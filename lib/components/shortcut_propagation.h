/******************************************************************************
 * shortcut_propagation.h
 *
 * Distributed shortcutted label propagation
 ******************************************************************************
 * Copyright (C) 2017 Sebastian Lamm <lamm@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef _SHORTCUT_PROPAGATION_H_
#define _SHORTCUT_PROPAGATION_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "cag_builder.h"
#include "utils.h"
#include "union_find.h"
#include "dynamic_graph_access.h"

class ShortcutPropagation {
 public:
  ShortcutPropagation(const Config &conf, const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      config_(conf),
      iteration_(0),
      number_of_hitters_(conf.number_of_hitters) { 
    heavy_hitters_.set_empty_key(-1);
  }

  virtual ~ShortcutPropagation() = default;

  void FindComponents(DynamicGraphAccess &g, std::vector<VertexID> &g_labels) {
    if (config_.use_contraction) {
      FindLocalComponents(g, g_labels);

      CAGBuilder<DynamicGraphAccess> 
        first_contraction(g, g_labels, rank_, size_);
      DynamicGraphAccess cag = first_contraction.BuildDynamicComponentAdjacencyGraph();
      OutputStats<DynamicGraphAccess>(cag);

      // TODO: Delete original graph?
      // Keep contraction labeling for later
      std::vector<VertexID> cag_labels(cag.GetNumberOfVertices(), 0);
      FindLocalComponents(cag, cag_labels);

      CAGBuilder<DynamicGraphAccess> 
        second_contraction(cag, cag_labels, rank_, size_);
      DynamicGraphAccess ccag = second_contraction.BuildDynamicComponentAdjacencyGraph();
      OutputStats<DynamicGraphAccess>(ccag);

      PerformShortcutting(ccag);

      ApplyToLocalComponents(ccag, cag, cag_labels);
      ApplyToLocalComponents(cag, cag_labels, g, g_labels);
    } else {
      PerformShortcutting(g);
      g.ForallLocalVertices([&](const VertexID v) {
          g_labels[v] = g.GetVertexLabel(v);
      });
    }
  }

  void Output(DynamicGraphAccess &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Counters
  unsigned int iteration_;

  // Local labels
  std::vector<VertexID> labels_;
  std::vector<PEID> ranks_;

  // Heavy hitters
  VertexID number_of_hitters_;
  google::dense_hash_set<VertexID> heavy_hitters_;

  // Statistics
  Timer iteration_timer_;
  Timer shortcut_timer_;

  void PerformShortcutting(DynamicGraphAccess &g) {
    // Init 
    labels_.resize(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) { labels_[v] = g.GetGlobalID(v); });
    ranks_.resize(g.GetNumberOfLocalVertices(), rank_);

    // Iterate until converged
    do {
      iteration_timer_.Restart();
      if (rank_ == ROOT) std::cout << "[STATUS] Starting iteration " << iteration_ << std::endl;
      PropagateLabels(g);
      FindMinLabels(g);
      if (number_of_hitters_ > 0) 
        FindHeavyHitters(g);
      if (rank_ == ROOT) std::cout << "[STATUS] |- Propagating labels took " 
                                   << "[TIME] " << iteration_timer_.Elapsed() << std::endl;
      Shortcut(g);
      if (rank_ == ROOT) std::cout << "[STATUS] |- Building shortcuts took " 
                                   << "[TIME] " << iteration_timer_.Elapsed() << std::endl;
      OutputStats<DynamicGraphAccess>(g);

      iteration_++;
    } while (!CheckConvergence(g));
  }

  void FindLocalComponents(DynamicGraphAccess &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<DynamicGraphAccess>::BFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void PropagateLabels(DynamicGraphAccess &g) {
    g.ForallLocalVertices([&](const VertexID v) {
      if (labels_[v] < g.GetVertexLabel(v))
        g.SetVertexPayload(v, {g.GetVertexDeviate(v), 
                               labels_[v], 
#ifdef TIEBREAK_DEGREE
                               0,
#endif
                               ranks_[v]});
    });
    g.SendAndReceiveGhostVertices();
  } 

  void FindMinLabels(DynamicGraphAccess &g) {
    g.ForallLocalVertices([&](VertexID v) {
      // Gather min label of all neighbors
      VertexID min_label = g.GetVertexLabel(v);
      VertexID min_rank = g.GetVertexRoot(v);
      g.ForallNeighbors(v, [&](VertexID u) {
        if (g.GetVertexLabel(u) < min_label) {
          min_label = g.GetVertexLabel(u);
          min_rank = g.GetVertexRoot(u);
        }
      });
      labels_[v] = min_label;
      ranks_[v] = min_rank;
    });
  }

  void FindHeavyHitters(DynamicGraphAccess &g) {
    google::dense_hash_map<VertexID, VertexID> number_hits;
    number_hits.set_empty_key(-1);
    g.ForallLocalVertices([&](const VertexID v) {
      const VertexID target = labels_[v];
      if (number_hits.find(target) == end(number_hits))
        number_hits[target] = 0;
      if (++number_hits[target] > g.GetNumberOfLocalVertices() / number_of_hitters_) {
        heavy_hitters_.insert(labels_[v]);
        if (heavy_hitters_.size() >= number_of_hitters_) return;
      }
    });
  }

  void Shortcut(DynamicGraphAccess &g) {
    std::vector<std::vector<std::tuple<VertexID, VertexID, VertexID>>> update_buffers(size_);
    std::vector<std::vector<VertexID>> request_buffers(size_);
    google::dense_hash_map<VertexID, std::vector<VertexID>> update_lists;
    update_lists.set_empty_key(-1);
    google::dense_hash_set<VertexID> request_set;
    request_set.set_empty_key(-1);

    shortcut_timer_.Restart();
    g.ForallLocalVertices([&](const VertexID v) {
      if (labels_[v] < g.GetVertexLabel(v)) {
        // Send l'(v) to l(v)
        update_buffers[g.GetVertexRoot(v)].emplace_back(g.GetVertexLabel(v), labels_[v], ranks_[v]);
      }
      // Request l(l'(v)) from l'(v)
      // Check for heavy hitter
      if (number_of_hitters_ == 0 || (number_of_hitters_ > 0 && heavy_hitters_.find(labels_[v]) != end(heavy_hitters_))) {
        // Check for uniqueness
        if (request_set.find(labels_[v]) == end(request_set)) {
          request_set.insert(labels_[v]);
          request_buffers[ranks_[v]].emplace_back(labels_[v]);
        }
      } 
      update_lists[labels_[v]].emplace_back(v);
    });

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Filling buffers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    MPI_Datatype MPI_LABEL_UPDATE;
    MPI_Type_vector(1, 3, 0, MPI_LONG, &MPI_LABEL_UPDATE);
    MPI_Type_commit(&MPI_LABEL_UPDATE);

    // Send updates and requests
    std::vector<MPI_Request*> answer_requests;
    std::vector<MPI_Request*> update_requests;
    answer_requests.clear();
    update_requests.clear();

    shortcut_timer_.Restart();
    for (PEID i = 0; i < size_; ++i) {
      if (i == rank_) continue;
      {
        auto *req = new MPI_Request();
        MPI_Isend(&update_buffers[i][0], update_buffers[i].size(), MPI_LABEL_UPDATE, i, 
                  6 * size_ + i, MPI_COMM_WORLD, req);
        update_requests.emplace_back(req);
      }

      {
        auto *req = new MPI_Request();
        MPI_Isend(&request_buffers[i][0], request_buffers[i].size(), MPI_LONG, i, 
                  7 * size_ + i, MPI_COMM_WORLD, req);
        answer_requests.emplace_back(req);
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Sending buffers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Receive request and send shortcuts 
    std::vector<std::vector<std::tuple<VertexID, VertexID, VertexID>>> answers(size_);

    // Process local requests
    shortcut_timer_.Restart();
    if (request_buffers[rank_].size() > 0) {
      for (const VertexID &request : request_buffers[rank_]) {
        update_buffers[rank_].emplace_back(request, g.GetVertexLabel(g.GetLocalID(request)), g.GetVertexRoot(g.GetLocalID(request)));
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving local requests took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Process remote requests
    MPI_Status st{};
    int flag = 1;
    shortcut_timer_.Restart();
    do {
      // Probe for requests
      MPI_Iprobe(MPI_ANY_SOURCE, 7 * size_ + rank_, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_LONG, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_LONG, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Request
        if (st.MPI_TAG == 7 * size_ + rank_) {
          for (const VertexID &request : message) {
            answers[st.MPI_SOURCE].emplace_back(request, g.GetVertexLabel(g.GetLocalID(request)), g.GetVertexRoot(g.GetLocalID(request)));
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }
    } while (flag);

    WaitForRequests(answer_requests);

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving remote requests took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    shortcut_timer_.Restart();
    for (PEID i = 0; i < size_; ++i) {
      if (i == rank_) continue;
      {
        auto *req = new MPI_Request();
        MPI_Isend(&answers[i][0], answers[i].size(), MPI_LABEL_UPDATE, i, 6 * size_ + i, MPI_COMM_WORLD, req);
        update_requests.emplace_back(req);
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Sending answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Process request answers
    // Process local answers
    shortcut_timer_.Restart();
    if (update_buffers[rank_].size() > 0) {
      for (const auto &update : update_buffers[rank_]) {
        const VertexID target = std::get<0>(update);
        const VertexID label = std::get<1>(update);
        const VertexID root = std::get<2>(update);
        for (const VertexID v : update_lists[target]) {
          if (label < labels_[v]) {
            labels_[v] = label;
            ranks_[v] = root;
          }
        }
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving local answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Process remote answers
    flag = 1;
    shortcut_timer_.Restart();
    MPI_Barrier(MPI_COMM_WORLD);
    do {
      // Probe for updates
      MPI_Iprobe(MPI_ANY_SOURCE, 6 * size_ + rank_, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_LABEL_UPDATE, &message_length);
        // std::cout << "R" << 
        std::vector<std::tuple<VertexID, VertexID, VertexID>> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_LABEL_UPDATE, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Updates
        if (st.MPI_TAG == 6 * size_ + rank_) {
          for (const auto &update : message) {
            const VertexID target = std::get<0>(update);
            const VertexID label = std::get<1>(update);
            const VertexID root = std::get<2>(update);
            for (const VertexID v : update_lists[target]) {
              if (label < labels_[v]) {
                labels_[v] = label;
                ranks_[v] = root;
              }
            }
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }
    } while (flag);
    WaitForRequests(update_requests);

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving remote answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

  }

  void WaitForRequests(std::vector<MPI_Request*>& requests) {
    for (unsigned int i = 0; i < requests.size(); ++i) {
      MPI_Status st;
      MPI_Wait(requests[i], &st);
      delete requests[i];
    }
  }

  bool CheckConvergence(DynamicGraphAccess &g) {
    int converged_globally = 0;

    // Check local convergence
    int converged_locally = 1;
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetVertexLabel(v) != labels_[v]) converged_locally = 0;
    });

    MPI_Allreduce(&converged_locally,
                  &converged_globally,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    return converged_globally;
  }

  void ApplyToLocalComponents(DynamicGraphAccess &cag, 
                              DynamicGraphAccess &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(DynamicGraphAccess &cag, 
                              std::vector<VertexID> &cag_label, 
                              DynamicGraphAccess &g, 
                              std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
    });
  }

  template <typename GraphType>
  void OutputStats(GraphType &g) {
    VertexID n = g.GatherNumberOfGlobalVertices();
    EdgeID m = g.GatherNumberOfGlobalEdges();

    // Determine min/maximum cut size
    EdgeID m_cut = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      std::cout << "TEMP "
                << "s=" << config_.seed << ", "
                << "p=" << size_  << ", "
                << "n=" << n << ", "
                << "m=" << m << ", "
                << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }
};

#endif
