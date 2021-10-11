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
#include "dynamic_graph_comm.h"
#include "static_graph_comm.h"

class ShortcutPropagation {
 public:
  ShortcutPropagation(const Config &conf, const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      config_(conf),
      iteration_(0),
      number_of_hitters_(conf.number_of_hitters) { 
    heavy_hitters_.set_empty_key(EmptyKey);
    heavy_hitters_.set_deleted_key(DeleteKey);
  }

  virtual ~ShortcutPropagation() = default;

  template <typename GraphType>
  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    rng_offset_ = global_vertices;
    if constexpr (std::is_same<GraphType, StaticGraph>::value) {
      FindLocalComponents(g, g_labels);

      CAGBuilder<StaticGraph> 
        first_contraction(g, g_labels, config_, rank_, size_);
      auto cag = first_contraction.BuildComponentAdjacencyGraph<StaticGraph>();
#ifndef NDEBUG
      OutputStats<StaticGraph>(cag);
#endif
#ifndef NSTATUS
      if (rank_ == ROOT || config_.print_verbose) 
        std::cout << "[STATUS] R" << rank_ << " Computed first cag " << std::endl;
#endif

      // Keep contraction labeling for later
      std::vector<VertexID> cag_labels(cag.GetVertexVectorSize(), 0);
      FindLocalComponents(cag, cag_labels);

      CAGBuilder<StaticGraph> 
        second_contraction(cag, cag_labels, config_, rank_, size_);
      auto ccag = second_contraction.BuildComponentAdjacencyGraph<StaticGraphCommunicator>();
#ifndef NDEBUG
      OutputStats<StaticGraphCommunicator>(ccag);
#endif
#ifndef NSTATUS
      if (rank_ == ROOT || config_.print_verbose) 
        std::cout << "[STATUS] R" << rank_ << " Computed second cag " << std::endl;
#endif

      PerformShortcutting(ccag);

      ApplyToLocalComponents(ccag, cag, cag_labels);
      ApplyToLocalComponents(cag, cag_labels, g, g_labels);
    } else if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value) {
      PerformShortcutting(g);
      g.ForallLocalVertices([&](const VertexID v) {
        g_labels[v] = g.GetVertexLabel(v);
      });
    }
  }

  void Output(StaticGraphCommunicator &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Counters
  unsigned int iteration_;
  VertexID rng_offset_;

  // Local labels
  std::vector<VertexID> labels_;
  std::vector<PEID> ranks_;

  // Heavy hitters
  VertexID number_of_hitters_;
  google::dense_hash_set<VertexID> heavy_hitters_;

  // Statistics
  Timer iteration_timer_;
  Timer shortcut_timer_;

  void PerformShortcutting(StaticGraphCommunicator &g) {
    // Init 
    labels_.resize(g.GetLocalVertexVectorSize());
    g.ForallLocalVertices([&](const VertexID v) { labels_[v] = g.GetGlobalID(v); });
    ranks_.resize(g.GetLocalVertexVectorSize(), rank_);

    // Iterate until converged
    do {
      iteration_timer_.Restart();
#ifndef NSTATUS
      if (rank_ == ROOT || config_.print_verbose) 
        std::cout << "[STATUS] R" << rank_ << " Starting iteration " << iteration_ << std::endl;
#endif
      PropagateLabels(g);
      FindMinLabels(g);
      if (number_of_hitters_ > 0) 
        FindHeavyHitters(g);
#ifndef NSTATUS
      if (rank_ == ROOT || config_.print_verbose) 
        std::cout << "[STATUS] |- R" << rank_ << " Propagating labels took " 
                                   << "[TIME] " << iteration_timer_.Elapsed() << std::endl;
#endif
      Shortcut(g);
#ifndef NSTATUS
      if (rank_ == ROOT || config_.print_verbose) 
        std::cout << "[STATUS] |- R" << rank_ << " Building shortcuts took " 
                  << "[TIME] " << iteration_timer_.Elapsed() << std::endl;
#endif
#ifndef NDEBUG
      OutputStats<StaticGraphCommunicator>(g);
#endif

      iteration_++;
    } while (!CheckConvergence(g));
  }

  void FindLocalComponents(StaticGraph &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetVertexVectorSize(), false);
    std::vector<VertexID> parent(g.GetVertexVectorSize(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
      parent[v] = v;
      marked[v] = false;
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS<StaticGraph>(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void PropagateLabels(StaticGraphCommunicator &g) {
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

  void FindMinLabels(StaticGraphCommunicator &g) {
    g.ForallLocalVertices([&](VertexID v) {
      // Gather min label of all neighbors
      VertexID v_label = g.GetVertexLabel(v);
      VertexID v_rank = g.GetVertexRoot(v);
      g.ForallNeighbors(v, [&](VertexID u) {
        if (g.GetVertexLabel(u) < v_label) {
          v_label = g.GetVertexLabel(u);
          v_rank = g.GetVertexRoot(u);
        }
      });
      labels_[v] = v_label;
      ranks_[v] = v_rank;
    });
  }

  void FindHeavyHitters(StaticGraphCommunicator &g) {
    google::dense_hash_map<VertexID, VertexID> number_hits;
    number_hits.set_empty_key(EmptyKey);
    number_hits.set_deleted_key(DeleteKey);
    g.ForallLocalVertices([&](const VertexID v) {
      const VertexID target = labels_[v];
      if (number_hits.find(target) == end(number_hits))
        number_hits[target] = 0;
      if (++number_hits[target] > g.GetLocalVertexVectorSize() / number_of_hitters_) {
        heavy_hitters_.insert(labels_[v]);
        if (heavy_hitters_.size() >= number_of_hitters_) return;
      }
    });
  }

  void Shortcut(StaticGraphCommunicator &g) {
    google::dense_hash_map<PEID, VertexBuffer> update_buffers;
    update_buffers.set_empty_key(EmptyKey);
    update_buffers.set_deleted_key(DeleteKey);

    google::dense_hash_map<PEID, VertexBuffer> request_buffers;
    request_buffers.set_empty_key(EmptyKey);
    request_buffers.set_deleted_key(DeleteKey);

    google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
    receive_buffers.set_empty_key(EmptyKey);
    receive_buffers.set_deleted_key(DeleteKey);

    google::dense_hash_map<VertexID, std::vector<VertexID>> update_lists;
    update_lists.set_empty_key(EmptyKey);
    update_lists.set_deleted_key(DeleteKey);

    google::dense_hash_set<VertexID> request_set;
    request_set.set_empty_key(EmptyKey);
    request_set.set_deleted_key(DeleteKey);

    shortcut_timer_.Restart();
    g.ForallLocalVertices([&](const VertexID v) {
      if (labels_[v] < g.GetVertexLabel(v)) {
        // Send l'(v) to l(v)
        update_buffers[g.GetVertexRoot(v)].emplace_back(g.GetVertexLabel(v));
        update_buffers[g.GetVertexRoot(v)].emplace_back(labels_[v]);
        update_buffers[g.GetVertexRoot(v)].emplace_back(ranks_[v]);
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

#ifndef NSTATUS
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |-- R" << rank_ << " Filling buffers took " 
                << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;
#endif

    // Process local requests
    shortcut_timer_.Restart();
    if (request_buffers[rank_].size() > 0) {
      for (const VertexID &request : request_buffers[rank_]) {
        update_buffers[rank_].emplace_back(request);
        update_buffers[rank_].emplace_back(g.GetVertexLabel(g.GetLocalID(request)));
        update_buffers[rank_].emplace_back(g.GetVertexRoot(g.GetLocalID(request)));
      }
    }
    request_buffers[rank_].clear();

#ifndef NSTATUS
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |-- R" << rank_ << " Resolving local requests took " 
                << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;
#endif

    shortcut_timer_.Restart();
    CommunicationUtility::AllToAll(request_buffers, receive_buffers, rank_, size_, ExpTag, config_.use_regular);
    CommunicationUtility::ClearBuffers(request_buffers);

    for (auto &kv : receive_buffers) {
      for (const VertexID &m : kv.second) {
        update_buffers[kv.first].emplace_back(m);
        update_buffers[kv.first].emplace_back(g.GetVertexLabel(g.GetLocalID(m)));
        update_buffers[kv.first].emplace_back(g.GetVertexRoot(g.GetLocalID(m)));
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);

#ifndef NSTATUS
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |-- R" << rank_ << " Resolving remote requests took " 
                << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;
#endif

    // Process request answers
    // Process local answers
    shortcut_timer_.Restart();
    if (update_buffers[rank_].size() > 0) {
      for (VertexID i = 0; i < update_buffers[rank_].size(); i += 3) {
        const VertexID target = update_buffers[rank_][i];
        const VertexID label = update_buffers[rank_][i + 1];
        const VertexID root = update_buffers[rank_][i + 2];
        for (const VertexID v : update_lists[target]) {
          if (label < labels_[v]) {
            labels_[v] = label;
            ranks_[v] = root;
          }
        }
      }
    }
    update_buffers[rank_].clear();

#ifndef NSTATUS
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |-- R" << rank_ << " Resolving local answers took " 
                << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;
#endif

    // Process remote answers
    shortcut_timer_.Restart();
    CommunicationUtility::AllToAll(update_buffers, receive_buffers, rank_, size_, ExpTag, config_.use_regular);
    CommunicationUtility::ClearBuffers(update_buffers);

    for (auto &kv : receive_buffers) {
        for (VertexID i = 0; i < kv.second.size(); i += 3) {
        const VertexID target = kv.second[i];
        const VertexID label = kv.second[i + 1];
        const VertexID root = kv.second[i + 2];
        for (const VertexID v : update_lists[target]) {
          if (label < labels_[v]) {
            labels_[v] = label;
            ranks_[v] = root;
          }
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);

#ifndef NSTATUS
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |-- R" << rank_ << " Resolving remote answers took " 
                << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;
#endif
  }

  bool CheckConvergence(StaticGraphCommunicator &g) {
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

  void ApplyToLocalComponents(StaticGraphCommunicator &cag, 
                              StaticGraph &g, std::vector<VertexID> &g_label) {
    std::vector<bool> resolved(g.GetLocalVertexVectorSize(), false);
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
      resolved[v] = true;
    });
    // Turn on remaining vertices and set their labels
    g.SetAllVerticesActive(true);
    g.ForallLocalVertices([&](const VertexID v) {
      if (!resolved[v]) g_label[v] = rng_offset_ + g.GetGlobalID(v);
    });
  }

  void ApplyToLocalComponents(StaticGraph &cag, 
                              std::vector<VertexID> &cag_label, 
                              StaticGraph &g, 
                              std::vector<VertexID> &g_label) {
    std::vector<bool> resolved(g.GetLocalVertexVectorSize(), false);
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
      resolved[v] = true;
    });
    // Turn on remaining vertices and set their labels
    g.SetAllVerticesActive(true);
    g.ForallLocalVertices([&](const VertexID v) {
      if (!resolved[v]) g_label[v] = g.GetGlobalID(v);
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
