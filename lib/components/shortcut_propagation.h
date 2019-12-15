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
    heavy_hitters_.set_empty_key(-1);
  }

  virtual ~ShortcutPropagation() = default;

  void FindComponents(StaticGraphCommunicator &g, std::vector<VertexID> &g_labels) {
    if (config_.use_contraction) {
      // FindLocalComponents(g, g_labels);

      // CAGBuilder<StaticGraphCommunicator> 
      //   first_contraction(g, g_labels, rank_, size_);
      // StaticGraphCommunicator cag = first_contraction.BuildDynamicComponentAdjacencyGraph();
      // OutputStats<StaticGraphCommunicator>(cag);

      // // TODO: Delete original graph?
      // // Keep contraction labeling for later
      // std::vector<VertexID> cag_labels(cag.GetNumberOfVertices(), 0);
      // FindLocalComponents(cag, cag_labels);

      // CAGBuilder<StaticGraphCommunicator> 
      //   second_contraction(cag, cag_labels, rank_, size_);
      // StaticGraphCommunicator ccag = second_contraction.BuildDynamicComponentAdjacencyGraph();
      // OutputStats<StaticGraphCommunicator>(ccag);

      // PerformShortcutting(ccag);

      // ApplyToLocalComponents(ccag, cag, cag_labels);
      // ApplyToLocalComponents(cag, cag_labels, g, g_labels);
    } else {
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
      OutputStats<StaticGraphCommunicator>(g);

      iteration_++;
    } while (!CheckConvergence(g));
  }

  void FindLocalComponents(StaticGraphCommunicator &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<StaticGraphCommunicator>::BFS(g, v, marked, parent);
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
    // TODO: Use BFS instead
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

  void Shortcut(StaticGraphCommunicator &g) {
    google::dense_hash_map<PEID, std::vector<VertexID>> update_buffers;
    update_buffers.set_empty_key(-1);
    google::dense_hash_map<PEID, std::vector<VertexID>> request_buffers;
    request_buffers.set_empty_key(-1);

    google::dense_hash_map<VertexID, std::vector<VertexID>> update_lists;
    update_lists.set_empty_key(-1);

    google::dense_hash_set<VertexID> request_set;
    request_set.set_empty_key(-1);

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

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Filling buffers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Send updates and requests
    int num_requests = 0;
    for (const auto &kv : request_buffers) {
      PEID pe = kv.first;
      if (pe == rank_) continue;
      if (request_buffers[pe].size() > 0) num_requests++;
    }
    std::vector<MPI_Request> answer_requests(num_requests);

    shortcut_timer_.Restart();
    int req = 0;
    for (const auto &kv : request_buffers) {
      PEID pe = kv.first;
      if (pe == rank_) continue;
      if (request_buffers[pe].size() > 0) {
        MPI_Issend(request_buffers[pe].data(), request_buffers[pe].size(), MPI_VERTEX, pe, 
                   7 * size_ + pe, MPI_COMM_WORLD, &answer_requests[req++]);
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Sending buffers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Process local requests
    shortcut_timer_.Restart();
    if (request_buffers[rank_].size() > 0) {
      for (const VertexID &request : request_buffers[rank_]) {
        update_buffers[rank_].emplace_back(request);
        update_buffers[rank_].emplace_back(g.GetVertexLabel(g.GetLocalID(request)));
        update_buffers[rank_].emplace_back(g.GetVertexRoot(g.GetLocalID(request)));
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving local requests took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    shortcut_timer_.Restart();
    std::vector<MPI_Status> statuses(num_requests);
    int isend_done = 0;
    while(isend_done == 0) {
      // Check for messages
      int iprobe_success = 0;
      MPI_Status st{};
      MPI_Iprobe(MPI_ANY_SOURCE, 7 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
      if (iprobe_success) {
        int message_length;
        MPI_Get_count(&st, MPI_VERTEX, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_VERTEX, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Request
        if (st.MPI_TAG == 7 * size_ + rank_) {
          if (st.MPI_SOURCE == rank_) {
            std::cout << "[ERROR] R" << rank_ << " self message!" << std::endl;
            exit(1);
          }
          for (const VertexID &m : message) {
            update_buffers[st.MPI_SOURCE].emplace_back(m);
            update_buffers[st.MPI_SOURCE].emplace_back(g.GetVertexLabel(g.GetLocalID(m)));
            update_buffers[st.MPI_SOURCE].emplace_back(g.GetVertexRoot(g.GetLocalID(m)));
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }

      // Check if all ISend successful
      isend_done = 0;
      MPI_Testall(num_requests, answer_requests.data(), &isend_done, statuses.data());
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

    int ibarrier_done = 0;
    while(ibarrier_done == 0) {
      // Check for messages
      int iprobe_success = 0;
      MPI_Status st{};
      MPI_Iprobe(MPI_ANY_SOURCE, 7 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
      if (iprobe_success) {
        int message_length;
        MPI_Get_count(&st, MPI_VERTEX, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_VERTEX, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Request
        if (st.MPI_TAG == 7 * size_ + rank_) {
          if (st.MPI_SOURCE == rank_) {
            std::cout << "[ERROR] R" << rank_ << " self message!" << std::endl;
            exit(1);
          }
          for (const VertexID &m : message) {
            update_buffers[st.MPI_SOURCE].emplace_back(m);
            update_buffers[st.MPI_SOURCE].emplace_back(g.GetVertexLabel(g.GetLocalID(m)));
            update_buffers[st.MPI_SOURCE].emplace_back(g.GetVertexRoot(g.GetLocalID(m)));
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }

      // Check if all reached Ibarrier
      MPI_Status tst{};
      MPI_Test(&barrier_request, &ibarrier_done, &tst);
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving remote requests took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    num_requests = 0;
    for (const auto &kv : request_buffers) {
      PEID pe = kv.first;
      if (pe == rank_) continue;
      if (update_buffers[pe].size() > 0) num_requests++;
    }
    std::vector<MPI_Request> update_requests(num_requests);

    shortcut_timer_.Restart();
    statuses.resize(num_requests);
    req = 0;
    for (const auto &kv : request_buffers) {
      PEID pe = kv.first;
      if (pe == rank_) continue;
      if (update_buffers[pe].size() > 0) {
        MPI_Issend(update_buffers[pe].data(), update_buffers[pe].size(), MPI_VERTEX, pe, 
                   72 * size_ + pe, MPI_COMM_WORLD, &update_requests[req++]);
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Sending updates/answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

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

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving local answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    // Process remote answers
    shortcut_timer_.Restart();
    std::vector<VertexID> receive_buffer;
    isend_done = 0;
    while(!isend_done) {
      // Check for messages
      int iprobe_success = 0;
      MPI_Status st{};
      MPI_Iprobe(MPI_ANY_SOURCE, 72 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
      if (iprobe_success) {
        int message_length;
        MPI_Get_count(&st, MPI_VERTEX, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_VERTEX, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Request
        if (st.MPI_TAG == 72 * size_ + rank_) {
          if (st.MPI_SOURCE == rank_) {
            std::cout << "[ERROR] R" << rank_ << " self message!" << std::endl;
            exit(1);
          }
          for (const auto &m : message) {
            receive_buffer.emplace_back(m);
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }

      // Check if all ISend successful
      isend_done = 0;
      MPI_Testall(num_requests, update_requests.data(), &isend_done, statuses.data());
    }

    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

    ibarrier_done = 0;
    while(!ibarrier_done) {
      // Check for messages
      int iprobe_success = 0;
      MPI_Status st{};
      MPI_Iprobe(MPI_ANY_SOURCE, 72 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
      if (iprobe_success) {
        int message_length;
        MPI_Get_count(&st, MPI_VERTEX, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0], message_length, MPI_VERTEX, st.MPI_SOURCE,
                 st.MPI_TAG, MPI_COMM_WORLD, &rst);
        // Request
        if (st.MPI_TAG == 72 * size_ + rank_) {
          if (st.MPI_SOURCE == rank_) {
            std::cout << "[ERROR] R" << rank_ << " self message!" << std::endl;
            exit(1);
          }
          for (const auto &request : message) {
            receive_buffer.emplace_back(request);
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }

      // Check if all reached Ibarrier
      MPI_Status tst{};
      MPI_Test(&barrier_request, &ibarrier_done, &tst);
    }

    for (VertexID i = 0; i < receive_buffer.size(); i += 3) {
      const VertexID target = receive_buffer[i];
      const VertexID label = receive_buffer[i + 1];
      const VertexID root = receive_buffer[i + 2];
      for (const VertexID v : update_lists[target]) {
        if (label < labels_[v]) {
          labels_[v] = label;
          ranks_[v] = root;
        }
      }
    }

    if (rank_ == ROOT) std::cout << "[STATUS] |-- Resolving remote answers took " 
                                 << "[TIME] " << shortcut_timer_.Elapsed() << std::endl;

    CheckRequests(answer_requests);
    CheckRequests(update_requests);
  }

  void CheckRequests(std::vector<MPI_Request> &requests) {
    unsigned int unresolved_requests = 0;
    for (unsigned int i = 0; i < requests.size(); ++i) {
      if (requests[i] != MPI_REQUEST_NULL) {
        MPI_Request_free(&requests[i]);
        unresolved_requests++;
      }
    }
    if (unresolved_requests > 0) {
      std::cerr << "R" << rank_ << " Error unresolved requests in shortcut propagation" << std::endl;
      exit(0);
    }
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
                              StaticGraphCommunicator &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(StaticGraphCommunicator &cag, 
                              std::vector<VertexID> &cag_label, 
                              StaticGraphCommunicator &g, 
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
