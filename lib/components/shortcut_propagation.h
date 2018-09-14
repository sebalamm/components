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

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "graph_access.h"

class ShortcutPropagation {
 public:
  ShortcutPropagation(const Config &conf, const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      config_(conf),
      iteration_(0),
      number_of_hitters_(0) { }
  virtual ~ShortcutPropagation() = default;

  void FindComponents(GraphAccess &g) {
    // Init 
    labels_.resize(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) { labels_[v] = g.GetGlobalID(v); });
    ranks_.resize(g.GetNumberOfLocalVertices(), rank_);

    // Iterate until converged
    do {
      PropagateLabels(g);
      if (number_of_hitters_ > 0) 
        FindHeavyHitters(g);
      Shortcut(g);
      iteration_++;
    } while (!CheckConvergence(g));
  }

  void Output(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "Component labels" << std::endl;
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
  std::unordered_set<VertexID> heavy_hitters_;

  void PropagateLabels(GraphAccess &g) {
    g.ForallLocalVertices([&](const VertexID v) {
      if (labels_[v] < g.GetVertexLabel(v))
        g.SetVertexPayload(v, {g.GetVertexDeviate(v), labels_[v], ranks_[v]});
      // std::cout << "[R" << rank_ << ":" << iteration_ << "]" << " v (" << g.GetGlobalID(v) << "," << labels_[v] << "," << ranks_[v] << ")" << std::endl;
    });
    g.SendAndReceiveGhostVertices();
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
      // std::cout << "[R" << rank_ << ":" << iteration_ << "] p (" << g.GetGlobalID(v) << "," << labels_[v] << "," << ranks_[v] << ")" << std::endl;
    });
  } 

  void FindHeavyHitters(GraphAccess &g) {
    std::unordered_map<VertexID, VertexID> number_hits;
    g.ForallLocalVertices([&](const VertexID v) {
      const VertexID target = labels_[v];
      if (number_hits.find(target) == end(number_hits))
        number_hits[target] = 0;
      if (++number_hits[target] > g.GetNumberOfLocalVertices() / number_of_hitters_) {
        heavy_hitters_.emplace(labels_[v]);
        if (heavy_hitters_.size() >= number_of_hitters_) return;
      }
    });
  }

  void Shortcut(GraphAccess &g) {
    std::vector<std::vector<std::tuple<VertexID, VertexID, VertexID>>> update_buffers(size_);
    std::vector<std::vector<VertexID>> request_buffers(size_);
    std::unordered_map<VertexID, std::vector<VertexID>> update_lists;
    std::unordered_set<VertexID> request_set;
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
          request_set.emplace(labels_[v]);
          request_buffers[ranks_[v]].emplace_back(labels_[v]);
        }
      } 
      update_lists[labels_[v]].emplace_back(v);
    });

    MPI_Datatype MPI_LABEL_UPDATE;
    MPI_Type_vector(1, 3, 0, MPI_LONG, &MPI_LABEL_UPDATE);
    MPI_Type_commit(&MPI_LABEL_UPDATE);

    // Send updates and requests
    for (PEID i = 0; i < size_; ++i) {
      {
        MPI_Request req;
        // std::cout << "[R" << rank_ << ":" << iteration_ << "] num updates for " << i << " = " << update_buffers[i].size() << std::endl;
        MPI_Isend(&update_buffers[i][0], update_buffers[i].size(), MPI_LABEL_UPDATE, i, 
                  6 * size_ + i, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
      }

      {
        MPI_Request req;
        // std::cout << "[R" << rank_ << ":" << iteration_ << "] num requests for " << i << " = " << request_buffers[i].size() << std::endl;
        MPI_Isend(&request_buffers[i][0], request_buffers[i].size(), MPI_LONG, i, 
                  7 * size_ + i, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
      }
    }

    // Process updates
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Status st{};
    // int flag = 1;
    // do {
    //   // Probe for updates
    //   MPI_Iprobe(MPI_ANY_SOURCE, 6 * size_ + rank_, MPI_COMM_WORLD, &flag, &st);
    //   if (flag) {
    //     int message_length;
    //     MPI_Get_count(&st, MPI_LABEL_UPDATE, &message_length);
    //     std::vector<std::tuple<VertexID, VertexID, VertexID>> message(message_length);
    //     MPI_Status rst{};
    //     MPI_Recv(&message[0], message_length, MPI_LABEL_UPDATE, st.MPI_SOURCE,
    //              st.MPI_TAG, MPI_COMM_WORLD, &rst);
    //     // Updates
    //     if (st.MPI_TAG == 6 * size_ + rank_) {
    //       for (const auto &update : message) {
    //         const VertexID target = std::get<0>(update);
    //         // const VertexID local_target = g.GetLocalID(target);
    //         const VertexID label = std::get<1>(update);
    //         const VertexID root = std::get<2>(update);
    //         // std::cout << "[R" << rank_ << ":" << iteration_ << "] update " << target << " to " << label << std::endl;
    //         for (const VertexID v : update_lists[target]) {
    //           if (label < labels_[v]) {
    //             labels_[v] = label;
    //             ranks_[v] = root;
    //           }
    //         }
    //       }
    //     } else std::cout << "Unexpected tag." << std::endl;
    //   }
    // } while (flag);

    // Receive request and send shortcuts 
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<std::vector<std::tuple<VertexID, VertexID, VertexID>>> answers(size_);
    // flag = 1;
    MPI_Status st{};
    int flag = 1;
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
            // std::cout << "[R" << rank_ << ":" << iteration_ << "] request from " << st.MPI_SOURCE<< " get label of " << request << std::endl;
            answers[st.MPI_SOURCE].emplace_back(request, g.GetVertexLabel(g.GetLocalID(request)), g.GetVertexRoot(g.GetLocalID(request)));
          }
        } else std::cout << "Unexpected tag." << std::endl;
      }
    } while (flag);

    for (PEID i = 0; i < size_; ++i) {
      {
        MPI_Request req;
        MPI_Isend(&answers[i][0], answers[i].size(), MPI_LABEL_UPDATE, i, 6 * size_ + i, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
      }
    }

    // Process request answers
    MPI_Barrier(MPI_COMM_WORLD);
    flag = 1;
    do {
      // Probe for updates
      MPI_Iprobe(MPI_ANY_SOURCE, 6 * size_ + rank_, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_LABEL_UPDATE, &message_length);
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
            // std::cout << "[R" << rank_ << ":" << iteration_ << "] request answer from " << st.MPI_SOURCE << " label of " << target << " is "  << label << std::endl;
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

    MPI_Barrier(MPI_COMM_WORLD);
  }

  bool CheckConvergence(GraphAccess &g) {
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
};

#endif
