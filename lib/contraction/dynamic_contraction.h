/******************************************************************************
 * graph_contraction.h
 *
 * Contraction of distributed graph
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

#ifndef _DYNAMIC_CONTRACTION_H_
#define _DYNAMIC_CONTRACTION_H_

#include <iostream>
#include <google/sparse_hash_set>

#include "config.h"
#include "definitions.h"
#include "dynamic_graph_access.h"
#include "static_graph_access.h"
#include "edge_hash.h"

class DynamicContraction {
 public:
  DynamicContraction(DynamicGraphAccess &g, const PEID rank, const PEID size)
      : g_(g), 
        rank_(rank), 
        size_(size),
        contraction_level_(0) { 
    inactive_level_.resize(g_.GetNumberOfVertices(), -1);
  }
  virtual ~DynamicContraction() = default;

  inline bool IsActive(VertexID v) const {
    return inactive_level_[v] == -1;
  }

  void ExponentialContraction() {
    // Statistics
    Timer contract_timer;
    contract_timer.Restart();

    VertexID num_global_vertices = g_.GatherNumberOfGlobalVertices();
    VertexID num_vertices = g_.GetNumberOfVertices();

    // Update with new vertices added during last contraction
    inactive_level_.resize(num_vertices, -1);

    // Determine edges to communicate
    // Gather labels to communicate
    google::dense_hash_set<VertexID> send_ids(size_); 
    send_ids.set_empty_key(-1);

    std::vector<std::vector<VertexID>> send_buffers_a(size_);
    std::vector<std::vector<VertexID>> send_buffers_b(size_);
    std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
    std::vector<std::vector<VertexID>> receive_buffers(size_);

    for (int i = 0; i < size_; ++i) {
      send_buffers_a[i].clear();
      send_buffers_b[i].clear();
      receive_buffers[i].clear();
    }

    google::dense_hash_set<VertexID> inserted_edges; 
    inserted_edges.set_empty_key(-1);
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);

    FindExponentialConflictingEdges(num_global_vertices, 
                                    inserted_edges, 
                                    edges_to_add, 
                                    send_ids, 
                                    current_send_buffers);

    std::vector<bool> is_adj(size_);
    PEID num_adj = FindAdjacentPEs(is_adj);

    // Propagate edge buffers until all vertices are converged
    std::vector<MPI_Request*> requests;
    requests.clear();
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {

      SendMessages(is_adj, current_send_buffers, requests);
      ReceiveMessages(num_adj, requests, current_send_buffers, receive_buffers);
      SwapBuffers(current_send_buffers, send_buffers_a, send_buffers_b);

      int converged_locally = ProcessExponentialMessages(num_global_vertices, inserted_edges, edges_to_add, 
                                                         send_ids, receive_buffers, current_send_buffers);

      // Check if all PEs are done
      // if (++local_iterations % 6 == 0) {
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      // } 
      local_iterations++;
    }

    if (rank_ == ROOT) 
      std::cout << "[STATUS] |--- Propagation done " 
                << "[INFO] rounds "  << local_iterations << " "
                << "[TIME] " << contract_timer.Elapsed() << std::endl;

    // Insert edges and keep corresponding vertices
    g_.ForallLocalVertices([&](VertexID v) { g_.RemoveAllEdges(v); });
    contraction_level_++;

    UpdateActiveVertices();
    g_.ResetNumberOfCutEdges();
    InsertEdges(edges_to_add);

    // max_degree_computed_ = false;
    UpdateGraphVertices();
  }

  void LocalContraction() {
    // Statistics
    Timer contract_timer;
    contract_timer.Restart();

    VertexID num_global_vertices = g_.GatherNumberOfGlobalVertices();
    VertexID num_vertices = g_.GetNumberOfVertices();

    // Update with new vertices added during last contraction
    inactive_level_.resize(num_vertices, -1);

    // Determine edges to communicate
    // Gather labels to communicate
    google::dense_hash_set<VertexID> send_ids(size_); 
    send_ids.set_empty_key(-1);

    std::vector<std::vector<VertexID>> send_buffers_a(size_);
    std::vector<std::vector<VertexID>> send_buffers_b(size_);
    std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
    std::vector<std::vector<VertexID>> receive_buffers(size_);

    for (int i = 0; i < size_; ++i) {
      send_buffers_a[i].clear();
      send_buffers_b[i].clear();
      receive_buffers[i].clear();
    }

    google::dense_hash_set<VertexID> inserted_edges; 
    inserted_edges.set_empty_key(-1);
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);

    // TODO: Factor out in new method
    FindLocalConflictingEdges(num_global_vertices, 
                              send_ids, 
                              current_send_buffers);

    std::vector<bool> is_adj(size_);
    PEID num_adj = FindAdjacentPEs(is_adj);

    // Propagate edge buffers until all vertices are converged
    std::vector<MPI_Request*> requests;
    requests.clear();
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {
      SendMessages(is_adj, current_send_buffers, requests);
      ReceiveMessages(num_adj, requests, current_send_buffers, receive_buffers);
      SwapBuffers(current_send_buffers, send_buffers_a, send_buffers_b);

      int converged_locally = ProcessLocalMessages(num_global_vertices, inserted_edges, edges_to_add, 
                                                   send_ids, receive_buffers, current_send_buffers);


      // Check if all PEs are done
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      local_iterations++;
    }
    if (rank_ == ROOT) 
    std::cout << "[STATUS] |--- Propagation done " 
              << "[TIME] " << contract_timer.Elapsed() << std::endl;

    // Insert edges and keep corresponding vertices
    g_.ForallLocalVertices([&](VertexID v) { g_.RemoveAllEdges(v); });
    contraction_level_++;

    UpdateActiveVertices();
    g_.ResetNumberOfCutEdges();
    InsertEdges(edges_to_add);

    // max_degree_computed_ = false;
    UpdateGraphVertices();
  }

  void FindExponentialConflictingEdges(VertexID num_global_vertices,
                                       google::dense_hash_set<VertexID> &inserted_edges, 
                                       std::vector<std::vector<std::pair<VertexID, VertexID>>> &local_edges,
                                       google::dense_hash_set<VertexID> &sent_edges,
                                       std::vector<std::vector<VertexID>> *send_buffers) {
    g_.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g_.GetVertexLabel(v);
      g_.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g_.GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + num_global_vertices * wlabel;
          PEID wroot = g_.GetVertexRoot(w);

          if (inserted_edges.find(update_id) == end(inserted_edges) 
                && g_.IsLocalFromGlobal(vlabel)) {
            local_edges[wroot].emplace_back(vlabel, wlabel);
            inserted_edges.insert(update_id);
            sent_edges.insert(update_id);
            if (wroot == rank_) {
              local_edges[rank_].emplace_back(wlabel, vlabel);
              inserted_edges.insert(wlabel + num_global_vertices * vlabel);
              sent_edges.insert(wlabel + num_global_vertices * vlabel);
            }
          } else if (sent_edges.find(update_id) == end(sent_edges)) {
            // Local propagation with shortcuts
            VertexID parent = g_.GetShortcutForLabel(vlabel);
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            // Send edge
            sent_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, parent, send_buffers);
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    // Sentinel
    removed_edges_.emplace(std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max());
  }

  void FindLocalConflictingEdges(VertexID num_global_vertices,
                                 google::dense_hash_set<VertexID> &sent_edges,
                                 std::vector<std::vector<VertexID>> *send_buffers) {
    g_.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g_.GetVertexLabel(v);
      g_.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g_.GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + num_global_vertices * wlabel;

          if (sent_edges.find(update_id) == sent_edges.end()) {
            // Local propagation
            VertexID parent = g_.GetParent(v);
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            // Send edge
            sent_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, g_.GetVertexRoot(w), parent, send_buffers);
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    removed_edges_.emplace(std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max());
  }

  PEID FindAdjacentPEs(std::vector<bool> &is_adj) {
    // Get adjacency (otherwise we get deadlocks with added edges)
    PEID num_adj = 0;
    for (PEID pe = 0; pe < size_; pe++) {
      is_adj[pe] = g_.IsAdjacentPE(pe);
      if (is_adj[pe]) num_adj++;
    }
    return num_adj;
  }

  void ReceiveMessages(PEID adjacent_pes,
                       std::vector<MPI_Request*> &requests,
                       std::vector<std::vector<VertexID>> *send_buffers,
                       std::vector<std::vector<VertexID>> &receive_buffers) {
    // Receive edges
    PEID messages_recv = 0;
    int message_length = 0;
    for (int i = 0; i < size_; ++i) receive_buffers[i].clear();
    receive_buffers[rank_] = (*send_buffers)[rank_];
    while (messages_recv < adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
      MPI_Get_count(&st, MPI_VERTEX, &message_length);
      messages_recv++;

      receive_buffers[st.MPI_SOURCE].resize(message_length);
      MPI_Status rst{};
      MPI_Recv(&receive_buffers[st.MPI_SOURCE][0], 
               message_length, MPI_VERTEX, 
               st.MPI_SOURCE, 0, MPI_COMM_WORLD, &rst);
    }

    for (unsigned int i = 0; i < requests.size(); ++i) {
      MPI_Status st;
      MPI_Wait(requests[i], &st);
      delete requests[i];
    }
    requests.clear();
  }

  void SendMessages(std::vector<bool> &is_adj,
                    std::vector<std::vector<VertexID>> *send_buffers,
                    std::vector<MPI_Request*> &requests) {
    for (PEID pe = 0; pe < size_; ++pe) {
      if (is_adj[pe]) {
        if ((*send_buffers)[pe].size() == 0) 
          (*send_buffers)[pe].emplace_back(0);
        auto *req = new MPI_Request();
        MPI_Isend((*send_buffers)[pe].data(), 
                  static_cast<int>((*send_buffers)[pe].size()), 
                  MPI_VERTEX, pe, 0, MPI_COMM_WORLD, req);
        requests.emplace_back(req);
      }
    }
  }

  int ProcessExponentialMessages(VertexID num_global_vertices,
                                 google::dense_hash_set<VertexID> &inserted_edges, 
                                 std::vector<std::vector<std::pair<VertexID, VertexID>>> &new_edges,
                                 google::dense_hash_set<VertexID> &propagated_edges,
                                 std::vector<std::vector<VertexID>> &receive_buffers,
                                 std::vector<std::vector<VertexID>> *send_buffers) {
    int propagate = 0;
    // Receive edges and apply updates
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffers[pe].size() < 4) continue;
      for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
        VertexID vlabel = receive_buffers[pe][i];
        VertexID wlabel = receive_buffers[pe][i + 1];
        VertexID wroot = receive_buffers[pe][i + 2];
        VertexID link = receive_buffers[pe][i + 3];
        VertexID update_id = vlabel + num_global_vertices * wlabel;

        // Continue if already inserted
        if (inserted_edges.find(update_id) != end(inserted_edges)) continue;
        if (propagated_edges.find(update_id) != end(propagated_edges)) continue;

        // If vlabel is on same PE just insert the edge
        if (g_.IsLocalFromGlobal(vlabel)) {
          new_edges[wroot].emplace_back(vlabel, wlabel);
          inserted_edges.insert(update_id);
          propagated_edges.insert(update_id);
          if (wroot == rank_) {
            new_edges[rank_].emplace_back(wlabel, vlabel);
            inserted_edges.insert(wlabel + num_global_vertices * vlabel);
            propagated_edges.insert(wlabel + num_global_vertices * vlabel);
          }
        } else {
          // Local propagation with shortcuts
          VertexID parent = g_.GetShortcutForLabel(vlabel);
          PEID pe = g_.GetPE(g_.GetLocalID(parent));
          // Send edge
          propagated_edges.insert(update_id);
          PlaceInBuffer(pe, vlabel, wlabel, wroot, parent, send_buffers);
          propagate = 1;
        }
      }
    }
    return !propagate;
  }

  int ProcessLocalMessages(VertexID num_global_vertices,
                           google::dense_hash_set<VertexID> &inserted_edges, 
                           std::vector<std::vector<std::pair<VertexID, VertexID>>> &new_edges,
                           google::dense_hash_set<VertexID> &propagated_edges,
                           std::vector<std::vector<VertexID>> &receive_buffers,
                           std::vector<std::vector<VertexID>> *send_buffers) {
    int propagate = 0;
    // Receive edges and apply updates
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffers[pe].size() < 4) continue;
      for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
        VertexID vlabel = receive_buffers[pe][i];
        VertexID wlabel = receive_buffers[pe][i + 1];
        VertexID wroot = receive_buffers[pe][i + 2];
        VertexID link = receive_buffers[pe][i + 3];
        VertexID update_id = vlabel + num_global_vertices * wlabel;

        // Continue if already inserted
        if (inserted_edges.find(update_id) != end(inserted_edges)) continue;
        if (propagated_edges.find(update_id) != end(propagated_edges)) continue;

        // Get link information
        VertexID parent = g_.GetParent(g_.GetLocalID(link));
        PEID pe = g_.GetPE(g_.GetLocalID(parent));

        if (g_.IsLocalFromGlobal(vlabel)) {
          new_edges[wroot].emplace_back(vlabel, wlabel);
          inserted_edges.insert(update_id);
          propagated_edges.insert(update_id);
          if (wroot == rank_) {
            new_edges[rank_].emplace_back(wlabel, vlabel);
            inserted_edges.insert(wlabel + num_global_vertices * vlabel);
            propagated_edges.insert(wlabel + num_global_vertices * vlabel);
          }
        } else {
          if (g_.GetVertexLabel(g_.GetLocalID(link)) == vlabel) {
            propagated_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, parent, send_buffers);
            propagate = 1;
          } else {
            // Parent has to be connected to vlabel (N(N(v))
            VertexID local_vlabel = g_.GetLocalID(vlabel);
            pe = g_.GetPE(local_vlabel);
            // Send edge
            propagated_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, parent, send_buffers);
            propagate = 1;
          }
        }
      }
    }
    return ! propagate;
  }

  void PlaceInBuffer(PEID pe,
                     VertexID vlabel,
                     VertexID wlabel,
                     PEID wroot,
                     VertexID parent,
                     std::vector<std::vector<VertexID>> *send_buffers) {
    (*send_buffers)[pe].emplace_back(vlabel);
    (*send_buffers)[pe].emplace_back(wlabel);
    (*send_buffers)[pe].emplace_back(wroot);
    (*send_buffers)[pe].emplace_back(parent);
  }

  void UpdateActiveVertices() {
    for (VertexID i = 0; i < inactive_level_.size(); ++i) {
      if (inactive_level_[i] == -1) inactive_level_[i] = contraction_level_ - 1;
    }
  }

  void InsertEdges(std::vector<std::vector<std::pair<VertexID, VertexID>>> &new_edges) {
    for (PEID pe = 0; pe < size_; pe++) {
      for (auto &e : new_edges[pe]) {
        VertexID vlabel = e.first;
        VertexID wlabel = e.second;
        VertexID vlocal = g_.GetLocalID(vlabel);
        // TODO: Check if this is needed
        if (!g_.IsGhostFromGlobal(wlabel)) g_.AddGhostVertex(wlabel);
        // Add edge
        g_.AddEdge(vlocal, wlabel, pe);
        VertexID wlocal = g_.GetLocalID(wlabel);
        // Vertices remain active
        inactive_level_[vlocal] = -1;
        inactive_level_[wlocal] = -1;
        g_.SetVertexMessage(vlocal, {
                              std::numeric_limits<VertexID>::max() - 1,
                              vlabel, 
#ifdef TIEBREAK_DEGREE
                              0,
#endif
                              rank_
                           });

        g_.SetVertexMessage(wlocal, {
                              std::numeric_limits<VertexID>::max() - 1, 
                              wlabel, 
#ifdef TIEBREAK_DEGREE
                              0,
#endif
                              pe
                            });
      }
    }
  }

  void UpdateGraphVertices() {
    for (VertexID i = 0; i < inactive_level_.size(); ++i) {
      g_.SetActive(i, inactive_level_[i] == -1);
    }
  }

  void SwapBuffers(std::vector<std::vector<VertexID>>* buffers,
                   std::vector<std::vector<VertexID>> &a_buffers,
                   std::vector<std::vector<VertexID>> &b_buffers) {
    // Switch buffers
    if (buffers == &a_buffers) {
      for (int i = 0; i < size_; ++i) b_buffers[i].clear();
      buffers = &b_buffers;
    } else {
      for (int i = 0; i < size_; ++i) a_buffers[i].clear();
      buffers = &a_buffers;
    }
  }

  void UndoContraction() {
    // TODO [FIX]: Does not work if graph is completely empty
    // Remove last sentinel
    removed_edges_.pop();
    while (contraction_level_ > 0) {
      // Remove current edges from current level
      google::dense_hash_map<VertexID, VertexID> current_components; 
      current_components.set_empty_key(-1);
      g_.ForallLocalVertices([&](VertexID v) {
        g_.RemoveAllEdges(v);
      });

      // Decrease level
      contraction_level_--;

      // Update active graph portion
      EnableActiveVertices();
      AddRemovedEdges();
      UpdateGraphVertices();

      // Update local labels
      g_.ForallLocalVertices([&](VertexID v) {
        g_.ForceVertexPayload(v, {0, 
                               g_.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                               0,
#endif
                               rank_});
      });

      // Propagate labels
      int converged_globally = 0;
      while (converged_globally == 0) {
        int converged_locally = 1;
        // Receive variates
        g_.SendAndReceiveGhostVertices();

        // Send current label from root
        g_.ForallLocalVertices([&](VertexID v) {
          VertexID parent = g_.GetParent(v);
          if (g_.GetVertexLabel(g_.GetLocalID(parent)) != g_.GetVertexLabel(v)) {
            g_.SetVertexPayload(v, {0, 
                                 g_.GetVertexLabel(g_.GetLocalID(parent)), 
#ifdef TIEBREAK_DEGREE
                                 0,
#endif
                                 rank_});
            converged_locally = 0;
          }
        });

        // Check if all PEs are done
        MPI_Allreduce(&converged_locally,
                      &converged_globally,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
      }

      // Vertices at current level are roots at previous one
      g_.ForallLocalVertices([&](VertexID v) {
        g_.SetParent(v, g_.GetGlobalID(v));
      });
    }
  }

  void EnableActiveVertices() {
    for (VertexID i = 0; i < inactive_level_.size(); ++i) {
      if (inactive_level_[i] == contraction_level_) inactive_level_[i] = - 1;
    }
  }

  void AddRemovedEdges() {
    while (!removed_edges_.empty()) {
      auto e = removed_edges_.top();
      removed_edges_.pop();
      if (std::get<0>(e) == std::numeric_limits<VertexID>::max()) break;
      g_.AddEdge(std::get<0>(e), std::get<1>(e), g_.GetPE(g_.GetLocalID(std::get<1>(e))));
    }
  }

 private:
  // Original graph instance
  DynamicGraphAccess &g_;

  // Network information
  PEID rank_, size_;

  // Variables
  VertexID contraction_level_;
  std::stack<std::pair<VertexID, VertexID>> removed_edges_;
  std::vector<short> inactive_level_;
};

#endif
