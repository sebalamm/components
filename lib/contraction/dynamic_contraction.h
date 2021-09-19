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
#include "dynamic_graph_comm.h"
#include "static_graph.h"
#include "edge_hash.h"

class DynamicContraction {
 public:
  DynamicContraction(DynamicGraphCommunicator &g, const Config &conf, const PEID rank, const PEID size)
      : g_(g), 
        rank_(rank), 
        size_(size),
        config_(conf),
        global_num_vertices_(0),
        contraction_level_(0),
        comm_time_(0.0),
        send_volume_(0),
        recv_volume_(0) { 
    is_active_.set_empty_key(EmptyKey);
    is_active_.set_deleted_key(DeleteKey);
    send_buffers_.set_empty_key(EmptyKey);
    send_buffers_.set_deleted_key(DeleteKey);
    receive_buffers_.set_empty_key(EmptyKey);
    receive_buffers_.set_deleted_key(DeleteKey);
    inserted_edges_.set_empty_key(EmptyKey);
    inserted_edges_.set_deleted_key(DeleteKey);
    propagated_edges_.set_empty_key(EmptyKey);
    propagated_edges_.set_deleted_key(DeleteKey);
    edges_to_add_.set_empty_key(EmptyKey);
    edges_to_add_.set_deleted_key(DeleteKey);
  }

  virtual ~DynamicContraction() = default;

  void ExponentialContraction() {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();
    global_num_vertices_ = g_.GatherNumberOfGlobalVertices();

    g_.ForallVertices([&](VertexID v) {
      is_active_[contraction_level_][v] = true;
    });

    contraction_timer_.Restart();
    FindExponentialConflictingEdges();
    
    // std::cout << "[STATUS] |--- R" << rank_ << " Detecting conflicting edges took " 
    //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    // Propagate edge buffers until all vertices are converged
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {

      contraction_timer_.Restart();
      comm_time_ += CommunicationUtility::AllToAll(send_buffers_, receive_buffers_, rank_, size_, ContractionTag + local_iterations, config_.use_regular);
      // std::cout << "[STATUS] |---- R" << rank_ << " Message exchange took " 
      //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      send_volume_ += CommunicationUtility::ClearBuffers(send_buffers_);

      contraction_timer_.Restart();
      int converged_locally = ProcessExponentialMessages();
      // std::cout << "[STATUS] |---- R" << rank_ << " Processing messages took " 
      //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers_);

      // Check if all PEs are done
      contraction_timer_.Restart();
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
      local_iterations++;
      // std::cout << "[STATUS] |---- R" << rank_ << " Convergence test took " 
      //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    // std::cout << "[STATUS] |---- R" << rank_ << " Propagation done " 
    //           << "[INFO] rounds "  << local_iterations << " "
    //           << "[TIME] " << propagation_timer.Elapsed() << std::endl;
    contraction_level_++;

    // Clear propagation buffers
    inserted_edges_.clear();
    propagated_edges_.clear();

    // Remove all edges and disable interface vertices
    contraction_timer_.Restart();
    g_.ForallVertices([&](VertexID v) { 
      g_.RemoveAllEdges(v); 
      g_.SetInterface(v, false);
    });
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
    ResetActiveVertices();

    // Insert edges and keep corresponding vertices
    InsertAndClearEdges();
    // std::cout << "[STATUS] |---- R" << rank_ << " Updating edges took " 
    //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    // Reenable vertices that have at least one edge
    UpdateGraphVertices();
  }

  void DirectContraction() {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();
    global_num_vertices_ = g_.GatherNumberOfGlobalVertices();

    // Update with new vertices added during last contraction
    g_.ForallVertices([&](VertexID v) {
      is_active_[contraction_level_][v] = true;
    });

    contraction_timer_.Restart();
    FindDirectConflictingEdges();
    
    // std::cout << "[STATUS] |--- R" << rank_ << " Detecting conflicting edges took " 
    //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    // Propagate edge buffers until all vertices are converged
    int num_requests = 0;

    comm_time_ += CommunicationUtility::AllToAll(send_buffers_, receive_buffers_, rank_, size_, ContractionTag, config_.use_regular);
    send_volume_ += CommunicationUtility::ClearBuffers(send_buffers_);
    ProcessDirectMessages();
    recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers_);

    inserted_edges_.clear();
    propagated_edges_.clear();

    // Insert edges and keep corresponding vertices
    contraction_timer_.Restart();
    g_.ForallVertices([&](VertexID v) { 
      g_.RemoveAllEdges(v); 
      g_.SetInterface(v, false);
    });
    contraction_level_++;

    ResetActiveVertices();
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
    InsertAndClearEdges();

    // std::cout << "[STATUS] |---- R" << rank_ << " Updating edges took " 
    //           << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    UpdateGraphVertices();
  }

  void LocalContraction(google::dense_hash_map<VertexID, VertexID> &initial_parents) {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();
    global_num_vertices_ = g_.GatherNumberOfGlobalVertices();

    // Update with new vertices added during last contraction
    g_.ForallVertices([&](VertexID v) {
      is_active_[contraction_level_][v] = true;
    });

    FindLocalConflictingEdges();

    // Propagate edge buffers until all vertices are converged
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {
      comm_time_ += CommunicationUtility::AllToAll(send_buffers_, receive_buffers_, rank_, size_, ContractionTag + local_iterations, config_.use_regular);
      send_volume_ += CommunicationUtility::ClearBuffers(send_buffers_);

      int converged_locally = ProcessLocalMessages(initial_parents);
      recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers_);


      // Check if all PEs are done
      contraction_timer_.Restart();
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
      local_iterations++;
      if (rank_ == ROOT || config_.print_verbose)
        std::cout << "[STATUS] |--- R" << rank_ << " Convergence test took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    if (rank_ == ROOT || config_.print_verbose) 
      std::cout << "[STATUS] |---- R" << rank_ << " Propagation done " 
                << "[TIME] " << propagation_timer.Elapsed() << std::endl;

    inserted_edges_.clear();
    propagated_edges_.clear();

    // Insert edges and keep corresponding vertices
    g_.ForallVertices([&](VertexID v) { 
      g_.RemoveAllEdges(v); 
      g_.SetInterface(v, false);
    });
    contraction_level_++;

    ResetActiveVertices();
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
    InsertAndClearEdges();

    // max_degree_computed_ = false;
    UpdateGraphVertices();
  }

  void FindExponentialConflictingEdges() {
    g_.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g_.GetVertexLabel(v);
      g_.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g_.GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + global_num_vertices_ * wlabel;
          PEID wroot = g_.GetVertexRoot(w);

          if (inserted_edges_.find(update_id) == end(inserted_edges_) 
                && g_.IsLocalFromGlobal(vlabel)) {
            edges_to_add_[wroot].emplace_back(vlabel);
            edges_to_add_[wroot].emplace_back(wlabel);
            inserted_edges_.insert(update_id);
            propagated_edges_.insert(update_id);
            if (wroot == rank_) {
              edges_to_add_[rank_].emplace_back(wlabel);
              edges_to_add_[rank_].emplace_back(vlabel);
              inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
              propagated_edges_.insert(wlabel + global_num_vertices_ * vlabel);
            }
          } else if (propagated_edges_.find(update_id) == end(propagated_edges_)) {
            // Local propagation with shortcuts
            VertexID parent = g_.GetShortcutForLabel(vlabel);
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            // Send edge
            propagated_edges_.insert(update_id);
            send_buffers_[pe].emplace_back(vlabel);
            send_buffers_[pe].emplace_back(wlabel);
            send_buffers_[pe].emplace_back(wroot);
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    // Sentinel
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
    receive_buffers_[rank_].swap(send_buffers_[rank_]);
    send_buffers_[rank_].clear();
  }

  void FindDirectConflictingEdges() {
    g_.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g_.GetVertexLabel(v);
      g_.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g_.GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + global_num_vertices_ * wlabel;
          PEID wroot = g_.GetVertexRoot(w);

          if (inserted_edges_.find(update_id) == end(inserted_edges_) 
                && g_.IsLocalFromGlobal(vlabel)) {
            edges_to_add_[wroot].emplace_back(vlabel);
            edges_to_add_[wroot].emplace_back(wlabel);
            inserted_edges_.insert(update_id);
            propagated_edges_.insert(update_id);
            if (wroot == rank_) {
              edges_to_add_[rank_].emplace_back(wlabel);
              edges_to_add_[rank_].emplace_back(vlabel);
              inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
              propagated_edges_.insert(wlabel + global_num_vertices_ * vlabel);
            }
          } else if (propagated_edges_.find(update_id) == end(propagated_edges_)) {
            // Direct propagation to root
            PEID pe = g_.GetVertexRoot(v);
            // Send edge
            propagated_edges_.insert(update_id);
            send_buffers_[pe].emplace_back(vlabel);
            send_buffers_[pe].emplace_back(wlabel);
            send_buffers_[pe].emplace_back(wroot);
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    // Sentinel
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
    receive_buffers_[rank_].swap(send_buffers_[rank_]);
    send_buffers_[rank_].clear();
  }

  void FindLocalConflictingEdges() {
    g_.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g_.GetVertexLabel(v);
      g_.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g_.GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + global_num_vertices_ * wlabel;
          PEID wroot = g_.GetVertexRoot(w);

          if (inserted_edges_.find(update_id) == end(inserted_edges_) 
                && g_.IsLocalFromGlobal(vlabel)) {
            edges_to_add_[wroot].emplace_back(vlabel);
            edges_to_add_[wroot].emplace_back(wlabel);
            inserted_edges_.insert(update_id);
            propagated_edges_.insert(update_id);
            if (wroot == rank_) {
              edges_to_add_[rank_].emplace_back(wlabel);
              edges_to_add_[rank_].emplace_back(vlabel);
              inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
              propagated_edges_.insert(wlabel + global_num_vertices_ * vlabel);
            }
          } else if (propagated_edges_.find(update_id) == propagated_edges_.end()) {
            // Local propagation
            VertexID parent = g_.GetParent(v);
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            // Send edge
            propagated_edges_.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, g_.GetVertexRoot(w), parent);
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
    receive_buffers_[rank_].swap(send_buffers_[rank_]);
    send_buffers_[rank_].clear();
  }

  int ProcessExponentialMessages() {
    int propagate = 0;
    // Receive edges and apply updates
    for (auto &kv : receive_buffers_) {
      auto &buffer = kv.second;
      for (int i = 0; i < buffer.size(); i += 3) {
        VertexID vlabel = buffer[i];
        VertexID wlabel = buffer[i + 1];
        VertexID wroot = buffer[i + 2];

        // Continue if already inserted
        VertexID update_id = vlabel + global_num_vertices_ * wlabel;
        if (inserted_edges_.find(update_id) != end(inserted_edges_)) continue;
        if (propagated_edges_.find(update_id) != end(propagated_edges_)) continue;

        // If vlabel is on same PE just insert the edge
        if (g_.IsLocalFromGlobal(vlabel)) {
          edges_to_add_[wroot].emplace_back(vlabel);
          edges_to_add_[wroot].emplace_back(wlabel);
          inserted_edges_.insert(update_id);
          propagated_edges_.insert(update_id);
          if (wroot == rank_) {
            edges_to_add_[rank_].emplace_back(wlabel);
            edges_to_add_[rank_].emplace_back(vlabel);
            inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
            propagated_edges_.insert(wlabel + global_num_vertices_ * vlabel);
          }
        } else {
          // Local propagation with shortcuts
          VertexID parent = g_.GetShortcutForLabel(vlabel);
          PEID pe = g_.GetPE(g_.GetLocalID(parent));
          // Send edge
          propagated_edges_.insert(update_id);
          send_buffers_[pe].emplace_back(vlabel);
          send_buffers_[pe].emplace_back(wlabel);
          send_buffers_[pe].emplace_back(wroot);
          propagate = 1;
        }
      }
    }
    return !propagate;
  }

  int ProcessDirectMessages() {
    // Handle receive buffers
    for (auto &kv : receive_buffers_) {
      auto &buffer = kv.second;
      for (int i = 0; i < buffer.size(); i += 3) {
        VertexID vlabel = buffer[i];
        VertexID wlabel = buffer[i + 1];
        VertexID wroot = buffer[i + 2];

        // Continue if already inserted
        VertexID update_id = vlabel + global_num_vertices_ * wlabel;
        if (inserted_edges_.find(update_id) != end(inserted_edges_)) continue;

        // If vlabel is on same PE just insert the edge
        if (g_.IsLocalFromGlobal(vlabel)) {
          edges_to_add_[wroot].emplace_back(vlabel);
          edges_to_add_[wroot].emplace_back(wlabel);
          inserted_edges_.insert(update_id);
          if (wroot == rank_) {
            edges_to_add_[rank_].emplace_back(wlabel);
            edges_to_add_[rank_].emplace_back(vlabel);
            inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
          }
        } else {
          std::cout << "R" << rank_ << " incorrect message in direct contraction" << std::endl;
          exit(1);
        }
      }
    }
  }

  int ProcessLocalMessages(google::dense_hash_map<VertexID, VertexID> &initial_parents) {
    int propagate = 0;
    // Receive edges and apply updates
    for (auto &kv : receive_buffers_) {
      auto &buffer = kv.second;
      for (int i = 0; i < buffer.size(); i += 4) {
        VertexID vlabel = buffer[i];
        VertexID wlabel = buffer[i + 1];
        VertexID wroot = buffer[i + 2];
        VertexID link = buffer[i + 3];

        // Continue if already inserted
        VertexID update_id = vlabel + global_num_vertices_ * wlabel;
        if (inserted_edges_.find(update_id) != end(inserted_edges_)) continue;
        if (propagated_edges_.find(update_id) != end(propagated_edges_)) continue;

        // Get link information
        if (g_.IsLocalFromGlobal(vlabel)) {
          edges_to_add_[wroot].emplace_back(vlabel);
          edges_to_add_[wroot].emplace_back(wlabel);
          inserted_edges_.insert(update_id);
          propagated_edges_.insert(update_id);
          if (wroot == rank_) {
            edges_to_add_[rank_].emplace_back(wlabel);
            edges_to_add_[rank_].emplace_back(vlabel);
            inserted_edges_.insert(wlabel + global_num_vertices_ * vlabel);
            propagated_edges_.insert(wlabel + global_num_vertices_ * vlabel);
          }
        } else {
          if (g_.GetVertexLabel(g_.GetLocalID(link)) == vlabel) {
            VertexID parent = g_.GetParent(g_.GetLocalID(link));
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            propagated_edges_.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, parent);
            propagate = 1;
          } else {
            // Parent has to be connected to vlabel (N(N(v))
            VertexID initial_parent = initial_parents[g_.GetLocalID(vlabel)];
            PEID pe = g_.GetPE(initial_parent);
            // Send edge
            propagated_edges_.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, initial_parent);
            propagate = 1;
          }
        }
      }
    }
    return !propagate;
  }

  inline void PlaceInBuffer(PEID pe, VertexID vlabel, VertexID wlabel, PEID wroot, VertexID parent) {
    send_buffers_[pe].emplace_back(vlabel);
    send_buffers_[pe].emplace_back(wlabel);
    send_buffers_[pe].emplace_back(wroot);
    send_buffers_[pe].emplace_back(parent);
  }

  void ResetActiveVertices() {
    // Set vertices from previously level to inactive
    std::vector<VertexID> prev_active;
    for (auto &kv : is_active_[contraction_level_ - 1]) {
      prev_active.emplace_back(kv.first);
    }
    // Decouple loop
    for (const VertexID &v : prev_active) {
      is_active_[contraction_level_][v] = false;
    }
  }

  void InsertAndClearEdges() {
    for (auto &kv: edges_to_add_) {
      PEID pe = kv.first;
      auto &buffer = kv.second;
      for (VertexID i = 0; i < buffer.size(); i += 2) {
        VertexID vlabel = buffer[i];
        VertexID wlabel = buffer[i + 1];
        VertexID vlocal = g_.GetLocalID(vlabel);
        if (!g_.IsLocalFromGlobal(wlabel) && !g_.IsGhostFromGlobal(wlabel)) {
          g_.AddGhostVertex(wlabel, pe);
        }
        VertexID wlocal = g_.GetLocalID(wlabel);
        // Add edge
        g_.AddEdge(vlocal, wlabel, pe);
        // Add reverse add for ghosts
        if (g_.IsGhost(wlocal)) {
          g_.AddEdge(wlocal, vlabel, rank_);
        }
        // std::cout << "R" << rank_ << " add contraction edge (" << vlabel << "," << wlabel << "(R" << pe << "))" << std::endl;
        // Set corresponding vertices to active
        is_active_[contraction_level_][vlocal] = true;
        is_active_[contraction_level_][wlocal] = true;
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
      buffer.clear();
    }
    for (auto &kv : edges_to_add_) {
      edges_to_add_[kv.first].clear();
    }
    edges_to_add_.clear();
  }

  void UpdateGraphVertices() {
    // Activate remaining vertices in graph representation
    for (auto &kv : is_active_[contraction_level_]) {
      g_.SetActive(kv.first, kv.second);
    }
  }

  void UndoContraction() {
    // Remove last sentinel
    removed_edges_.pop();
    while (contraction_level_ > 0) {
      contraction_timer_.Restart();
      // Remove current edges from current level
      google::dense_hash_map<VertexID, VertexID> current_components; 
      current_components.set_empty_key(EmptyKey);
      current_components.set_deleted_key(DeleteKey);
      g_.ForallVertices([&](VertexID v) {
        g_.RemoveAllEdges(v);
        g_.SetInterface(v, false);
      });

      // Decrease level
      contraction_level_--;

      // Update active graph portion
      AddRemovedEdges();
      UpdateGraphVertices();
      if (rank_ == ROOT || config_.print_verbose)
        std::cout << "[STATUS] |-- R" << rank_ << " Updating edges took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

      contraction_timer_.Restart();
      // Update local labels
      g_.ForallLocalVertices([&](VertexID v) {
        g_.ForceVertexPayload(v, {0, 
                               g_.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                               0,
#endif
                               rank_});
      });
      if (rank_ == ROOT || config_.print_verbose)
        std::cout << "[STATUS] |-- R" << rank_ << " Updating payloads took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

      // Propagate labels
      int converged_globally = 0;
      while (converged_globally == 0) {
        int converged_locally = 1;
        contraction_timer_.Restart();
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
        if (rank_ == ROOT || config_.print_verbose) 
          std::cout << "[STATUS] |--- R" << rank_ << " Message exchange took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

        // Check if all PEs are done
        contraction_timer_.Restart();
        comm_timer_.Restart();
        MPI_Allreduce(&converged_locally,
                      &converged_globally,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        comm_time_ += comm_timer_.Elapsed();
        if (rank_ == ROOT || config_.print_verbose)
          std::cout << "[STATUS] |--- R" << rank_ << " Convergence test took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }
      // Vertices at current level are roots at previous one
      g_.ForallLocalVertices([&](VertexID v) {
        g_.SetParent(v, g_.GetGlobalID(v));
      });
    }
  }

  // TODO: Does this work with ghosts are handled now?
  void AddRemovedEdges() {
    while (!removed_edges_.empty()) {
      auto e = removed_edges_.top();
      removed_edges_.pop();
      if (std::get<0>(e) == std::numeric_limits<VertexID>::max() - 1) break;
      g_.AddEdge(std::get<0>(e), std::get<1>(e), g_.GetPE(g_.GetLocalID(std::get<1>(e))));
      if (!g_.IsLocalFromGlobal(std::get<1>(e))) {
        g_.AddEdge(g_.GetLocalID(std::get<1>(e)), g_.GetGlobalID(std::get<0>(e)), rank_);
      }
    }
  }

  inline float GetCommTime() {
    return comm_time_;
  }

  inline VertexID GetSendVolume() {
    return send_volume_;
  }

  inline VertexID GetReceiveVolume() {
    return recv_volume_;
  }

 private:
  // Original graph instance
  DynamicGraphCommunicator &g_;

  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Variables
  VertexID contraction_level_;
  VertexID global_num_vertices_;
  std::stack<std::pair<VertexID, VertexID>> removed_edges_;
  google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, bool>> is_active_;

  // Buffers
  google::dense_hash_map<PEID, VertexBuffer> send_buffers_;
  google::dense_hash_map<PEID, VertexBuffer> receive_buffers_;

  // Unique sets
  google::dense_hash_set<VertexID> inserted_edges_;
  google::dense_hash_set<VertexID> propagated_edges_;
  google::dense_hash_map<PEID, std::vector<VertexID>> edges_to_add_;

  // Statistics
  float comm_time_;
  Timer contraction_timer_;
  Timer comm_timer_;
  VertexID send_volume_;
  VertexID recv_volume_;
};

#endif
