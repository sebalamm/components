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
  DynamicContraction(DynamicGraphCommunicator &g, const PEID rank, const PEID size)
      : g_(g), 
        rank_(rank), 
        size_(size),
        contraction_level_(0),
        comm_time_(0.0) { 
    inactive_level_.set_empty_key(-1);
  }

  virtual ~DynamicContraction() = default;

  void ExponentialContraction() {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();
    edge_counter_ = 0;

    contraction_timer_.Restart();
    VertexID num_global_vertices = g_.GatherNumberOfGlobalVertices();
    VertexID num_vertices = g_.GetNumberOfVertices();

    // Update with new vertices added during last contraction
    g_.ForallVertices([&](VertexID v) {
      inactive_level_[v] = -1;
    });

    // Determine edges to communicate
    // Gather labels to communicate
    google::dense_hash_set<VertexID> send_ids; 
    send_ids.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<VertexID>> send_buffers_a(size_);
    std::vector<std::vector<VertexID>> send_buffers_b(size_);
    std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
    std::vector<std::vector<VertexID>> receive_buffers(size_);

    // TODO: Fix iteration
    for (int i = 0; i < size_; ++i) {
      send_buffers_a[i].clear();
      send_buffers_b[i].clear();
      receive_buffers[i].clear();
    }

    google::dense_hash_set<VertexID> inserted_edges; 
    inserted_edges.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);

    std::cout << "[STATUS] |--- R" << rank_ << " Allocation took " 
              << "[TIME] " << contraction_timer_.Elapsed() << std::endl;


    contraction_timer_.Restart();
    FindExponentialConflictingEdges(num_global_vertices, 
                                    inserted_edges, 
                                    edges_to_add, 
                                    send_ids, 
                                    current_send_buffers);
    
    std::cout << "[STATUS] |--- R" << rank_ << " Detecting conflicting edges took " 
              << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    // TODO: Fix size
    std::vector<bool> is_adj(size_);
    PEID num_adj = FindAdjacentPEs(is_adj);

    // Propagate edge buffers until all vertices are converged
    std::vector<MPI_Request> requests;
    requests.clear();
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {

      contraction_timer_.Restart();
      comm_timer_.Restart();
      SendMessages(is_adj, current_send_buffers, requests);
      ReceiveMessages(num_adj, requests, current_send_buffers, receive_buffers);
      SwapBuffers(current_send_buffers, send_buffers_a, send_buffers_b);
      comm_time_ += comm_timer_.Elapsed();

      std::cout << "[STATUS] |---- R" << rank_ << " Message exchange took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

      contraction_timer_.Restart();
      int converged_locally = ProcessExponentialMessages(num_global_vertices, inserted_edges, edges_to_add, 
                                                         send_ids, receive_buffers, current_send_buffers);
      std::cout << "[STATUS] |---- R" << rank_ << " Processing messages took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

      std::cout << "[STATUS] |---- R" << rank_ << " Processing messages took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

      // Check if all PEs are done
      // if (++local_iterations % 6 == 0) {
      contraction_timer_.Restart();
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
      // } 
      local_iterations++;
        std::cout << "[STATUS] |---- R" << rank_ << " Convergence test took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    std::cout << "[STATUS] |---- R" << rank_ << " Propagation done " 
              << "[INFO] rounds "  << local_iterations << " "
              << "[TIME] " << propagation_timer.Elapsed() << std::endl;

    // Insert edges and keep corresponding vertices
    contraction_timer_.Restart();
    g_.ForallLocalVertices([&](VertexID v) { g_.RemoveAllEdges(v); });
    contraction_level_++;

    UpdateActiveVertices();
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
    InsertEdges(edges_to_add);

    std::cout << "[STATUS] |---- R" << rank_ << " Updating edges took " 
              << "[TIME] " << contraction_timer_.Elapsed() 
              << " " << edge_counter_ << " edges" << std::endl;

    UpdateGraphVertices();
  }

  void DirectContraction() {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();
    edge_counter_ = 0;

    contraction_timer_.Restart();
    VertexID num_global_vertices = g_.GatherNumberOfGlobalVertices();
    VertexID num_vertices = g_.GetNumberOfVertices();

    // Determine edges to communicate
    // Gather labels to communicate
    google::dense_hash_set<VertexID> send_ids; 
    send_ids.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<VertexID>> send_buffers(size_);
    std::vector<std::vector<VertexID>> receive_buffers(size_);

    // TODO: Fix iteration
    for (int i = 0; i < size_; ++i) {
      send_buffers[i].clear();
      receive_buffers[i].clear();
    }

    google::dense_hash_set<VertexID> inserted_edges; 
    inserted_edges.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);

    std::cout << "[STATUS] |--- R" << rank_ << " Allocation took " 
              << "[TIME] " << contraction_timer_.Elapsed() << std::endl;


    contraction_timer_.Restart();
    FindDirectConflictingEdges(num_global_vertices, 
                               inserted_edges, 
                               edges_to_add, 
                               send_ids, 
                               send_buffers);
    
    std::cout << "[STATUS] |--- R" << rank_ << " Detecting conflicting edges took " 
              << "[TIME] " << contraction_timer_.Elapsed() << std::endl;

    // Propagate edge buffers until all vertices are converged
    int num_requests = 0;

    comm_timer_.Restart();
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (send_buffers[pe].size() > 0) num_requests++; 
    }
    std::vector<MPI_Request> requests(num_requests);

    int req = 0;
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (send_buffers[pe].size() > 0) {
        MPI_Issend(send_buffers[pe].data(), 
                   static_cast<int>(send_buffers[pe].size()), 
                   MPI_VERTEX, pe, 1000 * size_ + pe, MPI_COMM_WORLD, &requests[req++]);
        if (pe == rank_) {
          std::cout << "R" << rank_ << " ERROR selfmessage" << std::endl;
          exit(1);
        }
      } 
    }

    std::vector<MPI_Status> statuses(num_requests);
    int isend_done = 0;
    while (isend_done == 0) {
      // Check for messages
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, 1000 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          std::vector<VertexID> message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
      // Check if all ISend successful
      isend_done = 0;
      MPI_Testall(num_requests, requests.data(), &isend_done, statuses.data());
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

    int ibarrier_done = 0;
    while (ibarrier_done == 0) {
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, 1000 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          std::vector<VertexID> message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
        
      // Check if all reached Ibarrier
      MPI_Status tst{};
      MPI_Test(&barrier_request, &ibarrier_done, &tst);
      if (tst.MPI_ERROR != MPI_SUCCESS) {
        std::cout << "R" << rank_ << " mpi_test (barrier) failed" << std::endl;
        exit(1);
      }
    }
    comm_time_ += comm_timer_.Elapsed();

    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffers[pe].size() < 3) continue;
      for (int i = 0; i < receive_buffers[pe].size(); i += 3) {
        VertexID vlabel = receive_buffers[pe][i];
        VertexID wlabel = receive_buffers[pe][i + 1];
        VertexID wroot = receive_buffers[pe][i + 2];

        // Check for dummy message
        if (vlabel == std::numeric_limits<VertexID>::max()) continue;

        // Continue if already inserted
        VertexID update_id = vlabel + num_global_vertices * wlabel;
        if (inserted_edges.find(update_id) != end(inserted_edges)) continue;

        // If vlabel is on same PE just insert the edge
        if (g_.IsLocalFromGlobal(vlabel)) {
          edges_to_add[wroot].emplace_back(vlabel, wlabel);
          edge_counter_++;
          inserted_edges.insert(update_id);
          if (wroot == rank_) {
            edges_to_add[rank_].emplace_back(wlabel, vlabel);
            edge_counter_++;
            inserted_edges.insert(wlabel + num_global_vertices * vlabel);
          }
        } else {
          std::cout << "R" << rank_ << " incorrect message in direct contraction" << std::endl;
          exit(1);
        }
      }
    }

    // Insert edges and keep corresponding vertices
    contraction_timer_.Restart();
    g_.ForallLocalVertices([&](VertexID v) { g_.RemoveAllEdges(v); });
    contraction_level_++;

    UpdateActiveVertices();
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
    InsertEdges(edges_to_add);

    std::cout << "[STATUS] |---- R" << rank_ << " Updating edges took " 
              << "[TIME] " << contraction_timer_.Elapsed() 
              << " " << edge_counter_ << " edges" << std::endl;

    // max_degree_computed_ = false;
    UpdateGraphVertices();

    unsigned int unresolved_requests = 0;
    for (unsigned int i = 0; i < num_requests; ++i) {
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

  void LocalContraction() {
    // Statistics
    Timer propagation_timer;
    propagation_timer.Restart();

    VertexID num_global_vertices = g_.GatherNumberOfGlobalVertices();
    VertexID num_vertices = g_.GetNumberOfVertices();

    // Update with new vertices added during last contraction
    g_.ForallVertices([&](VertexID v) {
      inactive_level_[v] = -1;
    });

    // Determine edges to communicate
    // Gather labels to communicate
    google::dense_hash_set<VertexID> send_ids;
    send_ids.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<VertexID>> send_buffers_a(size_);
    std::vector<std::vector<VertexID>> send_buffers_b(size_);
    std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
    std::vector<std::vector<VertexID>> receive_buffers(size_);

    // TODO: Fix iteration
    for (int i = 0; i < size_; ++i) {
      send_buffers_a[i].clear();
      send_buffers_b[i].clear();
      receive_buffers[i].clear();
    }

    google::dense_hash_set<VertexID> inserted_edges; 
    inserted_edges.set_empty_key(-1);

    // TODO: Fix size
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);

    FindLocalConflictingEdges(num_global_vertices, 
                              inserted_edges, 
                              edges_to_add, 
                              send_ids, 
                              current_send_buffers);

    // TODO: Fix size
    std::vector<bool> is_adj(size_);
    PEID num_adj = FindAdjacentPEs(is_adj);

    // Propagate edge buffers until all vertices are converged
    std::vector<MPI_Request> requests;
    requests.clear();
    int converged_globally = 0;
    int local_iterations = 0;
    while (converged_globally == 0) {
      comm_timer_.Restart();
      SendMessages(is_adj, current_send_buffers, requests);
      ReceiveMessages(num_adj, requests, current_send_buffers, receive_buffers);
      SwapBuffers(current_send_buffers, send_buffers_a, send_buffers_b);
      comm_time_ += comm_timer_.Elapsed();

      int converged_locally = ProcessLocalMessages(num_global_vertices, inserted_edges, edges_to_add, 
                                                   send_ids, receive_buffers, current_send_buffers);


      // Check if all PEs are done
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
      local_iterations++;
    }
    if (rank_ == ROOT) 
      std::cout << "[STATUS] |---- Propagation done " 
                << "[TIME] " << propagation_timer.Elapsed() << std::endl;

    // Insert edges and keep corresponding vertices
    g_.ForallLocalVertices([&](VertexID v) { g_.RemoveAllEdges(v); });
    contraction_level_++;

    UpdateActiveVertices();
    g_.ResetNumberOfCutEdges();
    g_.ResetAdjacentPEs();
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
            edge_counter_++;
            inserted_edges.insert(update_id);
            sent_edges.insert(update_id);
            if (wroot == rank_) {
              local_edges[rank_].emplace_back(wlabel, vlabel);
              edge_counter_++;
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
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
  }

  void FindDirectConflictingEdges(VertexID num_global_vertices,
                                  google::dense_hash_set<VertexID> &inserted_edges, 
                                  std::vector<std::vector<std::pair<VertexID, VertexID>>> &local_edges,
                                  google::dense_hash_set<VertexID> &sent_edges,
                                  std::vector<std::vector<VertexID>> &send_buffers) {
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
            edge_counter_++;
            inserted_edges.insert(update_id);
            sent_edges.insert(update_id);
            if (wroot == rank_) {
              local_edges[rank_].emplace_back(wlabel, vlabel);
              edge_counter_++;
              inserted_edges.insert(wlabel + num_global_vertices * vlabel);
              sent_edges.insert(wlabel + num_global_vertices * vlabel);
            }
          } else if (sent_edges.find(update_id) == end(sent_edges)) {
            // Direct propagation to root
            PEID pe = g_.GetVertexRoot(v);
            // Send edge
            sent_edges.insert(update_id);
            send_buffers[pe].emplace_back(vlabel);
            send_buffers[pe].emplace_back(wlabel);
            send_buffers[pe].emplace_back(wroot);
            // std::cout << "R" << rank_ << " send (" << vlabel << "," << wlabel << ") (" << wroot << ") to " << pe << std::endl;
          }
        }
        removed_edges_.emplace(v, g_.GetGlobalID(w));
      });
    });
    // Sentinel
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
  }

  void FindLocalConflictingEdges(VertexID num_global_vertices,
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
            edge_counter_++;
            inserted_edges.insert(update_id);
            sent_edges.insert(update_id);
            if (wroot == rank_) {
              local_edges[rank_].emplace_back(wlabel, vlabel);
              edge_counter_++;
              inserted_edges.insert(wlabel + num_global_vertices * vlabel);
              sent_edges.insert(wlabel + num_global_vertices * vlabel);
            }
          } else if (sent_edges.find(update_id) == sent_edges.end()) {
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
    removed_edges_.emplace(std::numeric_limits<VertexID>::max() - 1, std::numeric_limits<VertexID>::max() - 1);
  }

  PEID FindAdjacentPEs(std::vector<bool> &is_adj) {
    // Get adjacency (otherwise we get deadlocks with added edges)
    PEID num_adj = 0;
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; pe++) {
      is_adj[pe] = g_.IsAdjacentPE(pe);
      if (is_adj[pe]) num_adj++;
    }
    return num_adj;
  }

  void ReceiveMessages(PEID adjacent_pes,
                       std::vector<MPI_Request> &requests,
                       std::vector<std::vector<VertexID>> *send_buffers,
                       std::vector<std::vector<VertexID>> &receive_buffers) {
    // Receive edges
    PEID messages_recv = 0;
    int message_length = 0;
    // TODO: Fix iteration
    for (int i = 0; i < size_; ++i) receive_buffers[i].clear();
    receive_buffers[rank_] = (*send_buffers)[rank_]; // copy (TODO: avoid)
    while (messages_recv < adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, 95 * size_ + rank_, MPI_COMM_WORLD, &st);
      MPI_Get_count(&st, MPI_VERTEX, &message_length);
      messages_recv++;

      receive_buffers[st.MPI_SOURCE].resize(message_length);
      MPI_Status rst{};
      MPI_Recv(&receive_buffers[st.MPI_SOURCE][0], 
               message_length, MPI_VERTEX, 
               st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, &rst);
    }

    for (unsigned int i = 0; i < requests.size(); ++i) {
      if (requests[i] != MPI_REQUEST_NULL) {
        MPI_Request_free(&requests[i]);
      }
    }
    requests.clear();
  }

  void SendMessages(std::vector<bool> &is_adj,
                    std::vector<std::vector<VertexID>> *send_buffers,
                    std::vector<MPI_Request> &requests) {
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (is_adj[pe]) {
        if ((*send_buffers)[pe].empty()) {
          (*send_buffers)[pe].emplace_back(std::numeric_limits<VertexID>::max());
          (*send_buffers)[pe].emplace_back(0);
          (*send_buffers)[pe].emplace_back(0);
          (*send_buffers)[pe].emplace_back(0);
        }
        requests.emplace_back(MPI_Request());
        MPI_Isend((*send_buffers)[pe].data(), 
                  static_cast<int>((*send_buffers)[pe].size()), 
                  MPI_VERTEX, pe, 95 * size_ + pe, MPI_COMM_WORLD, &requests.back());
        if (pe == rank_) {
          std::cout << "R" << rank_ << " ERROR selfmessage" << std::endl;
          exit(1);
        }
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
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffers[pe].size() < 4) continue;
      for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
        VertexID vlabel = receive_buffers[pe][i];
        VertexID wlabel = receive_buffers[pe][i + 1];
        VertexID wroot = receive_buffers[pe][i + 2];
        VertexID link = receive_buffers[pe][i + 3];

        // Check for dummy message
        if (vlabel == std::numeric_limits<VertexID>::max()) continue;

        // Continue if already inserted
        VertexID update_id = vlabel + num_global_vertices * wlabel;
        if (inserted_edges.find(update_id) != end(inserted_edges)) continue;
        if (propagated_edges.find(update_id) != end(propagated_edges)) continue;

        // If vlabel is on same PE just insert the edge
        if (g_.IsLocalFromGlobal(vlabel)) {
          new_edges[wroot].emplace_back(vlabel, wlabel);
          edge_counter_++;
          inserted_edges.insert(update_id);
          propagated_edges.insert(update_id);
          if (wroot == rank_) {
            new_edges[rank_].emplace_back(wlabel, vlabel);
            edge_counter_++;
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
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffers[pe].size() < 4) continue;
      for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
        VertexID vlabel = receive_buffers[pe][i];
        VertexID wlabel = receive_buffers[pe][i + 1];
        VertexID wroot = receive_buffers[pe][i + 2];
        VertexID link = receive_buffers[pe][i + 3];

        // Check for dummy message
        if (vlabel == std::numeric_limits<VertexID>::max()) continue;

        // Continue if already inserted
        VertexID update_id = vlabel + num_global_vertices * wlabel;
        if (inserted_edges.find(update_id) != end(inserted_edges)) continue;
        if (propagated_edges.find(update_id) != end(propagated_edges)) continue;

        // Get link information
        // TODO: We use this parent as link, this is wrong if the current (link) vertex points to a different partition
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
            VertexID parent = g_.GetParent(g_.GetLocalID(link));
            PEID pe = g_.GetPE(g_.GetLocalID(parent));
            propagated_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, parent, send_buffers);
            propagate = 1;
          } else {
            // Parent has to be connected to vlabel (N(N(v))
            VertexID local_vlabel = g_.GetLocalID(vlabel);
            pe = g_.GetPE(local_vlabel);
            // Send edge
            propagated_edges.insert(update_id);
            PlaceInBuffer(pe, vlabel, wlabel, wroot, vlabel, send_buffers);
            propagate = 1;
          }
        }
      }
    }
    return !propagate;
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
    for (auto &kv : inactive_level_) {
      if (kv.second == -1) inactive_level_[kv.first] = contraction_level_ - 1;
    }
  }

  void InsertEdges(std::vector<std::vector<std::pair<VertexID, VertexID>>> &new_edges) {
    // TODO: Fix iteration
    for (PEID pe = 0; pe < size_; pe++) {
      for (auto &e : new_edges[pe]) {
        VertexID vlabel = e.first;
        VertexID wlabel = e.second;
        VertexID vlocal = g_.GetLocalID(vlabel);
        // TODO: Check if this is needed
        if (!g_.IsLocalFromGlobal(wlabel) && !g_.IsGhostFromGlobal(wlabel)) {
          g_.AddGhostVertex(wlabel);
        }
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
    for (auto &kv : inactive_level_) {
      g_.SetActive(kv.first, kv.second == -1);
    }
  }

  void SwapBuffers(std::vector<std::vector<VertexID>>* buffers,
                   std::vector<std::vector<VertexID>> &a_buffers,
                   std::vector<std::vector<VertexID>> &b_buffers) {
    // Switch buffers
    // TODO: Fix iteration
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
      contraction_timer_.Restart();
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
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Updating edges took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

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
      if (rank_ == ROOT) {
        std::cout << "[status] |-- Updating payloads took " 
                  << "[time] " << contraction_timer_.Elapsed() << std::endl;
      }

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
        if (rank_ == ROOT) {
          std::cout << "[status] |--- Message exchange took " 
                    << "[time] " << contraction_timer_.Elapsed() << std::endl;
        }

        contraction_timer_.Restart();
        // Check if all PEs are done
        comm_timer_.Restart();
        MPI_Allreduce(&converged_locally,
                      &converged_globally,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        comm_time_ += comm_timer_.Elapsed();
        if (rank_ == ROOT) {
          std::cout << "[status] |--- Convergence test took " 
                    << "[time] " << contraction_timer_.Elapsed() << std::endl;
        }
      }
      // Vertices at current level are roots at previous one
      g_.ForallLocalVertices([&](VertexID v) {
        g_.SetParent(v, g_.GetGlobalID(v));
      });
    }
  }

  void EnableActiveVertices() {
    for (auto &kv : inactive_level_) {
      if (kv.second == contraction_level_) inactive_level_[kv.first] = -1;
    }
  }

  void AddRemovedEdges() {
    while (!removed_edges_.empty()) {
      auto e = removed_edges_.top();
      removed_edges_.pop();
      if (std::get<0>(e) == std::numeric_limits<VertexID>::max() - 1) break;
      g_.AddEdge(std::get<0>(e), std::get<1>(e), g_.GetPE(g_.GetLocalID(std::get<1>(e))));
    }
  }

  float GetCommTime() {
    return comm_time_;
  }

 private:
  // Original graph instance
  DynamicGraphCommunicator &g_;

  // Network information
  PEID rank_, size_;

  // Variables
  VertexID contraction_level_;
  std::stack<std::pair<VertexID, VertexID>> removed_edges_;
  google::dense_hash_map<VertexID, short> inactive_level_;

  // Statistics
  float comm_time_;
  Timer contraction_timer_;
  Timer comm_timer_;
  VertexID edge_counter_;
};

#endif
