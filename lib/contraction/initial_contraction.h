/******************************************************************************
 * initial_contraction.h
 *
 * Initial contraction of distributed graph
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

#ifndef _INITIAL_CONTRACTION_H_
#define _INITIAL_CONTRACTION_H_

#include <iostream>
#include <google/sparse_hash_set>

#include "config.h"
#include "definitions.h"
#include "graph_access.h"
#include "base_graph_access.h"
#include "edge_hash.h"

template <typename GraphInputType>
class InitialContraction {
 public:
  InitialContraction(GraphInputType &g, std::vector<VertexID> &vertex_labels, const PEID rank, const PEID size)
      : g_(g), vertex_labels_(vertex_labels), rank_(rank), size_(size),
        num_smaller_components_(0),
        num_local_components_(0),
        num_global_components_(0),
        node_buffers_(size) {
    local_components_.set_empty_key(-1);
  }
  virtual ~InitialContraction() = default;

  GraphAccess BuildComponentAdjacencyGraph() {
    ComputeComponentPrefixSum();
    ComputeLocalContractionMapping();
    ExchangeGhostContractionMapping();
    GenerateLocalContractionEdges();
    return BuildContractionGraph();
  }

  BaseGraphAccess ReduceBaseGraph() {
    ComputeComponentPrefixSum();
    ComputeLocalContractionMapping();
    ExchangeGhostContractionMapping();
    GenerateLocalContractionEdges();
    return BuildReducedBaseGraph();
  }

 private:
  // Original graph instance
  GraphInputType &g_;
  std::vector<VertexID> &vertex_labels_;

  // Network information
  PEID rank_, size_;

  // Component information
  VertexID num_smaller_components_, 
           num_local_components_,
           num_global_components_;

  google::dense_hash_set<VertexID> local_components_; 

  // Local edges
  EdgeHash local_edges_{};

  // Send buffers
  std::vector<std::vector<VertexID>> node_buffers_;

  std::vector<bool> received_message_;

  void ComputeComponentPrefixSum() {
    // Gather local components O(max(#component))
    num_local_components_ = FindLocalComponents();

    // Build prefix sum over local components O(log P)
    VertexID component_prefix_sum;
    MPI_Scan(&num_local_components_,
             &component_prefix_sum,
             1,
             MPI_VERTEX,
             MPI_SUM,
             MPI_COMM_WORLD);

    // Broadcast number of global components
    num_global_components_ = component_prefix_sum;
    MPI_Bcast(&num_global_components_,
              1,
              MPI_VERTEX,
              size_ - 1,
              MPI_COMM_WORLD);

    num_smaller_components_ = component_prefix_sum - num_local_components_;
  }

  VertexID FindLocalComponents() {
    // Add local components to hash set
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID v_label = vertex_labels_[v];
      if (local_components_.find(v_label) == end(local_components_)) {
        local_components_.insert(v_label);
      }
    });
    return local_components_.size();
  }

  void ComputeLocalContractionMapping() {
    // Map local components to contraction vertices O(n/P)
    google::dense_hash_map<VertexID, VertexID> label_map; 
    label_map.set_empty_key(-1);
    VertexID current_component = num_smaller_components_;
    for (const VertexID c : local_components_) {
      label_map[c] = current_component++;
    }

    // Setup contraction vertices for local vertices O(n/P)
    g_.AllocateContractionVertices();
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID component = label_map[vertex_labels_[v]];
      g_.SetContractionVertex(v, component);
      // std::cout << "R" << rank_ << " v " << g_.GetGlobalID(v) << " set cid " << component << std::endl;
    });
  }

  void ExchangeGhostContractionMapping() {
    IdentifyLargestInterfaceComponents();
    AddComponentMessages();

    // Send ghost vertex updates O(cut size) (communication)
    std::vector<MPI_Request*> requests = SendBuffers();

    google::dense_hash_map<PEID, VertexID> largest_component; 
    google::dense_hash_map<VertexID, VertexID> vertex_message; 
    largest_component.set_empty_key(-1); 
    vertex_message.set_empty_key(-1);

    received_message_.resize(g_.GetNumberOfVertices(), false);
    ReceiveBuffers(largest_component, vertex_message);

    ApplyUpdatesToGhostVertices(largest_component, vertex_message);
    WaitForRequests(requests);
  }

  void IdentifyLargestInterfaceComponents() {
    // Compute sizes of components for interface vertices
    google::dense_hash_map<VertexID, VertexID> interface_component_size;
    interface_component_size.set_empty_key(-1);

    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        if (interface_component_size.find(cv) == interface_component_size.end()) 
          interface_component_size[cv] = 0;
        interface_component_size[cv]++;
      }
    });

    std::vector<VertexID> largest_component_size(size_, 0);
    std::vector<VertexID> largest_component_id(size_, 0);

    // Identify largest component for each adjacent PE
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            if (interface_component_size[cv] > largest_component_size[target_pe]) {
              largest_component_size[target_pe] = interface_component_size[cv];
              largest_component_id[target_pe] = cv;
            } 
          }
        });
      }
    });

    for (PEID i = 0; i < size_; ++i) {
      if (largest_component_size[i] > 0) {
        node_buffers_[i].push_back(std::numeric_limits<VertexID>::max()); 
        node_buffers_[i].push_back(largest_component_id[i]); 
        // std::cout << "R" << rank_ << " set largest " << largest_component_id[i] << " PE " << i << std::endl;
      }
    }
  }

  void AddComponentMessages() {
    // Helper functions
    auto pair = [&](VertexID x, VertexID y) {
      return static_cast<VertexID>((0.5 * ((x + y) * (x + y + 1))) + y);
    };

    // Gather components with the same neighbor
    google::sparse_hash_set<VertexID> unique_neighbors;
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            VertexID contraction_vertex = g_.GetContractionVertex(v);
            VertexID largest_component = node_buffers_[target_pe][1];
            // Only send message if not part of largest component
            if (contraction_vertex != largest_component) {
              // Avoid duplicates by hashing the message
              VertexID comp_pair = pair(w, g_.GetContractionVertex(v));
              if (unique_neighbors.find(comp_pair) == end(unique_neighbors)) {
                unique_neighbors.insert(comp_pair);
                node_buffers_[target_pe].push_back(g_.GetGlobalID(v));
                node_buffers_[target_pe].push_back(contraction_vertex);
              }
            }
          }
        });
      }
    });

  }

  std::vector<MPI_Request*> SendBuffers() {
    std::vector<MPI_Request*> requests;
    requests.clear();

    for (PEID i = 0; i < size_; ++i) {
      if (g_.IsAdjacentPE(i)) {
        if (node_buffers_[i].empty()) 
          node_buffers_[i].push_back(0);
        auto *req = new MPI_Request();
        MPI_Isend(&node_buffers_[i][0],
                  static_cast<int>(node_buffers_[i].size()),
                  MPI_UNSIGNED_LONG,
                  i,
                  i + 6 * size_,
                  MPI_COMM_WORLD,
                  req);
        requests.emplace_back(req);
      };
    }

    return requests;
  }

  void ReceiveBuffers(google::dense_hash_map<PEID, VertexID> & largest_component, 
                      google::dense_hash_map<VertexID, VertexID> & vertex_message) {
    // Optimization for receiving largest component
    // Receive updates O(cut size)
    PEID num_adjacent_pes = g_.GetNumberOfAdjacentPEs();
    PEID recv_messages = 0;
    while (recv_messages < num_adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);

      int message_length;
      MPI_Get_count(&st, MPI_VERTEX, &message_length);
      std::vector<VertexID> message(static_cast<unsigned long>(message_length));

      MPI_Status rst{};
      MPI_Recv(&message[0],
               message_length,
               MPI_VERTEX,
               st.MPI_SOURCE,
               rank_ + 6 * size_,
               MPI_COMM_WORLD,
               &rst);
      recv_messages++;

      for (int i = 0; i < message_length - 1; i += 2) {
        VertexID global_id = message[i];
        VertexID contraction_id = message[i + 1];
        if (global_id == std::numeric_limits<VertexID>::max()) {
          largest_component[st.MPI_SOURCE] = contraction_id;
        } else {
          vertex_message[g_.GetLocalID(global_id)] = contraction_id;
          received_message_[g_.GetLocalID(global_id)] = true;
        }
      }
    }
  }

  void ApplyUpdatesToGhostVertices(google::dense_hash_map<PEID, VertexID> & largest_component, 
                                   google::dense_hash_map<VertexID, VertexID> & vertex_message) {
    g_.ForallGhostVertices([&](VertexID v) {
      PEID pe = g_.GetPE(v);
      VertexID cid = vertex_message[v];
      if (received_message_[v]) {
        g_.SetContractionVertex(v, cid);
      } else {
        g_.SetContractionVertex(v, largest_component[pe]);
      }
    });
  }

  void WaitForRequests(std::vector<MPI_Request*>& requests) {
    for (unsigned int i = 0; i < requests.size(); ++i) {
      MPI_Status st;
      MPI_Wait(requests[i], &st);
      delete requests[i];
    }
  }

  void GenerateLocalContractionEdges() {
    // Gather local edges (there shouldn't be any) O(m/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g_.GetContractionVertex(v);
      g_.ForallNeighbors(v, [&](const VertexID w) {
        VertexID cw = g_.GetContractionVertex(w);
        if (cv != cw) {
          auto h_edge = HashedEdge{num_global_components_, cv, cw, g_.GetPE(w)};
          if (local_edges_.find(h_edge) == end(local_edges_)) {
            local_edges_.insert(h_edge);
          }
        }
      });
    });
  }

  GraphAccess BuildContractionGraph() {
    VertexID from = num_smaller_components_;
    VertexID to = num_smaller_components_ + num_local_components_ - 1;

    EdgeID edge_counter = 0;
    std::vector<std::vector<std::pair<VertexID, VertexID>>>
        local_edge_lists(num_local_components_);
    for (const auto &e : local_edges_) {
      // Edge runs between local vertices
      if (from <= e.target && e.target < to) {
        local_edge_lists[e.source - from].emplace_back(e.target, rank_);
        local_edge_lists[e.target - from].emplace_back(e.source, rank_);
        edge_counter += 2;
      }
      // Edge runs between interface and ghost vertices
      else {
        local_edge_lists[e.source - from].emplace_back(e.target, e.rank);
        edge_counter++;
      }
    }


    GraphAccess cg(rank_, size_);
    cg.StartConstruct(num_local_components_,
                      num_smaller_components_);

    for (VertexID i = 0; i < num_local_components_; ++i) {
      VertexID v = cg.AddVertex();
      cg.SetVertexPayload(v, {cg.GetVertexDeviate(v), 
                              from + v, 
#ifdef TIEBREAK_DEGREE
                              0,
#endif
                              rank_});

      for (auto &j : local_edge_lists[i]) {
        VertexID target = j.first;
        cg.AddEdge(v, target, static_cast<PEID>(j.second));
      }
    }
    cg.FinishConstruct();
    return cg;
  }

  BaseGraphAccess BuildReducedBaseGraph() {
    VertexID from = num_smaller_components_;
    VertexID to = num_smaller_components_ + num_local_components_ - 1;

    VertexID number_of_ghost_vertices = 0;
    google::dense_hash_map<VertexID, VertexID> num_edges_for_ghost; 
    num_edges_for_ghost.set_empty_key(-1);

    std::vector<std::vector<std::pair<VertexID, VertexID>>>
        local_edge_lists(num_local_components_);
    for (const auto &e : local_edges_) {
      // Add edge from source to target
      local_edge_lists[e.source - from].emplace_back(e.target, e.rank);

      if (from <= e.target && e.target <= to) {
        // Target is local
        local_edge_lists[e.target - from].emplace_back(e.source, rank_);
      } else {
        // Target is ghost
        if (num_edges_for_ghost.find(e.target) == end(num_edges_for_ghost)) {
          num_edges_for_ghost[e.target] = 0;
          number_of_ghost_vertices++;
        }
        num_edges_for_ghost[e.target]++;
      }
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size_);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, MPI_COMM_WORLD);

    BaseGraphAccess cg(rank_, size_);
    cg.StartConstruct(num_local_components_,
                      number_of_ghost_vertices,
                      num_smaller_components_);


    cg.SetOffsetArray(std::move(vertex_dist));

    // Reserve memory for outgoing edges from ghost vertices
    for (auto &kv : num_edges_for_ghost) {
      VertexID local_id = cg.AddGhostVertex(kv.first);
      cg.ReserveEdgesForVertex(local_id, kv.second);
    }

    // Reserve memory for outgoing edges from local vertices
    for (VertexID v = 0; v < to - from + 1; ++v) {
      cg.ReserveEdgesForVertex(v, local_edge_lists[v].size());
    }

    // Add edges
    for (VertexID v = 0; v < to - from + 1; ++v) {
      for (auto &kv : local_edge_lists[v]) {
        cg.AddEdge(v, kv.first, static_cast<PEID>(kv.second));
      }
    }

    cg.FinishConstruct();
    return cg;
  }
};

#endif