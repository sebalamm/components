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

#ifndef _CAG_BUILDER_H_
#define _CAG_BUILDER_H_

#include <iostream>
#include <google/sparse_hash_set>
#include <google/dense_hash_set>

#include "config.h"
#include "definitions.h"
#include "comm_utils.h"
#include "dynamic_graph_comm.h"
#include "static_graph.h"
#include "edge_hash.h"


template <typename GraphType, typename LabelType = std::vector<VertexID>>
class CAGBuilder {
 public:
  CAGBuilder(GraphType &g, LabelType &vertex_labels, const PEID rank, const PEID size)
      : g_(g), vertex_labels_(vertex_labels), rank_(rank), size_(size),
        num_smaller_components_(0),
        num_local_components_(0),
        num_global_components_(0),
        comm_time_(0.0) {
    local_components_.set_empty_key(EmptyKey);
    local_components_.set_deleted_key(DeleteKey);
    send_buffers_.set_empty_key(EmptyKey);
    send_buffers_.set_deleted_key(DeleteKey);
    receive_buffers_.set_empty_key(EmptyKey);
    receive_buffers_.set_deleted_key(DeleteKey);
    offset_ = g_.GatherNumberOfGlobalVertices();
    smallest_vertex_id_ = offset_;
  }
  virtual ~CAGBuilder() = default;

  template <typename GraphOutputType>
  GraphOutputType BuildComponentAdjacencyGraph() {
    PerformContraction();
    return BuildContractionGraph<GraphOutputType>();
  }

  template <typename GraphOutputType>
  GraphOutputType BuildLocalComponentGraph() {
    PerformLocalContraction();
    return BuildLocalContractionGraph<GraphOutputType>();
  }

  float GetCommTime() {
    return comm_time_;
  }

 private:
  // Original graph instance
  GraphType &g_;
  LabelType &vertex_labels_;

  // Network information
  PEID rank_, size_;

  // Offset for pairing
  VertexID offset_;

  // Smallest vertex ID
  VertexID smallest_vertex_id_;

  // Component information
  VertexID num_smaller_components_, 
           num_local_components_,
           num_global_components_;

  google::dense_hash_set<VertexID> local_components_; 

  // Local edges
  EdgeHash local_edges_{};
  std::vector<std::tuple<VertexID, VertexID, PEID>> edges_;

  // Send buffers
  google::dense_hash_map<PEID, VertexBuffer> send_buffers_;
  google::dense_hash_map<PEID, VertexBuffer> receive_buffers_;

  // Statistics
  float comm_time_;
  Timer contraction_timer_;
  Timer comm_timer_;

  void PerformContraction() {
    contraction_timer_.Restart();
    comm_timer_.Restart();
    ComputeComponentPrefixSum();
    comm_time_ += comm_timer_.Elapsed();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing component prefix sum took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart();
    ComputeLocalContractionMapping();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing local contraction mapping took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart();
    ExchangeGhostContractionMapping();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- R" << rank_ << " Exchanging ghost contraction mapping took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    
    contraction_timer_.Restart();
    GenerateLocalContractionEdges();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Generating contraction edges took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
  }

  void PerformLocalContraction() {
    // Determine remaining vertices
    g_.AllocateContractionVertices();
    g_.ForallLocalVertices([&](const VertexID v) {
      // Set contraction vertex
      VertexID cv;
      if (!g_.IsInterface(v)) {
        cv = vertex_labels_[v];
      } else {
        cv = g_.GetGlobalID(v);
      }
      g_.SetContractionVertex(v, cv);

      // Store component 
      if (local_components_.find(cv) == end(local_components_)) {
        local_components_.insert(cv);
      }
      // Store smallest vertex label
      if (cv < smallest_vertex_id_) {
        smallest_vertex_id_ = cv;
      }
    });
    num_local_components_ = local_components_.size();

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

    // Determine remaining edges O(m/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g_.GetContractionVertex(v);
      g_.ForallNeighbors(v, [&](const VertexID w) {
        VertexID cw;
        if (g_.IsGhost(w)) {
          cw = g_.GetGlobalID(w);
        } else {
          cw = g_.GetContractionVertex(w);
        }
        if (cv != cw) {
          PEID pe = g_.GetPE(w);
          auto h_edge = HashedEdge{offset_, cv, cw, pe};
          if (local_edges_.find(h_edge) == end(local_edges_)) {
            local_edges_.insert(h_edge);
            edges_.emplace_back(cv, cw, pe);
            // TODO: Does this tie breaking work?
            // Seems to work for now
            if (!g_.IsLocal(w)) {
              edges_.emplace_back(cw, cv, rank_);
            }
          }
        }
      });
    });
  }

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
    label_map.set_empty_key(EmptyKey);
    label_map.set_deleted_key(DeleteKey);
    VertexID current_component = num_smaller_components_;
    for (const VertexID c : local_components_) {
      label_map[c] = current_component++;
    }

    // Setup contraction vertices for local vertices O(n/P)
    g_.AllocateContractionVertices();
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID component = label_map[vertex_labels_[v]];
      g_.SetContractionVertex(v, component);
    });
  }

  void ExchangeGhostContractionMapping() {
    Timer exchange_timer;
    exchange_timer.Restart();
    IdentifyLargestInterfaceComponents();
    AddComponentMessages();

    std::cout << "[STATUS] |--- R" << rank_ << " Filling buffers took " 
              << "[TIME] " << exchange_timer.Elapsed() << std::endl;

    exchange_timer.Restart();
    google::dense_hash_map<PEID, VertexID> largest_component; 
    google::dense_hash_map<VertexID, VertexID> vertex_message; 
    largest_component.set_empty_key(EmptyKey); 
    largest_component.set_deleted_key(DeleteKey); 
    vertex_message.set_empty_key(EmptyKey);
    vertex_message.set_deleted_key(DeleteKey);

    // Send ghost vertex updates O(cut size) (communication)
    exchange_timer.Restart();
    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers_, receive_buffers_, rank_, size_, CAGTag);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers_);

    std::cout << "[STATUS] |--- Sparse exchange took " 
              << "[TIME] " << exchange_timer.Elapsed() << std::endl;

    exchange_timer.Restart();
    HandleMessages(largest_component, vertex_message);
    CommunicationUtility::ClearBuffers(receive_buffers_);
    std::cout << "[STATUS] |--- Message handling took " 
              << "[TIME] " << exchange_timer.Elapsed() << std::endl;

    exchange_timer.Restart();
    ApplyUpdatesToGhostVertices(largest_component, vertex_message);
    std::cout << "[STATUS] |--- R" << rank_ << " Local updates took " 
              << "[TIME] " << exchange_timer.Elapsed() << std::endl;
  }

  void IdentifyLargestInterfaceComponents() {
    // Compute sizes of components for interface vertices
    google::dense_hash_map<VertexID, VertexID> interface_component_size;
    interface_component_size.set_empty_key(EmptyKey);
    interface_component_size.set_deleted_key(DeleteKey);

    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        if (interface_component_size.find(cv) == interface_component_size.end()) 
          interface_component_size[cv] = 0;
        interface_component_size[cv]++;
      }
    });

    google::dense_hash_map<PEID, std::pair<VertexID, VertexID>> largest_component;
    largest_component.set_empty_key(EmptyKey);
    largest_component.set_deleted_key(DeleteKey);

    // Identify largest component for each adjacent PE
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            if (largest_component.find(target_pe) == largest_component.end()) {
              largest_component[target_pe] = std::make_pair(interface_component_size[cv], cv);
            } else if (interface_component_size[cv] > largest_component[target_pe].first) {
              largest_component[target_pe] = std::make_pair(interface_component_size[cv], cv);
            } 
          }
        });
      }
    });

    for (const auto &kv : largest_component) {
      send_buffers_[kv.first].emplace_back(std::numeric_limits<VertexID>::max() - 1);
      send_buffers_[kv.first].emplace_back(kv.second.second);
    }
  }

  void AddComponentMessages() {
    // Helper functions
    auto pair = [&](VertexID x, VertexID y) {
      return x * offset_ + y;
    };

    // Gather components with the same neighbor
    // TODO: Can we use something else than a hashmap of hashmaps?
    google::dense_hash_map<PEID, google::sparse_hash_set<VertexID>> unique_neighbors;
    unique_neighbors.set_empty_key(EmptyKey);
    unique_neighbors.set_deleted_key(DeleteKey);
    google::dense_hash_set<VertexID> targets;
    targets.set_empty_key(EmptyKey);
    targets.set_deleted_key(DeleteKey);
    VertexID buffer_size = 0;
    VertexID num_checks = 0;
    VertexID num_edges = 0;
    VertexID messages_for_vertex = 0;
    VertexID max_messages = 0;
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        google::dense_hash_set<PEID> receiving_pes;
        receiving_pes.set_empty_key(EmptyKey);
        receiving_pes.set_deleted_key(DeleteKey);
        messages_for_vertex = 0;
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            VertexID contraction_vertex = g_.GetContractionVertex(v);
            VertexID largest_component = send_buffers_[target_pe][1];
            num_edges++;
            // Only send message if not part of largest component
            if (contraction_vertex != largest_component) {
              // Avoid duplicates by hashing the message
              VertexID comp_pair = pair(w, contraction_vertex);
              num_checks++;
              if (targets.find(w) == targets.end()) {
                targets.insert(w);
              }
              if (unique_neighbors[target_pe].find(comp_pair) == end(unique_neighbors[target_pe]) 
                && receiving_pes.find(target_pe) == end(receiving_pes)) {
                unique_neighbors[target_pe].insert(comp_pair);
                receiving_pes.insert(target_pe);
                send_buffers_[target_pe].push_back(g_.GetGlobalID(v));
                send_buffers_[target_pe].push_back(contraction_vertex);
                messages_for_vertex++;
                buffer_size++;
              }
            }
          }
        });
        if (messages_for_vertex > max_messages) max_messages = messages_for_vertex;
      }
    });
  }

  void HandleMessages(google::dense_hash_map<PEID, VertexID> &largest_component, 
                      google::dense_hash_map<VertexID, VertexID> &vertex_message) {
    for (auto &kv : receive_buffers_) {
      auto &buffer = kv.second;
      for (VertexID i = 0; i < buffer.size(); i += 2) {
        VertexID global_id = buffer[i];
        VertexID contraction_id = buffer[i + 1];

        if (global_id == std::numeric_limits<VertexID>::max() - 1) {
          largest_component[kv.first] = contraction_id;
        } else {
          vertex_message[g_.GetLocalID(global_id)] = contraction_id;
        }
      }
    }
  }

  void ApplyUpdatesToGhostVertices(google::dense_hash_map<PEID, VertexID> &largest_component, 
                                   google::dense_hash_map<VertexID, VertexID> &vertex_message) {
    g_.ForallGhostVertices([&](const VertexID v) {
      PEID pe = g_.GetPE(v);
      if (vertex_message.find(v) != vertex_message.end()) {
        g_.SetContractionVertex(v, vertex_message[v]);
      } else {
        g_.SetContractionVertex(v, largest_component[pe]);
      }
    });

    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        google::dense_hash_map<PEID, VertexID> component_id;
        component_id.set_empty_key(EmptyKey);
        component_id.set_deleted_key(DeleteKey);
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID pe = g_.GetPE(w);
            VertexID cid = g_.GetContractionVertex(w);
            if (component_id.find(pe) == component_id.end()) {
              component_id[pe] = cid;
            } else {
              // There are two adjacent vertices on the same PE with different component IDs?
              if (cid != component_id[pe] && cid != largest_component[pe]) {
                component_id[pe] = cid;
              }
            }
          }
        });
        // Apply updates
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID pe = g_.GetPE(w);
            g_.SetContractionVertex(w, component_id[pe]);
          }
        });
      }
    });
  }

  void GenerateLocalContractionEdges() {
    // Gather local edges (there shouldn't be any) O(m/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g_.GetContractionVertex(v);
      g_.ForallNeighbors(v, [&](const VertexID w) {
        VertexID cw = g_.GetContractionVertex(w);
        if (cv != cw) {
          PEID pe = g_.GetPE(w);
          auto h_edge = HashedEdge{num_global_components_, cv, cw, pe};
          if (local_edges_.find(h_edge) == end(local_edges_)) {
            local_edges_.insert(h_edge);
            edges_.emplace_back(cv, cw, pe);
            edges_.emplace_back(cw, cv, rank_);
          }
        }
      });
    });
  }

  template <typename GraphOutputType>
  GraphOutputType BuildContractionGraph() {
    VertexID from = num_smaller_components_;

    // Identify ghost vertices
    google::dense_hash_map<VertexID, PEID> ghost_pe; 
    ghost_pe.set_empty_key(EmptyKey);
    ghost_pe.set_deleted_key(DeleteKey);
    for (auto &e : edges_) {
      VertexID target = std::get<1>(e);
      PEID pe = std::get<2>(e);
      if (std::get<2>(e) != rank_) {
        if (ghost_pe.find(target) == ghost_pe.end()) {
            ghost_pe[target] = pe;
        } 
      }
    }

    VertexID number_of_ghost_vertices = ghost_pe.size();
    VertexID number_of_edges = edges_.size();

    GraphOutputType cg(rank_, size_);

    // Build graph
    // Static graphs also take the number of edges
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, StaticGraph>::value) {
      cg.StartConstruct(num_local_components_, 
                        number_of_ghost_vertices, 
                        number_of_edges,
                        from);
    } else if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraph>::value) {
      cg.StartConstruct(num_local_components_, 
                        number_of_ghost_vertices, 
                        num_global_components_);
    } else {
      cg.StartConstruct(num_local_components_, 
                        number_of_ghost_vertices, 
                        from);
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraph>::value) {
      for (VertexID v = 0; v < num_local_components_; v++) {
          cg.AddVertex(from + v);
      }
    }
  

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < num_local_components_; v++) {
          cg.SetVertexLabel(v, from + v);
          cg.SetVertexRoot(v, rank_);
      }
    }

    // Initialize ghost vertices
    for (const auto &kv : ghost_pe) {
      cg.AddGhostVertex(kv.first, kv.second);
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, StaticGraph>::value) {
      SortEdges<GraphOutputType>(cg, edges_);
    }

    for (auto &edge : edges_) {
      cg.AddEdge(cg.GetLocalID(std::get<0>(edge)), std::get<1>(edge), std::get<2>(edge));
    }

    cg.FinishConstruct();
    return cg;
  }

  template <typename GraphOutputType>
  GraphOutputType BuildLocalContractionGraph() {
    // Identify ghost vertices
    google::dense_hash_map<VertexID, PEID> ghost_pe; 
    ghost_pe.set_empty_key(EmptyKey);
    ghost_pe.set_deleted_key(DeleteKey);
    for (auto &e : edges_) {
      VertexID target = std::get<1>(e);
      PEID pe = std::get<2>(e);
      if (std::get<2>(e) != rank_) {
        if (ghost_pe.find(target) == ghost_pe.end()) {
            ghost_pe[target] = pe;
        } 
      }
    }

    VertexID number_of_ghost_vertices = ghost_pe.size();
    VertexID number_of_edges = edges_.size();
    VertexID number_of_local_vertices = local_components_.size();

    GraphOutputType cg(rank_, size_);

    // Build graph
    // Static graphs also take the number of edges
    // num_global_components has to be original number of vertices?
    // from has to be smallest id 
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, StaticGraph>::value) {
      cg.StartConstruct(number_of_local_vertices, 
                        number_of_ghost_vertices, 
                        number_of_edges,
                        smallest_vertex_id_);
    } else if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraph>::value) {
      cg.StartConstruct(number_of_local_vertices, 
                        number_of_ghost_vertices, 
                        offset_);
    } else {
      cg.StartConstruct(number_of_local_vertices, 
                        number_of_ghost_vertices, 
                        smallest_vertex_id_);
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraph>::value) {
      for (auto &v : local_components_) {
        cg.AddVertex(v);
      }
    }
  
    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphOutputType, SemidynamicGraphCommunicator>::value) {
      cg.ForallLocalVertices([&](const VertexID v) {
          cg.SetVertexLabel(v, cg.GetGlobalID(v));
          cg.SetVertexRoot(v, rank_);
      });
    }

    // Initialize ghost vertices
    for (const auto &kv : ghost_pe) {
      cg.AddGhostVertex(kv.first, kv.second);
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value
                  || std::is_same<GraphOutputType, StaticGraph>::value) {
      SortEdges<GraphOutputType>(cg, edges_);
    }

    for (auto &edge : edges_) {
      cg.AddEdge(cg.GetLocalID(std::get<0>(edge)), std::get<1>(edge), std::get<2>(edge));
    }

    cg.FinishConstruct();
    return cg;
  }

  template <typename GraphOutputType>
  static void SortEdges(const GraphOutputType &g, std::vector<std::tuple<VertexID, VertexID, PEID>> &edge_list) {
    std::sort(edge_list.begin(), edge_list.end(), [&](std::tuple<VertexID, VertexID, PEID> &left, std::tuple<VertexID, VertexID, PEID> &right) {
        VertexID lhs_source = g.GetLocalID(std::get<0>(left));
        VertexID lhs_target = g.GetLocalID(std::get<1>(left));
        VertexID rhs_source = g.GetLocalID(std::get<0>(right));
        VertexID rhs_target = g.GetLocalID(std::get<1>(right));
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });
  }
};

#endif
