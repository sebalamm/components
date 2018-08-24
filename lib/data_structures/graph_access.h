/******************************************************************************
 * graph_access.h
 *
 * Data structure for maintaining the (undirected) graph
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

#ifndef _GRAPH_ACCESS_H_
#define _GRAPH_ACCESS_H_

#include <mpi.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <sstream>
#include <deque>
#include <tuple>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <google/sparse_hash_set>

#include "config.h"

struct Vertex {
  EdgeID first_edge_;

  Vertex() : first_edge_(0) {}
  explicit Vertex(EdgeID e) : first_edge_(e) {}
};

struct LocalVertexData {
  bool is_interface_vertex_;

  LocalVertexData()
      : is_interface_vertex_(false) {}
  LocalVertexData(const VertexID id, bool interface)
      : is_interface_vertex_(interface) {}
};

struct VertexPayload {
  VertexID deviate_;
  VertexID label_;
  PEID root_;

  VertexPayload()
      : deviate_(std::numeric_limits<VertexID>::max() - 1),
        label_(0),
        root_(0) {}

  VertexPayload(VertexID deviate,
                VertexID label,
                PEID root)
      : deviate_(deviate),
        label_(label),
        root_(root) {}

  bool operator==(const VertexPayload &rhs) const {
    return std::tie(deviate_, label_, root_)
        == std::tie(rhs.deviate_, rhs.label_, rhs.root_);
  }

  bool operator!=(const VertexPayload &rhs) const {
    return !(*this == rhs);
  }
};

struct GhostVertexData {
  PEID rank_;
  VertexID global_id_;

  GhostVertexData()
      : rank_(0), global_id_(0) {}
  GhostVertexData(PEID rank, VertexID global_id)
      : rank_(rank), global_id_(global_id) {}
};

struct Edge {
  VertexID target_;

  Edge() : target_(0) {}
  explicit Edge(VertexID target) : target_(target) {}
};

class NodeCommunicator;
class GraphAccess {
 public:
  GraphAccess(const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        number_of_vertices_(0),
        number_of_local_vertices_(0),
        number_of_global_vertices_(0),
        number_of_edges_(0),
        number_of_global_edges_(0),
        local_offset_(0),
        ghost_offset_(0),
        contraction_level_(0),
        max_contraction_level_(0),
        edge_buffers_(size),
        ghost_comm_(nullptr),
        vertex_counter_(0),
        edge_counter_(0) {}
  virtual ~GraphAccess() {
    free(ghost_comm_);
    ghost_comm_ = nullptr;
  };

  GraphAccess(GraphAccess &&rhs) = default;

  GraphAccess(const GraphAccess &rhs) = default;

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void StartConstruct(VertexID local_n, EdgeID local_m, VertexID local_offset);

  void FinishConstruct() { number_of_edges_ = edge_counter_; }

  //////////////////////////////////////////////
  // Graph iterators
  //////////////////////////////////////////////
  template<typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfLocalVertices(); ++v) {
      if (active_vertices_[contraction_level_][v]) callback(v);
    }
  }

  template<typename F>
  void ForallGhostVertices(F &&callback) {
    for (VertexID v = GetNumberOfLocalVertices(); v < GetNumberOfVertices(); ++v) {
      if (active_vertices_[contraction_level_][v]) callback(v);
    }
  }

  // TODO: If we decrease the contraction level, there more vertices than currently active
  // Store number of vertices in each level
  template<typename F>
  void ForallVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfVertices(); ++v) {
      if (active_vertices_[contraction_level_][v]) callback(v);
    }
  }

  template<typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { callback(edges_[v][e].target_); });
  }

  template<typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = 0; e < GetVertexDegree(v); ++e) {
      callback(e);
    }
  }

  bool IsAdjacent(const VertexID source, const VertexID target) {
    bool adj = false;
    ForallNeighbors(source, [&](const VertexID w) {
      if (w == target) adj = true;
    });
    return adj;
  }

  //////////////////////////////////////////////
  // Graph contraction
  //////////////////////////////////////////////
  inline void AllocateContractionVertices() {
    contraction_vertices_.resize(GetNumberOfVertices());
  }

  inline void SetContractionVertex(VertexID v, VertexID cv) {
    contraction_vertices_[v] = cv;
  }

  inline VertexID GetContractionVertex(VertexID v) const {
    return contraction_vertices_[v];
  }

  void DetermineActiveVertices() {
    // if (rank_ == 0 && contraction_level_ == 1) OutputLocal();
    active_vertices_.resize(contraction_level_ + 2);
    active_vertices_[contraction_level_ + 1].resize(number_of_vertices_, false);
    vertex_payload_.resize(contraction_level_ + 2);
    vertex_payload_[contraction_level_ + 1].resize(number_of_vertices_);

    // Parent information
    parent_.resize(contraction_level_ + 2);
    parent_[contraction_level_ + 1].resize(number_of_vertices_);

    // Update stacks
    added_edges_.resize(contraction_level_ + 2);
    removed_edges_.resize(contraction_level_ + 2);

    // Determine edges to communicate
    // Gather labels to communicate
    VertexID offset = GatherNumberOfGlobalVertices();
    google::sparse_hash_set<VertexID> send_ids(size_);
    std::vector<std::vector<VertexID>> send_buffers_a(size_);
    std::vector<std::vector<VertexID>> send_buffers_b(size_);
    std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
    std::vector<std::vector<VertexID>> receive_buffers(size_);
    for (int i = 0; i < size_; ++i) {
      send_buffers_a[i].clear();
      send_buffers_b[i].clear();
      receive_buffers[i].clear();
    }

    google::sparse_hash_set<VertexID> inserted_edges;
    google::sparse_hash_set<VertexID> remaining_vertices;
    std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);
    ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = GetVertexLabel(v);
      ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = GetVertexLabel(w);
        if (vlabel != wlabel) {
          VertexID update_id = vlabel + offset * wlabel;
          // TODO: Try to apply updates directly
          if (send_ids.find(update_id) == send_ids.end()) {
            // Local propagation
            PEID cv = v;
            std::pair<PEID, VertexID> parent = GetParent(cv);
            PEID pe = parent.first;
            while (GetVertexRoot(v) != rank_ && pe == rank_) {
              cv = GetLocalID(parent.second);
              parent = GetParent(cv);
              pe = parent.first;
            }
            // Send edge
            send_ids.insert(update_id);
            (*current_send_buffers)[pe].emplace_back(vlabel);
            (*current_send_buffers)[pe].emplace_back(wlabel);
            (*current_send_buffers)[pe].emplace_back(GetVertexRoot(w));
            (*current_send_buffers)[pe].emplace_back(parent.second);
          }
        }
        removed_edges_[contraction_level_].emplace_back(v, GetGlobalID(w));
      });
      RemoveAllEdges(v);
      vertex_payload_[contraction_level_ + 1][v] =
          {std::numeric_limits<VertexID>::max() - 1, GetVertexLabel(v), GetVertexRoot(v)};
      parent_[contraction_level_ + 1][v] = parent_[contraction_level_][v];
      active_vertices_[contraction_level_ + 1][v] = true;
    });

    ForallGhostVertices([&](VertexID v) {
      vertex_payload_[contraction_level_ + 1][v] =
          {std::numeric_limits<VertexID>::max() - 1, GetVertexLabel(v), GetVertexRoot(v)};
    });

    // Get adjacency (otherwise we get deadlocks with added edges)
    std::vector<bool> is_adj(size_);
    PEID num_adj = 0;
    for (PEID pe = 0; pe < size_; pe++) {
      is_adj[pe] = IsAdjacentPE(pe);
      if (is_adj[pe]) num_adj++;
    }

    // Propagate edge buffers until all vertices are converged
    contraction_level_++;
    std::vector<MPI_Request*> requests;
    requests.clear();
    int converged_globally = 0;
    while (converged_globally == 0) {
      int converged_locally = 1;

      // Send edges
      for (PEID pe = 0; pe < size_; ++pe) {
        if (is_adj[pe]) {
          auto *req = new MPI_Request();
          MPI_Isend((*current_send_buffers)[pe].data(), static_cast<int>((*current_send_buffers)[pe].size()), MPI_LONG, pe, 0, MPI_COMM_WORLD, req);
          requests.emplace_back(req);
        }
      }

      // Receive edges
      PEID messages_recv = 0;
      int message_length = 0;
      for (int i = 0; i < size_; ++i) receive_buffers[i].clear();
      receive_buffers[rank_] = (*current_send_buffers)[rank_];
      while (messages_recv < num_adj) {
        MPI_Status st{};
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
        MPI_Get_count(&st, MPI_LONG, &message_length);
        messages_recv++;

        receive_buffers[st.MPI_SOURCE].resize(message_length);
        MPI_Status rst{};
        MPI_Recv(&receive_buffers[st.MPI_SOURCE][0], message_length, MPI_LONG, st.MPI_SOURCE, 0, MPI_COMM_WORLD, &rst);
      }

      // Clear buffers
      if (current_send_buffers == &send_buffers_a) {
        for (int i = 0; i < size_; ++i) send_buffers_b[i].clear();
        current_send_buffers = &send_buffers_b;
      } else {
        for (int i = 0; i < size_; ++i) send_buffers_a[i].clear();
        current_send_buffers = &send_buffers_a;
      }
      requests.clear();

      // Receive edges and apply updates
      for (PEID pe = 0; pe < size_; ++pe) {
        if (receive_buffers[pe].size() < 4) continue;
        for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
          VertexID vlabel = receive_buffers[pe][i];
          VertexID wlabel = receive_buffers[pe][i + 1];
          VertexID wroot = receive_buffers[pe][i + 2];
          VertexID link = receive_buffers[pe][i + 3];
          VertexID update_id = vlabel + offset * wlabel;

          // If link root is on same PE just jump to it
          if (GetVertexRoot(GetLocalID(link)) == rank_ && inserted_edges.find(update_id) == end(inserted_edges)) {
            // V -> W
            edges_to_add[wroot].emplace_back(vlabel, wlabel);
            inserted_edges.insert(update_id);
            send_ids.insert(update_id);
            // W -> V
            if (wroot == rank_) {
              edges_to_add[rank_].emplace_back(wlabel, vlabel);
              inserted_edges.insert(wlabel + offset * vlabel);
              send_ids.insert(wlabel + offset * vlabel);
            }
            converged_locally = 0;
          // } else if (GetVertexRoot(GetLocalID(link)) != rank_ && send_ids.find(update_id) == send_ids.end()) {
          } else if (GetVertexRoot(GetLocalID(link)) != rank_) {
            // Local propagation
            PEID cv = GetLocalID(link);
            std::pair<PEID, VertexID> parent = GetParent(cv);
            PEID pe = parent.first;
            while (pe == rank_) {
              cv = GetLocalID(parent.second);
              parent = GetParent(cv);
              pe = parent.first;
            }
            // Send edge
            // send_ids.insert(update_id);
            (*current_send_buffers)[pe].emplace_back(vlabel);
            (*current_send_buffers)[pe].emplace_back(wlabel);
            (*current_send_buffers)[pe].emplace_back(wroot);
            (*current_send_buffers)[pe].emplace_back(parent.second);
            converged_locally = 0;
          }
        }
      }

      // Check if all PEs are done
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
    }

    // Insert edges and keep corresponding vertices
    std::fill(begin(active_vertices_[contraction_level_]), 
              end(active_vertices_[contraction_level_]), false);
    for (PEID pe = 0; pe < size_; pe++) {
      for (auto &e : edges_to_add[pe]) {
        VertexID vlabel = e.first;
        VertexID wlabel = e.second;
        VertexID vlocal = GetLocalID(vlabel);
        // Add edge
        AddEdge(vlocal, wlabel, pe);
        added_edges_[contraction_level_ - 1].emplace_back(vlocal, wlabel);
        VertexID wlocal = GetLocalID(wlabel);
        // Vertices remain active
        active_vertices_[contraction_level_][vlocal] = true;
        active_vertices_[contraction_level_][wlocal] = true;
        vertex_payload_[contraction_level_][vlocal] =
            {std::numeric_limits<VertexID>::max() - 1, vlabel, rank_};
        vertex_payload_[contraction_level_][wlocal] =
            {std::numeric_limits<VertexID>::max() - 1, wlabel, pe};
      }
    }
  }

  void MoveUpContraction() {
    // TODO: Does not work if graph is completely empty
    // OutputLocal();
    while (contraction_level_ > 0) {
      // Remove current edges from current level
      ForallLocalVertices([&](VertexID v) {
        RemoveAllEdges(v);
      });

      // Decrease level
      contraction_level_--;

      for (auto &e : removed_edges_[contraction_level_])
        AddEdge(std::get<0>(e), std::get<1>(e), GetPE(GetLocalID(std::get<1>(e))));

      // OutputLocal();
      // Update local labels
      ForallLocalVertices([&](VertexID v) {
        if (vertex_payload_[contraction_level_][v].label_ !=
            vertex_payload_[contraction_level_ + 1][v].label_)
          SetVertexPayload(v,
                           {0,
                            vertex_payload_[contraction_level_ + 1][v].label_,
                            rank_});
      });

      // Propagate labels
      int converged_globally = 0;
      while (converged_globally == 0) {
        int converged_locally = 1;
        // Receive variates
        SendAndReceiveGhostVertices();

        // Send current label from root
        ForallLocalVertices([&](VertexID v) {
          std::pair<PEID, VertexID> parent = GetParent(v);
          if (GetVertexLabel(GetLocalID(parent.second)) != GetVertexLabel(v)) {
            SetVertexPayload(v, {0, GetVertexLabel(GetLocalID(parent.second)), rank_});
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
    }
  }

  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline void SetOffsetArray(std::vector<VertexID> &&vertex_dist) {
    offset_array_ = vertex_dist;
  }

  PEID GetPEFromOffset(const VertexID v) const {
    for (PEID i = 1; i < (PEID) offset_array_.size(); ++i) {
      if (v < offset_array_[i]) return i - 1;
    }
    return rank_;
  }

  inline bool IsLocal(VertexID v) const {
    return v < number_of_local_vertices_;
  }

  inline bool IsLocalFromGlobal(VertexID v) const {
    return local_offset_ <= v && v < local_offset_ + number_of_local_vertices_;
  }

  inline bool IsGhost(VertexID v) const {
    return global_to_local_map_.find(GetGlobalID(v))
        != global_to_local_map_.end();
  }


  inline bool IsGhostFromGlobal(VertexID v) const {
    return global_to_local_map_.find(v) != global_to_local_map_.end();
  }

  inline bool IsInterface(VertexID v) const {
    return local_vertices_data_[v].is_interface_vertex_;
  }

  inline bool IsInterfaceFromGlobal(VertexID v) const {
    return local_vertices_data_[GetLocalID(v)].is_interface_vertex_;
  }

  inline VertexID GetLocalID(VertexID v) const {
    return IsLocalFromGlobal(v) ? v - local_offset_
                                : global_to_local_map_.find(v)->second;
  }

  inline VertexID GetGlobalID(VertexID v) const {
    return IsLocal(v) ? v + local_offset_
                      : ghost_vertices_data_[v - ghost_offset_].global_id_;
  }

  inline PEID GetPE(VertexID v) const {
    return IsLocal(v) ? rank_
                      : ghost_vertices_data_[v - ghost_offset_].rank_;

  }

  //////////////////////////////////////////////
  // Manage local vertices/edges
  //////////////////////////////////////////////
  inline VertexID GetNumberOfVertices() const { return number_of_vertices_; }

  inline VertexID GetNumberOfGlobalVertices() const { return number_of_global_vertices_; }

  inline VertexID GetNumberOfGlobalEdges() const { return number_of_global_edges_; }

  inline VertexID GetLocalOffset() const {
    return local_offset_;
  }

  inline VertexID GetNumberOfLocalVertices() const {
    return number_of_local_vertices_;
  }

  inline VertexID GetNumberOfGhostVertices() const { return number_of_vertices_ - number_of_local_vertices_; }

  inline EdgeID GetNumberOfEdges() const { return number_of_edges_; }

  VertexID GatherNumberOfGlobalVertices() {
    VertexID local_vertices = 0;
    ForallLocalVertices([&](const VertexID v) { local_vertices++; });
    // Check if all PEs are done
    MPI_Allreduce(&local_vertices,
                  &number_of_global_vertices_,
                  1,
                  MPI_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return number_of_global_vertices_;
  }

  VertexID GatherNumberOfGlobalEdges() {
    VertexID local_edges = 0;
    ForallLocalVertices([&](const VertexID v) { 
        ForallNeighbors(v, [&](const VertexID w) { local_edges++; });
    });
    // Check if all PEs are done
    MPI_Allreduce(&local_edges,
                  &number_of_global_edges_,
                  1,
                  MPI_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    number_of_global_edges_ /= 2;
    return number_of_global_edges_;
  }

  void OutputComponents() {
    VertexID global_num_vertices = GatherNumberOfGlobalVertices();
    // Gather component sizes
    std::unordered_map<VertexID, VertexID> local_component_sizes;
    ForallLocalVertices([&](const VertexID v) {
      VertexID c = GetVertexLabel(v);
      if (local_component_sizes.find(c) == end(local_component_sizes))
        local_component_sizes[c] = 0;
      local_component_sizes[c]++;
    });

    // Gather component message
    std::vector<std::pair<VertexID, VertexID>> local_components;
    // local_components.reserve(local_component_sizes.size());
    for(auto &kv : local_component_sizes)
      local_components.emplace_back(kv.first, kv.second);
    int num_local_components = local_components.size();

    // Exchange number of local components
    std::vector<int> num_components(size_);
    MPI_Gather(&num_local_components, 1, MPI_INT, &num_components[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute diplacements
    std::vector<int> displ_components(size_, 0);
    int num_global_components = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_components[i] = num_global_components;
      num_global_components += num_components[i];
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_LONG, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Exchange components
    std::vector<std::pair<VertexID, VertexID>> global_components(num_global_components);
    MPI_Gatherv(&local_components[0], num_local_components, MPI_COMP,
                &global_components[0], &num_components[0], &displ_components[0], MPI_COMP,
                ROOT, MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      std::unordered_map<VertexID, VertexID> global_component_sizes;
      for (auto &comp : global_components) {
        VertexID c = comp.first;
        VertexID size = comp.second;
        if (global_component_sizes.find(c) == end(global_component_sizes))
          global_component_sizes[c] = 0;
        global_component_sizes[c] += size;
      }

      std::unordered_map<VertexID, VertexID> condensed_component_sizes;
      for (auto &cs : global_component_sizes) {
        VertexID c = cs.first;
        VertexID size = cs.second;
        if (condensed_component_sizes.find(size) == end(condensed_component_sizes)) {
          condensed_component_sizes[size] = 0;
        }
        condensed_component_sizes[size]++;
      }

      // Build final vector
      std::vector<std::pair<VertexID, VertexID>> components;
      components.reserve(condensed_component_sizes.size());
      for(auto &kv : condensed_component_sizes)
        components.emplace_back(kv.first, kv.second);
      std::sort(begin(components), end(components));

      std::cout << "[ ";
      for (auto &comp : components)
        std::cout << comp.first << "(" << comp.second << ") ";
      std::cout << "]" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void SetVertexPayload(VertexID v, VertexPayload &&msg, bool propagate = true);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return vertex_payload_[contraction_level_][v];
  }

  void SetVertexMessage(const VertexID v, VertexPayload &&msg) {
    vertex_payload_[contraction_level_][v] = msg;
  }

  void SetParent(const VertexID v, const PEID parent_pe, const VertexID parent_v) {
    parent_[contraction_level_][v] = std::make_pair(parent_pe, parent_v);
  }

  inline std::string GetVertexString(const VertexID v) {
    std::stringstream out;
    out << "(" << GetVertexDeviate(v) << ","
        << GetVertexLabel(v) << ","
        << GetVertexRoot(v) << ")";
    return out.str();
  }

  inline VertexID GetVertexDeviate(const VertexID v) const {
    return vertex_payload_[contraction_level_][v].deviate_;
  }

  inline VertexID GetVertexLabel(const VertexID v) const {
    return vertex_payload_[contraction_level_][v].label_;
  }

  inline PEID GetVertexRoot(const VertexID v) const {
    return vertex_payload_[contraction_level_][v].root_;
  }

  inline std::pair<PEID, VertexID> & GetParent(const VertexID v) {
    return parent_[contraction_level_][v];
  }

  inline VertexID AddVertex() {
    return vertex_counter_++;
  }

  inline void RemoveVertex() {
  }

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank);

  void RemoveAllEdges(VertexID from);

  // Local IDs
  bool IsConnected(VertexID from, VertexID to) {
    ForallNeighbors(from, [&](VertexID v) {
        if (v == to) return true; 
    });
    return false;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return edges_[v].size();
  }

  void AddSubgraph(std::vector<std::tuple<VertexID, VertexID, VertexID>> &vertices,
                   std::vector<std::tuple<VertexID, VertexID, VertexID>> &edges) {
    // TODO: This does not work yet
    for (const auto &vp : vertices) {
      VertexID v = AddVertex();
      SetVertexPayload(v, {GetVertexDeviate(v), std::get<1>(vp), std::get<1>(vp)});
    }

    for (const auto &e : edges) {
      AddEdge(GetLocalID(std::get<0>(e)), std::get<1>(e), std::get<2>(e));
    }     
  }

  //////////////////////////////////////////////
  // Manage ghost vertices
  //////////////////////////////////////////////
  void SendAndReceiveGhostVertices();

  void ReceiveAndSendGhostVertices();

  inline void HandleGhostUpdate(const VertexID v,
                                const VertexID label,
                                const VertexID deviate,
                                const PEID root) {
    SetVertexPayload(v, {deviate, label, root});
  }

  //////////////////////////////////////////////
  // Manage adjacent PEs
  //////////////////////////////////////////////
  inline PEID GetNumberOfAdjacentPEs() const {
    PEID counter = 0;
    for (const bool is_adj : adjacent_pes_)
      if (is_adj) counter++;
    return counter;
  }

  inline std::vector<PEID> GetAdjacentPEs() const {
    std::vector<PEID> adjacent_pes;
    for (PEID i = 0; i < adjacent_pes_.size(); ++i) {
      if (adjacent_pes_[i]) adjacent_pes.push_back(i);
    }
    return adjacent_pes;
  }

  inline bool IsAdjacentPE(const PEID pe) const {
    return adjacent_pes_[pe];
  }

  inline void SetAdjacentPE(const PEID pe, const bool is_adj) {
    adjacent_pes_[pe] = is_adj;
  }

  //////////////////////////////////////////////
  // Communication
  //////////////////////////////////////////////
  void GatherGraphOnRoot(std::vector<VertexID> &global_vertices,
                         std::vector<int> &num_vertices,
                         std::vector<VertexID> &global_labels,
                         std::vector<std::pair<VertexID, VertexID>> &global_edges) {
    // Gather components of local graph
    std::vector<VertexID> local_vertices;
    std::vector<VertexID> local_labels;
    std::vector<std::pair<VertexID, VertexID>> local_edges;
    int num_local_vertices = 0;
    int num_local_edges = 0;
    ForallLocalVertices([&](const VertexID &v) {
      local_vertices.push_back(GetGlobalID(v));
      local_labels.push_back(GetVertexLabel(v));
      num_local_vertices++;
      ForallNeighbors(v, [&](const VertexID &w) {
        local_edges.emplace_back(GetGlobalID(v), GetGlobalID(w));
        num_local_edges++;
      });
    });

    // Gather number of vertices/edges for each PE
    std::vector<int> num_edges(size_);
    MPI_Gather(&num_local_vertices, 1, MPI_INT, &num_vertices[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&num_local_edges, 1, MPI_INT, &num_edges[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displ_vertices(size_);
    std::vector<int> displ_edges(size_);
    int num_global_vertices = 0;
    int num_global_edges = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_vertices[i] = num_global_vertices;
      displ_edges[i] = num_global_edges;
      num_global_vertices += num_vertices[i];
      num_global_edges += num_edges[i];
    }

    // Build datatype for edge
    MPI_Datatype MPI_EDGE;
    MPI_Type_vector(1, 2, 0, MPI_LONG, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);
    
    // Gather vertices/edges and labels for each PE
    global_vertices.resize(num_global_vertices);
    global_labels.resize(num_global_vertices);
    global_edges.resize(num_global_edges);
    MPI_Gatherv(&local_vertices[0], num_local_vertices, MPI_LONG,
                &global_vertices[0], &num_vertices[0], &displ_vertices[0], MPI_LONG,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_labels[0], num_local_vertices, MPI_LONG,
                &global_labels[0], &num_vertices[0], &displ_vertices[0], MPI_LONG,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_edges[0], num_local_edges, MPI_EDGE,
                &global_edges[0], &num_edges[0], &displ_edges[0], MPI_EDGE,
                ROOT, MPI_COMM_WORLD);
  } 

  void DistributeLabelsFromRoot(std::vector<VertexID> &global_labels, 
                                std::vector<int> &num_labels) {
    // Compute displacements
    std::vector<int> displ_labels(size_);
    int num_global_labels = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_labels[i] = num_global_labels;
      num_global_labels += num_labels[i];
    }

    // Gather local vertices
    std::vector<VertexID> local_vertices;
    int num_local_vertices = 0;
    ForallLocalVertices([&](const VertexID &v) {
      local_vertices.push_back(v);
      num_local_vertices++;
    });

    // Scatter to other PEs
    std::vector<VertexID> local_labels(num_local_vertices);
    MPI_Scatterv(&global_labels[0], &num_labels[0], &displ_labels[0], MPI_LONG, 
                 &local_labels[0], num_local_vertices, MPI_LONG, 
                 ROOT, MPI_COMM_WORLD);

    for (int i = 0; i < num_local_vertices; ++i) {
      VertexID v = local_vertices[i];
      SetVertexPayload(v, {GetVertexDeviate(v), 
                           local_labels[i], 
                           GetVertexRoot(v)});
    }
    SendAndReceiveGhostVertices();
  }

  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  void OutputLocal();

  void OutputLabels();

  void OutputGhosts();

  void Logging(bool active);

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<std::vector<Edge>> edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;
  std::vector<std::vector<VertexPayload>> vertex_payload_;
  std::vector<std::vector<std::pair<PEID, VertexID>>> parent_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_edges_;
  EdgeID number_of_global_edges_;

  // Vertex mapping
  VertexID local_offset_;
  std::vector<VertexID> offset_array_;

  VertexID ghost_offset_;
  std::unordered_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  VertexID contraction_level_, max_contraction_level_;
  std::vector<VertexID> contraction_vertices_;
  std::vector<std::vector<bool>> active_vertices_;
  std::vector<std::vector<std::pair<VertexID, VertexID>>> added_edges_;
  std::vector<std::vector<std::pair<VertexID, VertexID>>> removed_edges_;

  // Buffers
  std::vector<std::vector<VertexID>> edge_buffers_;

  // Adjacent PEs
  std::vector<bool> adjacent_pes_;

  // Communication interface
  NodeCommunicator *ghost_comm_;

  // Temporary counters
  VertexID vertex_counter_;
  EdgeID edge_counter_;
};

#endif
