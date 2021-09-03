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

#ifndef _STATIC_GRAPH_H_
#define _STATIC_GRAPH_H_

#include <mpi.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <stack>
#include <sstream>
#include <deque>
#include <tuple>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <google/sparse_hash_set>
#include <google/dense_hash_set>
#include <google/sparse_hash_map>
#include <google/dense_hash_map>

#include "config.h"
#include "timer.h"


class StaticGraph {

 public:
  StaticGraph(const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_cut_edges_(0),
      number_of_global_edges_(0),
      local_offset_(0),
      ghost_offset_(0),
      vertex_counter_(0),
      original_duplicate_id_(std::numeric_limits<VertexID>::max()),
      local_duplicate_id_(std::numeric_limits<VertexID>::max()),
      global_duplicate_id_(std::numeric_limits<VertexID>::max()),
      edge_counter_(0),
      ghost_counter_(0),
      last_source_(0),
      comm_time_(0.0),
      send_volume_(0),
      recv_volume_(0) {
    global_to_local_map_.set_empty_key(EmptyKey);
    global_to_local_map_.set_deleted_key(DeleteKey);
    // duplicates_.set_empty_key(EmptyKey);
    adjacent_pes_.set_empty_key(EmptyKey);
    adjacent_pes_.set_deleted_key(DeleteKey);
  }

  virtual ~StaticGraph() {};

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void StartConstruct(const VertexID local_n, 
                      const VertexID ghost_n, 
                      const VertexID total_m,
                      const VertexID local_offset) {
    number_of_local_vertices_ = local_n;
    number_of_vertices_ = local_n + ghost_n;
    number_of_edges_ = total_m;

    vertices_.resize(number_of_vertices_ + 1);
    // Fix first node being isolated
    if (local_n > 0) vertices_[0].first_edge_ = 0;
    edges_.resize(number_of_edges_);

    local_vertices_data_.resize(number_of_vertices_);
    ghost_vertices_data_.resize(ghost_n);

    local_offset_ = local_offset;
    ghost_offset_ = local_n;

    // Temp counter for properly counting new ghost vertices
    vertex_counter_ = local_n; 
    ghost_counter_ = local_n;
  }

  void FinishConstruct() { 
    vertices_.resize(vertex_counter_ + 1);
    edges_.resize(edge_counter_ + 1);

    number_of_vertices_ = vertex_counter_;
    for (VertexID v = 1; v <= vertex_counter_; v++) {
      if (vertices_[v].first_edge_ == std::numeric_limits<EdgeID>::max()) {
        vertices_[v].first_edge_ = vertices_[v - 1].first_edge_;
      }
    }
  }

  //////////////////////////////////////////////
  // Graph iterators
  //////////////////////////////////////////////
  template<typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfLocalVertices(); ++v) {
      callback(v);
    }
  }

  template<typename F>
  void ForallGhostVertices(F &&callback) {
    for (VertexID v = GetNumberOfLocalVertices(); v < GetNumberOfVertices(); ++v) {
      callback(v);
    }
  }

  template<typename F>
  void ForallVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfVertices(); ++v) {
      callback(v);
    }
  }

  template<typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { 
        callback(edges_[e].target_); 
    });
  }

  template<typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = GetFirstEdge(v); e < GetFirstInvalidEdge(v); ++e) {
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

  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline bool IsLocal(VertexID v) const {
    return v < number_of_local_vertices_;
  }

  inline bool IsLocalFromGlobal(VertexID v) const {
    return v == global_duplicate_id_ 
            || (v != original_duplicate_id_ && local_offset_ <= v && v < local_offset_ + number_of_local_vertices_);
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
    if (IsLocalFromGlobal(v)) {
      if (global_duplicate_id_ < std::numeric_limits<VertexID>::max() 
          && global_duplicate_id_ == v) {
        return local_duplicate_id_;
      } else {
        return v - local_offset_;
      }
    } else {
      return global_to_local_map_.find(v)->second;
    }
  }

  inline VertexID GetGlobalID(VertexID v) const {
    // TODO This says 0 is local (even tough its 63)
    if (IsLocal(v)) {
      if (local_duplicate_id_ < std::numeric_limits<VertexID>::max()
          && local_duplicate_id_ == v) {
        return global_duplicate_id_;
      } else {
        return v + local_offset_;
      }
    } else {
      return ghost_vertices_data_[v - ghost_offset_].global_id_;
    }
  }

  inline PEID GetPE(VertexID v) const {
    return IsLocal(v) ? rank_
                      : ghost_vertices_data_[v - ghost_offset_].rank_;
  }

  inline std::pair<VertexID, VertexID> GetDuplicate() {
    return std::make_pair(global_duplicate_id_, local_duplicate_id_);
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

  inline EdgeID GetNumberOfCutEdges() const { return number_of_cut_edges_; }

  inline void ResetNumberOfCutEdges() { number_of_cut_edges_ = 0; }

  inline EdgeID GetFirstEdge(const VertexID v) const {
    return vertices_[v].first_edge_;
  }

  inline EdgeID GetFirstInvalidEdge(const VertexID v) const {
    return vertices_[v + 1].first_edge_;
  }

  VertexID GatherNumberOfGlobalVertices() {
    VertexID local_vertices = 0;
    ForallLocalVertices([&](const VertexID v) { 
        local_vertices++; 
    });
    // Check if all PEs are done
    comm_timer_.Restart();
    MPI_Allreduce(&local_vertices,
                  &number_of_global_vertices_,
                  1,
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    comm_time_ += comm_timer_.Elapsed();
    return number_of_global_vertices_;
  }

  VertexID GatherNumberOfGlobalEdges() {
    VertexID local_edges = 0;
    ForallLocalVertices([&](const VertexID v) { 
        ForallNeighbors(v, [&](const VertexID w) { local_edges++; });
    });
    // Check if all PEs are done
    comm_timer_.Restart();
    MPI_Allreduce(&local_edges,
                  &number_of_global_edges_,
                  1,
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    comm_time_ += comm_timer_.Elapsed();
    number_of_global_edges_ /= 2;
    return number_of_global_edges_;
  }

  inline VertexID AddVertex() {
    return vertex_counter_++;
  }

  inline VertexID AddGhostVertex(VertexID v, PEID pe) {
    global_to_local_map_[v] = ghost_counter_;

    if (vertex_counter_ >= local_vertices_data_.size()) {
      local_vertices_data_.resize(vertex_counter_ + 1);
    }
    if (ghost_counter_ >= ghost_vertices_data_.size()) {
      ghost_vertices_data_.resize(ghost_counter_ + 1);
    }

    // Update data
    local_vertices_data_[vertex_counter_].is_interface_vertex_ = false;
    ghost_vertices_data_[ghost_counter_ - ghost_offset_].rank_ = pe;
    ghost_vertices_data_[ghost_counter_ - ghost_offset_].global_id_ = v;

    vertex_counter_++;
    return ghost_counter_++;
  }

  inline void AddDuplicateVertex(VertexID original_id, VertexID global_id) {
    // "Remap" duplicate
    original_duplicate_id_ = original_id;
    local_duplicate_id_ = original_id - local_offset_;
    global_duplicate_id_ = global_id;
  }

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank) {
    if (IsLocalFromGlobal(to)) {
      AddLocalEdge(from, to);
    } else {
      if (rank == size_) {
        std::cout << "This shouldn't happen" << std::endl;
        exit(1);
      }
      if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
        local_vertices_data_[from].is_interface_vertex_ = true;
        number_of_cut_edges_++;
        AddLocalEdge(from, to);
        SetAdjacentPE(rank, true);
      } else {
        std::cout << "This shouldn't happen" << std::endl;
        exit(1);
      }
    }
    if (from > last_source_) last_source_ = from;
    return edge_counter_;
  }

  void AddLocalEdge(VertexID from, VertexID to) {
    if (edge_counter_ >= edges_.size()) {
      edges_.resize(edge_counter_ + 1);
    }
    edges_[edge_counter_].target_ = GetLocalID(to);
    edge_counter_++;
    if (from + 1 >= vertices_.size()) {
      vertices_.resize(from + 2);
    }
    // std::cout << "R" << rank_ << " add edge (" << GetGlobalID(from) << "," << to << ") local (" << from << "," << GetLocalID(to) << ")" << std::endl;
    vertices_[from + 1].first_edge_ = edge_counter_;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return vertices_[v + 1].first_edge_ - vertices_[v].first_edge_; 
  }

  //////////////////////////////////////////////
  // Manage adjacent PEs
  //////////////////////////////////////////////
  inline PEID GetNumberOfAdjacentPEs() const {
    return adjacent_pes_.size();
  }

  template<typename F>
  void ForallAdjacentPEs(F &&callback) {
    for (const PEID &pe : adjacent_pes_) {
      callback(pe);
    }
  }

  inline bool IsAdjacentPE(const PEID pe) const {
    return adjacent_pes_.find(pe) != adjacent_pes_.end();
  }

  void SetAdjacentPE(const PEID pe, const bool is_adj) {
    if (pe == rank_) return;
    if (is_adj) {
      if (IsAdjacentPE(pe)) return;
      else adjacent_pes_.insert(pe);
    } else {
      if (!IsAdjacentPE(pe)) return;
      else adjacent_pes_.erase(pe);
    }
  }

  void ResetAdjacentPEs() {
    adjacent_pes_.clear();
  }
  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  void OutputLocal() {
    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [V] "
          << GetGlobalID(v) << " (local_id=" << v
           << ", pe=" << rank_ << ")";
      std::cout << out.str() << std::endl;
    });

    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [N] "
          << GetGlobalID(v) << " -> ";
      ForallNeighbors(v, [&](VertexID u) {
        // out << GetGlobalID(u) << " ";
        out << GetGlobalID(u) << " (local_id=" << u
            << ", pe=" << GetPE(u) << ") ";
      });
      std::cout << out.str() << std::endl;
    });
  }

  void OutputGhosts() {
    std::cout << "[R" << rank_ << "] [G] [ ";
    for (auto &e : global_to_local_map_) {
      std::cout << e.first << " ";
    }
    std::cout << "]" << std::endl;
  }

  void OutputComponents(std::vector<VertexID> &labels) {
    VertexID global_num_vertices = GatherNumberOfGlobalVertices();
    // Gather component sizes
    google::dense_hash_map<VertexID, VertexID> local_component_sizes; 
    local_component_sizes.set_empty_key(EmptyKey);
    local_component_sizes.set_deleted_key(DeleteKey);
    ForallLocalVertices([&](const VertexID v) {
      if (GetGlobalID(v) <= global_num_vertices) {
        VertexID c = labels[v];
        if (local_component_sizes.find(c) == end(local_component_sizes))
          local_component_sizes[c] = 0;
        local_component_sizes[c]++;
      }
    });

    // Gather component message
    std::vector<std::pair<VertexID, VertexID>> local_components;
    for(auto &kv : local_component_sizes)
      local_components.emplace_back(kv.first, kv.second);
    // [MEMORY]: Might be too small
    int num_local_components = local_components.size();

    // Exchange number of local components
    std::vector<int> num_components(size_);
    MPI_Gather(&num_local_components, 1, MPI_INT, &num_components[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute diplacements
    std::vector<int> displ_components(size_, 0);
    // [MEMORY]: Might be too small
    int num_global_components = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_components[i] = num_global_components;
      num_global_components += num_components[i];
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Exchange components
    std::vector<std::pair<VertexID, VertexID>> global_components(num_global_components);
    MPI_Gatherv(&local_components[0], num_local_components, MPI_COMP,
                &global_components[0], &num_components[0], &displ_components[0], MPI_COMP,
                ROOT, MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      google::dense_hash_map<VertexID, VertexID> global_component_sizes; 
      global_component_sizes.set_empty_key(EmptyKey);
      global_component_sizes.set_deleted_key(DeleteKey);
      for (auto &comp : global_components) {
        VertexID c = comp.first;
        VertexID size = comp.second;
        if (global_component_sizes.find(c) == end(global_component_sizes))
          global_component_sizes[c] = 0;
        global_component_sizes[c] += size;
      }

      google::dense_hash_map<VertexID, VertexID> condensed_component_sizes; 
      condensed_component_sizes.set_empty_key(EmptyKey);
      condensed_component_sizes.set_deleted_key(DeleteKey);
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

      std::cout << "COMPONENTS [ ";
      for (auto &comp : components)
        std::cout << "size=" << comp.first << " (num=" << comp.second << ") ";
      std::cout << "]" << std::endl;
    }

  }

  void Logging(bool active);

  inline float GetCommTime() {
    return comm_time_;
  }

  inline VertexID GetSendVolume() {
    return send_volume_;
  }

  inline VertexID GetReceiveVolume() {
    return recv_volume_;
  }

 protected:
  // Structs
  struct Vertex {
    EdgeID first_edge_;

    Vertex() : first_edge_(std::numeric_limits<EdgeID>::max()) {}
    explicit Vertex(EdgeID e) : first_edge_(e) {}
  };

  struct LocalVertexData {
    bool is_interface_vertex_;

    LocalVertexData()
        : is_interface_vertex_(false) {}
    LocalVertexData(const VertexID id, bool interface)
        : is_interface_vertex_(interface) {}
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

  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_edges_;
  EdgeID number_of_cut_edges_;
  EdgeID number_of_global_edges_;

  // Vertex mapping
  VertexID local_offset_;
  VertexID ghost_offset_;
  google::dense_hash_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  std::vector<VertexID> contraction_vertices_;

  // Duplicates
  // google::dense_hash_set<VertexID> duplicates_;
  VertexID original_duplicate_id_;
  VertexID local_duplicate_id_;
  VertexID global_duplicate_id_;

  // Adjacent PEs
  google::dense_hash_set<PEID> adjacent_pes_;

  // Temporary counters
  VertexID vertex_counter_;
  EdgeID edge_counter_;
  VertexID ghost_counter_;
  VertexID last_source_;

  // Statistics
  float comm_time_;
  Timer comm_timer_;
  VertexID send_volume_;
  VertexID recv_volume_;
};

#endif
