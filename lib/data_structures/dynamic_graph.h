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

#ifndef _DYNAMIC_GRAPH_H_
#define _DYNAMIC_GRAPH_H_

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

class DynamicGraph {
 public:
  DynamicGraph(const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_cut_edges_(0),
      number_of_global_edges_(0),
      vertex_counter_(0),
      ghost_vertex_counter_(0),
      edge_counter_(0),
      ghost_offset_(0),
      comm_time_(0.0) {
    label_shortcut_.set_empty_key(EmptyKey);
    label_shortcut_.set_deleted_key(DeleteKey);
    global_to_local_map_.set_empty_key(EmptyKey);
    global_to_local_map_.set_deleted_key(DeleteKey);
    adjacent_pes_.set_empty_key(EmptyKey);
    adjacent_pes_.set_deleted_key(DeleteKey);
  }

  virtual ~DynamicGraph() {};

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void StartConstruct(VertexID local_n, VertexID ghost_n, VertexID ghost_offset) {
    ghost_offset_ = ghost_offset;
  }

  void FinishConstruct() { number_of_edges_ = edge_counter_; }

  //////////////////////////////////////////////
  // Graph iterators
  //////////////////////////////////////////////
  template<typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfLocalVertices(); ++v) {
      if (IsActive(v)) callback(v);
    }
  }

  template<typename F>
  void ForallGhostVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfGhostVertices(); ++v) {
      if (IsActive(v + ghost_offset_)) callback(v + ghost_offset_);
    }
  }

  template<typename F>
  void ForallVertices(F &&callback) {
    ForallLocalVertices(callback);
    ForallGhostVertices(callback);
  }

  template<typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    if (IsLocal(v)) {
      ForallAdjacentEdges(v, [&](EdgeID e) { 
          callback(local_adjacent_edges_[v][e].target_); 
      });
    } else {
      ForallAdjacentEdges(v, [&](EdgeID e) { 
          callback(ghost_adjacent_edges_[v - ghost_offset_][e].target_); 
      });
    }
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

  inline bool IsActive(const VertexID v) {
    return IsLocal(v) ? local_active_[v]
                      : ghost_active_[v - ghost_offset_];
  }

  void SetActive(VertexID v, bool is_active) {
    if (IsLocal(v)) local_active_[v] = is_active;
    else ghost_active_[v - ghost_offset_] = is_active;
  }


  //////////////////////////////////////////////
  // Graph contraction
  //////////////////////////////////////////////
  inline void AllocateContractionVertices() {
    local_contraction_vertices_.resize(GetNumberOfLocalVertices());
    ghost_contraction_vertices_.resize(GetNumberOfGhostVertices());
  }

  inline void SetContractionVertex(VertexID v, VertexID cv) {
    if (IsLocal(v)) local_contraction_vertices_[v] = cv;
    else ghost_contraction_vertices_[v - ghost_offset_] = cv;
  }

  inline VertexID GetContractionVertex(VertexID v) {
    return IsLocal(v) ? local_contraction_vertices_[v]
                      : ghost_contraction_vertices_[v - ghost_offset_];
  }


  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline bool IsLocal(VertexID v) const {
    return v < ghost_offset_;
  }

  inline bool IsLocalFromGlobal(VertexID v) {
    return global_to_local_map_.find(v) != global_to_local_map_.end() ? IsLocal(global_to_local_map_[v]) : false;
  }

  inline bool IsGhost(VertexID v) const {
    return v >= ghost_offset_;
  }

  inline bool IsGhostFromGlobal(VertexID v) {
    return global_to_local_map_.find(v) != global_to_local_map_.end() ? IsGhost(global_to_local_map_[v]) : false;
  }

  inline bool IsInterface(VertexID v) {
    return IsLocal(v) ? local_vertices_data_[v].is_interface_vertex_ 
                      : false;
  }

  inline void SetInterface(VertexID v, bool is_interface) {
    if (IsLocal(v)) local_vertices_data_[v].is_interface_vertex_ = is_interface;
  }

  inline bool IsInterfaceFromGlobal(VertexID v) {
    return IsLocalFromGlobal(v) ? local_vertices_data_[GetLocalID(v)].is_interface_vertex_ 
                                : false;
  }

  inline VertexID GetLocalID(VertexID v) {
    if (global_to_local_map_.find(v) != global_to_local_map_.end()) return global_to_local_map_[v];
    else {
      std::cout << "R" << rank_ << " This shouldn't happen (illegal get local on v=" << v << ")" << std::endl;
      exit(1);
    }
  }

  inline VertexID GetGlobalID(VertexID v) {
    return IsLocal(v) ? local_vertices_data_[v].global_id_ 
                      : ghost_vertices_data_[v - ghost_offset_].global_id_;
  }

  inline VertexID GetGhostOffset() { return ghost_offset_; }

  inline PEID GetPE(VertexID v) {
    return IsLocal(v) ? rank_
                      : ghost_vertices_data_[v - ghost_offset_].rank_;
  }

  inline void SetPE(VertexID v, PEID pe) {
    ghost_vertices_data_[v - ghost_offset_].rank_ = pe;
  }
  //////////////////////////////////////////////
  // Manage local vertices/edges
  //////////////////////////////////////////////
  inline VertexID GetNumberOfVertices() const { return number_of_vertices_; }

  inline VertexID GetNumberOfGlobalVertices() const { return number_of_global_vertices_; }

  inline VertexID GetNumberOfGlobalEdges() const { return number_of_global_edges_; }

  inline VertexID GetNumberOfLocalVertices() const {
    return number_of_local_vertices_;
  }

  inline VertexID GetNumberOfGhostVertices() const { return number_of_vertices_ - number_of_local_vertices_; }

  inline EdgeID GetNumberOfEdges() const { return number_of_edges_; }

  inline EdgeID GetNumberOfCutEdges() const { return number_of_cut_edges_; }

  inline void ResetNumberOfCutEdges() { number_of_cut_edges_ = 0; }

  VertexID GatherNumberOfGlobalVertices() {
    VertexID local_vertices = 0;
    ForallLocalVertices([&](const VertexID v) { local_vertices++; });
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

  void SetParent(const VertexID v, const VertexID parent_v) {
    if (IsLocal(v)) local_parent_[v] = parent_v;
    else ghost_parent_[v - ghost_offset_] = parent_v;
  }

  inline VertexID GetParent(const VertexID v) {
    return IsLocal(v) ? local_parent_[v] : ghost_parent_[v - ghost_offset_];
  }

  inline VertexID AddVertex(VertexID v) {
    global_to_local_map_[v] = vertex_counter_;

    // Update data
    if (vertex_counter_ >= local_vertices_data_.size()) {
      local_vertices_data_.resize(vertex_counter_ + 1);
      local_adjacent_edges_.resize(vertex_counter_ + 1);
      local_parent_.resize(vertex_counter_ + 1);
      local_active_.resize(vertex_counter_ + 1);
    }

    // Update data
    local_vertices_data_[vertex_counter_].is_interface_vertex_ = false;
    local_vertices_data_[vertex_counter_].global_id_ = v;
    local_active_[vertex_counter_] = true;

    number_of_vertices_++;
    number_of_local_vertices_++;
    return vertex_counter_++;
  }

  VertexID AddGhostVertex(VertexID v, PEID pe) {
    VertexID local_id = ghost_vertex_counter_ + ghost_offset_;
    global_to_local_map_[v] = local_id;

    // Update data
    if (ghost_vertex_counter_ >= ghost_vertices_data_.size()) {
      ghost_vertices_data_.resize(ghost_vertex_counter_ + 1);
      ghost_adjacent_edges_.resize(ghost_vertex_counter_ + 1);
      ghost_parent_.resize(ghost_vertex_counter_ + 1);
      ghost_active_.resize(ghost_vertex_counter_ + 1);
    }

    // Update data
    ghost_vertices_data_[local_id - ghost_offset_].global_id_ = v;
    ghost_vertices_data_[local_id - ghost_offset_].rank_ = pe;
    ghost_active_[local_id - ghost_offset_] = true;

    number_of_vertices_++;
    return ghost_vertex_counter_++;
  }

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank) {
    if (IsLocalFromGlobal(to)) {
      AddLocalEdge(from, to);
    } else {
      if (rank == size_) {
        std::cout << "This shouldn't happen (illegal add edge)" << std::endl;
        exit(1);
      }
      // NOTE: from always local
      local_vertices_data_[from].is_interface_vertex_ = true;
      if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
        number_of_cut_edges_++;
        AddGhostEdge(from, to);
        SetAdjacentPE(rank, true);
      } else {
        std::cout << "This shouldn't happen (illegal add edge)" << std::endl;
        exit(1);
      }
    }
    edge_counter_++;
    return edge_counter_;
  }

  void AddLocalEdge(VertexID from, VertexID to) {
    if (IsLocal(from)) {
      if (from >= local_adjacent_edges_.size()) {
        local_adjacent_edges_.resize(from + 1);
      }
      local_adjacent_edges_[from].emplace_back(global_to_local_map_[to]);
    }
    else if (IsGhost(from)) {
      if (from - ghost_offset_ >= ghost_adjacent_edges_.size()) {
        ghost_adjacent_edges_.resize(from - ghost_offset_ + 1);
      }
      ghost_adjacent_edges_[from - ghost_offset_].emplace_back(global_to_local_map_[to]);
    } else {
      std::cout << "This shouldn't happen (illegal add local edge)" << std::endl;
      exit(1);
    }
  }

  void AddGhostEdge(VertexID from, VertexID to) {
    AddLocalEdge(from, to);
  }

  EdgeID RemoveEdge(VertexID from, VertexID to) {
    VertexID delete_pos = ghost_offset_;
    if (IsLocal(from)) {
      for (VertexID i = 0; i < local_adjacent_edges_[from].size(); i++) {
        if (local_adjacent_edges_[from][i].target_ == to) {
          delete_pos = i;
          break;
        }
      }
      if (delete_pos != ghost_offset_) {
        local_adjacent_edges_[from].erase(local_adjacent_edges_[from].begin() + delete_pos);
        if (IsGhost(to)) number_of_cut_edges_--;
      } else {
        std::cout << "This shouldn't happen (illegal remove edge)" << std::endl;
        exit(1);
      }
    } else {
      std::cout << "This shouldn't happen (illegal remove edge)" << std::endl;
      exit(1);
    }
  }

  bool RelinkEdge(VertexID from, VertexID old_to, VertexID new_to, PEID rank) {
    VertexID old_to_local = GetLocalID(old_to);
    VertexID new_to_local = GetLocalID(new_to);

    if (IsLocal(new_to_local)) {
      if (IsGhost(old_to_local)) number_of_cut_edges_--;
    }
    else {
      if (!IsGhost(old_to_local)) number_of_cut_edges_++;
      if (rank == size_) {
        std::cout << "This shouldn't happen (illegal relink edge)" << std::endl;
        exit(1);
      }
      SetAdjacentPE(rank, true);
    }

    // Actual relink
    // NOTE: from should always be local
    bool success = false;
    for (VertexID i = 0; i < local_adjacent_edges_[from].size(); i++) {
      if (local_adjacent_edges_[from][i].target_ == old_to_local) {
        local_adjacent_edges_[from][i].target_ = new_to_local;
        success = true;
      }
    }

    return success;
  }

  void ReserveEdgesForVertex(VertexID v, VertexID num_edges) {
    if (IsLocal(v)) local_adjacent_edges_[v].reserve(num_edges);
    else ghost_adjacent_edges_[v - ghost_offset_].reserve(num_edges);
  }

  void RemoveAllEdges(VertexID from) {
    ForallNeighbors(from, [&](const VertexID w) {
      if (IsGhost(w)) number_of_cut_edges_--;
    });
    if (IsLocal(from)) local_adjacent_edges_[from].clear();
    else ghost_adjacent_edges_[from - ghost_offset_].clear();
  }

  // Local IDs
  bool IsConnected(VertexID from, VertexID to) {
    ForallNeighbors(from, [&](VertexID v) {
        if (v == to) return true; 
    });
    return false;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return IsLocal(v) ? local_adjacent_edges_[v].size()
                      : ghost_adjacent_edges_[v - ghost_offset_].size();
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
  bool CheckDuplicates() {
    // google::dense_hash_set<VertexID> neighbors;
    ForallLocalVertices([&](const VertexID v) {
      std::unordered_set<VertexID> neighbors;
      ForallNeighbors(v, [&](const VertexID w) {
        if (neighbors.find(w) != end(neighbors)) {
          std::cout << "[R" << rank_ << ":0] DUPL (" << GetGlobalID(v) << "," << GetGlobalID(w) << "[" << GetPE(w) << "])" << std::endl;
          return true;
        }
        neighbors.insert(w);
      });
    });
    return false;
  }

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
        out << GetGlobalID(u) << " (local_id=" << u
            << ", pe=" << GetPE(u) << ") ";
      });
      std::cout << out.str() << std::endl;
    });
  }

  void OutputGhosts() {
    PEID rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "[R" << rank << "] [G] [ ";
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
      VertexID c = labels[v];
      if (local_component_sizes.find(c) == end(local_component_sizes))
        local_component_sizes[c] = 0;
      local_component_sizes[c]++;
    });

    // Gather component message
    std::vector<std::pair<VertexID, VertexID>> local_components;
    // local_components.reserve(local_component_sizes.size());
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

  float GetCommTime() {
    return comm_time_;
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
    VertexID global_id_;

    LocalVertexData()
        : global_id_(0), is_interface_vertex_(false) {}
    LocalVertexData(VertexID global_id, bool interface)
        : global_id_(global_id), is_interface_vertex_(interface) {}
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
  std::vector<std::vector<Edge>> local_adjacent_edges_;
  std::vector<std::vector<Edge>> ghost_adjacent_edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;

  // Shortcutting
  std::vector<VertexID> local_parent_;
  std::vector<VertexID> ghost_parent_;
  google::dense_hash_map<VertexID, VertexID> label_shortcut_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_edges_;
  EdgeID number_of_cut_edges_;
  EdgeID number_of_global_edges_;

  // Vertex mapping
  google::dense_hash_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  std::vector<VertexID> local_contraction_vertices_;
  std::vector<VertexID> ghost_contraction_vertices_;
  std::vector<bool> local_active_;
  std::vector<bool> ghost_active_;

  // Adjacent PEs
  google::dense_hash_set<PEID> adjacent_pes_;

  // Temporary counters
  VertexID vertex_counter_;
  VertexID ghost_vertex_counter_;
  EdgeID edge_counter_;
  VertexID ghost_offset_;

  // Statistics
  float comm_time_;
  Timer comm_timer_;
};

#endif
