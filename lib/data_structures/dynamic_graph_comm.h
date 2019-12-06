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

#ifndef _DYNAMIC_GRAPH_COMM_H_
#define _DYNAMIC_GRAPH_COMM_H_

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
#include "payload.h"

class DynamicVertexCommunicator;
class DynamicGraphCommunicator {
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
    VertexID global_id_;
    PEID rank_;

    GhostVertexData()
        : global_id_(0), rank_(0) {}
    GhostVertexData(VertexID global_id, PEID rank)
        : global_id_(global_id_), rank_(rank) {}
  };

  struct Edge {
    VertexID target_;

    Edge() : target_(0) {}
    explicit Edge(VertexID target) : target_(target) {}
  };

 public:
  DynamicGraphCommunicator(const PEID rank, const PEID size);

  virtual ~DynamicGraphCommunicator();

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void ResetCommunicator();
  
  void StartConstruct(VertexID local_n, VertexID ghost_n, VertexID local_offset);

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

  void BuildLabelShortcuts() {
    // Gather labels
    google::dense_hash_set<VertexID> labels; 
    labels.set_empty_key(-1);
    ForallLocalVertices([&](const VertexID v) {
        labels.insert(GetVertexLabel(v));
    });

    // Init shortcuts
    // std::unordered_map<VertexID, std::pair<VertexID, VertexID>> smallest_deviate;
    google::dense_hash_map<VertexID, std::pair<VertexID, VertexID>> smallest_deviate; 
    smallest_deviate.set_empty_key(-1);
    FindSmallestDeviates(labels, smallest_deviate);

    // Set actual shortcuts
    for (auto &kv : smallest_deviate) 
      label_shortcut_[kv.first] = kv.second.second;
  }

  void FindSmallestDeviates(google::dense_hash_set<VertexID> labels,
                            google::dense_hash_map<VertexID, std::pair<VertexID, VertexID>> &smallest_deviate) {
    for (auto &v : labels) 
      smallest_deviate[v] = std::make_pair(std::numeric_limits<VertexID>::max(), 0);
    ForallLocalVertices([&](const VertexID v) {
      auto payload = GetVertexMessage(v);
      VertexID label = payload.label_;
      VertexID deviate = payload.deviate_;
#ifdef TIEBREAK_DEGREE
      VertexID degree = payload.degree_;
#endif
      VertexID root = payload.root_;
      if (smallest_deviate[label].first > deviate) {
        smallest_deviate[label].first = deviate;
        smallest_deviate[label].second = GetParent(v);
      }
    });
  }

  inline VertexID GetShortcutForLabel(VertexID label) {
    return label_shortcut_[label];
  }

  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline void SetOffsetArray(std::vector<std::pair<VertexID, VertexID>> &&vertex_dist) {
    offset_array_ = vertex_dist;
  }

  PEID GetPEFromOffset(const VertexID v) const {
    for (PEID i = 0; i < offset_array_.size(); ++i) {
      if (v >= offset_array_[i].first && v < offset_array_[i].second) return i;
    }
    return rank_;
  }

  inline bool IsLocal(VertexID v) const {
    return v < ghost_offset_;
  }

  inline bool IsLocalFromGlobal(VertexID v) {
    return global_to_local_map_.find(v) != global_to_local_map_.end() && IsLocal(global_to_local_map_[v]);
  }

  inline bool IsGhost(VertexID v) const {
    return v >= ghost_offset_;
  }

  inline bool IsGhostFromGlobal(VertexID v) {
    return global_to_local_map_.find(v) != global_to_local_map_.end() && IsGhost(global_to_local_map_[v]);
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
    return global_to_local_map_[v];
  }

  inline VertexID GetGlobalID(VertexID v) {
    return IsLocal(v) ? local_vertices_data_[v].global_id_ 
                      : ghost_vertices_data_[v - ghost_offset_].global_id_;
  }

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

  void SetVertexPayload(VertexID v, VertexPayload &&msg, bool propagate = true);

  void ForceVertexPayload(VertexID v, VertexPayload &&msg);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return IsLocal(v) ? local_payload_[v]
                      : ghost_payload_[v - ghost_offset_];
  }

  void SetVertexMessage(const VertexID v, VertexPayload &&msg) {
    if (IsLocal(v)) local_payload_[v] = msg;
    else ghost_payload_[v - ghost_offset_] = msg;
  }

  void SetParent(const VertexID v, const VertexID parent_v) {
    if (IsLocal(v)) local_parent_[v] = parent_v;
    else ghost_parent_[v - ghost_offset_] = parent_v;
  }

  inline std::string GetVertexString(const VertexID v) {
    std::stringstream out;
    out << "(" << GetVertexDeviate(v) << ","
        << GetVertexLabel(v) << ","
        << GetVertexRoot(v) << ")";
    return out.str();
  }

  inline VertexID GetVertexDeviate(const VertexID v) {
    return IsLocal(v) ? local_payload_[v].deviate_
                      : ghost_payload_[v - ghost_offset_].deviate_;
  }

  inline void SetVertexDeviate(const VertexID v, const VertexID deviate) {
    if (IsLocal(v)) local_payload_[v].deviate_ = deviate;
    else ghost_payload_[v - ghost_offset_].deviate_ = deviate;
  }

  inline VertexID GetVertexLabel(const VertexID v) {
    return IsLocal(v) ? local_payload_[v].label_
                      : ghost_payload_[v - ghost_offset_].label_;
  }

  inline void SetVertexLabel(const VertexID v, const VertexID label) {
    if (IsLocal(v)) local_payload_[v].label_ = label;
    else ghost_payload_[v - ghost_offset_].label_ = label;
  }

  inline PEID GetVertexRoot(const VertexID v) {
    return IsLocal(v) ? local_payload_[v].root_
                      : ghost_payload_[v - ghost_offset_].root_;
  }

  inline void SetVertexRoot(const VertexID v, const PEID root) {
    if (IsLocal(v)) local_payload_[v].root_ = root;
    else ghost_payload_[v - ghost_offset_].root_ = root;
  }

  inline VertexID GetParent(const VertexID v) {
    return IsLocal(v) ? local_parent_[v] : ghost_parent_[v - ghost_offset_];
  }

  inline VertexID AddVertex(VertexID v) {
    VertexID local_id = vertex_counter_++;
    global_to_local_map_[v] = local_id;

    // Update data
    local_vertices_data_.resize(local_vertices_data_.size() + 1);
    local_payload_.resize(local_payload_.size() + 1);
    local_adjacent_edges_.resize(local_adjacent_edges_.size() + 1);
    local_parent_.resize(local_parent_.size() + 1);
    local_active_.resize(local_active_.size() + 1);
    local_vertices_data_[local_id].is_interface_vertex_ = false;
    local_vertices_data_[local_id].global_id_ = v;

    // Set active
    local_active_[local_id] = true;

    number_of_vertices_++;
    number_of_local_vertices_++;
    return local_id;
  }

  VertexID AddGhostVertex(VertexID v);

  VertexID AddGhostVertex(VertexID v, PEID pe);

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank);

  EdgeID RelinkEdge(VertexID from, VertexID old_to, VertexID new_to, PEID rank);

  EdgeID RemoveEdge(VertexID from, VertexID to);

  void RemoveAllEdges(VertexID from);

  void AddLocalEdge(VertexID from, VertexID to);

  void AddGhostEdge(VertexID from, VertexID to);

  void ReserveEdgesForVertex(VertexID v, VertexID num_edges);

  // Local IDs
  bool IsConnected(VertexID from, VertexID to) {
    ForallNeighbors(from, [&](VertexID v) {
        if (v == to) return true; 
    });
    return false;
  }

  inline VertexID GetVertexDegree(const VertexID v) {
    return IsLocal(v) ? local_adjacent_edges_[v].size()
                      : ghost_adjacent_edges_[v - ghost_offset_].size();
  }

  //////////////////////////////////////////////
  // Manage ghost vertices
  //////////////////////////////////////////////
  void SendAndReceiveGhostVertices();

  void ReceiveAndSendGhostVertices();

  inline void HandleGhostUpdate(const VertexID v,
                                const VertexID label,
                                const VertexID deviate,
#ifdef TIEBREAK_DEGREE
                                const VertexID degree,
#endif
                                const PEID root) {
    SetVertexPayload(v, {deviate, 
                         label, 
#ifdef TIEBREAK_DEGREE
                         degree,
#endif
                         root});
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

  void SetAdjacentPE(const PEID pe, const bool is_adj);

  void ResetAdjacentPEs();


  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  bool CheckDuplicates();

  void OutputLocal();

  void OutputLabels();

  void OutputGhosts();

  void OutputComponents(std::vector<VertexID> &labels);

  void Logging(bool active);

  float GetCommTime();

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<std::vector<Edge>> local_adjacent_edges_;
  std::vector<std::vector<Edge>> ghost_adjacent_edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;

  std::vector<VertexPayload> local_payload_;
  std::vector<VertexPayload> ghost_payload_;

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
  std::vector<std::pair<VertexID, VertexID>> offset_array_;
  google::dense_hash_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  std::vector<VertexID> local_contraction_vertices_;
  std::vector<VertexID> ghost_contraction_vertices_;
  std::vector<bool> local_active_;
  std::vector<bool> ghost_active_;

  // Adjacent PEs
  std::vector<bool> adjacent_pes_;

  // Communication interface
  DynamicVertexCommunicator *ghost_comm_;

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
