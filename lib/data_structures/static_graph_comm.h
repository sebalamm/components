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

#ifndef _STATIC_GRAPH_COMM_H_
#define _STATIC_GRAPH_COMM_H_

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

class StaticVertexCommunicator;
class StaticGraphCommunicator {
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

 public:
  StaticGraphCommunicator(const PEID rank, const PEID size);

  virtual ~StaticGraphCommunicator();

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void ResetCommunicator();

  void StartConstruct(const VertexID local_n, 
                      const VertexID ghost_n, 
                      const VertexID total_m,
                      const VertexID local_offset);

  void FinishConstruct() { 
    vertices_.resize(vertex_counter_ + 1);
    edges_.resize(edge_counter_ + 1);

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
  inline void SetOffsetArray(std::vector<std::pair<VertexID, VertexID>> &&vertex_dist) {
    offset_array_ = vertex_dist;
  }

  PEID GetPEFromOffset(const VertexID v) const {
    for (PEID i = 0; i < offset_array_.size(); ++i) {
      if (v >= offset_array_[i].first && v < offset_array_[i].second) {
        return i;
      }
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
    ForallLocalVertices([&](const VertexID v) { local_vertices++; });
    // Check if all PEs are done
    MPI_Allreduce(&local_vertices,
                  &number_of_global_vertices_,
                  1,
                  MPI_VERTEX,
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
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    number_of_global_edges_ /= 2;
    return number_of_global_edges_;
  }

  inline VertexID AddVertex() {
    return vertex_counter_++;
  }

  VertexID AddGhostVertex(VertexID v);

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank);

  void AddLocalEdge(VertexID from, VertexID to);

  void SetVertexPayload(VertexID v, VertexPayload &&msg, bool propagate = true);

  void ForceVertexPayload(VertexID v, VertexPayload &&msg);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return vertex_payload_[v];
  }

  void SetVertexMessage(const VertexID v, VertexPayload &&msg) {
    vertex_payload_[v] = msg;
  }

  inline std::string GetVertexString(const VertexID v) {
    std::stringstream out;
    out << "(" << GetVertexDeviate(v) << ","
        << GetVertexLabel(v) << ","
        << GetVertexRoot(v) << ")";
    return out.str();
  }

  inline VertexID GetVertexDeviate(const VertexID v) const {
    return vertex_payload_[v].deviate_;
  }

  inline void SetVertexDeviate(const VertexID v, const VertexID deviate) {
    vertex_payload_[v].deviate_ = deviate;
  }

  inline VertexID GetVertexLabel(const VertexID v) const {
    return vertex_payload_[v].label_;
  }

  inline void SetVertexLabel(const VertexID v, const VertexID label) {
    vertex_payload_[v].label_ = label;
  }

  inline PEID GetVertexRoot(const VertexID v) const {
    return vertex_payload_[v].root_;
  }

  inline void SetVertexRoot(const VertexID v, const PEID root) {
    vertex_payload_[v].root_ = root;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return vertices_[v + 1].first_edge_ - vertices_[v].first_edge_; 
  }

  VertexID GetMaxDegree() {
    if (!max_degree_computed_) {
      max_degree_ = 0;
      ForallVertices([&](const VertexID v) {
          if (GetVertexDegree(v) > max_degree_) 
            max_degree_ = GetVertexDegree(v);
      });
      max_degree_computed_ = true;
    }
    return max_degree_;
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
  void OutputLocal();

  void OutputLabels();

  void OutputGhosts();

  void OutputComponents(std::vector<VertexID> &labels);

  void Logging(bool active);

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;
  std::vector<VertexPayload> vertex_payload_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_edges_;
  EdgeID number_of_cut_edges_;
  EdgeID number_of_global_edges_;

  VertexID max_degree_;
  bool max_degree_computed_;

  // Vertex mapping
  VertexID local_offset_;
  std::vector<std::pair<VertexID, VertexID>> offset_array_;

  VertexID ghost_offset_;
  google::dense_hash_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  std::vector<VertexID> contraction_vertices_;

  // Adjacent PEs
  std::vector<bool> adjacent_pes_;

  // Communication interface
  StaticVertexCommunicator *ghost_comm_;

  // Temporary counters
  VertexID vertex_counter_;
  EdgeID edge_counter_;
  VertexID ghost_counter_;
  VertexID last_source_;
};

#endif