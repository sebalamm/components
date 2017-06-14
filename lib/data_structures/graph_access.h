/******************************************************************************
 * graph_io.h
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

#include <vector>

#include "config.h"

struct Vertex {
  EdgeID first_edge_;
};

struct LocalVertexData {
  VertexID label_;
  bool is_interface_node_;
};

struct NonLocalVertexData {
  PEID rank;
  VertexID global_id_;
};

struct Edge {
  EdgeID local_target_;
};

class GraphAccess {
 public:
  GraphAccess() : max_degree_(0) {}
  virtual ~GraphAccess() {}

  GraphAccess(GraphAccess &&rhs) noexcept { std::cout << "move" << std::endl; }

  GraphAccess(const GraphAccess &rhs) { std::cout << "copy" << std::endl; }

  template <typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < NumberOfLocalVertices(); ++v) {
      callback(v);
    }
  }

  template <typename F>
  void ForallNeighbors(VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { callback(edges_[e].local_target_); });
  }

  template <typename F>
  void ForallLocalEdges(F &&callback) {
    for (EdgeID e = 0; e < NumberOfLocalEdges(); ++e) {
      callback(e);
    }
  }

  template <typename F>
  void ForallAdjacentEdges(VertexID v, F &&callback) {
    for (EdgeID e = FirstEdge(v); e <= LastEdge(v); ++e) {
      callback(e);
    }
  }

  inline VertexID VertexDegree(VertexID v) const {
    return FirstEdge(v + 1) - FirstEdge(v);
  }

  inline VertexID NumberOfLocalVertices() const {
    return number_of_local_vertices_;
  }

  inline VertexID NumberOfGhostVertices() const {
    return vertices_.size() - number_of_local_vertices_ - 1;
  }

  inline VertexID NumberOfGlobalVertices() const {
    return number_of_global_vertices_;
  }

  inline EdgeID NumberOfLocalEdges() const { return number_of_local_edges_; }

  inline EdgeID NumberOfGlobalEdges() const { return number_of_global_edges_; }

  VertexID MaxVertexDegree() {
    if (max_degree_ == 0) {
      VertexID local_max = 0;
      ForallLocalVertices([&](VertexID v) {
        if (VertexDegree(v) > local_max) local_max = VertexDegree(v);
      });
      MPI_Reduce(&local_max, &max_degree_, 1, MPI_LONG, MPI_MAX, ROOT,
                 MPI_COMM_WORLD);
    }
    return max_degree_;
  }

  inline bool IsLocal(VertexID v) {
    return (range_from_ <= v && range_to_ <= v);
  }

  inline VertexID GetLocalID(VertexID v) {
    return IsLocal(v) ? v - range_from_ : global_to_local_map_[v];
  }

  inline VertexID GetGlobalID(VertexID v) {
    return IsLocal(v) ? v + range_from_ : non_local_vertices_data_[v-ghost_offset_].global_id_;
  }

  VertexID CreateVertex() { 
    return 0;
  }

  EdgeID CreateEdge(VertexID from, VertexID to) {
    return 0;
  }

  inline VertexID GetVertexLabel(VertexID v) const {
    return local_vertices_data_[v].label_;
  }

  inline void SetVertexLabel(VertexID v, VertexID label) {
    local_vertices_data_[v].label_ = label;
  }

  inline void InitRangeArray(std::vector<VertexID> &&dist) {
    vertex_ranges_ = std::move(dist);
  }

  inline void SetLocalRange(VertexID from, VertexID to) {
    range_from_ = from; 
    range_to_ = to;
  }

  inline std::pair<VertexID, VertexID> GetLocalRange() const {
    return std::make_pair(range_from_, range_to_);
  }

  PEID GetPEFromRange(VertexID v) const {
    for (PEID i = 1; i < (PEID)vertex_ranges_.size(); i++) {
      if (v < vertex_ranges_[i]) {
        return (i - 1);
      }
    }
    return -1;
  }

 private:
  VertexID max_degree_;
  std::vector<Vertex> vertices_;
  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<NonLocalVertexData> non_local_vertices_data_;
  std::vector<Edge> edges_;
  std::vector<VertexID> vertex_ranges_;
  
  std::unordered_map<VertexID, VertexID> global_to_local_map_;

  VertexID range_from_, range_to_;
  VertexID ghost_offset_;
  VertexID number_of_local_vertices_, number_of_global_vertices_; 
  EdgeID number_of_local_edges_, number_of_global_edges_;

  inline EdgeID FirstEdge(VertexID v) const { return vertices_[v].first_edge_; }

  inline EdgeID LastEdge(VertexID v) const {
    return vertices_[v + 1].first_edge_ - 1;
  }
};

#endif
