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

#include <unordered_map>
#include <vector>

#include "config.h"

struct Vertex {
  EdgeID first_edge_;

  Vertex(EdgeID e) : first_edge_(e) {}
};

struct LocalVertexData {
  VertexID label_;
  bool is_interface_node_;

  LocalVertexData(VertexID label, bool interface)
      : label_(label), is_interface_node_(interface) {}
};

struct NonLocalVertexData {
  PEID rank_;
  VertexID global_id_;

  NonLocalVertexData(PEID rank, VertexID global_id)
      : rank_(rank), global_id_(global_id) {}
};

struct Edge {
  VertexID local_target_;

  Edge(VertexID target) : local_target_(target) {}
};

class GraphAccess {
 public:
  GraphAccess() {}
  GraphAccess(const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        max_degree_(0),
        pe_div_(0),
        range_from_(0),
        range_to_(0),
        ghost_offset_(0),
        number_of_local_vertices_(0),
        number_of_global_vertices_(0),
        number_of_local_edges_(0),
        number_of_global_edges_(0),
        vertex_counter_(0),
        degree_counter_(0),
        edge_counter_(0) {}
  virtual ~GraphAccess() {}

  GraphAccess(GraphAccess &&rhs) noexcept { std::cout << "move" << std::endl; }

  GraphAccess(const GraphAccess &rhs) { std::cout << "copy" << std::endl; }

  void Construct(const VertexID local_n, const EdgeID local_m,
                 const VertexID global_n, const EdgeID global_m) {
    number_of_local_vertices_ = local_n;
    number_of_global_vertices_ = global_n;
    number_of_local_edges_ = local_m;
    number_of_global_edges_ = global_m;

    vertices_.resize(local_n + 1, 0);
    local_vertices_data_.resize(local_n + 1, {0, false});
    edges_.resize(local_m + 1, 0);

    pe_div_ = ceil(global_n / (double)size_);
  }

  template <typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < NumberOfLocalVertices(); ++v) {
      callback(v);
    }
  }

  template <typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v,
                        [&](EdgeID e) { callback(edges_[e].local_target_); });
  }

  template <typename F>
  void ForallLocalEdges(F &&callback) {
    for (EdgeID e = 0; e < NumberOfLocalEdges(); ++e) {
      callback(e);
    }
  }

  template <typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = FirstEdge(v); e <= LastEdge(v); ++e) {
      callback(e);
    }
  }

  inline VertexID VertexDegree(const VertexID v) const {
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

  inline bool IsLocal(VertexID v) const {
    return (range_from_ <= v && range_to_ <= v);
  }

  inline bool IsLocalTarget(VertexID v) const {
    return (range_from_ <= v && v <= range_to_);
  }

  inline bool IsGhost(VertexID v) {
    return global_to_local_map_.find(v) != global_to_local_map_.end();
  }

  inline VertexID GetLocalID(VertexID v) {
    return IsLocal(v) ? v - range_from_ : global_to_local_map_[v];
  }

  inline VertexID GetGlobalID(VertexID v) const {
    return IsLocal(v) ? v + range_from_
                      : non_local_vertices_data_[v - ghost_offset_].global_id_;
  }

  PEID GetPE(VertexID v) const {
    return IsLocal(v) ? rank_
                      : non_local_vertices_data_[v - ghost_offset_].rank_;
  }

  VertexID CreateVertex() {
    degree_counter_ = 0;
    return vertex_counter_++;
  }

  EdgeID CreateEdge(VertexID from, VertexID to) {
    if (IsLocalTarget(to))
      edges_[edge_counter_].local_target_ = to - range_from_;
    else {
      local_vertices_data_[from].is_interface_node_ = true;
      if (IsGhost(to))
        edges_[edge_counter_].local_target_ = global_to_local_map_[to];
      else {
        global_to_local_map_[to] = number_of_local_vertices_++;
        edges_[edge_counter_].local_target_ = global_to_local_map_[to];

        PEID neighbor = GetPEFromRange(to);
        vertices_.emplace_back(0);
        local_vertices_data_.emplace_back(to, false);
        non_local_vertices_data_.emplace_back(neighbor, to);
      }
    }

    EdgeID e_prime = edge_counter_++;
    vertices_[from + 1].first_edge_ = edge_counter_;

    // if (last_source + 1 < from) {
    //   for (VertexID i = from; i > last_source + 1; --i) {
    //     vertices_[i].first_edge_ = vertices[last_source + 1].first_edge_;
    //   }
    // }
    // last_source = from;

    degree_counter_++;
    if (max_degree_ < degree_counter_) max_degree_ = degree_counter_;

    return e_prime;
  }

  inline void SetVertexLabel(const VertexID v, const VertexID label) {
    local_vertices_data_[v].label_ = label;
  }

  inline VertexID GetVertexLabel(const VertexID v) const {
    return local_vertices_data_[v].label_;
  }

  inline void SetLocalRange(const VertexID from, const VertexID to) {
    range_from_ = from;
    range_to_ = to;
  }

  inline std::pair<VertexID, VertexID> GetLocalRange() const {
    return std::make_pair(range_from_, range_to_);
  }

  inline void SetRangeArray(std::vector<VertexID> &&dist) {
    vertex_ranges_ = std::move(dist);
  }

  // Get target PE from vertex ranges
  PEID GetPEFromRange(const VertexID v) const {
    for (PEID i = 1; i < (PEID)vertex_ranges_.size(); i++) {
      if (v < vertex_ranges_[i]) {
        return (i - 1);
      }
    }
    return -1;
  }

 private:
  PEID rank_, size_;

  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;
  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<NonLocalVertexData> non_local_vertices_data_;
  std::vector<VertexID> vertex_ranges_;

  std::unordered_map<VertexID, VertexID> global_to_local_map_;

  VertexID max_degree_;
  VertexID pe_div_;

  VertexID range_from_, range_to_;
  VertexID ghost_offset_;
  VertexID number_of_local_vertices_, number_of_global_vertices_;
  EdgeID number_of_local_edges_, number_of_global_edges_;

  VertexID vertex_counter_;
  VertexID degree_counter_;
  EdgeID edge_counter_;

  inline EdgeID FirstEdge(const VertexID v) const {
    return vertices_[v].first_edge_;
  }

  inline EdgeID LastEdge(const VertexID v) const {
    return vertices_[v + 1].first_edge_ - 1;
  }
};

#endif
