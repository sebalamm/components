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

#ifndef _MINIMAL_GRAPH_ACCESS_H_
#define _MINIMAL_GRAPH_ACCESS_H_

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


class MinimalGraphAccess {
  struct Edge {
    VertexID target_;

    Edge() : target_(0) {}
    explicit Edge(VertexID target) : target_(target) {}
  };

 public:
  MinimalGraphAccess(const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_local_edges_(0),
      number_of_global_edges_(0) {
    local_vertices_.set_empty_key(-1);
    local_vertices_.clear();
    ghost_vertices_.set_empty_key(-1);
    ghost_vertices_.clear();
    adjacent_edges_.set_empty_key(-1);
    adjacent_edges_.clear();
  }

  virtual ~MinimalGraphAccess() {};

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void StartConstruct() { }

  void FinishConstruct() { }

  //////////////////////////////////////////////
  // Graph iterators
  //////////////////////////////////////////////
  template<typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v : local_vertices_) {
      callback(v);
    }
  }

  template<typename F>
  void ForallGhostVertices(F &&callback) {
    for (auto &v : ghost_vertices_) {
      callback(v.first);
    }
  }

  template<typename F>
  void ForallVertices(F &&callback) {
    ForallLocalVertices(callback);
    ForallGhostVertices(callback);
  }
  
  template<typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { callback(adjacent_edges_[v][e].target_); });
  }

  template<typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = 0; e < GetVertexDegree(v); ++e) {
      callback(e);
    }
  }

  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline bool IsLocal(VertexID v) const {
    return local_vertices_.find(v) != end(local_vertices_);
  }

  inline bool IsGhost(VertexID v) const {
    return ghost_vertices_.find(v) != end(ghost_vertices_);
  }

  void PrintGhosts() const {
    for (const auto &kv : ghost_vertices_) {
      std::cout << "R" << rank_ << " g " << kv.first << " p " << kv.second << std::endl;
    }
  }

  //////////////////////////////////////////////
  // Manage local vertices/edges
  //////////////////////////////////////////////
  inline VertexID GetNumberOfLocalVertices() const { return number_of_local_vertices_; }

  inline EdgeID GetNumberOfLocalEdges() const { return number_of_local_edges_; }

  inline VertexID GetNumberOfGlobalVertices() const { return number_of_global_vertices_; }

  inline VertexID GetNumberOfGlobalEdges() const { return number_of_global_edges_; }

  VertexID GatherNumberOfGlobalVertices() {
    MPI_Allreduce(&number_of_local_vertices_,
                  &number_of_global_vertices_,
                  1,
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return number_of_global_vertices_;
  }

  VertexID GatherNumberOfGlobalEdges() {
    MPI_Allreduce(&number_of_local_edges_,
                  &number_of_global_edges_,
                  1,
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    number_of_global_edges_ /= 2;
    return number_of_global_edges_;
  }

  inline VertexID AddLocalVertex(VertexID global_id) {
    if (IsLocal(global_id)) {
      std::cout << "R" << rank_ << " This shouldn't happen (added local vertex)" << std::endl;
      exit(1);
    } else {
      local_vertices_.insert(global_id);
      // Ensure that vertex is either ghost or local not both
      if (IsGhost(global_id)) {
        ghost_vertices_.erase(global_id);
      }
    }
    return number_of_local_vertices_++;
  }

  inline VertexID AddGhostVertex(VertexID global_id, PEID rank) {
    if (IsLocal(global_id)) {
      std::cout << "R" << rank_ << " This shouldn't happen (added ghost)" << std::endl;
      exit(1);
    }
    ghost_vertices_[global_id] = rank;
    return number_of_local_vertices_;
  }

  EdgeID AddEdge(VertexID global_from, VertexID global_to) {
    if (IsLocal(global_from)) {
      AddLocalEdge(global_from, global_to);
    } else {
      std::cout << "R" << rank_ << " This shouldn't happen (added edge)" << std::endl;
      exit(1);
    }
    return number_of_local_edges_++;
  }

  void RelinkEdge(VertexID global_from, VertexID global_to, VertexID new_global_to) {
    for (VertexID i = 0; i < adjacent_edges_[global_from].size(); ++i) {
      if (adjacent_edges_[global_from][i].target_ == global_to) {
        adjacent_edges_[global_from][i].target_ = new_global_to;
      }
    }
  }

  void AddLocalEdge(VertexID global_from, VertexID global_to) {
    adjacent_edges_[global_from].emplace_back(global_to);
  }

  inline VertexID GetVertexDegree(const VertexID global_v) {
    return adjacent_edges_[global_v].size();
  }

  inline PEID GetPE(VertexID global_id) {
    if (!IsGhost(global_id)) {
      std::cout << "R" << rank_ << " This shouldn't happen (added edge)" << std::endl;
      exit(1);
    }
    return ghost_vertices_[global_id];
  }

  inline void SetPE(VertexID global_id, PEID rank) {
    if (!IsGhost(global_id)) {
      std::cout << "R" << rank_ << " This shouldn't happen (added edge)" << std::endl;
      exit(1);
    }
    ghost_vertices_[global_id] = rank;
  }

  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  void OutputLocal() {
    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [V] " << v;
      std::cout << out.str() << std::endl;
    });

    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [N] "
          << v << " -> ";
      ForallNeighbors(v, [&](VertexID u) {
        out << u << " ";
      });
      std::cout << out.str() << std::endl;
    });
  }

  void Logging(bool active);

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  google::dense_hash_set<VertexID> local_vertices_;
  google::dense_hash_map<VertexID, PEID> ghost_vertices_;
  google::dense_hash_map<VertexID, std::vector<Edge>> adjacent_edges_;

  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_local_edges_;
  EdgeID number_of_global_edges_;
};

#endif
