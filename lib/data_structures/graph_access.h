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

#include <unordered_map>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <sstream>
#include <deque>
#include <tuple>

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

class BlockingCommunicator;
class GraphAccess {
 public:
  GraphAccess(const PEID rank, const PEID size) 
      : rank_(rank),
        size_(size),
        number_of_vertices_(0),
        number_of_local_vertices_(0),
        number_of_edges_(0),
        local_offset_(0),
        ghost_offset_(0),
        contraction_level_(0),
        max_contraction_level_(0),
        ghost_comm_(nullptr),
        vertex_counter_(0),
        edge_counter_(0) {}
  virtual ~GraphAccess() = default;

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
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { callback(edges_[v][e].target_); });
  }

  template<typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = 0; e < GetVertexDegree(v); ++e) {
      callback(e);
    }
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
    active_vertices_.resize(contraction_level_ + 2);
    active_vertices_[contraction_level_ + 1].resize(number_of_local_vertices_, false);
    vertex_payload_.resize(contraction_level_ + 2);
    vertex_payload_[contraction_level_ + 1].resize(number_of_vertices_);

    // Parent information
    parent_.resize(contraction_level_ + 2);
    parent_[contraction_level_ + 1].resize(number_of_vertices_);

    // Update stacks
    added_edges_.resize(contraction_level_ + 2);
    removed_edges_.resize(contraction_level_ + 2);

    // Update data type
    MPI_Datatype MPI_UPDATE;
    MPI_Type_vector(1, 3, 0, MPI_LONG, &MPI_UPDATE);
    MPI_Type_commit(&MPI_UPDATE);

    // Determine remaining active vertices
    std::vector<std::tuple<VertexID, VertexID>> edges_to_remove;
    ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = GetVertexLabel(v);
      ForallAdjacentEdges(v, [&](EdgeID e) {
        VertexID w = edges_[v][e].target_;
        VertexID wlabel = GetVertexLabel(w);
        // Edge needs to be linked to root 
        if (vlabel != wlabel) {
          std::vector<VertexID> update = {vlabel, wlabel, GetVertexRoot(w)};
          if (logging_) {
            std::cout << "[R"  << rank_ << ":" << contraction_level_
                      << "] [Link] send edge (" << vlabel << "," << wlabel << ") to "
                      << GetVertexRoot(v) << std::endl;
          }
          auto *request = new MPI_Request();
          MPI_Isend(&update[0], 1, MPI_UPDATE, GetVertexRoot(v), 0, MPI_COMM_WORLD, request);
        }
        // Edge can be removed 
        // Store removed edges for resolution
        edges_to_remove.emplace_back(v, w);
      });
      vertex_payload_[contraction_level_ + 1][v] = {std::numeric_limits<VertexID>::max() - 1, GetVertexLabel(v), rank_};
    });

    // Increase contraction level
    contraction_level_++;
    max_contraction_level_++;

    // Remove inactive edges
    for (auto &e : edges_to_remove) {
      RemoveEdge(std::get<0>(e), std::get<1>(e));
      removed_edges_[contraction_level_].emplace_back(GetGlobalID(std::get<0>(e)), GetGlobalID(std::get<1>(e)));
    }

    // Gather edge updates
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status st{};
    int flag = 1;
    do {
      MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_UPDATE, &message_length);
        std::vector<VertexID> update(3);
        MPI_Status rst{};
        MPI_Recv(&update[0], message_length, MPI_UPDATE, st.MPI_SOURCE, 0, MPI_COMM_WORLD, &rst);
        if (logging_) {
          std::cout << "[R"  << rank_ << ":" << contraction_level_
                    << "] [Link] recv edge (" << update[0] << "," << update[1] 
                    << "(R" << update[2] << ")) from " << st.MPI_SOURCE << std::endl;
        }
        // Insert actual edge
        AddEdge(GetLocalID(update[0]), update[1], update[2]);
        added_edges_[contraction_level_].emplace_back(update[0], update[1]);
        active_vertices_[contraction_level_][GetLocalID(update[0])] = true;
      }
    } while (flag);

  }

  void MoveUpContraction() {
    while (contraction_level_ > 0) {
      contraction_level_--;

      for (auto &e : added_edges_[contraction_level_ + 1]) 
        RemoveEdge(GetLocalID(std::get<0>(e)), GetLocalID(std::get<1>(e)));
      for (auto &e : removed_edges_[contraction_level_ + 1]) 
        AddEdge(GetLocalID(std::get<0>(e)), std::get<1>(e), GetPE(GetLocalID(std::get<1>(e))));

      // Update local labels
      ForallLocalVertices([&](VertexID v) {
        if (vertex_payload_[contraction_level_][v].label_ != 
              vertex_payload_[contraction_level_ + 1][v].label_) 
          SetVertexPayload(v, {0, vertex_payload_[contraction_level_ + 1][v].label_, rank_});
      });

      // Propagate labels
      int converged_globally = 0;
      while (converged_globally == 0) {
        int converged_locally = 1;
        // Receive variates
        UpdateGhostVertices();

        // Send current label from root
        ForallLocalVertices([&](VertexID v) {
          if (GetVertexLabel(GetParent(v)) != GetVertexLabel(v)) {
            SetVertexPayload(v, {0, GetVertexLabel(GetParent(v)), rank_});
            converged_locally = false;
          }
        });

        // Check if all PEs are done
        MPI_Allreduce(&converged_locally, &converged_globally, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      }
    }
    UpdateGhostVertices();
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

  inline VertexID GetNumberOfLocalVertices() const {
    return number_of_local_vertices_;
  }

  inline EdgeID GetNumberOfEdges() const { return number_of_edges_; }

  void SetVertexPayload(VertexID v, const VertexPayload &msg);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return vertex_payload_[contraction_level_][v];
  }

  void SetVertexMessage(const VertexID v, const VertexPayload &msg) {
    vertex_payload_[contraction_level_][v] = msg;
  }

  void SetParent(const VertexID v, const VertexID parent) {
    parent_[contraction_level_][v] = parent;
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

  inline PEID GetParent(const VertexID v) const {
    return parent_[contraction_level_][v];
  }

  inline VertexID AddVertex() {
    return vertex_counter_++;
  }

  inline void RemoveVertex() {
  }

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank);

  void RemoveEdge(VertexID from, VertexID to);

  inline VertexID GetVertexDegree(const VertexID v) const {
    return edges_[v].size();
  }

  //////////////////////////////////////////////
  // Manage ghost vertices
  //////////////////////////////////////////////
  void UpdateGhostVertices();

  inline VertexID NumberOfGhostVertices() const {
    return number_of_vertices_ - number_of_local_vertices_ - 1;
  }

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
  // I/O
  //////////////////////////////////////////////
  void OutputLocal();

  void OutputLabels();

  void Logging(bool active);

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<std::vector<Edge>> edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;
  std::vector<std::vector<VertexPayload>> vertex_payload_;
  std::vector<std::vector<VertexID>> parent_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;

  EdgeID number_of_edges_;

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

  // Adjacent PEs
  std::vector<bool> adjacent_pes_;

  // Communication interface
  BlockingCommunicator *ghost_comm_;

  // Temporary counters
  VertexID vertex_counter_;
  EdgeID edge_counter_;

  // Logging
  bool logging_;
};

#endif
