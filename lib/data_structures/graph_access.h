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
#include <unordered_set>
#include <boost/functional/hash.hpp>

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
        local_offset_(0),
        ghost_offset_(0),
        contraction_level_(0),
        max_contraction_level_(0),
        edge_buffers_(size),
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
  void ForallGhostVertices(F &&callback) {
    for (VertexID v = GetNumberOfLocalVertices(); v < GetNumberOfVertices(); ++v) {
      if (active_vertices_[contraction_level_][v]) callback(v);
    }
  }

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
    std::vector<std::unordered_set<VertexID>> send_ids(size_);
    for (PEID i = 0; i < size_; ++i) edge_buffers_[i].clear();

    // Gather remaining edges and reset vertex payloads
    ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = GetVertexLabel(v);
      ForallAdjacentEdges(v, [&](EdgeID e) {
        VertexID w = edges_[v][e].target_;
        VertexID wlabel = GetVertexLabel(w);
        // Edge needs to be linked to root 
        if (vlabel != wlabel) {
          PEID pe = GetVertexRoot(v);
          VertexID update_id = vlabel + GetNumberOfLocalVertices() * wlabel;
          if (send_ids[pe].find(update_id) == send_ids[pe].end()) {
            send_ids[pe].insert(update_id);
            // TODO: Encode edges to reduce volume
            edge_buffers_[pe].push_back(vlabel);
            edge_buffers_[pe].push_back(wlabel);
            edge_buffers_[pe].push_back(GetVertexRoot(w));
#ifndef NDEBUG
            std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_
                      << "] [Link] send edge (" << vlabel << "," << wlabel
                      << "(R" << GetVertexRoot(w) << ")) to " << pe
                      << std::endl;
#endif
          }
        } 
        removed_edges_[contraction_level_].emplace_back(v, GetGlobalID(w));
      });
      RemoveAllEdges(v);
      vertex_payload_[contraction_level_ + 1][v] =
          {std::numeric_limits<VertexID>::max() - 1, GetVertexLabel(v), rank_};
    });

    // Send gathered edges
    for (PEID i = 0; i < size_; ++i) {
      if (!IsAdjacentPE(i) || i == rank_) continue;
      if (!(edge_buffers_[i].size() > 0)) continue;
      MPI_Request request;
      MPI_Isend(&edge_buffers_[i][0], edge_buffers_[i].size(), MPI_LONG, i, 0,
                MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
    }

    // Increase contraction level
    contraction_level_++;
    max_contraction_level_++;

    // Gather edge updates
    std::unordered_set<VertexID> inserted_edges;

    // Local updates
    if (edge_buffers_[rank_].size() > 0) {
      for (int i = 0; i < edge_buffers_[rank_].size() - 1; i += 3) {
        VertexID source = GetLocalID(edge_buffers_[rank_][i]);
        VertexID target = edge_buffers_[rank_][i+1];
        PEID target_pe = static_cast<PEID>(edge_buffers_[rank_][i+2]);
        VertexID edge_id = source + target * GetNumberOfLocalVertices();
        if (inserted_edges.find(edge_id) == inserted_edges.end()) {
          inserted_edges.insert(edge_id);
          active_vertices_[contraction_level_][source] = true;
          AddEdge(source, target, target_pe);
          added_edges_[contraction_level_ - 1].emplace_back(source, target);
#ifndef NDEBUG
          std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_ - 1
                    << "] [Link] recv edge (" << GetGlobalID(source) << "," << target
                    << "(R" << target_pe << ")) from " << rank_
                    << std::endl;
#endif
        }
      }
    }

    // Non-local updates
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status st{};
    int flag = 1;
    do {
      MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_LONG, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0],
                 message_length,
                 MPI_LONG,
                 st.MPI_SOURCE,
                 0,
                 MPI_COMM_WORLD,
                 &rst);
        if (message_length == 1) continue;

        // Insert edges
        for (int i = 0; i < message_length - 1; i += 3) {
          VertexID source = GetLocalID(message[i]);
          VertexID target = message[i+1];
          PEID target_pe = static_cast<PEID>(message[i+2]);
          VertexID edge_id = source + target * GetNumberOfLocalVertices();
          if (inserted_edges.find(edge_id) == inserted_edges.end()) {
            inserted_edges.insert(edge_id);
            active_vertices_[contraction_level_][source] = true;
            AddEdge(source, target, target_pe);
            added_edges_[contraction_level_ - 1].emplace_back(source, target);
#ifndef NDEBUG
            std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_ - 1
                      << "] [Link] recv edge (" << GetGlobalID(source) << "," << target
                      << "(R" << target_pe << ")) from " << st.MPI_SOURCE
                      << std::endl;
#endif
          }
        }
      }
    } while (flag);
  }

  void MoveUpContraction() {
    while (contraction_level_ > 0) {
      // Remove current edges from current level
      ForallLocalVertices([&](VertexID v) {
          RemoveAllEdges(v);
      });

      // Decrease level
      contraction_level_--;

      // Add previously removed edges
      for (auto &e : removed_edges_[contraction_level_])
        AddEdge(std::get<0>(e), std::get<1>(e), GetPE(GetLocalID(std::get<1>(e))));

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
          if (GetVertexLabel(GetParent(v)) != GetVertexLabel(v)) {
            SetVertexPayload(v, {0, GetVertexLabel(GetParent(v)), rank_});
            converged_locally = false;
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

  inline VertexID GetLocalOffset() const {
    return local_offset_;
  }

  inline VertexID GetNumberOfLocalVertices() const {
    return number_of_local_vertices_;
  }

  inline VertexID GetNumberOfGhostVertices() const { return number_of_vertices_ - number_of_local_vertices_; }

  inline EdgeID GetNumberOfEdges() const { return number_of_edges_; }

  inline VertexID GatherNumberOfGlobalVertices() {
    VertexID local_vertices = 0;
    ForallLocalVertices([&](VertexID v) { local_vertices++; });
    // Check if all PEs are done
    MPI_Allreduce(&local_vertices,
                  &number_of_global_vertices_,
                  1,
                  MPI_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return number_of_global_vertices_;
  }


  void SetVertexPayload(VertexID v, VertexPayload &&msg, bool propagate = true);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return vertex_payload_[contraction_level_][v];
  }

  void SetVertexMessage(const VertexID v, VertexPayload &&msg) {
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

  inline VertexID GetParent(const VertexID v) const {
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
  std::vector<std::vector<VertexID>> parent_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

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
