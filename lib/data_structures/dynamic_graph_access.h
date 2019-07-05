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

#ifndef _DYNAMIC_GRAPH_ACCESS_
#define _DYNAMIC_GRAPH_ACCESS_

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
#include "static_graph_access.h"

struct VertexPayload {
  VertexID deviate_;
  VertexID label_;
#ifdef TIEBREAK_DEGREE
  VertexID degree_;
#endif
  PEID root_;

  VertexPayload()
      : deviate_(std::numeric_limits<VertexID>::max() - 1),
        label_(0),
#ifdef TIEBREAK_DEGREE
        degree_(0),
#endif
        root_(0) {}

  VertexPayload(VertexID deviate,
                VertexID label,
#ifdef TIEBREAK_DEGREE
                VertexID degree,
#endif
                PEID root)
      : deviate_(deviate),
        label_(label),
#ifdef TIEBREAK_DEGREE
        degree_(degree),
#endif
        root_(root) {}

  bool operator==(const VertexPayload &rhs) const {
    return std::tie(deviate_, 
                    label_, 
#ifdef TIEBREAK_DEGREE
                    rhs.degree_,
#endif
                    root_)
        == std::tie(rhs.deviate_, 
                    rhs.label_, 
#ifdef TIEBREAK_DEGREE
                    rhs.degree_,
#endif
                    rhs.root_);
  }

  bool operator!=(const VertexPayload &rhs) const {
    return !(*this == rhs);
  }
};

class VertexCommunicator;
class DynamicGraphAccess {
 public:
  DynamicGraphAccess(const PEID rank, const PEID size);
  virtual ~DynamicGraphAccess();

  DynamicGraphAccess(DynamicGraphAccess &&rhs) = default;

  DynamicGraphAccess(const DynamicGraphAccess &rhs) = default;

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
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
    for (VertexID v = GetNumberOfLocalVertices(); v < GetNumberOfVertices(); ++v) {
      if (IsActive(v)) callback(v);
    }
  }

  template<typename F>
  void ForallVertices(F &&callback) {
    for (VertexID v = 0; v < GetNumberOfVertices(); ++v) {
      if (IsActive(v)) callback(v);
    }
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

  bool IsAdjacent(const VertexID source, const VertexID target) {
    bool adj = false;
    ForallNeighbors(source, [&](const VertexID w) {
      if (w == target) adj = true;
    });
    return adj;
  }

  inline bool IsActive(const VertexID v) const {
    return is_active_[v];
  }

  void SetActive(VertexID v, bool is_active) {
    is_active_[v] = is_active;
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

  // void ContractLocal() {
  //   // Statistics
  //   Timer contract_timer;
  //   contract_timer.Restart();
  //   
  //   // Determine edges to communicate
  //   // Gather labels to communicate
  //   VertexID num_global_vertices = GatherNumberOfGlobalVertices();
  //   google::dense_hash_set<VertexID> send_ids(size_); send_ids.set_empty_key(-1);
  //   std::vector<std::vector<VertexID>> send_buffers_a(size_);
  //   std::vector<std::vector<VertexID>> send_buffers_b(size_);
  //   std::vector<std::vector<VertexID>>* current_send_buffers = &send_buffers_a;
  //   std::vector<std::vector<VertexID>> receive_buffers(size_);
  //   for (int i = 0; i < size_; ++i) {
  //     send_buffers_a[i].clear();
  //     send_buffers_a[i].reserve(number_of_vertices_);
  //     send_buffers_b[i].clear();
  //     send_buffers_b[i].reserve(number_of_vertices_);
  //     receive_buffers[i].clear();
  //     receive_buffers[i].reserve(number_of_vertices_);
  //   }

  //   // std::unordered_set<VertexID> inserted_edges;
  //   google::dense_hash_set<VertexID> inserted_edges; inserted_edges.set_empty_key(-1);
  //   std::vector<std::vector<std::pair<VertexID, VertexID>>> edges_to_add(size_);
  //   ForallLocalVertices([&](VertexID v) {
  //     VertexID vlabel = GetVertexLabel(v);
  //     ForallNeighbors(v, [&](VertexID w) {
  //       VertexID wlabel = GetVertexLabel(w);
  //       if (vlabel != wlabel) {
  //         VertexID update_id = vlabel + num_global_vertices * wlabel;
  //         if (send_ids.find(update_id) == send_ids.end()) {
  //           // Local propagation
  //           VertexID parent = GetParent(v);
  //           PEID pe = GetPE(GetLocalID(parent));
  //           // Send edge
  //           send_ids.insert(update_id);
  //           (*current_send_buffers)[pe].emplace_back(vlabel);
  //           (*current_send_buffers)[pe].emplace_back(wlabel);
  //           (*current_send_buffers)[pe].emplace_back(GetVertexRoot(w));
  //           (*current_send_buffers)[pe].emplace_back(parent);
  //         }
  //       }
  //       removed_edges_.emplace(v, GetGlobalID(w));
  //     });
  //   });
  //   removed_edges_.emplace(std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max());

  //   // Get adjacency (otherwise we get deadlocks with added edges)
  //   std::vector<bool> is_adj(size_);
  //   PEID num_adj = 0;
  //   for (PEID pe = 0; pe < size_; pe++) {
  //     is_adj[pe] = IsAdjacentPE(pe);
  //     if (is_adj[pe]) num_adj++;
  //   }

  //   if (rank_ == ROOT) 
  //     std::cout << "[STATUS] |--- Send done " 
  //               << "[TIME] " << contract_timer.Elapsed() << std::endl;

  //   // Propagate edge buffers until all vertices are converged
  //   std::vector<MPI_Request*> requests;
  //   requests.clear();
  //   VertexID iteration = 0;
  //   int converged_globally = 0;
  //   while (converged_globally == 0) {
  //     int converged_locally = 1;

  //     // Send edges
  //     for (PEID pe = 0; pe < size_; ++pe) {
  //       if (is_adj[pe] && pe != rank_) {
  //         if ((*current_send_buffers)[pe].size() == 0) 
  //           (*current_send_buffers)[pe].emplace_back(0);
  //         auto *req = new MPI_Request();
  //         MPI_Isend((*current_send_buffers)[pe].data(), static_cast<int>((*current_send_buffers)[pe].size()), MPI_VERTEX, pe, 0, MPI_COMM_WORLD, req);
  //         requests.emplace_back(req);
  //       }
  //     }

  //     // Receive edges
  //     PEID messages_recv = 0;
  //     int message_length = 0;
  //     for (int i = 0; i < size_; ++i) receive_buffers[i].clear();
  //     for (int i = 0; i < (*current_send_buffers)[rank_].size(); i++) {
  //       receive_buffers[rank_].emplace_back((*current_send_buffers)[rank_][i]);
  //     }
  //     while (messages_recv < num_adj) {
  //       MPI_Status st{};
  //       MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
  //       MPI_Get_count(&st, MPI_VERTEX, &message_length);
  //       messages_recv++;

  //       receive_buffers[st.MPI_SOURCE].resize(message_length);
  //       MPI_Status rst{};
  //       MPI_Recv(&receive_buffers[st.MPI_SOURCE][0], message_length, MPI_VERTEX, st.MPI_SOURCE, 0, MPI_COMM_WORLD, &rst);
  //     }

  //     for (unsigned int i = 0; i < requests.size(); ++i) {
  //       MPI_Status st;
  //       MPI_Wait(requests[i], &st);
  //       delete requests[i];
  //     }
  //     requests.clear();

  //     // Clear buffers
  //     if (current_send_buffers == &send_buffers_a) {
  //       for (int i = 0; i < size_; ++i) send_buffers_b[i].clear();
  //       current_send_buffers = &send_buffers_b;
  //     } else {
  //       for (int i = 0; i < size_; ++i) send_buffers_a[i].clear();
  //       current_send_buffers = &send_buffers_a;
  //     }

  //     // Receive edges and apply updates
  //     for (PEID pe = 0; pe < size_; ++pe) {
  //       // if (pe == rank_ && rank_ == 0) std::cout << "recv size " << receive_buffers[pe].size() << std::endl;
  //       if (receive_buffers[pe].size() < 4) continue;
  //       for (int i = 0; i < receive_buffers[pe].size(); i += 4) {
  //         VertexID vlabel = receive_buffers[pe][i];
  //         VertexID wlabel = receive_buffers[pe][i + 1];
  //         VertexID wroot = receive_buffers[pe][i + 2];
  //         VertexID link = receive_buffers[pe][i + 3];
  //         VertexID update_id = vlabel + num_global_vertices * wlabel;

  //         // Continue if already inserted
  //         if (inserted_edges.find(update_id) != end(inserted_edges)) continue;
  //         // if (send_ids.find(update_id) != end(send_ids) && pe != rank_) continue;

  //         // Get link information
  //         VertexID parent = GetParent(GetLocalID(link));
  //         PEID pe = GetPE(GetLocalID(parent));

  //         if (IsLocalFromGlobal(vlabel)) {
  //           edges_to_add[wroot].emplace_back(vlabel, wlabel);
  //           inserted_edges.insert(update_id);
  //           send_ids.insert(update_id);
  //           if (wroot == rank_) {
  //             edges_to_add[rank_].emplace_back(wlabel, vlabel);
  //             inserted_edges.insert(wlabel + num_global_vertices * vlabel);
  //             send_ids.insert(wlabel + num_global_vertices * vlabel);
  //           }
  //         } else {
  //           if (GetVertexLabel(GetLocalID(link)) == vlabel) {
  //             send_ids.insert(update_id);
  //             (*current_send_buffers)[pe].emplace_back(vlabel);
  //             (*current_send_buffers)[pe].emplace_back(wlabel);
  //             (*current_send_buffers)[pe].emplace_back(wroot);
  //             (*current_send_buffers)[pe].emplace_back(parent);
  //             converged_locally = 0;
  //           } else {
  //             // Parent has to be connected to vlabel (N(N(v))
  //             VertexID local_vlabel = GetLocalID(vlabel);
  //             pe = GetPE(local_vlabel);

  //             // Send edge
  //             send_ids.insert(update_id);
  //             (*current_send_buffers)[pe].emplace_back(vlabel);
  //             (*current_send_buffers)[pe].emplace_back(wlabel);
  //             (*current_send_buffers)[pe].emplace_back(wroot);
  //             (*current_send_buffers)[pe].emplace_back(vlabel);
  //             converged_locally = 0;
  //           }
  //         }
  //       }
  //     }

  //     // Check if all PEs are done
  //     MPI_Allreduce(&converged_locally,
  //                   &converged_globally,
  //                   1,
  //                   MPI_INT,
  //                   MPI_MIN,
  //                   MPI_COMM_WORLD);
  //   }
  //   if (rank_ == ROOT) 
  //   std::cout << "[STATUS] |--- Propagation done " 
  //             << "[TIME] " << contract_timer.Elapsed() << std::endl;

  //   // Insert edges and keep corresponding vertices
  //   ForallLocalVertices([&](VertexID v) { RemoveAllEdges(v); });
  //   contraction_level_++;
  //   for (VertexID i = 0; i < inactive_level_.size(); ++i) {
  //     if (inactive_level_[i] == -1) inactive_level_[i] = contraction_level_ - 1;
  //   }
  //   for (PEID pe = 0; pe < size_; pe++) {
  //     for (auto &e : edges_to_add[pe]) {
  //       VertexID vlabel = e.first;
  //       VertexID wlabel = e.second;
  //       VertexID vlocal = GetLocalID(vlabel);
  //       // Add edge
  //       AddEdge(vlocal, wlabel, pe);
  //       VertexID wlocal = GetLocalID(wlabel);
  //       // Vertices remain active
  //       inactive_level_[vlocal] = -1;
  //       inactive_level_[wlocal] = -1;
  //       vertex_payload_[vlocal] = {std::numeric_limits<VertexID>::max() - 1, 
  //                                  vlabel, 
  // #ifdef TIEBREAK_DEGREE
  //                                  0,
  // #endif
  //                                  rank_};
  //       vertex_payload_[wlocal] = {std::numeric_limits<VertexID>::max() - 1, 
  //                                  wlabel, 
  //#ifdef TIEBREAK_DEGREE
  //                                  0,
  // #endif
  //                                  pe};
  //     }
  //   }
  //   if (rank_ == ROOT) 
  //     std::cout << "[STATUS] |--- Insertion done " 
  //               << "[TIME] " << contract_timer.Elapsed() << std::endl;
  // }

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

  void SetVertexPayload(VertexID v, VertexPayload &&msg, bool propagate = true);

  void ForceVertexPayload(VertexID v, VertexPayload &&msg);

  inline VertexPayload &GetVertexMessage(const VertexID v) {
    return vertex_payload_[v];
  }

  void SetVertexMessage(const VertexID v, VertexPayload &&msg) {
    vertex_payload_[v] = msg;
  }

  void SetParent(const VertexID v, const VertexID parent_v) {
    parent_[v] = parent_v;
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

  inline VertexID GetParent(const VertexID v) {
    return parent_[v];
  }

  inline VertexID AddVertex() {
    return vertex_counter_++;
  }

  VertexID  AddGhostVertex(VertexID v);

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank);

  void AddLocalEdge(VertexID from, VertexID to);

  void AddGhostEdge(VertexID from, VertexID to);

  void ReserveEdgesForVertex(VertexID v, VertexID num_edges);

  void RemoveAllEdges(VertexID from);

  // Local IDs
  bool IsConnected(VertexID from, VertexID to) {
    ForallNeighbors(from, [&](VertexID v) {
        if (v == to) return true; 
    });
    return false;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return adjacent_edges_[v].size();
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
    // TODO [MEMORY]: Might be too small
    int num_local_vertices = 0;
    // TODO [MEMORY]: Might be too small
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
    // TODO [MEMORY]: Might be too small
    int num_global_vertices = 0;
    // TODO [MEMORY]: Might be too small
    int num_global_edges = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_vertices[i] = num_global_vertices;
      displ_edges[i] = num_global_edges;
      num_global_vertices += num_vertices[i];
      num_global_edges += num_edges[i];
    }

    // Build datatype for edge
    MPI_Datatype MPI_EDGE;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);
    
    // Gather vertices/edges and labels for each PE
    global_vertices.resize(num_global_vertices);
    global_labels.resize(num_global_vertices);
    global_edges.resize(num_global_edges);
    MPI_Gatherv(&local_vertices[0], num_local_vertices, MPI_VERTEX,
                &global_vertices[0], &num_vertices[0], &displ_vertices[0], MPI_VERTEX,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_labels[0], num_local_vertices, MPI_VERTEX,
                &global_labels[0], &num_vertices[0], &displ_vertices[0], MPI_VERTEX,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_edges[0], num_local_edges, MPI_EDGE,
                &global_edges[0], &num_edges[0], &displ_edges[0], MPI_EDGE,
                ROOT, MPI_COMM_WORLD);
  } 

  void DistributeLabelsFromRoot(std::vector<VertexID> &global_labels, 
                                std::vector<int> &num_labels) {
    // Compute displacements
    std::vector<int> displ_labels(size_);
    // TODO [MEMORY]: Might be too small
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
    MPI_Scatterv(&global_labels[0], &num_labels[0], &displ_labels[0], MPI_VERTEX, 
                 &local_labels[0], num_local_vertices, MPI_VERTEX, 
                 ROOT, MPI_COMM_WORLD);

    for (int i = 0; i < num_local_vertices; ++i) {
      VertexID v = local_vertices[i];
      SetVertexPayload(v, {GetVertexDeviate(v), 
                           local_labels[i], 
#ifdef TIEBREAK_DEGREE
                           0,
#endif
                           GetVertexRoot(v)});
    }
  }

  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  bool CheckDuplicates();

  void OutputLocal();

  void OutputLabels();

  void OutputGhosts();

  void OutputComponents(std::vector<VertexID> &labels);

  void Logging(bool active);

 private:
  // Network information
  PEID rank_, size_;

  // Vertices and edges
  std::vector<std::vector<Edge>> adjacent_edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;
  std::vector<VertexPayload> vertex_payload_;

  // Shortcutting
  std::vector<VertexID> parent_;
  google::dense_hash_map<VertexID, VertexID> label_shortcut_;

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
  std::vector<bool> is_active_;

  // Adjacent PEs
  std::vector<bool> adjacent_pes_;

  // Communication interface
  VertexCommunicator *ghost_comm_;

  // Temporary counters
  VertexID vertex_counter_;
  VertexID ghost_counter_;
  EdgeID edge_counter_;
};

#endif
