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

#include "dynamic_graph.h"
#include "config.h"
#include "timer.h"
#include "payload.h"

template<typename GraphType> 
class VertexCommunicator;

class DynamicGraphCommunicator : public DynamicGraph {
 public:
  DynamicGraphCommunicator(const PEID rank, const PEID size);

  virtual ~DynamicGraphCommunicator();

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void ResetCommunicator();

  //////////////////////////////////////////////
  // Graph contraction
  //////////////////////////////////////////////
  void BuildLabelShortcuts() {
    // Gather labels
    google::dense_hash_set<VertexID> labels; 
    labels.set_empty_key(-1);
    ForallLocalVertices([&](const VertexID v) {
        labels.insert(GetVertexLabel(v));
    });

    // Init shortcuts
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
  // Manage local vertices/edges
  //////////////////////////////////////////////
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

  VertexID AddGhostVertex(VertexID v, PEID pe);

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
  void SetAdjacentPE(const PEID pe, const bool is_adj);

  void ResetAdjacentPEs();


  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  void OutputLocal();

  void OutputLabels();

  void Logging(bool active);

  float GetCommTime();

 private:
  // Payloads
  std::vector<VertexPayload> local_payload_;
  std::vector<VertexPayload> ghost_payload_;

  // Communication interface
  VertexCommunicator<DynamicGraphCommunicator> *ghost_comm_;
};

#endif
