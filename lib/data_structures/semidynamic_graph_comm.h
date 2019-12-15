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

#ifndef _SEMIDYNAMIC_GRAPH_COMM_H_
#define _SEMIDYNAMIC_GRAPH_COMM_H_

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

#include "semidynamic_graph.h"
#include "config.h"
#include "timer.h"
#include "payload.h"

template<typename GraphType> 
class VertexCommunicator;

class SemidynamicGraphCommunicator : public SemidynamicGraph {
 public:
  SemidynamicGraphCommunicator(const PEID rank, const PEID size);

  virtual ~SemidynamicGraphCommunicator();

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void ResetCommunicator();
  
  void StartConstruct(VertexID local_n, VertexID ghost_n, VertexID local_offset);

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
  // Manage local vertices/edges
  //////////////////////////////////////////////
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

  VertexID AddGhostVertex(VertexID v);

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
  std::vector<VertexPayload> vertex_payload_;

  VertexID max_degree_;
  bool max_degree_computed_;

  // Communication interface
  VertexCommunicator<SemidynamicGraphCommunicator> *ghost_comm_;
};

#endif
