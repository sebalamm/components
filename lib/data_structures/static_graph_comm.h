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

#include "static_graph.h"
#include "config.h"
#include "timer.h"
#include "payload.h"

template<typename GraphType> 
class VertexCommunicator;

class StaticGraphCommunicator : public StaticGraph {
 public:
  StaticGraphCommunicator(const Config& conf, const PEID rank, const PEID size);

  virtual ~StaticGraphCommunicator();

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void ResetCommunicator();

  //////////////////////////////////////////////
  // Manage local vertices/edges
  //////////////////////////////////////////////
  VertexID AddGhostVertex(VertexID v, PEID pe);

  void SampleVertexNeighborhood(const VertexID &v, const float sampling_factor);

  void AllocatePayloads() {
    vertex_payload_.resize(vertex_counter_);
  }

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
    std::string deviate = GetVertexDeviate(v) == MaxDeviate ? "-" : std::to_string(GetVertexDeviate(v));
    out << "(" << deviate << ","
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

  //////////////////////////////////////////////
  // Manage ghost vertices
  //////////////////////////////////////////////
  void SendAndReceiveGhostVertices();

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
  // I/O
  //////////////////////////////////////////////
  void OutputLocal();

  void OutputLabels();

  void Logging(bool active);

  float GetCommTime();

  VertexID GetSendVolume();

  VertexID GetReceiveVolume();

 private:
  std::vector<VertexPayload> vertex_payload_;

  // Communication interface
  VertexCommunicator<StaticGraphCommunicator> *ghost_comm_;
};

#endif
