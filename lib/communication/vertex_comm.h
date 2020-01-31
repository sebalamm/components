/******************************************************************************
 * ghost_communicator.h
 *
 * Communication patterns for ghost vertices in distributed graph.
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

#ifndef _VERTEX_COMMUNICATOR_H_
#define _VERTEX_COMMUNICATOR_H_

#include <memory>

#include "definitions.h"
#include "config.h"
#include "dynamic_graph_comm.h"
#include "semidynamic_graph_comm.h"
#include "static_graph_comm.h"
#include "comm_utils.h"

template<typename GraphType>
class VertexCommunicator {
 public:
  VertexCommunicator(const PEID rank,
                     const PEID size)
      : g_(nullptr),
        use_sampling_(false),
        rank_(rank),
        size_(size),
        comm_time_(0.0) {
    packed_pes_.set_empty_key(-1);
    packed_pes_.set_deleted_key(-1);
    send_buffers_.set_empty_key(-1);
    send_buffers_.set_deleted_key(-1);
    receive_buffers_.set_empty_key(-1);
    receive_buffers_.set_deleted_key(-1);
    neighborhood_sample_.set_empty_key(-1);
    neighborhood_sample_.set_deleted_key(-1);
    message_tag_ = static_cast<unsigned int>(100 * size_);
  }
  virtual ~VertexCommunicator() {};

  VertexCommunicator(const VertexCommunicator &rhs) = default;
  VertexCommunicator(VertexCommunicator &&rhs) = default;
 
  inline void SetGraph(GraphType *g) {
    g_ = g;
  }

  inline bool IsPackedPE(const PEID pe) const {
    return packed_pes_.find(pe) != packed_pes_.end();
  }

  void SetPackedPE(const PEID pe, const bool is_packed) {
    if (pe == rank_) return;
    if (is_packed) {
      if (IsPackedPE(pe)) return;
      else packed_pes_.insert(pe);
    } else {
      if (!IsPackedPE(pe)) return;
      else packed_pes_.erase(pe);
    }
  }

  void AddMessage(VertexID v, const VertexPayload &msg);

  void SampleVertexNeighborhood(const VertexID &v, const float sampling_factor);

  void UpdateGhostVertices();

  void SendAndReceiveGhostVertices() {
    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers_, receive_buffers_, 
                                         rank_, size_, message_tag_++);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers_);
    UpdateGhostVertices();
    CommunicationUtility::ClearBuffers(receive_buffers_);
  }

  float GetCommTime() {
    return comm_time_;
  }


 private:
  GraphType *g_;

  PEID rank_, size_;

  google::dense_hash_set<PEID> packed_pes_;
  google::dense_hash_map<PEID, VertexBuffer> send_buffers_;
  google::dense_hash_map<PEID, VertexBuffer> receive_buffers_;

  // Neighborhood sampling
  bool use_sampling_;
  google::dense_hash_map<VertexID, google::sparse_hash_set<VertexID>> neighborhood_sample_;

  VertexID message_tag_;

  float comm_time_;
  Timer comm_timer_;

  void PlaceInBuffer(const PEID &pe,
                     const VertexID &v,
                     const VertexPayload &msg);
};

template class VertexCommunicator<DynamicGraphCommunicator>;
template class VertexCommunicator<SemidynamicGraphCommunicator>;
template class VertexCommunicator<StaticGraphCommunicator>;

#endif
