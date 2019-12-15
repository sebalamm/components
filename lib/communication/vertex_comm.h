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

#include <unordered_map>
#include <vector>
#include <memory>

#include "config.h"
#include "dynamic_graph_comm.h"
#include "semidynamic_graph_comm.h"
#include "static_graph_comm.h"

using Buffer = std::vector<VertexID>;

template<typename GraphType>
class VertexCommunicator {
 public:
  VertexCommunicator(const PEID rank,
                   const PEID size,
                   MPI_Comm communicator)
      : communicator_(communicator),
        g_(nullptr),
        rank_(rank),
        size_(size),
        comm_time_(0.0) {
    packed_pes_.set_empty_key(-1);
    adjacent_pes_.set_empty_key(-1);
    send_buffers_a_.set_empty_key(-1);
    send_buffers_b_.set_empty_key(-1);
    current_send_buffers_ = &send_buffers_a_;
    send_tag_ = static_cast<unsigned int>(100 * size_);
    recv_tag_ = static_cast<unsigned int>(100 * size_);
  }
  virtual ~VertexCommunicator() {};

  VertexCommunicator(const VertexCommunicator &rhs) = default;
  VertexCommunicator(VertexCommunicator &&rhs) = default;
 
  inline void SetGraph(GraphType *g) {
    g_ = g;
  }

  inline PEID GetNumberOfAdjacentPEs() const {
    return adjacent_pes_.size();
  }

  template<typename F>
  void ForallAdjacentPEs(F &&callback) {
    for (const PEID &pe : adjacent_pes_) {
      callback(pe);
    }
  }

  inline bool IsAdjacentPE(const PEID pe) const {
    return adjacent_pes_.find(pe) != adjacent_pes_.end();
  }

  inline bool IsPackedPE(const PEID pe) const {
    return packed_pes_.find(pe) != packed_pes_.end();
  }

  void SetAdjacentPE(const PEID pe, const bool is_adj) {
    if (pe == rank_) return;
    if (is_adj) {
      if (IsAdjacentPE(pe)) return;
      else adjacent_pes_.insert(pe);
    } else {
      if (!IsAdjacentPE(pe)) return;
      else adjacent_pes_.erase(pe);
    }
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

  void ResetAdjacentPEs() {
    adjacent_pes_.clear();
  }

  void AddMessage(VertexID v, const VertexPayload &msg);

  void ReceiveAndSendGhostVertices() {
    comm_timer_.Restart();
    if (send_tag_ > 100 * size_) ReceiveMessages();
    SendMessages();
    ClearAndSwitchBuffers();
    comm_time_ += comm_timer_.Elapsed();
  }

  void SendAndReceiveGhostVertices() {
    comm_timer_.Restart();
    SendMessages();
    ReceiveMessages();
    ClearAndSwitchBuffers();
    comm_time_ += comm_timer_.Elapsed();
  }

  float GetCommTime() {
    return comm_time_;
  }


 private:
  MPI_Comm communicator_;
  GraphType *g_;

  PEID rank_, size_;

  google::dense_hash_set<PEID> adjacent_pes_;
  google::dense_hash_set<PEID> packed_pes_;
  google::dense_hash_map<PEID, Buffer> *current_send_buffers_;
  google::dense_hash_map<PEID, Buffer> send_buffers_a_;
  google::dense_hash_map<PEID, Buffer> send_buffers_b_;
  std::vector<MPI_Request> isend_requests_;

  unsigned int send_tag_;
  unsigned int recv_tag_;

  float comm_time_;
  Timer comm_timer_;

  void SendMessages() {
    send_tag_++;
    ForallAdjacentPEs([&](const PEID &pe) {
      if ((*current_send_buffers_)[pe].empty()) {
        (*current_send_buffers_)[pe].emplace_back(std::numeric_limits<VertexID>::max());
        (*current_send_buffers_)[pe].emplace_back(0);
        (*current_send_buffers_)[pe].emplace_back(0);
        (*current_send_buffers_)[pe].emplace_back(0);
      }
      isend_requests_.emplace_back(MPI_Request());
      MPI_Isend((*current_send_buffers_)[pe].data(),
                static_cast<int>((*current_send_buffers_)[pe].size()),
                MPI_VERTEX, pe,
                send_tag_, communicator_, &isend_requests_.back());
    });
  }

  void ReceiveMessages();

  void ClearAndSwitchBuffers() {
    if (current_send_buffers_ == &send_buffers_a_) {
      for (auto &kv : send_buffers_b_) kv.second.clear();
      send_buffers_b_.clear();
      current_send_buffers_ = &send_buffers_b_;
    } else {
      for (auto &kv : send_buffers_a_) kv.second.clear();
      send_buffers_a_.clear();
      current_send_buffers_ = &send_buffers_a_;
    }
  }
};

template class VertexCommunicator<DynamicGraphCommunicator>;
template class VertexCommunicator<SemidynamicGraphCommunicator>;
template class VertexCommunicator<StaticGraphCommunicator>;

#endif
