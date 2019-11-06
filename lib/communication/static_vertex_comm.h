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

#ifndef _STATIC_VERTEX_COMMUNICATOR_H_
#define _STATIC_VERTEX_COMMUNICATOR_H_

#include <unordered_map>
#include <vector>
#include <memory>

#include "config.h"
#include "static_graph_comm.h"

using Buffer = std::vector<VertexID>;

class StaticVertexCommunicator {
 public:
  StaticVertexCommunicator(const PEID rank,
                   const PEID size,
                   MPI_Comm communicator)
      : communicator_(communicator),
        g_(nullptr),
        rank_(rank),
        size_(size) {
    packed_pes_.resize(static_cast<unsigned long>(size_), false);
    adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
    send_buffers_a_.resize(static_cast<unsigned long>(size_));
    send_buffers_b_.resize(static_cast<unsigned long>(size_));
    current_send_buffers_ = &send_buffers_a_;
    send_tag_ = static_cast<unsigned int>(100 * size_);
    recv_tag_ = static_cast<unsigned int>(100 * size_);
  }
  virtual ~StaticVertexCommunicator() {};

  StaticVertexCommunicator(const StaticVertexCommunicator &rhs) = default;
  StaticVertexCommunicator(StaticVertexCommunicator &&rhs) = default;
 
  inline void SetGraph(StaticGraphCommunicator *g) {
    g_ = g;
  }

  inline void SetAdjacentPE(const PEID neighbor, const bool is_adj) {
    adjacent_pes_[neighbor] = is_adj;
  }

  inline PEID GetNumberOfAdjacentPEs() const {
    PEID counter = 0;
    for (PEID i = 0; i < adjacent_pes_.size(); i++) {
      bool is_adj = adjacent_pes_[i];
      if (is_adj) {
        counter++;
      }
    }
    return counter;
  }

  void AddMessage(VertexID v, const VertexPayload &msg);

  void ReceiveAndSendGhostVertices() {
    if (send_tag_ > 100 * size_) ReceiveMessages();
    SendMessages();
    ClearAndSwitchBuffers();
  }

  void SendAndReceiveGhostVertices() {
    SendMessages();
    ReceiveMessages();
    ClearAndSwitchBuffers();
  }

 private:
  MPI_Comm communicator_;
  StaticGraphCommunicator *g_;

  PEID rank_, size_;

  std::vector<bool> adjacent_pes_;
  std::vector<bool> packed_pes_;
  std::vector<Buffer> *current_send_buffers_;
  std::vector<Buffer> send_buffers_a_;
  std::vector<Buffer> send_buffers_b_;
  std::vector<MPI_Request> isend_requests_;

  unsigned int send_tag_;
  unsigned int recv_tag_;

  void SendMessages() {
    send_tag_++;
    for (PEID pe = 0; pe < size_; ++pe) {
      if (adjacent_pes_[pe]) {
        if ((*current_send_buffers_)[pe].empty()) {
          (*current_send_buffers_)[pe].emplace_back(std::numeric_limits<VertexID>::max());
          (*current_send_buffers_)[pe].emplace_back(0);
          (*current_send_buffers_)[pe].emplace_back(0);
          (*current_send_buffers_)[pe].emplace_back(0);
        }
        isend_requests_.emplace_back(MPI_Request());
        MPI_Isend(&(*current_send_buffers_)[pe][0],
                  static_cast<int>((*current_send_buffers_)[pe].size()),
                  MPI_VERTEX, pe,
                  send_tag_, communicator_, &isend_requests_.back());
      }
    }
  }

  void ReceiveMessages();

  void ClearAndSwitchBuffers() {
    if (current_send_buffers_ == &send_buffers_a_) {
      for (int i = 0; i < size_; ++i) send_buffers_b_[i].clear();
      current_send_buffers_ = &send_buffers_b_;
    } else {
      for (int i = 0; i < size_; ++i) send_buffers_a_[i].clear();
      current_send_buffers_ = &send_buffers_a_;
    }
  }
};

#endif