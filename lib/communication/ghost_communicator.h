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

#ifndef _GHOST_COMMUNICATOR_H_
#define _GHOST_COMMUNICATOR_H_

#include <unordered_map>
#include <vector>
#include <memory>

#include "config.h"
#include "graph_access.h"

using Buffer = std::vector<VertexID>;

class GhostCommunicator {
 public:
  GhostCommunicator(GraphAccess *g,
                    const PEID rank,
                    const PEID size,
                    MPI_Comm communicator)
      : communicator_(communicator),
        g_(g),
        rank_(rank),
        size_(size) {
    packed_pes_.resize(static_cast<unsigned long>(size_), false);
    adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
    send_buffers_a_.resize(static_cast<unsigned long>(size_));
    send_buffers_b_.resize(static_cast<unsigned long>(size_));
    current_send_buffers_ = &send_buffers_a_;
    current_send_tag_ = static_cast<unsigned int>(100 * size_);
    current_recv_tag_ = static_cast<unsigned int>(100 * size_);
  }
  virtual ~GhostCommunicator() = default;

  GhostCommunicator(const GhostCommunicator &rhs) = default;
  GhostCommunicator(GhostCommunicator &&rhs) = default;

  inline void SetAdjacentPE(const PEID neighbor, const bool is_adj) {
    adjacent_pes_[neighbor] = is_adj;
  }

  inline PEID GetNumberOfAdjacentPEs() const {
    PEID counter = 0;
    for (const bool is_adj : adjacent_pes_)
      if (is_adj) counter++;
    return counter;
  }

  void AddMessage(VertexID v, const VertexPayload &msg);

  void UpdateGhostVertices() {
    if (current_send_tag_ > 100 * size_) ReceiveIncomingMessages();
    SendMessages();
    ClearAndSwitchBuffers();
  }

 private:
  MPI_Comm communicator_;
  GraphAccess *g_;

  PEID rank_, size_;

  std::vector<bool> adjacent_pes_;
  std::vector<bool> packed_pes_;
  std::vector<Buffer> *current_send_buffers_;
  std::vector<Buffer> send_buffers_a_;
  std::vector<Buffer> send_buffers_b_;
  std::vector<MPI_Request *> isend_requests_;

  unsigned int current_send_tag_;
  unsigned int current_recv_tag_;

  void SendMessages() {
    current_send_tag_++;

    for (PEID pe = 0; pe < size_; ++pe) {
      if (adjacent_pes_[pe]) {
        if ((*current_send_buffers_)[pe].empty())
          (*current_send_buffers_)[pe].emplace_back(0);
      }

      auto *request = new MPI_Request();
      MPI_Isend(&(*current_send_buffers_)[pe][0],
                static_cast<int>((*current_send_buffers_)[pe].size()),
                MPI_LONG, pe,
                current_send_tag_, communicator_, request);
#ifndef NDEBUG
      if ((*current_send_buffers_)[pe].size() > 1) {
        for (int i = 0; i < (*current_send_buffers_)[pe].size() - 1; i += 4) {
          std::cout << "[R" << rank_ << "] send ("
                    << (*current_send_buffers_)[pe][i + 1] << ","
                    << (*current_send_buffers_)[pe][i] << ","
                    << (*current_send_buffers_)[pe][i + 2] << ") to pe "
                    << pe << " with tag " << current_send_tag_ << " length "
                    << (*current_send_buffers_)[pe].size() << std::endl;
        }
      }
#endif
      isend_requests_.push_back(request);
    }
  }

  void ReceiveIncomingMessages();

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
