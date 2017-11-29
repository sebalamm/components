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

#ifndef _BLOCKING_COMMUNICATOR_H_
#define _BLOCKING_COMMUNICATOR_H_

#include <unordered_map>
#include <vector>
#include <memory>

#include "config.h"
#include "graph_access.h"

using Buffer = std::vector<VertexID>;

class BlockingCommunicator {
 public:
  BlockingCommunicator(GraphAccess *g,
                    const PEID rank,
                    const PEID size,
                    MPI_Comm communicator)
      : communicator_(communicator), g_(g), rank_(rank), size_(size), logging_(false) {
    packed_pes_.resize(static_cast<unsigned long>(size_), false);
    adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
    send_buffers_.resize(static_cast<unsigned long>(size_));
    send_tag_ = static_cast<unsigned int>(100 * size_);
    recv_tag_ = static_cast<unsigned int>(100 * size_);
  }
  virtual ~BlockingCommunicator() = default;

  BlockingCommunicator(const BlockingCommunicator &rhs) = default;
  BlockingCommunicator(BlockingCommunicator &&rhs) = default;

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
    SendMessages();
    MPI_Barrier(communicator_);
    ReceiveMessages();
    ClearBuffers();
  }

  void Logging(bool active);

 private:
  MPI_Comm communicator_;
  GraphAccess *g_;

  PEID rank_, size_;

  std::vector<bool> adjacent_pes_;
  std::vector<bool> packed_pes_;
  std::vector<Buffer> send_buffers_;
  std::vector<MPI_Request *> isend_requests_;

  unsigned int send_tag_;
  unsigned int recv_tag_;

  bool logging_;

  void SendMessages() {
    send_tag_++;

    for (PEID pe = 0; pe < size_; ++pe) {
      if (adjacent_pes_[pe]) {
        if (send_buffers_[pe].empty())
          send_buffers_[pe].emplace_back(0);
        auto *request = new MPI_Request();
        MPI_Isend(&send_buffers_[pe][0],
                  static_cast<int>(send_buffers_[pe].size()),
                  MPI_LONG, pe,
                  send_tag_, communicator_, request);

        if (logging_) {
          if (send_buffers_[pe].size() > 1) {
            for (int i = 0; i < send_buffers_[pe].size() - 1; i += 4) {
              std::cout << "[R" << rank_ << "] send [" << send_buffers_[pe][i] << "]("
                        << send_buffers_[pe][i + 1] << ","
                        << send_buffers_[pe][i + 2] << ","
                        << send_buffers_[pe][i + 3] << ") to pe "
                        << pe << " with tag " << send_tag_ << " length "
                        << send_buffers_[pe].size() << std::endl;
            }
          }
        }
        isend_requests_.push_back(request);
      }
    }
  }

  void ReceiveMessages();

  void ClearBuffers() {
    for (int i = 0; i < size_; ++i) send_buffers_[i].clear();
  }
};

#endif
