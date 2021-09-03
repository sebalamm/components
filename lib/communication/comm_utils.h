/******************************************************************************
 * comm_utils.h
 *
 * Utility algorithms needed for connected components
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

#ifndef _COMM_UTILITY_H_
#define _COMM_UTILITY_H_

#include "definitions.h"

class CommunicationUtility {
 public:

  static float SparseAllToAll(google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                              google::dense_hash_map<PEID, VertexBuffer> &receive_buffers,
                              PEID rank, PEID size, PEID message_tag = 0) {
    Timer comm_timer;
    comm_timer.Restart();
    PEID num_requests = 0;
    for (const auto &kv : send_buffers) {
      PEID pe = kv.first;
      if (send_buffers[pe].size() > 0) num_requests++; 
    }
    std::vector<MPI_Request> requests(num_requests);

    int req = 0;
    for (const auto &kv : send_buffers) {
      PEID pe = kv.first;
      if (send_buffers[pe].size() > 0) {
        MPI_Issend(send_buffers[pe].data(), 
                   static_cast<int>(send_buffers[pe].size()), 
                   MPI_VERTEX, pe, message_tag * size + pe, MPI_COMM_WORLD, &requests[req++]);
        if (pe == rank) {
          std::cout << "R" << rank << " ERROR selfmessage" << std::endl;
          exit(1);
        }
      } 
    }

    std::vector<MPI_Status> statuses(num_requests);
    int isend_done = 0;
    while (isend_done == 0) {
      // Check for messages
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, message_tag * size + rank, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          VertexBuffer message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
      // Check if all ISend successful
      isend_done = 0;
      MPI_Testall(num_requests, requests.data(), &isend_done, statuses.data());
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

    int ibarrier_done = 0;
    while (ibarrier_done == 0) {
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, message_tag * size + rank, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          VertexBuffer message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
        
      // Check if all reached Ibarrier
      MPI_Status tst{};
      MPI_Test(&barrier_request, &ibarrier_done, &tst);
      if (tst.MPI_ERROR != MPI_SUCCESS) {
        std::cout << "R" << rank << " mpi_test (barrier) failed" << std::endl;
        exit(1);
      }
    }
    return comm_timer.Elapsed();
  }

  static VertexID ClearBuffers(google::dense_hash_map<PEID, VertexBuffer> &buffers) {
    VertexID messages = 0;
    for (auto &kv: buffers) {
      kv.second.clear();
      messages += kv.second.size();
    }
    buffers.clear();
    return messages;
  }
};

#endif
