/******************************************************************************
 * hierarchy.h
 *
 * Contraction hierarchy of distributed graph
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

#ifndef _HIERARCHY_H_
#define _HIERARCHY_H_

#include <iostream>
#include <google/dense_hash_map>

#include "config.h"
#include "definitions.h"
#include "graph_access.h"
#include "edge_hash.h"

class Hierarchy {
 public:
  Hierarchy(GraphAccess &g, const PEID rank, const PEID size)
      : rank_(rank), size_(size),
        contraction_level_(0) {
    msg_buffers_.resize(size_);
  }

  virtual ~Hierarchy() = default;

  void ContractVertices(GraphAccess &g) {
    // Active vertices
    is_active_.resize(contraction_level_ + 2);
    is_active_[contraction_level_ + 1].resize(g.GetNumberOfVertices(), false);

    // Gather vertex information from graph
    vertex_payload_.resize(contraction_level_ + 2);
    vertex_payload_[contraction_level_ + 1].resize(g.GetNumberOfVertices());

    parent_.resize(contraction_level_ + 2);
    parent_[contraction_level_ + 1].resize(g.GetNumberOfVertices());

    // Update hierarchy
    g.ForallVertices([&](const VertexID v) {
      vertex_payload_[contraction_level_][v] = g.GetVertexPayload(v);
      parent_[contraction_level_][v] = g.GetParent(v);
    });

    // Update stacks
    added_edges_.resize(contraction_level_ + 2);
    removed_edges_.resize(contraction_level_ + 2);

    // Determine edges to communicate
    std::vector<std::unordered_set<VertexID>> send_ids(size_);
    for (PEID i = 0; i < size_; ++i) msg_buffers_[i].clear();

    // Gather remaining edges and reset vertex payloads
    g.ForallLocalVertices([&](VertexID v) {
      VertexID vlabel = g.GetVertexLabel(v);
      g.ForallNeighbors(v, [&](VertexID w) {
        VertexID wlabel = g.GetVertexLabel(w);
        // Edge needs to be linked to root 
        if (vlabel != wlabel) {
          PEID pe = g.GetVertexRoot(v);
          VertexID update_id = vlabel + g.GetNumberOfLocalVertices() * wlabel;
          if (send_ids[pe].find(update_id) == send_ids[pe].end()) {
            send_ids[pe].insert(update_id);
            // TODO: Encode edges to reduce volume
            msg_buffers_[pe].push_back(vlabel);
            msg_buffers_[pe].push_back(wlabel);
            msg_buffers_[pe].push_back(g.GetVertexRoot(w));
#ifndef NDEBUG
            std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_
                      << "] [Link] send edge (" << vlabel << "," << wlabel
                      << "(R" << g.GetVertexRoot(w) << ")) to " << pe
                      << std::endl;
#endif
          }
        } 
        removed_edges_[contraction_level_].emplace_back(v, g.GetGlobalID(w));
      });
      g.RemoveAllEdges(v);
      vertex_payload_[contraction_level_ + 1][v] =
          {std::numeric_limits<VertexID>::max() - 1, g.GetVertexLabel(v), rank_};
    });

    // Send gathered edges
    for (PEID i = 0; i < size_; ++i) {
      if (!g.IsAdjacentPE(i) || i == rank_) continue;
      if (!(msg_buffers_[i].size() > 0)) continue;
      MPI_Request request;
      MPI_Isend(&msg_buffers_[i][0], msg_buffers_[i].size(), MPI_LONG, i, 0,
                MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
    }

    // Increase contraction level
    contraction_level_++;

    // Gather edge updates
    std::unordered_set<VertexID> inserted_edges;

    // Local updates
    if (msg_buffers_[rank_].size() > 0) {
      for (int i = 0; i < msg_buffers_[rank_].size() - 1; i += 3) {
        VertexID source = g.GetLocalID(msg_buffers_[rank_][i]);
        VertexID target = msg_buffers_[rank_][i+1];
        PEID target_pe = static_cast<PEID>(msg_buffers_[rank_][i+2]);
        VertexID edge_id = source + target * g.GetNumberOfLocalVertices();
        if (inserted_edges.find(edge_id) == inserted_edges.end()) {
          inserted_edges.insert(edge_id);
          is_active_[contraction_level_][source] = true;
          g.AddEdge(source, target, target_pe);
          added_edges_[contraction_level_ - 1].emplace_back(source, target);
#ifndef NDEBUG
          std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_ - 1
                    << "] [Link] recv edge (" << g.GetGlobalID(source) << "," << target
                    << "(R" << target_pe << ")) from " << rank_
                    << std::endl;
#endif
        }
      }
    }

    // Non-local updates
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status st{};
    int flag = 1;
    do {
      MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st);
      if (flag) {
        int message_length;
        MPI_Get_count(&st, MPI_LONG, &message_length);
        std::vector<VertexID> message(message_length);
        MPI_Status rst{};
        MPI_Recv(&message[0],
                 message_length,
                 MPI_LONG,
                 st.MPI_SOURCE,
                 0,
                 MPI_COMM_WORLD,
                 &rst);
        if (message_length == 1) continue;

        // Insert edges
        for (int i = 0; i < message_length - 1; i += 3) {
          VertexID source = g.GetLocalID(message[i]);
          VertexID target = message[i+1];
          PEID target_pe = static_cast<PEID>(message[i+2]);
          VertexID edge_id = source + target * g.GetNumberOfLocalVertices();
          if (inserted_edges.find(edge_id) == inserted_edges.end()) {
            inserted_edges.insert(edge_id);
            is_active_[contraction_level_][source] = true;
            g.AddEdge(source, target, target_pe);
            added_edges_[contraction_level_ - 1].emplace_back(source, target);
#ifndef NDEBUG
            std::cout << "[LOG] [R" << rank_ << ":" << contraction_level_ - 1
                      << "] [Link] recv edge (" << g.GetGlobalID(source) << "," << target
                      << "(R" << target_pe << ")) from " << st.MPI_SOURCE
                      << std::endl;
#endif
          }
        }
      }
    } while (flag);

    // Update graph
    g.SetVertexPayloads(GetVertexPayloads());
    g.SetActiveVertices(GetActiveVertices());
  }

  void UncontractVertices(GraphAccess &g) {
    while (contraction_level_ > 0) {
      // Update hierarchy
      g.ForallVertices([&](const VertexID v) {
        if (g.IsLocal(v)) g.RemoveAllEdges(v);
        vertex_payload_[contraction_level_][v] = g.GetVertexPayload(v);
        parent_[contraction_level_][v] = g.GetParent(v);
      });

      // Decrease level
      contraction_level_--;
      g.SetActiveVertices(GetActiveVertices());

      // Add previously removed edges
      for (auto &e : removed_edges_[contraction_level_])
        g.AddEdge(std::get<0>(e), std::get<1>(e), g.GetPE(g.GetLocalID(std::get<1>(e))));

      // Update local labels
      g.ForallLocalVertices([&](VertexID v) {
        if (vertex_payload_[contraction_level_][v].label_ != 
            vertex_payload_[contraction_level_ + 1][v].label_)
          g.SetVertexPayload(v, {0, vertex_payload_[contraction_level_ + 1][v].label_, rank_});
      });

      // Propagate labels
      int converged_globally = 0;
      while (converged_globally == 0) {
        int converged_locally = 1;
        // Receive variates
        g.SendAndReceiveGhostVertices();

        // Send current label from root
        g.ForallLocalVertices([&](VertexID v) {
          if (g.GetVertexLabel(g.GetParent(v)) != g.GetVertexLabel(v)) {
            g.SetVertexPayload(v, {0, g.GetVertexLabel(g.GetParent(v)), rank_});
            converged_locally = false;
          }
        });

        // Check if all PEs are done
        MPI_Allreduce(&converged_locally,
                      &converged_globally,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
      }
    }
  }

 private:
  // Network information
  PEID rank_, size_;

  // Contraction
  VertexID contraction_level_;
  std::vector<std::vector<std::pair<VertexID, VertexID>>> added_edges_;
  std::vector<std::vector<std::pair<VertexID, VertexID>>> removed_edges_;
  
  // Active vertices
  std::vector<std::vector<bool>> is_active_;

  // Vertex information
  std::vector<std::vector<VertexPayload>> vertex_payload_;
  std::vector<std::vector<VertexID>> parent_;

  // Buffers
  std::vector<std::vector<VertexID>> msg_buffers_;

  inline std::vector<bool> & GetActiveVertices() {
    return is_active_[contraction_level_];
  }

  inline std::vector<VertexPayload> & GetVertexPayloads() {
    return vertex_payload_[contraction_level_];
  }

  inline std::vector<VertexID> & GetParents() {
    return parent_[contraction_level_];
  }
};

#endif
