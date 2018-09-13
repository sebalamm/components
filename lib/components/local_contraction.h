/******************************************************************************
 * components.h *
 * Distributed computation of connected components
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

#ifndef _LOCAL_CONTRACTION_H_
#define _LOCAL_CONTRACTION_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "graph_access.h"
#include "graph_contraction.h"
#include "utils.h"
#include "union_find.h"
#include "propagation.h"

class LocalContraction {
 public:
  LocalContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0) {}
  virtual ~LocalContraction() = default;

  void FindComponents(GraphAccess &g) {
    Timer t;
    t.Restart();
    PerformDecomposition(g);
  }

  void Output(GraphAccess &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;
  VertexID rng_offset_;

  // Local components
  std::vector<VertexID> parent_;

  void PerformDecomposition(GraphAccess &g) {
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      RunContraction(g);
    }
    PropagateLabelsUp(g);
  }

  void RunContraction(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "iteration " << iteration_ << std::endl;
    // Draw exponential deviate per local vertex
    std::uniform_int_distribution<unsigned int> distribution(0, 99);
    g.ForallVertices([&](const VertexID v) {
      // Set preliminary deviate
      std::mt19937
          generator(static_cast<unsigned int>(config_.seed + g.GetVertexLabel(v)
          + iteration_ * rng_offset_));
      if (g.IsLocal(v)) g.SetParent(v, rank_, g.GetGlobalID(v));

      g.SetVertexPayload(v, {static_cast<VertexID>(distribution(generator)),
                             g.GetVertexLabel(v), g.GetVertexRoot(v)}, 
                         false);
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
    });

    // Perform update for local vertices
    // Find smallest label in N(v)
    std::vector<VertexPayload> n_smallest_neighbor(g.GetNumberOfVertices());
    g.ForallLocalVertices([&](VertexID v) {
      n_smallest_neighbor[v] = g.GetVertexMessage(v);
      g.ForallNeighbors(v, [&](VertexID w) {
        // Store neighbor label
        if (g.GetVertexDeviate(w) < n_smallest_neighbor[v].deviate_ ||
            (g.GetVertexDeviate(w) == n_smallest_neighbor[v].deviate_ &&
                g.GetVertexLabel(w) < n_smallest_neighbor[v].label_)) {
          g.SetParent(v, g.GetPE(w), g.GetGlobalID(w));
          n_smallest_neighbor[v] = {g.GetVertexDeviate(w), g.GetVertexLabel(w),
                                 g.GetVertexRoot(w)};
        }
      });
    });

    g.ForallLocalVertices([&](VertexID v) {
      g.SetVertexPayload(v, std::move(n_smallest_neighbor[v]));
    });

    // Receive variates
    g.SendAndReceiveGhostVertices();

    // Perform update for local vertices
    // Find smallest label in N(N(v))
    std::vector<VertexPayload> nn_smallest_neighbor(g.GetNumberOfVertices());
    g.ForallLocalVertices([&](VertexID v) {
      nn_smallest_neighbor[v] = g.GetVertexMessage(v);
      g.ForallNeighbors(v, [&](VertexID w) {
        // Store neighbor label
        if (g.GetVertexDeviate(w) < nn_smallest_neighbor[v].deviate_ ||
            (g.GetVertexDeviate(w) == nn_smallest_neighbor[v].deviate_ &&
                g.GetVertexLabel(w) < nn_smallest_neighbor[v].label_)) {
          g.SetParent(v, g.GetPE(w), g.GetGlobalID(w));
          nn_smallest_neighbor[v] = {g.GetVertexDeviate(w), g.GetVertexLabel(w),
                                 g.GetVertexRoot(w)};
        }
      });
    });

    g.ForallLocalVertices([&](VertexID v) {
      g.SetVertexPayload(v, std::move(nn_smallest_neighbor[v]));
    });

    // Receive variates
    g.SendAndReceiveGhostVertices();

    // Determine remaining active vertices
    g.ContractLocal();

    // Count remaining number of vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      RunContraction(g);
    }
  }

  void PropagateLabelsUp(GraphAccess &g) {
    g.MoveUpContraction();
  }
};

#endif
