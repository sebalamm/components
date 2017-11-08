/******************************************************************************
 * components.h
 *
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

#ifndef _COMPONENTS_H_
#define _COMPONENTS_H_

#include <iostream>
#include <unordered_set>
#include <random>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "graph_access.h"
#include "graph_contraction.h"
#include "utils.h"

class Components {
 public:
  Components(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf) {}
  virtual ~Components() = default;

  void FindComponents(GraphAccess &g) {
    // if (rank_ == 1) g.OutputLocal();
    FindLocalComponents(g);
    Contraction cont(g, rank_, size_);
    GraphAccess cag = cont.BuildComponentAdjacencyGraph();
    PerformDecomposition(cag);
    // if (rank_ == 1) cag.OutputLocal();
  }

  void Output(GraphAccess &g, const PEID rank) {
    // if (rank == ROOT) g.OutputLocal();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  void PerformDecomposition(GraphAccess &g) {
    std::cout << "rank " << rank_ << " perform decomposition" << std::endl;
    // ExchangeNeighborReductions(g);
    // RunExponentialBFS(g);
    // DetermineSupernodes(g);
    // GraphAccess cg = ContractDecomposition(g);
    // PerformDecomposition(cg);
    // UncontractDecomposition(g, cg);
  }

  void FindLocalComponents(GraphAccess &g) {
    std::vector<bool> marked(g.GetNumberOfLocalVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfLocalVertices());

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS(g, v, marked, parent);
    });

    // Set vertex labels for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      g.SetVertexLabel(v, parent[v]);
    });
  }

  void ExchangeNeighborReductions(GraphAccess &g) {
    std::vector<PEID> adjacent_pes;
    g.ForallLocalVertices([&](const VertexID v) {
      PEID neighbor_pe = 0;
      bool neighbor = false;
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (g.IsGhost(w)) {
          PEID target_pe = g.GetPE(w);
          // reductions[target_pe].push_back(w);
        }
      });
    });

    // Send reductions to neighbors
    for (PEID p : adjacent_pes) {
      // Send locally determined CCs
    }
  }

  void RunExponentialBFS(GraphAccess &g) {
    // Initialize PRNG
    std::mt19937 generator(static_cast<unsigned int>(config_.seed + rank_));
    std::exponential_distribution<double> distribution(config_.beta);

    // Draw exponential deviate per vertex
    std::vector<double> delta(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) { delta[v] = distribution(generator); });

    // Exchange deviates
    ExchangeDeviates(delta);

    // Each vertex maintains vertices with lower/higher deviate
    std::vector<std::unordered_set<VertexID>> ancestors(g.GetNumberOfLocalVertices());
    std::vector<std::unordered_set<VertexID>> successors(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) {
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (delta[w] < delta[v] - 1) ancestors[v].insert(w);
        else if (delta[w] - 1 < delta[v]) successors[v].insert(w);
      });
    });

    // Each vertex with an unprocessed ancestor is idle
    std::vector<VertexID> active_partition(g.GetNumberOfLocalVertices());
    std::iota(begin(active_partition), end(active_partition), 0);
    VertexID num_active_vertices = g.GetNumberOfLocalVertices();
    for (VertexID i = 0; i < g.GetNumberOfLocalVertices(); ++i) {
      VertexID v = active_partition[i];
      if (!ancestors[v].empty()) {
        std::swap(active_partition[i], active_partition[num_active_vertices - 1]);
        num_active_vertices--;
      }
    }

    // While we still have idle vertices
    VertexID visited_vertices = 0;
    while (visited_vertices < g.GetNumberOfLocalVertices()) {
      for (VertexID i = 0; i < num_active_vertices; ++i) {
        VertexID v = active_partition[i];
        // Make changes to successors
        for (const VertexID w : successors[v]) {
          g.SetVertexLabel(w, v + 1);
        }
      }

      // Propagate successor changes
      g.UpdateGhostVertices();
      HandleGhostUpdates();

      // Check for newly activated vertices
      for (VertexID i = num_active_vertices; i < g.GetNumberOfLocalVertices(); ++i) {
        VertexID v = active_partition[i];
        // Search for ancestors to remove
        for (VertexID w : ancestors[v]) { if (delta[w] >= delta[v] - 1) w = ancestors[v].erase(w); }
        // Vertex active?
        if (ancestors[v].empty()) {
          std::swap(active_partition[i], active_partition[num_active_vertices]);
          num_active_vertices++;
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void ExchangeDeviates(std::vector<double> &deviates) {}

  void HandleGhostUpdates() {}
};

#endif
