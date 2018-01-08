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

#ifndef _COMPONENTS_H_
#define _COMPONENTS_H_

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
#include "propagation.h"

class Components {
 public:
  Components(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0) {}
  virtual ~Components() = default;

  void FindComponents(GraphAccess &g) {
    FindLocalComponents(g);
    Contraction cont(g, rank_, size_);
    GraphAccess cag = cont.BuildComponentAdjacencyGraph();
    PerformDecomposition(cag);
  }

  void Output(GraphAccess &g) {
    MPI_Barrier(MPI_COMM_WORLD);
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;

  void PerformDecomposition(GraphAccess &g) {
    // FindGhostReductions(g);
    g.Logging(false);
    RunExponentialBFS(g);
    PropagateLabelsUp(g);
    Output(g);
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
      // g.SetVertexLabel(v, parent[v]);
      g.SetVertexPayload(v, {g.GetVertexDeviate(v), g.GetVertexLabel(parent[v]), rank_});
    });
  }

  // TODO: This is an optimization
  void FindGhostReductions(GraphAccess &g) {
    // Find local vertices connected by same ghost vertex
    std::unordered_map<VertexID, std::vector<VertexID>>
        vertex_buckets(g.NumberOfGhostVertices());
    g.ForallLocalVertices([&](const VertexID v) {
      ASSERT_TRUE(g.IsInterface(v))
      g.ForallNeighbors(v, [&](const VertexID w) {
        ASSERT_TRUE(g.IsGhost(w))
        vertex_buckets[w].push_back(v);
      });
    });

    // Update labels for local vertices
    for (auto &bucket : vertex_buckets) {
      if (bucket.second.size() >= 2) {
        VertexID min_label = g.GetNumberOfVertices();
        for (VertexID &v : bucket.second)
          min_label = std::min(g.GetVertexLabel(v),
                               min_label);
        for (VertexID &v : bucket.second)
          // g.SetVertexLabel(v, min_label);
          g.SetVertexPayload(v,
                             {g.GetVertexDeviate(v), min_label,
                              g.GetVertexRoot(v)});
      }
    }

    // Find ghost vertices connected by same local vertex
    std::unordered_map<VertexID, std::vector<VertexID>>
        ghost_buckets(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) {
      ASSERT_TRUE(g.IsInterface(v))
      g.ForallNeighbors(v, [&](const VertexID w) {
        ASSERT_TRUE(g.IsGhost(w))
        ghost_buckets[v].push_back(w);
      });
    });

    // Group buckets by PE
    std::unordered_map<PEID,
                       std::unordered_map<VertexID, std::vector<VertexID>>>
        pe_ghost_buckets
        (static_cast<unsigned long>(g.GetNumberOfAdjacentPEs()));
    for (auto &bucket : ghost_buckets) {
      for (VertexID &v : bucket.second) {
        PEID target_pe = g.GetPE(v);
        pe_ghost_buckets[target_pe][bucket.first].push_back(v);
      }
    }

    // Update labels for ghost vertices
    for (auto &pe_bucket : pe_ghost_buckets) {
      for (auto &pe_vertex_bucket: pe_bucket.second) {
        VertexID min_label = g.GetNumberOfVertices();
        for (VertexID &v : pe_vertex_bucket.second)
          min_label = std::min(g.GetVertexLabel(v), min_label);
        for (VertexID &v : pe_vertex_bucket.second)
          // g.SetVertexLabel(v, min_label);
          g.SetVertexPayload(v,
                             {g.GetVertexDeviate(v), min_label,
                              g.GetVertexRoot(v)});
      }
    }

    // Merge local vertices and remove excess edges
    g.ForallLocalVertices([&](const VertexID v) {
      // Active component
      if (g.GetVertexLabel(v) == g.GetGlobalID(v)) {
        std::vector<std::pair<VertexID, VertexID>> edges_to_remove;
        g.ForallNeighbors(v, [&](VertexID w) {
          // Remove edges to inactive neigbors (all ghosts)
          if (g.GetVertexLabel(w) != g.GetGlobalID(w))
            edges_to_remove.emplace_back(v, w);
        });
        for (auto &edge : edges_to_remove)
          g.RemoveEdge(edge.first,
                       edge.second);
      }
      // Inactive component
      else {
        std::vector<std::pair<VertexID, VertexID>> edges_to_remove;
        g.ForallNeighbors(v, [&](VertexID w) {
          // Remove all edges
          edges_to_remove.emplace_back(v, w);
        });
        for (auto &edge : edges_to_remove)
          g.RemoveEdge(edge.first, edge.second);
      }
    });
  }

  void RunExponentialBFS(GraphAccess &g) {
    // Initialize PRNG
    std::exponential_distribution<double> distribution(config_.beta);

    // Draw exponential deviate per vertex
    g.ForallLocalVertices([&](const VertexID v) {
      std::mt19937
          generator(static_cast<unsigned int>(config_.seed + g.GetVertexLabel(v) + iteration_ * g.GetNumberOfVertices() * size_));
      g.SetParent(v, v);
      g.SetVertexPayload(v, {static_cast<VertexID>(distribution(generator)), g.GetVertexLabel(v), g.GetVertexRoot(v)});
      std::cout << "[R" << rank_ << ":" << iteration_ << "] draw deviate " << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v) << std::endl;
    });

    int converged_globally = 0;
    while (converged_globally == 0) {
      int converged_locally = 1;
      // Receive variates
      g.UpdateGhostVertices();

      // Perform update for local vertices
      g.ForallLocalVertices([&](VertexID v) {
        auto smallest_payload = g.GetVertexMessage(v);
        g.ForallNeighbors(v, [&](VertexID w) {
          if (g.GetVertexDeviate(w) + 1 < smallest_payload.deviate_ ||
              (g.GetVertexDeviate(w) + 1 == smallest_payload.deviate_ && 
               g.GetVertexLabel(w) < smallest_payload.label_)) {
            g.SetParent(v, w);
            smallest_payload = {g.GetVertexDeviate(w) + 1, g.GetVertexLabel(w), g.GetVertexRoot(w)};
            converged_locally = 0;
          }
        });
        g.SetVertexPayload(v, smallest_payload);
      });

      // Check if all PEs are done
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
    }
    // Output converged deviates
    g.UpdateGhostVertices();
    g.OutputLocal();
    g.DetermineActiveVertices();

    // Count remaining number of vertices
    VertexID local_vertices = 0;
    VertexID global_vertices = 0;
    g.ForallLocalVertices([&](VertexID v) { local_vertices++; });
    // Check if all PEs are done
    MPI_Allreduce(&local_vertices,
                  &global_vertices,
                  1,
                  MPI_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    iteration_++;
    if (global_vertices > 0) RunExponentialBFS(g);
  }

  void PropagateLabelsUp(GraphAccess &g) {
    g.MoveUpContraction();
  }
};

#endif
