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
        config_(conf) {}
  virtual ~Components() = default;

  void FindComponents(GraphAccess &g) {
    FindLocalComponents(g);
    Contraction cont(g, rank_, size_);
    GraphAccess cag = cont.BuildComponentAdjacencyGraph();
    PerformDecomposition(cag);
  }

  void Output(GraphAccess &g, const PEID rank) {}

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  void PerformDecomposition(GraphAccess &g) {
    // FindGhostReductions(g);
    RunExponentialBFS(g);
    DetermineSupernodes(g);
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

  // TODO: This is an optimization
  void FindGhostReductions(GraphAccess &g) {
    // Find local vertices connected by same ghost vertex
    std::unordered_map<VertexID, std::vector<VertexID>> vertex_buckets(g.NumberOfGhostVertices());
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
        for (VertexID &v : bucket.second) min_label = std::min(g.GetVertexLabel(v), min_label);
        for (VertexID &v : bucket.second) g.SetVertexLabel(v, min_label);
      }
    }

    // Find ghost vertices connected by same local vertex
    std::unordered_map<VertexID, std::vector<VertexID>> ghost_buckets(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](const VertexID v) {
      ASSERT_TRUE(g.IsInterface(v))
      g.ForallNeighbors(v, [&](const VertexID w) {
        ASSERT_TRUE(g.IsGhost(w))
        ghost_buckets[v].push_back(w);
      });
    });

    // Group buckets by PE
    std::unordered_map<PEID, std::unordered_map<VertexID, std::vector<VertexID>>>
        pe_ghost_buckets(static_cast<unsigned long>(g.GetNumberOfAdjacentPEs()));
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
        for (VertexID &v : pe_vertex_bucket.second) min_label = std::min(g.GetVertexLabel(v), min_label);
        for (VertexID &v : pe_vertex_bucket.second) g.SetVertexLabel(v, min_label);
      }
    }

    // Merge local vertices and remove excess edges
    g.ForallLocalVertices([&](const VertexID v) {
      // Active component
      if (g.GetVertexLabel(v) == g.GetGlobalID(v)) {
        std::vector<std::pair<VertexID, VertexID>> edges_to_remove;
        g.ForallNeighbors(v, [&](VertexID w) {
          // Remove edges to inactive neigbors (all ghosts)
          if (g.GetVertexLabel(w) != g.GetGlobalID(w)) edges_to_remove.emplace_back(v, w);
        });
        for (auto &edge : edges_to_remove) g.RemoveEdge(edge.first, edge.second);
      }
        // Inactive component
      else {
        std::vector<std::pair<VertexID, VertexID>> edges_to_remove;
        g.ForallNeighbors(v, [&](VertexID w) {
          // Remove all edges
          edges_to_remove.emplace_back(v, w);
        });
        for (auto &edge : edges_to_remove) g.RemoveEdge(edge.first, edge.second);
      }
    });

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void RunExponentialBFS(GraphAccess &g) {
    // Initialize PRNG
    std::mt19937 generator(static_cast<unsigned int>(config_.seed + config_.n + rank_));
    std::exponential_distribution<double> distribution(config_.beta);

    // Draw exponential deviate per vertex
    g.ForallLocalVertices([&](const VertexID v) {
      g.SetVertexMsg(v, static_cast<VertexID>(distribution(generator)));
    });

    // Send initial deviates
    g.UpdateGhostVertices();

    unsigned int iteration = 0;
    bool converged_globally = false;
    while (!converged_globally) {
      bool converged_locally = true;
      // Receive variates
      g.UpdateGhostVertices();

      // Perform update for local vertices
      g.ForallLocalVertices([&](VertexID v) {
        g.ForallNeighbors(v, [&](VertexID w) {
          if (g.GetVertexMsg(w) + 1 < g.GetVertexMsg(v)) {
            g.SetVertexLabel(v, g.GetVertexLabel(w), g.GetVertexMsg(w) + 1);
            converged_locally = false;
          }
        });
      });

      // Check if all PEs are done
      MPI_Allreduce(&converged_locally, &converged_globally, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    // Output converged deviates
    g.OutputLocal();
  }

  void DetermineSupernodes(GraphAccess &g) {
    // Build MPI groups based on local deviates
    std::cout << "[R" << rank_ << "] global comm " << rank_ << "/" << size_ << std::endl;
    std::unordered_set<VertexID> groups;
    g.ForallLocalVertices([&](VertexID v) {
      VertexID group = g.GetVertexLabel(v);
      if (groups.find(group) == groups.end()) {
        groups.insert(group);
        std::cout << "[R" << rank_ << "] label=" << group << std::endl;
      }
    });

    // std::unordered_map<VertexID, MPI_Comm> comms;
    // for (VertexID id : groups) {
    //   MPI_Comm label_comm;
    //   MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(id), rank_, &label_comm);
    //   comms[id] = label_comm;
    //   PEID label_rank, label_size;
    //   MPI_Comm_rank(label_comm, &label_rank);
    //   MPI_Comm_size(label_comm, &label_size);
    //   std::cout << "[R" << rank_ << "] join comm for label= " << id << " with rank " << label_rank << "/" << label_size
    //     << std::endl;
    // }
  }

};

#endif
