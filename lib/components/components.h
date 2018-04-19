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
#include "union_find.h"
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
    if (rank_ == ROOT) std::cout << "[STATUS] Find local components" << std::endl;
    FindLocalComponents(g);
    if (rank_ == ROOT) std::cout << "[STATUS] Contract local components" << std::endl;
    Contraction cont(g, rank_, size_);
    GraphAccess cag = cont.BuildComponentAdjacencyGraph();
    if (rank_ == ROOT) std::cout << "[STATUS] Perform main algorithm" << std::endl;
    // FindHighDegreeVertices(cag);
    PerformDecomposition(cag);
    ApplyToLocalComponents(cag, g);
  }

  void Output(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "Component labels" << std::endl;
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;

  // Local components
  std::vector<VertexID> parent;

  void PerformDecomposition(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "[STATUS] |- Start exponential BFS" << std::endl;
    RunExponentialBFS(g);
    if (rank_ == ROOT) std::cout << "[STATUS] |- Propagate labels upward" << std::endl;
    PropagateLabelsUp(g);
  }

  void FindLocalComponents(GraphAccess &g) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    parent.resize(g.GetNumberOfVertices());

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS(g, v, marked, parent);
    });

    // Set vertex labels for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      // g.SetVertexLabel(v, parent[v]);
      g.SetVertexPayload(v,
                         {g.GetVertexDeviate(v), g.GetVertexLabel(parent[v]),
                          rank_});
    });
  }

  void FindHighDegreeVertices(GraphAccess &g) {
    std::vector<VertexID> local_high_degree;
    int num_local_high_degree = 0;

    // Gather local high degree vertices
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID degree = g.GetVertexDegree(v);
      if (degree >= config_.degree_limit) {
        local_high_degree.push_back(g.GetGlobalID(v));
        num_local_high_degree++;
      }
    });

    // Gather number of high degrees via all-gather
    std::vector<int> num_high_degree(size_);
    MPI_Allgather(&num_local_high_degree, 1, MPI_INT,
                  &num_high_degree[0], 1, MPI_INT,
                  MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displ(size_);
    int num_global_high_degree = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ[i] = num_global_high_degree;
      num_global_high_degree += num_high_degree[i];
    }

    // Distribte vertices using all-to-all
    std::vector<VertexID> global_high_degree(num_global_high_degree);
    MPI_Allgatherv(&local_high_degree[0], num_local_high_degree, MPI_LONG,
                   &global_high_degree[0], &num_high_degree[0], &displ[0], MPI_LONG,
                   MPI_COMM_WORLD);

    for (VertexID i = 0; i < num_global_high_degree; ++i) {
      std::cout << "r " << rank_ << " v " << global_high_degree[i] << std::endl;
    }
  }

  void RunExponentialBFS(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "[STATUS] |-- Iteration " << iteration_ << std::endl;
    std::exponential_distribution<double> distribution(config_.beta);

    // TODO: Prioritize high degree vertices to limit reduce number of messages
    // Draw exponential deviate per ghost vertex
    // g.ForallGhostVertices([&](const VertexID v) {
    //   std::mt19937
    //       generator(static_cast<unsigned int>(config_.seed + g.GetVertexLabel(v)
    //       + iteration_ * g.GetNumberOfVertices() * size_));
    //   g.SetVertexPayload(v,
    //                      {static_cast<VertexID>(distribution(generator)),
    //                       g.GetVertexLabel(v), g.GetVertexRoot(v)});
    // });

    // Draw exponential deviate per local vertex
    g.ForallVertices([&](const VertexID v) {
      // Set preliminary deviate
      std::mt19937
          generator(static_cast<unsigned int>(config_.seed + g.GetVertexLabel(v)
          + iteration_ * g.GetNumberOfVertices() * size_));
      // TODO: Also initialize ghost and don't send deviates during init
      if (g.IsLocal(v)) {
        g.SetParent(v, v);
        VertexPayload smallest_payload = {static_cast<VertexID>(distribution(generator)),
                                          g.GetVertexLabel(v), g.GetVertexRoot(v)};
        g.SetVertexPayload(v, std::move(smallest_payload));
      }
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
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
            smallest_payload = {g.GetVertexDeviate(w) + 1, g.GetVertexLabel(w),
                                g.GetVertexRoot(w)};
            converged_locally = 0;
          }
        });
        g.SetVertexPayload(v, std::move(smallest_payload));
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
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    // g.DetermineActiveVertices();

    // Count remaining number of vertices
    VertexID remaining_global_vertices = g.GatherNumberOfGlobalVertices();
    if (remaining_global_vertices > 0) {
      iteration_++;
      if (remaining_global_vertices < config_.sequential_limit) {
        if (rank_ == ROOT) 
          std::cout << "[STATUS] Perform sequential computation (n=" 
                    << remaining_global_vertices << ")" << std::endl;
        RunSequentialCC(g);
      }
      else RunExponentialBFS(g);
    }
  }

  void PropagateLabelsUp(GraphAccess &g) {
    g.MoveUpContraction();
  }

  void ApplyToLocalComponents(GraphAccess &cag, GraphAccess &g) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g.GetContractionVertex(v);
      g.SetVertexPayload(v, {0, cag.GetVertexLabel(cag.GetLocalID(cv)), rank_});
    });
  }

  void RunSequentialCC(GraphAccess &g) {
    // Perform gather of graph on root 
    std::vector<VertexID> vertices;
    std::vector<int> num_vertices_per_pe(size_);
    std::vector<VertexID> labels;
    std::vector<std::pair<VertexID, VertexID>> edges;
    if (rank_ == ROOT) 
      std::cout << "[STATUS] Gather vertices on root" << std::endl;
    g.GatherGraphOnRoot(vertices, num_vertices_per_pe, labels, edges);

    // Root computes labels
    if (rank_ == ROOT) {
      std::cout << "[STATUS] Compute labels on root" << std::endl;
      // Build vertex mapping 
      std::unordered_map<VertexID, int> vertex_map;
      std::unordered_map<int, VertexID> reverse_vertex_map;
      int current_vertex = 0;
      for (const VertexID &v : vertices) {
        vertex_map[v] = current_vertex;
        reverse_vertex_map[current_vertex++] = v;
      }

      // Build edge lists
      std::vector<std::vector<int>> edge_lists(vertices.size());
      for (const auto &e : edges) 
        edge_lists[vertex_map[e.first]].push_back(vertex_map[e.second]);

      // Construct temporary graph
      GraphAccess sg(ROOT, 1);
      sg.StartConstruct(vertices.size(), edges.size(), ROOT);
      for (int i = 0; i < vertices.size(); ++i) {
        VertexID v = sg.AddVertex();
        sg.SetVertexPayload(v, {sg.GetVertexDeviate(v), labels[v], ROOT});

        for (const int &e : edge_lists[v]) 
          sg.AddEdge(v, e, 1);
      }
      sg.FinishConstruct();
      FindLocalComponents(sg);

      // Gather labels
      sg.ForallLocalVertices([&](const VertexID &v) {
        labels[v] = sg.GetVertexLabel(v);
      });
    }

    // Distribute labels to other PEs
    if (rank_ == ROOT) 
      std::cout << "[STATUS] Distribute labels from root" << std::endl;
    g.DistributeLabelsFromRoot(labels, num_vertices_per_pe);
  }

};

#endif
