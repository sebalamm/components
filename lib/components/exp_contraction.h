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

#ifndef _EXP_CONTRACTION_H_
#define _EXP_CONTRACTION_H_

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

class ExponentialContraction {
 public:
  ExponentialContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0) {}
  virtual ~ExponentialContraction() = default;

  void FindComponents(GraphAccess &g) {
    Timer t;
    t.Restart();
    rng_offset_ = size_ + config_.seed;
    if (config_.use_contraction) {
      FindLocalComponents(g);
      Contraction cont(g, rank_, size_);
      GraphAccess cag = cont.BuildComponentAdjacencyGraph();
      FindLocalComponents(cag);
      Contraction ccont(cag, rank_, size_);
      GraphAccess ccag = ccont.BuildComponentAdjacencyGraph();
      // rng_offset_ = ccag.GatherNumberOfGlobalVertices();

      PerformDecomposition(ccag);

      ApplyToLocalComponents(ccag, cag);
      ApplyToLocalComponents(cag, g);
    } else PerformDecomposition(g);
  }

  void Output(GraphAccess &g) {
    // if (rank_ == ROOT) std::cout << "Component labels" << std::endl;
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

  // Statistics
  Timer iteration_timer_;

  void PerformDecomposition(GraphAccess &g) {
    // if (rank_ == ROOT) std::cout << "[STATUS] |- Start exponential BFS" << std::endl;
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_timer_.Restart();
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
        RunSequentialCC(g);
      else RunContraction(g);
    }
    // if (rank_ == ROOT) std::cout << "[STATUS] |- Propagate labels upward" << std::endl;
    PropagateLabelsUp(g);
  }

  void FindLocalComponents(GraphAccess &g) {
    Timer t;
    t.Restart();
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    parent_.resize(g.GetNumberOfVertices());

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS(g, v, marked, parent_);
    });

    // Set vertex labels for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      // g.SetVertexLabel(v, parent_[v]);
      g.SetVertexPayload(v, {g.GetVertexDeviate(v), g.GetVertexLabel(parent_[v]), rank_});
    });
  }

  void FindHighDegreeVertices(GraphAccess &g) {
    std::vector<VertexID> local_vertices;
    std::vector<VertexID> local_degrees;
    // TODO: Might be too small
    int num_local_vertices = 0;

    // Gather local high degree vertices
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID degree = g.GetVertexDegree(v);
      if (degree >= config_.degree_limit) {
        local_vertices.push_back(g.GetGlobalID(v));
        local_degrees.push_back(g.GetVertexDegree(v));
        num_local_vertices++;
      }
    });

    // Gather number of high degrees via all-gather
    std::vector<int> num_vertices(size_);
    MPI_Allgather(&num_local_vertices, 1, MPI_INT,
                  &num_vertices[0], 1, MPI_INT,
                  MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displ(size_);
    // TODO: Might be too small
    int num_global_vertices = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ[i] = num_global_vertices;
      num_global_vertices += num_vertices[i];
    }

    // Distribute vertices/degrees using all-gather
    std::vector<VertexID> global_vertices(num_global_vertices);
    std::vector<VertexID> global_degrees(num_global_vertices);
    MPI_Allgatherv(&local_vertices[0], num_local_vertices, MPI_VERTEX,
                   &global_vertices[0], &num_vertices[0], &displ[0], MPI_VERTEX,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(&local_degrees[0], num_local_vertices, MPI_VERTEX,
                   &global_degrees[0], &num_vertices[0], &displ[0], MPI_VERTEX,
                   MPI_COMM_WORLD);
  }

  void RunContraction(GraphAccess &g) {
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (rank_ == ROOT) {
      if (iteration_ == 1)
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << "-" 
                  << " [ADD] " << global_vertices << std::endl;
      else
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << iteration_timer_.Elapsed() 
                  << " [ADD] " << global_vertices << std::endl;
    }
    iteration_timer_.Restart();

    // if (rank_ == ROOT) std::cout << "[STATUS] Find high degree" << std::endl;
    // FindHighDegreeVertices(g);
    
    // TODO: Alternative deviates
    std::exponential_distribution<LPFloat> distribution(config_.beta);
    std::mt19937
        generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
    // g.ForallVertices([&](const VertexID v) {
    g.ForallLocalVertices([&](const VertexID v) {
      // Set preliminary deviate
      g.SetParent(v, g.GetGlobalID(v));
      LPFloat weight = static_cast<LPFloat>(log2(global_vertices) / g.GetVertexDegree(v));
      g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
                             g.GetVertexLabel(v), g.GetVertexRoot(v)}, 
                         true);
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
    });
    g.SendAndReceiveGhostVertices();

    // // Draw exponential deviate per local vertex
    // std::exponential_distribution<LPFloat> distribution(config_.beta);
    // g.ForallVertices([&](const VertexID v) {
    //   // Set preliminary deviate
    //   std::mt19937
    //       generator(static_cast<unsigned int>(config_.seed + g.GetVertexLabel(v)
    //       + iteration_ * rng_offset_));
    //   if (g.IsLocal(v)) g.SetParent(v, g.GetGlobalID(v));

    //   // Weigh distribution towards high degree vertices
    //   // TODO: Test different weighing functions 
    //   // LPFloat weight = static_cast<LPFloat>(log2(g.GetNumberOfGlobalVertices()) / g.GetVertexDegree(v));
    //   LPFloat weight = 1.;
    //   g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
    //                          g.GetVertexLabel(v), g.GetVertexRoot(v)}, 
    //                      false);
    // #ifndef NDEBUG
    //   std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
    //             << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
    //             << std::endl;
    // #endif
    // });

    // if (rank_ == ROOT)
    //   std::cout << "[STATUS] |-- Pick deviates " 
    //             << "[TIME] " << iteration_timer_.Elapsed() << std::endl;

    VertexID exchange_rounds = 0;
    int converged_globally = 0;
    int local_iterations = 0;
    Timer round_timer;
    while (converged_globally == 0) {
      round_timer.Restart();
      int converged_locally = 1;

      // Perform update for local vertices
      g.ForallLocalVertices([&](VertexID v) {
        auto smallest_payload = g.GetVertexMessage(v);
        g.ForallNeighbors(v, [&](VertexID w) {
          // Store neighbor label
          if (g.GetVertexDeviate(w) + 1 < smallest_payload.deviate_ ||
              (g.GetVertexDeviate(w) + 1 == smallest_payload.deviate_ &&
                  g.GetVertexLabel(w) < smallest_payload.label_)) {
            g.SetParent(v, g.GetGlobalID(w));
            smallest_payload = {g.GetVertexDeviate(w) + 1, g.GetVertexLabel(w),
                                g.GetVertexRoot(w)};
            converged_locally = 0;
          }
        });
        g.SetVertexPayload(v, std::move(smallest_payload));
      });

      // Check if all PEs are done
      // if (++local_iterations % 6 == 0) {
        MPI_Allreduce(&converged_locally,
                      &converged_globally,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        exchange_rounds++;

        // Receive variates
        g.SendAndReceiveGhostVertices();
      // } 

      // if (rank_ == ROOT) 
      //   std::cout << "[STATUS] |--- Round finished " 
      //             << "[TIME] " << round_timer.Elapsed() << std::endl;
    }

    // Determine remaining active vertices
    g.BuildLabelShortcuts();
    g.ContractExponential();

    // Count remaining number of vertices
    global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
        RunSequentialCC(g);
      else RunContraction(g);
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
    g.GatherGraphOnRoot(vertices, num_vertices_per_pe, labels, edges);

    // Root computes labels
    if (rank_ == ROOT) {
      // Build vertex mapping 
      std::unordered_map<VertexID, int> vertex_map;
      std::unordered_map<int, VertexID> reverse_vertex_map;
      // TODO: Might be too small
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
      // TODO: Might be too small
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
    g.DistributeLabelsFromRoot(labels, num_vertices_per_pe);
  }
};

#endif