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
#include "base_graph_access.h"
#include "initial_contraction.h"
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
        iteration_(0) { }

  virtual ~ExponentialContraction() {
    delete exp_contraction_;
    exp_contraction_ = nullptr;
  };

  void FindComponents(BaseGraphAccess &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    FindLocalComponents(g, g_labels);
    
    // First round of contraction
    InitialContraction<BaseGraphAccess> 
      first_contraction(g, g_labels, rank_, size_);
    BaseGraphAccess cag 
      = first_contraction.ReduceBaseGraph();

    // Delete original graph?
    // Keep contraction labeling for later

    std::vector<VertexID> cag_labels(g.GetNumberOfVertices(), 0);
    FindLocalComponents(cag, cag_labels);

    // cag.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);

    // Second round of contraction
    InitialContraction<BaseGraphAccess> 
      second_contraction(cag, cag_labels, rank_, size_);
    GraphAccess ccag 
      = second_contraction.BuildComponentAdjacencyGraph();

    // ccag.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);

    // Delete intermediate graph?
    // Keep contraction labeling for later
    exp_contraction_ = new GraphContraction(ccag, rank_, size_);

    // Main decomposition algorithm
    PerformDecomposition(ccag);

    // CCAG -> CAG (labels)
    ApplyToLocalComponents(ccag, cag, cag_labels);
    // CAG (labels) -> G (labels)
    ApplyToLocalComponents(cag, cag_labels, g, g_labels);
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

  // Statistics
  Timer iteration_timer_;
  
  // Contraction
  GraphContraction *exp_contraction_;

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
    exp_contraction_->UndoContraction();
  }

  void FindLocalComponents(BaseGraphAccess &g, std::vector<VertexID> & label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<BaseGraphAccess>::BFS(g, v, marked, parent);
    });

    // Set vertex labe for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
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
    // TODO: Number of vertices seems correct so something is probably wrong with backwards edges
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
      LPFloat weight = 
#ifdef TIEBREAK_DEGREE
        // static_cast<LPFloat>(g.GetVertexDegree(v) / g.GetMaxDegree());
        // static_cast<LPFloat>(log2(g.GetNumberOfVertices()) / g.GetVertexDegree(v));
        1.0;
#else
        1.0;
#endif
      g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
                             g.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)}, 
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

    if (rank_ == ROOT)
      std::cout << "[STATUS] |-- Pick deviates " 
                << "[TIME] " << iteration_timer_.Elapsed() << std::endl;

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
#ifdef TIEBREAK_DEGREE
                  g.GetVertexDegree(w) < smallest_payload.degree_)) {
#else 
                  g.GetVertexLabel(w) < smallest_payload.label_)) {
#endif
            g.SetParent(v, g.GetGlobalID(w));
            smallest_payload = {g.GetVertexDeviate(w) + 1, 
                                g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                // TODO: Retrieve degree from non-local vertices
                                g.GetVertexDegree(w),
#endif
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

      if (rank_ == ROOT) 
        std::cout << "[STATUS] |--- Round finished " 
                  << "[TIME] " << round_timer.Elapsed() << std::endl;
    }

    // Determine remaining active vertices
    g.BuildLabelShortcuts();
    exp_contraction_->ExponentialContraction();

    // Count remaining number of vertices
    global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
        RunSequentialCC(g);
      else RunContraction(g);
    }
  }

  void ApplyToLocalComponents(GraphAccess &cag, BaseGraphAccess &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(BaseGraphAccess &cag, std::vector<VertexID> &cag_label, BaseGraphAccess &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
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
      BaseGraphAccess sg(ROOT, 1);

      sg.StartConstruct(vertices.size(), 0, edges.size(), ROOT);
      for (int v = 0; v < vertices.size(); ++v) {
        // sg.ReserveEdgesForVertex(v, edge_lists[v].size());
        for (const int &e : edge_lists[v]) 
          sg.AddEdge(v, e, ROOT);
      }
      sg.FinishConstruct();
      FindLocalComponents(sg, labels);
    }

    // Distribute labels to other PEs
    g.DistributeLabelsFromRoot(labels, num_vertices_per_pe);
  }
};

#endif
