/******************************************************************************
 * bloom_reduce.h
 *
 * Distributed bloom filter merging via hypercube reduce
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

#ifndef _CONTRACTION_REDUCE_
#define _CONTRACTION_REDUCE_

#include <iostream>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "graph_access.h"

class ContractionReduce {
 public:
  ContractionReduce(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf) {}
  virtual ~ContractionReduce() = default;

  void FindComponents(GraphAccess &g) {
    Timer t;
    t.Restart()
    if (rank_ == ROOT) std::cout << "[STATUS] Find local components (" << t.Elapsed() << ")" << std::endl;
    FindLocalComponents(g);
    if (rank_ == ROOT) std::cout << "[STATUS] Contract local components (" << t.Elapsed() << ")" << std::endl;
    Contraction cont(g, rank_, size_);
    GraphAccess cag = cont.BuildComponentAdjacencyGraph();
    if (rank_ == ROOT) std::cout << "[STATUS] Perform main algorithm (" << t.Elapsed() << ")" << std::endl;
    PerformHypercubeReduce(cag);
    if (rank_ == ROOT) std::cout << "[STATUS] Apply labels (" << t.Elapsed() << ")" << std::endl;
    ApplyToLocalComponents(cag, g);
  }

  void Output(GraphAccess &g) { }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Local components
  std::vector<VertexID> parent;

  void FindLocalComponents(GraphAccess &g) {
    Timer t;
    t.Restart();
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    parent.resize(g.GetNumberOfVertices());

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

  void PerformHypercubeReduce(GraphAccess &g) {
    if (rank_ == ROOT) std::cout << "[STATUS] |- Start hypercube reduce" << std::endl;
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices < config_.sequential_limit) {
        if (rank_ == ROOT) 
          std::cout << "[STATUS] |-- Perform sequential computation (n=" 
                    << global_vertices << ")" << std::endl;
        RunSequentialCC(g);
      }
      else RunReduceCC(g);
    }
    if (rank_ == ROOT) std::cout << "[STATUS] |- Propagate labels upward" << std::endl;
    // PropagateLabelsUp(g);
  }

  void RunReduceCC(GraphAccess &g) {
    PEID dims = math.ceil(log2(size_));
    for (PEID d = 0; d < dims; ++d) {
      // Gather components of local graph
      std::vector<std::tuple<VertexID, VertexID, VertexID>> vertex_buffer;
      std::vector<std::tuple<VertexID, VertexID, VertexID>> edge_buffer;
      ForallLocalVertices([&](const VertexID &v) {
        label_buffer.emplace_back(v, GetVertexLabel(v));
        ForallNeighbors(v, [&](const VertexID &w) {
          edge_buffer.emplace_back(GetGlobalID(v), GetGlobalID(w));
        });
      });

      // Build datatype for edge
      MPI_Datatype MPI_EDGE;
      MPI_Type_vector(1, 2, 0, MPI_LONG, &MPI_EDGE);
      MPI_Type_commit(&MPI_EDGE);

      // Send local graph to neighbor in current dimension
      PEID target_rank = rank_ & ~((PEID)1 << d);
      MPI_Request;
      MPI_Isend(&edge_buffer[0], edge_buffer.size(), MPI_EDGE, target_rank, 
                0, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);

      // Receive neighbor graph and add to current graph
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Status st{};
      int flag = 1;

      // TODO: Implement subgraph merging
      // g.AddSubgraph(...);
      
      // Compute updated labels (remove duplicate edges and non-leader vertices)
    }
  }

  VertexID ComputeComponentPrefixSum(GraphAccess &g) {
    // Gather local components O(max(#component))
    VertexID num_local_components = 0, num_global_components = 0;
    std::unordered_set<VertexID> local_components;
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID v_label = g.GetVertexLabel(v);
      if (local_components.find(v_label) == end(local_components)) {
        local_components.insert(v_label);
        num_local_components++;
      }
    });

    // Build prefix sum over local components O(log P)
    VertexID component_prefix_sum;
    MPI_Scan(&num_local_components,
             &component_prefix_sum,
             1,
             MPI_LONG,
             MPI_SUM,
             MPI_COMM_WORLD);

    num_global_components = component_prefix_sum;
    MPI_Bcast(&num_global_components,
              1,
              MPI_LONG,
              size_ - 1,
              MPI_COMM_WORLD);

    return component_prefix_sum - num_local_components;
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
    g.DistributeLabelsFromRoot(labels, num_vertices_per_pe);
  }
};

#endif

