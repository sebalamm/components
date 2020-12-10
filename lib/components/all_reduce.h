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

#ifndef _ALL_REDUCE_H_
#define _ALL_REDUCE_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "utils.h"

template <typename GraphType>
class AllReduce {
 public:
  AllReduce(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0),
        num_vertices_per_pe_(size, 0) { }

  virtual ~AllReduce() { };

  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    contraction_timer_.Restart();
    // Init local data
    InitLocalData(g, g_labels);

    // Perform gather of graph on root 
    GatherGraphOnRoot(g);
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Gather on root took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    
    contraction_timer_.Restart();
    FindComponentsOnRoot();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Local computation on root took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart();
    // Distribute labels to other PEs
    DistributeLabelsFromRoot(g, g_labels);
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Distributing graph from root took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
  }

  void Output(GraphType &g) {
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

  // Vertex distribution
  std::vector<int> num_vertices_per_pe_;
  
  // Local information
  std::vector<VertexID> local_vertices_;
  std::vector<VertexID> local_labels_;
  std::vector<std::pair<VertexID, VertexID>> local_edges_;

  // Global information
  std::vector<VertexID> global_vertices_;
  std::vector<VertexID> global_labels_;
  std::vector<std::pair<VertexID, VertexID>> global_edges_;

  // Statistics
  Timer contraction_timer_;

  void FindComponentsOnRoot() {
    if (rank_ == ROOT) {
      // Build vertex mapping 
      google::dense_hash_map<VertexID, int> vertex_map; 
      vertex_map.set_empty_key(EmptyKey);
      vertex_map.set_deleted_key(DeleteKey);
      std::vector<VertexID> reverse_vertex_map(global_vertices_.size());
      int current_vertex = 0;
      for (const VertexID &v : global_vertices_) {
        vertex_map[v] = current_vertex;
        reverse_vertex_map[current_vertex++] = v;
      }

      // Build edge lists
      std::vector<std::vector<int>> edge_lists(global_vertices_.size());
      for (const auto &e : global_edges_) {
        edge_lists[vertex_map[e.first]].push_back(vertex_map[e.second]);
      }

      // Construct temporary graph
      StaticGraph sg(ROOT, 1);

      sg.StartConstruct(global_vertices_.size(), 0, global_edges_.size(), ROOT);
      for (int v = 0; v < global_vertices_.size(); ++v) {
        for (const int &e : edge_lists[v]) {
          sg.AddEdge(v, e, ROOT);
        }
      }
      sg.FinishConstruct();
      FindLocalComponents(sg, global_labels_);

    }
  }

  void FindLocalComponents(StaticGraph &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = global_labels_[v];
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS<StaticGraph>(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void InitLocalData(GraphType &g, std::vector<VertexID> g_labels) {
    g.ForallLocalVertices([&](const VertexID &v) {
      local_vertices_.push_back(g.GetGlobalID(v));
      local_labels_.push_back(g_labels[v]);
      g.ForallNeighbors(v, [&](const VertexID &w) {
        local_edges_.emplace_back(g.GetGlobalID(v), g.GetGlobalID(w));
      });
    });
  }

  void GatherGraphOnRoot(GraphType &g) {
    // Gather local labels/vertices
    int num_local_vertices = local_vertices_.size();
    int num_local_edges = local_edges_.size();

    // Gather number of vertices/edges for each PE
    std::vector<int> num_edges(size_);
    MPI_Gather(&num_local_vertices, 1, MPI_INT, &num_vertices_per_pe_[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&num_local_edges, 1, MPI_INT, &num_edges[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displ_vertices(size_);
    std::vector<int> displ_edges(size_);
    int num_global_vertices = 0;
    int num_global_edges = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_vertices[i] = num_global_vertices;
      displ_edges[i] = num_global_edges;
      num_global_vertices += num_vertices_per_pe_[i];
      num_global_edges += num_edges[i];
    }

    // Build datatype for edge
    MPI_Datatype MPI_EDGE;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);
    
    // Gather vertices/edges and labels for each PE
    global_vertices_.resize(num_global_vertices);
    global_labels_.resize(num_global_vertices);
    global_edges_.resize(num_global_edges);
    MPI_Gatherv(&local_vertices_[0], num_local_vertices, MPI_VERTEX,
                &global_vertices_[0], &num_vertices_per_pe_[0], &displ_vertices[0], MPI_VERTEX,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_labels_[0], num_local_vertices, MPI_VERTEX,
                &global_labels_[0], &num_vertices_per_pe_[0], &displ_vertices[0], MPI_VERTEX,
                ROOT, MPI_COMM_WORLD);
    MPI_Gatherv(&local_edges_[0], num_local_edges, MPI_EDGE,
                &global_edges_[0], &num_edges[0], &displ_edges[0], MPI_EDGE,
                ROOT, MPI_COMM_WORLD);
  } 

  void DistributeLabelsFromRoot(GraphType &g, std::vector<VertexID> &g_labels) {
    // Compute displacements
    std::vector<int> displ_labels(size_);
    int num_global_labels = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_labels[i] = num_global_labels;
      num_global_labels += num_vertices_per_pe_[i];
    }

    // Scatter to other PEs
    int num_local_vertices = local_vertices_.size();
    MPI_Scatterv(&global_labels_[0], &num_vertices_per_pe_[0], &displ_labels[0], MPI_VERTEX, 
                 &local_labels_[0], num_local_vertices, MPI_VERTEX, 
                 ROOT, MPI_COMM_WORLD);

    for (int i = 0; i < num_local_vertices; ++i) {
      VertexID v = local_vertices_[i];
      g_labels[i] = local_labels_[i];
    }
  }

  void OutputStats(GraphType &g) {
    VertexID n = g.GatherNumberOfGlobalVertices();
    EdgeID m = g.GatherNumberOfGlobalEdges();

    // Determine min/maximum cut size
    EdgeID m_cut = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      std::cout << "TEMP "
                << "s=" << config_.seed << ", "
                << "p=" << size_  << ", "
                << "n=" << n << ", "
                << "m=" << m << ", "
                << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }
};

#endif
