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

#ifndef _REDUCE_CONTRACTION_H_
#define _REDUCE_CONTRACTION_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include <sys/sysinfo.h>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "dynamic_graph_access.h"
#include "static_graph_access.h"
#include "cag_builder.h"
#include "dynamic_contraction.h"
#include "utils.h"
#include "union_find.h"
#include "propagation.h"

template <typename GraphType>
class ReduceContraction {
 public:
  ReduceContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0),
        num_vertices_per_pe_(size, 0) { }

  virtual ~ReduceContraction() { };

  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    contraction_timer_.Restart();

    // Store original graph for later resolution
    GraphType og = g;

    // Perform successive reductions
    bool is_active = true;
    for (PEID i = 0; i < ceil(log2(size_)); i++) {
      if (!is_active) continue;
      if (IsBitSet(rank_, i)) {
        std::cout << "Send R" << rank_ << " to R" << rank_ - pow(2,i) << " (round " << i << ")"<< std::endl;
        SendGraph(g, g_labels, rank_ - pow(2,i));
      } else if (rank_ + pow(2,i) < size_) {
        std::cout << "Reduce R" << rank_ << " with R" << rank_ + pow(2,i) << " (round " << i << ")"<< std::endl;
        ReceiveGraph(g, g_labels, rank_ + pow(2,i));
      }
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Exchange round took " 
          << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }
    }

    // DistributeLabelsFromRoot(og, g_labels);
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

  bool IsBitSet(PEID val, int bit) {
    return 1 == ((val >> bit) & 1);
  }

  void SendGraph(GraphType &g, std::vector<VertexID> &g_labels, PEID target) {
    std::vector<VertexID> local_vertices;
    std::vector<VertexID> local_labels;
    std::vector<std::pair<VertexID,VertexID>> local_edges;
    g.ForallLocalVertices([&](const VertexID &v) {
      local_vertices.push_back(g.GetGlobalID(v));
      local_labels.push_back(g_labels[v]);
      g.ForallNeighbors(v, [&](const VertexID &w) {
        local_edges.emplace_back(g.GetGlobalID(v), g.GetGlobalID(w));
      });
    });

    // Send graph data
    // Vertices
    std::cout << "R" << rank_ << " send " << local_vertices.size() << " vertices to " << target << std::endl;
    MPI_Send(&local_vertices[0],
             static_cast<int>(local_vertices.size()),
             MPI_VERTEX, target, target + 6 * size_, MPI_COMM_WORLD);

    // Labels
    std::cout << "R" << rank_ << " send " << local_labels.size() << " labels to " << target << std::endl;
    MPI_Send(&local_labels[0],
             static_cast<int>(local_labels.size()),
             MPI_VERTEX, target, target + 6 * size_, MPI_COMM_WORLD);

    // Edges
    std::cout << "R" << rank_ << " send " << local_edges.size() << " edges to " << target << std::endl;
    MPI_Send(&local_edges[0],
             static_cast<int>(local_edges.size()),
             MPI_VERTEX, target, target + 6 * size_, MPI_COMM_WORLD);

  }

  void ReceiveGraph(GraphType& g, std::vector<VertexID> &g_labels, PEID sender) {
    // Receive graph data
    MPI_Status st{};
    MPI_Status rst{};
    int message_length;

    // Vertices
    MPI_Probe(MPI_ANY_SOURCE, sender + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);

    std::cout << "R" << rank_ << " recv " << message_length << " vertices from " << sender << std::endl;
    std::vector<VertexID> message(static_cast<unsigned long>(message_length));
    MPI_Recv(&message[0], message_length,
             MPI_VERTEX, st.MPI_SOURCE,
             rank_ + 6 * size_, MPI_COMM_WORLD, &rst);

    // Labels
    MPI_Probe(MPI_ANY_SOURCE, sender + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);

    std::cout << "R" << rank_ << " recv " << message_length << " labels from " << sender << std::endl;
    message.resize(static_cast<unsigned long>(message_length));
    MPI_Recv(&message[0], message_length,
             MPI_VERTEX, st.MPI_SOURCE,
             rank_ + 6 * size_, MPI_COMM_WORLD, &rst);

    // Edges
    MPI_Probe(MPI_ANY_SOURCE, sender + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);

    std::cout << "R" << rank_ << " recv " << message_length << " edges from " << sender << std::endl;
    message.resize(static_cast<unsigned long>(message_length));
    MPI_Recv(&message[0], message_length,
             MPI_VERTEX, st.MPI_SOURCE,
             rank_ + 6 * size_, MPI_COMM_WORLD, &rst);
  }

  // void FindComponentsOnRoot() {
  //   if (rank_ == ROOT) {
  //     // Build vertex mapping 
  //     google::dense_hash_map<VertexID, int> vertex_map; 
  //     vertex_map.set_empty_key(-1);
  //     google::dense_hash_map<int, VertexID> reverse_vertex_map; 
  //     reverse_vertex_map.set_empty_key(-1);
  //     int current_vertex = 0;
  //     for (const VertexID &v : global_vertices_) {
  //       vertex_map[v] = current_vertex;
  //       reverse_vertex_map[current_vertex++] = v;
  //     }

  //     // Build edge lists
  //     std::vector<std::vector<int>> edge_lists(global_vertices_.size());
  //     for (const auto &e : global_edges_) 
  //       edge_lists[vertex_map[e.first]].push_back(vertex_map[e.second]);

  //     // Construct temporary graph
  //     StaticGraphAccess sg(ROOT, 1);

  //     sg.StartConstruct(global_vertices_.size(), 0, global_edges_.size(), ROOT);
  //     for (int v = 0; v < global_vertices_.size(); ++v) {
  //       for (const int &e : edge_lists[v]) 
  //         sg.AddEdge(v, e, ROOT);
  //     }
  //     sg.FinishConstruct();
  //     FindLocalComponents(sg, global_labels_);
  //   }
  // }

  // void FindLocalComponents(StaticGraphAccess &g, std::vector<VertexID> &label) {
  //   std::vector<bool> marked(g.GetNumberOfVertices(), false);
  //   std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

  //   g.ForallVertices([&](const VertexID v) {
  //     label[v] = g.GetGlobalID(v);
  //   });

  //   // Compute components
  //   g.ForallLocalVertices([&](const VertexID v) {
  //     if (!marked[v]) Utility<StaticGraphAccess>::BFS(g, v, marked, parent);
  //   });

  //   // Set vertex label for contraction
  //   g.ForallLocalVertices([&](const VertexID v) {
  //     label[v] = label[parent[v]];
  //   });
  // }

  // void InitLocalData(GraphType &g, std::vector<VertexID> g_labels) {
  // }

  // void DistributeLabelsFromRoot(GraphType &g, std::vector<VertexID> &g_labels) {
  //   // Compute displacements
  //   std::vector<int> displ_labels(size_);
  //   int num_global_labels = 0;
  //   for (PEID i = 0; i < size_; ++i) {
  //     displ_labels[i] = num_global_labels;
  //     num_global_labels += num_vertices_per_pe_[i];
  //   }

  //   // Scatter to other PEs
  //   int num_local_vertices = local_vertices_.size();
  //   MPI_Scatterv(&global_labels_[0], &num_vertices_per_pe_[0], &displ_labels[0], MPI_VERTEX, 
  //                &local_labels_[0], num_local_vertices, MPI_VERTEX, 
  //                ROOT, MPI_COMM_WORLD);

  //   for (int i = 0; i < num_local_vertices; ++i) {
  //     VertexID v = local_vertices_[i];
  //     g_labels[i] = local_labels_[i];
  //   }
  // }

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

  static long long GetFreePhysMem() {
    struct sysinfo memInfo;
    sysinfo (&memInfo);
    long long totalPhysMem = memInfo.totalram;
    long long freePhysMem = memInfo.freeram;

    totalPhysMem *= memInfo.mem_unit;
    freePhysMem *= memInfo.mem_unit;
    totalPhysMem *= 1e-9;
    freePhysMem *= 1e-9;

    return freePhysMem;
  } 
};

#endif
