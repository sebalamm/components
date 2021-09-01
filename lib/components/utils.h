/******************************************************************************
 * utils.h
 *
 * Utility algorithms needed for connected components
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

#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <queue>
#ifdef MEMINFO
#include <sys/sysinfo.h>
#endif

#include "dynamic_graph_comm.h"
#include "semidynamic_graph.h"
#include "static_graph.h"
#include "static_graph_comm.h"

class Utility {
 public:
  template <typename GraphType>
  static void BFS(GraphType &g,
                  const VertexID &start,
                  std::vector<bool> &marked,
                  std::vector<VertexID> &parent) {
    // Standard BFS
    std::queue<VertexID> q;
    q.push(start);
    marked[start] = true;
    parent[start] = start;
    while (!q.empty()) {
      VertexID v = q.front();
      q.pop();
      g.ForallNeighbors(v, [&](const VertexID &w) {
        if (!marked[w]) {
          q.push(w);
          marked[w] = true;
          parent[w] = start;
        }
      });
    }
  }

  template <typename GraphType>
  static void BFS(GraphType &g,
                  const VertexID &start,
                  google::dense_hash_map<VertexID, bool> &marked,
                  google::dense_hash_map<VertexID, VertexID> &parent) {
    // Standard BFS
    std::queue<VertexID> q;
    q.push(start);
    marked[start] = true;
    parent[start] = start;
    while (!q.empty()) {
      VertexID v = q.front();
      q.pop();
      g.ForallNeighbors(v, [&](const VertexID &w) {
        if (!marked[w]) {
          q.push(w);
          marked[w] = true;
          parent[w] = start;
        }
      });
    }
  }

  template <typename GraphType>
  static VertexID ComputeAverageMaxDegree(GraphType &g,
                                          const PEID rank, const PEID size) {
    // Determine local max degree
    VertexID local_max_deg = 0;
    g.ForallLocalVertices([&](const VertexID v) {
        VertexID v_deg = g.GetVertexDegree(v);
        if (v_deg > local_max_deg) {
          local_max_deg = v_deg; 
        }
    });

    // Get global average max degree
    VertexID global_max_deg = 0;
    MPI_Allreduce(&local_max_deg, &global_max_deg, 
                  1, MPI_VERTEX, 
                  MPI_SUM, MPI_COMM_WORLD);
    return global_max_deg / size; 
  }

  template <typename GraphType>
  static void SelectHighDegreeVertices(GraphType &g, 
                                       VertexID degree_threshold,
                                       std::vector<std::pair<VertexID, VertexID>> &high_degree_vertices) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID v_deg = g.GetVertexDegree(v);
      if (v_deg >= degree_threshold) {
        high_degree_vertices.emplace_back(v, v_deg);
      }
    });
  }

  static long long GetFreePhysMem() {
#ifdef MEMINFO
    struct sysinfo memInfo;
    sysinfo (&memInfo);
    long long totalPhysMem = memInfo.totalram;
    long long freePhysMem = memInfo.freeram;

    totalPhysMem *= memInfo.mem_unit;
    freePhysMem *= memInfo.mem_unit;
    totalPhysMem *= 1e-9;
    freePhysMem *= 1e-9;

    return freePhysMem;
#else
    return 0;
#endif
  } 
};

#endif
