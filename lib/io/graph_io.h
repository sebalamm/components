/******************************************************************************
 * graph_io.h
 *
 * I/O class for reading/writing the graph acess data structure
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

#ifndef _GRAPH_IO_H_
#define _GRAPH_IO_H_

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "config.h"
#include "static_graph_access.h"

class GraphIO {
 public:
  GraphIO() = default;
  virtual ~GraphIO() = default;

  static StaticGraphAccess ReadDistributedEdgeList(Config &config, PEID rank,
                                                   PEID size, const MPI_Comm &comm,
                                                   auto &edge_list) {
    // Gather local edge lists (transpose)
    VertexID from = edge_list[0].first, to = edge_list[0].second;
    VertexID number_of_local_vertices = to - from + 1;
    edge_list.erase(begin(edge_list));

    // Count ghost vertices
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(-1);
    for (auto &edge : edge_list) {
      VertexID source = edge.first;
      VertexID target = edge.second;

      // Target ghost
      if (from > target || target > to) {
        if (ghost_vertices.find(target) == end(ghost_vertices)) {
            ghost_vertices.insert(target);
        } 
      } 
    }

    VertexID number_of_ghost_vertices = ghost_vertices.size();
    VertexID number_of_edges = edge_list.size();

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    // Build graph
    StaticGraphAccess G(rank, size);
    G.StartConstruct(number_of_local_vertices, 
                     number_of_ghost_vertices, 
                     number_of_edges,
                     from);

    G.SetOffsetArray(std::move(vertex_dist));

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      G.AddGhostVertex(v);
    }

    // Reserve local vertices
    // for (VertexID v = 0; v < number_of_local_vertices; ++v) {
    //   VertexID num_edges = num_edges_for_local_vertex[v];
    //   G.ReserveEdgesForVertex(v, num_edges);
    // }

    // Reserve ghost vertices
    // for (auto &kv : num_edges_for_ghost_vertex) {
    //   VertexID global_id = kv.first;
    //   VertexID num_edges = kv.second;
    //   VertexID local_id = G.AddGhostVertex(global_id);
    //   G.ReserveEdgesForVertex(local_id, num_edges);
    // }

    // Add local edges
    // for (VertexID v = 0; v < number_of_local_vertices; ++v) {

    std::sort(edge_list.begin(), edge_list.end(), [&](auto &left, auto &right) {
        VertexID offset = to + 1;
        VertexID left_source_distance = (left.first < from ? 
                                         left.first + offset : 
                                         left.first) - from;
        VertexID left_target_distance = (left.second < from ? 
                                          left.second + offset : 
                                          left.second) - from;

        VertexID right_source_distance = (right.first < from ? 
                                          right.first + offset : 
                                          right.first) - from;
        VertexID right_target_distance = (right.second < from ? 
                                          right.second + offset : 
                                          right.second) - from;

        return (left_source_distance < right_source_distance
                  || (left_source_distance == right_source_distance && left_target_distance < right_target_distance));
    });
    for (auto &edge : edge_list) {
      G.AddEdge(G.GetLocalID(edge.first), edge.second, size);
    }
    
    // if (rank == ROOT) G.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);

    // Add ghost edges
    // std::sort(edge_list.begin(), edge_list.end(), [](auto &left, auto &right) {
    //     return left.second < right.second;
    // });
    // for (auto &edge : edge_list) {
    //   if (from > edge.second || edge.second > to) {
    //     G.AddEdge(G.GetLocalID(edge.second), edge.first, size);
    //   }
    // }

    G.FinishConstruct();

    // if (rank == ROOT) G.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);

    return G;
  }

  static StaticGraphAccess ReadDistributedGraph(Config &config, PEID rank,
                                          PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);

    // open file for reading
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_vertices;
    EdgeID number_of_edges;

    std::getline(in, line);
    while (line[0] == '%') std::getline(in, line);

    std::stringstream ss(line);
    ss >> number_of_vertices;
    ss >> number_of_edges;

    config.n = number_of_vertices;
    config.m = number_of_edges;

    // Read the lines i*ceil(n/size) to (i+1)*floor(n/size) lines of that file
    VertexID leftover_vertices = number_of_vertices % size;
    VertexID number_of_local_vertices = (number_of_vertices / size)
        + static_cast<VertexID>(rank < leftover_vertices);
    VertexID from = (rank * number_of_local_vertices)
        + static_cast<VertexID>(rank >= leftover_vertices ? leftover_vertices
                                                          : 0);
    VertexID to = from + number_of_local_vertices - 1;

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);
    std::cout << "rank " << rank << " from " << from << " to " << to
              << " amount " << number_of_local_vertices << std::endl;

    std::vector<std::vector<VertexID>> local_edge_lists;
    local_edge_lists.resize(number_of_local_vertices);

    VertexID counter = 0;
    VertexID node_counter = 0;
    EdgeID edge_counter = 0;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        old_str = &line[0];
        new_str = nullptr;

        for (;;) {
          VertexID target;
          target = (VertexID) strtol(old_str, &new_str, 10);

          if (target == 0) break;
          old_str = new_str;

          local_edge_lists[node_counter].push_back(target);
          edge_counter++;
        }

        node_counter++;
      }

      counter++;

      if (in.eof()) break;
    }

    MPI_Barrier(comm);

    StaticGraphAccess G(rank, size);
    // TODO: Add number of ghost vertices and reserve memory
    // G.StartConstruct(number_of_local_vertices, from);

    G.SetOffsetArray(std::move(vertex_dist));

    for (VertexID i = 0; i < number_of_local_vertices; ++i) {
      VertexID v = G.AddVertex();
      for (VertexID j : local_edge_lists[i])
        G.AddEdge(v, j - 1, size);
    }
    G.FinishConstruct();
    return G;
  }

 private:
};

#endif
