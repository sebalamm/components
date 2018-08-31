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
#include "graph_access.h"

class GraphIO {
 public:
  GraphIO() = default;
  virtual ~GraphIO() = default;

  static GraphAccess ReadDistributedEdgeList(Config &config, PEID rank,
                                             PEID size, const MPI_Comm &comm,
                                             auto &edge_list) {
    // Gather local edge lists (transpose)
    VertexID from = edge_list[0].first, to = edge_list[0].second;
    VertexID number_of_local_vertices = to - from + 1;
    edge_list.erase(begin(edge_list));

    EdgeID number_of_local_edges = 0;
    std::vector<std::vector<VertexID>> local_edge_lists(number_of_local_vertices);
    for (auto &edge : edge_list) {
      VertexID source = edge.first;
      VertexID target = edge.second;
      if (from <= source && source <= to) 
        local_edge_lists[source - from].emplace_back(target);
      if (from <= target && target <= to) 
        local_edge_lists[target - from].emplace_back(source);
      number_of_local_edges++;
    }

    // Gather vertex distribution
    VertexID next_from = to + 1;
    std::vector<VertexID> vertex_dist(size, 0);
    MPI_Allgather(&next_from, 1, MPI_LONG,
                  &vertex_dist[0], 1, MPI_LONG, comm);
    vertex_dist.insert(begin(vertex_dist), 0);
    std::cout << "rank " << rank << " from " << from << " to " << to
              << " amount " << number_of_local_vertices << std::endl;

    // TODO: Reshuffling here probably won't work since we have 
    // to make changes to our entire data structure.
    // Better to shuffle how we read the input file (like Tom).
    // // Determine random target for each local vertex
    // srand(rank);
    // std::vector<std::vector<VertexID>> vertices_for_pe(size);
    // for (VertexID v = 0; v < number_of_local_vertices; ++v) {
    //   PEID target = rand() % size;
    //   vertices_for_pe[target].emplace_back(from + v);
    // }

    // // Compute send displacements and actual message
    // {
    //   VertexID displ = 0;
    //   std::vector<VertexID> msg;
    //   std::vector<int> msg_size(size, 0);
    //   std::vector<int> vertex_displ(size, 0);
    //   for (PEID pe = 0; pe < size; ++pe) {
    //     vertex_displ[pe] = displ;
    //     for (VertexID & v : vertices_for_pe[pe]) msg.emplace_back(v);
    //     msg_size[pe] = vertices_for_pe[pe].size();
    //     displ += vertices_for_pe[pe].size();
    //   }

    //   // Exchange displacements
    //   std::vector<int> global_msg_size(size, 0);
    //   MPI_Alltoall(msg_size.data(), 1, MPI_INT, 
    //                global_msg_size.data(), 1, MPI_INT,
    //                MPI_COMM_WORLD);

    //   // Compute receive displacements
    //   VertexID global_displ = 0;
    //   std::vector<int> global_vertex_displ(size, 0);
    //   for (PEID pe = 0; pe < size; ++pe) {
    //     global_vertex_displ[pe] = global_displ;
    //     global_displ += global_msg_size[pe];
    //   }

    //   // Send/receive vertices
    //   std::vector<VertexID> global_msg(global_displ, 0);
    //   MPI_Alltoallv(msg.data(), msg_size.data(), vertex_displ.data(), MPI_LONG, 
    //                 global_msg.data(), global_msg_size.data(), global_vertex_displ.data(), MPI_LONG,
    //                 MPI_COMM_WORLD);
    // }

    // // Compute send displacements and actual message
    // {
    //   VertexID displ = 0;
    //   std::vector<VertexID> msg;
    //   std::vector<int> msg_size(size, 0);
    //   std::vector<int> edge_displ(size, 0);
    //   for (PEID pe = 0; pe < size; ++pe) {
    //     edge_displ[pe] = displ;
    //     for (VertexID & v : vertices_for_pe[pe]) {
    //       for (VertexID & w : local_edge_lists[v - from]) {
    //         msg.emplace_back(w);
    //       }
    //       msg_size[pe] += local_edge_lists[v - from].size();
    //       displ += local_edge_lists[v - from].size();
    //     }
    //   }

    //   // Exchange displacements
    //   std::vector<int> global_msg_size(size, 0);
    //   MPI_Alltoall(msg_size.data(), 1, MPI_INT, 
    //                global_msg_size.data(), 1, MPI_INT,
    //                MPI_COMM_WORLD);

    //   // Compute receive displacements
    //   VertexID global_displ = 0;
    //   std::vector<int> global_edge_displ(size, 0);
    //   for (PEID pe = 0; pe < size; ++pe) {
    //     global_edge_displ[pe] = global_displ;
    //     global_displ += global_msg_size[pe];
    //   }

    //   // Send/receive vertices
    //   std::vector<VertexID> global_msg(global_displ, 0);
    //   MPI_Alltoallv(msg.data(), msg_size.data(), edge_displ.data(), MPI_LONG, 
    //                 global_msg.data(), global_msg_size.data(), global_edge_displ.data(), MPI_LONG,
    //                 MPI_COMM_WORLD);
    // }

    // Build graph
    GraphAccess G(rank, size);
    G.StartConstruct(number_of_local_vertices, 2 * number_of_local_edges, from);

    G.SetOffsetArray(std::move(vertex_dist));

    for (VertexID i = 0; i < number_of_local_vertices; ++i) {
      VertexID v = G.AddVertex();
      G.SetVertexPayload(v, {G.GetVertexDeviate(v), from + v, rank});

      for (VertexID w : local_edge_lists[v]) 
          G.AddEdge(v, w, size);
    }

    G.FinishConstruct();

    return G;
  }

  static GraphAccess ReadDistributedGraph(Config &config, PEID rank,
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
    VertexID num_vertices = (number_of_vertices / size)
        + static_cast<VertexID>(rank < leftover_vertices);
    VertexID from = (rank * num_vertices)
        + static_cast<VertexID>(rank >= leftover_vertices ? leftover_vertices
                                                          : 0);
    VertexID to = from + num_vertices - 1;

    VertexID number_of_local_vertices = to - from + 1;
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

    GraphAccess G(rank, size);
    G.StartConstruct(number_of_local_vertices, 2 * edge_counter, from);

    std::vector<VertexID> vertex_dist(static_cast<unsigned long>(size + 1), 0);
    for (PEID pe_id = 1; pe_id <= size; pe_id++) {
      VertexID num_vertices_for_pe = (number_of_vertices / size)
          + static_cast<VertexID>(pe_id < leftover_vertices);
      vertex_dist[pe_id] = static_cast<VertexID>((pe_id * num_vertices_for_pe)
          + static_cast<VertexID>(pe_id >= leftover_vertices
                                  ? leftover_vertices : 0));
    }
    G.SetOffsetArray(std::move(vertex_dist));

    for (VertexID i = 0; i < number_of_local_vertices; ++i) {
      VertexID v = G.AddVertex();
      // G.SetVertexLabel(v, from + v);
      G.SetVertexPayload(v, {G.GetVertexDeviate(v), from + v, rank});

      for (VertexID j : local_edge_lists[i])
        G.AddEdge(v, j - 1, size);
    }
    G.FinishConstruct();
    MPI_Barrier(comm);

    return G;
  }

 private:
};

#endif
