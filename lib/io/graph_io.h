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
    VertexID num_vertices = (number_of_vertices / size) + static_cast<VertexID>(rank < leftover_vertices);
    VertexID from = (rank * num_vertices) + static_cast<VertexID>(rank >= leftover_vertices ? leftover_vertices : 0);
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
      VertexID num_vertices_for_pe = (number_of_vertices / size) + static_cast<VertexID>(pe_id < leftover_vertices);
      vertex_dist[pe_id] = static_cast<VertexID>((pe_id * num_vertices_for_pe)
          + static_cast<VertexID>(num_vertices_for_pe >= leftover_vertices ? leftover_vertices : 0));
    }
    G.SetOffsetArray(std::move(vertex_dist));

    for (VertexID i = 0; i < number_of_local_vertices; ++i) {
      VertexID v = G.AddVertex();
      G.SetVertexLabel(v, from + v);

      for (VertexID j : local_edge_lists[i]) {
        G.AddEdge(v, j - 1, size);
      }
    }
    G.FinishConstruct();
    MPI_Barrier(comm);
    std::cout << "done construction" << std::endl;

    return G;
  }

 private:
};

#endif
