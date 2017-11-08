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

  static GraphAccess ReadDistributedGraph(const Config &config, PEID rank,
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

    // Read the lines i*ceil(n/size) to (i+1)*floor(n/size) lines of that file
    VertexID from = static_cast<VertexID>(rank * ceil(number_of_vertices / (double) size));
    VertexID to = std::min(
        (VertexID) ((rank + 1) * ceil(number_of_vertices / (double) size) - 1),
        number_of_vertices - 1);

    VertexID number_of_local_vertices = to - from + 1;
    std::cout << "rank " << rank << " from " << from << " to " << to
              << " amount " << number_of_local_vertices << std::endl;

    std::vector<std::vector<VertexID>> local_edge_lists;
    local_edge_lists.resize(number_of_local_vertices);

    VertexID counter = 0;
    VertexID node_counter = 0;
    EdgeID edge_counter = 0;

    char *oldstr, *newstr;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        oldstr = &line[0];
        newstr = 0;

        for (;;) {
          VertexID target;
          target = (VertexID) strtol(oldstr, &newstr, 10);

          if (target == 0) break;
          oldstr = newstr;

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
    for (PEID peid = 0; peid <= size; peid++)
      vertex_dist[peid] = static_cast<VertexID>(peid * ceil(number_of_vertices / (double) size));
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
