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
#include "utils.h"
#include "comm_utils.h"
#include "dynamic_graph.h"
#include "dynamic_graph_comm.h"
#include "semidynamic_graph.h"
#include "semidynamic_graph_comm.h"
#include "static_graph.h"
#include "static_graph_comm.h"

typedef struct {
  VertexID first_vertex, rank;
} Boundary;

// Operation is inoutvec[i](remote and local) = invec[i](remote) op inoutvec[i](local)
void compare_boundary(Boundary *invec, Boundary *inoutvec, int *len, MPI_Datatype *dtype) {
  VertexID local_last_vertex = inoutvec[0].first_vertex;
  VertexID local_rank = inoutvec[0].rank;

  VertexID remote_last_vertex = invec[0].first_vertex;
  VertexID remote_rank = invec[0].rank;

  if (local_last_vertex != remote_last_vertex) {
    inoutvec[0].first_vertex = local_last_vertex;
    inoutvec[0].rank = local_rank;
  }
}

class GraphIO {
 public:
  GraphIO() = default;
  virtual ~GraphIO() = default;

  template<typename GraphType>
  static void ReadMETISGenerator(GraphType &g,
                                 Config &config, 
                                 PEID rank, PEID size, const MPI_Comm &comm,
                                 auto &edge_list) {
    // Gather local edge lists (transpose)
    VertexID from = edge_list[0].first, to = edge_list[0].second;
    VertexID number_of_local_vertices = to - from + 1;
    edge_list.erase(begin(edge_list));
    VertexID number_of_edges = edge_list.size();

    // Count ghost vertices
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);
    VertexID number_of_ghost_vertices 
      = DetermineGhostVertices(edge_list, from, to, ghost_vertices);
    if (rank == ROOT) std::cout << "done finding ghosts... mem " << Utility::GetFreePhysMem() << std::endl;

    // Gather number of global vertices
    VertexID number_of_global_vertices = 0;
    MPI_Allreduce(&number_of_local_vertices,
                  &number_of_global_vertices,
                  1,
                  MPI_VERTEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // Build graph
    // Static graphs also take the number of edges
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       number_of_ghost_vertices, 
                       number_of_edges,
                       from);
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       number_of_ghost_vertices, 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       number_of_ghost_vertices, 
                       from);
    }
    if (rank == ROOT) std::cout << "done start construct... mem " << Utility::GetFreePhysMem() << std::endl;

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(from + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    GatherPERanges(from, to, comm, vertex_dist);

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }
    if (rank == ROOT) std::cout << "done adding ghosts... mem " << Utility::GetFreePhysMem() << std::endl;

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }
    if (rank == ROOT) std::cout << "done adding edges... mem " << Utility::GetFreePhysMem() << std::endl;

    g.FinishConstruct();
  }

  template<typename GraphType>
  static void ReadPartitionedMETISFile(GraphType &g, 
                                       Config &config, 
                                       PEID rank, PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);

    // open file for reading
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_global_vertices = 0;
    EdgeID number_of_global_edges = 0;
    PEID number_of_partitions = 0;

    std::getline(in, line);
    while (line[0] == '%') std::getline(in, line);

    std::stringstream ss(line);
    ss >> number_of_global_vertices;
    ss >> number_of_global_edges;
    ss >> number_of_partitions;

    config.n = number_of_global_vertices;
    config.m = number_of_global_edges;

    PEID leftover_partitions = number_of_partitions % size;
    PEID number_of_local_partitions = (number_of_partitions / size)
        + static_cast<PEID>(rank < leftover_partitions);
    PEID from_partition = (rank * number_of_local_partitions)
        + static_cast<PEID>(rank >= leftover_partitions ? leftover_partitions : 0);
    PEID to_partition = from_partition + number_of_local_partitions - 1;

    // Build offset array
    VertexID vertices_in_partition;
    VertexID from, to;
    VertexID current_offset = 0;
    for (VertexID i = 0; i < number_of_partitions; ++i) {
      std::getline(in, line);
      std::stringstream ss(line);
      ss >> vertices_in_partition;
      if (i == from_partition) from = current_offset;
      current_offset += vertices_in_partition;
      if (i == to_partition) to = current_offset - 1;
    }
    VertexID number_of_local_vertices = to - from + 1;

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);

    ParseVertexFilestream(in, from, to, ghost_vertices, edge_list);

    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       edge_list.size(),
                       from); 
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       from); 
    }

    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    GatherPERanges(from, to, comm, vertex_dist);

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(from + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphType>
  static void ReadMETISFile(GraphType &g, 
                            Config &config, 
                            PEID rank, PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);

    // open file for reading
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_global_vertices = 0;
    EdgeID number_of_global_edges = 0;

    std::getline(in, line);
    while (line[0] == '%') std::getline(in, line);

    std::stringstream ss(line);
    ss >> number_of_global_vertices;
    ss >> number_of_global_edges;

    config.n = number_of_global_vertices;
    config.m = number_of_global_edges;

    // Read the lines i*ceil(n/size) to (i+1)*floor(n/size) lines of that file
    VertexID leftover_vertices = number_of_global_vertices % size;
    VertexID number_of_local_vertices = (number_of_global_vertices / size)
      + static_cast<VertexID>(rank < leftover_vertices);
    VertexID from = (rank * number_of_local_vertices)
      + static_cast<VertexID>(rank >= leftover_vertices ? leftover_vertices : 0);
    VertexID to = from + number_of_local_vertices - 1;

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);

    ParseVertexFilestream(in, from, to, ghost_vertices, edge_list);

    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       edge_list.size(),
                       from); 
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       from); 
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(from + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    GatherPERanges(from, to, comm, vertex_dist);

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }


    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphType>
  static void ReadSortedEdgeFile(GraphType &g, 
                                 Config &config, 
                                 PEID rank, PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);

    // open file for reading
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_global_vertices = 0;
    EdgeID number_of_global_edges = 0;

    std::getline(in, line);
    while (line[0] == '%') std::getline(in, line);

    std::stringstream ss(line);
    std::string type;
    ss >> type;
    ss >> number_of_global_vertices;
    ss >> number_of_global_edges;

    config.n = number_of_global_vertices;
    config.m = number_of_global_edges;

    // Read the lines i*ceil(m/size) to (i+1)*floor(m/size) lines of that file
    VertexID leftover_edges = number_of_global_edges % size;
    VertexID number_of_local_edges = (number_of_global_edges / size)
      + static_cast<VertexID>(rank < leftover_edges);
    VertexID from = (rank * number_of_local_edges)
      + static_cast<VertexID>(rank >= leftover_edges ? leftover_edges : 0);
    VertexID to = from + number_of_local_edges - 1;
    // std::cout << "R" << rank << " (edge) from " << from << " to " << to << std::endl;

    // Gather local edges
    std::vector<std::pair<VertexID, VertexID>> edge_list;
    VertexID first_vertex = std::numeric_limits<VertexID>::max();
    VertexID last_vertex = 0;
    std::pair<VertexID, VertexID> first_vertex_range 
      = {std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max()};
    ParseEdgeFilestream(in, from, to, first_vertex, last_vertex, first_vertex_range, edge_list);

    // Determine local and ghost vertices
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);

    VertexID number_of_ghost_vertices 
      = DetermineGhostVertices(edge_list, first_vertex, last_vertex, ghost_vertices);

    VertexID number_of_local_vertices = last_vertex - first_vertex + 1;

    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices,
                       ghost_vertices.size(),
                       edge_list.size(),
                       first_vertex); 
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       first_vertex); 
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(first_vertex + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, first_vertex + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 4, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::vector<std::tuple<VertexID, VertexID, VertexID, VertexID>> vertex_dist(size);
    std::tuple<VertexID, VertexID, VertexID, VertexID> local_dist(first_vertex, 
                                                                  last_vertex, 
                                                                  first_vertex_range.first, 
                                                                  first_vertex_range.second);
    MPI_Allgather(&local_dist, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    // std::cout << "R" << rank << " V(R)=[" << first_vertex << "," << last_vertex << "]" << std::endl;

    // Resulting mapping
    google::dense_hash_map<VertexID, 
                          std::vector<std::tuple<VertexID, 
                                                 VertexID, 
                                                 VertexID,
                                                 PEID>>> vertex_ranges;
    vertex_ranges.set_empty_key(EmptyKey);

    // Construct mapping: v -> list<[first edge, last edge], duplicate ID, pe>
    VertexID prev_range_last = std::get<1>(vertex_dist[0]);
    VertexID prev_vertex_range_first = std::get<2>(vertex_dist[0]);
    VertexID prev_vertex_range_last = std::get<3>(vertex_dist[0]);
    VertexID prev_last_pe = 0;
    for (PEID i = 1; i < vertex_dist.size(); ++i) {
      VertexID current_range_first = std::get<0>(vertex_dist[i]);
      VertexID current_range_last = std::get<1>(vertex_dist[i]);
      VertexID current_vertex_range_first = std::get<2>(vertex_dist[i]);
      VertexID current_vertex_range_last = std::get<3>(vertex_dist[i]);
      if (prev_range_last == current_range_last
          || (prev_range_last != current_range_last 
            && prev_range_last == current_range_first)) {
        if (vertex_ranges.find(prev_range_last) == vertex_ranges.end()) {
          vertex_ranges[prev_range_last].emplace_back(0, current_vertex_range_first - 1, prev_range_last, prev_last_pe);
          if (prev_last_pe != rank) {
            ghost_vertices.erase(prev_range_last);
            // std::cout << "R" << rank << " add ghost (dist gen init) v " << prev_range_last << " pe(v)=" << prev_last_pe << std::endl;
            g.AddGhostVertex(prev_range_last, prev_last_pe);
          }
        }
        VertexID duplicate_id = (4 * number_of_global_vertices) * (i + size);
        vertex_ranges[prev_range_last].emplace_back(current_vertex_range_first, current_vertex_range_last, duplicate_id, i);
        if (i != rank) {
          // std::cout << "R" << rank << " add ghost (dist gen) v " << duplicate_id << " pe(v)=" << i << std::endl;
          g.AddGhostVertex(duplicate_id, i);
          if (prev_last_pe == rank) {
            edge_list.emplace_back(duplicate_id, prev_range_last);
            edge_list.emplace_back(prev_range_last, duplicate_id);
          } 
        } else {
          if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                        || std::is_same<GraphType, StaticGraph>::value
                        || std::is_same<GraphType, SemidynamicGraphCommunicator>::value
                        || std::is_same<GraphType, SemidynamicGraph>::value) {
            // std::cout << "R" << rank << " add dupl v " << duplicate_id << " of " << prev_range_last << std::endl;
            g.AddDuplicateVertex(prev_range_last, duplicate_id);
          } else {
            // std::cout << "R" << rank << " add local v " << duplicate_id << std::endl;
            g.AddVertex(duplicate_id);
          }
          edge_list.emplace_back(duplicate_id, prev_range_last);
          edge_list.emplace_back(prev_range_last, duplicate_id);
        }
      }
      if (prev_range_last != current_range_last) {
        prev_range_last = current_range_last;
        prev_vertex_range_first = current_vertex_range_first;
        prev_vertex_range_last = current_vertex_range_last;
        prev_last_pe = i;
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      // Get PE for ghost
      PEID pe = rank;
      for (PEID i = 0; i < vertex_dist.size(); ++i) {
        if (v >= std::get<0>(vertex_dist[i]) && v <= std::get<1>(vertex_dist[i])) {
          pe = i;
        }
      }
      // std::cout << "R" << rank << " add ghost (regular) v " << v << " pe(v)=" << pe << std::endl;
      g.AddGhostVertex(v, pe);
    }

    // Update edges with data from duplicates
    for (auto &edge : edge_list) {
      VertexID source = std::numeric_limits<VertexID>::max();
      VertexID target = std::numeric_limits<VertexID>::max();
      PEID target_pe = std::numeric_limits<PEID>::max();
      // Find updated source
      // TODO: Replace with proper binary search
      if (vertex_ranges.find(edge.first) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.first]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.second >= first_edge && edge.second <= last_edge) {
            source = dupl_id;
            break;
          }
        }
      }
      // Find updated target
      if (vertex_ranges.find(edge.second) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.second]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.first >= first_edge && edge.first <= last_edge) {
            target = dupl_id;
            target_pe = pe;
            break;
          }
        }
      }

      // Properly update edge if not duplicates were found
      if (source == std::numeric_limits<VertexID>::max()) 
        source = edge.first;
      if (target == std::numeric_limits<VertexID>::max())
        target = edge.second;

      // Update edge
      edge.first = source;
      edge.second = target;
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    // Finally add edges
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, g.GetPE(g.GetLocalID(edge.second)));
    }

    g.FinishConstruct();
    // g.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
  }

  template<typename GraphType>
  static void ReadPartitionedSortedEdgeFile(GraphType &g, 
                                            Config &config, 
                                            PEID rank, PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);
    filename += "_" + std::to_string(rank);

    // open file for reading
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_global_vertices = 0;
    EdgeID number_of_global_edges = 0;

    std::getline(in, line);
    while (line[0] == '%') std::getline(in, line);

    std::stringstream ss(line);
    std::string type;
    ss >> type;
    ss >> number_of_global_vertices;
    ss >> number_of_global_edges;

    config.n = number_of_global_vertices;
    config.m = number_of_global_edges;

    // Gather local edges
    std::vector<std::pair<VertexID, VertexID>> edge_list;
    VertexID first_vertex = std::numeric_limits<VertexID>::max();
    VertexID last_vertex = 0;
    std::pair<VertexID, VertexID> first_vertex_range 
      = {std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max()};
    ParseEdgeFilestream(in, 0, std::numeric_limits<VertexID>::max(), first_vertex, last_vertex, first_vertex_range, edge_list);
    VertexID number_of_local_edges = edge_list.size();

    // Determine local and ghost vertices
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);

    VertexID number_of_ghost_vertices 
      = DetermineGhostVertices(edge_list, first_vertex, last_vertex, ghost_vertices);

    VertexID number_of_local_vertices = last_vertex - first_vertex + 1;

    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices,
                       ghost_vertices.size(),
                       edge_list.size(),
                       first_vertex); 
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       first_vertex); 
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(first_vertex + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, first_vertex + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 4, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::vector<std::tuple<VertexID, VertexID, VertexID, VertexID>> vertex_dist(size);
    std::tuple<VertexID, VertexID, VertexID, VertexID> local_dist(first_vertex, 
                                                                  last_vertex, 
                                                                  first_vertex_range.first, 
                                                                  first_vertex_range.second);
    MPI_Allgather(&local_dist, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    // std::cout << "R" << rank << " V(R)=[" << first_vertex << "," << last_vertex << "]" << std::endl;

    // Resulting mapping
    google::dense_hash_map<VertexID, 
                          std::vector<std::tuple<VertexID, 
                                                 VertexID, 
                                                 VertexID,
                                                 PEID>>> vertex_ranges;
    vertex_ranges.set_empty_key(EmptyKey);

    // Construct mapping: v -> list<[first edge, last edge], duplicate ID, pe>
    VertexID prev_range_last = std::get<1>(vertex_dist[0]);
    VertexID prev_vertex_range_first = std::get<2>(vertex_dist[0]);
    VertexID prev_vertex_range_last = std::get<3>(vertex_dist[0]);
    VertexID prev_last_pe = 0;
    for (PEID i = 1; i < vertex_dist.size(); ++i) {
      VertexID current_range_first = std::get<0>(vertex_dist[i]);
      VertexID current_range_last = std::get<1>(vertex_dist[i]);
      VertexID current_vertex_range_first = std::get<2>(vertex_dist[i]);
      VertexID current_vertex_range_last = std::get<3>(vertex_dist[i]);
      if (prev_range_last == current_range_last
          || (prev_range_last != current_range_last 
            && prev_range_last == current_range_first)) {
        if (vertex_ranges.find(prev_range_last) == vertex_ranges.end()) {
          vertex_ranges[prev_range_last].emplace_back(0, current_vertex_range_first - 1, prev_range_last, prev_last_pe);
          if (prev_last_pe != rank) {
            ghost_vertices.erase(prev_range_last);
            // std::cout << "R" << rank << " add ghost (dist gen init) v " << prev_range_last << " pe(v)=" << prev_last_pe << std::endl;
            g.AddGhostVertex(prev_range_last, prev_last_pe);
          }
        }
        VertexID duplicate_id = (4 * number_of_global_vertices) * (i + size);
        vertex_ranges[prev_range_last].emplace_back(current_vertex_range_first, current_vertex_range_last, duplicate_id, i);
        if (i != rank) {
          // std::cout << "R" << rank << " add ghost (dist gen) v " << duplicate_id << " pe(v)=" << i << std::endl;
          g.AddGhostVertex(duplicate_id, i);
          if (prev_last_pe == rank) {
            edge_list.emplace_back(duplicate_id, prev_range_last);
            edge_list.emplace_back(prev_range_last, duplicate_id);
          } 
        } else {
          if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                        || std::is_same<GraphType, StaticGraph>::value
                        || std::is_same<GraphType, SemidynamicGraphCommunicator>::value
                        || std::is_same<GraphType, SemidynamicGraph>::value) {
            // std::cout << "R" << rank << " add dupl v " << duplicate_id << " of " << prev_range_last << std::endl;
            g.AddDuplicateVertex(prev_range_last, duplicate_id);
          } else {
            // std::cout << "R" << rank << " add local v " << duplicate_id << std::endl;
            g.AddVertex(duplicate_id);
          }
          edge_list.emplace_back(duplicate_id, prev_range_last);
          edge_list.emplace_back(prev_range_last, duplicate_id);
        }
      }
      if (prev_range_last != current_range_last) {
        prev_range_last = current_range_last;
        prev_vertex_range_first = current_vertex_range_first;
        prev_vertex_range_last = current_vertex_range_last;
        prev_last_pe = i;
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      // Get PE for ghost
      PEID pe = rank;
      for (PEID i = 0; i < vertex_dist.size(); ++i) {
        if (v >= std::get<0>(vertex_dist[i]) && v <= std::get<1>(vertex_dist[i])) {
          pe = i;
        }
      }
      // std::cout << "R" << rank << " add ghost (regular) v " << v << " pe(v)=" << pe << std::endl;
      g.AddGhostVertex(v, pe);
    }

    // Update edges with data from duplicates
    for (auto &edge : edge_list) {
      VertexID source = std::numeric_limits<VertexID>::max();
      VertexID target = std::numeric_limits<VertexID>::max();
      PEID target_pe = std::numeric_limits<PEID>::max();
      // Find updated source
      // TODO: Replace with proper binary search
      if (vertex_ranges.find(edge.first) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.first]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.second >= first_edge && edge.second <= last_edge) {
            source = dupl_id;
            break;
          }
        }
      }
      // Find updated target
      if (vertex_ranges.find(edge.second) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.second]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.first >= first_edge && edge.first <= last_edge) {
            target = dupl_id;
            target_pe = pe;
            break;
          }
        }
      }

      // Properly update edge if not duplicates were found
      if (source == std::numeric_limits<VertexID>::max()) 
        source = edge.first;
      if (target == std::numeric_limits<VertexID>::max())
        target = edge.second;

      // Update edge
      edge.first = source;
      edge.second = target;
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    // Finally add edges
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, g.GetPE(g.GetLocalID(edge.second)));
    }

    g.FinishConstruct();
    // g.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
  }

  template<typename GraphType>
  static void ReadSortedBinaryFile(GraphType &g, 
                                   Config &config, 
                                   PEID rank, PEID size, const MPI_Comm &comm) {
    std::string line;
    std::string filename(config.input_file);

    // open file for reading
    std::ifstream in(filename.c_str(), std::ios::binary);

    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(0);
    }

    VertexID number_of_global_vertices = 0;
    EdgeID number_of_global_edges = 0;

    // Jump to beginning and get number of global vertices and edges
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(&number_of_global_vertices), sizeof(VertexID));
    in.seekg(sizeof(VertexID), std::ios::beg);
    in.read(reinterpret_cast<char*>(&number_of_global_edges), sizeof(EdgeID));

    config.n = number_of_global_vertices;
    config.m = number_of_global_edges;

    // Read the lines i*ceil(m/size) to (i+1)*floor(m/size) lines of that file
    VertexID leftover_edges = number_of_global_edges % size;
    VertexID number_of_local_edges = (number_of_global_edges / size)
      + static_cast<VertexID>(rank < leftover_edges);
    VertexID from = (rank * number_of_local_edges)
      + static_cast<VertexID>(rank >= leftover_edges ? leftover_edges : 0);
    VertexID to = from + number_of_local_edges - 1;

    // Gather local edges
    std::vector<std::pair<VertexID, VertexID>> edge_list(number_of_local_edges);
    auto header_offset = sizeof(VertexID) + sizeof(EdgeID);
    auto edge_offset = from * sizeof(VertexID) * 2;
    auto edge_buffer_size = number_of_local_edges * sizeof(VertexID) * 2;
    in.seekg(header_offset + edge_offset, std::ios::beg);
    in.read(reinterpret_cast<char*>(&edge_list[0]), edge_buffer_size);

    // Compute vertex ranges
    VertexID first_vertex = std::numeric_limits<VertexID>::max();
    VertexID last_vertex = 0;
    std::pair<VertexID, VertexID> first_vertex_range 
      = {std::numeric_limits<VertexID>::max(), std::numeric_limits<VertexID>::max()};
    for (EdgeID e = 0; e < edge_list.size(); ++e) {
      edge_list[e].first--;
      edge_list[e].second--;

      if (first_vertex == std::numeric_limits<VertexID>::max()) {
        first_vertex = edge_list[e].first;
      }
      last_vertex = edge_list[e].first;

      if (edge_list[e].first == first_vertex) {
        if (first_vertex_range.first == std::numeric_limits<VertexID>::max()) {
          first_vertex_range.first = edge_list[e].second;
        }
        first_vertex_range.second = edge_list[e].second;
      }
    }

    // Determine local and ghost vertices
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(EmptyKey);
    ghost_vertices.set_deleted_key(DeleteKey);

    VertexID number_of_ghost_vertices 
      = DetermineGhostVertices(edge_list, first_vertex, last_vertex, ghost_vertices);

    VertexID number_of_local_vertices = last_vertex - first_vertex + 1;

    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      g.StartConstruct(number_of_local_vertices,
                       ghost_vertices.size(),
                       edge_list.size(),
                       first_vertex); 
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       number_of_global_vertices);
    } else {
      g.StartConstruct(number_of_local_vertices, 
                       ghost_vertices.size(), 
                       first_vertex); 
    }

    // Add vertices for dynamic graphs
    if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraph>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.AddVertex(first_vertex + v);
      }
    }

    // Initialize payloads for graphs with communicator 
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, DynamicGraphCommunicator>::value
                  || std::is_same<GraphType, SemidynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, first_vertex + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 4, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::vector<std::tuple<VertexID, VertexID, VertexID, VertexID>> vertex_dist(size);
    std::tuple<VertexID, VertexID, VertexID, VertexID> local_dist(first_vertex, 
                                                                  last_vertex, 
                                                                  first_vertex_range.first, 
                                                                  first_vertex_range.second);
    MPI_Allgather(&local_dist, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    // Resulting mapping
    google::dense_hash_map<VertexID, 
                          std::vector<std::tuple<VertexID, 
                                                 VertexID, 
                                                 VertexID,
                                                 PEID>>> vertex_ranges;
    vertex_ranges.set_empty_key(EmptyKey);

    // Construct mapping: v -> list<[first edge, last edge], duplicate ID, pe>
    VertexID prev_range_last = std::get<1>(vertex_dist[0]);
    VertexID prev_vertex_range_first = std::get<2>(vertex_dist[0]);
    VertexID prev_vertex_range_last = std::get<3>(vertex_dist[0]);
    VertexID prev_last_pe = 0;
    for (PEID i = 1; i < vertex_dist.size(); ++i) {
      VertexID current_range_first = std::get<0>(vertex_dist[i]);
      VertexID current_range_last = std::get<1>(vertex_dist[i]);
      VertexID current_vertex_range_first = std::get<2>(vertex_dist[i]);
      VertexID current_vertex_range_last = std::get<3>(vertex_dist[i]);
      if (prev_range_last == current_range_last
          || (prev_range_last != current_range_last 
            && prev_range_last == current_range_first)) {
        if (vertex_ranges.find(prev_range_last) == vertex_ranges.end()) {
          vertex_ranges[prev_range_last].emplace_back(0, current_vertex_range_first - 1, prev_range_last, prev_last_pe);
          if (prev_last_pe != rank) {
            ghost_vertices.erase(prev_range_last);
            // std::cout << "R" << rank << " add ghost (dist gen init) v " << prev_range_last << " pe(v)=" << prev_last_pe << std::endl;
            g.AddGhostVertex(prev_range_last, prev_last_pe);
          }
        }
        VertexID duplicate_id = (4 * number_of_global_vertices) * (i + size);
        vertex_ranges[prev_range_last].emplace_back(current_vertex_range_first, current_vertex_range_last, duplicate_id, i);
        if (i != rank) {
          // std::cout << "R" << rank << " add ghost (dist gen) v " << duplicate_id << " pe(v)=" << i << std::endl;
          g.AddGhostVertex(duplicate_id, i);
          if (prev_last_pe == rank) {
            edge_list.emplace_back(duplicate_id, prev_range_last);
            edge_list.emplace_back(prev_range_last, duplicate_id);
          } 
        } else {
          if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                        || std::is_same<GraphType, StaticGraph>::value
                        || std::is_same<GraphType, SemidynamicGraphCommunicator>::value
                        || std::is_same<GraphType, SemidynamicGraph>::value) {
            // std::cout << "R" << rank << " add dupl v " << duplicate_id << " of " << prev_range_last << std::endl;
            g.AddDuplicateVertex(prev_range_last, duplicate_id);
          } else {
            // std::cout << "R" << rank << " add local v " << duplicate_id << std::endl;
            g.AddVertex(duplicate_id);
          }
          edge_list.emplace_back(duplicate_id, prev_range_last);
          edge_list.emplace_back(prev_range_last, duplicate_id);
        }
      }
      if (prev_range_last != current_range_last) {
        prev_range_last = current_range_last;
        prev_vertex_range_first = current_vertex_range_first;
        prev_vertex_range_last = current_vertex_range_last;
        prev_last_pe = i;
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      // Get PE for ghost
      PEID pe = rank;
      for (PEID i = 0; i < vertex_dist.size(); ++i) {
        if (v >= std::get<0>(vertex_dist[i]) && v <= std::get<1>(vertex_dist[i])) {
          pe = i;
        }
      }
      // std::cout << "R" << rank << " add ghost (regular) v " << v << " pe(v)=" << pe << std::endl;
      g.AddGhostVertex(v, pe);
    }

    // Update edges with data from duplicates
    for (auto &edge : edge_list) {
      VertexID source = std::numeric_limits<VertexID>::max();
      VertexID target = std::numeric_limits<VertexID>::max();
      PEID target_pe = std::numeric_limits<PEID>::max();
      // Find updated source
      // TODO: Replace with proper binary search
      if (vertex_ranges.find(edge.first) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.first]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.second >= first_edge && edge.second <= last_edge) {
            source = dupl_id;
            break;
          }
        }
      }
      // Find updated target
      if (vertex_ranges.find(edge.second) != vertex_ranges.end()) {
        for (auto &range : vertex_ranges[edge.second]) {
          VertexID first_edge = std::get<0>(range);
          VertexID last_edge = std::get<1>(range);
          VertexID dupl_id = std::get<2>(range);
          PEID pe = std::get<3>(range);
          if (edge.first >= first_edge && edge.first <= last_edge) {
            target = dupl_id;
            target_pe = pe;
            break;
          }
        }
      }

      // Properly update edge if not duplicates were found
      if (source == std::numeric_limits<VertexID>::max()) 
        source = edge.first;
      if (target == std::numeric_limits<VertexID>::max())
        target = edge.second;

      // Update edge
      edge.first = source;
      edge.second = target;
    }

    // Sort edges for static graphs
    if constexpr (std::is_same<GraphType, StaticGraphCommunicator>::value
                  || std::is_same<GraphType, StaticGraph>::value) {
      SortEdges<GraphType>(g, edge_list);
    }

    // Finally add edges
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, g.GetPE(g.GetLocalID(edge.second)));
    }

    g.FinishConstruct();
    // g.OutputLocal();
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
  }

  static PEID GetPEFromOffset(const VertexID v, 
                              std::vector<std::pair<VertexID, VertexID>> offset_array,
                              PEID default_rank) {
    for (PEID i = 0; i < offset_array.size(); ++i) {
      if (v >= offset_array[i].first && v < offset_array[i].second) {
        return i;
      }
    }
    return default_rank;
  }

 private:

  static VertexID DetermineGhostVertices(auto &edge_list, 
                                         VertexID local_from, VertexID local_to, 
                                         google::dense_hash_set<VertexID> &ghost_vertices) {
    for (auto &edge : edge_list) {
      VertexID source = edge.first;
      VertexID target = edge.second;

      // Target ghost
      if (target < local_from || target > local_to) {
        if (ghost_vertices.find(target) == ghost_vertices.end()) {
          ghost_vertices.insert(target);
        } 
      } 
    }
    return ghost_vertices.size();
  }

  static void ParseVertexFilestream(std::ifstream &in, 
                                    VertexID local_from, VertexID local_to, 
                                    google::dense_hash_set<VertexID> &ghost_vertices,
                                    std::vector<std::pair<VertexID, VertexID>> &edge_list) {
    VertexID counter = 0;
    VertexID vertex_counter = 0;

    std::string line;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > local_to) break;
      if (line[0] == '%') continue;

      if (counter >= local_from) {
        old_str = &line[0];
        new_str = nullptr;

        VertexID source = local_from + vertex_counter;
        for (;;) {
          VertexID target; 
          target = (VertexID) strtol(old_str, &new_str, 10);
          if (target == 0) break;
          old_str = new_str;
          // Decrement target to get proper range
          target--;
          // Add edges
          edge_list.emplace_back(source, target);
          if (local_from > target || target > local_to) {
            if (ghost_vertices.find(target) == end(ghost_vertices)) {
                ghost_vertices.insert(target);
            } 
            // We need the backwards edge here
            edge_list.emplace_back(target, source);
          } 
        }
        vertex_counter++;
      }
      counter++;
      if (in.eof()) break;
    }
  }

  static void ParseEdgeFilestream(std::ifstream &in, 
                                  VertexID local_from, VertexID local_to, 
                                  VertexID &first_local_vertex, VertexID &last_local_vertex,
                                  std::pair<VertexID, VertexID> &first_local_vertex_range,
                                  std::vector<std::pair<VertexID, VertexID>> &edge_list) {
    VertexID edge_counter = 0;

    std::string line;
    while (std::getline(in, line)) {
      if (edge_counter > local_to) break;
      if (line[0] == '%') continue;

      if (edge_counter >= local_from) {
        std::stringstream ss(line);
        std::string type;
        VertexID source = 0; 
        VertexID target = 0;
        ss >> type;
        ss >> source;
        ss >> target;

        // Decrement (1,n) to get proper range (0,n-1)
        source--;
        target--;

        if (first_local_vertex == std::numeric_limits<VertexID>::max()) {
          first_local_vertex = source;
        }
        last_local_vertex = source;

        // Add edge(s)
        edge_list.emplace_back(source, target);
        edge_list.emplace_back(target, source);

        if (source == first_local_vertex) {
          if (first_local_vertex_range.first == std::numeric_limits<VertexID>::max()) {
          first_local_vertex_range.first = target;
          } 
          first_local_vertex_range.second = target;
        }
      }
      edge_counter++;
      if (in.eof()) break;
    }
  }

  static void GatherPERanges(VertexID local_from, VertexID local_to, 
                             const MPI_Comm &comm,
                             std::vector<std::pair<VertexID, VertexID>> &pe_ranges) {
    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> local_range(local_from, local_to + 1);
    MPI_Allgather(&local_range, 1, MPI_COMP,
                  &pe_ranges[0], 1, MPI_COMP, comm);
  }

  template <typename GraphType>
  static void SortEdges(const GraphType &g, auto &edge_list) {
    std::sort(edge_list.begin(), edge_list.end(), [&](auto &left, auto &right) {
        VertexID lhs_source = g.GetLocalID(left.first);
        VertexID lhs_target = g.GetLocalID(left.second);
        VertexID rhs_source = g.GetLocalID(right.first);
        VertexID rhs_target = g.GetLocalID(right.second);
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });
  }
};

#endif
