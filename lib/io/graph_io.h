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

#include <sys/sysinfo.h>

#include "config.h"
#include "utils.h"
#include "dynamic_graph_comm.h"
#include "semidynamic_graph.h"
#include "static_graph.h"
#include "static_graph_comm.h"

class GraphIO {
 public:
  GraphIO() = default;
  virtual ~GraphIO() = default;

  template<typename GraphOutputType>
  static void ReadStaticDistributedEdgeList(GraphOutputType &g,
                                            Config &config, PEID rank,
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

    if (rank == ROOT) std::cout << "done finding ghosts... mem " << Utility::GetFreePhysMem() << std::endl;

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
    g.StartConstruct(number_of_local_vertices, 
                     number_of_ghost_vertices, 
                     number_of_edges,
                     from);


    if (rank == ROOT) std::cout << "done start construct... mem " << Utility::GetFreePhysMem() << std::endl;

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    if (rank == ROOT) std::cout << "done adding ghosts... mem " << Utility::GetFreePhysMem() << std::endl;

    std::sort(edge_list.begin(), edge_list.end(), [&](auto &left, auto &right) {
        VertexID lhs_source = g.GetLocalID(left.first);
        VertexID lhs_target = g.GetLocalID(left.second);
        VertexID rhs_source = g.GetLocalID(right.first);
        VertexID rhs_target = g.GetLocalID(right.second);
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    if (rank == ROOT) std::cout << "done adding edges... mem " << Utility::GetFreePhysMem() << std::endl;

    g.FinishConstruct();
  }

  template<typename GraphOutputType>
  static void ReadDynamicDistributedEdgeList(GraphOutputType &g,
                                             Config &config, PEID rank,
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
    g.StartConstruct(number_of_local_vertices, 
                     number_of_ghost_vertices, 
                     from);

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphOutputType>
  static void ReadStaticDistributedFile(GraphOutputType &g, 
                                        Config &config, PEID rank,
                                        PEID size, const MPI_Comm &comm) {
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

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);
    
    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(-1);

    VertexID counter = 0;
    VertexID vertex_counter = 0;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        old_str = &line[0];
        new_str = nullptr;

        VertexID source = from + vertex_counter;
        for (;;) {
          VertexID target; 
          target = (VertexID) strtol(old_str, &new_str, 10);
          if (target == 0) break;
          old_str = new_str;
          // Decrement target to get proper range
          target--;
          // Add edges
          edge_list.emplace_back(source, target);
          if (from > target || target > to) {
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

    g.StartConstruct(number_of_local_vertices, 
                     ghost_vertices.size(), 
                     edge_list.size(),
                     from); 

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    std::sort(edge_list.begin(), edge_list.end(), [&](auto &left, auto &right) {
        VertexID lhs_source = g.GetLocalID(left.first);
        VertexID lhs_target = g.GetLocalID(left.second);
        VertexID rhs_source = g.GetLocalID(right.first);
        VertexID rhs_target = g.GetLocalID(right.second);
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphOutputType>
  static void ReadDynamicDistributedFile(GraphOutputType &g,
                                         Config &config, PEID rank,
                                         PEID size, const MPI_Comm &comm) {
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

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(-1);

    VertexID counter = 0;
    VertexID vertex_counter = 0;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        old_str = &line[0];
        new_str = nullptr;

        VertexID source = from + vertex_counter;
        for (;;) {
          VertexID target;
          target = (VertexID) strtol(old_str, &new_str, 10);
          if (target == 0) break;
          old_str = new_str;
          // Decrement target to get proper range
          target--;
          // Add edges
          edge_list.emplace_back(source, target);
          if (from > target || target > to) {
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

    g.StartConstruct(number_of_local_vertices, 
                     ghost_vertices.size(), 
                     from);

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphOutputType>
  static void ReadStaticFile(GraphOutputType &g, 
                             Config &config, PEID rank,
                             PEID size, const MPI_Comm &comm) {
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

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);
    
    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    google::dense_hash_set<VertexID> ghost_vertices; 
    ghost_vertices.set_empty_key(-1);

    VertexID counter = 0;
    VertexID vertex_counter = 0;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        old_str = &line[0];
        new_str = nullptr;

        VertexID source = from + vertex_counter;
        for (;;) {
          VertexID target; 
          target = (VertexID) strtol(old_str, &new_str, 10);
          if (target == 0) break;
          old_str = new_str;
          // Decrement target to get proper range
          target--;
          // Add edges
          edge_list.emplace_back(source, target);
          if (from > target || target > to) {
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

    g.StartConstruct(number_of_local_vertices, 
                     ghost_vertices.size(), 
                     edge_list.size(),
                     from); 

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, StaticGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    std::sort(edge_list.begin(), edge_list.end(), [&](auto &left, auto &right) {
        VertexID lhs_source = g.GetLocalID(left.first);
        VertexID lhs_target = g.GetLocalID(left.second);
        VertexID rhs_source = g.GetLocalID(right.first);
        VertexID rhs_target = g.GetLocalID(right.second);
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });
    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
  }

  template<typename GraphOutputType>
  static void ReadDynamicFile(GraphOutputType &g,
                              Config &config, PEID rank,
                              PEID size, const MPI_Comm &comm) {
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

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Gather vertex distribution
    std::pair<VertexID, VertexID> range(from, to + 1);
    std::vector<std::pair<VertexID, VertexID>> vertex_dist(size);
    MPI_Allgather(&range, 1, MPI_COMP,
                  &vertex_dist[0], 1, MPI_COMP, comm);

    std::vector<std::pair<VertexID, VertexID>> edge_list;
    // google::dense_hash_set<VertexID> ghost_vertices; 
    // ghost_vertices.set_empty_key(-1);
    google::sparse_hash_set<VertexID> ghost_vertices; 

    VertexID counter = 0;
    VertexID vertex_counter = 0;

    char *old_str, *new_str;
    while (std::getline(in, line)) {
      if (counter > to) break;
      if (line[0] == '%') continue;

      if (counter >= from) {
        old_str = &line[0];
        new_str = nullptr;

        VertexID source = from + vertex_counter;
        for (;;) {
          VertexID target;
          target = (VertexID) strtol(old_str, &new_str, 10);
          if (target == 0) break;
          old_str = new_str;
          // Decrement target to get proper range
          target--;
          // Add edges
          edge_list.emplace_back(source, target);
          if (from > target || target > to) {
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

    g.StartConstruct(number_of_local_vertices, 
                     ghost_vertices.size(), 
                     from);

    // Initialize local vertices
    if constexpr (std::is_same<GraphOutputType, DynamicGraphCommunicator>::value) {
      for (VertexID v = 0; v < number_of_local_vertices; v++) {
          g.SetVertexLabel(v, from + v);
          g.SetVertexRoot(v, rank);
      }
    }

    // Initialize ghost vertices
    for (auto &v : ghost_vertices) {
      g.AddGhostVertex(v, GetPEFromOffset(v, vertex_dist, rank));
    }

    for (auto &edge : edge_list) {
      g.AddEdge(g.GetLocalID(edge.first), edge.second, GetPEFromOffset(edge.second, vertex_dist, rank));
    }

    g.FinishConstruct();
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
};

#endif
