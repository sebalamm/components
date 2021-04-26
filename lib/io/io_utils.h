/******************************************************************************
 * io_utils.h
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

#ifndef _IO_UTILITY_H_
#define _IO_UTILITY_H_

#include "config.h"
#include "definitions.h"
#include "io/graph_io.h"
#include "kagen_interface.h"

class IOUtility {
 public:
  template<typename GraphType>
  static void LoadGraph(GraphType &g, 
                        Config &config,
                        PEID rank, PEID size) {
    // File I/O
    if (config.input_type == "file") {
      GraphIO::ReadMETISFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "edgefile") {
      GraphIO::ReadSortedEdgeFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "partition") {
      GraphIO::ReadPartitionedSortedEdgeFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "binary") {
      GraphIO::ReadSortedBinaryFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.gen != "null") {
      // Generator I/O
      kagen::KaGen gen(rank, size);
      kagen::EdgeList edge_list;
      GenerateSyntheticGraph(gen, config, edge_list);
      GraphIO::ReadMETISGenerator<GraphType>(g, config, rank, size, MPI_COMM_WORLD, edge_list);
      edge_list.clear();
    } else {
      std::cout << "I/O type not supported" << std::endl;
      MPI_Finalize();
      exit(1);
    }
  }

  static void GenerateSyntheticGraph(kagen::KaGen &gen,
                                     Config &config,
                                     kagen::EdgeList &edge_list) {
      if (config.gen == "gnm_undirected")
          edge_list = gen.GenerateUndirectedGNM(config.gen_n, config.gen_m, config.gen_k, config.seed);
      else if (config.gen == "rdg_2d")
          edge_list = gen.Generate2DRDG(config.gen_n, config.gen_k, config.seed);
      else if (config.gen == "rdg_3d")
          edge_list = gen.Generate3DRDG(config.gen_n, config.gen_k, config.seed);
      else if (config.gen == "rgg_2d")
          edge_list = gen.Generate2DRGG(config.gen_n, config.gen_r, config.gen_k, config.seed);
      else if (config.gen == "rgg_3d")
          edge_list = gen.Generate3DRGG(config.gen_n, config.gen_r, config.gen_k, config.seed);
      else if (config.gen == "rhg")
          edge_list = gen.GenerateRHG(config.gen_n, config.gen_gamma, config.gen_d, config.gen_k, config.seed);
      else if (config.gen == "ba")
          edge_list = gen.GenerateBA(config.gen_n, config.gen_d, config.gen_k, config.seed);
      else if (config.gen == "grid_2d")
          edge_list = gen.Generate2DGrid(config.gen_n, config.gen_m, config.gen_p, config.gen_periodic, config.gen_k, config.seed);
      else {
        std::cout << "Generator not supported" << std::endl;
        MPI_Finalize();
        exit(1);
      }
  }

  template<typename GraphType>
  static void PrintGraphParams(GraphType &g,
                               Config &config,
                               PEID rank, PEID size) {
    VertexID n = g.GatherNumberOfGlobalVertices();
    EdgeID m = g.GatherNumberOfGlobalEdges();

    VertexID highest_degree = 0;
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetVertexDegree(v) > highest_degree) {
        highest_degree = g.GetVertexDegree(v);
      }
    });

    // Determine min/maximum cut size
    EdgeID m_cut = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (rank == ROOT) {
      std::cout << "INPUT "
                << "s=" << config.seed << ", "
                << "p=" << size  << ", "
                << "n=" << n << ", "
                << "m=" << m << ", "
                << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
    std::cout << "R" << rank << " LOCAL INPUT" << " c=" << m_cut << ", hd=" << highest_degree << std::endl;
  }
};

#endif
