/******************************************************************************
 * components.cpp
 *
 * Main application
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

#include <mpi.h>

#include "config.h"
#include "benchmark.h"
#include "io/graph_io.h"
#include "parse_parameters.h"
#include "timer.h"
#include "kagen_interface.h"

#include "components/exponential_contraction.h"

int main(int argn, char **argv) {
  // Init MPI
  MPI_Init(&argn, &argv);
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read command-line args
  Config conf;
  ParseParameters(argn, argv, conf);
  
  StaticGraphAccess G(rank, size);
  if (conf.input_file != "null") {
    // File I/O
    GraphIO::ReadStaticDistributedFile(G, conf, rank, size, MPI_COMM_WORLD);
  } else if (conf.gen != "null") {
    // Generator I/O
    kagen::KaGen gen(rank, size);
    kagen::EdgeList edge_list;
    if (conf.gen == "gnm_undirected")
        edge_list = gen.GenerateUndirectedGNM(conf.gen_n, conf.gen_m, conf.gen_k);
    else if (conf.gen == "rdg_2d")
        edge_list = gen.Generate2DRDG(conf.gen_n, conf.gen_k);
    else if (conf.gen == "rdg_3d")
        edge_list = gen.Generate3DRDG(conf.gen_n, conf.gen_k);
    else if (conf.gen == "rgg_2d")
        edge_list = gen.Generate2DRGG(conf.gen_n, conf.gen_r, conf.gen_k);
    else if (conf.gen == "rgg_3d")
        edge_list = gen.Generate3DRGG(conf.gen_n, conf.gen_r, conf.gen_k);
    else if (conf.gen == "rhg")
        edge_list = gen.GenerateRHG(conf.gen_n, conf.gen_gamma, conf.gen_d, conf.gen_k);
    else if (conf.gen == "ba")
        edge_list = gen.GenerateBA(conf.gen_n, conf.gen_d, conf.gen_k);
    else {
      if (rank == ROOT) 
        std::cout << "Generator not supported" << std::endl;
      MPI_Finalize();
      exit(1);
    }
    if (rank == ROOT) std::cout << "Graph generated" << std::endl;
    GraphIO::ReadStaticDistributedEdgeList(G, conf, rank, size, MPI_COMM_WORLD, edge_list);
  } else {
    if (rank == ROOT) 
      std::cout << "I/O type not supported" << std::endl;
    MPI_Finalize();
    exit(1);
  }

  VertexID n = G.GatherNumberOfGlobalVertices();
  EdgeID m = G.GatherNumberOfGlobalEdges();

  // Determine min/maximum cut size
  EdgeID m_cut = G.GetNumberOfCutEdges();
  EdgeID min_cut, max_cut;
  MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
             MPI_COMM_WORLD);
  MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
             MPI_COMM_WORLD);

  if (rank == ROOT) {
    std::cout << "INPUT "
              << "s=" << conf.seed << ", "
              << "p=" << size  << ", "
              << "n=" << n << ", "
              << "m=" << m << ", "
              << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
  }

  // Timers
  Timer t;
  Statistics stats;

  double local_time = 0.0;
  double total_time = 0.0;

  int user_seed = conf.seed;
  std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
  for (int i = 0; i < conf.iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    t.Restart();

    // Determine labels
    conf.seed = user_seed + i;
    ExponentialContraction comp(conf, rank, size);
    comp.FindComponents(G, labels);

    // Gather total time
    local_time = t.Elapsed();
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) stats.Push(total_time);
    
    // Print labels
    G.OutputComponents(labels);
  }

  if (rank == ROOT) {
    std::cout << "RESULT runner=exp"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
