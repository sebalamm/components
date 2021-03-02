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
#include "io/io_utils.h"
#include "parse_parameters.h"
#include "timer.h"

#include "components/all_reduce.h"

int main(int argn, char **argv) {
  // Init MPI
  MPI_Init(&argn, &argv);
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read command-line args
  Config conf;
  ParseParameters(argn, argv, conf);
  int initial_seed = conf.seed;

  StaticGraph G(rank, size);
  IOUtility::LoadGraph(G, conf, rank, size);
  IOUtility::PrintGraphParams(G, conf, rank, size);

  // WARMUP RUN
  if (rank == ROOT) std::cout << "WARMUP RUN" << std::endl;

  {
    // Determine labels
    std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
    G.ForallLocalVertices([&](const VertexID v) {
      labels[v] = G.GetGlobalID(v);
    });
    AllReduce<StaticGraph> ar(conf, rank, size);
    ar.FindComponents(G, labels);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ACTUAL RUN
  if (rank == ROOT) std::cout << "BENCH RUN" << std::endl;

  Statistics stats;
  
  for (int i = 0; i < conf.iterations; ++i) {
    int round_seed = initial_seed + i + 1000;
    conf.seed = round_seed;

    Timer t;
    double local_time = 0.0;
    double total_time = 0.0;

    std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
    G.ForallLocalVertices([&](const VertexID v) {
      labels[v] = G.GetGlobalID(v);
    });
    t.Restart();

    // Determine labels
    AllReduce<StaticGraph> ar(conf, rank, size);
    ar.FindComponents(G, labels);

    // Gather total time
    local_time = t.Elapsed();
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) stats.Push(total_time);
    
    // Print labels
    G.OutputComponents(labels);
  }

  if (rank == ROOT) {
    std::cout << "RESULT runner=allreduce"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

