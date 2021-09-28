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

#include "components/local_contraction.h"

int main(int argn, char **argv) {
  // Init MPI
  MPI_Init(&argn, &argv);
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read command-line args
  Config conf;
  ParseParameters(argn, argv, conf);
  if (rank == ROOT) PrintParameters(conf);
  int initial_seed = conf.seed;

#ifndef NWARMUP
  if (rank == ROOT) std::cout << "WARMUP RUN" << std::endl;
  {
    if (conf.use_contraction) {
      StaticGraph G(conf, rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      MPI_Barrier(MPI_COMM_WORLD);

      LocalContraction comp(conf, rank, size);
      comp.FindComponents(G, labels);
    } else {
      DynamicGraphCommunicator G(conf, rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);
      G.ResetCommunicator();

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      MPI_Barrier(MPI_COMM_WORLD);

      LocalContraction comp(conf, rank, size);
      comp.FindComponents(G, labels);
    }
  }
#endif

  // ACTUAL RUN
  if (rank == ROOT) std::cout << "BENCH RUN" << std::endl;
  Statistics stats;
  
  for (int i = 0; i < conf.iterations; ++i) {
    int round_seed = initial_seed + i + 1000;
    conf.seed = round_seed;

    Timer t;
    double local_time = 0.0;
    double total_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);

    if (conf.use_contraction) {
      t.Restart();
      StaticGraph G(conf, rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);
      if (i == 0) IOUtility::PrintGraphParams(G, conf, rank, size);

      // Reset timers
      G.ResetCommTime();
      G.ResetSendVolume();
      G.ResetReceiveVolume();

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      local_time = t.Elapsed();
      std::cout << "IO rank=" << rank << " time=" << local_time << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);

      t.Restart();
      LocalContraction comp(conf, rank, size);
      comp.FindComponents(G, labels);
      local_time = t.Elapsed();
      
      // Print labels
      G.OutputComponents(labels);
    } else {
      t.Restart();
      DynamicGraphCommunicator G(conf, rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);
      if (i == 0) IOUtility::PrintGraphParams(G, conf, rank, size);
      G.ResetCommunicator();

      // Reset timers
      G.ResetCommTime();
      G.ResetSendVolume();
      G.ResetReceiveVolume();

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      local_time = t.Elapsed();
      std::cout << "IO rank=" << rank << " time=" << local_time << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);

      t.Restart();
      LocalContraction comp(conf, rank, size);
      comp.FindComponents(G, labels);
      local_time = t.Elapsed();
      
      // Print labels
      G.OutputComponents(labels);
    }

    // Gather total time
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) stats.Push(total_time);
  }

  if (rank == ROOT) {
    std::cout << "RESULT runner=local"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

