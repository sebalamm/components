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

#include "components/shortcut_propagation.h"

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

  StaticGraph SG(conf, rank, size);
  StaticGraphCommunicator SGC(conf, rank, size);
  MPI_Barrier(MPI_COMM_WORLD);
  if (conf.use_contraction) {
      IOUtility::LoadGraph(SG, conf, rank, size);
      IOUtility::PrintGraphParams(SG, conf, rank, size);
  } else {
      IOUtility::LoadGraph(SGC, conf, rank, size);
      IOUtility::PrintGraphParams(SGC, conf, rank, size);
  }

  // WARMUP RUN
  if (rank == ROOT) std::cout << "WARMUP RUN" << std::endl;
  {
    ShortcutPropagation comp(conf, rank, size);
    if (conf.use_contraction) {
      StaticGraph CG = SG;

      // Determine labels
      std::vector<VertexID> labels(CG.GetNumberOfVertices(), 0);
      CG.ForallLocalVertices([&](const VertexID v) {
        labels[v] = CG.GetGlobalID(v);
      });
      comp.FindComponents(CG, labels);
    } else {
      StaticGraphCommunicator CG = SGC;
      CG.ResetCommunicator();

      // Determine labels
      std::vector<VertexID> labels(CG.GetNumberOfVertices(), 0);
      CG.ForallLocalVertices([&](const VertexID v) {
        labels[v] = CG.GetGlobalID(v);
      });
      comp.FindComponents(CG, labels);
    }
  }

  // ACTUAL RUN
  if (rank == ROOT) std::cout << "BENCH RUN" << std::endl;

  Statistics stats;
  Statistics global_stats;
  
  for (int i = 0; i < conf.iterations; ++i) {
    int round_seed = initial_seed + i + 1000;
    conf.seed = round_seed;

    Timer t;
    double local_time = 0.0;
    double total_time = 0.0;

    ShortcutPropagation comp(conf, rank, size);

    if (conf.use_contraction) {
      StaticGraph CG = SG;

      // Reset timers
      CG.ResetCommTime();
      CG.ResetSendVolume();
      CG.ResetReceiveVolume();

      // Determine labels
      std::vector<VertexID> labels(CG.GetNumberOfVertices(), 0);
      CG.ForallLocalVertices([&](const VertexID v) {
        labels[v] = CG.GetGlobalID(v);
      });
      t.Restart();
      comp.FindComponents(CG, labels);
      local_time = t.Elapsed();

      // Print labels
      CG.OutputComponents(labels);
    } else {
      StaticGraphCommunicator CG = SGC;
      CG.ResetCommunicator();

      // Reset timers
      CG.ResetCommTime();
      CG.ResetSendVolume();
      CG.ResetReceiveVolume();

      // Determine labels
      std::vector<VertexID> labels(CG.GetNumberOfVertices(), 0);
      CG.ForallLocalVertices([&](const VertexID v) {
        labels[v] = CG.GetGlobalID(v);
      });
      t.Restart();
      comp.FindComponents(CG, labels);
      local_time = t.Elapsed();

      // Print labels
      CG.OutputComponents(labels);
    }

    // Gather total time
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    stats.Push(local_time);
    if (rank == ROOT) global_stats.Push(total_time);
  }

  std::cout << "LOCAL RESULT rank=" << rank << " runner=exp"
            << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
            << " iterations=" << conf.iterations << std::endl;
  if (rank == ROOT) {
    std::cout << "GLOBAL RESULT runner=exp"
              << " time=" << global_stats.Avg() << " stddev=" << global_stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

