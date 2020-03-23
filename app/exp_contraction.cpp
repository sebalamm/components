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
  int initial_seed = conf.seed;

  Timer io_t;
  io_t.Restart();

  StaticGraph SG(rank, size);
  DynamicGraphCommunicator DG(rank, size);
  if (conf.use_contraction) {
      IOUtility::LoadGraph(SG, conf, rank, size);
      IOUtility::PrintGraphParams(SG, conf, rank, size);
  } else {
      IOUtility::LoadGraph(DG, conf, rank, size);
      IOUtility::PrintGraphParams(DG, conf, rank, size);
  }
  if (rank == ROOT) std::cout << "[INFO] I/O took time=" << io_t.Elapsed() << std::endl;

  // WARMUP RUN
  if (rank == ROOT) std::cout << "WARMUP RUN" << std::endl;

  {
    // Use static graph for contraction
    ExponentialContraction comp(conf, rank, size);
    if (conf.use_contraction) {
      StaticGraph G(rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);
      IOUtility::PrintGraphParams(G, conf, rank, size);

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      comp.FindComponents(G, labels);
    } else {
      DynamicGraphCommunicator G(rank, size);
      IOUtility::LoadGraph(G, conf, rank, size);
      IOUtility::PrintGraphParams(G, conf, rank, size);

      // Determine labels
      std::vector<VertexID> labels(G.GetNumberOfVertices(), 0);
      G.ForallLocalVertices([&](const VertexID v) {
        labels[v] = G.GetGlobalID(v);
      });
      comp.FindComponents(G, labels);
    }
  }

  // ACTUAL RUN
  if (rank == ROOT) std::cout << "BENCH RUN" << std::endl;
  Statistics stats;
  Statistics comm_stats;

  for (int i = 0; i < conf.iterations; ++i) {
    int round_seed = initial_seed + i + 1000;
    conf.seed = round_seed;

    Timer t;
    float local_time = 0.0;
    float total_time = 0.0;

    ExponentialContraction comp(conf, rank, size);
    if (conf.use_contraction) {
      io_t.Restart();
      StaticGraph CG = SG;
      if (rank == ROOT) std::cout << "[INFO] copy took time=" << io_t.Elapsed() << std::endl;

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
      io_t.Restart();
      DynamicGraphCommunicator CG = DG;
      if (rank == ROOT) std::cout << "[INFO] copy took time=" << io_t.Elapsed() << std::endl;

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
    MPI_Reduce(&local_time, &total_time, 1, MPI_FLOAT, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) stats.Push(total_time);

    // Gather comm time
    float comm_time = comp.GetCommTime();
    float total_comm_time = 0.0;
    MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_FLOAT, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) comm_stats.Push(total_comm_time);
  }

  if (rank == ROOT) {
    std::cout << "RESULT runner=exp"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " comm_time=" << comm_stats.Avg() << " stddev=" << comm_stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

