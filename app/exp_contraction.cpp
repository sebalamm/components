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
  if (rank == ROOT) PrintParameters(conf);
  int initial_seed = conf.seed;

#ifndef NWARMUP
  if (rank == ROOT) std::cout << "WARMUP RUN" << std::endl;
  {
    StaticGraph G(conf, rank, size);
    IOUtility::LoadGraph(G, conf, rank, size);

    // Determine labels
    std::vector<VertexID> labels(G.GetVertexVectorSize(), 0);
    G.ForallLocalVertices([&](const VertexID v) {
      labels[v] = G.GetGlobalID(v);
    });
    MPI_Barrier(MPI_COMM_WORLD);

    ExponentialContraction comp(conf, rank, size);
    comp.FindComponents(G, labels);
  }
#endif

  // ACTUAL RUN
  if (rank == ROOT) std::cout << "BENCH RUN" << std::endl;

  Statistics stats;
  Statistics global_stats;
  Statistics comm_stats;
  Statistics global_comm_stats;
  Statistics send_stats;
  Statistics global_send_stats;
  Statistics recv_stats;
  Statistics global_recv_stats;

  for (int i = 0; i < conf.iterations; ++i) {
    int round_seed = initial_seed + i + 1000;
    conf.seed = round_seed;

    Timer t;
    float local_time = 0.0;
    float total_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);

    t.Restart();
    StaticGraph G(conf, rank, size);
    IOUtility::LoadGraph(G, conf, rank, size);
    if (i == 0) IOUtility::PrintGraphParams(G, conf, rank, size);

    // Reset timers
    G.ResetCommTime();
    G.ResetSendVolume();
    G.ResetReceiveVolume();

    // Determine labels
    std::vector<VertexID> labels(G.GetVertexVectorSize(), 0);
    G.ForallLocalVertices([&](const VertexID v) {
      labels[v] = G.GetGlobalID(v);
    });
    local_time = t.Elapsed();
    std::cout << "IO rank=" << rank << " time=" << local_time << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    t.Restart();
    ExponentialContraction comp(conf, rank, size);
    comp.FindComponents(G, labels);
    local_time = t.Elapsed();

    // Print labels
    // G.OutputComponents(labels);

    // Gather total time
    MPI_Reduce(&local_time, &total_time, 1, MPI_FLOAT, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    stats.Push(local_time);
    if (rank == ROOT) global_stats.Push(total_time);

    // Gather comm time
    float comm_time = comp.GetCommTime();
    float total_comm_time = 0.0;
    MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_FLOAT, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    comm_stats.Push(comm_time);
    if (rank == ROOT) global_comm_stats.Push(total_comm_time);

    VertexID send_volume = comp.GetSendVolume();
    VertexID total_send_volume = 0;
    MPI_Reduce(&send_volume, &total_send_volume, 1, MPI_LONG, MPI_SUM, ROOT,
               MPI_COMM_WORLD);
    send_stats.Push(send_volume);
    if (rank == ROOT) global_send_stats.Push(total_send_volume);

    VertexID recv_volume = comp.GetReceiveVolume();
    VertexID total_recv_volume = 0;
    MPI_Reduce(&recv_volume, &total_recv_volume, 1, MPI_LONG, MPI_SUM, ROOT,
               MPI_COMM_WORLD);
    recv_stats.Push(recv_volume);
    if (rank == ROOT) global_recv_stats.Push(total_recv_volume);
  }

  if (conf.print_verbose) {
    std::cout << "LOCAL RESULT rank=" << rank << " runner=exp"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " comm_time=" << comm_stats.Avg() << " stddev=" << comm_stats.Stddev()
              << " send_volume=" << send_stats.Avg() << " stddev=" << send_stats.Stddev()
              << " recv_volume=" << recv_stats.Avg() << " stddev=" << recv_stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }
  if (rank == ROOT) {
    std::cout << "GLOBAL RESULT runner=exp"
              << " time=" << global_stats.Avg() << " stddev=" << global_stats.Stddev()
              << " comm_time=" << global_comm_stats.Avg() << " stddev=" << global_comm_stats.Stddev()
              << " send_volume=" << global_send_stats.Avg() << " stddev=" << global_send_stats.Stddev()
              << " recv_volume=" << global_recv_stats.Avg() << " stddev=" << global_recv_stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

