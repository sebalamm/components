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
  GraphAccess
      G = GraphIO::ReadDistributedGraph(conf, rank, size, MPI_COMM_WORLD);

  if (rank == ROOT) {
    std::cout << "compute ccs (s=" << conf.seed << ", p=" << size << ")"
              << std::endl;
  }

  // Timers
  Timer t;
  Statistics stats;

  double local_time = 0.0;
  double total_time = 0.0;

  int user_seed = conf.seed;
  for (int i = 0; i < conf.iterations; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    t.Restart();

    // Determine labels
    conf.seed = user_seed + i;
    ShortcutPropagation comp(conf, rank, size);
    comp.FindComponents(G);

    // Gather total time
    local_time = t.Elapsed();
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) stats.Push(total_time);
    
    // Print labels
    // comp.Output(G);
  }

  if (rank == ROOT) {
    std::cout << "RESULT runner=comp"
              << " time=" << stats.Avg() << " stddev=" << stats.Stddev()
              << " iterations=" << conf.iterations << std::endl;
  }

  // Finalize MPI
  MPI_Finalize();
}
