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

#include "components/components.h"

int main(int argn, char **argv) {
  // Init MPI
  MPI_Init(&argn, &argv);
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read command-line args
  Config conf;
  ParseParameters(argn, argv, conf);

  // int b = rank==0? -1 : rank;
  // while(b) ;
  
  GraphAccess G = GraphIO::ReadDistributedGraph(conf, rank, size, MPI_COMM_WORLD);
  // std::cout << "id " << G.GetGlobalID(0) << " rank " << rank << std::endl;
  if(rank==2) {
    std::cout << "Rank " << rank << " local id" << G.GetLocalID(1312) << std::endl;
    std::cout << "Rank " << rank << " Vertex with global ID 1312 belongs to " << G.GetPE(G.GetLocalID(1312)) << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);


  // Finalize MPI
  MPI_Finalize();
}

