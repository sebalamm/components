/******************************************************************************
 * propagation.h
 *
 * Distributed label propagation
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

#ifndef _PROPAGATION_H_
#define _PROPAGATION_H_

#include <iostream>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "graph_access.h"

class Propagation {
  public:
    Propagation() {}
    virtual ~Propagation() {}

    void FindComponents(GraphAccess &g, const Config &conf, const PEID rank) {
      // Iterate for fixed number of rounds
      for (unsigned int i = 0; i < conf.prop_iterations; ++i) {
        FindLocalComponents(g);
        // Propagate ghost vertex labels
        g.UpdateGhostVertices();       
        // Sync to prevent race conditions
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    void Output(GraphAccess &g) {
      GatherComponents(g);
    }

  private:
    void FindLocalComponents(GraphAccess &g) {
      g.ForallLocalVertices([&](VertexID v)  {
        // Gather min label of all neighbors
        VertexID min_label = g.GetVertexLabel(v);
        g.ForallNeighbors(v, [&](VertexID u) {
          min_label = std::min(g.GetVertexLabel(u), min_label);
        });
        g.SetVertexLabel(v, min_label);
      });
    }

    void GatherComponents(GraphAccess &g) {
      // Gather local components
      std::vector<VertexID> local_components(g.NumberOfLocalVertices());
      g.ForallLocalVertices([&](VertexID v)  {
          local_components[v] = g.GetVertexLabel(v);
      });

      // Eliminate duplicates
      sort(local_components.begin(), local_components.end());
      local_components.erase(unique(local_components.begin(), local_components.end()), local_components.end());
      int num_local_components = local_components.size();

      // Gather number of components for each PE
      PEID rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      std::vector<int> num_components(size);
      MPI_Allgather(&num_local_components, 1, MPI_INT, &num_components[0], 1, MPI_INT, MPI_COMM_WORLD);
      
      VertexID num_total_components = 0;
      for (VertexID comps : num_components) num_total_components += comps;

      // Concatenate components on root 
      std::vector<VertexID> global_components(num_total_components);
      std::vector<int> displ(size);

      if (rank == ROOT) {
        VertexID sum = 0;
        for (PEID i = 0; i < size; i++) {
          displ[i] = sum;
          sum += num_components[i];
        }
      }

      MPI_Gatherv(&local_components[0], num_components[rank], MPI_LONG, &global_components[0], &num_components[0], &displ[0], MPI_LONG, ROOT, MPI_COMM_WORLD);
      sort(global_components.begin(), global_components.end());
      global_components.erase(unique(global_components.begin(), global_components.end()), global_components.end());

      // Output
      if (rank == ROOT) {
        std::cout << "num ccs " << global_components.size() << std::endl;
        std::cout << "ccs (";
        for (VertexID i = 0; i < global_components.size() - 1; ++i) 
          std::cout << global_components[i] << " ";
        std::cout << global_components[global_components.size()-1] << ")" << std::endl;
      }

    }
};

#endif
