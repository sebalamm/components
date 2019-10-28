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
#include <unordered_set>
#include <random>
#include <set>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "cag_builder.h"
#include "utils.h"
#include "union_find.h"
#include "dynamic_graph_comm.h"
#include "static_graph_comm.h"

class Propagation {
 public:
  Propagation(const Config &conf, const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      config_(conf),
      iteration_(0) { }

  virtual ~Propagation() = default;

  void FindComponents(StaticGraphCommunicator &g, std::vector<VertexID> &g_labels) {
    if (config_.use_contraction) {
      // FindLocalComponents(g, g_labels);

      // CAGBuilder<StaticGraphCommunicator> 
      //   first_contraction(g, g_labels, rank_, size_);
      // StaticGraphCommunicator cag = first_contraction.BuildDynamicComponentAdjacencyGraph();
      // OutputStats<StaticGraphCommunicator>(cag);

      // // TODO: Delete original graph?
      // // Keep contraction labeling for later
      // std::vector<VertexID> cag_labels(cag.GetNumberOfVertices(), 0);
      // FindLocalComponents(cag, cag_labels);

      // CAGBuilder<StaticGraphCommunicator> 
      //   second_contraction(cag, cag_labels, rank_, size_);
      // StaticGraphCommunicator ccag = second_contraction.BuildDynamicComponentAdjacencyGraph();
      // OutputStats<StaticGraphCommunicator>(ccag);

      // PerformPropagation(ccag);

      // ApplyToLocalComponents(ccag, cag, cag_labels);
      // ApplyToLocalComponents(cag, cag_labels, g, g_labels);
    } else {
      PerformPropagation(g);
      g.ForallLocalVertices([&](const VertexID v) {
          g_labels[v] = g.GetVertexLabel(v);
      });
    }
  }

  void Output(StaticGraphCommunicator &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Counters
  unsigned int iteration_;

  // Local labels
  std::vector<VertexID> prev_labels_;

  void PerformPropagation(StaticGraphCommunicator &g) {
    prev_labels_.resize(g.GetNumberOfLocalVertices());
    // Iterate until converged
    do {
      PropagateLabels(g);
      FindMinLabels(g);

      OutputStats<StaticGraphCommunicator>(g);

      iteration_++;
    } while (!CheckConvergence(g));
  }

  void FindLocalComponents(StaticGraphCommunicator &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetVertexLabel(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<StaticGraphCommunicator>::BFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void PropagateLabels(StaticGraphCommunicator &g) {
    g.ForallLocalVertices([&](const VertexID v) { 
      prev_labels_[v] = g.GetVertexLabel(v); 
    });
    g.SendAndReceiveGhostVertices();
  }

  void FindMinLabels(StaticGraphCommunicator &g) {
    // std::vector<bool> changed(g.GetNumberOfLocalVertices(), false);
    // bool label_changed = true;
    // while(label_changed) {
    //   label_changed = false;
      g.ForallLocalVertices([&](VertexID v) {
        // Gather min label of all neighbors
        VertexID v_label = g.GetVertexLabel(v);
        g.ForallNeighbors(v, [&](VertexID u) {
          if (g.GetVertexLabel(u) < v_label) {
            // label_changed = true;
            // changed[v] = true;
            v_label = g.GetVertexLabel(u);
          }
        });
        // g.SetVertexLabel(v, v_label);
        g.SetVertexPayload(v,
                           {g.GetVertexDeviate(v), 
                            v_label,
#ifdef TIEBREAK_DEGREE
                            0,
#endif
                            g.GetVertexRoot(v)});
        });
    // }

//     g.ForallLocalVertices([&](VertexID v) {
//         if (changed[v] && g.IsInterface(v)) {
//           g.ForceVertexPayload(v,
//                                {g.GetVertexDeviate(v), 
//                                 g.GetVertexLabel(v),
// #ifdef TIEBREAK_DEGREE
//                                 0,
// #endif
//                                 g.GetVertexRoot(v)});
//         } 
//       });
  }

  bool CheckConvergence(StaticGraphCommunicator &g) {
    int converged_globally = 0;

    // Check local convergence
    int converged_locally = 1;
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetVertexLabel(v) != prev_labels_[v]) converged_locally = 0;
    });

    MPI_Allreduce(&converged_locally,
                  &converged_globally,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    return converged_globally;
  }

  void ApplyToLocalComponents(StaticGraphCommunicator &cag, 
                              StaticGraphCommunicator &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(StaticGraphCommunicator &cag, 
                              std::vector<VertexID> &cag_label, 
                              StaticGraphCommunicator &g, 
                              std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
    });
  }

  template <typename GraphType>
  void OutputStats(GraphType &g) {
    VertexID n = g.GatherNumberOfGlobalVertices();
    EdgeID m = g.GatherNumberOfGlobalEdges();

    // Determine min/maximum cut size
    EdgeID m_cut = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      std::cout << "TEMP "
                << "s=" << config_.seed << ", "
                << "p=" << size_  << ", "
                << "n=" << n << ", "
                << "m=" << m << ", "
                << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }
};

#endif
