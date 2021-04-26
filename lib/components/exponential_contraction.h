/******************************************************************************
 * components.h *
 * Distributed computation of connected components;
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

#ifndef _EXPONENTIAL_CONTRACTION_H_
#define _EXPONENTIAL_CONTRACTION_H_

#include <iostream>

#include <experimental/random>
#include <tlx/define.hpp>
#include <tlx/math.hpp>

#include "config.h"
#include "definitions.h"
#include "static_graph.h"
#include "dynamic_graph_comm.h"
#include "cag_builder.h"
#include "dynamic_contraction.h"
#include "utils.h"
#include "comm_utils.h"
#include "all_reduce.h"

class ExponentialContraction {
 public:
  ExponentialContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0),
        comm_time_(0.0) { 
    // replicated_vertices_.set_empty_key(EmptyKey);
    // replicated_vertices_.set_deleted_key(DeleteKey);
  }

  virtual ~ExponentialContraction() {
    delete exp_contraction_;
    exp_contraction_ = nullptr;
  };

  template <typename GraphType>
  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    contraction_timer_.Restart();

    if constexpr (std::is_same<GraphType, StaticGraph>::value) {
      FindLocalComponents(g, g_labels);
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |- Finding local components on input took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }
      
      if (config_.single_level_contraction) {
        contraction_timer_.Restart();
        CAGBuilder<StaticGraph> 
          contraction(g, g_labels, rank_, size_);
        auto cag 
          = contraction.BuildComponentAdjacencyGraph<DynamicGraphCommunicator>();
        cag.ResetCommunicator();
        OutputStats<DynamicGraphCommunicator>(cag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Building cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          contraction_timer_.Restart();
          DistributeHighDegreeVertices(cag);
          OutputStats<DynamicGraphCommunicator>(cag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Distributing high degree vertices took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }
        }

        // Keep contraction labeling for later
        exp_contraction_ = new DynamicContraction(cag, rank_, size_);

        // Main decomposition algorithm
        contraction_timer_.Restart(); 
        PerformDecomposition(cag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Resolving connectivity took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          RemoveReplicatedVertices(cag);
        }

        ApplyToLocalComponents(cag, g, g_labels);
        comm_time_ += contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + cag.GetCommTime() + g.GetCommTime();
      } else if (config_.use_contraction) {
        // First round of contraction
        contraction_timer_.Restart();
        CAGBuilder<StaticGraph> 
          first_contraction(g, g_labels, rank_, size_);
        auto cag 
          = first_contraction.BuildComponentAdjacencyGraph<StaticGraph>();
        OutputStats<StaticGraph>(cag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Building first cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        // Keep contraction labeling for later
        contraction_timer_.Restart();
        std::vector<VertexID> cag_labels(cag.GetNumberOfVertices(), 0);
        FindLocalComponents(cag, cag_labels);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Finding local components on cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        // Second round of contraction
        contraction_timer_.Restart();
        CAGBuilder<StaticGraph> 
          second_contraction(cag, cag_labels, rank_, size_);
        auto ccag 
          = second_contraction.BuildComponentAdjacencyGraph<DynamicGraphCommunicator>();
        ccag.ResetCommunicator();
        OutputStats<DynamicGraphCommunicator>(ccag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Building second cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          contraction_timer_.Restart();
          DistributeHighDegreeVertices(ccag);
          OutputStats<DynamicGraphCommunicator>(ccag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Distributing high degree vertices took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }
          // ccag.OutputLocal();
          // ccag.OutputGhosts();
        }

        // Keep contraction labeling for later
        exp_contraction_ = new DynamicContraction(ccag, rank_, size_);

        // Main decomposition algorithm
        contraction_timer_.Restart(); 
        PerformDecomposition(ccag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Resolving connectivity took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          RemoveReplicatedVertices(ccag);
        }

        ApplyToLocalComponents(ccag, cag, cag_labels);
        ApplyToLocalComponents(cag, cag_labels, g, g_labels);
        comm_time_ += first_contraction.GetCommTime() + second_contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + ccag.GetCommTime() + cag.GetCommTime() + g.GetCommTime();
      } else {
        // At least contract locally
        contraction_timer_.Restart();
        CAGBuilder<StaticGraph> 
          local_contraction(g, g_labels, rank_, size_);
        auto lcag 
          = local_contraction.BuildLocalComponentGraph<DynamicGraphCommunicator>();
        lcag.ResetCommunicator();
        OutputStats<DynamicGraphCommunicator>(lcag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Building local cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          contraction_timer_.Restart();
          DistributeHighDegreeVertices(lcag);
          OutputStats<DynamicGraphCommunicator>(lcag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Distributing high degree vertices took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }
        }

        exp_contraction_ = new DynamicContraction(lcag, rank_, size_);

        PerformDecomposition(lcag);

        if (config_.replicate_high_degree) {
          RemoveReplicatedVertices(lcag);
        }

        ApplyToLocalComponents(lcag, g, g_labels);
        comm_time_ += local_contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + lcag.GetCommTime() + g.GetCommTime();

        // TODO: I think this fails 
        // google::sparse_hash_map<VertexID, VertexID> comp_sizes;
        // g.ForallLocalVertices([&](const VertexID v) {
        //     if (comp_sizes.find(g.GetVertexLabel(v)) == comp_sizes.end()) {
        //       comp_sizes[g.GetVertexLabel(v)] = 0;
        //     }
        //     comp_sizes[g.GetVertexLabel(v)]++;
        //     if (v >= g_labels.size()) {
        //       std::cout << "R" << rank_ << " This shouldn't happen: Invalid vertex remap v=" << g.GetGlobalID(v) << " lid=" << v << " label=" << g.GetVertexLabel(v) << std::endl;
        //       exit(1);
        //     }
        //     g_labels[v] = g.GetVertexLabel(v);
        // });
      }
    } 
  }

  void Output(DynamicGraphCommunicator &g) {
    g.OutputLabels();
  }

  float GetCommTime() {
    return comm_time_; 
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;
  VertexID rng_offset_;

  // Statistics
  float comm_time_;
  Timer iteration_timer_;
  Timer contraction_timer_;
  Timer comm_timer_;
  
  // Contraction
  DynamicContraction *exp_contraction_;

  // Node replication
  // google::dense_hash_map<VertexID, VertexID> replicated_vertices_;
  VertexID global_repl_offset_;

  void PerformDecomposition(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart(); 
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_timer_.Restart();
      iteration_++;
      if (global_vertices <= config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        if (config_.use_bfs) RunContractionBFS(g);
        else RunContractionLP(g);
    }
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |- Running contraction took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart(); 
    exp_contraction_->UndoContraction();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |- Undoing contraction took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
  }

  void FindLocalComponents(StaticGraph &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS<StaticGraph>(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void RunContractionLP(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart();
    if (rank_ == ROOT) {
      if (iteration_ == 1)
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << "-" << std::endl;
      else
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << iteration_timer_.Elapsed() << std::endl;
    }
    iteration_timer_.Restart();
    // if (iteration_ == 2) g.OutputLocal();

    std::exponential_distribution<LPFloat> distribution(config_.beta);
    std::mt19937
        generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
    g.ForallLocalVertices([&](const VertexID v) {
      // Set preliminary deviate
      g.SetParent(v, g.GetGlobalID(v));
      LPFloat weight = 
#ifdef TIEBREAK_DEGREE
        // static_cast<LPFloat>(g.GetVertexDegree(v) / g.GetMaxDegree());
        // static_cast<LPFloat>(log2(g.GetNumberOfVertices()) / g.GetVertexDegree(v));
        1.0;
#else
        1.0;
#endif
      g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
                             g.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)}, 
                         true);
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
    });
    g.SendAndReceiveGhostVertices();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing and exchanging deviates took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT)
      std::cout << "[STATUS] |-- Pick deviates " 
                << "[TIME] " << iteration_timer_.Elapsed() << std::endl;

    VertexID exchange_rounds = 0;
    int converged_globally = 0;
    int local_iterations = 0;
    Timer round_timer;
    while (converged_globally == 0) {
      round_timer.Restart();
      contraction_timer_.Restart();
      int converged_locally = 1;

      // Perform update for local vertices
      g.ForallLocalVertices([&](VertexID v) {
        auto smallest_payload = g.GetVertexMessage(v);
        g.ForallNeighbors(v, [&](VertexID w) {
          // Store neighbor label
          if (g.GetVertexDeviate(w) + 1 < smallest_payload.deviate_ ||
              (g.GetVertexDeviate(w) + 1 == smallest_payload.deviate_ &&
#ifdef TIEBREAK_DEGREE
                  g.GetVertexDegree(w) < smallest_payload.degree_)) {
#else 
                  g.GetVertexLabel(w) < smallest_payload.label_)) {
#endif
            g.SetParent(v, g.GetGlobalID(w));
            smallest_payload = {g.GetVertexDeviate(w) + 1, 
                                g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                g.GetVertexDegree(w),
#endif
                                g.GetVertexRoot(w)};
            converged_locally = 0;
          }
        });
        g.SetVertexPayload(v, std::move(smallest_payload));
      });
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Updating payloads took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      contraction_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      exchange_rounds++;
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Convergence test took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      contraction_timer_.Restart();
      // Receive variates
      g.SendAndReceiveGhostVertices();
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Exchanging payloads took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      if (rank_ == ROOT) 
        std::cout << "[STATUS] |--- Round finished " 
                  << "[TIME] " << round_timer.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done propagating... mem " << Utility::GetFreePhysMem() << std::endl;

    contraction_timer_.Restart();
    // Determine remaining active vertices
    g.BuildLabelShortcuts();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Building shortcuts took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done shortcutting... mem " << Utility::GetFreePhysMem() << std::endl;

    contraction_timer_.Restart();
    if (config_.direct_contraction) {
      exp_contraction_->DirectContraction();
    } else {
      exp_contraction_->ExponentialContraction();
    }
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Exponential contraction took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done contraction... mem " << Utility::GetFreePhysMem() << std::endl;
    OutputStats<DynamicGraphCommunicator>(g);

    // Count remaining number of vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices <= config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        RunContractionLP(g);
    }
  }

  void RunContractionBFS(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart();
    if (rank_ == ROOT) {
      if (iteration_ == 1)
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << "-" << std::endl;
      else
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << iteration_timer_.Elapsed() << std::endl;
    }
    iteration_timer_.Restart();
    
    // Compute starting times
    // Set initial parent and payload
    int active_round = 0;
    int max_round = 0;
    std::exponential_distribution<LPFloat> distribution(config_.beta);
    std::mt19937
        generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
    g.ForallLocalVertices([&](const VertexID v) {
      // Set preliminary deviate
      g.SetParent(v, g.GetGlobalID(v));
      LPFloat weight = 
#ifdef TIEBREAK_DEGREE
        // static_cast<LPFloat>(g.GetVertexDegree(v) / g.GetMaxDegree());
        // static_cast<LPFloat>(log2(g.GetNumberOfVertices()) / g.GetVertexDegree(v));
        1.0;
#else
        1.0;
#endif
      VertexID vertex_round = static_cast<VertexID>(weight * distribution(generator));
      if (vertex_round > max_round) max_round = vertex_round;
      // Don't propagate payloads at the beginning
      // We only want to send payloads if its a vertex turn to do so
      g.SetVertexPayload(v, {vertex_round,
                             g.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)}, 
                         vertex_round == active_round);
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
    });
    g.SendAndReceiveGhostVertices();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing deviates and starting BFS took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT)
      std::cout << "[STATUS] |-- Pick deviates " 
                << "[TIME] " << iteration_timer_.Elapsed() << std::endl;

    VertexID exchange_rounds = 0;
    int converged_globally = 0;
    int local_iterations = 0;
    Timer round_timer;
    google::dense_hash_set<VertexID> active_vertices; 
    active_vertices.set_empty_key(EmptyKey);
    active_vertices.set_deleted_key(DeleteKey);
    VertexID num_active = 0;
    while (converged_globally == 0) {
      round_timer.Restart();
      contraction_timer_.Restart();
      int converged_locally = 1;
      // Increase active round to simulate time step
      if (max_round > active_round 
          && num_active < g.GetNumberOfVertices()) 
        converged_locally = 0;

      // Mark vertices active if their deviate (starting time) is the current round
      // This includes ghost vertices
      g.ForallVertices([&](VertexID v) {
        VertexID vertex_round = g.GetVertexDeviate(v);
        if (vertex_round == active_round && active_vertices.find(v) == active_vertices.end()) {
          active_vertices.insert(v);
          num_active++;
        }
        // Vertex becomes active next round so we need to schedule its payload
        else if (vertex_round == active_round + 1) {
          auto vertex_payload = g.GetVertexMessage(v);
          vertex_payload = {g.GetVertexDeviate(v),
                            g.GetVertexLabel(v),
#ifdef TIEBREAK_DEGREE
                            g.GetVertexDegree(v),
#endif
                            g.GetVertexRoot(v)};
          g.ForceVertexPayload(v, std::move(vertex_payload));
          converged_locally = 0;
        }
      });

      // Iterate over currently active vertices (including ghosts)
      google::dense_hash_map<VertexID, VertexPayload> updated_payloads; 
      updated_payloads.set_empty_key(EmptyKey);
      updated_payloads.set_deleted_key(DeleteKey);
      for (const VertexID &v : active_vertices) {
        // Iterate over neighborhood and look for non-active vertices
        g.ForallNeighbors(v, [&](VertexID w) {
          // Local neighbor (no message) 
          // If the neighbor is a ghost, the other PE takes care of it after receiving the payload in the last round
          if (g.IsLocal(w) && active_vertices.find(w) == active_vertices.end()) {
            auto vertex_payload = g.GetVertexMessage(w);
            // Neighbor might get more than one update
            // Choose min for now
            if (g.GetVertexDeviate(v) + 1 < vertex_payload.deviate_) { 
              g.SetParent(w, g.GetGlobalID(v));
              vertex_payload = {g.GetVertexDeviate(v) + 1,
                                g.GetVertexLabel(v),
#ifdef TIEBREAK_DEGREE
                                g.GetVertexDegree(v),
#endif
                                g.GetVertexRoot(v)};
              converged_locally = 0;
              // Delay inserting the neighbor into active vertices and sending the payload
              // updated_payloads[w] = vertex_payload;
              g.SetVertexPayload(w, std::move(vertex_payload));
            }
          } 
        });
      }

      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Updating payloads took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      contraction_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      exchange_rounds++;
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Convergence test took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      contraction_timer_.Restart();
      // Receive variates
      g.SendAndReceiveGhostVertices();
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Exchanging payloads took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      active_round++;
      active_vertices.clear();
      if (rank_ == ROOT) 
        std::cout << "[STATUS] |--- Round finished " 
                  << "[TIME] " << round_timer.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done propagating... mem " << Utility::GetFreePhysMem() << std::endl;

    contraction_timer_.Restart();
    // Determine remaining active vertices
    g.BuildLabelShortcuts();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Building shortcuts took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done shortcutting... mem " << Utility::GetFreePhysMem() << std::endl;

    contraction_timer_.Restart();
    if (config_.direct_contraction) {
      exp_contraction_->DirectContraction();
    } else {
      exp_contraction_->ExponentialContraction();
    }
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Exponential contraction took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done contraction... mem " << Utility::GetFreePhysMem() << std::endl;
    OutputStats<DynamicGraphCommunicator>(g);

    // Count remaining number of vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices <= config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        RunContractionBFS(g);
    }
  }

  void ApplyToLocalComponents(DynamicGraphCommunicator &cag, 
                              StaticGraph &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(StaticGraph &cag, 
                              std::vector<VertexID> &cag_label, 
                              StaticGraph &g, 
                              std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
    });
  }

  void RunSequentialCC(DynamicGraphCommunicator &g) {
    // Build vertex mapping 
    google::dense_hash_map<VertexID, int> vertex_map; 
    vertex_map.set_empty_key(EmptyKey);
    vertex_map.set_deleted_key(DeleteKey);
    std::vector<VertexID> reverse_vertex_map(g.GetNumberOfLocalVertices());
    int current_vertex = 0;
    g.ForallLocalVertices([&](const VertexID v) {
      vertex_map[v] = current_vertex;
      reverse_vertex_map[current_vertex++] = v;
    });

    // Init labels
    std::vector<VertexID> labels(g.GetNumberOfLocalVertices());
    for (VertexID i = 0; i < labels.size(); ++i) {
      labels[i] = g.GetVertexLabel(reverse_vertex_map[i]);
    }
    g.ForallLocalVertices([&](const VertexID v) {
      labels[v] = g.GetVertexLabel(v);
    });

    // Run all-reduce
    AllReduce<DynamicGraphCommunicator> ar(config_, rank_, size_);
    ar.FindComponents(g, labels);

    g.ForallLocalVertices([&](const VertexID v) {
      g.SetVertexLabel(v, labels[vertex_map[v]]);
    });
  }

  // void SampleHighDegreeNeighborhoods(DynamicGraphCommunicator &g) {
  //   std::vector<std::pair<VertexID, VertexID>> high_degree_vertices;
  //   VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_);
  //   Utility::SelectHighDegreeVertices(g, config_.degree_threshold * avg_max_deg, high_degree_vertices);

  //   for (const auto &vd : high_degree_vertices) {
  //     g.SampleVertexNeighborhood(v, config_.neighborhood_sampling_factor);
  //   }
  // }

  void DistributeHighDegreeVertices(DynamicGraphCommunicator &g) {
    // Determine high degree vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    std::vector<std::pair<VertexID, VertexID>> high_degree_vertices;
    VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_);
    // Use sqrt(n) as a degree threshold
    config_.degree_threshold = static_cast<VertexID>(config_.degree_threshold*sqrt(global_vertices));
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |- High degree threshold " << config_.degree_threshold
                << " [TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    Utility::SelectHighDegreeVertices(g, config_.degree_threshold, high_degree_vertices);
    
    // Split high degree vertices into one layer of proxies with degree sqrt(n)
    SplitHighDegreeVerticesSqrtEdge(g, avg_max_deg, high_degree_vertices);
    // SplitHighDegreeVerticesSqrtPseudo(g, avg_max_deg, high_degree_vertices);
    // SplitHighDegreeVerticesSqrt(g, avg_max_deg, high_degree_vertices);
    // Split high degree vertices into binomial trees
    // SplitHighDegreeVerticesIntoTrees(g, avg_max_deg, high_degree_vertices);
  }

  void SplitHighDegreeVerticesSqrtEdge(DynamicGraphCommunicator &g,
                                         const VertexID &avg_max_deg,
                                         std::vector<std::pair<VertexID, VertexID>> &high_degree_vertices) {
    // Compute offset for IDs of replicated vertices
    // g.OutputLocal();
    VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
    global_repl_offset_ = num_global_vertices;

    // Default buffers for message exchange
    google::dense_hash_map<PEID, VertexBuffer> send_buffers;
    send_buffers.set_empty_key(EmptyKey);
    send_buffers.set_deleted_key(DeleteKey);
    google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
    receive_buffers.set_empty_key(EmptyKey);
    receive_buffers.set_deleted_key(DeleteKey);
    
    // New edges to replicated vertices
    std::vector<VertexID> local_edges;

    // Map to relink edges
    // Source, Target, Pair(Repl, Repl PE)
    google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> replicated_edges;
    replicated_edges.set_empty_key(EmptyKey);
    replicated_edges.set_deleted_key(DeleteKey);

    //////////////////////////////////////////
    // Send high degree neighbors (their degree)
    //////////////////////////////////////////
    google::dense_hash_set<PEID> packed_pes;
    packed_pes.set_empty_key(EmptyKey);
    packed_pes.set_deleted_key(DeleteKey);
    for (VertexID i = 0; i < high_degree_vertices.size(); ++i) {
      VertexID v = high_degree_vertices[i].first;
      VertexID deg_v = high_degree_vertices[i].second;

      g.ForallNeighbors(v, [&](const VertexID w) {
        if (!g.IsLocal(w)) {
          PEID target_pe = g.GetPE(w);
          // Only send vertex once
          if (packed_pes.find(target_pe) == packed_pes.end()) {
            send_buffers[target_pe].emplace_back(g.GetGlobalID(v));
            send_buffers[target_pe].emplace_back(deg_v);
            packed_pes.insert(target_pe);
          }
        }
      });
      packed_pes.clear();
    }

    // Send updates
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);

    //////////////////////////////////////////
    // Receive high degree neighbors (their degree)
    //////////////////////////////////////////
    for (auto &kv : receive_buffers) {
      PEID source_pe = kv.first;
      auto &vertex_degrees = kv.second;
      for (VertexID i = 0; i < vertex_degrees.size(); i += 2) {
        VertexID vertex = vertex_degrees[i];
        VertexID degree = vertex_degrees[i + 1];
        high_degree_vertices.emplace_back(g.GetLocalID(vertex), degree);
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);

    //////////////////////////////////////////
    // Each PE now has a vector of his and neighboring high degree vertices
    // Compute distribution of replicates
    //////////////////////////////////////////
    std::uniform_int_distribution<PEID> dist(0, size_ - 1);
    PEID num_replicates = std::min(size_ - 1, static_cast<PEID>(config_.degree_threshold));
    // This now also includes ghosts
    for (VertexID i = 0; i < high_degree_vertices.size(); ++i) {
      VertexID v = high_degree_vertices[i].first;
      VertexID vertex_seed = config_.seed + g.GetGlobalID(v);
      PEID taboo_rank = g.GetPE(v);
      // If we compute the repl_vertices_id here, this should be recomputed consistently
      // VertexID vertex_offset = num_global_vertices * (taboo_rank + 1) + g.GetGlobalID(v);
      VertexID vertex_offset = num_global_vertices * size_ * (taboo_rank + 1) + g.GetGlobalID(v) * size_;
      VertexID repl_vertices_id = vertex_offset;

      // Compute set of min(P - 1, sqrt(n)) _unique_ possible targets to sample from
      // Associate a unique replicate with each target
      // We have to initialize a mersenne twister (or something similar) for each vertex
      // so that a neighboring PE can reproduce this set without communication
      std::mt19937 gen(vertex_seed);
      google::dense_hash_map<PEID, VertexID> pes_for_replication;
      pes_for_replication.set_empty_key(EmptyKey);
      pes_for_replication.set_deleted_key(DeleteKey);
      // Vector to actually sample PEs from
      std::vector<std::pair<PEID, VertexID>> sampling_vector;
      while (pes_for_replication.size() < num_replicates) {
        PEID target_pe = size_;
        do {
          target_pe = dist(gen);
        } while (target_pe == taboo_rank || pes_for_replication.find(target_pe) != pes_for_replication.end());
        pes_for_replication[target_pe] = repl_vertices_id;
        // std::cout << "R" << rank_ << " store replicate vertex " << repl_vertices_id << " PE(rep) " << target_pe << " for v " << g.GetGlobalID(v) << " PE(v) " << taboo_rank << std::endl;
        sampling_vector.emplace_back(target_pe, repl_vertices_id);
        repl_vertices_id++;
      }

      // For each outgoing edge select a random PE from the set of targets
      std::uniform_int_distribution<PEID> dist(0, num_replicates - 1);
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (!g.IsLocal(v) || !g.IsLocal(w)) {
          // Create a unique seed for the current edge
          VertexID edge_seed = config_.seed + g.GetGlobalID(v) * num_global_vertices + g.GetGlobalID(w);
          // Select the (random) target PE
          std::mt19937 gen(edge_seed);
          PEID index = dist(gen);
          // Get the associated elements from the sampling vector
          PEID target_pe = sampling_vector[index].first;
          VertexID repl_vertices_id = sampling_vector[index].second;

          // PE now knows that edge (v, w) will be replicated to repl_vertices_id on target_pe
          // For each edge store its associated replicate
          // std::cout << "R" << rank_ << " store replicate edge (" << repl_vertices_id << "," << g.GetGlobalID(w) << ") for e (" << g.GetGlobalID(v) << "," << g.GetGlobalID(w) << ")" << std::endl;
          replicated_edges[g.GetGlobalID(v)][g.GetGlobalID(w)] = std::make_pair(repl_vertices_id, target_pe);
        }
      });
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank_ == ROOT) std::cout << "[R" << rank_ << "] Computed initial replicates" << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);

    // We have to relink our replicated edges based on the recomputed information 
    // I.e. if for an edge (v,w) both endpoints have been replicated (replicated_edges[v][w] and replicated_edges[w][v] exist)
    // In this case update the edge with the correct target
    // Finally, send the edges to their respective replicates
    // TODO: We also have to insert edges that run from ghost replicates to local vertices
    // since these edges are not present otherwise
    // for this purpose relink the old edges to the replicates
    for (VertexID i = 0; i < high_degree_vertices.size(); ++i) {
      VertexID v = high_degree_vertices[i].first;

      // 1.  Keep (local, local) edges
      // 2.  Distribute (local, ghost) edges
      // 3.  Relink (ghost, local) edges
      VertexID edge_source_id = g.GetGlobalID(v);
      PEID edge_source_pe = g.GetPE(v);
      if (g.IsLocal(v)) {
        g.ForallNeighbors(v, [&](const VertexID w) {
          VertexID edge_target_id = g.GetGlobalID(w);
          PEID edge_target_pe = g.GetPE(w);

          // Non local edge (can be redistributed)
          if (edge_target_pe != rank_) {
            auto &replicate = replicated_edges[edge_source_id][edge_target_id];
            VertexID repl_source_id = replicate.first;
            PEID repl_source_pe = replicate.second;

            // Add replicates (and corresponding edge) if they are not already present
            if (!(g.IsLocalFromGlobal(repl_source_id) || g.IsGhostFromGlobal(repl_source_id))) {
              g.AddGhostVertex(repl_source_id, repl_source_pe);
              local_edges.emplace_back(repl_source_id);
              local_edges.emplace_back(repl_source_pe);
            }

            // Always start message with original and replicated source
            send_buffers[repl_source_pe].emplace_back(edge_source_id);
            send_buffers[repl_source_pe].emplace_back(repl_source_id);

            // Now check if the target is still valid or was also replicated
            // Update the edge target accordingly
            if (replicated_edges.find(edge_target_id) != replicated_edges.end() &&
                replicated_edges[edge_target_id].find(edge_source_id) != replicated_edges[edge_target_id].end()) {
              auto &reverse_replicate = replicated_edges[edge_target_id][edge_source_id];
              // If so, overwrite the edge target with the replicate and the associated pe
              edge_target_id = reverse_replicate.first;
              edge_target_pe = reverse_replicate.second;
            }

            // Send (updated) edge target to replicate vertex
            send_buffers[repl_source_pe].emplace_back(edge_target_id);
            send_buffers[repl_source_pe].emplace_back(edge_target_pe);

            // std::cout << "R" << rank_ << " send edge (" << repl_source_id << "," << edge_target_id << " (PE " << edge_target_pe << ")) to PE " << repl_source_pe << std::endl;
          } 
          // Keep local edges
          else {
            local_edges.emplace_back(edge_target_id);
            local_edges.emplace_back(edge_target_pe);
          }
        });
      } 
      else {
        g.ForallNeighbors(v, [&](const VertexID w) {
          VertexID edge_target_id = g.GetGlobalID(w);
          PEID edge_target_pe = g.GetPE(w);
          // This should always be rank
          if (edge_target_pe == rank_) {
            // Check if w is also high degree (v, w both high degree)
            // If so, this should be covered be the previous case 
            // And we should not relink or add any edges
            if (replicated_edges.find(edge_target_id) == replicated_edges.end()) {
              // Get the replicate and relink
              if (replicated_edges.find(edge_source_id) != replicated_edges.end() &&
                  replicated_edges[edge_source_id].find(edge_target_id) != replicated_edges[edge_source_id].end()) {
                auto &replicate = replicated_edges[edge_source_id][edge_target_id];
                // Add replicates if they are not already present
                if (!(g.IsLocalFromGlobal(replicate.first) || g.IsGhostFromGlobal(replicate.first))) {
                  if (replicate.second == rank_) {
                    // std::cout << "R" << rank_ << " add repl (local) " << replicate.first << std::endl;
                    g.AddVertex(replicate.first);
                    // Also add original source if not existent yet
                    if (!(g.IsLocalFromGlobal(edge_source_id) || g.IsGhostFromGlobal(edge_source_id))) {
                      g.AddGhostVertex(edge_source_id, edge_source_pe);
                    }
                    // Add local edge from replicate to source (and vice versa)
                    g.AddEdge(g.GetLocalID(replicate.first), edge_source_id, edge_source_pe);
                    g.AddEdge(g.GetLocalID(edge_source_id), replicate.first, rank_);
                    // std::cout << "R" << rank_ << " add edge (repl) (" << replicate.first << "," << edge_source_id << " (PE " << edge_source_pe << "))" << std::endl;
                  } else {
                    // std::cout << "R" << rank_ << " add repl (ghost) " << replicate.first << std::endl;
                    g.AddGhostVertex(replicate.first, replicate.second);
                  }
                }
                // Relink
                bool relink_success = g.RelinkEdge(g.GetLocalID(edge_target_id), edge_source_id, replicate.first, replicate.second);
                if (!relink_success) {
                  std::cout << "R" << rank_ << " This shouldn't happen: Invalid (local) relink (" << edge_target_id << "," << edge_source_id << ") -> (" << edge_target_id << "," << replicate.first << ") from R" << edge_source_pe << " to R" << replicate.second << std::endl;
                  exit(1);
                }
              }
            }
          } else {
            std::cout << "R" << rank_ << " This shouldn't happen: Invalid high degree edge" << std::endl;
            exit(1);
          }
        });
      }
      // Replace edges with new edges to replicates
      g.RemoveAllEdges(v);
      for (VertexID i = 0; i < local_edges.size(); i+= 2) {
        g.AddEdge(v, local_edges[i], local_edges[i+1]);
        // Add reverse edge for non local edges
        if (local_edges[i+1] != rank_) {
          g.AddEdge(g.GetLocalID(local_edges[i]), g.GetGlobalID(v), rank_);
        }
      }
      local_edges.clear();
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank_ == ROOT) std::cout << "[R" << rank_ << "] Updated and sent replicates" << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
    
    // Send updates
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);

    //////////////////////////////////////////
    // Process received edges 
    //////////////////////////////////////////
    // First add any replicate vertices
    for (auto &kv : receive_buffers) {
      PEID edge_source_pe = kv.first;
      auto &edges = kv.second;

      // Iterate over received edges
      for (VertexID i = 0; i < edges.size(); i += 4) {
        VertexID edge_source_id = edges[i];
        VertexID repl_source_id = edges[i + 1];
        VertexID edge_target_id = edges[i + 2];
        PEID edge_target_pe = edges[i + 3];

        // Remove edges from edge source id
        // TODO: Is this corect?

        // Add replicate if not existent yet
        if (!(g.IsLocalFromGlobal(repl_source_id) || g.IsGhostFromGlobal(repl_source_id))) {
          g.AddVertex(repl_source_id);
          // std::cout << "R" << rank_ << " add repl (local) " << repl_source_id << std::endl;

          // Also add original source if not existent yet
          if (!(g.IsLocalFromGlobal(edge_source_id) || g.IsGhostFromGlobal(edge_source_id))) {
            g.AddGhostVertex(edge_source_id, edge_source_pe);
          }
          // Add local edge from replicate to source (and vice versa)
          g.AddEdge(g.GetLocalID(repl_source_id), edge_source_id, edge_source_pe);
          g.AddEdge(g.GetLocalID(edge_source_id), repl_source_id, rank_);
          // std::cout << "R" << rank_ << " add edge (repl) (" << repl_source_id << "," << edge_source_id << " (PE " << edge_source_pe << "))" << std::endl;
        }
      }
    }

    // Now add edges once all replicates are in place
    for (auto &kv : receive_buffers) {
      PEID edge_source_pe = kv.first;
      auto &edges = kv.second;

      // Iterate over received edges
      for (VertexID i = 0; i < edges.size(); i += 4) {
        VertexID edge_source_id = edges[i];
        VertexID repl_source_id = edges[i + 1];
        VertexID edge_target_id = edges[i + 2];
        PEID edge_target_pe = edges[i + 3];

        // Add any remaining ghosts
        if (!(g.IsLocalFromGlobal(edge_target_id) || g.IsGhostFromGlobal(edge_target_id))) {
          g.AddGhostVertex(edge_target_id, edge_target_pe);
        }

        // If this is a local edge only insert it for one of the two 
        g.AddEdge(g.GetLocalID(repl_source_id), edge_target_id, edge_target_pe);
        if (edge_target_pe != rank_) {
          g.AddEdge(g.GetLocalID(edge_target_id), repl_source_id, rank_);
        }
        // if (edge_target_pe == rank_ && repl_source_id < edge_target_id) {
        //   std::cout << "R" << rank_ << " add edge (direct local) (" << repl_source_id << "," << edge_target_id << " (PE " << edge_target_pe << "))" << std::endl;
        // } 
        // // Otherwise we can insert it without problems
        // else if (edge_target_pe != rank_){
        //   g.AddEdge(g.GetLocalID(repl_source_id), edge_target_id, edge_target_pe);
        //   g.AddEdge(g.GetLocalID(edge_target_id), repl_source_id, rank_);
        //   std::cout << "R" << rank_ << " add edge (direct remote) (" << repl_source_id << "," << edge_target_id << " (PE " << edge_target_pe << "))" << std::endl;
        // }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank_ == ROOT) std::cout << "[R" << rank_ << "] Finished replication" << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
  }

  void SplitHighDegreeVerticesSqrtPseudo(DynamicGraphCommunicator &g,
                                         const VertexID &avg_max_deg,
                                         const std::vector<std::pair<VertexID, VertexID>> &high_degree_vertices) {
    // Compute offset for IDs of replicated vertices
    VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
    global_repl_offset_ = num_global_vertices;
    VertexID vertex_offset = num_global_vertices * (rank_ + 1);
    VertexID repl_vertices_id = vertex_offset;

    // Default buffers for message exchange
    google::dense_hash_map<PEID, VertexBuffer> send_buffers;
    send_buffers.set_empty_key(EmptyKey);
    send_buffers.set_deleted_key(DeleteKey);
    google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
    receive_buffers.set_empty_key(EmptyKey);
    receive_buffers.set_deleted_key(DeleteKey);
    
    // New edges to replicated vertices
    std::vector<VertexID> local_edges;

    // Map for relinked vertices
    google::dense_hash_map<VertexID, std::vector<std::pair<VertexID, PEID>>> replicates_for_vertex;
    replicates_for_vertex.set_empty_key(EmptyKey);
    replicates_for_vertex.set_deleted_key(DeleteKey);

    // Map to answer relink messages
    google::dense_hash_map<VertexID, 
                           google::sparse_hash_map<VertexID, 
                                                   std::pair<VertexID, PEID>>> replicated_edges;
    replicated_edges.set_empty_key(EmptyKey);
    replicated_edges.set_deleted_key(DeleteKey);


    //////////////////////////////////////////
    // Compute distribution of replicates
    //////////////////////////////////////////
    std::mt19937 gen(config_.seed + rank_);
    std::uniform_int_distribution<PEID> dist(0, size_ - 1);
    for (VertexID i = 0; i < high_degree_vertices.size(); ++i) {
      VertexID v = high_degree_vertices[i].first;
      // Select random (different) target pe 
      PEID target_pe = dist(gen);
      while (target_pe == rank_) {
        target_pe = dist(gen);
      }
      // Increase replicate ID
      repl_vertices_id++;
      VertexID edge_counter = 0;

      // Add replicate
      replicates_for_vertex[g.GetGlobalID(v)].emplace_back(std::make_pair(repl_vertices_id, target_pe));

      g.ForallNeighbors(v, [&](const VertexID w) {
        // Create new replicate
        if (edge_counter >= config_.degree_threshold) {
          // Select new random target pe
          target_pe = dist(gen);
          while (target_pe == rank_) {
            target_pe = dist(gen);
          }
          // Increase replicate ID
          repl_vertices_id++;
          edge_counter = 0;

          // Add replicate
          replicates_for_vertex[g.GetGlobalID(v)].emplace_back(std::make_pair(repl_vertices_id, target_pe));
        }

        // For each edge store its associated replicate
        // std::cout << "R" << rank_ << " comp replicate (" << g.GetGlobalID(v) << "," << g.GetGlobalID(w) << ") -> " << "(" << repl_vertices_id << "," << g.GetGlobalID(w) << ") on PE " << target_pe << std::endl; 
        replicated_edges[g.GetGlobalID(v)][g.GetGlobalID(w)] = std::make_pair(repl_vertices_id, target_pe);

        // Send edge to corresponding neighbor
        send_buffers[g.GetPE(w)].emplace_back(g.GetGlobalID(w));
        send_buffers[g.GetPE(w)].emplace_back(g.GetGlobalID(v));
        send_buffers[g.GetPE(w)].emplace_back(repl_vertices_id);
        send_buffers[g.GetPE(w)].emplace_back(target_pe);
        edge_counter++;
      });
    }

    // Send updates
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);

    //////////////////////////////////////////
    // Process received edges and store relinks
    //////////////////////////////////////////
    google::dense_hash_map<VertexID, 
                           google::sparse_hash_map<VertexID, 
                                                   std::pair<VertexID, PEID>>> relink_edges;
    relink_edges.set_empty_key(EmptyKey);
    relink_edges.set_deleted_key(DeleteKey);
    for (auto &kv : receive_buffers) {
      PEID target_pe = kv.first;
      auto &updates = kv.second;
      for (VertexID i = 0; i < updates.size(); i += 4) {
        VertexID edge_source = updates[i];
        VertexID edge_target = updates[i + 1];
        VertexID new_edge_target = updates[i + 2];
        PEID new_target_pe = updates[i + 3];
        // std::cout << "R" << rank_ << " recv replicate (" << edge_source << "," << edge_target << ") -> " << "(" << edge_source << "," << new_edge_target << ") on PE " << new_target_pe << std::endl; 
        relink_edges[edge_source][edge_target] = std::make_pair(new_edge_target, new_target_pe);
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);

    //////////////////////////////////////////
    // Send edges to replicates
    // (With repeating the distribution computation)
    //////////////////////////////////////////
    gen.seed(config_.seed + rank_);
    std::mt19937 gen_san(config_.seed + rank_);
    std::uniform_int_distribution<PEID> dist_san(0, size_ - 1);
    for (VertexID i = 0; i < high_degree_vertices.size(); ++i) {
      VertexID v = high_degree_vertices[i].first;
      VertexID source_vertex = g.GetGlobalID(v);
      // Load replicates
      VertexID replicate_counter = 0;
      auto& replicates = replicates_for_vertex[source_vertex];
      // std::cout << "R" << rank_ << " split high degree v " << source_vertex << " d " << g.GetVertexDegree(v) << " thres " << config_.degree_threshold << std::endl;
      // Pick replicate
      VertexID repl_vertices_id = replicates[replicate_counter].first;
      PEID target_pe = replicates[replicate_counter].second;
      replicate_counter++;
      VertexID edge_counter = 0;
      // Start message with (Vertex, Replicate)
      // std::cout << "R" << rank_ << " repl v " << repl_vertices_id << " for v " << g.GetGlobalID(v) << " to PE " << target_pe  << std::endl;
      send_buffers[target_pe].emplace_back(source_vertex);
      send_buffers[target_pe].emplace_back(repl_vertices_id);
      send_buffers[target_pe].emplace_back(0);
      // Add ghost vertex if not already present
      if (!(g.IsLocalFromGlobal(repl_vertices_id) || g.IsGhostFromGlobal(repl_vertices_id))) {
        g.AddGhostVertex(repl_vertices_id, target_pe);
      }
      // Store edge
      local_edges.emplace_back(repl_vertices_id);
      local_edges.emplace_back(target_pe);

      g.ForallNeighbors(v, [&](const VertexID w) {
        VertexID target_vertex = g.GetGlobalID(w);
        if (edge_counter >= config_.degree_threshold) {
          // Load next replicate
          repl_vertices_id = replicates[replicate_counter].first;
          target_pe = replicates[replicate_counter].second;
          replicate_counter++;
          edge_counter = 0;
          // Start message with (Vertex, Replicate)
          // std::cout << "R" << rank_ << " repl v " << repl_vertices_id << " for v " << g.GetGlobalID(v) << " to PE " << target_pe  << std::endl;
          send_buffers[target_pe].emplace_back(g.GetGlobalID(v));
          send_buffers[target_pe].emplace_back(repl_vertices_id);
          send_buffers[target_pe].emplace_back(0);
          // Add ghost vertex if not already present
          if (!(g.IsLocalFromGlobal(repl_vertices_id) || g.IsGhostFromGlobal(repl_vertices_id))) {
            g.AddGhostVertex(repl_vertices_id, target_pe);
          }
          // Store edge
          local_edges.emplace_back(repl_vertices_id);
          local_edges.emplace_back(target_pe);
        }
        // Add message to buffer
        if (relink_edges.find(source_vertex) != relink_edges.end() 
            && (relink_edges[source_vertex].find(target_vertex) != relink_edges[source_vertex].end())) {
          // std::cout << "R" << rank_ << " send e (relink) (" << g.GetGlobalID(v) << "(" << repl_vertices_id << ")," << relink_edges[source_vertex][target_vertex].first << "(" << relink_edges[source_vertex][target_vertex].second << ")" << ") to PE " << target_pe << std::endl;
          send_buffers[target_pe].emplace_back(relink_edges[source_vertex][target_vertex].first);
          send_buffers[target_pe].emplace_back(relink_edges[source_vertex][target_vertex].second);
          send_buffers[target_pe].emplace_back(1);
        } else {
          // std::cout << "R" << rank_ << " send e (direct) (" << g.GetGlobalID(v) << "(" << repl_vertices_id << ")," << target_vertex << ") to PE " << target_pe  << std::endl;
          send_buffers[target_pe].emplace_back(target_vertex);
          send_buffers[target_pe].emplace_back(g.GetPE(w));
          send_buffers[target_pe].emplace_back(2);
        }
        edge_counter++;
      });
      // Replace edges with new edges to replicates
      g.RemoveAllEdges(v);
      for (VertexID i = 0; i < local_edges.size(); i+= 2) {
        g.AddEdge(v, local_edges[i], local_edges[i+1]);
        g.AddEdge(g.GetLocalID(local_edges[i]), g.GetGlobalID(v), rank_);
      }
      local_edges.clear();
    }

    // Send edges
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);

    //////////////////////////////////////////
    // Process received edges and send relink messages
    //////////////////////////////////////////
    google::dense_hash_map<VertexID, google::sparse_hash_set<VertexID>> edges_to_add;
    edges_to_add.set_empty_key(EmptyKey);
    edges_to_add.set_deleted_key(DeleteKey);
    global_repl_offset_ = num_global_vertices;
    for (auto &kv : receive_buffers) {
      PEID source_pe = kv.first;
      auto &edges = kv.second;

      // Group messages based on target vertices
      VertexID source_vertex;
      VertexID replicate_vertex;
      for (VertexID i = 0; i < edges.size(); i += 3) {
        VertexID target_vertex = edges[i];
        VertexID target_pe = edges[i + 1];
        VertexID message_type = edges[i + 2];
        // Check if message is replicate vertex ID
        if (message_type == 0) {
          source_vertex = target_vertex;
          replicate_vertex = target_pe;
          // Create replicate vertex
          g.AddVertex(replicate_vertex);
          // std::cout << "R" << rank_ << " added rep v " << replicate_vertex << " for v " << source_vertex << " lid " << g.GetLocalID(replicate_vertex) << std::endl;
          // Add ghost vertex if parent does not exist
          if (!(g.IsLocalFromGlobal(source_vertex) || g.IsGhostFromGlobal(source_vertex))) {
            g.AddGhostVertex(source_vertex, source_pe);
          }
          // Add local edge from replicate to source (and vice versa)
          // std::cout << "R" << rank_ << " add (1) e (" << replicate_vertex << "," << source_vertex << ")" << std::endl;
          g.AddEdge(g.GetLocalID(replicate_vertex), source_vertex, source_pe);
          g.AddEdge(g.GetLocalID(source_vertex), replicate_vertex, rank_);
        } else {
          // In case of an already relinked edge (type 1) we don't have to send an additional relink message
          if (message_type == 2) {
            // Send relink message to target
            send_buffers[target_pe].emplace_back(target_vertex);
            send_buffers[target_pe].emplace_back(source_vertex);
            send_buffers[target_pe].emplace_back(replicate_vertex);
          }
          // Add the final edge (we have to delay the insertion to get to know all replicates)
          local_edges.emplace_back(replicate_vertex);
          local_edges.emplace_back(target_vertex);
          local_edges.emplace_back(target_pe);
        }
      }
    }
    // Add stored edges
    for (VertexID i = 0; i < local_edges.size(); i+= 3) {
      VertexID replicate_vertex = local_edges[i];
      VertexID target_vertex = local_edges[i + 1];
      PEID target_pe = local_edges[i + 2];
      // This should make no problems since we know the correct target
      // std::cout << "R" << rank_ << " add (2) e (" << replicate_vertex << "," << target_vertex << ") PE(target)=" << target_pe  << std::endl;
      if (!(g.IsLocalFromGlobal(target_vertex) || g.IsGhostFromGlobal(target_vertex))) {
        g.AddGhostVertex(target_vertex, target_pe);
      }
      g.AddEdge(g.GetLocalID(replicate_vertex), target_vertex, target_pe);
      g.AddEdge(g.GetLocalID(target_vertex), replicate_vertex, rank_);
    }
    local_edges.clear();

    // Exchange relink messages
    CommunicationUtility::ClearBuffers(receive_buffers);
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);

    //////////////////////////////////////////
    // Check relink messages 
    //////////////////////////////////////////
    for (auto &kv : receive_buffers) {
      PEID target_pe = kv.first;
      auto &relink_buffer = kv.second;

      for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
        VertexID source_vertex = relink_buffer[i];
        VertexID old_target_vertex = relink_buffer[i + 1];
        VertexID new_target_vertex = relink_buffer[i + 2];
        // Case: Target is remote and was not replicated
        // We don't need to send answers, since the other side is already fixed
        if (!(g.IsLocalFromGlobal(new_target_vertex) || g.IsGhostFromGlobal(new_target_vertex))) {
          g.AddGhostVertex(new_target_vertex, target_pe);
        }
        bool relink_success = g.RelinkEdge(g.GetLocalID(source_vertex), old_target_vertex, new_target_vertex, target_pe);
        // std::cout << "R" << rank_ << " relink (" << source_vertex << "," << old_target_vertex << ") -> (" << source_vertex << "," << new_target_vertex << ") from R" << target_pe << std::endl;
        if (!relink_success) {
          std::cout << "R" << rank_ << " This shouldn't happen: Invalid (local answer) relink (" << source_vertex << "," << old_target_vertex << ") -> (" << source_vertex << "," << new_target_vertex << ") from R" << target_pe << std::endl;
        }
        g.AddEdge(g.GetLocalID(new_target_vertex), source_vertex, rank_);
        // Do we have to remove the edge from the old target (not when its a ghost)
        if (g.IsLocalFromGlobal(old_target_vertex)) {
          g.RemoveEdge(g.GetLocalID(old_target_vertex), g.GetLocalID(source_vertex));
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);
  }

  // void SplitHighDegreeVerticesSqrt(DynamicGraphCommunicator &g,
  //                                  const VertexID &avg_max_deg,
  //                                  const std::vector<VertexID> &high_degree_vertices) {
  //   // Compute offset for IDs of replicated vertices
  //   VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
  //   // TODO: Check if this offset if correct
  //   VertexID vertex_offset = num_global_vertices * (rank_ + 1);
  //   VertexID repl_vertices_id = vertex_offset;

  //   // Default buffers for message exchange
  //   google::dense_hash_map<PEID, VertexBuffer> send_buffers;
  //   send_buffers.set_empty_key(EmptyKey);
  //   send_buffers.set_deleted_key(DeleteKey);
  //   google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
  //   receive_buffers.set_empty_key(EmptyKey);
  //   receive_buffers.set_deleted_key(DeleteKey);
  //   
  //   // New edges to replicated vertices
  //   std::vector<VertexID> local_edges;

  //   // Map to answer relink messages
  //   google::dense_hash_map<VertexID, 
  //                          google::sparse_hash_map<VertexID, 
  //                                                  std::pair<VertexID, PEID>>> replicated_edges;
  //   replicated_edges.set_empty_key(EmptyKey);
  //   replicated_edges.set_deleted_key(DeleteKey);

  //   //////////////////////////////////////////
  //   // Send edges to replicates
  //   //////////////////////////////////////////
  //   // TODO: Optimize to keep local edges in separate buffer
  //   std::mt19937 gen(config_.seed + rank_);
  //   std::uniform_int_distribution<PEID> dist(0, size_ - 1);
  //   for (const VertexID &v : high_degree_vertices) {
  //     std::cout << "R" << rank_ << " split high degree v " << g.GetGlobalID(v) << " d " << g.GetVertexDegree(v) << " thres " << config_.degree_threshold << std::endl;
  //     VertexID edge_counter = 0;
  //     // Select random (different) target pe 
  //     PEID target_pe = dist(gen);
  //     while (target_pe == rank_) target_pe = dist(gen);
  //     // Start message with (Vertex, Replicate)
  //     repl_vertices_id++;
  //     send_buffers[target_pe].emplace_back(g.GetGlobalID(v));
  //     send_buffers[target_pe].emplace_back(repl_vertices_id);
  //     g.ForallNeighbors(v, [&](const VertexID w) {
  //       if (edge_counter >= config_.degree_threshold) {
  //         // Select new random target pe
  //         target_pe = dist(gen);
  //         while (target_pe == rank_) target_pe = dist(gen);
  //         // Increase replicate ID
  //         repl_vertices_id++;
  //         // Start message with (Vertex, Replicate)
  //         std::cout << "R" << rank_ << " repl v " << repl_vertices_id << " for v " << g.GetGlobalID(v) << " to PE " << target_pe  << std::endl;
  //         send_buffers[target_pe].emplace_back(g.GetGlobalID(v));
  //         send_buffers[target_pe].emplace_back(repl_vertices_id);
  //         // Add ghost vertex if not already present
  //         if (!(g.IsLocalFromGlobal(repl_vertices_id) || g.IsGhostFromGlobal(repl_vertices_id))) {
  //           g.AddGhostVertex(repl_vertices_id, target_pe);
  //         }
  //         // Store edge
  //         local_edges.emplace_back(repl_vertices_id);
  //         local_edges.emplace_back(target_pe);
  //         edge_counter = 0;
  //       }
  //       // Add message to buffer
  //       std::cout << "R" << rank_ << " send e (" << g.GetGlobalID(v) << "(" << repl_vertices_id << ")," << g.GetGlobalID(w) << ") to PE " << target_pe  << std::endl;
  //       send_buffers[target_pe].emplace_back(g.GetGlobalID(w));
  //       send_buffers[target_pe].emplace_back(g.GetPE(w));
  //       // Store edge to later answer relink messages
  //       // replicated_edges[g.GetGlobalID(v)][g.GetGlobalID(w)] = std::make_pair(repl_vertices_id, g.GetPE(w));
  //       // TODO: Is this correct
  //       replicated_edges[g.GetGlobalID(v)][g.GetGlobalID(w)] = std::make_pair(repl_vertices_id, target_pe);
  //       edge_counter++;
  //     });
  //     // Replace edges with new edges to replicates
  //     g.RemoveAllEdges(v);
  //     for (VertexID i = 0; i < local_edges.size(); i+= 2) {
  //       g.AddEdge(v, local_edges[i], local_edges[i+1]);
  //       g.AddEdge(g.GetLocalID(local_edges[i]), g.GetGlobalID(v), rank_);
  //     }
  //   }
  //   // Send edges
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);

  //   // MPI_Barrier(MPI_COMM_WORLD);
  //   // if (rank_ == ROOT) std::cout << "Finished distributing replicates" << std::endl;

  //   //////////////////////////////////////////
  //   // Process received edges and send relink messages
  //   //////////////////////////////////////////
  //   google::dense_hash_map<VertexID, google::sparse_hash_set<VertexID>> edges_to_add;
  //   edges_to_add.set_empty_key(EmptyKey);
  //   edges_to_add.set_deleted_key(DeleteKey);
  //   global_repl_offset_ = num_global_vertices;
  //   for (auto &kv : receive_buffers) {
  //     PEID source_pe = kv.first;
  //     auto &edges = kv.second;

  //     // Group messages based on target vertices
  //     VertexID source_vertex;
  //     VertexID replicate_vertex;
  //     for (VertexID i = 0; i < edges.size(); i += 2) {
  //       VertexID target_vertex = edges[i];
  //       VertexID target_pe = edges[i + 1];
  //       // Check if message is replicate vertex ID
  //       if (target_pe >= global_repl_offset_) {
  //         source_vertex = target_vertex;
  //         replicate_vertex = target_pe;
  //         // Create replicate vertex
  //         g.AddVertex(replicate_vertex);
  //         // std::cout << "R" << rank_ << " added rep v " << replicate_vertex << " for v " << source_vertex << " lid " << g.GetLocalID(replicate_vertex) << std::endl;
  //         // Add ghost vertex if parent does not exist
  //         if (!(g.IsLocalFromGlobal(source_vertex) || g.IsGhostFromGlobal(source_vertex))) {
  //           g.AddGhostVertex(source_vertex, source_pe);
  //         }
  //         // Add local edge from replicate to source (and vice versa)
  //         // std::cout << "R" << rank_ << " add e (" << replicate_vertex << "," << source_vertex << ")" << std::endl;
  //         g.AddEdge(g.GetLocalID(replicate_vertex), source_vertex, source_pe);
  //         g.AddEdge(g.GetLocalID(source_vertex), replicate_vertex, rank_);
  //         // }
  //       } else {
  //         // Add temporary edge (might still have the wrong target)
  //         // TODO: This ghost might become obsolete later
  //         if (!(g.IsLocalFromGlobal(target_vertex) || g.IsGhostFromGlobal(target_vertex))) {
  //           g.AddGhostVertex(target_vertex, target_pe);
  //         }
  //         // std::cout << "R" << rank_ << " add temp e (" << replicate_vertex << "," << target_vertex << ") PE(target)=" << target_pe  << std::endl;
  //         // TODO: We potentially have to remove the (target, replicate) edge when it is relinked later
  //         g.AddEdge(g.GetLocalID(replicate_vertex), target_vertex, target_pe);
  //         g.AddEdge(g.GetLocalID(target_vertex), replicate_vertex, rank_);
  //         // Send relink message to target
  //         send_buffers[target_pe].emplace_back(target_vertex);
  //         send_buffers[target_pe].emplace_back(source_vertex);
  //         send_buffers[target_pe].emplace_back(replicate_vertex);
  //       }
  //     }
  //   }
  //   // Exchange relink messages
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);

  //   // MPI_Barrier(MPI_COMM_WORLD);
  //   // if (rank_ == ROOT) std::cout << "Finished sending relink messages" << std::endl;

  //   //////////////////////////////////////////
  //   // Check relink messages for necessary answer messages
  //   //////////////////////////////////////////
  //   for (auto &kv : receive_buffers) {
  //     PEID target_pe = kv.first;
  //     auto &relink_buffer = kv.second;

  //     for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
  //       VertexID source_vertex = relink_buffer[i];
  //       VertexID old_target_vertex = relink_buffer[i + 1];
  //       VertexID new_target_vertex = relink_buffer[i + 2];

  //       // If so send answer message back to sender
  //       if (replicated_edges.find(source_vertex) != replicated_edges.end() 
  //           && (replicated_edges[source_vertex].find(old_target_vertex) != replicated_edges[source_vertex].end())) {
  //         // Case: Target is remote and was replicated
  //         // Send back (new target, source vertex, replicated vertex, replicated vertex PE)
  //         // Receiving PE can then relink edge (new target, source vertex) to (new target, replicated vertex)
  //         send_buffers[target_pe].emplace_back(new_target_vertex);
  //         send_buffers[target_pe].emplace_back(source_vertex);
  //         send_buffers[target_pe].emplace_back(replicated_edges[source_vertex][old_target_vertex].first);
  //         send_buffers[target_pe].emplace_back(replicated_edges[source_vertex][old_target_vertex].second);
  //       }
  //       else {
  //         // Case: Target is remote and was not replicated
  //         if (!(g.IsLocalFromGlobal(new_target_vertex) || g.IsGhostFromGlobal(new_target_vertex))) {
  //           g.AddGhostVertex(new_target_vertex, target_pe);
  //         }
  //         bool relink_success = g.RelinkEdge(g.GetLocalID(source_vertex), old_target_vertex, new_target_vertex, target_pe);
  //         if (!relink_success) {
  //           std::cout << "R" << rank_ << " This shouldn't happen: Invalid (local answer) relink (" << source_vertex << "," << old_target_vertex << ") -> (" << source_vertex << "," << new_target_vertex << ") from R" << target_pe << std::endl;
  //         }
  //         // TODO: Also add backward edge here (is this correct?)
  //         g.AddEdge(g.GetLocalID(new_target_vertex), source_vertex, rank_);
  //       }
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   // Send second round of relink (answer) messages
  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);

  //   // MPI_Barrier(MPI_COMM_WORLD);
  //   // if (rank_ == ROOT) std::cout << "Finished sending answer messages" << std::endl;

  //   //////////////////////////////////////////
  //   // TODO: Send (final) updates from replicates to neighbors
  //   //////////////////////////////////////////
  //   for (auto &kv : receive_buffers) {
  //     PEID sender_pe = kv.first;
  //     auto &answer_buffer = kv.second;
  //     for (VertexID i = 0; i < answer_buffer.size(); i += 4) {
  //       VertexID source_vertex = answer_buffer[i];
  //       VertexID old_target_vertex = answer_buffer[i + 1];
  //       VertexID new_target_vertex = answer_buffer[i + 2];
  //       PEID new_target_pe = answer_buffer[i + 3];

  //       if (!(g.IsLocalFromGlobal(new_target_vertex) || g.IsGhostFromGlobal(new_target_vertex))) {
  //         g.AddGhostVertex(new_target_vertex, new_target_pe);
  //       }
  //       bool relink_success = g.RelinkEdge(g.GetLocalID(source_vertex), old_target_vertex, new_target_vertex, new_target_pe);
  //       if (!relink_success) {
  //         std::cout << "R" << rank_ << " This shouldn't happen: Invalid (second round) relink (" << source_vertex << "(lid=" << g.GetLocalID(source_vertex) << ")," << old_target_vertex << ") -> (" << source_vertex << "," << new_target_vertex << ") from R" << sender_pe << std::endl;
  //       }
  //       // TODO: Also add backward edge here (is this correct?)
  //       g.AddEdge(g.GetLocalID(new_target_vertex), source_vertex, rank_);
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);

  //   // MPI_Barrier(MPI_COMM_WORLD);
  //   // if (rank_ == ROOT) std::cout << "Finished relinking remaining vertices" << std::endl;
  // }

  // void SplitHighDegreeVerticesIntoTrees(DynamicGraphCommunicator &g,
  //                                       const VertexID &avg_max_deg,
  //                                       const std::vector<VertexID> &high_degree_vertices) {
  //   // Compute offset for IDs for replicated vertices
  //   VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
  //   VertexID vertex_offset = num_global_vertices * (rank_ + size_);

  //   // Default buffers for message exchange
  //   google::dense_hash_map<PEID, VertexBuffer> send_buffers;
  //   send_buffers.set_empty_key(-1);
  //   send_buffers.set_deleted_key(-1);
  //   google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
  //   receive_buffers.set_empty_key(-1);
  //   receive_buffers.set_deleted_key(-1);
  //   
  //   // New edges to replicated vertices
  //   google::dense_hash_map<VertexID, std::vector<VertexID>> local_edges;
  //   local_edges.set_empty_key(-1);
  //   local_edges.set_deleted_key(-1);
  //   google::dense_hash_map<VertexID, std::vector<VertexID>> parent_edges;
  //   parent_edges.set_empty_key(-1);
  //   parent_edges.set_deleted_key(-1);

  //   // Split adjacency list for high degree vertices
  //   VertexID repl_vertices_id = vertex_offset;

  //   // TODO: Make these command line options
  //   float edge_quotient = 2.0;
  //   VertexID edge_threshold = 2;

  //   // Hashmap for grouping edges before building send buffers
  //   google::dense_hash_map<VertexID, std::vector<VertexID>> vertex_messages;
  //   vertex_messages.set_empty_key(-1);
  //   vertex_messages.set_deleted_key(-1);

  //   // Hashmap for storing parents of vertices
  //   google::dense_hash_map<VertexID, PEID> parent;
  //   parent.set_empty_key(-1);
  //   parent.set_deleted_key(-1);
  //   
  //   google::dense_hash_map<VertexID, 
  //                          google::sparse_hash_map<VertexID, 
  //                                                  std::pair<VertexID, PEID>>> high_degree_edge_distribution;
  //   high_degree_edge_distribution.set_empty_key(-1);
  //   high_degree_edge_distribution.set_deleted_key(-1);

  //   google::dense_hash_set<VertexID> tree_leaf_set;
  //   tree_leaf_set.set_empty_key(-1);
  //   tree_leaf_set.set_deleted_key(-1);

  //   ComputeInitialBinomialPartitioning(g, high_degree_vertices, parent, vertex_messages);

  //   PropagateEdgesAlongTree(g, 
  //                           edge_threshold, edge_quotient, 
  //                           repl_vertices_id,
  //                           local_edges, parent_edges,
  //                           parent, tree_leaf_set,
  //                           vertex_messages,
  //                           send_buffers, receive_buffers);

  //   GatherEdgesOnLeaves(local_edges, parent_edges, tree_leaf_set, vertex_messages);

  //   GatherEdgeDistributionOnRoot(g, 
  //                                high_degree_edge_distribution,
  //                                local_edges, parent_edges,
  //                                parent,
  //                                vertex_messages,
  //                                send_buffers, receive_buffers);

  //   AddLocalEdges(g, local_edges);

  //   SendInitialRelinkMessages(g, high_degree_edge_distribution, local_edges, send_buffers, receive_buffers);

  //   ReplyRelinkMessages(g, high_degree_edge_distribution, send_buffers, receive_buffers);

  //   SendFinalRelinkMessages(g, send_buffers, receive_buffers);

  //   ApplyRelinks(g, receive_buffers);
  // }

  // void ComputeInitialBinomialPartitioning(DynamicGraphCommunicator &g,
  //                                         const std::vector<VertexID> &high_degree_vertices,
  //                                         google::dense_hash_map<VertexID, PEID> &parents,
  //                                         google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers) {
  //   // Initial distribution of high degree vertices
  //   for (const VertexID &v : high_degree_vertices) {
  //     VertexID v_deg = g.GetVertexDegree(v);
  //     VertexID num_parts = tlx::integer_log2_ceil(v_deg);
  //     if (num_parts <= 1) continue;
  //     if (rank_ != 0 && rank_ != 1)
  //     std::cout << "R" << rank_ << " split v " << g.GetGlobalID(v) << " deg(v)=" << v_deg << " in " << num_parts << " parts" << std::endl;

  //     // Gather all non-local neighbors
  //     g.ForallNeighbors(v, [&](const VertexID w) {
  //       if (!g.IsLocal(w)) { 
  //         propagation_buffers[g.GetGlobalID(v)].emplace_back(g.GetGlobalID(w));
  //         propagation_buffers[g.GetGlobalID(v)].emplace_back(g.GetPE(w));
  //       }
  //     });
  //     parents[g.GetGlobalID(v)] = rank_;
  //     g.RemoveAllEdges(v);
  //   }
  // }

  // void PropagateEdgesAlongTree(DynamicGraphCommunicator &g,
  //                              const VertexID minimal_message_size,
  //                              const float message_size_fraction,
  //                              const VertexID replicate_offset,
  //                              google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
  //                              google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
  //                              google::dense_hash_map<VertexID, PEID> &parents,
  //                              google::dense_hash_set<VertexID> &leaves,
  //                              google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers,
  //                              google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
  //                              google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   // First loop: Reroute message to leafs of binomial tree
  //   VertexID replicate_counter = 0;
  //   int converged_globally = 0;
  //   int converged_locally = 0;
  //   while (converged_globally == 0) {
  //     converged_locally = 1;
  //     for (auto &kv : propagation_buffers) {
  //       VertexID source = kv.first;
  //       VertexID replicate = source;
  //       if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
  //         replicate = replicated_vertices_[source];
  //       }
  //       auto &edges = kv.second;
  //       if (edges.size() <= 0) continue;

  //       // Push half of the messages in the corresponding send buffer
  //       // if (rank_ == 6 || rank_ == 10) {
  //       //   std::cout << "R" << rank_ << " has " << edges.size() / 2 << " remaining edges (size=" << edges.size() << ") for v " << source << std::endl;
  //       if ((edges.size() / 2) > minimal_message_size) {
  //         VertexID message_size = ceil((edges.size() / 2) / message_size_fraction) * 2;
  //         PEID current_target_pe = size_;
  //         for (VertexID i = 0; i < edges.size(); i += 2) {
  //           VertexID target = edges[i];
  //           PEID target_pe = edges[i + 1];

  //           // Pick first PE as receiver
  //           // TODO: Develop some better scheme for picking receivers
  //           if (current_target_pe >= size_ && target_pe != rank_) {
  //             current_target_pe = target_pe;
  //           }

  //           if (target_pe != rank_) {
  //             send_buffers[current_target_pe].emplace_back(source);
  //             send_buffers[current_target_pe].emplace_back(replicate);
  //             send_buffers[current_target_pe].emplace_back(target);
  //             send_buffers[current_target_pe].emplace_back(target_pe);
  //             if (current_target_pe >= size_) {
  //               std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (down propagation) PE R" << current_target_pe << std::endl;
  //             }
  //             leaves.erase(source);
  //             // TODO: -- or -= 2?
  //             message_size -= 2;
  //           } else {
  //             local_edges[source].emplace_back(replicate);
  //             local_edges[source].emplace_back(target);
  //             local_edges[source].emplace_back(target_pe);
  //           }
  //           // Mark elements for deletion
  //           edges[i] = std::numeric_limits<PEID>::max() - 1;
  //           edges[i + 1] = std::numeric_limits<PEID>::max() - 1;
  //         }
  //         // Remove processed edges from buffer
  //         edges.erase(std::remove(edges.begin(), edges.end(), std::numeric_limits<PEID>::max() - 1), edges.end());

  //         message_size = std::min(message_size, static_cast<VertexID>(edges.size()));
  //         if (message_size > 0) {
  //           for (VertexID i = 0; i < message_size; i += 2) {
  //             VertexID target = edges[i];
  //             PEID target_pe = edges[i + 1];

  //             // Message (source, replicate, target, pe(target))
  //             send_buffers[current_target_pe].emplace_back(source);
  //             send_buffers[current_target_pe].emplace_back(replicate);
  //             send_buffers[current_target_pe].emplace_back(target);
  //             send_buffers[current_target_pe].emplace_back(target_pe);
  //             if (current_target_pe >= size_) {
  //               std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (down propagation) PE R" << current_target_pe << std::endl;
  //             }
  //             leaves.erase(source);
  //           }
  //           // Remove processed edges from buffer
  //           edges.erase(edges.begin(), edges.begin() + message_size);
  //         }
  //       } 
  //       // Insert remaining edges locally
  //       else {
  //         for (VertexID i = 0; i < edges.size(); i += 2) {
  //           VertexID replicate = source;
  //           if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
  //             replicate = replicated_vertices_[source];
  //           }
  //           local_edges[source].emplace_back(replicate);
  //           local_edges[source].emplace_back(edges[i]);
  //           local_edges[source].emplace_back(edges[i + 1]);
  //         }
  //         edges.clear();
  //       }
  //     }

  //     comm_timer_.Restart();
  //     CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //     comm_time_ += comm_timer_.Elapsed();
  //     CommunicationUtility::ClearBuffers(send_buffers);

  //     // Process received messages
  //     for (auto &kv : receive_buffers) {
  //       PEID sender_pe = kv.first;
  //       auto &edges = kv.second;

  //       // Group messages based on target vertices
  //       for (VertexID i = 0; i < edges.size(); i += 4) {
  //         VertexID source = edges[i];
  //         VertexID parent_replicate = edges[i + 1];
  //         VertexID target = edges[i + 2];
  //         PEID target_pe = edges[i + 3];
  //         leaves.insert(source);

  //         // Add replicate
  //         if (replicated_vertices_.find(source) == end(replicated_vertices_)) {
  //           VertexID replicate = replicate_offset + replicate_counter++;
  //           replicated_vertices_[source] = replicate;
  //           g.AddVertex(replicate);
  //           parents[replicate] = sender_pe;
  //           // Add ghost vertex if parent replicate does not exist
  //           if (!(g.IsLocalFromGlobal(parent_replicate) || g.IsGhostFromGlobal(parent_replicate))) {
  //             g.AddGhostVertex(parent_replicate, sender_pe);
  //           }
  //           // Add local edge from replicate to parent replicate
  //           parent_edges[source].emplace_back(replicate);
  //           parent_edges[source].emplace_back(parent_replicate);
  //           parent_edges[source].emplace_back(sender_pe);
  //           g.AddEdge(g.GetLocalID(replicate), parent_replicate, sender_pe);
  //         }

  //         // Add messages to propagation buffer
  //         propagation_buffers[source].emplace_back(target);
  //         propagation_buffers[source].emplace_back(target_pe);
  //         converged_locally = 0;
  //       }
  //     }
  //     CommunicationUtility::ClearBuffers(receive_buffers);

  //     // Repeat until convergence
  //     comm_timer_.Restart();
  //     MPI_Allreduce(&converged_locally,
  //                   &converged_globally,
  //                   1,
  //                   MPI_INT,
  //                   MPI_MIN,
  //                   MPI_COMM_WORLD);
  //     comm_time_ += comm_timer_.Elapsed();
  //   }
  //   propagation_buffers.clear();
  // }

  // void GatherEdgesOnLeaves(google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
  //                          google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
  //                          google::dense_hash_set<VertexID> &leaves,
  //                          google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers) {
  //   // Initial grouping of leaf vertices
  //   if (leaves.size() > 0) {
  //     for (const VertexID &v: leaves) {
  //       // if (rank_ == 0 || rank_ == 1 || rank_ == 2) {
  //       //   std::cout << "R" << rank_ << " is tree leaf for v " << v << std::endl;
  //       // }
  //       for (VertexID i = 0; i < local_edges[v].size(); i+= 3) {
  //         VertexID replicate = local_edges[v][i];
  //         VertexID target = local_edges[v][i + 1];
  //         PEID target_pe = local_edges[v][i + 2];

  //         // Gather all neighbors
  //         propagation_buffers[v].emplace_back(replicate);
  //         propagation_buffers[v].emplace_back(target);
  //         propagation_buffers[v].emplace_back(rank_);
  //       }

  //       for (VertexID i = 0; i < parent_edges[v].size(); i+= 3) {
  //         VertexID replicate = parent_edges[v][i];
  //         VertexID target = parent_edges[v][i + 1];
  //         PEID target_pe = parent_edges[v][i + 2];

  //         // Gather all neighbors
  //         propagation_buffers[v].emplace_back(replicate);
  //         propagation_buffers[v].emplace_back(target);
  //         propagation_buffers[v].emplace_back(rank_);
  //       }
  //     }
  //   }
  // }

  // void GatherEdgeDistributionOnRoot(DynamicGraphCommunicator &g,
  //                                   google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
  //                                   google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
  //                                   google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
  //                                   google::dense_hash_map<VertexID, PEID> &parents,
  //                                   google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers,
  //                                   google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
  //                                   google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   // Second loop: Gather message on root
  //   int converged_globally = 0;
  //   int converged_locally = 0;
  //   google::dense_hash_set<VertexID> added_local;
  //   added_local.set_empty_key(-1);
  //   added_local.set_deleted_key(-1);
  //   while (converged_globally == 0) {
  //     converged_locally = 1;
  //     for (auto &kv : propagation_buffers) {
  //       VertexID source = kv.first;
  //       VertexID replicate = source;
  //       if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
  //         replicate = replicated_vertices_[source];
  //       }
  //       // TODO: Strictly parent has to be decided on an edge to edge base
  //       // (PE can receive several blocks of the same adjacency list from different vertices)
  //       PEID parent_pe = parents[replicate];
  //       if (parent_pe == rank_) {
  //         continue;
  //       }
  //       auto &edges = kv.second;
  //       if (edges.size() <= 0) continue;

  //       // Send edges back to parent
  //       for (VertexID i = 0; i < edges.size(); i += 3) {
  //         send_buffers[parent_pe].emplace_back(source);
  //         send_buffers[parent_pe].emplace_back(replicate);
  //         send_buffers[parent_pe].emplace_back(edges[i]);
  //         send_buffers[parent_pe].emplace_back(edges[i + 1]);
  //         send_buffers[parent_pe].emplace_back(edges[i + 2]);
  //         if (parent_pe >= size_) {
  //           std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (up propagation) PE R" << parent_pe << std::endl;
  //         }
  //       }
  //       edges.clear();
  //     }
  //     propagation_buffers.clear();

  //     comm_timer_.Restart();
  //     CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //     comm_time_ += comm_timer_.Elapsed();
  //     CommunicationUtility::ClearBuffers(send_buffers);

  //     // Process received messages
  //     for (auto &kv : receive_buffers) {
  //       PEID sender_pe = kv.first;
  //       auto &edges = kv.second;

  //       // Group messages based on target vertices
  //       for (VertexID i = 0; i < edges.size(); i += 5) {
  //         VertexID source = edges[i];
  //         VertexID replicate = source;
  //         if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
  //           replicate = replicated_vertices_[source];
  //         }
  //         VertexID child_replicate = edges[i + 1];
  //         VertexID leaf_replicate = edges[i + 2];
  //         VertexID target = edges[i + 3];
  //         PEID leaf_pe = edges[i + 4];

  //         // Add local edge from replicate to child replicate
  //         if (target == replicate) {
  //           // Add ghost vertex if child replicate does not exist
  //           if (!(g.IsLocalFromGlobal(child_replicate) || g.IsGhostFromGlobal(child_replicate))) {
  //             g.AddGhostVertex(child_replicate, sender_pe);
  //           }
  //           g.AddEdge(g.GetLocalID(replicate), child_replicate, sender_pe);
  //         }

  //         // Store mapping of edge to PE if root
  //         if (g.IsLocalFromGlobal(source)) {
  //           edge_distribution[source][target] = std::make_pair(leaf_replicate, leaf_pe);
  //         } else {
  //           // Add corresponding local edges to propagation buffer
  //           if (added_local.find(source) == end(added_local)) {
  //             for (VertexID i = 0; i < local_edges[source].size(); i+= 3) {
  //               VertexID replicate = local_edges[source][i];
  //               VertexID target = local_edges[source][i + 1];
  //               PEID target_pe = local_edges[source][i + 2];

  //               propagation_buffers[source].emplace_back(replicate);
  //               propagation_buffers[source].emplace_back(target);
  //               propagation_buffers[source].emplace_back(rank_);
  //               converged_locally = 0;
  //             }

  //             for (VertexID i = 0; i < parent_edges[source].size(); i+= 3) {
  //               VertexID replicate = parent_edges[source][i];
  //               VertexID target = parent_edges[source][i + 1];
  //               PEID target_pe = parent_edges[source][i + 2];

  //               propagation_buffers[source].emplace_back(replicate);
  //               propagation_buffers[source].emplace_back(target);
  //               propagation_buffers[source].emplace_back(rank_);
  //               converged_locally = 0;
  //             }
  //             added_local.insert(source);
  //           }
  //           propagation_buffers[source].emplace_back(leaf_replicate);
  //           propagation_buffers[source].emplace_back(target);
  //           propagation_buffers[source].emplace_back(leaf_pe);
  //           converged_locally = 0;
  //         }
  //       }
  //     }
  //     CommunicationUtility::ClearBuffers(receive_buffers);

  //     // Repeat until convergence
  //     comm_timer_.Restart();
  //     MPI_Allreduce(&converged_locally,
  //                   &converged_globally,
  //                   1,
  //                   MPI_INT,
  //                   MPI_MIN,
  //                   MPI_COMM_WORLD);
  //     comm_time_ += comm_timer_.Elapsed();
  //   }
  //   propagation_buffers.clear();
  // }

  // void AddLocalEdges(DynamicGraphCommunicator &g,
  //                    google::dense_hash_map<VertexID, VertexBuffer> &local_edges) {
  //   // Add local edges
  //   for (auto &kv : local_edges) {
  //     VertexID source = kv.first;
  //     auto &edges = kv.second;
  //     for (VertexID i = 0; i < edges.size(); i+= 3) {
  //       VertexID replicate = edges[i];
  //       VertexID target = edges[i + 1];
  //       PEID target_pe = edges[i + 2];
  //       // Add ghost vertex if target does not exist
  //       if (!(g.IsLocalFromGlobal(target) || g.IsGhostFromGlobal(target))) {
  //         g.AddGhostVertex(target, target_pe);
  //       }
  //       g.AddEdge(g.GetLocalID(replicate), target, target_pe);
  //     }
  //   }
  // }

  // void SendInitialRelinkMessages(DynamicGraphCommunicator &g,
  //                                google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
  //                                google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
  //                                google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
  //                                google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   // Send first round of relink messages
  //   for (auto &kv : local_edges) {
  //     VertexID source = kv.first;
  //     auto &edges = kv.second;
  //     for (VertexID i = 0; i < edges.size(); i+= 3) {
  //       VertexID replicate = edges[i];
  //       VertexID target = edges[i + 1];
  //       PEID target_pe = edges[i + 2];
  //       if (replicate == source) continue;

  //       if (target_pe != rank_) {
  //         send_buffers[target_pe].emplace_back(target);
  //         send_buffers[target_pe].emplace_back(source);
  //         send_buffers[target_pe].emplace_back(replicate);
  //         if (target_pe >= size_) {
  //           std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (first round) PE R" << target_pe << std::endl;
  //         }
  //       } else {
  //         if (edge_distribution.find(target) == edge_distribution.end()
  //             || (edge_distribution.find(target) != edge_distribution.end()
  //               && edge_distribution[target].find(source) == edge_distribution[target].end())) {
  //           bool relink_success = g.RelinkEdge(g.GetLocalID(target), source, replicate, rank_);
  //           if (!relink_success) {
  //             std::cout << "R" << rank_ << " This shouldn't happen: Invalid (first round) relink (" << target << "," << source << ") -> (" << target << "," << replicate << ") from R" << rank_ << " isLocal(target)=" << g.IsLocalFromGlobal(target) << " isLocal(source)=" << g.IsLocalFromGlobal(source) << " isLocal(replicate)=" << g.IsLocalFromGlobal(replicate) << std::endl;
  //           }
  //         } else {
  //           send_buffers[rank_].emplace_back(target);
  //           send_buffers[rank_].emplace_back(source);
  //           send_buffers[rank_].emplace_back(replicate);
  //         }
  //       }
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);
  // }

  // void ReplyRelinkMessages(DynamicGraphCommunicator &g,
  //                          google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
  //                          google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
  //                          google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   // Check relink messages for necessary answer messages
  //   for (auto &kv : receive_buffers) {
  //     PEID target_pe = kv.first;
  //     auto &relink_buffer = kv.second;

  //     for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
  //       VertexID source = relink_buffer[i];
  //       VertexID old_target = relink_buffer[i + 1];
  //       VertexID new_target = relink_buffer[i + 2];

  //       // Check if both endpoints have been replicated
  //       if (edge_distribution.find(source) != edge_distribution.end() 
  //           && (edge_distribution[source].find(old_target) != edge_distribution[source].end())) {
  //           send_buffers[target_pe].emplace_back(edge_distribution[source][old_target].first);
  //           send_buffers[target_pe].emplace_back(edge_distribution[source][old_target].second);
  //           send_buffers[target_pe].emplace_back(old_target);
  //           send_buffers[target_pe].emplace_back(new_target);
  //           if (target_pe >= size_) {
  //             std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (answer reply) PE R" << target_pe << std::endl;
  //           }
  //       }
  //       // Otherwise, perform remote relinking operation
  //       else {
  //         if (!(g.IsLocalFromGlobal(new_target) || g.IsGhostFromGlobal(new_target))) {
  //           g.AddGhostVertex(new_target, target_pe);
  //         }
  //         bool relink_success = g.RelinkEdge(g.GetLocalID(source), old_target, new_target, target_pe);
  //         if (!relink_success) {
  //           std::cout << "R" << rank_ << " This shouldn't happen: Invalid (local answer) relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") from R" << target_pe << std::endl;
  //         }
  //       }
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   // Send second round of relink (answer) messages
  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);
  // }

  // void SendFinalRelinkMessages(DynamicGraphCommunicator &g,
  //                              google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
  //                              google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   for (auto &kv : receive_buffers) {
  //     PEID sender_pe = kv.first;
  //     auto &answer_buffer = kv.second;
  //     for (VertexID i = 0; i < answer_buffer.size(); i += 4) {
  //       VertexID source = answer_buffer[i];
  //       PEID source_pe = answer_buffer[i + 1];
  //       VertexID old_target = answer_buffer[i + 2];
  //       VertexID new_target = answer_buffer[i + 3];

  //       // Reroute relink to pe that now holds the initial edge
  //       send_buffers[source_pe].emplace_back(source);
  //       send_buffers[source_pe].emplace_back(old_target);
  //       send_buffers[source_pe].emplace_back(new_target);
  //       if (source_pe >= size_) {
  //         std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (second round) PE R" << source_pe << std::endl;
  //       }
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  //   receive_buffers[rank_] = send_buffers[rank_];
  //   send_buffers[rank_].clear();

  //   // Send final round of relink messages
  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);
  // }

  // void ApplyRelinks(DynamicGraphCommunicator &g,
  //                   google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
  //   for (auto &kv : receive_buffers) {
  //     PEID sender_pe = kv.first;
  //     auto &relink_buffer = kv.second;

  //     for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
  //       VertexID source = relink_buffer[i];
  //       VertexID old_target = relink_buffer[i + 1];
  //       VertexID new_target = relink_buffer[i + 2];

  //       if (!(g.IsLocalFromGlobal(new_target) || g.IsGhostFromGlobal(new_target))) {
  //         g.AddGhostVertex(new_target, sender_pe);
  //       }
  //       bool relink_success = g.RelinkEdge(g.GetLocalID(source), old_target, new_target, sender_pe);
  //       if (!relink_success) {
  //         std::cout << "R" << rank_ << " This shouldn't happen: Invalid (second round) relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") from R" << sender_pe << std::endl;
  //       }
  //     }
  //   }
  //   CommunicationUtility::ClearBuffers(receive_buffers);
  // }

  void UpdateInterfaceVertices(DynamicGraphCommunicator &g) {
    // Check if PEs are still connected
    google::dense_hash_set<PEID> neighboring_pes;
    neighboring_pes.set_empty_key(-1);
    neighboring_pes.set_deleted_key(-1);
    g.ForallLocalVertices([&](const VertexID v) {
      bool ghost_neighbor = false;
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (!g.IsLocal(w)) {
          if (neighboring_pes.find(g.GetPE(v)) == neighboring_pes.end()) {
            neighboring_pes.insert(g.GetPE(v));
          }
          ghost_neighbor = true;
        }
      });
      g.SetInterface(v, ghost_neighbor);
    });

    // Update PEs
    for (const auto &pe : neighboring_pes) {
      g.SetAdjacentPE(pe, true);
    }
  }

  void RemoveReplicatedVertices(DynamicGraphCommunicator &g) {
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetGlobalID(v) >= global_repl_offset_) {
        g.SetActive(v, false);
      }
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
