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
        comm_time_(0.0),
        send_volume_(0),
        recv_volume_(0) { 
    replicated_vertices_.set_empty_key(EmptyKey);
    replicated_vertices_.set_deleted_key(DeleteKey);
  }

  virtual ~ExponentialContraction() {
    delete exp_contraction_;
    exp_contraction_ = nullptr;
  };

  template <typename GraphType>
  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    contraction_timer_.Restart();
    if constexpr (std::is_same<GraphType, StaticGraph>::value) {
      FindLocalComponents<StaticGraph>(g, g_labels);
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
          IOUtility::PrintGraphParams(cag, config_, rank_, size_);
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
        // Get stats
        comm_time_ += contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + cag.GetCommTime() + g.GetCommTime();
        send_volume_ += contraction.GetSendVolume()
                      + exp_contraction_->GetSendVolume()
                      + cag.GetSendVolume() + g.GetSendVolume();
        recv_volume_ += contraction.GetReceiveVolume()
                      + exp_contraction_->GetReceiveVolume()
                      + cag.GetReceiveVolume() + g.GetReceiveVolume();
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
        FindLocalComponents<StaticGraph>(cag, cag_labels);
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
          IOUtility::PrintGraphParams(ccag, config_, rank_, size_);
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
        // Get stats
        comm_time_ += first_contraction.GetCommTime() + second_contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + ccag.GetCommTime() + cag.GetCommTime() + g.GetCommTime();
        send_volume_ += first_contraction.GetSendVolume() + second_contraction.GetSendVolume()
                      + exp_contraction_->GetSendVolume()
                      + ccag.GetSendVolume() + cag.GetSendVolume() + g.GetSendVolume();
        recv_volume_ += first_contraction.GetReceiveVolume() + second_contraction.GetReceiveVolume()
                      + exp_contraction_->GetReceiveVolume()
                      + ccag.GetReceiveVolume() + cag.GetReceiveVolume() + g.GetReceiveVolume();
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
          // SampleHighDegreeNeighborhoods(lcag);
          OutputStats<DynamicGraphCommunicator>(lcag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Distributing high degree vertices took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }
          IOUtility::PrintGraphParams(lcag, config_, rank_, size_);

          contraction_timer_.Restart();
          google::dense_hash_map<VertexID, VertexID> lcag_labels;
          lcag_labels.set_empty_key(EmptyKey);
          lcag_labels.set_deleted_key(DeleteKey);
          FindLocalComponents<DynamicGraphCommunicator>(lcag, lcag_labels);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Finding local components on lcag took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }

          // Contract again
          contraction_timer_.Restart();
          CAGBuilder<DynamicGraphCommunicator, google::dense_hash_map<VertexID, VertexID>> 
            local_contraction(lcag, lcag_labels, rank_, size_);
          auto hd_lcag 
            = local_contraction.BuildLocalComponentGraph<DynamicGraphCommunicator>();
          hd_lcag.ResetCommunicator();
          OutputStats<DynamicGraphCommunicator>(lcag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Building high degree local cag took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }

          // Keep contraction labeling for later
          exp_contraction_ = new DynamicContraction(hd_lcag, rank_, size_);

          // Main decomposition algorithm
          contraction_timer_.Restart(); 
          PerformDecomposition(hd_lcag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Resolving connectivity took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }

          ApplyToLocalComponents(hd_lcag, lcag);
          RemoveReplicatedVertices(lcag);
        } else {
          // Keep contraction labeling for later
          exp_contraction_ = new DynamicContraction(lcag, rank_, size_);

          // Main decomposition algorithm
          contraction_timer_.Restart(); 
          PerformDecomposition(lcag);
          if (rank_ == ROOT) {
            std::cout << "[STATUS] |- Resolving connectivity took " 
                      << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
          }
        }
        ApplyToLocalComponents(lcag, g, g_labels);
        // Get stats
        comm_time_ += local_contraction.GetCommTime() 
                      + exp_contraction_->GetCommTime()
                      + lcag.GetCommTime() + g.GetCommTime();
        send_volume_ += local_contraction.GetSendVolume()
                      + exp_contraction_->GetSendVolume()
                      + lcag.GetSendVolume() + g.GetSendVolume();
        recv_volume_ += local_contraction.GetReceiveVolume() 
                      + exp_contraction_->GetReceiveVolume()
                      + lcag.GetReceiveVolume() + g.GetReceiveVolume();
      }
    } 
  }

  void Output(DynamicGraphCommunicator &g) {
    g.OutputLabels();
  }

  inline float GetCommTime() {
    return comm_time_; 
  }

  inline VertexID GetSendVolume() {
    return send_volume_; 
  }

  inline VertexID GetReceiveVolume() {
    return recv_volume_; 
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
  Timer comm_timer_;
  float comm_time_;
  Timer iteration_timer_;
  Timer contraction_timer_;
  VertexID send_volume_;
  VertexID recv_volume_;
  
  // Contraction
  DynamicContraction *exp_contraction_;

  // Node replication
  google::dense_hash_map<VertexID, VertexID> replicated_vertices_;
  VertexID global_repl_offset_;

  void PerformDecomposition(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart(); 
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    rng_offset_ = global_vertices;
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

  template <typename GraphType>
  void FindLocalComponents(GraphType &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      VertexID gid = g.GetGlobalID(v);
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS<GraphType>(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  template <typename GraphType>
  void FindLocalComponents(GraphType &g, google::dense_hash_map<VertexID, VertexID> &label) {
    google::dense_hash_map<VertexID, bool> marked;
    marked.set_empty_key(EmptyKey);
    marked.set_deleted_key(DeleteKey);
    google::dense_hash_map<VertexID, VertexID> parent;
    parent.set_empty_key(EmptyKey);
    parent.set_deleted_key(DeleteKey);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
      marked[v] = false;
      parent[v] = 0;
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility::BFS<GraphType>(g, v, marked, parent);
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
    
    // std::exponential_distribution<LPFloat> distribution(config_.beta);
    // std::mt19937
    //     generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
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
      // Skip replicated vertices
      if (replicated_vertices_.find(g.GetGlobalID(v)) == replicated_vertices_.end()) {
        std::exponential_distribution<LPFloat> distribution(config_.beta);
        std::mt19937
            generator(static_cast<unsigned int>(g.GetGlobalID(v) + config_.seed + iteration_ * rng_offset_));
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
      }
    });

    // Iterate over replicates and assign same variate as root
    for (auto &kv : replicated_vertices_) {
      VertexID v = g.GetLocalID(kv.first);
      LPFloat weight = 
#ifdef TIEBREAK_DEGREE
        // static_cast<LPFloat>(g.GetVertexDegree(v) / g.GetMaxDegree());
        // static_cast<LPFloat>(log2(g.GetNumberOfVertices()) / g.GetVertexDegree(v));
        1.0;
#else
        1.0;
#endif
      std::exponential_distribution<LPFloat> distribution(config_.beta);
      std::mt19937
          generator(static_cast<unsigned int>(kv.second + config_.seed + iteration_ * rng_offset_));
      g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
                             g.GetVertexLabel(v),
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)},
                         true);
    }
    
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
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
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
    // std::exponential_distribution<LPFloat> distribution(config_.beta);
    // std::mt19937
    //     generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
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
      // Skip replicated vertices
      if (replicated_vertices_.find(g.GetGlobalID(v)) == replicated_vertices_.end()) {
        std::exponential_distribution<LPFloat> distribution(config_.beta);
        std::mt19937
            generator(static_cast<unsigned int>(g.GetGlobalID(v) + config_.seed + iteration_ * rng_offset_));
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
      }
    });

    // Iterate over replicates and assign same variate as root
    for (auto &kv : replicated_vertices_) {
      VertexID v = g.GetLocalID(kv.first);
      LPFloat weight = 
#ifdef TIEBREAK_DEGREE
        // static_cast<LPFloat>(g.GetVertexDegree(v) / g.GetMaxDegree());
        // static_cast<LPFloat>(log2(g.GetNumberOfVertices()) / g.GetVertexDegree(v));
        1.0;
#else
        1.0;
#endif
      std::exponential_distribution<LPFloat> distribution(config_.beta);
      std::mt19937
          generator(static_cast<unsigned int>(kv.second + config_.seed + iteration_ * rng_offset_));
      g.SetVertexPayload(v, {static_cast<VertexID>(weight * distribution(generator)),
                             g.GetVertexLabel(v),
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)},
                         true);
    }
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
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
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
                              DynamicGraphCommunicator &g) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g.SetVertexLabel(v, cag.GetVertexLabel(cv));
    });
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

  void SampleHighDegreeNeighborhoods(DynamicGraphCommunicator &g) {
    std::vector<std::pair<VertexID, VertexID>> high_degree_vertices;
    VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_, comm_time_);
    Utility::SelectHighDegreeVertices(g, avg_max_deg, high_degree_vertices);

    for (const auto &vd : high_degree_vertices) {
      std::cout << "R" << rank_ << " sample neighborhood v " << vd.first << " d " << vd.second << " (ad " << avg_max_deg << ") s " << vd.second * config_.neighborhood_sampling_factor << std::endl;
      g.SampleVertexNeighborhood(vd.first, config_.neighborhood_sampling_factor);
    }
  }

  void DistributeHighDegreeVertices(DynamicGraphCommunicator &g) {
    // Determine high degree vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    std::vector<std::pair<VertexID, VertexID>> high_degree_vertices;
    VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_, comm_time_);
    // Use sqrt(n) as a degree threshold
    config_.degree_threshold = static_cast<VertexID>(config_.degree_threshold*sqrt(global_vertices));
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |- High degree threshold " << config_.degree_threshold
                << " [TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    Utility::SelectHighDegreeVertices(g, config_.degree_threshold, high_degree_vertices);
    std::cout << "[STATUS] |- R" << rank_ << " Num high degree to distributed " 
              << high_degree_vertices.size() << std::endl;
    // Split high degree vertices into one layer of proxies with degree sqrt(n)
    SplitHighDegreeVerticesSqrtEdge(g, avg_max_deg, high_degree_vertices);
  }

  void SplitHighDegreeVerticesSqrtEdge(DynamicGraphCommunicator &g,
                                         const VertexID &avg_max_deg,
                                         std::vector<std::pair<VertexID, VertexID>> &high_degree_vertices) {
    // Compute offset for IDs of replicated vertices
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

    comm_time_ += CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    send_volume_ += CommunicationUtility::ClearBuffers(send_buffers);

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
    recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers);

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
          replicated_edges[g.GetGlobalID(v)][g.GetGlobalID(w)] = std::make_pair(repl_vertices_id, target_pe);
        }
      });
    }

    // We have to relink our replicated edges based on the recomputed information 
    // I.e. if for an edge (v,w) both endpoints have been replicated (replicated_edges[v][w] and replicated_edges[w][v] exist)
    // In this case update the edge with the correct target
    // Finally, send the edges to their respective replicates
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
                    g.AddVertex(replicate.first);
                    replicated_vertices_[replicate.first] = edge_source_id;
                    // Also add original source if not existent yet
                    if (!(g.IsLocalFromGlobal(edge_source_id) || g.IsGhostFromGlobal(edge_source_id))) {
                      g.AddGhostVertex(edge_source_id, edge_source_pe);
                    }
                    // Add local edge from replicate to source (and vice versa)
                    g.AddEdge(g.GetLocalID(replicate.first), edge_source_id, edge_source_pe);
                    g.AddEdge(g.GetLocalID(edge_source_id), replicate.first, rank_);
                  } else {
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
    
    // Send updates
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_time_ += CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    send_volume_ += CommunicationUtility::ClearBuffers(send_buffers);

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

        // Add replicate if not existent yet
        if (!(g.IsLocalFromGlobal(repl_source_id) || g.IsGhostFromGlobal(repl_source_id))) {
          g.AddVertex(repl_source_id);
          replicated_vertices_[repl_source_id] = edge_source_id;

          // Also add original source if not existent yet
          if (!(g.IsLocalFromGlobal(edge_source_id) || g.IsGhostFromGlobal(edge_source_id))) {
            g.AddGhostVertex(edge_source_id, edge_source_pe);
          }
          // Add local edge from replicate to source (and vice versa)
          g.AddEdge(g.GetLocalID(repl_source_id), edge_source_id, edge_source_pe);
          g.AddEdge(g.GetLocalID(edge_source_id), repl_source_id, rank_);
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
      }
    }
    recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers);
  }

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
    VertexID n_local = g.GetNumberOfVertices();
    EdgeID m_local = g.GetNumberOfEdges();
    VertexID n_global = g.GatherNumberOfGlobalVertices();
    EdgeID m_global = g.GatherNumberOfGlobalEdges();

    VertexID highest_degree = 0;
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetVertexDegree(v) > highest_degree) {
        highest_degree = g.GetVertexDegree(v);
      }
    });

    // Determine min/maximum cut size
    EdgeID cut_local = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    comm_timer_.Restart();
    MPI_Reduce(&cut_local, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&cut_local, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);
    comm_time_ += comm_timer_.Elapsed();

    std::cout << "LOCAL TEMP IMPUT"
              << " rank=" << rank_
              << " n=" << n_local 
              << " m=" << m_local 
              << " c=" << cut_local 
              << " max_d=" << highest_degree << std::endl;
    if (rank_ == ROOT) {
      std::cout << "GLOBAL TEMP IMPUT"
                << " s=" << config_.seed 
                << " p=" << size_ 
                << " n=" << n_global
                << " m=" << m_global
                << " c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }

};

#endif
