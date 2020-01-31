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

#include <sys/sysinfo.h>
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
    replicated_vertices_.set_empty_key(-1);
    replicated_vertices_.set_deleted_key(-1);
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
        OutputStats<DynamicGraphCommunicator>(cag);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Building cag took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }

        if (config_.replicate_high_degree) {
          contraction_timer_.Restart();
          // MPI_Barrier(MPI_COMM_WORLD);
          // cag.OutputLocal();
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
      } else {
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
      }
    } else if constexpr (std::is_same<GraphType, DynamicGraphCommunicator>::value) {
      if (config_.replicate_high_degree) {
        contraction_timer_.Restart();
        DistributeHighDegreeVertices(g);
        OutputStats<DynamicGraphCommunicator>(g);
        if (rank_ == ROOT) {
          std::cout << "[STATUS] |- Distributing high degree vertices took " 
                    << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
        }
      }

      exp_contraction_ = new DynamicContraction(g, rank_, size_);

      PerformDecomposition(g);

      if (config_.replicate_high_degree) {
        RemoveReplicatedVertices(g);
      }

      g.ForallLocalVertices([&](const VertexID v) {
          g_labels[v] = g.GetVertexLabel(v);
      });
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
  google::dense_hash_map<VertexID, VertexID> replicated_vertices_;

  void PerformDecomposition(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart(); 
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_timer_.Restart();
      iteration_++;
      if (global_vertices <= config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        RunContraction(g);
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

  void RunContraction(DynamicGraphCommunicator &g) {
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
        RunContraction(g);
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
    vertex_map.set_empty_key(-1);
    vertex_map.set_deleted_key(-1);
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
    std::vector<VertexID> high_degree_vertices;
    VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_);
    Utility::SelectHighDegreeVertices(g, config_.degree_threshold * avg_max_deg, high_degree_vertices);

    for (const VertexID &v : high_degree_vertices) {
      g.SampleVertexNeighborhood(v, config_.neighborhood_sampling_factor);
    }
  }

  void DistributeHighDegreeVertices(DynamicGraphCommunicator &g) {
    // if (rank_ == 0) {
    //   g.OutputLocal();
    // }
    // Determine high degree vertices
    std::vector<VertexID> high_degree_vertices;
    VertexID avg_max_deg = Utility::ComputeAverageMaxDegree(g, rank_, size_);
    // Utility::SelectHighDegreeVertices(g, config_.degree_threshold * avg_max_deg, high_degree_vertices);
    Utility::SelectHighDegreeVertices(g, config_.degree_threshold, high_degree_vertices);
    std::cout << "R" << rank_ << " num high degree " << high_degree_vertices.size() << std::endl;
    
    // Split high degree vertices into binomial trees
    SplitHighDegreeVerticesIntoTrees(g, avg_max_deg, high_degree_vertices);
    // Split vertices into stars if they have a high degree
    // SplitHighDegreeVerticesIntoStars(g, avg_max_deg, high_degree_vertices);
  }

  // void SplitHighDegreeVerticesIntoStars(DynamicGraphCommunicator &g,
  //                                       const VertexID &avg_max_deg,
  //                                       const std::vector<VertexID> &high_degree_vertices) {
  //   // Compute offset for IDs for replicated vertices
  //   VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
  //   VertexID vertex_offset = num_global_vertices * (rank_ + size_);

  //   google::dense_hash_map<PEID, VertexBuffer> send_buffers;
  //   send_buffers.set_empty_key(-1);
  //   google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
  //   receive_buffers.set_empty_key(-1);

  //   // Split adjacency list for high degree vertices
  //   VertexID clamped_avg_max_deg = std::max<VertexID>(avg_max_deg, 2);
  //   for (const VertexID &v : high_degree_vertices) {
  //     VertexID v_deg = g.GetVertexDegree(v);
  //     VertexID num_parts = static_cast<VertexID>(ceil(v_deg / clamped_avg_max_deg));
  //     std::cout << "R" << rank_ << " split v " << g.GetGlobalID(v) << " deg(v)=" << v_deg << " in " << num_parts << " rounds" << std::endl;
  //     VertexID part_size = v_deg / num_parts;
  //     ComputeEdgePartitioning(g, v, num_parts, part_size, vertex_offset, send_buffers);
  //   }

  //   // Inform neighbors about replication 
  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);

  //   // Insert replicated vertices and local edges
  //   ProcessAdjLists(g, num_global_vertices, receive_buffers, send_buffers);
  //   CommunicationUtility::ClearBuffers(receive_buffers);

  //   // Inform neighbors about new vertices
  //   comm_timer_.Restart();
  //   CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
  //   comm_time_ += comm_timer_.Elapsed();
  //   CommunicationUtility::ClearBuffers(send_buffers);

  //   // Relink edges to new replicated vertices
  //   ProcessRelinking(g, receive_buffers);
  //   CommunicationUtility::ClearBuffers(receive_buffers);

  //   // Update interface vertices and neihgboring PEs
  //   UpdateInterfaceVertices(g);
  // }

  void SplitHighDegreeVerticesIntoTrees(DynamicGraphCommunicator &g,
                                        const VertexID &avg_max_deg,
                                        const std::vector<VertexID> &high_degree_vertices) {
    // Compute offset for IDs for replicated vertices
    VertexID num_global_vertices = g.GatherNumberOfGlobalVertices();
    VertexID vertex_offset = num_global_vertices * (rank_ + size_);

    // Default buffers for message exchange
    google::dense_hash_map<PEID, VertexBuffer> send_buffers;
    send_buffers.set_empty_key(-1);
    send_buffers.set_deleted_key(-1);
    google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
    receive_buffers.set_empty_key(-1);
    receive_buffers.set_deleted_key(-1);
    
    // New edges to replicated vertices
    google::dense_hash_map<VertexID, std::vector<VertexID>> local_edges;
    local_edges.set_empty_key(-1);
    local_edges.set_deleted_key(-1);
    google::dense_hash_map<VertexID, std::vector<VertexID>> parent_edges;
    parent_edges.set_empty_key(-1);
    parent_edges.set_deleted_key(-1);

    // Split adjacency list for high degree vertices
    VertexID repl_vertices_id = vertex_offset;

    // TODO: Make these command line options
    float edge_quotient = 2.0;
    VertexID edge_threshold = 2;

    // Hashmap for grouping edges before building send buffers
    google::dense_hash_map<VertexID, std::vector<VertexID>> vertex_messages;
    vertex_messages.set_empty_key(-1);
    vertex_messages.set_deleted_key(-1);

    // Hashmap for storing parents of vertices
    google::dense_hash_map<VertexID, PEID> parent;
    parent.set_empty_key(-1);
    parent.set_deleted_key(-1);
    
    google::dense_hash_map<VertexID, 
                           google::sparse_hash_map<VertexID, 
                                                   std::pair<VertexID, PEID>>> high_degree_edge_distribution;
    high_degree_edge_distribution.set_empty_key(-1);
    high_degree_edge_distribution.set_deleted_key(-1);

    google::dense_hash_set<VertexID> tree_leaf_set;
    tree_leaf_set.set_empty_key(-1);
    tree_leaf_set.set_deleted_key(-1);

    ComputeInitialBinomialPartitioning(g, high_degree_vertices, parent, vertex_messages);

    PropagateEdgesAlongTree(g, 
                            edge_threshold, edge_quotient, 
                            repl_vertices_id,
                            local_edges, parent_edges,
                            parent, tree_leaf_set,
                            vertex_messages,
                            send_buffers, receive_buffers);

    GatherEdgesOnLeaves(local_edges, parent_edges, tree_leaf_set, vertex_messages);

    GatherEdgeDistributionOnRoot(g, 
                                 high_degree_edge_distribution,
                                 local_edges, parent_edges,
                                 parent,
                                 vertex_messages,
                                 send_buffers, receive_buffers);

    AddLocalEdges(g, local_edges);

    SendInitialRelinkMessages(g, high_degree_edge_distribution, local_edges, send_buffers, receive_buffers);

    ReplyRelinkMessages(g, high_degree_edge_distribution, send_buffers, receive_buffers);

    SendFinalRelinkMessages(g, send_buffers, receive_buffers);

    ApplyRelinks(g, receive_buffers);
  }

  // void ComputeEdgePartitioning(DynamicGraphCommunicator &g,
  //                              const VertexID &vertex_id,
  //                              const VertexID &num_parts,
  //                              const VertexID &part_size,
  //                              const VertexID &repl_vertices_id,
  //                              google::dense_hash_map<PEID, VertexBuffer> &send_buffers) {
  //   // Temporary vector for storing new local edges 
  //   std::vector<VertexID> local_edges;

  //   google::dense_hash_map<PEID, std::vector<VertexID>> edges_for_pe;
  //   edges_for_pe.set_empty_key(-1);
  //   g.ForallNeighbors(vertex_id, [&](const VertexID w) {
  //       if (!g.IsLocal(w)) {
  //         edges_for_pe[g.GetPE(w)].emplace_back(g.GetGlobalID(w));
  //       }
  //   });
  //   
  //   // Copy hash map to vector
  //   std::vector<std::pair<PEID, VertexID>> num_edges_for_pe;
  //   for (const auto &kv : edges_for_pe) {
  //     PEID pe = kv.first;
  //     VertexID num_edges = kv.second.size();
  //     num_edges_for_pe.emplace_back(pe, num_edges);
  //   }

  //   PEID num_neighboring_pes = edges_for_pe.size();
  //   std::random_shuffle(num_edges_for_pe.begin(), num_edges_for_pe.end());
  //   std::sort(num_edges_for_pe.begin(), num_edges_for_pe.end(), [](const auto& lhs, const auto& rhs) {
  //       return lhs.second > rhs.second;
  //   });

  //   VertexID repl_vertices_counter = 0;
  //   VertexID remaining_parts = num_parts;
  //   // Check if we can send single large part to PE
  //   for (const auto &kv : num_edges_for_pe) {
  //     PEID pe = kv.first;
  //     if (pe >= size_) {
  //       std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (initial partition) PE R" << pe << std::endl;
  //     }
  //     VertexID num_edges = kv.second;
  //     if (num_edges >= part_size) {
  //       send_buffers[pe].emplace_back(g.GetGlobalID(vertex_id));
  //       send_buffers[pe].emplace_back(repl_vertices_id + repl_vertices_counter);
  //       local_edges.emplace_back(repl_vertices_id + repl_vertices_counter);
  //       local_edges.emplace_back(pe);
  //       repl_vertices_counter++;
  //       for (const auto &v : edges_for_pe[pe]) {
  //         send_buffers[pe].emplace_back(std::move(v));
  //         send_buffers[pe].emplace_back(pe);
  //       }
  //       remaining_parts -= floor(num_edges / part_size);
  //       edges_for_pe[pe].clear();
  //     }
  //   }

  //   // Distribute the remaining parts greedily between neighboring PEs
  //   while (remaining_parts > 0) {
  //     // Greedily select PE with most remaining edges
  //     PEID current_target_pe = std::numeric_limits<PEID>::max();
  //     int remaining_part_size = part_size;
  //     for (const auto &kv : num_edges_for_pe) {
  //       PEID pe = kv.first;
  //       PEID num_edges = kv.second;
  //       if (num_edges > 0 && remaining_part_size > 0) {
  //         // Set current PE as target for following messages
  //         if (current_target_pe >= size_) {
  //           current_target_pe = pe;
  //           send_buffers[current_target_pe].emplace_back(g.GetGlobalID(vertex_id));
  //           send_buffers[current_target_pe].emplace_back(repl_vertices_id + repl_vertices_counter);
  //           if (current_target_pe >= size_) {
  //             std::cout << "R" << rank_ << " This shouldn't happen: Invalid target PE R" << current_target_pe << std::endl;
  //           }
  //           local_edges.emplace_back(repl_vertices_id + repl_vertices_counter);
  //           local_edges.emplace_back(current_target_pe);
  //           repl_vertices_counter++;
  //         }
  //         // Add edges in current block
  //         for (const auto &v : edges_for_pe[pe]) {
  //           send_buffers[current_target_pe].emplace_back(std::move(v));
  //           send_buffers[current_target_pe].emplace_back(pe);
  //           if (current_target_pe >= size_) {
  //             std::cout << "R" << rank_ << " This shouldn't happen: Invalid target PE R" << current_target_pe << std::endl;
  //           }
  //         }
  //         remaining_part_size -= num_edges; 
  //         if (remaining_part_size <= 0) {
  //           PEID current_target_pe = std::numeric_limits<PEID>::max();
  //           // First step guarantees that edgelists are now smaller than a part
  //           remaining_parts--;
  //         }
  //         edges_for_pe[pe].clear();
  //       }
  //     }
  //   }
  //   edges_for_pe.clear();

  //   // Update the adjacency list of the original (replicated) vertex
  //   g.RemoveAllEdges(vertex_id);
  //   for (VertexID i = 0; i < local_edges.size(); i += 2) {
  //     VertexID target =  local_edges[i];
  //     VertexID target_pe =  local_edges[i+1];
  //     g.AddGhostVertex(target, target_pe);
  //     g.AddEdge(vertex_id, target, target_pe);
  //   }
  // }

  // void ProcessAdjLists(DynamicGraphCommunicator &g, 
  //                      const VertexID &offset,
  //                      google::dense_hash_map<PEID, VertexBuffer> &receive_buffer,
  //                      google::dense_hash_map<PEID, VertexBuffer> &send_buffer) {
  //   // Clear send buffers
  //   for (const auto &kv : send_buffer) {
  //     PEID pe = kv.first;
  //     send_buffer[pe].clear();
  //   }
  //   send_buffer.clear();
  //   // Process incoming vertices/edges
  //   for (const auto &kv : receive_buffer) {
  //     PEID pe = kv.first;
  //     auto& buffer = kv.second;
  //     if (buffer.size() > 0) {
  //       VertexID source, copy_vertex;
  //       for (VertexID i = 0; i < buffer.size(); i+=2) {
  //         // New source
  //         // Note: This might not work if number of global vertices is smaller than number of PEs
  //         if (buffer[i+1] >= offset) {
  //           source = buffer[i];
  //           copy_vertex = buffer[i+1];
  //           VertexID local_copy_id = g.AddVertex(copy_vertex);
  //           replicated_vertices_[source] = copy_vertex;
  //           g.AddEdge(local_copy_id, source, pe);
  //         } else {
  //           VertexID target = buffer[i];
  //           VertexID target_pe = buffer[i+1];
  //           // Neighbor is local vertex
  //           if (target_pe == rank_) {
  //             g.RelinkEdge(g.GetLocalID(target), source, copy_vertex, rank_);
  //           // Neighbor is remote vertex
  //           } else {
  //             g.AddGhostVertex(target, target_pe);
  //             // Target PE needs to reroute (target, source) from (original) PE to (target, copy) on sender PE
  //             send_buffer[target_pe].emplace_back(target);
  //             send_buffer[target_pe].emplace_back(source);
  //             send_buffer[target_pe].emplace_back(copy_vertex);
  //           }
  //           g.AddEdge(g.GetLocalID(copy_vertex), target, target_pe);
  //         }
  //       }
  //     }
  //   }
  // }

  // void ProcessRelinking(DynamicGraphCommunicator &g, 
  //                     google::dense_hash_map<PEID, VertexBuffer> &receive_buffer) {
  //   // Process incoming vertices/edges
  //   for (const auto &kv : receive_buffer) {
  //     PEID pe = kv.first;
  //     auto& buffer = kv.second;
  //     if (buffer.size() > 0) {
  //       for (VertexID i = 0; i < buffer.size(); i+=3) {
  //         VertexID source = buffer[i];
  //         VertexID old_target = buffer[i+1];
  //         VertexID new_target = buffer[i+2];

  //         if (!g.IsGhostFromGlobal(new_target)) {
  //           g.AddGhostVertex(new_target, pe);
  //         }
  //         g.RelinkEdge(g.GetLocalID(source), old_target, new_target, pe);
  //       }
  //     }
  //   }
  // }

  void ComputeInitialBinomialPartitioning(DynamicGraphCommunicator &g,
                                          const std::vector<VertexID> &high_degree_vertices,
                                          google::dense_hash_map<VertexID, PEID> &parents,
                                          google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers) {
    // Initial distribution of high degree vertices
    for (const VertexID &v : high_degree_vertices) {
      VertexID v_deg = g.GetVertexDegree(v);
      VertexID num_parts = tlx::integer_log2_ceil(v_deg);
      if (num_parts <= 1) continue;
      if (rank_ != 0 && rank_ != 1)
      std::cout << "R" << rank_ << " split v " << g.GetGlobalID(v) << " deg(v)=" << v_deg << " in " << num_parts << " parts" << std::endl;

      // Gather all non-local neighbors
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (!g.IsLocal(w)) { 
          propagation_buffers[g.GetGlobalID(v)].emplace_back(g.GetGlobalID(w));
          propagation_buffers[g.GetGlobalID(v)].emplace_back(g.GetPE(w));
        }
      });
      parents[g.GetGlobalID(v)] = rank_;
      g.RemoveAllEdges(v);
    }
  }

  void PropagateEdgesAlongTree(DynamicGraphCommunicator &g,
                               const VertexID minimal_message_size,
                               const float message_size_fraction,
                               const VertexID replicate_offset,
                               google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
                               google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
                               google::dense_hash_map<VertexID, PEID> &parents,
                               google::dense_hash_set<VertexID> &leaves,
                               google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers,
                               google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                               google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    // First loop: Reroute message to leafs of binomial tree
    VertexID replicate_counter = 0;
    int converged_globally = 0;
    int converged_locally = 0;
    while (converged_globally == 0) {
      converged_locally = 1;
      for (auto &kv : propagation_buffers) {
        VertexID source = kv.first;
        VertexID replicate = source;
        if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
          replicate = replicated_vertices_[source];
        }
        auto &edges = kv.second;
        if (edges.size() <= 0) continue;

        // Push half of the messages in the corresponding send buffer
        // if (rank_ == 6 || rank_ == 10) {
        //   std::cout << "R" << rank_ << " has " << edges.size() / 2 << " remaining edges (size=" << edges.size() << ") for v " << source << std::endl;
        // }
        if ((edges.size() / 2) > minimal_message_size) {
          VertexID message_size = ceil((edges.size() / 2) / message_size_fraction) * 2;
          PEID current_target_pe = size_;
          for (VertexID i = 0; i < message_size; i += 2) {
            VertexID target = edges[i];
            PEID target_pe = edges[i + 1];

            // Pick first PE as receiver
            // TODO: Develop some better scheme for picking receivers
            if (current_target_pe >= size_ && target_pe != rank_) {
              current_target_pe = target_pe;
            }

            // Message (source, replicate, target, pe(target))
            if (target_pe != rank_) {
              send_buffers[current_target_pe].emplace_back(source);
              send_buffers[current_target_pe].emplace_back(replicate);
              send_buffers[current_target_pe].emplace_back(target);
              send_buffers[current_target_pe].emplace_back(target_pe);
              if (current_target_pe >= size_) {
                std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (down propagation) PE R" << current_target_pe << std::endl;
              }
              // if (rank_ == 0 || rank_ == 2 || rank_ == 1) {
              //   std::cout << "R" << rank_ << " propagate (" << replicate << " (repl of " << source << ")," << target << ") target(pe)=" << target_pe << " to R" << current_target_pe << std::endl;
              // }
              leaves.erase(source);
            } else {
              local_edges[source].emplace_back(replicate);
              local_edges[source].emplace_back(target);
              local_edges[source].emplace_back(target_pe);
              // if (rank_ == 0 || rank_ == 2 || rank_ == 1) {
              //   std::cout << "R" << rank_ << " add (" << replicate << " (repl of " << source << ")," << target << ") target(pe)=" << target_pe << " to local edges (during propagation)" << std::endl;
              // }
            }
          }
          // Remove processed edges from buffer
          edges.erase(edges.begin(), edges.begin() + message_size);
        } 
        // Insert remaining edges locally
        else {
          for (VertexID i = 0; i < edges.size(); i += 2) {
            VertexID replicate = source;
            if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
              replicate = replicated_vertices_[source];
            }
            local_edges[source].emplace_back(replicate);
            local_edges[source].emplace_back(edges[i]);
            local_edges[source].emplace_back(edges[i + 1]);
            // if (rank_ == 0 || rank_ == 2 || rank_ == 1) {
            //   std::cout << "R" << rank_ << " add (" << replicate << " (repl of " << source << ")," << edges[i] << ") target(pe)=" << edges[i+1] << " to local edges (remaining local edge)" << std::endl;
            // }
          }
          edges.clear();
        }
      }

      comm_timer_.Restart();
      CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
      comm_time_ += comm_timer_.Elapsed();
      CommunicationUtility::ClearBuffers(send_buffers);

      // Process received messages
      for (auto &kv : receive_buffers) {
        PEID sender_pe = kv.first;
        auto &edges = kv.second;

        // Group messages based on target vertices
        // if (rank_ == 6 || rank_ == 10) {
        //   std::cout << "R" << rank_ << " recv " << edges.size() / 4 << " edges from R" << sender_pe << std::endl;
        // }
        for (VertexID i = 0; i < edges.size(); i += 4) {
          VertexID source = edges[i];
          VertexID parent_replicate = edges[i + 1];
          VertexID target = edges[i + 2];
          PEID target_pe = edges[i + 3];
          leaves.insert(source);

          // Add replicate
          if (replicated_vertices_.find(source) == end(replicated_vertices_)) {
            VertexID replicate = replicate_offset + replicate_counter++;
            replicated_vertices_[source] = replicate;
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " add replicate " << replicate << " of v " << source << " parent(replicate)=" << sender_pe << std::endl;
            // }
            g.AddVertex(replicate);
            parents[replicate] = sender_pe;
            // Add ghost vertex if parent replicate does not exist
            if (!(g.IsLocalFromGlobal(parent_replicate) || g.IsGhostFromGlobal(parent_replicate))) {
              g.AddGhostVertex(parent_replicate, sender_pe);
            }
            // Add local edge from replicate to parent replicate
            parent_edges[source].emplace_back(replicate);
            parent_edges[source].emplace_back(parent_replicate);
            parent_edges[source].emplace_back(sender_pe);
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " add (" << replicate << "," << parent_replicate << ") target(pe)=" << sender_pe << " to local edges (during propagation)" << std::endl;
            // }
            g.AddEdge(g.GetLocalID(replicate), parent_replicate, sender_pe);
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " add edge (" << replicate << "," << parent_replicate << ") pe(target)=" << sender_pe << std::endl;
            // }
          }

          // Add messages to propagation buffer
          propagation_buffers[source].emplace_back(target);
          propagation_buffers[source].emplace_back(target_pe);
          converged_locally = 0;
        }
      }
      CommunicationUtility::ClearBuffers(receive_buffers);

      // Repeat until convergence
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
    }
    propagation_buffers.clear();
  }

  void GatherEdgesOnLeaves(google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
                           google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
                           google::dense_hash_set<VertexID> &leaves,
                           google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers) {
    // Initial grouping of leaf vertices
    if (leaves.size() > 0) {
      for (const VertexID &v: leaves) {
        // if (rank_ == 0 || rank_ == 1 || rank_ == 2) {
        //   std::cout << "R" << rank_ << " is tree leaf for v " << v << std::endl;
        // }
        for (VertexID i = 0; i < local_edges[v].size(); i+= 3) {
          VertexID replicate = local_edges[v][i];
          VertexID target = local_edges[v][i + 1];
          PEID target_pe = local_edges[v][i + 2];

          // Gather all neighbors
          propagation_buffers[v].emplace_back(replicate);
          propagation_buffers[v].emplace_back(target);
          propagation_buffers[v].emplace_back(rank_);
        }

        for (VertexID i = 0; i < parent_edges[v].size(); i+= 3) {
          VertexID replicate = parent_edges[v][i];
          VertexID target = parent_edges[v][i + 1];
          PEID target_pe = parent_edges[v][i + 2];

          // Gather all neighbors
          propagation_buffers[v].emplace_back(replicate);
          propagation_buffers[v].emplace_back(target);
          propagation_buffers[v].emplace_back(rank_);
        }
      }
    }
  }

  void GatherEdgeDistributionOnRoot(DynamicGraphCommunicator &g,
                                    google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
                                    google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
                                    google::dense_hash_map<VertexID, VertexBuffer> &parent_edges,
                                    google::dense_hash_map<VertexID, PEID> &parents,
                                    google::dense_hash_map<VertexID, VertexBuffer> &propagation_buffers,
                                    google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                                    google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    // Second loop: Gather message on root
    int converged_globally = 0;
    int converged_locally = 0;
    google::dense_hash_set<VertexID> added_local;
    added_local.set_empty_key(-1);
    added_local.set_deleted_key(-1);
    while (converged_globally == 0) {
      converged_locally = 1;
      for (auto &kv : propagation_buffers) {
        VertexID source = kv.first;
        VertexID replicate = source;
        if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
          replicate = replicated_vertices_[source];
        }
        // TODO: Strictly parent has to be decided on an edge to edge base
        // (PE can receive several blocks of the same adjacency list from different vertices)
        PEID parent_pe = parents[replicate];
        if (parent_pe == rank_) {
          continue;
        }
        auto &edges = kv.second;
        if (edges.size() <= 0) continue;

        // Send edges back to parent
        for (VertexID i = 0; i < edges.size(); i += 3) {
          send_buffers[parent_pe].emplace_back(source);
          send_buffers[parent_pe].emplace_back(replicate);
          send_buffers[parent_pe].emplace_back(edges[i]);
          send_buffers[parent_pe].emplace_back(edges[i + 1]);
          send_buffers[parent_pe].emplace_back(edges[i + 2]);
          if (parent_pe >= size_) {
            std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (up propagation) PE R" << parent_pe << std::endl;
          }
          // if (rank_ == 0 || rank_ == 2 || rank_ == 1) {
          //   std::cout << "R" << rank_ << " (up-)propagate (" << edges[i] << " (repl of " << source << ")," << edges[i + 1] << ") pe(source)=" << edges[i + 2] << " back to R" << parent_pe << " via local repl " << replicate << std::endl;
          // }
        }
        edges.clear();
      }
      propagation_buffers.clear();

      comm_timer_.Restart();
      CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
      comm_time_ += comm_timer_.Elapsed();
      CommunicationUtility::ClearBuffers(send_buffers);

      // Process received messages
      for (auto &kv : receive_buffers) {
        PEID sender_pe = kv.first;
        auto &edges = kv.second;

        // Group messages based on target vertices
        // if (rank_ == 6 || rank_ == 10) {
        //   std::cout << "R" << rank_ << " recv " << edges.size() / 4 << " edges from " << sender_pe << std::endl;
        // }
        for (VertexID i = 0; i < edges.size(); i += 5) {
          VertexID source = edges[i];
          VertexID replicate = source;
          if (replicated_vertices_.find(source) != replicated_vertices_.end()) {
            replicate = replicated_vertices_[source];
          }
          VertexID child_replicate = edges[i + 1];
          VertexID leaf_replicate = edges[i + 2];
          VertexID target = edges[i + 3];
          PEID leaf_pe = edges[i + 4];

          // Add local edge from replicate to child replicate
          if (target == replicate) {
            // Add ghost vertex if child replicate does not exist
            if (!(g.IsLocalFromGlobal(child_replicate) || g.IsGhostFromGlobal(child_replicate))) {
              g.AddGhostVertex(child_replicate, sender_pe);
            }
            g.AddEdge(g.GetLocalID(replicate), child_replicate, sender_pe);
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " add edge (" << replicate << "," << child_replicate << ") pe(target)=" << sender_pe << std::endl;
            // }
          }

          // Store mapping of edge to PE if root
          if (g.IsLocalFromGlobal(source)) {
            edge_distribution[source][target] = std::make_pair(leaf_replicate, leaf_pe);
          } else {
            // Add corresponding local edges to propagation buffer
            if (added_local.find(source) == end(added_local)) {
              for (VertexID i = 0; i < local_edges[source].size(); i+= 3) {
                VertexID replicate = local_edges[source][i];
                VertexID target = local_edges[source][i + 1];
                PEID target_pe = local_edges[source][i + 2];

                propagation_buffers[source].emplace_back(replicate);
                propagation_buffers[source].emplace_back(target);
                propagation_buffers[source].emplace_back(rank_);
                converged_locally = 0;
              }

              for (VertexID i = 0; i < parent_edges[source].size(); i+= 3) {
                VertexID replicate = parent_edges[source][i];
                VertexID target = parent_edges[source][i + 1];
                PEID target_pe = parent_edges[source][i + 2];

                propagation_buffers[source].emplace_back(replicate);
                propagation_buffers[source].emplace_back(target);
                propagation_buffers[source].emplace_back(rank_);
                converged_locally = 0;
              }
              added_local.insert(source);
            }
            propagation_buffers[source].emplace_back(leaf_replicate);
            propagation_buffers[source].emplace_back(target);
            propagation_buffers[source].emplace_back(leaf_pe);
            converged_locally = 0;
          }
        }
      }
      CommunicationUtility::ClearBuffers(receive_buffers);

      // Repeat until convergence
      comm_timer_.Restart();
      MPI_Allreduce(&converged_locally,
                    &converged_globally,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
    }
    propagation_buffers.clear();
  }

  void AddLocalEdges(DynamicGraphCommunicator &g,
                     google::dense_hash_map<VertexID, VertexBuffer> &local_edges) {
    // Add local edges
    for (auto &kv : local_edges) {
      VertexID source = kv.first;
      auto &edges = kv.second;
      for (VertexID i = 0; i < edges.size(); i+= 3) {
        VertexID replicate = edges[i];
        VertexID target = edges[i + 1];
        PEID target_pe = edges[i + 2];
        // Add ghost vertex if target does not exist
        if (!(g.IsLocalFromGlobal(target) || g.IsGhostFromGlobal(target))) {
          g.AddGhostVertex(target, target_pe);
        }
        g.AddEdge(g.GetLocalID(replicate), target, target_pe);
        // if (rank_ == 6 || rank_ == 10) {
        //   std::cout << "R" << rank_ << " add edge (" << replicate << "," << target << ") pe(target)=" << target_pe << std::endl;
        // }
      }
    }
  }

  void SendInitialRelinkMessages(DynamicGraphCommunicator &g,
                                 google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
                                 google::dense_hash_map<VertexID, VertexBuffer> &local_edges,
                                 google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                                 google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    // Send first round of relink messages
    for (auto &kv : local_edges) {
      VertexID source = kv.first;
      auto &edges = kv.second;
      for (VertexID i = 0; i < edges.size(); i+= 3) {
        VertexID replicate = edges[i];
        VertexID target = edges[i + 1];
        PEID target_pe = edges[i + 2];
        if (replicate == source) continue;

        if (target_pe != rank_) {
          send_buffers[target_pe].emplace_back(target);
          send_buffers[target_pe].emplace_back(source);
          send_buffers[target_pe].emplace_back(replicate);
          if (target_pe >= size_) {
            std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (first round) PE R" << target_pe << std::endl;
          }
          // if (rank_ == 6 || rank_ == 10) {
          //   std::cout << "R" << rank_ << " send relink (" << target << "," << source << ") -> (" << target << "," << replicate << ") pe(target)=" << rank_ << " to R" << target_pe << std::endl;
          // }
        } else {
          if (edge_distribution.find(target) == edge_distribution.end()
              || (edge_distribution.find(target) != edge_distribution.end()
                && edge_distribution[target].find(source) == edge_distribution[target].end())) {
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " local relink (" << target << "," << source << ") -> (" << target << "," << replicate << ") pe(target)=" << target_pe << std::endl;
            // }
            bool relink_success = g.RelinkEdge(g.GetLocalID(target), source, replicate, rank_);
            if (!relink_success) {
              std::cout << "R" << rank_ << " This shouldn't happen: Invalid (first round) relink (" << target << "," << source << ") -> (" << target << "," << replicate << ") from R" << rank_ << " isLocal(target)=" << g.IsLocalFromGlobal(target) << " isLocal(source)=" << g.IsLocalFromGlobal(source) << " isLocal(replicate)=" << g.IsLocalFromGlobal(replicate) << std::endl;
            }
          } else {
            send_buffers[rank_].emplace_back(target);
            send_buffers[rank_].emplace_back(source);
            send_buffers[rank_].emplace_back(replicate);
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " send relink (" << target << "," << source << ") -> (" << target << "," << replicate << ") pe(target)=" << rank_ << " to R" << rank_ << std::endl;
            // }
          }
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);
  }

  void ReplyRelinkMessages(DynamicGraphCommunicator &g,
                           google::dense_hash_map<VertexID, google::sparse_hash_map<VertexID, std::pair<VertexID, PEID>>> &edge_distribution,
                           google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                           google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    // Check relink messages for necessary answer messages
    for (auto &kv : receive_buffers) {
      PEID target_pe = kv.first;
      auto &relink_buffer = kv.second;

      for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
        VertexID source = relink_buffer[i];
        VertexID old_target = relink_buffer[i + 1];
        VertexID new_target = relink_buffer[i + 2];

        // Check if both endpoints have been replicated
        if (edge_distribution.find(source) != edge_distribution.end() 
            && (edge_distribution[source].find(old_target) != edge_distribution[source].end())) {
            // if (rank_ == 6 || rank_ == 10) {
            //   std::cout << "R" << rank_ << " answer relink reply (" << source << "," << new_target << ") from R" << target_pe << " with (" << edge_distribution[source][old_target].first << "," << new_target << ") pe(source)=" << edge_distribution[source][old_target].second << " to R" << target_pe << std::endl;
            // }
            send_buffers[target_pe].emplace_back(edge_distribution[source][old_target].first);
            send_buffers[target_pe].emplace_back(edge_distribution[source][old_target].second);
            send_buffers[target_pe].emplace_back(old_target);
            send_buffers[target_pe].emplace_back(new_target);
            if (target_pe >= size_) {
              std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (answer reply) PE R" << target_pe << std::endl;
            }
        }
        // Otherwise, perform remote relinking operation
        else {
          if (!(g.IsLocalFromGlobal(new_target) || g.IsGhostFromGlobal(new_target))) {
            g.AddGhostVertex(new_target, target_pe);
          }
          // if (rank_ == 6 || rank_ == 10) {
          //   std::cout << "R" << rank_ << " answer relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") pe(target)=" << target_pe << std::endl;
          // }
          bool relink_success = g.RelinkEdge(g.GetLocalID(source), old_target, new_target, target_pe);
          if (!relink_success) {
            std::cout << "R" << rank_ << " This shouldn't happen: Invalid (local answer) relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") from R" << target_pe << std::endl;
          }
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    // Send second round of relink (answer) messages
    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);
  }

  void SendFinalRelinkMessages(DynamicGraphCommunicator &g,
                               google::dense_hash_map<PEID, VertexBuffer> &send_buffers,
                               google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    for (auto &kv : receive_buffers) {
      PEID sender_pe = kv.first;
      auto &answer_buffer = kv.second;
      for (VertexID i = 0; i < answer_buffer.size(); i += 4) {
        VertexID source = answer_buffer[i];
        PEID source_pe = answer_buffer[i + 1];
        VertexID old_target = answer_buffer[i + 2];
        VertexID new_target = answer_buffer[i + 3];

        // Reroute relink to pe that now holds the initial edge
        send_buffers[source_pe].emplace_back(source);
        send_buffers[source_pe].emplace_back(old_target);
        send_buffers[source_pe].emplace_back(new_target);
        if (source_pe >= size_) {
          std::cout << "R" << rank_ << " This shouldn't happen: Invalid target (second round) PE R" << source_pe << std::endl;
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);
    receive_buffers[rank_] = send_buffers[rank_];
    send_buffers[rank_].clear();

    // Send final round of relink messages
    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);
  }

  void ApplyRelinks(DynamicGraphCommunicator &g,
                    google::dense_hash_map<PEID, VertexBuffer> &receive_buffers) {
    for (auto &kv : receive_buffers) {
      PEID sender_pe = kv.first;
      auto &relink_buffer = kv.second;

      for (VertexID i = 0; i < relink_buffer.size(); i += 3) {
        VertexID source = relink_buffer[i];
        VertexID old_target = relink_buffer[i + 1];
        VertexID new_target = relink_buffer[i + 2];

        if (!(g.IsLocalFromGlobal(new_target) || g.IsGhostFromGlobal(new_target))) {
          g.AddGhostVertex(new_target, sender_pe);
        }
        // if (rank_ == 6 || rank_ == 10) {
        //   std::cout << "R" << rank_ << " relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") pe(target)=" << sender_pe << std::endl;
        // }
        bool relink_success = g.RelinkEdge(g.GetLocalID(source), old_target, new_target, sender_pe);
        if (!relink_success) {
          std::cout << "R" << rank_ << " This shouldn't happen: Invalid (second round) relink (" << source << "," << old_target << ") -> (" << source << "," << new_target << ") from R" << sender_pe << std::endl;
        }
      }
    }
    CommunicationUtility::ClearBuffers(receive_buffers);
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
    for (auto &kv : replicated_vertices_) {
      g.SetActive(g.GetLocalID(kv.second), false);
    }
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
