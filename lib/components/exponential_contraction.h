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
        comm_time_(0.0) { }

  virtual ~ExponentialContraction() {
    delete exp_contraction_;
    exp_contraction_ = nullptr;
  };

  void FindComponents(StaticGraph &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    contraction_timer_.Restart();
    if (config_.use_contraction) {
      FindLocalComponents(g, g_labels);
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |- Finding local components on input took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }
      
      // First round of contraction
      contraction_timer_.Restart();
      CAGBuilder<StaticGraph> 
        first_contraction(g, g_labels, rank_, size_);
      StaticGraph cag 
        = first_contraction.BuildStaticComponentAdjacencyGraph();
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
      DynamicGraphCommunicator ccag 
        = second_contraction.BuildDynamicComponentAdjacencyGraph();
      OutputStats<DynamicGraphCommunicator>(ccag);
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |- Building second cag took " 
                  << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

      if (config_.replicate_high_degree) {
        DistributeHighDegreeVertices(ccag);
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
      // exp_contraction_ = new DynamicContraction(g, rank_, size_);

      // PerformDecomposition(g);

      // g.ForallLocalVertices([&](const VertexID v) {
      //     g_labels[v] = g.GetVertexLabel(v);
      // });
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
  std::vector<VertexID> replicated_vertices_;

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
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |--- Barrier before convergence test took " 
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

    // if (iteration_ == 2) {
    //   g.OutputLocal();
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   exit(1);
    // }

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

  void DistributeHighDegreeVertices(DynamicGraphCommunicator &g) {
    // Offset for new vertex IDs
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    VertexID vertex_offset = global_vertices * (rank_ + 1);
    VertexID local_max_deg = 0;
    g.ForallLocalVertices([&](const VertexID v) {
        VertexID v_deg = g.GetVertexDegree(v);
        if (v_deg > local_max_deg) {
          local_max_deg = v_deg; 
        }
    });
    VertexID global_max_deg = 0;
    MPI_Allreduce(&local_max_deg, &global_max_deg, 
                  1, MPI_VERTEX, 
                  MPI_SUM, MPI_COMM_WORLD);
    VertexID avg_max_deg = global_max_deg / size_; 

    VertexID num_new_vertices = 0;
    google::dense_hash_map<PEID, VertexBuffer> send_buffers;
    send_buffers.set_empty_key(-1);
    google::dense_hash_map<PEID, VertexBuffer> receive_buffers;
    receive_buffers.set_empty_key(-1);

    if (local_max_deg >= 4 * avg_max_deg) {
      g.ForallLocalVertices([&](const VertexID v) {
        VertexID v_deg = g.GetVertexDegree(v);
        if (v_deg >= 4 * avg_max_deg) {
          // Split adjacency list for building the star
          ComputeEdgePartitioning(g, avg_max_deg, v, v_deg, vertex_offset, num_new_vertices, send_buffers);
        }
      });
    }

    comm_timer_.Restart();
    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_time_ += comm_timer_.Elapsed();
    CommunicationUtility::ClearBuffers(send_buffers);
    ProcessAdjLists(g, global_vertices, receive_buffers, send_buffers);
    CommunicationUtility::ClearBuffers(receive_buffers);

    CommunicationUtility::SparseAllToAll(send_buffers, receive_buffers, rank_, size_, 993);
    comm_timer_.Restart();
    CommunicationUtility::ClearBuffers(send_buffers);
    comm_time_ += comm_timer_.Elapsed();
    ProcessRelinking(g, receive_buffers);
    CommunicationUtility::ClearBuffers(receive_buffers);

    UpdateInterfaceVertices(g);
  }

  void ComputeEdgePartitioning(DynamicGraphCommunicator &g,
                               const VertexID &avg_max_deg,
                               const VertexID &vertex_id,
                               const VertexID &vertex_deg,
                               const VertexID &new_vertices_offset,
                               VertexID &new_vertices_counter,
                               google::dense_hash_map<PEID, VertexBuffer> &send_buffers) {
    // Ensure that its not just a single partition
    VertexID clamped_avg_max_deg = std::max<VertexID>(avg_max_deg, 2);

    // Number of partitions to generate
    VertexID num_parts = static_cast<VertexID>(ceil(vertex_deg / clamped_avg_max_deg));
    VertexID part_size = vertex_deg / num_parts;

    // Temporary vector for storing new local edges 
    std::vector<VertexID> local_edges;

    google::dense_hash_map<PEID, std::vector<VertexID>> edges_for_pe;
    edges_for_pe.set_empty_key(-1);
    g.ForallNeighbors(vertex_id, [&](const VertexID w) {
        edges_for_pe[g.GetPE(w)].emplace_back(g.GetGlobalID(w));
    });
    
    // Copy hash map to vector
    std::vector<std::pair<PEID, VertexID>> num_edges_for_pe;
    for (const auto &kv : edges_for_pe) {
      PEID pe = kv.first;
      VertexID num_edges = kv.second.size();
      num_edges_for_pe.emplace_back(pe, num_edges);
    }

    PEID num_neighboring_pes = edges_for_pe.size();
    std::random_shuffle(num_edges_for_pe.begin(), num_edges_for_pe.end());
    std::sort(num_edges_for_pe.begin(), num_edges_for_pe.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    VertexID remaining_edges = vertex_deg;
    VertexID remaining_parts = num_parts;
    // Check if we can send single large part to PE
    for (const auto &kv : num_edges_for_pe) {
      PEID pe = kv.first;
      VertexID num_edges = kv.second;
      if (num_edges >= part_size) {
        send_buffers[pe].emplace_back(g.GetGlobalID(vertex_id));
        send_buffers[pe].emplace_back(new_vertices_offset + new_vertices_counter);
        local_edges.emplace_back(new_vertices_offset + new_vertices_counter);
        local_edges.emplace_back(pe);
        new_vertices_counter++;
        for (const auto &v : edges_for_pe[pe]) {
          send_buffers[pe].emplace_back(std::move(v));
          send_buffers[pe].emplace_back(pe);
        }
        remaining_parts -= floor(num_edges / part_size);
        edges_for_pe[pe].clear();
      }
    }

    // Distribute the remaining parts greedily between neighboring PEs
    while (remaining_parts > 0) {
      // Greedily select PE with most remaining edges
      PEID current_target_pe = std::numeric_limits<PEID>::max();
      int remaining_part_size = part_size;
      for (const auto &kv : num_edges_for_pe) {
        PEID pe = kv.first;
        PEID num_edges = kv.second;
        if (num_edges > 0 && remaining_part_size > 0) {
          // Set current PE as target for following messages
          if (current_target_pe >= size_) {
            current_target_pe = pe;
            send_buffers[current_target_pe].emplace_back(g.GetGlobalID(vertex_id));
            send_buffers[current_target_pe].emplace_back(new_vertices_offset + new_vertices_counter);
            local_edges.emplace_back(new_vertices_offset + new_vertices_counter);
            local_edges.emplace_back(current_target_pe);
            new_vertices_counter++;
          }
          // Add edges in current block
          for (const auto &v : edges_for_pe[pe]) {
            send_buffers[current_target_pe].emplace_back(std::move(v));
            send_buffers[current_target_pe].emplace_back(pe);
          }
          remaining_part_size -= num_edges; 
          if (remaining_part_size <= 0) {
            PEID current_target_pe = std::numeric_limits<PEID>::max();
            // First step guarantees that edgelists are now smaller than a part
            remaining_parts--;
          }
          edges_for_pe[pe].clear();
        }
      }
    }
    edges_for_pe.clear();

    // Update the adjacency list of the original (replicated) vertex
    g.RemoveAllEdges(vertex_id);
    for (VertexID i = 0; i < local_edges.size(); i += 2) {
      VertexID target =  local_edges[i];
      VertexID target_pe =  local_edges[i+1];
      g.AddGhostVertex(target, target_pe);
      g.AddEdge(vertex_id, target, target_pe);
    }
  }

  void ProcessAdjLists(DynamicGraphCommunicator &g, 
                       const VertexID &offset,
                       google::dense_hash_map<PEID, VertexBuffer> &receive_buffer,
                       google::dense_hash_map<PEID, VertexBuffer> &send_buffer) {
    // Clear send buffers
    for (const auto &kv : send_buffer) {
      PEID pe = kv.first;
      send_buffer[pe].clear();
    }
    send_buffer.clear();
    // Process incoming vertices/edges
    for (const auto &kv : receive_buffer) {
      PEID pe = kv.first;
      auto& buffer = kv.second;
      if (buffer.size() > 0) {
        VertexID source, copy_vertex;
        for (VertexID i = 0; i < buffer.size(); i+=2) {
          // New source
          // Note: This might not work if number of global vertices is smaller than number of PEs
          if (buffer[i+1] >= offset) {
            source = buffer[i];
            copy_vertex = buffer[i+1];
            VertexID local_copy_id = g.AddVertex(copy_vertex);
            replicated_vertices_.emplace_back(local_copy_id);
            g.AddEdge(local_copy_id, source, pe);
          } else {
            VertexID target = buffer[i];
            VertexID target_pe = buffer[i+1];
            // Neighbor is local vertex
            if (target_pe == rank_) {
              g.RelinkEdge(g.GetLocalID(target), source, copy_vertex, rank_);
            // Neighbor is remote vertex
            } else {
              g.AddGhostVertex(target, target_pe);
              // Target PE needs to reroute (target, source) from (original) PE to (target, copy) on sender PE
              send_buffer[target_pe].emplace_back(target);
              send_buffer[target_pe].emplace_back(source);
              send_buffer[target_pe].emplace_back(copy_vertex);
            }
            g.AddEdge(g.GetLocalID(copy_vertex), target, target_pe);
          }
        }
      }
    }
  }

  void ProcessRelinking(DynamicGraphCommunicator &g, 
                      google::dense_hash_map<PEID, VertexBuffer> &receive_buffer) {
    // Process incoming vertices/edges
    for (const auto &kv : receive_buffer) {
      PEID pe = kv.first;
      auto& buffer = kv.second;
      if (buffer.size() > 0) {
        for (VertexID i = 0; i < buffer.size(); i+=3) {
          VertexID source = buffer[i];
          VertexID old_target = buffer[i+1];
          VertexID new_target = buffer[i+2];

          if (!g.IsGhostFromGlobal(new_target)) {
            g.AddGhostVertex(new_target, pe);
          }
          g.RelinkEdge(g.GetLocalID(source), old_target, new_target, pe);
        }
      }
    }
  }

  void UpdateInterfaceVertices(DynamicGraphCommunicator &g) {
    // Check if PEs are still connected
    google::dense_hash_set<PEID> neighboring_pes;
    neighboring_pes.set_empty_key(-1);
    g.ForallLocalVertices([&](const VertexID v) {
      bool ghost_neighbor = false;
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (g.IsGhost(w)) {
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
    for (const VertexID &v : replicated_vertices_) {
      g.SetActive(v, false);
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
