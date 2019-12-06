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
#include <unordered_set>
#include <random>
#include <set>

#include <sys/sysinfo.h>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "dynamic_graph_comm.h"
#include "static_graph.h"
#include "cag_builder.h"
#include "dynamic_contraction.h"
#include "utils.h"
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

      // MEMORY: Delete original graph?
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

      // MEMORY: Delete intermediate graph?
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
      // TODO: Contraction does not work on static graph
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
  google::sparse_hash_map<VertexID, VertexID> distribution_map_;
  std::vector<VertexID> replicated_vertices_;

  void PerformDecomposition(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart(); 
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_timer_.Restart();
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
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
      if (!marked[v]) Utility<StaticGraph>::BFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void FindHighDegreeVertices(DynamicGraphCommunicator &g) {
    std::vector<VertexID> local_vertices;
    std::vector<VertexID> local_degrees;
    // MEMORY: Might be too small
    int num_local_vertices = 0;

    // Gather local high degree vertices
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID degree = g.GetVertexDegree(v);
      if (degree >= config_.degree_limit) {
        local_vertices.push_back(g.GetGlobalID(v));
        local_degrees.push_back(g.GetVertexDegree(v));
        num_local_vertices++;
      }
    });

    // Gather number of high degrees via all-gather
    std::vector<int> num_vertices(size_);
    MPI_Allgather(&num_local_vertices, 1, MPI_INT,
                  &num_vertices[0], 1, MPI_INT,
                  MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displ(size_);
    // MEMORY: Might be too small
    int num_global_vertices = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ[i] = num_global_vertices;
      num_global_vertices += num_vertices[i];
    }

    // Distribute vertices/degrees using all-gather
    std::vector<VertexID> global_vertices(num_global_vertices);
    std::vector<VertexID> global_degrees(num_global_vertices);
    MPI_Allgatherv(&local_vertices[0], num_local_vertices, MPI_VERTEX,
                   &global_vertices[0], &num_vertices[0], &displ[0], MPI_VERTEX,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(&local_degrees[0], num_local_vertices, MPI_VERTEX,
                   &global_degrees[0], &num_vertices[0], &displ[0], MPI_VERTEX,
                   MPI_COMM_WORLD);
  }

  void RunContraction(DynamicGraphCommunicator &g) {
    contraction_timer_.Restart();
    // VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (rank_ == ROOT) {
      if (iteration_ == 1)
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << "-" << std::endl;
                  // << " [ADD] " << global_vertices << std::endl;
      else
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << iteration_timer_.Elapsed() << std::endl;
                  // << " [ADD] " << global_vertices << std::endl;
    }
    iteration_timer_.Restart();

    // if (rank_ == ROOT) std::cout << "[STATUS] Find high degree" << std::endl;
    // FindHighDegreeVertices(g);
    
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

    if (rank_ == ROOT) std::cout << "done propagating... mem " << GetFreePhysMem() << std::endl;

    contraction_timer_.Restart();
    // Determine remaining active vertices
    g.BuildLabelShortcuts();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Building shortcuts took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    if (rank_ == ROOT) std::cout << "done shortcutting... mem " << GetFreePhysMem() << std::endl;

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

    if (rank_ == ROOT) std::cout << "done contraction... mem " << GetFreePhysMem() << std::endl;

    OutputStats<DynamicGraphCommunicator>(g);

    // Count remaining number of vertices
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
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
    VertexID vertex_offset = g.GatherNumberOfGlobalVertices() * rank_;
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
    if (rank_ == ROOT) std::cout << "avg max deg " << avg_max_deg << std::endl;

    VertexID total_parts = 0;
    VertexID num_new_vertices = 0;
    std::vector<std::vector<VertexID>> send_buffers(size_);
    std::vector<std::vector<VertexID>> receive_buffers(size_);
    if (local_max_deg >= 4 * avg_max_deg) {
      g.ForallLocalVertices([&](const VertexID v) {
        VertexID v_deg = g.GetVertexDegree(v);
        if (v_deg >= 4 * avg_max_deg) {
          // std::cout << "R" << rank_ << " v " << g.GetGlobalID(v) << " d " << v_deg << std::endl;

          // Split adjacency list for building the star
          avg_max_deg = std::max<VertexID>(avg_max_deg, 2);
          VertexID num_parts = static_cast<VertexID>(ceil(v_deg / avg_max_deg));
          std::vector<std::vector<VertexID>> adj_list_parts(num_parts);
          std::vector<std::vector<VertexID>> num_adj(num_parts);
          std::vector<VertexID> edges_to_insert;
          for (VertexID i = 0; i < num_parts; ++i) num_adj[i].resize(size_, 0);

          VertexID adj_counter = 0;
          VertexID current_part = 0;
          g.ForallNeighbors(v, [&](const VertexID w) {
              // Reset counter for new part
              if (adj_counter >= avg_max_deg) {
                current_part++;
                adj_counter = 0;
              }
              // Add v to the beginning of each part
              if (adj_counter == 0) {
                adj_list_parts[current_part].emplace_back(g.GetGlobalID(v));
                adj_list_parts[current_part].emplace_back(vertex_offset + num_new_vertices);
                num_new_vertices++;
              }
              // Add neighbors to end of part
              PEID w_pe = g.GetPE(w);
              adj_list_parts[current_part].emplace_back(g.GetGlobalID(w));
              adj_list_parts[current_part].emplace_back(w_pe);
              adj_counter++;
              num_adj[current_part][w_pe]++;
          });
          total_parts += current_part;

          for (VertexID i = 0; i < num_parts; ++i) {
            PEID max_pe = 0;
            PEID max_adj = 0;
            for (PEID j = 0; j < size_; ++j) {
              if (num_adj[i][j] > max_adj) {
                max_adj = num_adj[i][j];
                max_pe = j;
              }
            }
            send_buffers[max_pe].swap(adj_list_parts[i]);
            edges_to_insert.emplace_back(send_buffers[max_pe][1]);
            edges_to_insert.emplace_back(max_pe);
          }
          g.RemoveAllEdges(v);
          for (VertexID i = 0; i < edges_to_insert.size(); i += 2) {
            VertexID target =  edges_to_insert[i];
            VertexID target_pe =  edges_to_insert[i+1];
            g.AddGhostVertex(target, target_pe);
            // std::cout << "R" << rank_ << " add ghost vertex " << target << " pe(ghost)=" << target_pe << std::endl;
            g.AddEdge(v, target, target_pe);
            distribution_map_[target] = v;
            // std::cout << "R" << rank_ << " add edge (" << g.GetGlobalID(v) << "," << target << ") pe(target)=" << target_pe << std::endl;
          }
        }
      });
    }

    ExchangeMessages(send_buffers, receive_buffers);
    ProcessAdjLists(g, receive_buffers, send_buffers);
    ExchangeMessages(send_buffers, receive_buffers);
    ProcessRouting(g, receive_buffers);
    // TODO: Optimize detection of neighboring PEs and interface vertices
    UpdateInterfaceVertices(g);

    // TODO: Relink local edges to original vertex to replicated ones (further reduces messages)
    //       Actually do this by partitioning the vertices properly on the sender side
    // TODO: Reintroduce Semidynamic graph class
  }

  void ExchangeMessages(std::vector<std::vector<VertexID>> &send_buffers,
                        std::vector<std::vector<VertexID>> &receive_buffers) {
    PEID num_requests = 0;
    for (PEID pe = 0; pe < size_; ++pe) {
      if (send_buffers[pe].size() > 0) num_requests++; 
    }
    std::vector<MPI_Request> requests(num_requests);

    int req = 0;
    for (PEID pe = 0; pe < size_; ++pe) {
      if (send_buffers[pe].size() > 0) {
        MPI_Issend(send_buffers[pe].data(), 
                   static_cast<int>(send_buffers[pe].size()), 
                   MPI_VERTEX, pe, 993 * size_ + pe, MPI_COMM_WORLD, &requests[req++]);
        if (pe == rank_) {
          std::cout << "R" << rank_ << " ERROR selfmessage" << std::endl;
          exit(1);
        }
      } 
    }

    std::vector<MPI_Status> statuses(num_requests);
    int isend_done = 0;
    while (isend_done == 0) {
      // Check for messages
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, 993 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          std::vector<VertexID> message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
      // Check if all ISend successful
      isend_done = 0;
      MPI_Testall(num_requests, requests.data(), &isend_done, statuses.data());
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

    int ibarrier_done = 0;
    while (ibarrier_done == 0) {
      int iprobe_success = 1;
      while (iprobe_success > 0) {
        iprobe_success = 0;
        MPI_Status st{};
        MPI_Iprobe(MPI_ANY_SOURCE, 993 * size_ + rank_, MPI_COMM_WORLD, &iprobe_success, &st);
        if (iprobe_success > 0) {
          int message_length;
          MPI_Get_count(&st, MPI_VERTEX, &message_length);
          std::vector<VertexID> message(message_length);
          MPI_Status rst{};
          MPI_Recv(message.data(), message_length, MPI_VERTEX, st.MPI_SOURCE,
                   st.MPI_TAG, MPI_COMM_WORLD, &rst);

          for (const VertexID &m : message) {
            receive_buffers[st.MPI_SOURCE].emplace_back(m);
          }
        }
      }
        
      // Check if all reached Ibarrier
      MPI_Status tst{};
      MPI_Test(&barrier_request, &ibarrier_done, &tst);
      if (tst.MPI_ERROR != MPI_SUCCESS) {
        std::cout << "R" << rank_ << " mpi_test (barrier) failed" << std::endl;
        exit(1);
      }
    }
  }

  void ProcessAdjLists(DynamicGraphCommunicator &g, 
                       std::vector<std::vector<VertexID>> & receive_buffer,
                       std::vector<std::vector<VertexID>> & send_buffer) {
    // Clear send buffers
    for (PEID pe = 0; pe < size_; ++pe) {
      send_buffer[pe].clear();
    }
    // Process incoming vertices/edges
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffer[pe].size() > 0) {
        // std::cout << "R" << rank_ << " recv " << (receive_buffer[pe].size()-1)/2<< " edges from " << pe << std::endl;
        VertexID source = receive_buffer[pe][0];
        VertexID copy_vertex = receive_buffer[pe][1];
        // std::cout << "R" << rank_ << " recv vertex " << source << " new_id " << copy_vertex << " from " << pe << std::endl;
        VertexID local_copy_id = g.AddVertex(copy_vertex);
        replicated_vertices_.emplace_back(local_copy_id);
        // std::cout << "R" << rank_ << " add local vertex " << copy_vertex << std::endl;
        g.AddEdge(local_copy_id, source, pe);
        // std::cout << "R" << rank_ << " add edge (" << copy_vertex << "," << source << ") pe(target)=" << pe << std::endl;
        for (VertexID i = 2; i < receive_buffer[pe].size(); i+=2) {
          VertexID target = receive_buffer[pe][i];
          VertexID target_pe = receive_buffer[pe][i+1];
          // std::cout << "R" << rank_ << " recv edge (" << copy_vertex << "," << target << ") local(target)=" << (target_pe == rank_) << " pe(target)=" << target_pe << " from " << pe << std::endl;
          // Neighbor is local vertex
          if (target_pe == rank_) {
            // g.AddEdge(g.GetLocalID(target), copy_vertex, rank_);
            // std::cout << "R" << rank_ << " add edge (" << target << "," << copy_vertex << ") pe(target)=" << rank_ << std::endl;
            g.RelinkEdge(g.GetLocalID(target), source, copy_vertex, rank_);
            // std::cout << "R" << rank_ << " reroute edge (" << target << "," << source << ") pe(target)=" << g.GetPE(source) << " -> (" << target << "," << copy_vertex << ") pe(target)=" << rank_ << std::endl;
          // Neighbor is remote vertex
          } else {
            g.AddGhostVertex(target);
            // Target PE needs to reroute (target, source) from (original) PE to (target, copy) on sender PE
            send_buffer[target_pe].emplace_back(target);
            send_buffer[target_pe].emplace_back(source);
            send_buffer[target_pe].emplace_back(copy_vertex);
            // std::cout << "R" << rank_ << " send reroute (" << target << "," << source << ") pe(target)=" << pe << " -> (" << target << "," << copy_vertex << ") pe(target)=" << rank_ << " to " << target_pe << std::endl;
          }
          g.AddEdge(g.GetLocalID(copy_vertex), target, target_pe);
          // std::cout << "R" << rank_ << " add edge (" << copy_vertex << "," << target << ") pe(target)=" << target_pe << std::endl;
        }
        // Clear receive buffers
        receive_buffer[pe].clear();
      }
    }
  }

  void ProcessRouting(DynamicGraphCommunicator &g, 
                      std::vector<std::vector<VertexID>> & receive_buffer) {
    // Process incoming vertices/edges
    for (PEID pe = 0; pe < size_; ++pe) {
      if (receive_buffer[pe].size() > 0) {
        // std::cout << "R" << rank_ << " recv " << receive_buffer[pe].size()/3 << " reroutes from " << pe << std::endl;
        for (VertexID i = 0; i < receive_buffer[pe].size(); i+=3) {
          VertexID source = receive_buffer[pe][i];
          VertexID old_target = receive_buffer[pe][i+1];
          VertexID new_target = receive_buffer[pe][i+2];
          // std::cout << "R" << rank_ << " recv reroute (" << source << "," << old_target << ") pe(target)=" << g.GetPE(old_target) << " -> (" << source << "," << new_target << ") pe(target)=" << pe << " from " << pe << std::endl;

          if (!g.IsGhostFromGlobal(new_target)) {
            // std::cout << "R" << rank_ << " add ghost vertex " << new_target << " pe(vertex)=" << pe << std::endl;
            g.AddGhostVertex(new_target, pe);
          }
          g.RelinkEdge(g.GetLocalID(source), old_target, new_target, pe);
          // std::cout << "R" << rank_ << " reroute edge (" << source << "," << old_target << ") pe(target)=" << g.GetPE(old_target) << " -> (" << source << "," << new_target << ") pe(target)=" << pe << std::endl;
        }
      }
    }
  }

  void UpdateInterfaceVertices(DynamicGraphCommunicator &g) {
    // Check if PEs are still connected
    std::vector<bool> is_neighbor(size_, false);
    g.ForallLocalVertices([&](const VertexID v) {
      bool ghost_neighbor = false;
      g.ForallNeighbors(v, [&](const VertexID w) {
        if (g.IsGhost(w)) {
          is_neighbor[g.GetPE(v)] = true;
          ghost_neighbor = true;
        }
      });
      g.SetInterface(v, ghost_neighbor);
    });

    // Update PEs
    for (PEID i = 0; i < size_; i++) {
      g.SetAdjacentPE(i, is_neighbor[i]);
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

  static long long GetFreePhysMem() {
    struct sysinfo memInfo;
    sysinfo (&memInfo);
    long long totalPhysMem = memInfo.totalram;
    long long freePhysMem = memInfo.freeram;

    totalPhysMem *= memInfo.mem_unit;
    freePhysMem *= memInfo.mem_unit;
    totalPhysMem *= 1e-9;
    freePhysMem *= 1e-9;

    return freePhysMem;
  } 
};

#endif
