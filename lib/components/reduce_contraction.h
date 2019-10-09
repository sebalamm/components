/******************************************************************************
 * components.h *
 * Distributed computation of connected components
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

#ifndef _REDUCE_CONTRACTION_H_
#define _REDUCE_CONTRACTION_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include <sys/sysinfo.h>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "dynamic_graph_access.h"
#include "static_graph_access.h"
#include "minimal_graph_access.h"
#include "cag_builder.h"
#include "dynamic_contraction.h"
#include "utils.h"
#include "union_find.h"
#include "propagation.h"

template <typename GraphType>
class ReduceContraction {
 public:
  ReduceContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0),
        is_active_(true),
        num_vertices_per_pe_(size, 0) { 
    vertex_map_.set_empty_key(-1);
    reverse_vertex_map_.set_empty_key(-1);
    vertex_to_contracted_vertex_.set_empty_key(-1);
    contracted_vertex_to_vertices_.set_empty_key(-1);
    inactive_vertices_.set_empty_key(-1);
  }

  virtual ~ReduceContraction() { };

  void FindComponents(GraphType &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    contraction_timer_.Restart();

    // Turn into minimal graph type
    MinimalGraphAccess mg(rank_, size_);
    ConvertGraph(g, mg);
    // g.OutputLocal();

    // Perform successive reductions
    google::sparse_hash_map<VertexID, VertexID> labels;
    mg.ForallLocalVertices([&](VertexID v) {
      labels[v] = v;
    });

    // Contraction
    vertex_buffer_.resize(ceil(log2(size_)));
    label_buffer_.resize(ceil(log2(size_)));
    for (PEID i = 0; i < ceil(log2(size_)); i++) {
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Start exchange (contraction)" << std::endl;
      }
      if (is_active_) {
        if (IsBitSet(rank_, i)) {
          // std::cout << "R" << rank_ << " send graph to " << rank_ - pow(2,i) << " (round " << i << ")" << std::endl;
          SendGraph(mg, labels, rank_ - pow(2,i), i);
          SendGhosts(mg, labels, rank_ - pow(2,i), i);
          is_active_ = false;
        } else if (rank_ + pow(2,i) < size_) {
          ReceiveGraph(mg, labels, rank_ + pow(2,i), i);
          ReceiveGhosts(mg, labels, rank_ + pow(2,i), i);
        }
      }

      if (is_active_) {
        BuildLocalGraph(mg, labels, rank_ + pow(2,i), i);
        // Compute new labels and contract local ones
        FindLocalComponents(mg, labels);
        ContractLocalComponents(mg, labels);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Exchange round (contraction) took " 
          << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }

    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_ == ROOT) {
    }
    // Uncontraction
    for (PEID i = ceil(log2(size_)); i-- > 0;) {
      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Start exchange (uncontraction)" << std::endl;
      }

      if (!is_active_ && (rank_ % (PEID)(pow(2,i))) == 0) is_active_ = true;
      if (is_active_) {
        if (IsBitSet(rank_, i)) {
          ReceiveLabels(mg, labels, rank_ - pow(2,i), i);
        } else if (rank_ + pow(2,i) < size_) {
          // TODO: Propagate labels locally
          SendLabels(mg, labels, rank_ + pow(2,i), i);
        }
      }

      if (rank_ == ROOT) {
        std::cout << "[STATUS] |-- Exchange round (uncontraction) took " 
          << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
      }
    }

    g.ForallLocalVertices([&](VertexID v) {
      VertexID gv = g.GetGlobalID(v);
      VertexID cv = g.GetGlobalID(v);
      if (vertex_to_contracted_vertex_.find(cv) != end(vertex_to_contracted_vertex_)) {
        cv = vertex_to_contracted_vertex_[cv];
      } 
      labels[gv] = labels[cv];
    });

    g.ForallLocalVertices([&](VertexID v) {
      VertexID gv = g.GetGlobalID(v);
      g_labels[v] = labels[gv];
      std::cout << "R" << rank_ << " v " << gv << " l " << labels[gv] << std::endl;
    });
  }

  void Output(GraphType &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  const PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;
  bool is_active_;
  VertexID offset_;
  VertexID rng_offset_;

  // Vertex distribution
  std::vector<int> num_vertices_per_pe_;
  
  // Local information
  std::vector<std::vector<VertexID>> vertex_buffer_;
  std::vector<std::vector<VertexID>> label_buffer_;
  std::vector<std::pair<VertexID, VertexID>> ghost_buffer_;
  std::vector<std::pair<VertexID, VertexID>> edge_buffer_;

  // Global information
  std::vector<VertexID> global_vertices_;
  std::vector<VertexID> global_labels_;
  std::vector<std::pair<VertexID, VertexID>> global_edges_;

  // Vertex maps
  google::dense_hash_map<VertexID, int> vertex_map_; 
  google::dense_hash_map<int, VertexID> reverse_vertex_map_; 

  // Contracted vertices
  google::dense_hash_map<VertexID, VertexID> vertex_to_contracted_vertex_;
  google::dense_hash_map<VertexID, std::vector<VertexID>> contracted_vertex_to_vertices_;
  google::dense_hash_set<VertexID> inactive_vertices_;

  // Statistics
  Timer contraction_timer_;

  void ConvertGraph(GraphType &g, MinimalGraphAccess &mg) {
    // Start construct
    offset_ = g.GatherNumberOfGlobalVertices();

    // Add local vertices
    g.ForallLocalVertices([&](const VertexID v) {
        mg.AddLocalVertex(g.GetGlobalID(v)) ;
    });

    // Add ghost vertices
    g.ForallGhostVertices([&](const VertexID v) {
        mg.AddGhostVertex(g.GetGlobalID(v), g.GetPE(v));
        if (g.GetPE(v) == rank_) {
          std::cout << "R" << rank_ << " This shouldn't happen! (init)" << std::endl;
          exit(1);
        }
    });

    // Add edges
    g.ForallLocalVertices([&](const VertexID v) {
      g.ForallNeighbors(v, [&](const VertexID w) {
        mg.AddEdge(g.GetGlobalID(v), g.GetGlobalID(w));
      });
    });
  }

  inline bool IsBitSet(PEID val, int bit) {
    return 1 == ((val >> bit) & 1);
  }

  void ReceiveGhosts(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &g_labels, PEID target, VertexID level) {
    std::vector<VertexID> ghost_vertices;
    google::dense_hash_set<VertexID> unique_ghosts;
    unique_ghosts.set_empty_key(-1);

    g.ForallLocalVertices([&](VertexID v) {
      g.ForallNeighbors(v, [&](VertexID w) {
        // TODO: Does this have to include ghost vertices from the previous step?
        if (g.IsGhost(w) && g.GetPE(w) == target) {
          // Only receive updates for ghosts located at the specified neighbor
          if (unique_ghosts.find(w) == end(unique_ghosts)) {
            unique_ghosts.insert(w);
            ghost_vertices.emplace_back(w);
            // std::cout << "R" << rank_ << " request update for " << w << std::endl;
          }
        }
      });
    });

    MPI_Send(&ghost_vertices[0], static_cast<int>(ghost_vertices.size()),
             MPI_VERTEX, target, 
             target + 6 * size_, MPI_COMM_WORLD);

    // Wait for other PE to update the ghost labels
    std::vector<VertexID> ghost_updates;
    MPI_Status st{};
    int message_length;

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    ghost_updates.resize(message_length);
    MPI_Recv(&ghost_updates[0], ghost_updates.size(),
             MPI_VERTEX, target,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    google::dense_hash_map<VertexID, VertexID> ghost_map;
    ghost_map.set_empty_key(-1);
    for (VertexID i = 0; i < ghost_updates.size(); i += 2) {
      // std::cout << "R" << rank_ << " received update (" << ghost_updates[i] << "," << ghost_updates[i + 1] << ") (round " << level << ")" << std::endl;
      ghost_map[ghost_updates[i]] = ghost_updates[i + 1];
    }

    // Relink edges
    g.ForallLocalVertices([&](VertexID v) {
      g.ForallNeighbors(v, [&](VertexID w) {
        if (ghost_map.find(w) != end(ghost_map)) {
          g.RelinkEdge(v, w, ghost_map[w]);
          if (rank_ == 0) {
            std::cout << "R" << rank_ << " relink edge (ghost) (" << v << "," << w << ") to (" << v << "," << ghost_map[w] << ")" << std::endl;
          }
        }
      });
    });
  }

  void SendGhosts(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &g_labels, PEID sender, VertexID level) {
    std::vector<VertexID> ghosts;
    MPI_Status st{};
    int message_length;

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    ghosts.resize(message_length);
    MPI_Recv(&ghosts[0], ghosts.size(),
             MPI_VERTEX, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    // Process we need to send new ghost labels 
    std::vector<VertexID> ghost_updates;
    for (VertexID i = 0; i < ghosts.size(); ++i) {
      VertexID v = ghosts[i];
      VertexID current_label = v;
      if (vertex_to_contracted_vertex_.find(v) != end(vertex_to_contracted_vertex_)) {
        current_label = vertex_to_contracted_vertex_[v];
        while (vertex_to_contracted_vertex_.find(current_label) != end(vertex_to_contracted_vertex_)) {
          current_label = vertex_to_contracted_vertex_[current_label];
        }
      } 
      ghost_updates.emplace_back(v);
      ghost_updates.emplace_back(current_label);
      // std::cout << "R" << rank_ << " send update (" << v << "," << current_label << ")" << std::endl;
    }

    MPI_Send(&ghost_updates[0], static_cast<int>(ghost_updates.size()),
             MPI_VERTEX, sender, 
             sender + 6 * size_, MPI_COMM_WORLD);
  }

  void SendGraph(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &g_labels, PEID target, VertexID level) {
    vertex_buffer_[level].clear();
    label_buffer_[level].clear();
    ghost_buffer_.clear();
    edge_buffer_.clear();

    google::sparse_hash_set<VertexID> ghost_set;
    auto pair = [&](VertexID v, PEID rank) {
      return v * size_ + rank;
    };

    // Gather vertices based on 
    g.ForallLocalVertices([&](const VertexID v) {
    // for (auto &kv : g_labels) {
      // VertexID v = kv.first;
      // VertexID v_label = kv.second;
      VertexID v_label = g_labels[v];
      // This ensures that inactive vertices are not send
      if (v == v_label) {
        vertex_buffer_[level].push_back(v);
        label_buffer_[level].push_back(v_label);
        g.ForallNeighbors(v, [&](const VertexID &w) {
          if (g.IsGhost(w)) {
            PEID updated_rank = g.GetPE(w);
            // Directly update the ranks of ghost vertices
            for (VertexID i = 0; i <= level; i++) {
              updated_rank = IsBitSet(updated_rank, i) ? updated_rank - pow(2,i) : updated_rank;
            }
            g.SetPE(w, updated_rank);
            if (updated_rank != target && ghost_set.find(pair(w, updated_rank)) == end(ghost_set)) {
              // std::cout << "R" << rank_ << " send ghost (" << w << "," << updated_rank << ") to " << target << " (round " << level << ")"<< std::endl;
              ghost_buffer_.emplace_back(w, updated_rank);
              ghost_set.insert(pair(w, updated_rank));
            }
          }
          if (rank_ == 8) std::cout << "R" << rank_ << " send edge (" << v << "," << w << ") to " << target << " (round " << level << ")"<< std::endl;
          edge_buffer_.emplace_back(v, w);
        });
      } 
    // }
    });

    // Send graph data
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    MPI_Send(&vertex_buffer_[level][0], static_cast<int>(vertex_buffer_[level].size()),
             MPI_VERTEX, target, 
             target + 6 * size_, MPI_COMM_WORLD);
    MPI_Send(&label_buffer_[level][0], static_cast<int>(label_buffer_[level].size()),
             MPI_VERTEX, target, 
             target + 6 * size_, MPI_COMM_WORLD);
    MPI_Send(&ghost_buffer_[0], static_cast<int>(ghost_buffer_.size()),
             MPI_COMP, target, 
             target + 6 * size_, MPI_COMM_WORLD);
    MPI_Send(&edge_buffer_[0], static_cast<int>(edge_buffer_.size()),
             MPI_COMP, target, 
             target + 6 * size_, MPI_COMM_WORLD);
  }

  void ReceiveGraph(MinimalGraphAccess& g, google::sparse_hash_map<VertexID, VertexID> &g_labels, PEID sender, VertexID level) {
    vertex_buffer_[level].clear();
    label_buffer_[level].clear();
    ghost_buffer_.clear();
    edge_buffer_.clear();

    // Receive graph data
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);
    MPI_Status st{};
    int message_length;

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    vertex_buffer_[level].resize(message_length);
    MPI_Recv(&vertex_buffer_[level][0], vertex_buffer_[level].size(),
             MPI_VERTEX, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    label_buffer_[level].resize(message_length);
    MPI_Recv(&label_buffer_[level][0], label_buffer_[level].size(),
             MPI_VERTEX, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_COMP, &message_length);
    ghost_buffer_.resize(message_length);
    MPI_Recv(&ghost_buffer_[0], ghost_buffer_.size(),
             MPI_COMP, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_COMP, &message_length);
    edge_buffer_.resize(message_length);
    MPI_Recv(&edge_buffer_[0], edge_buffer_.size(),
             MPI_COMP, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);
  }

  void BuildLocalGraph(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &label, PEID sender, VertexID level) {
    // Add local vertex
    for (VertexID i = 0; i < vertex_buffer_[level].size(); i++) {
      VertexID v = vertex_buffer_[level][i];
      // std::cout << "R" << rank_ << " build local graph add local " << v << std::endl;
      g.AddLocalVertex(v);   
      // VertexID v_label = label_buffer_[level][i];
      // if (v == v_label) {
      //   g.AddLocalVertex(v);   
      // } else {
      //   vertex_to_contracted_vertex_[v] = v_label;
      //   std::cout << "R" << rank_ << "This shouldn't happen (build local)" << std::endl;
      // }
      if (g.IsGhost(v)) {
        std::cout << "R" << rank_ << " This shouldn't happen! (build add local)" << std::endl;
        exit(1);
      }
    }

    for (VertexID i = 0; i < ghost_buffer_.size(); i++) {
      VertexID v = ghost_buffer_[i].first;
      VertexID rank = ghost_buffer_[i].second;
      // std::cout << "R" << rank_ << " build local graph add ghost " << v << std::endl;
      g.AddGhostVertex(v, rank); 
      if (g.IsLocal(v) || rank == rank_) {
        std::cout << "R" << rank_ << " This shouldn't happen! (build add ghost) v " << v << " l " << g.IsLocal(v) << " r " << rank << std::endl;
        exit(1);
      }
    }

    // Add edges
    for (auto &e : edge_buffer_) {
      if (rank_ == 0) {
        std::cout << "R" << rank_ << " insert edge (" << e.first << "," << e.second << ")" << std::endl;
      }
      g.AddEdge(e.first, e.second);
    }

    // Updated edges
    g.ForallLocalVertices([&](VertexID v){
      g.ForallNeighbors(v, [&](VertexID w) {
        // If neighbor was previously local it was contracted and needs to be relinked
        // if ((!g.IsLocal(w)) && vertex_to_contracted_vertex_.find(w) != end(vertex_to_contracted_vertex_)) {
        if (vertex_to_contracted_vertex_.find(w) != end(vertex_to_contracted_vertex_)) {
          VertexID current_label = vertex_to_contracted_vertex_[w];
          while (vertex_to_contracted_vertex_.find(current_label) != end(vertex_to_contracted_vertex_)) {
            current_label = vertex_to_contracted_vertex_[current_label];
          }
          g.RelinkEdge(v, w, current_label);
          if (rank_ == 0) {
            std::cout << "R" << rank_ << " relink edge (" << v << "," << w << ") to (" << v << "," << current_label << ")" << std::endl;
          }
        }
      });
    });
    g.OutputLocal();
  }

  void FindLocalComponents(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &label) {
    google::sparse_hash_set<VertexID> marked;
    google::sparse_hash_map<VertexID, VertexID> parent;

    // Compute mapping for ghost vertices
    g.ForallVertices([&](const VertexID v) {
      label[v] = v;
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (marked.find(v) == end(marked)) ModifiedBFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void ModifiedBFS(MinimalGraphAccess &g,
                   const VertexID &start,
                   google::sparse_hash_set<VertexID> &marked,
                   google::sparse_hash_map<VertexID, VertexID> &parent) {
    // Standard BFS
    std::queue<VertexID> q;
    q.push(start);
    marked.insert(start);
    parent[start] = start;
    while (!q.empty()) {
      VertexID v = q.front();
      q.pop();
      g.ForallNeighbors(v, [&](VertexID &w) {
        if (g.IsLocal(w) && marked.find(w) == end(marked)) {
          q.push(w);
          marked.insert(w);
          parent[w] = start;
        }
      });
    }
  }

  void ContractLocalComponents(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &label) { 
    // Gather number of unique labels
    google::dense_hash_set<VertexID> contracted_vertices;
    contracted_vertices.set_empty_key(-1);
    EdgeHash contracted_edges;

    std::vector<std::pair<VertexID, VertexID>> new_ghosts;
    google::sparse_hash_set<VertexID> ghost_set;
    auto pair = [&](VertexID v, PEID rank) {
      return v * size_ + rank;
    };

    g.ForallLocalVertices([&](const VertexID v) {
      VertexID v_label = label[v];
      if (contracted_vertices.find(v_label) == end(contracted_vertices)) {
        contracted_vertices.insert(v_label);
      } 
      if (v_label != v) {
        vertex_to_contracted_vertex_[v] = v_label;
        contracted_vertex_to_vertices_[v_label].emplace_back(v);
        inactive_vertices_.insert(v);
        if (v_label == 18) std::cout << "R" << rank_ << " contract vertex " << v << " to " << v_label << std::endl;
        // if (v == 34) std::cout << "R" << rank_ << " contract vertex " << v << " to " << v_label << std::endl;
      }
      g.ForallNeighbors(v, [&](const VertexID w) {
        bool is_ghost = g.IsGhost(w);
        if (is_ghost) {
          PEID pe = g.GetPE(w);
          PEID id = pair(w, pe);
          if (ghost_set.find(id) == end(ghost_set)) {
            if (is_ghost && pe == rank_) {
              std::cout << "R" << rank_ << " This shouldn't happen! (contract w " << w << " g " << is_ghost << " p " << pe << ")" << std::endl;
              exit(1);
            }
            ghost_set.insert(id);
            new_ghosts.emplace_back(w, pe);
          }

          auto h_edge = HashedEdge{offset_, v_label, w, pe};
          if (is_ghost && contracted_edges.find(h_edge) == end(contracted_edges)) {
            contracted_edges.insert(h_edge);
          } 
        }
      });
    });

    // Add local vertices
    // label.clear();
    MinimalGraphAccess cg(rank_, size_);
    for (const VertexID v : contracted_vertices) {
      cg.AddLocalVertex(v);   
      label[v] = v;
    }

    // Add ghost vertices
    // std::cout << "R" << rank_ << " contract local components" << std::endl;
    for (VertexID i = 0; i < new_ghosts.size(); i++) {
      VertexID v = new_ghosts[i].first;
      VertexID rank = new_ghosts[i].second;
      // std::cout << "R" << rank_ << " contract local graph " << std::endl;
      cg.AddGhostVertex(v, rank);   
      if (rank == rank_) {
        std::cout << "R" << rank_ << " This shouldn't happen! (contract add ghost)" << std::endl;
        exit(1);
      }
    }

    // Add edges
    for (auto &e : contracted_edges) {
      cg.AddEdge(e.source, e.target);
    }
    // TODO: Vertex 18 is missing edge to 33 on R0 at end of contraction
    // cg.OutputLocal();

    g = cg;
  }

  void SendLabels(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &label, PEID target, VertexID level) {
    std::vector<VertexID> vertices, labels;
    for (VertexID i = 0; i < vertex_buffer_[level].size(); i++) {
      VertexID v = vertex_buffer_[level][i];
      VertexID v_label = label_buffer_[level][i];
      VertexID cv = v;
      if (vertex_to_contracted_vertex_.find(v) != end(vertex_to_contracted_vertex_)) {
        cv = vertex_to_contracted_vertex_[v];
        // while (vertex_to_contracted_vertex_.find(cv) != end(vertex_to_contracted_vertex_)) {
        //   cv = vertex_to_contracted_vertex_[cv];
        // }
      }
      VertexID cv_label = label[cv];
      vertices.push_back(v);
      labels.push_back(cv_label);
      // std::cout << "R" << rank_ << " send (" << v << "," << cv_label << ") to R" << target << "(round " << level << ")" << std::endl;
    }

    MPI_Send(&vertices[0], static_cast<int>(vertices.size()),
             MPI_VERTEX, target, 
             target + 6 * size_, MPI_COMM_WORLD);
    MPI_Send(&labels[0], static_cast<int>(labels.size()),
             MPI_VERTEX, target, 
             target + 6 * size_, MPI_COMM_WORLD);
  }

  void ReceiveLabels(MinimalGraphAccess &g, google::sparse_hash_map<VertexID, VertexID> &label, PEID sender, VertexID level) {
    std::vector<VertexID> vertices, labels;
    MPI_Status st{};
    int message_length;

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    vertices.resize(message_length);
    MPI_Recv(&vertices[0], vertices.size(),
             MPI_VERTEX, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);
    MPI_Get_count(&st, MPI_VERTEX, &message_length);
    labels.resize(message_length);
    MPI_Recv(&labels[0], labels.size(),
             MPI_VERTEX, sender,
             rank_ + 6 * size_, MPI_COMM_WORLD, &st);

    for (VertexID i = 0; i < vertices.size(); i++) {
      label[vertices[i]] = labels[i];
      for (VertexID v : contracted_vertex_to_vertices_[vertices[i]]) {
        label[v] = labels[i];
      }
    }

    // TODO: Propagate labels locally
  }

  // void DistributeLabelsFromRoot(GraphType &g, std::vector<VertexID> &g_labels) {
  //   // Compute displacements
  //   std::vector<int> displ_labels(size_);
  //   int num_global_labels = 0;
  //   for (PEID i = 0; i < size_; ++i) {
  //     displ_labels[i] = num_global_labels;
  //     num_global_labels += num_vertices_per_pe_[i];
  //   }

  //   // Scatter to other PEs
  //   int num_local_vertices = vertex_buffer_.size();
  //   MPI_Scatterv(&global_labels_[0], &num_vertices_per_pe_[0], &displ_labels[0], MPI_VERTEX, 
  //                &label_buffer_[0], num_local_vertices, MPI_VERTEX, 
  //                ROOT, MPI_COMM_WORLD);

  //   for (int i = 0; i < num_local_vertices; ++i) {
  //     VertexID v = vertex_buffer_[i];
  //     g_labels[i] = label_buffer_[i];
  //   }
  // }

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
