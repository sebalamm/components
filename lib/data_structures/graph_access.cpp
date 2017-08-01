#include <iostream>

#include "graph_access.h"
#include "ghost_communicator.h"

void GraphAccess::StartConstruct(const VertexID local_n, const EdgeID local_m,
                    const VertexID global_n, const EdgeID global_m) {
  number_of_local_vertices_ = local_n;
  number_of_vertices_ = local_n;
  number_of_global_vertices_ = global_n;
  number_of_local_edges_ = local_m;
  number_of_global_edges_ = global_m;

  ghost_offset_ = local_n;

  vertices_.resize(local_n + 1, 0);
  local_vertices_data_.resize(local_n, {0, false});
  edges_.resize(local_m + 1, 0);

  pe_div_ = ceil(global_n / (double)size_);
  ghost_comm_ = new GhostCommunicator(this, rank_, size_, MPI_COMM_WORLD);
}

void GraphAccess::UpdateGhostVertices() {
  ghost_comm_->UpdateGhostVertices();
}

void GraphAccess::SetVertexLabel(const VertexID v, const VertexID label) {
  if (local_vertices_data_[v].label_ != label && local_vertices_data_[v].is_interface_node_)
    ghost_comm_->AddLabel(v, label);
  local_vertices_data_[v].label_ = label;
}

EdgeID GraphAccess::CreateEdge(VertexID from, VertexID to) {
  // if (rank_ == 1) std::cout << "gen e (" << from << ","  << to << ")" << std::endl;
  if (IsLocalFromGlobal(to)) {
    edges_[edge_counter_].local_target_ = to - range_from_;
  } else {
    local_vertices_data_[from].is_interface_node_ = true;
    if (IsGhostFromGlobal(to)) {
      edges_[edge_counter_].local_target_ = global_to_local_map_[to];
    } else {
      global_to_local_map_[to] = number_of_vertices_++;
      // if (rank_ == 1) std::cout << "insert map " << to << " -> " << number_of_vertices_ - 1 << std::endl;
      edges_[edge_counter_].local_target_ = global_to_local_map_[to];

      PEID neighbor = GetPEFromRange(to);
      vertices_.emplace_back(0);
      local_vertices_data_.emplace_back(to, false);
      non_local_vertices_data_.emplace_back(neighbor, to);
      ghost_comm_->SetAdjacentPE(neighbor, true);
    }
  }

  EdgeID e_prime = edge_counter_++;
  vertices_[from + 1].first_edge_ = edge_counter_;

  // Isolated vertex
  if (prev_from_ + 1 < from) {
    for (VertexID i = from; i > prev_from_ + 1; i--) {
      vertices_[i].first_edge_ = vertices_[prev_from_+1].first_edge_;
    }
  }
  prev_from_ = from;

  degree_counter_++;
  if (max_degree_ < degree_counter_) max_degree_ = degree_counter_;

  return e_prime;
}

void GraphAccess::OutputLocal() {
  // std::cout << "Vertices " << NumberOfGlobalVertices() << std::endl;
  // std::cout << "|- Stored " << vertices_.size() - 1 << std::endl;
  // std::cout << "|- Local " << NumberOfLocalVertices() << std::endl;
  // std::cout << "|- Ghost " << NumberOfGhostVertices() << std::endl;
  // std::cout << "Edges " << NumberOfGlobalEdges() << std::endl;
  // std::cout << "|- Stored " << edges_.size() << std::endl;
  // std::cout << "|- Local " << NumberOfLocalEdges() << std::endl;
  // std::cout << std::endl;

  // std::cout << "check local" << std::endl;
  // ForallLocalVertices([&](VertexID v) {
  //     std::cout << v << " -> " << IsLocal(v) << std::endl;
  //   });
  // std::cout << std::endl;

  // std::cout << "check ids" << std::endl;
  // ForallLocalVertices([&](VertexID v) {
  //     std::cout << v << " -> " << GetGlobalID(v) << std::endl;
  //   });
  // std::cout << std::endl;

  std::cout << "check label" << std::endl;
  ForallLocalVertices([&](const VertexID v) {
      std::cout << v << " -> " << GetVertexLabel(v) << std::endl;
    });
  std::cout << std::endl;

  std::cout << "check neighbors" << std::endl;
  ForallLocalVertices([&](const VertexID v) {
      std::cout << v << " -> ";
      ForallNeighbors(v, [&](VertexID u) {
        std::cout << u << "(" << IsGhost(u) << "," << GetGlobalID(u) << "," << GetVertexLabel(u) << ") ";
        });
      std::cout << std::endl;
    });
  std::cout << std::endl;

  // std::cout << "check edges" << std::endl;
  // ForallLocalEdges([&](EdgeID e) {
  //     std::cout << edges_[e].local_target_ << "(" << GetPE(edges_[e].local_target_) << ")" << " ";
  //   });
  // std::cout << std::endl;
}

