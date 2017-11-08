#include <iostream>

#include "graph_access.h"
#include "ghost_communicator.h"

void GraphAccess::StartConstruct(const VertexID local_n, const EdgeID local_m, const VertexID local_offset) {
  number_of_vertices_ = local_n;
  number_of_local_vertices_ = local_n;
  number_of_edges_ = local_m;

  vertices_.resize(local_n);
  local_vertices_data_.resize(local_n, {0, false});
  edges_.resize(local_n);

  local_offset_ = local_offset;
  ghost_offset_ = number_of_local_vertices_;

  adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
  ghost_comm_ = new GhostCommunicator(this, rank_, size_, MPI_COMM_WORLD);
}

void GraphAccess::UpdateGhostVertices() {
  ghost_comm_->UpdateGhostVertices();
}

void GraphAccess::SetVertexLabel(const VertexID v, const VertexID label) {
  if (local_vertices_data_[v].label_ != label && local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddLabel(v, label);
  local_vertices_data_[v].label_ = label;
}

EdgeID GraphAccess::AddEdge(VertexID from, VertexID to, PEID rank) {
  if (IsLocalFromGlobal(to)) {
    edges_[from].emplace_back(to - local_offset_);
  } else {
    local_vertices_data_[from].is_interface_vertex_ = true;
    if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
      edges_[from].emplace_back(global_to_local_map_[to]);
    } else {
      global_to_local_map_[to] = number_of_vertices_++;
      edges_[from].emplace_back(global_to_local_map_[to]);

      if (rank_ == ROOT && rank == size_) std::cout << "get PE from offset for " << to << std::endl;
      PEID neighbor = (rank == size_) ? GetPEFromOffset(to) : rank;
      vertices_.emplace_back(0);
      local_vertices_data_.emplace_back(to, false);
      ghost_vertices_data_.emplace_back(neighbor, to);
      SetAdjacentPE(neighbor, true);
      ghost_comm_->SetAdjacentPE(neighbor, true);
    }
  }

  return edge_counter_++;
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
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "check label" << std::endl;
  ForallLocalVertices([&](const VertexID v) {
    std::cout << v << " -> " << GetVertexLabel(v) << " r " << rank << std::endl;
  });
  std::cout << std::endl;

  std::cout << "check neighbors" << std::endl;
  ForallLocalVertices([&](const VertexID v) {
    std::cout << v << " -> ";
    ForallNeighbors(v, [&](VertexID u) {
      std::cout << u << "(" << IsGhost(u) << "," << GetGlobalID(u) << "," << GetVertexLabel(u) << "," << GetPE(u)
                << ") ";
    });
    std::cout << " r " << rank << std::endl;
  });
  std::cout << std::endl;

  // std::cout << "check edges" << std::endl;
  // ForallLocalEdges([&](EdgeID e) {
  //     std::cout << edges_[e].target_ << "(" << GetPE(edges_[e].target_) << ")" << " ";
  //   });
  // std::cout << std::endl;
}

