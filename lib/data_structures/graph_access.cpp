#include <iostream>
#include <sstream>

#include "graph_access.h"
#include "blocking_communicator.h"

void GraphAccess::StartConstruct(const VertexID local_n,
                                 const EdgeID local_m,
                                 const VertexID local_offset) {
  number_of_vertices_ = local_n;
  number_of_local_vertices_ = local_n;
  number_of_edges_ = local_m;

  edges_.resize(local_n);
  local_vertices_data_.resize(local_n);
  vertex_payload_.resize(local_n);
  vertex_payload_[0].resize(local_n);

  local_offset_ = local_offset;
  ghost_offset_ = local_n;

  active_vertices_.resize(local_n);
  active_vertices_[0].resize(local_n, true);

  parent_.resize(local_n);
  parent_[0].resize(local_n);

  added_edges_.resize(local_n);
  removed_edges_.resize(local_n);

  adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
  ghost_comm_ = new BlockingCommunicator(this, rank_, size_, MPI_COMM_WORLD);
}

void GraphAccess::UpdateGhostVertices() {
  ghost_comm_->UpdateGhostVertices();
}

void GraphAccess::SetVertexPayload(const VertexID v,
                                   const VertexPayload &msg) {
  if (GetVertexMessage(v) != msg
      && local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, msg);
}

// Local ID, Global ID, target rank
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

      PEID neighbor = (rank == size_) ? GetPEFromOffset(to) : rank;
      local_vertices_data_.emplace_back(to, false);
      // TODO: Might be problematic
      vertex_payload_[contraction_level_].emplace_back(
          std::numeric_limits<VertexID>::max() - 1, to, rank);
      ghost_vertices_data_.emplace_back(neighbor, to);
      SetAdjacentPE(neighbor, true);
      ghost_comm_->SetAdjacentPE(neighbor, true);
    }
  }

  return edge_counter_++;
}

// Local IDs
void GraphAccess::RemoveEdge(const VertexID from, const VertexID to) {
  for (EdgeID i = 0; i < edges_[from].size() - 1; ++i) {
    if (edges_[from][i].target_ == to)
      iter_swap(edges_[from].begin() + i,
                edges_[from].end() - 1);
  }
  edges_[from].pop_back();
}

void GraphAccess::OutputLocal() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << ":" << contraction_level_ << "] [V] "
        << GetGlobalID(v) << " (local_id=" << v
        << ", msg=" << GetVertexString(v) << ", pe="
        << rank << ")";
    std::cout << out.str() << std::endl;
  });

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << ":" << contraction_level_ << "] [N] "
        << GetGlobalID(v) << " -> ";
    ForallNeighbors(v, [&](VertexID u) {
      out << GetGlobalID(u) << " (local_id=" << u << ", msg="
          << GetVertexString(u) << ", pe="
          << GetPE(u) << ") ";
    });
    std::cout << out.str() << std::endl;
  });
}

void GraphAccess::OutputLabels() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << "] [V] "
        << GetGlobalID(v) << " label="
        << vertex_payload_[contraction_level_][v].label_;
    std::cout << out.str() << std::endl;
  });
}

void GraphAccess::Logging(bool active) {
  ghost_comm_->Logging(active);
  logging_ = active;
}

