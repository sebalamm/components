#include <iostream>
#include <sstream>

#include "graph_access.h"
#include "node_communicator.h"

GraphAccess::GraphAccess(const PEID rank, const PEID size) 
    : rank_(rank),
      size_(size),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_global_edges_(0),
      max_degree_(0),
      max_degree_computed_(false),
      local_offset_(0),
      ghost_offset_(0),
      ghost_comm_(nullptr),
      vertex_counter_(0),
      edge_counter_(0) {
  ghost_comm_ = new NodeCommunicator(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
  label_shortcut_.set_empty_key(-1);
  global_to_local_map_.set_empty_key(-1);
}

GraphAccess::~GraphAccess() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void GraphAccess::StartConstruct(const VertexID local_n,
                                 const EdgeID local_m,
                                 const VertexID local_offset) {
  number_of_vertices_ = local_n;
  number_of_local_vertices_ = local_n;
  number_of_edges_ = local_m;

  edges_.resize(local_n);
  local_vertices_data_.resize(local_n);
  vertex_payload_.resize(local_n);

  local_offset_ = local_offset;
  ghost_offset_ = local_n;

  parent_.resize(local_n);
  is_active_.resize(local_n, true);

  adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
}

void GraphAccess::SendAndReceiveGhostVertices() {
  ghost_comm_->SendAndReceiveGhostVertices();
}

void GraphAccess::ReceiveAndSendGhostVertices() {
  ghost_comm_->ReceiveAndSendGhostVertices();
}

void GraphAccess::SetVertexPayload(const VertexID v,
                                   VertexPayload &&msg,
                                   bool propagate) {
  if (GetVertexMessage(v) != msg
      && local_vertices_data_[v].is_interface_vertex_
      && propagate)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

void GraphAccess::ForceVertexPayload(const VertexID v,
                                     VertexPayload &&msg) {
  if (local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

// Local ID, Global ID, target rank
EdgeID GraphAccess::AddEdge(VertexID from, VertexID to, PEID rank) {
  if (IsLocalFromGlobal(to)) {
    edges_[from].emplace_back(to - local_offset_);
  } else {
    PEID neighbor = (rank == size_) ? GetPEFromOffset(to) : rank;
    local_vertices_data_[from].is_interface_vertex_ = true;
    if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
      edges_[from].emplace_back(global_to_local_map_[to]);
      edges_[global_to_local_map_[to]].emplace_back(from);
      // inactive_level_[global_to_local_map_[to]] = -1;
      is_active_[global_to_local_map_[to]] = true;
      vertex_payload_[global_to_local_map_[to]] = {std::numeric_limits<VertexID>::max() - 1, 
                                                   GetVertexLabel(global_to_local_map_[to]), 
#ifdef TIEBREAK_DEGREE
                                                   0,
#endif
                                                   neighbor};
    } else {
      global_to_local_map_[to] = number_of_vertices_++;
      edges_[from].emplace_back(global_to_local_map_[to]);
      edges_.resize(number_of_vertices_);
      edges_[global_to_local_map_[to]].emplace_back(from);
      // inactive_level_.resize(number_of_vertices_);
      // inactive_level_[global_to_local_map_[to]] = -1;
      is_active_.resize(number_of_vertices_, true);
      local_vertices_data_.emplace_back(to, false);
      ghost_vertices_data_.emplace_back(neighbor, to);
      SetAdjacentPE(neighbor, true);
      ghost_comm_->SetAdjacentPE(neighbor, true);
      // Contraction additions
      vertex_payload_.emplace_back(std::numeric_limits<VertexID>::max() - 1, 
                                   to, 
#ifdef TIEBREAK_DEGREE
                                   0,
#endif
                                   neighbor);
    }
  }

  return edge_counter_++;
}

void GraphAccess::RemoveAllEdges(const VertexID from) {
  edges_[from].clear();
}

bool GraphAccess::CheckDuplicates() {
  // google::dense_hash_set<VertexID> neighbors;
  ForallLocalVertices([&](const VertexID v) {
    std::unordered_set<VertexID> neighbors;
    ForallNeighbors(v, [&](const VertexID w) {
      if (neighbors.find(w) != end(neighbors)) {
        std::cout << "[R" << rank_ << ":0] DUPL (" << GetGlobalID(v) << "," << GetGlobalID(w) << "[" << GetPE(w) << "])" << std::endl;
        return true;
      }
      neighbors.insert(w);
    });
  });
  return false;
}

void GraphAccess::OutputLocal() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << "] [V] "
        << GetGlobalID(v) << " (local_id=" << v
        << ", msg=" << GetVertexString(v) << ", pe="
        << rank << ")";
    std::cout << out.str() << std::endl;
  });

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << "] [N] "
        << GetGlobalID(v) << " -> ";
    ForallNeighbors(v, [&](VertexID u) {
      // out << GetGlobalID(u) << " ";
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
        << vertex_payload_[v].label_;
    std::cout << out.str() << std::endl;
  });
}

void GraphAccess::OutputGhosts() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "[R" << rank << "] [G] [ ";
  for (auto &e : global_to_local_map_) {
    std::cout << e.first << " ";
  }
  std::cout << "]" << std::endl;
}

