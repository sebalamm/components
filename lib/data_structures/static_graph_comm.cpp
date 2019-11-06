#include <iostream>
#include <sstream>

#include "static_graph_comm.h"
#include "static_vertex_comm.h"
  
StaticGraphCommunicator::StaticGraphCommunicator(const PEID rank, const PEID size) 
    : rank_(rank),
      size_(size),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_cut_edges_(0),
      number_of_global_edges_(0),
      max_degree_(0),
      max_degree_computed_(false),
      local_offset_(0),
      ghost_offset_(0),
      ghost_comm_(nullptr),
      vertex_counter_(0),
      edge_counter_(0),
      ghost_counter_(0),
      last_source_(0) {
  ghost_comm_ = new StaticVertexCommunicator(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
  global_to_local_map_.set_empty_key(-1);
}

StaticGraphCommunicator::~StaticGraphCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void StaticGraphCommunicator::ResetCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = new StaticVertexCommunicator(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
}

void StaticGraphCommunicator::StartConstruct(const VertexID local_n, 
                                             const VertexID ghost_n, 
                                             const VertexID total_m,
                                             const VertexID local_offset) {
    number_of_local_vertices_ = local_n;
    number_of_vertices_ = local_n + ghost_n;
    number_of_edges_ = total_m;

    vertices_.resize(number_of_vertices_ + 1);
    // Fix first node being isolated
    if (local_n > 0) vertices_[0].first_edge_ = 0;
    edges_.resize(number_of_edges_);

    local_vertices_data_.resize(number_of_vertices_);
    ghost_vertices_data_.resize(ghost_n);
    vertex_payload_.resize(number_of_vertices_);

    local_offset_ = local_offset;
    ghost_offset_ = local_n;

    // Temp counter for properly counting new ghost vertices
    vertex_counter_ = local_n; 
    edge_counter_ = 0;
    ghost_counter_ = local_n;

    adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
}

void StaticGraphCommunicator::SendAndReceiveGhostVertices() {
  ghost_comm_->SendAndReceiveGhostVertices();
}

void StaticGraphCommunicator::ReceiveAndSendGhostVertices() {
  ghost_comm_->ReceiveAndSendGhostVertices();
}

void StaticGraphCommunicator::SetVertexPayload(const VertexID v,
                                          VertexPayload &&msg,
                                          bool propagate) {
  if (GetVertexMessage(v) != msg
      && local_vertices_data_[v].is_interface_vertex_
      && propagate)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

void StaticGraphCommunicator::ForceVertexPayload(const VertexID v,
                                     VertexPayload &&msg) {
  if (local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

VertexID StaticGraphCommunicator::AddGhostVertex(VertexID v) {
  AddVertex();
  global_to_local_map_[v] = ghost_counter_;

  if (ghost_counter_ > local_vertices_data_.size()) {
    local_vertices_data_.resize(ghost_counter_ + 1);
    ghost_vertices_data_.resize(ghost_counter_ + 1);
    vertex_payload_.resize(ghost_counter_ + 1);
  }

  // Update data
  local_vertices_data_[ghost_counter_].is_interface_vertex_ = false;
  ghost_vertices_data_[ghost_counter_ - ghost_offset_].rank_ = GetPEFromOffset(v);
  ghost_vertices_data_[ghost_counter_ - ghost_offset_].global_id_ = v;

  // Set adjacent PE
  PEID neighbor = GetPEFromOffset(v);

  // Set payload
  vertex_payload_[ghost_counter_] = {std::numeric_limits<VertexID>::max() - 1, 
                                     v, 
#ifdef TIEBREAK_DEGREE
                                     0,
#endif
                                     neighbor};

  return ghost_counter_++;
}

// Local ID, Global ID, target rank
EdgeID StaticGraphCommunicator::AddEdge(VertexID from, VertexID to, PEID rank) {
  if (IsLocalFromGlobal(to)) {
    AddLocalEdge(from, to);
  } else {
    PEID neighbor = (rank == size_) ? GetPEFromOffset(to) : rank;
    local_vertices_data_[from].is_interface_vertex_ = true;
    if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
      number_of_cut_edges_++;
      AddLocalEdge(from, to);
      SetAdjacentPE(neighbor, true);
    } else {
      std::cout << "This shouldn't happen" << std::endl;
      exit(1);
    }
  }
  if (from > last_source_) last_source_ = from;
  return edge_counter_;
}

void StaticGraphCommunicator::AddLocalEdge(VertexID from, VertexID to) {
  edges_[edge_counter_].target_ = GetLocalID(to);
  edge_counter_++;
  vertices_[from + 1].first_edge_ = edge_counter_;
}

void StaticGraphCommunicator::SetAdjacentPE(const PEID pe, const bool is_adj) {
    if (pe == rank_) return;
    adjacent_pes_[pe] = is_adj;
    ghost_comm_->SetAdjacentPE(pe, is_adj);
}

void StaticGraphCommunicator::ResetAdjacentPEs() {
  for (PEID i = 0; i < adjacent_pes_.size(); ++i) {
    SetAdjacentPE(i, false);
  }
}

void StaticGraphCommunicator::OutputLocal() {
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

void StaticGraphCommunicator::OutputLabels() {
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

void StaticGraphCommunicator::OutputGhosts() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "[R" << rank << "] [G] [ ";
  for (auto &e : global_to_local_map_) {
    std::cout << e.first << " ";
  }
  std::cout << "]" << std::endl;
}

void StaticGraphCommunicator::OutputComponents(std::vector<VertexID> &labels) {
  VertexID global_num_vertices = GatherNumberOfGlobalVertices();
  // Gather component sizes
  google::dense_hash_map<VertexID, VertexID> local_component_sizes; 
  local_component_sizes.set_empty_key(-1);
  ForallLocalVertices([&](const VertexID v) {
    VertexID c = labels[v];
    if (local_component_sizes.find(c) == end(local_component_sizes))
      local_component_sizes[c] = 0;
    local_component_sizes[c]++;
  });

  // Gather component message
  std::vector<std::pair<VertexID, VertexID>> local_components;
  // local_components.reserve(local_component_sizes.size());
  for(auto &kv : local_component_sizes)
    local_components.emplace_back(kv.first, kv.second);
  // TODO [MEMORY]: Might be too small
  int num_local_components = local_components.size();

  // Exchange number of local components
  std::vector<int> num_components(size_);
  MPI_Gather(&num_local_components, 1, MPI_INT, &num_components[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

  // Compute diplacements
  std::vector<int> displ_components(size_, 0);
  // TODO [MEMORY]: Might be too small
  int num_global_components = 0;
  for (PEID i = 0; i < size_; ++i) {
    displ_components[i] = num_global_components;
    num_global_components += num_components[i];
  }

  // Add datatype
  MPI_Datatype MPI_COMP;
  MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
  MPI_Type_commit(&MPI_COMP);

  // Exchange components
  std::vector<std::pair<VertexID, VertexID>> global_components(num_global_components);
  MPI_Gatherv(&local_components[0], num_local_components, MPI_COMP,
              &global_components[0], &num_components[0], &displ_components[0], MPI_COMP,
              ROOT, MPI_COMM_WORLD);

  if (rank_ == ROOT) {
    google::dense_hash_map<VertexID, VertexID> global_component_sizes; global_component_sizes.set_empty_key(-1);
    for (auto &comp : global_components) {
      VertexID c = comp.first;
      VertexID size = comp.second;
      if (global_component_sizes.find(c) == end(global_component_sizes))
        global_component_sizes[c] = 0;
      global_component_sizes[c] += size;
    }

    google::dense_hash_map<VertexID, VertexID> condensed_component_sizes; condensed_component_sizes.set_empty_key(-1);
    for (auto &cs : global_component_sizes) {
      VertexID c = cs.first;
      VertexID size = cs.second;
      if (condensed_component_sizes.find(size) == end(condensed_component_sizes)) {
        condensed_component_sizes[size] = 0;
      }
      condensed_component_sizes[size]++;
    }

    // Build final vector
    std::vector<std::pair<VertexID, VertexID>> components;
    components.reserve(condensed_component_sizes.size());
    for(auto &kv : condensed_component_sizes)
      components.emplace_back(kv.first, kv.second);
    std::sort(begin(components), end(components));

    std::cout << "COMPONENTS [ ";
    for (auto &comp : components)
      std::cout << "size=" << comp.first << " (num=" << comp.second << ") ";
    std::cout << "]" << std::endl;
  }
}
