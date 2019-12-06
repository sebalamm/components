#include <iostream>
#include <sstream>

#include "dynamic_graph_comm.h"
#include "dynamic_vertex_comm.h"
  
DynamicGraphCommunicator::DynamicGraphCommunicator(const PEID rank, const PEID size) 
    : rank_(rank),
      size_(size),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_cut_edges_(0),
      number_of_global_edges_(0),
      ghost_comm_(nullptr),
      vertex_counter_(0),
      ghost_vertex_counter_(0),
      edge_counter_(0),
      ghost_offset_(0),
      comm_time_(0.0) {
  ghost_comm_ = new DynamicVertexCommunicator(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
  label_shortcut_.set_empty_key(-1);
  global_to_local_map_.set_empty_key(-1);
}

DynamicGraphCommunicator::~DynamicGraphCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void DynamicGraphCommunicator::ResetCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = new DynamicVertexCommunicator(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
}

void DynamicGraphCommunicator::StartConstruct(const VertexID local_n,
                                              const VertexID ghost_n,
                                              const VertexID ghost_offset) {
  ghost_offset_ = ghost_offset;
  adjacent_pes_.resize(static_cast<unsigned long>(size_), false);
}

void DynamicGraphCommunicator::SendAndReceiveGhostVertices() {
  ghost_comm_->SendAndReceiveGhostVertices();
}

void DynamicGraphCommunicator::ReceiveAndSendGhostVertices() {
  ghost_comm_->ReceiveAndSendGhostVertices();
}

// TODO: v should always be local?
void DynamicGraphCommunicator::SetVertexPayload(const VertexID v,
                                          VertexPayload &&msg,
                                          bool propagate) {
  if (GetVertexMessage(v) != msg
      && local_vertices_data_[v].is_interface_vertex_
      && propagate)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

// TODO: v should always be local?
void DynamicGraphCommunicator::ForceVertexPayload(const VertexID v,
                                     VertexPayload &&msg) {
  if (local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

void DynamicGraphCommunicator::ReserveEdgesForVertex(VertexID v, VertexID num_edges) {
  if (IsLocal(v)) local_adjacent_edges_[v].reserve(num_edges);
  else ghost_adjacent_edges_[v - ghost_offset_].reserve(num_edges);
}

VertexID DynamicGraphCommunicator::AddGhostVertex(VertexID v) {
  AddGhostVertex(v, GetPEFromOffset(v));
}

VertexID DynamicGraphCommunicator::AddGhostVertex(VertexID v, PEID pe) {
  VertexID local_id = ghost_vertex_counter_ + ghost_offset_;
  global_to_local_map_[v] = local_id;
  ghost_vertex_counter_++;

  // Update data
  ghost_vertices_data_.resize(ghost_vertices_data_.size() + 1);
  ghost_payload_.resize(ghost_payload_.size() + 1);
  ghost_adjacent_edges_.resize(ghost_adjacent_edges_.size() + 1);
  ghost_parent_.resize(ghost_parent_.size() + 1);
  ghost_active_.resize(ghost_active_.size() + 1);
  ghost_vertices_data_[local_id - ghost_offset_].global_id_ = v;
  ghost_vertices_data_[local_id - ghost_offset_].rank_ = pe;

  // Set adjacent PE
  PEID neighbor = pe;

  // Set active
  ghost_active_[local_id - ghost_offset_] = true;

  // Set payload
  ghost_payload_[local_id - ghost_offset_] = {std::numeric_limits<VertexID>::max() - 1, 
                                              v, 
#ifdef TIEBREAK_DEGREE
                                              0,
#endif
                                              neighbor};

  number_of_vertices_++;
  return local_id;
}

// Local ID, Global ID, target rank
EdgeID DynamicGraphCommunicator::AddEdge(VertexID from, VertexID to, PEID rank) {
  if (IsLocalFromGlobal(to)) {
    AddLocalEdge(from, to);
  } else {
    PEID neighbor = (rank == size_) ? GetPEFromOffset(to) : rank;
    // TODO: Is from always local?
    local_vertices_data_[from].is_interface_vertex_ = true;
    if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
      number_of_cut_edges_++;
      AddGhostEdge(from, to);
      SetAdjacentPE(neighbor, true);
    } else {
      std::cout << "This shouldn't happen" << std::endl;
      exit(1);
    }
  }
  edge_counter_++;
  return edge_counter_;
}

EdgeID DynamicGraphCommunicator::RelinkEdge(VertexID from, VertexID old_to, VertexID new_to, PEID rank) {
  VertexID old_to_local = GetLocalID(old_to);
  VertexID new_to_local = GetLocalID(new_to);

  // TODO: Unsure if the ghost offset works with distributing high degree vertices
  if (IsLocal(new_to_local)) {
    if (IsGhost(old_to_local)) number_of_cut_edges_--;
  }
  else {
    if (!IsGhost(old_to_local)) number_of_cut_edges_++;
    PEID neighbor = (rank == size_) ? GetPEFromOffset(new_to) : rank;
    SetAdjacentPE(neighbor, true);
  }

  // Actual relink
  // TODO: from should always be local
  for (VertexID i = 0; i < local_adjacent_edges_[from].size(); i++) {
    if (local_adjacent_edges_[from][i].target_ == old_to_local) {
      local_adjacent_edges_[from][i].target_ = new_to_local;
    }
  }

  // TODO: Remove later
  // => We might want to do a sweep after ALL relinking operations are done and update accordingly
  // => For now do it here and fix if everything works (very costly)
  std::vector<bool> is_neighbor(size_, false);
  ForallLocalVertices([&](const VertexID v) {
    ForallNeighbors(v, [&](const VertexID w) {
      if (IsGhost(w)) {
        is_neighbor[GetPE(v)] = true;
      }
    });
  });

  // Update PEs
  for (PEID i = 0; i < size_; i++) {
    SetAdjacentPE(i, is_neighbor[i]);
  }

  // Check if from is still an interface vertex
  ForallNeighbors(from, [&](const VertexID w) {
    if (IsGhost(w)) {
      local_vertices_data_[from].is_interface_vertex_ = true;
    }
  });

  return edge_counter_;
}

EdgeID DynamicGraphCommunicator::RemoveEdge(VertexID from, VertexID to) {
  VertexID delete_pos = ghost_offset_;
  if (IsLocal(from)) {
    for (VertexID i = 0; i < local_adjacent_edges_[from].size(); i++) {
      if (local_adjacent_edges_[from][i].target_ == to) {
        delete_pos = i;
        break;
      }
    }
    if (delete_pos != ghost_offset_) {
      local_adjacent_edges_[from].erase(local_adjacent_edges_[from].begin() + delete_pos);
    } else {
      std::cout << "This shouldn't happen" << std::endl;
      exit(1);
    }
  } else {
    std::cout << "This shouldn't happen" << std::endl;
    exit(1);
  }
}

void DynamicGraphCommunicator::RemoveAllEdges(const VertexID from) {
  if (IsLocal(from)) local_adjacent_edges_[from].clear();
  else ghost_adjacent_edges_[from - ghost_offset_].clear();
}

void DynamicGraphCommunicator::AddLocalEdge(VertexID from, VertexID to) {
  if (IsLocal(from)) local_adjacent_edges_[from].emplace_back(global_to_local_map_[to]);
  else ghost_adjacent_edges_[from - ghost_offset_].emplace_back(global_to_local_map_[to]);
}

void DynamicGraphCommunicator::AddGhostEdge(VertexID from, VertexID to) {
  AddLocalEdge(from, to);
}

void DynamicGraphCommunicator::SetAdjacentPE(const PEID pe, const bool is_adj) {
    if (pe == rank_) return;
    adjacent_pes_[pe] = is_adj;
    ghost_comm_->SetAdjacentPE(pe, is_adj);
}

void DynamicGraphCommunicator::ResetAdjacentPEs() {
  for (PEID i = 0; i < adjacent_pes_.size(); ++i) {
    SetAdjacentPE(i, false);
  }
}

bool DynamicGraphCommunicator::CheckDuplicates() {
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

void DynamicGraphCommunicator::OutputLocal() {
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

void DynamicGraphCommunicator::OutputLabels() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ForallLocalVertices([&](const VertexID v) {
    std::stringstream out;
    out << "[R" << rank << "] [V] "
        << GetGlobalID(v) << " label="
        << local_payload_[v].label_;
    std::cout << out.str() << std::endl;
  });
}

void DynamicGraphCommunicator::OutputGhosts() {
  PEID rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "[R" << rank << "] [G] [ ";
  for (auto &e : global_to_local_map_) {
    std::cout << e.first << " ";
  }
  std::cout << "]" << std::endl;
}

void DynamicGraphCommunicator::OutputComponents(std::vector<VertexID> &labels) {
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

float DynamicGraphCommunicator::GetCommTime() {
  return comm_time_ + ghost_comm_->GetCommTime();
}

