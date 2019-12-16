#include <iostream>
#include <sstream>

#include "dynamic_graph_comm.h"
#include "vertex_comm.h"
  
DynamicGraphCommunicator::DynamicGraphCommunicator(const PEID rank, const PEID size) 
    : DynamicGraph(rank, size), 
      ghost_comm_(nullptr) {
  ghost_comm_ = new VertexCommunicator<DynamicGraphCommunicator>(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
}

DynamicGraphCommunicator::~DynamicGraphCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void DynamicGraphCommunicator::ResetCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = new VertexCommunicator<DynamicGraphCommunicator>(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
}

void DynamicGraphCommunicator::SendAndReceiveGhostVertices() {
  ghost_comm_->SendAndReceiveGhostVertices();
}

void DynamicGraphCommunicator::ReceiveAndSendGhostVertices() {
  ghost_comm_->ReceiveAndSendGhostVertices();
}

// NOTE: v should always be local
void DynamicGraphCommunicator::SetVertexPayload(const VertexID v,
                                          VertexPayload &&msg,
                                          bool propagate) {
  if (GetVertexMessage(v) != msg
      && local_vertices_data_[v].is_interface_vertex_
      && propagate)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
}

// NOTE: v should always be local?
void DynamicGraphCommunicator::ForceVertexPayload(const VertexID v,
                                     VertexPayload &&msg) {
  if (local_vertices_data_[v].is_interface_vertex_)
    ghost_comm_->AddMessage(v, msg);
  SetVertexMessage(v, std::move(msg));
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

  // Set active
  ghost_active_[local_id - ghost_offset_] = true;

  // Set payload
  ghost_payload_[local_id - ghost_offset_] = {std::numeric_limits<VertexID>::max() - 1, 
                                              v, 
#ifdef TIEBREAK_DEGREE
                                              0,
#endif
                                              pe};

  number_of_vertices_++;
  return local_id;
}

void DynamicGraphCommunicator::SetAdjacentPE(const PEID pe, const bool is_adj) {
  DynamicGraph::SetAdjacentPE(pe, is_adj);
  ghost_comm_->SetAdjacentPE(pe, is_adj);
}

void DynamicGraphCommunicator::ResetAdjacentPEs() {
  DynamicGraph::ResetAdjacentPEs();
  ghost_comm_->ResetAdjacentPEs();
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

float DynamicGraphCommunicator::GetCommTime() {
  return comm_time_ + ghost_comm_->GetCommTime();
}

