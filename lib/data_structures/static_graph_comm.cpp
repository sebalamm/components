#include <iostream>
#include <sstream>

#include "static_graph_comm.h"
#include "vertex_comm.h"
  
StaticGraphCommunicator::StaticGraphCommunicator(const PEID rank, const PEID size) 
    : StaticGraph(rank, size),
      ghost_comm_(nullptr) {
  ghost_comm_ = new VertexCommunicator<StaticGraphCommunicator>(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
}

StaticGraphCommunicator::~StaticGraphCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void StaticGraphCommunicator::ResetCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = new VertexCommunicator<StaticGraphCommunicator>(rank_, size_, MPI_COMM_WORLD);
  ghost_comm_->SetGraph(this);
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

void StaticGraphCommunicator::SetAdjacentPE(const PEID pe, const bool is_adj) {
    StaticGraph::SetAdjacentPE(pe, is_adj);
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

float StaticGraphCommunicator::GetCommTime() {
  return comm_time_ + ghost_comm_->GetCommTime();
}


