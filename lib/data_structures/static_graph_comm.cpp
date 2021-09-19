#include <iostream>
#include <sstream>

#include "static_graph_comm.h"
#include "vertex_comm.h"
  
StaticGraphCommunicator::StaticGraphCommunicator(const Config& conf, const PEID rank, const PEID size) 
    : StaticGraph(conf, rank, size),
      ghost_comm_(nullptr) {
  ghost_comm_ = new VertexCommunicator<StaticGraphCommunicator>(config_, rank_, size_);
  ghost_comm_->SetGraph(this);
}

StaticGraphCommunicator::~StaticGraphCommunicator() {
  delete ghost_comm_;
  ghost_comm_ = nullptr;
}

void StaticGraphCommunicator::ResetCommunicator() {
  ghost_comm_ = new VertexCommunicator<StaticGraphCommunicator>(config_, rank_, size_);
  ghost_comm_->SetGraph(this);
}

void StaticGraphCommunicator::SendAndReceiveGhostVertices() {
  ghost_comm_->SendAndReceiveGhostVertices();
}

void StaticGraphCommunicator::SampleVertexNeighborhood(const VertexID &v, const float sampling_factor) {
  ghost_comm_->SampleVertexNeighborhood(v, sampling_factor);
}

void StaticGraphCommunicator::SetVertexPayload(const VertexID v,
                                          VertexPayload &&msg,
                                          bool propagate) {
  if (GetVertexMessage(v) != msg
      && IsLocal(v)
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

VertexID StaticGraphCommunicator::AddGhostVertex(VertexID v, PEID pe) {
  global_to_local_map_[v] = ghost_counter_;

  // Fix overflows
  if (vertex_counter_ >= is_active_.size()) {
    is_active_.resize(vertex_counter_ + 1);
  }
  if (vertex_counter_ >= vertex_payload_.size()) {
    vertex_payload_.resize(vertex_counter_ + 1);
  }
  if (ghost_counter_ - ghost_offset_ >= GetGhostVertexVectorSize()) {
    ghost_vertices_data_.resize(ghost_counter_ + 1);
  }

  // Update data
  ghost_vertices_data_[ghost_counter_ - ghost_offset_].rank_ = pe;
  ghost_vertices_data_[ghost_counter_ - ghost_offset_].global_id_ = v;
  is_active_[vertex_counter_] = true;

  // Set payload
  vertex_payload_[vertex_counter_] = {MaxDeviate, 
                                      v, 
#ifdef TIEBREAK_DEGREE
                                      0,
#endif
                                      pe};

  vertex_counter_++;
  return ghost_counter_++;
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

VertexID StaticGraphCommunicator::GetSendVolume() {
  return send_volume_ + ghost_comm_->GetSendVolume();
}

VertexID StaticGraphCommunicator::GetReceiveVolume() {
  return recv_volume_ + ghost_comm_->GetReceiveVolume();
}

