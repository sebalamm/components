#include <iostream>

#include "ghost_communicator.h"

void GhostCommunicator::AddLabel(const VertexID v, const VertexID label) {
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) {
      if (label == g_->GetVertexLabel(u)) return;
      PEID neighbor = g_->GetPE(u);
      if (!packed_pes_[neighbor]) {
        (*current_send_buffers_)[neighbor].emplace_back(g_->GetGlobalID(v));
        (*current_send_buffers_)[neighbor].emplace_back(label);
        packed_pes_[neighbor] = true;
      }
    }
  });

  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) packed_pes_[g_->GetPE(u)] = false;
  });
}

void GhostCommunicator::ReceiveIncomingMessages() {
  PEID messages_recv = 0;
  current_recv_tag_++;

  while (messages_recv < GetNumberOfAdjacentPEs()) {
    MPI_Status st{};
    MPI_Probe(MPI_ANY_SOURCE, current_recv_tag_, communicator_, &st);

    int message_length;
    MPI_Get_count(&st, MPI_LONG, &message_length);

    std::vector<VertexID> message(static_cast<unsigned long>(message_length));
    MPI_Status rst{};
    MPI_Recv(&message[0], message_length,
             MPI_LONG, st.MPI_SOURCE,
             current_recv_tag_, communicator_, &rst);
    messages_recv++;

    if (message_length == 1) continue;

    for (int i = 0; i < message_length - 1; i += 2) {
      VertexID global_id = message[i];
      VertexID label = message[i + 1];

      VertexID local_id = g_->GetLocalID(global_id);
      g_->HandleGhostUpdate(local_id, label);
    }
  }

  isend_requests_.clear();
}

