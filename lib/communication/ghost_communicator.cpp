#include <iostream>

#include "ghost_communicator.h"

void GhostCommunicator::AddMessage(const VertexID v, const VertexPayload &msg) {
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) {
      if (msg == g_->GetVertexMessage(u)) return;
      PEID neighbor = g_->GetPE(u);
      if (!packed_pes_[neighbor]) {
        // Unpack msg and add content (Sender, Deviate, Component, PE (of component))
        (*current_send_buffers_)[neighbor].emplace_back(g_->GetGlobalID(v));
        (*current_send_buffers_)[neighbor].emplace_back(msg.deviate_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.label_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.root_);
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
    if (message_length == 0) continue;
    messages_recv++;

    if (message_length == 1) continue;

#ifndef NDEBUG
    for (int i = 0; i < message_length - 1; i += 4) {
      VertexID local_id = g_->GetLocalID(message[i]);
      VertexID deviate = message[i + 1];
      VertexID label = message[i + 2];
      auto root = static_cast<PEID>(message[i + 3]);
      std::cout << "[R" << rank_ << "] recv (" << deviate << "," << label
                << "," << root << ") from pe "
                << st.MPI_SOURCE << " with tag " << current_recv_tag_
                << " length " << message_length << " ["
                << messages_recv << "/" << GetNumberOfAdjacentPEs() << "]"
                << std::endl;
      g_->HandleGhostUpdate(local_id, label, deviate, root);
    }
#endif
  }
  isend_requests_.clear();
}

