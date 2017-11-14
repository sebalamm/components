#include <iostream>

#include "ghost_communicator.h"

void GhostCommunicator::AddLabel(const VertexID v, const VertexID label, const VertexID msg) {
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) {
      if (label == g_->GetVertexLabel(u) && msg == g_->GetVertexMsg(u)) return;
      PEID neighbor = g_->GetPE(u);
      if (!packed_pes_[neighbor]) {
        (*current_send_buffers_)[neighbor].emplace_back(g_->GetGlobalID(v));
        (*current_send_buffers_)[neighbor].emplace_back(msg);
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

  // std::cout << "[R" << rank_ << "] recv tag " << current_recv_tag_ << std::endl;
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
    // TODO: we shouldn't need this right?
    if (message_length == 0) continue;
    messages_recv++;

    // std::cout << "[R" << rank_ << "] message found [" << messages_recv << "/" << GetNumberOfAdjacentPEs()
    //           << "] with tag " << current_recv_tag_ << " length " << message_length << std::endl;
    if (message_length == 1) continue;

    for (int i = 0; i < message_length - 1; i += 3) {
      VertexID global_id = message[i];
      VertexID msg = message[i + 1];
      VertexID label = message[i + 2];

      // std::cout << "[R" << rank_ << "] recv (" << global_id << "," << msg << "," << label << ") from pe "
      //           << st.MPI_SOURCE << " with tag " << current_recv_tag_ << " length " << message_length << " ["
      //           << messages_recv << "/" << GetNumberOfAdjacentPEs() << "]" << std::endl;

      VertexID local_id = g_->GetLocalID(global_id);
      g_->HandleGhostUpdate(local_id, label, msg);
    }
  }

  isend_requests_.clear();
}

