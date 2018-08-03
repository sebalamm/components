#include <iostream>

#include "node_communicator.h"

void NodeCommunicator::AddMessage(const VertexID v,
                                      const VertexPayload &msg) {
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) {
      PEID neighbor = g_->GetPE(u);
      if (!packed_pes_[neighbor]) {
        // Unpack msg and add content (Sender, Deviate, Component, PE (of component))
        (*current_send_buffers_)[neighbor].emplace_back(g_->GetGlobalID(v));
        (*current_send_buffers_)[neighbor].emplace_back(msg.deviate_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.label_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.root_);
        packed_pes_[neighbor] = true;

        // if (g_->GetGlobalID(v) == 6) {
        //   g_->OutputLocal();
        //   std::cout << rank_ << std::endl;
        // }
      }
    }
  });

  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) packed_pes_[g_->GetPE(u)] = false;
  });
}

void NodeCommunicator::ReceiveMessages() {
  PEID messages_recv = 0;
  recv_tag_++;
  while (messages_recv < GetNumberOfAdjacentPEs()) {
    MPI_Status st{};
    MPI_Probe(MPI_ANY_SOURCE, recv_tag_, communicator_, &st);

    int message_length;
    MPI_Get_count(&st, MPI_LONG, &message_length);

    // if (rank_ == 11) std::cout << "start recv " << messages_recv << " " << recv_tag_ << std::endl;
    std::vector<VertexID> message(static_cast<unsigned long>(message_length));
    MPI_Status rst{};
    MPI_Recv(&message[0], message_length,
             MPI_LONG, st.MPI_SOURCE,
             recv_tag_, communicator_, &rst);
    messages_recv++;
    // if (rank_ == 11) std::cout << "start ml " << message_length << std::endl;
    if (message_length < 4) continue;

    for (int i = 0; i < message_length; i += 4) {
    // if (rank_ == 0) std::cout 
    //                   << message[i] 
    //                   << " "
    //                   << g_->IsGhostFromGlobal(message[i]) 
    //                   << std::endl;
    // if (rank_ == 3 && message[i] == 6) {
    //   g_->OutputLocal();
    //   std::cout << st.MPI_SOURCE << std::endl;
    //   std::cout << g_->IsGhostFromGlobal(message[i]) << std::endl;
    // }
      VertexID local_id = g_->GetLocalID(message[i]);
      VertexID deviate = message[i + 1];
      VertexID label = message[i + 2];
      PEID root = static_cast<PEID>(message[i + 3]);
// #ifndef NDEBUG
// if (rank_ == 11) {
//       std::cout << "[R" << rank_ << "] recv [" << local_id << "]("
//                 << deviate << "," << label
//                 << "," << root << ") from pe "
//                 << st.MPI_SOURCE << " with tag " << recv_tag_
//                 << " length " << message_length << " ["
//                 << messages_recv << "/" << GetNumberOfAdjacentPEs() << "]"
//                 << std::endl;
//     }
// #endif
      g_->HandleGhostUpdate(local_id, label, deviate, root);
    }
    // if (rank_ == 11) std::cout << "done" << std::endl;
  }
  isend_requests_.clear();
}

