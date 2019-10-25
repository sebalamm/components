#include <iostream>

#include "vertex_communicator.h"

void VertexCommunicator::AddMessage(const VertexID v,
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
#ifdef TIEBREAK_DEGREE
        (*current_send_buffers_)[neighbor].emplace_back(msg.degree_);
#endif
        packed_pes_[neighbor] = true;
      }
    }
  });

  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) packed_pes_[g_->GetPE(u)] = false;
  });
}

void VertexCommunicator::ReceiveMessages() {
  PEID messages_recv = 0;
  recv_tag_++;
  while (messages_recv < GetNumberOfAdjacentPEs()) {
    MPI_Status st{};
    MPI_Probe(MPI_ANY_SOURCE, recv_tag_, communicator_, &st);

    int message_length;
    MPI_Get_count(&st, MPI_VERTEX, &message_length);

    std::vector<VertexID> message(static_cast<unsigned long>(message_length));
    MPI_Status rst{};
    MPI_Recv(&message[0], message_length,
             MPI_VERTEX, st.MPI_SOURCE,
             recv_tag_, communicator_, &rst);
    messages_recv++;

#ifdef TIEBREAK_DEGREE
    if (message_length < 5) continue;
    for (int i = 0; i < message_length; i += 5) {
#else 
    if (message_length < 4) continue;
    for (int i = 0; i < message_length; i += 4) {
#endif
    // if (rank_ == 0) std::cout 
    //                   << message[i] 
    //                   << " "
    //                   << g_->IsGhostFromGlobal(message[i]) 
    //                   << " "
    //                   << st.MPI_SOURCE 
    //                   << std::endl;
      VertexID global_id = message[i];
      VertexID deviate = message[i + 1];
      VertexID label = message[i + 2];
      PEID root = static_cast<PEID>(message[i + 3]);
#ifdef TIEBREAK_DEGREE
      VertexID degree = message[i + 4];
#endif
// #ifndef NDEBUG
// if (rank_ == 0) {
//       std::cout << "[R" << rank_ << "] recv [" << local_id << "]("
//                 << deviate << "," << label
//                 << "," << root << ") from pe "
//                 << st.MPI_SOURCE << " with tag " << recv_tag_
//                 << " length " << message_length << " ["
//                 << messages_recv << "/" << GetNumberOfAdjacentPEs() << "]"
//                 << std::endl;
//     }
// #endif

      if (global_id == std::numeric_limits<VertexID>::max()) continue;
      VertexID local_id = g_->GetLocalID(global_id);

      g_->HandleGhostUpdate(local_id, 
                            label, 
                            deviate, 
#ifdef TIEBREAK_DEGREE
                            degree,
#endif
                            root);
    }
    // if (rank_ == 11) std::cout << "done" << std::endl;
  }
  for (unsigned int i = 0; i < isend_requests_.size(); ++i) {
    MPI_Status st{};
    MPI_Wait(isend_requests_[i], &st);
  }
  isend_requests_.clear();
}

