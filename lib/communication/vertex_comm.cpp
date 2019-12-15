#include <iostream>

#include "vertex_comm.h"

template<typename GraphType>
void VertexCommunicator<GraphType>::AddMessage(const VertexID v,
                                               const VertexPayload &msg) {
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) {
      PEID neighbor = g_->GetPE(u);
      if (!IsPackedPE(neighbor)) {
        // Unpack msg and add content (Sender, Deviate, Component, PE (of component))
        (*current_send_buffers_)[neighbor].emplace_back(g_->GetGlobalID(v));
        (*current_send_buffers_)[neighbor].emplace_back(msg.deviate_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.label_);
        (*current_send_buffers_)[neighbor].emplace_back(msg.root_);
#ifdef TIEBREAK_DEGREE
        (*current_send_buffers_)[neighbor].emplace_back(msg.degree_);
#endif
        SetPackedPE(neighbor, true);
      }
    }
  });

  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) SetPackedPE(g_->GetPE(u), false);
  });
}

template<typename GraphType>
void VertexCommunicator<GraphType>::ReceiveMessages() {
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
      VertexID global_id = message[i];
      VertexID deviate = message[i + 1];
      VertexID label = message[i + 2];
      PEID root = static_cast<PEID>(message[i + 3]);
#ifdef TIEBREAK_DEGREE
      VertexID degree = message[i + 4];
#endif

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
  }
  for (unsigned int i = 0; i < isend_requests_.size(); ++i) {
    if (isend_requests_[i] != MPI_REQUEST_NULL) {
      MPI_Request_free(&isend_requests_[i]);
    }
  }
  isend_requests_.clear();
}

