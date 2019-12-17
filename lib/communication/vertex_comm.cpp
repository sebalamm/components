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
        send_buffers_[neighbor].emplace_back(g_->GetGlobalID(v));
        send_buffers_[neighbor].emplace_back(msg.deviate_);
        send_buffers_[neighbor].emplace_back(msg.label_);
        send_buffers_[neighbor].emplace_back(msg.root_);
#ifdef TIEBREAK_DEGREE
        send_buffers_[neighbor].emplace_back(msg.degree_);
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
void VertexCommunicator<GraphType>::UpdateGhostVertices() {
  for (const auto &kv : receive_buffers_) {
    const auto &buffer = kv.second;
#ifdef TIEBREAK_DEGREE
    for (VertexID i = 0; i < buffer.size(); i += 5) {
#else 
    for (VertexID i = 0; i < buffer.size(); i += 4) {
#endif
      VertexID global_id = buffer[i];
      VertexID deviate = buffer[i + 1];
      VertexID label = buffer[i + 2];
      PEID root = static_cast<PEID>(buffer[i + 3]);
#ifdef TIEBREAK_DEGREE
      VertexID degree = buffer[i + 4];
#endif

      g_->HandleGhostUpdate(g_->GetLocalID(global_id), 
                            label, 
                            deviate, 
#ifdef TIEBREAK_DEGREE
                            degree,
#endif
                            root);
    }
  }
}

