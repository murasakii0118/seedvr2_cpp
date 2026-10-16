// Pre-computed positive / negative text embeddings (pos_emb.pt / neg_emb.pt)
#pragma once

#include <torch/torch.h>
#include <string>

namespace seedvr2 {

struct TextEmbeddings {
    torch::Tensor pos;   // [L, 5120]
    torch::Tensor neg;   // [L, 5120]
};

// The embeddings are compiled into the binary (core/text_emb_data.h); the device
// argument selects where the resulting tensors live.
TextEmbeddings load_text_embeddings(const std::string& device = "cpu");

} // namespace seedvr2
