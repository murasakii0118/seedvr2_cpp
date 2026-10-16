// NaDiT (Native Resolution Diffusion Transformer) 3B inference on ggml/CUDA.
#pragma once

#include "core/gguf_model.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cstdlib>

namespace seedvr2 {

struct DitConfig {
    int vid_dim       = 2560;
    int txt_in_dim    = 5120;
    int txt_dim       = 2560;
    int emb_dim       = 15360;
    int heads         = 20;
    int head_dim      = 128;
    int expand_ratio  = 4;
    int num_layers    = 32;
    int mm_layers     = 10;      // layers with per-branch weights
    int swiglu_hidden = 6912;
    float norm_eps    = 1e-5f;
    int patch_t = 1, patch_h = 2, patch_w = 2;
    int vid_in_ch  = 33;
    int vid_out_ch = 16;
    int micro_batch = 0;         // 0 = off
};

class Dit {
public:
    // device: a ggml device name ("CUDA0", "Vulkan0", "CPU", ...) or a backend
    // keyword ("cuda", "vulkan", "cpu")
    void load(const std::string& gguf_path, const std::string& device = "CUDA0");
    void free();
    bool loaded() const { return loaded_; }
    const DitConfig& config() const { return cfg_; }
    std::string backend_name() const { return model_.backend_name(); }

    // vid_in : [T, H, W, 33] row-major (H, W = latent dims, divisible by patch)
    // txt    : [Lt, 5120]
    // t      : diffusion timestep
    // returns: [T, H, W, 16]
    torch::Tensor forward(const float* vid_in, int T, int H, int W,
                          const float* txt, int Lt, float t);

    // token count helper
    int64_t tokens(int T, int H, int W) const {
        return (int64_t)T * (H / cfg_.patch_h) * (W / cfg_.patch_w);
    }

private:
    DitConfig cfg_;
    GgufModel model_;
    bool loaded_ = false;

    // time-embedding MLP evaluated on the host (small, avoids a second graph)
    std::vector<float> embW1, embB1, embW2, embB2, embW3, embB3;
    std::vector<float> ropeFreq_;
    int embHid = 0;
    void compute_time_embedding(float t, std::vector<float>& out) const;

    ggml_gallocr_t galloc_ = nullptr;
    bool verbose_ = (std::getenv("SEEDVR2_VERBOSE") != nullptr);
};

} // namespace seedvr2
