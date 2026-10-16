// Causal 3D video VAE implemented with ggml (quantised GGUF weights, no libtorch).
#pragma once

#include "core/gguf_model.h"

#include <string>
#include <torch/torch.h>

namespace seedvr2 {

class VaeGgml {
public:
    VaeGgml();
    ~VaeGgml();

    // device: ggml device name ("CUDA0", "Vulkan0", "CPU") or keyword
    void load(const std::string& gguf_path, const std::string& device = "CPU");
    void free();
    bool ready() const { return loaded_; }
    const std::string& info() const;

    // [1,3,1,H,W] in [-1,1] -> [1,16,1,h,w] latent.
    // tiled: spatial tiling (see tiled_encode/tiled_decode), a port of the
    // Python VideoAutoencoderKL.tiled_encode/tiled_decode.  tile_size and
    // tile_overlap are in *output-space* pixels.
    torch::Tensor encode(const torch::Tensor& x, double sample_seed,
                         bool tiled = false, int tile_size = 1024, int tile_overlap = 128);
    // [1,16,1,h,w] latent -> [1,3,1,H,W]
    torch::Tensor decode(const torch::Tensor& z,
                         bool tiled = false, int tile_size = 1024, int tile_overlap = 128);

private:
    struct Impl;
    Impl* impl_;
    bool loaded_ = false;
};

} // namespace seedvr2
