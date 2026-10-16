// End-to-end upscaling pipeline.
#pragma once

#include "core/common.h"
#include "core/dit.h"
#include "core/vae_ggml.h"
#include "core/text_emb.h"

#include <torch/torch.h>
#include <string>
#include <vector>

namespace seedvr2 {

struct EngineConfig {
    std::string gguf_path;             // DiT
    std::string vae_path;              // quantised VAE .gguf (ggml)
    // compute backends: ggml device names ("CUDA0", "Vulkan0", "CPU", ...)
    // or the keywords "cuda"/"vulkan"/"cpu"
    std::string dit_device = "CUDA0";
    // CUDA is ~25x faster than the CPU here (256: 0.4s vs 12s encode); the
    // loader falls back to the CPU when the device is unavailable.
    std::string vae_device = "CUDA0";
    // Keep only one model resident on the device at a time: VAE(encode) -> DiT ->
    // VAE(decode).  Weights are re-uploaded per stage (cheap over PCIe) and the
    // VRAM of the other model is released - matters a lot on 8 GB laptop GPUs.
    bool staged_load = true;
};

// Owns the models; thread-safe for one job at a time.
class Engine {
public:
    void init(const EngineConfig& cfg, Progress& prog);
    bool ready() const { return ready_; }
    const std::string& info() const { return info_; }
    // currently active compute backends (for the UI)
    const std::string& dit_device() const { return dit_dev_; }
    const std::string& vae_device() const { return vae_dev_; }

    // Switch backends and reload the models.  Takes the job lock, so it must not
    // be called while an upscale is running.
    void reinit(const std::string& dit_device, const std::string& vae_device, Progress& prog);

    // Upscale one image (encoded bytes) -> PNG bytes
    std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                       const UpscaleParams& params, Progress& prog);

    // Diagnostic: VAE encode -> decode only (no DiT)
    std::vector<unsigned char> vae_roundtrip(const std::vector<unsigned char>& input,
                                             const UpscaleParams& params, Progress& prog);

    // "latent cache" batch: all VAE encodes, then all DiT passes, then all VAE
    // decodes - each model is loaded onto the device only once per batch instead
    // of once per image (avoids the repeated staged load/unload).
    std::vector<std::vector<unsigned char>> upscale_batch(
        const std::vector<std::vector<unsigned char>>& inputs,
        const std::vector<UpscaleParams>& params, Progress& prog);

private:
    // per-image state carried across the three stages
    struct Stage;
    void stage_encode(const std::vector<unsigned char>& input, const UpscaleParams& params,
                      Progress& prog, Stage& st);
    void stage_dit(const UpscaleParams& params, Progress& prog, Stage& st);
    std::vector<unsigned char> stage_decode(const UpscaleParams& params, Progress& prog, Stage& st);
    bool ready_ = false;
    std::string info_;
    EngineConfig cfg_;
    std::string dit_dev_ = "CUDA0", vae_dev_ = "CUDA0";
    Dit dit_;
    VaeGgml vae_g_;
    TextEmbeddings text_;
    std::mutex mu_;

    void init_locked(const EngineConfig& cfg, Progress& prog);

    // staged residency helpers (no-ops when staged_load is off)
    void ensure_vae();
    void ensure_dit();
    void release_models();

    // VAE dispatch (ggml only); the per-request tiling flags are honoured.
    torch::Tensor vae_encode(const torch::Tensor& x, double seed, const UpscaleParams& p) {
        return vae_g_.encode(x, seed, p.vae_encode_tiled, p.vae_tile_size, p.vae_tile_overlap);
    }
    torch::Tensor vae_decode(const torch::Tensor& z, const UpscaleParams& p) {
        return vae_g_.decode(z, p.vae_decode_tiled, p.vae_tile_size, p.vae_tile_overlap);
    }
};

} // namespace seedvr2
