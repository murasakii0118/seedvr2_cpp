// Pre-processing transforms (SideResize / DivisiblePad / Normalize), mirroring
// src/data/image/transforms in the Python reference.
#pragma once

#include <torch/torch.h>
#include <tuple>

namespace seedvr2 {

// SideResize: resize so the shortest edge becomes `size` (upscaling allowed),
// optionally capping the longest edge at max_size. Input [T,H,W,C] in [0,1].
torch::Tensor side_resize(const torch::Tensor& thwc, int size, int max_size);

// DivisiblePad: bottom/right zero padding so that H,W are multiples of (fh, fw).
torch::Tensor divisible_pad(const torch::Tensor& thwc, int fh, int fw);

// Full pipeline used by generation_utils.prepare_video_transforms:
//   SideResize -> clamp(0,1) -> DivisiblePad(16,16) -> Normalize(0.5,0.5)
//   -> permute(1,0,2,3)   (t c h w -> c t h w)
// Returns {transformed_cthw, true_h, true_w, padded_h, padded_w}
struct TransformResult {
    torch::Tensor cthw;   // [C,T,H,W] in [-1,1]
    int true_h = 0, true_w = 0, padded_h = 0, padded_w = 0;
};
TransformResult prepare_video(const torch::Tensor& thwc, int resolution, int max_resolution);

} // namespace seedvr2
