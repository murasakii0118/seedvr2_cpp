// Image loading / saving (OpenCV) and conversion helpers.
#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include "core/common.h"

namespace seedvr2 {

// Decode encoded image bytes (as uploaded from the browser) -> [H, W, 3] in [0,1].
torch::Tensor decode_image(const std::vector<unsigned char>& bytes);

// Save a float32 tensor [H, W, 3] (values in [0,1]) as PNG/JPEG bytes.
std::vector<unsigned char> encode_image(const torch::Tensor& img01, bool png, int quality = 95);

// Bicubic resize (antialiased when downscaling) matching torchvision's
// InterpolationMode.BICUBIC as used by SeedVR2's SideResize.
torch::Tensor resize_bicubic(const torch::Tensor& src_thwc, int dst_w, int dst_h);

} // namespace seedvr2
