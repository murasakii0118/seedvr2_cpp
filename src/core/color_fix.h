// Color correction (utils/color_fix.py)
#pragma once

#include <torch/torch.h>
#include <string>

namespace seedvr2 {

torch::Tensor wavelet_reconstruction(const torch::Tensor& content_, const torch::Tensor& style_);
torch::Tensor adaptive_instance_normalization(const torch::Tensor& content, const torch::Tensor& style);
torch::Tensor lab_color_transfer(const torch::Tensor& content, const torch::Tensor& style,
                                 double luminance_weight = 0.8);
torch::Tensor hsv_saturation_histogram_match(const torch::Tensor& content, const torch::Tensor& style);
torch::Tensor wavelet_adaptive_color_correction(const torch::Tensor& content, const torch::Tensor& style);

// Dispatcher by name (lab|wavelet|wavelet_adaptive|hsv|adain|none)
torch::Tensor apply_color_correction(const std::string& method,
                                     const torch::Tensor& content,
                                     const torch::Tensor& style);

} // namespace seedvr2
