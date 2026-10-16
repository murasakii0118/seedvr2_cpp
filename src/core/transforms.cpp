#include "core/transforms.h"
#include "core/image_io.h"

namespace seedvr2 {

torch::Tensor side_resize(const torch::Tensor& thwc, int size, int max_size) {
    int H = (int)thwc.size(1), W = (int)thwc.size(2);
    int short_edge = std::min(H, W);
    int target = size;
    // downsample_only == False for our pipeline -> always scale to `size`
    if (target == short_edge) {
        // still allow max_size handling below
    }
    // torchvision TVF.resize(image, size, BICUBIC): `size` is the *shortest*
    // edge; the other edge is scaled proportionally and rounded.
    double ratio = (double)target / (double)short_edge;
    int new_h = round_half_away(H * ratio);
    int new_w = round_half_away(W * ratio);
    if (new_h < 1) new_h = 1;
    if (new_w < 1) new_w = 1;

    torch::Tensor out = resize_bicubic(thwc, new_w, new_h);

    if (max_size > 0) {
        int mh = (int)out.size(1), mw = (int)out.size(2);
        if (std::max(mh, mw) > max_size) {
            double s = (double)max_size / (double)std::max(mh, mw);
            int nh = round_half_away(mh * s);
            int nw = round_half_away(mw * s);
            out = resize_bicubic(out, std::max(1, nw), std::max(1, nh));
        }
    }
    return out;
}

torch::Tensor divisible_pad(const torch::Tensor& thwc, int fh, int fw) {
    int H = (int)thwc.size(1), W = (int)thwc.size(2);
    int ph = (fh - (H % fh)) % fh;
    int pw = (fw - (W % fw)) % fw;
    if (ph == 0 && pw == 0) return thwc;
    // F.pad format (left, right, top, bottom) on [T,H,W,C] => pad last dims
    auto out = torch::constant_pad_nd(
        thwc.permute({ 0, 3, 1, 2 }).contiguous(), { 0, pw, 0, ph }, 0.0);
    return out.permute({ 0, 2, 3, 1 }).contiguous();
}

TransformResult prepare_video(const torch::Tensor& thwc_in, int resolution, int max_resolution) {
    TransformResult r;
    auto x = side_resize(thwc_in, resolution, max_resolution);
    r.true_h = (int)x.size(1);
    r.true_w = (int)x.size(2);
    x = torch::clamp(x, 0.0, 1.0);
    x = divisible_pad(x, 16, 16);
    r.padded_h = (int)x.size(1);
    r.padded_w = (int)x.size(2);
    x = (x - 0.5) / 0.5;                       // Normalize(0.5, 0.5)
    r.cthw = x.permute({ 3, 0, 1, 2 }).contiguous();  // t c h w -> c t h w
    return r;
}

} // namespace seedvr2
