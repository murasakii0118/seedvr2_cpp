// Color correction (utils/color_fix.py)
#include "core/color_fix.h"
#include "core/common.h"

#include <torch/torch.h>

namespace seedvr2 {

using torch::Tensor;

// ---------------------------------------------------------------------------
static Tensor wavelet_blur(const Tensor& image_in, int radius) {
    Tensor image = image_in;
    int m = std::min((int)image.size(-2), (int)image.size(-1));
    int max_safe = std::max(1, m / 8);
    if (radius > max_safe) radius = max_safe;

    int C = (int)image.size(1);
    static const float kv[3][3] = {
        { 0.0625f, 0.125f, 0.0625f },
        { 0.125f,  0.25f,  0.125f  },
        { 0.0625f, 0.125f, 0.0625f },
    };
    Tensor kernel = torch::empty({ C, 1, 3, 3 }, image.options());
    {
        auto k = kernel.accessor<float, 4>();
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) k[c][0][i][j] = kv[i][j];
    }
    Tensor padded = torch::nn::functional::pad(
        image, torch::nn::functional::PadFuncOptions({ radius, radius, radius, radius })
                   .mode(torch::kReplicate));
    return torch::nn::functional::conv2d(
        padded, kernel,
        torch::nn::functional::Conv2dFuncOptions()
            .stride({ 1, 1 }).padding({ 0, 0 }).dilation({ radius, radius }).groups(C));
}

static void wavelet_decomposition(const Tensor& image_in, Tensor& high, Tensor& low) {
    Tensor image = image_in;
    high = torch::zeros_like(image);
    low = image;
    for (int i = 0; i < 5; ++i) {
        int radius = 1 << i;
        Tensor lf = wavelet_blur(image, radius);
        high = high.add(image).sub(lf);
        image = lf;
        low = lf;
    }
}

Tensor wavelet_reconstruction(const Tensor& content_in, const Tensor& style_in) {
    Tensor content = content_in, style = style_in;
    if (content.sizes() != style.sizes()) {
        style = torch::nn::functional::interpolate(
            style, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{ content.size(-2), content.size(-1) })
                       .mode(torch::kBilinear).align_corners(false));
    }
    Tensor c_high, c_low, s_high, s_low;
    wavelet_decomposition(content, c_high, c_low);
    wavelet_decomposition(style, s_high, s_low);
    Tensor out = c_high.add(s_low);
    return out.clamp(-1.0, 1.0);
}

Tensor adaptive_instance_normalization(const Tensor& content, const Tensor& style) {
    auto ms = [](const Tensor& f) {
        int64_t b = f.size(0), c = f.size(1);
        Tensor v = f.reshape({ b, c, -1 });
        Tensor var = v.var(2, /*unbiased=*/false, /*keepdim=*/true) + 1e-5;
        Tensor std = var.sqrt().reshape({ b, c, 1, 1 });
        Tensor mean = v.mean(2, true).reshape({ b, c, 1, 1 });
        return std::make_pair(mean, std);
    };
    auto [s_mean, s_std] = ms(style);
    auto [c_mean, c_std] = ms(content);
    Tensor norm = (content - c_mean) / c_std;
    return norm * s_std + s_mean;
}

// ---------------------------------------------------------------------------
// LAB
// ---------------------------------------------------------------------------
static Tensor rgb_to_lab(const Tensor& rgb, const Tensor& mat, double eps, double kappa) {
    Tensor mask = rgb > 0.04045;
    Tensor lin = torch::where(mask, torch::pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92);
    int64_t B = lin.size(0), H = lin.size(2), W = lin.size(3);
    Tensor flat = lin.permute({ 0, 2, 3, 1 }).reshape({ -1, 3 });
    Tensor xyz = torch::matmul(flat, mat.transpose(0, 1)).reshape({ B, H, W, 3 }).permute({ 0, 3, 1, 2 });
    xyz.select(1, 0).div_(0.95047);
    xyz.select(1, 2).div_(1.08883);
    double ec = eps * eps * eps;
    Tensor m = xyz > ec;
    Tensor felse = (xyz * kappa + 16.0) / 116.0;
    Tensor f = torch::where(m, torch::pow(xyz, 1.0 / 3.0), felse);
    Tensor L = f.select(1, 1) * 116.0 - 16.0;
    Tensor a = (f.select(1, 0) - f.select(1, 1)) * 500.0;
    Tensor b = (f.select(1, 1) - f.select(1, 2)) * 200.0;
    return torch::stack({ L, a, b }, 1);
}

static Tensor lab_to_rgb(const Tensor& lab, const Tensor& mat_inv, double eps, double kappa) {
    Tensor L = lab.select(1, 0), a = lab.select(1, 1), b = lab.select(1, 2);
    Tensor fy = (L + 16.0) / 116.0;
    Tensor fx = a / 500.0 + fy;
    Tensor fz = fy - b / 200.0;
    auto inv = [&](const Tensor& t) {
        Tensor t1 = torch::pow(t, 3.0);
        Tensor t2 = (t * 116.0 - 16.0) / kappa;
        return torch::where(t > eps, t1, t2);
    };
    Tensor x = inv(fx), y = inv(fy), z = inv(fz);
    x = x * 0.95047;
    z = z * 1.08883;
    Tensor xyz = torch::stack({ x, y, z }, 1);
    int64_t B = xyz.size(0), H = xyz.size(2), W = xyz.size(3);
    Tensor flat = xyz.permute({ 0, 2, 3, 1 }).reshape({ -1, 3 });
    Tensor lin = torch::matmul(flat, mat_inv.transpose(0, 1)).reshape({ B, H, W, 3 }).permute({ 0, 3, 1, 2 });
    Tensor m = lin > 0.0031308;
    Tensor rgb = torch::where(m, torch::pow(torch::clamp_min(lin, 0.0), 1.0 / 2.4) * 1.055 - 0.055, lin * 12.92);
    return torch::clamp(rgb, 0.0, 1.0);
}

static Tensor histogram_matching_channel(const Tensor& source, const Tensor& reference) {
    auto orig = source.sizes();
    Tensor sf = source.flatten();
    Tensor rf = reference.flatten();
    auto s = sf.sort();
    Tensor s_sorted = std::get<0>(s);
    Tensor s_idx = std::get<1>(s);
    Tensor r_sorted = std::get<0>(rf.sort());
    int64_t ns = s_sorted.size(0), nr = r_sorted.size(0);
    Tensor matched;
    if (ns == nr) {
        matched = r_sorted;
    } else {
        Tensor q = torch::linspace(0, 1, ns, s_sorted.options());
        Tensor ri = (q * (double)(nr - 1)).to(torch::kLong).clamp(0, nr - 1);
        matched = r_sorted.index_select(0, ri);
    }
    Tensor inv = s_idx.argsort();
    return matched.index_select(0, inv).reshape(orig);
}

Tensor lab_color_transfer(const Tensor& content_in, const Tensor& style_in, double luminance_weight) {
    Tensor content = wavelet_reconstruction(content_in, style_in);
    Tensor style = style_in;
    if (content.sizes() != style.sizes()) {
        style = torch::nn::functional::interpolate(
            style, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{ content.size(-2), content.size(-1) })
                       .mode(torch::kBilinear).align_corners(false));
    }
    content = content.to(torch::kFloat32);
    style = style.to(torch::kFloat32);
    auto dev = content.device();

    Tensor rgb2xyz = torch::tensor({ { 0.4124564, 0.3575761, 0.1804375 },
                                     { 0.2126729, 0.7151522, 0.0721750 },
                                     { 0.0193339, 0.1191920, 0.9503041 } },
                                   content.options());
    Tensor xyz2rgb = torch::tensor({ { 3.2404542, -1.5371385, -0.4985314 },
                                     { -0.9692660, 1.8760108, 0.0415560 },
                                     { 0.0556434, -0.2040259, 1.0572252 } },
                                   content.options());
    double eps = 6.0 / 29.0;
    double kappa = std::pow(29.0 / 3.0, 3.0);

    content = (content + 1.0) * 0.5;
    content = content.clamp(0.0, 1.0);
    style = (style + 1.0) * 0.5;
    style = style.clamp(0.0, 1.0);

    Tensor c_lab = rgb_to_lab(content, rgb2xyz, eps, kappa);
    Tensor s_lab = rgb_to_lab(style, rgb2xyz, eps, kappa);

    Tensor ma = histogram_matching_channel(c_lab.select(1, 1), s_lab.select(1, 1));
    Tensor mb = histogram_matching_channel(c_lab.select(1, 2), s_lab.select(1, 2));
    Tensor result_L;
    if (luminance_weight < 1.0) {
        Tensor mL = histogram_matching_channel(c_lab.select(1, 0), s_lab.select(1, 0));
        result_L = c_lab.select(1, 0) * luminance_weight + mL * (1.0 - luminance_weight);
    } else {
        result_L = c_lab.select(1, 0);
    }
    Tensor out_lab = torch::stack({ result_L, ma, mb }, 1);
    Tensor rgb = lab_to_rgb(out_lab, xyz2rgb, eps, kappa);
    return rgb * 2.0 - 1.0;
}

// ---------------------------------------------------------------------------
// HSV
// ---------------------------------------------------------------------------
static Tensor rgb_to_hsv_batch(const Tensor& rgb) {
    Tensor r = rgb.select(1, 0), g = rgb.select(1, 1), b = rgb.select(1, 2);
    auto mx = std::get<0>(rgb.max(1));
    auto mn = std::get<0>(rgb.min(1));
    Tensor rangec = mx - mn;
    Tensor rnz = torch::where(rangec > 1e-10, rangec, torch::ones_like(rangec));
    Tensor h = torch::zeros_like(mx);
    Tensor m_r = (mx == r) & (rangec > 1e-10);
    h = torch::where(m_r, torch::remainder((g - b) / rnz, 6.0), h);
    Tensor m_g = (mx == g) & (rangec > 1e-10);
    h = torch::where(m_g, (b - r) / rnz + 2.0, h);
    Tensor m_b = (mx == b) & (rangec > 1e-10);
    h = torch::where(m_b, (r - g) / rnz + 4.0, h);
    h = h / 6.0;
    Tensor s = torch::where(mx > 1e-10, rangec / torch::clamp_min(mx, 1e-10), torch::zeros_like(mx));
    Tensor v = mx;
    return torch::stack({ h, s, v }, 1);
}

static Tensor hsv_to_rgb_batch(const Tensor& hsv) {
    Tensor h = hsv.select(1, 0) * 6.0;
    Tensor s = hsv.select(1, 1);
    Tensor v = hsv.select(1, 2);
    Tensor fl = torch::floor(h);
    Tensor i = torch::remainder(fl, 6.0).to(torch::kLong);
    Tensor f = h - fl;
    Tensor p = v * (1.0 - s);
    Tensor q = v * (1.0 - s * f);
    Tensor t = v * (1.0 - s * (1.0 - f));
    Tensor r = torch::zeros_like(v), g = torch::zeros_like(v), bl = torch::zeros_like(v);
    for (int k = 0; k < 6; ++k) {
        Tensor m = (i == k);
        Tensor cr, cg, cb;
        if (k == 0) { cr = v; cg = t; cb = p; }
        else if (k == 1) { cr = q; cg = v; cb = p; }
        else if (k == 2) { cr = p; cg = v; cb = t; }
        else if (k == 3) { cr = p; cg = q; cb = v; }
        else if (k == 4) { cr = t; cg = p; cb = v; }
        else { cr = v; cg = p; cb = q; }
        r = torch::where(m, cr, r);
        g = torch::where(m, cg, g);
        bl = torch::where(m, cb, bl);
    }
    return torch::stack({ r, g, bl }, 1);
}

static Tensor histogram_match_1d(const Tensor& source, const Tensor& reference) {
    auto s = source.sort();
    Tensor s_sorted = std::get<0>(s), s_idx = std::get<1>(s);
    Tensor r_sorted = std::get<0>(reference.sort());
    int64_t ns = s_sorted.size(0), nr = r_sorted.size(0);
    Tensor matched;
    if (ns == nr) matched = r_sorted;
    else {
        Tensor q = torch::linspace(0, 1, ns, s_sorted.options());
        Tensor ri = (q * (double)(nr - 1)).to(torch::kLong).clamp(0, nr - 1);
        matched = r_sorted.index_select(0, ri);
    }
    return matched.index_select(0, s_idx.argsort());
}

static Tensor hue_conditional_saturation_match(const Tensor& c_h, const Tensor& c_s,
                                               const Tensor& s_h, const Tensor& s_s) {
    const int num_bins = 12;
    const double bin_width = 1.0 / num_bins;
    const int min_pixels = 100;
    Tensor matched = c_s.clone();
    Tensor ch_f = c_h.flatten(), cs_f = c_s.flatten(), sh_f = s_h.flatten(), ss_f = s_s.flatten();
    Tensor mf = matched.flatten();
    for (int b = 0; b < num_bins; ++b) {
        double bs = b * bin_width, be = (b + 1) * bin_width;
        Tensor cm, sm;
        if (b == 0) {
            cm = ((ch_f >= 0) & (ch_f < be)) | (ch_f >= (1.0 - bin_width));
            sm = ((sh_f >= 0) & (sh_f < be)) | (sh_f >= (1.0 - bin_width));
        } else {
            cm = (ch_f >= bs) & (ch_f < be);
            sm = (sh_f >= bs) & (sh_f < be);
        }
        Tensor cb = cs_f.masked_select(cm);
        Tensor sb = ss_f.masked_select(sm);
        if (cb.size(0) > min_pixels && sb.size(0) > min_pixels) {
            Tensor mb = histogram_match_1d(cb, sb);
            mf.masked_scatter_(cm, mb);
        }
    }
    return mf.reshape(c_s.sizes());
}

Tensor hsv_saturation_histogram_match(const Tensor& content_in, const Tensor& style_in) {
    Tensor content = content_in, style = style_in;
    if (content.sizes() != style.sizes()) {
        style = torch::nn::functional::interpolate(
            style, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{ content.size(-2), content.size(-1) })
                       .mode(torch::kBilinear).align_corners(false));
    }
    content = content.to(torch::kFloat32);
    style = style.to(torch::kFloat32);
    Tensor c_rgb = (content + 1.0) * 0.5;
    c_rgb = c_rgb.clamp(0.0, 1.0);
    Tensor s_rgb = (style + 1.0) * 0.5;
    s_rgb = s_rgb.clamp(0.0, 1.0);
    Tensor c_hsv = rgb_to_hsv_batch(c_rgb);
    Tensor s_hsv = rgb_to_hsv_batch(s_rgb);
    Tensor c_h = c_hsv.select(1, 0), c_s = c_hsv.select(1, 1), c_v = c_hsv.select(1, 2);
    Tensor s_h = s_hsv.select(1, 0), s_s = s_hsv.select(1, 1);
    Tensor ms = hue_conditional_saturation_match(c_h, c_s, s_h, s_s);
    Tensor out_hsv = torch::stack({ c_h, ms, c_v }, 1);
    Tensor rgb = hsv_to_rgb_batch(out_hsv).clamp(0.0, 1.0);
    return rgb * 2.0 - 1.0;
}

static Tensor saturation_map(const Tensor& t_in) {
    Tensor rgb = (t_in + 1.0) * 0.5;
    rgb = rgb.clamp(0.0, 1.0);
    auto mx = std::get<0>(rgb.max(1, true));
    auto mn = std::get<0>(rgb.min(1, true));
    return torch::where(mx > 1e-10, (mx - mn) / torch::clamp_min(mx, 1e-10), torch::zeros_like(mx));
}

Tensor wavelet_adaptive_color_correction(const Tensor& content_in, const Tensor& style_in) {
    Tensor content = content_in, style = style_in;
    if (content.sizes() != style.sizes()) {
        style = torch::nn::functional::interpolate(
            style, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{ content.size(-2), content.size(-1) })
                       .mode(torch::kBilinear).align_corners(false));
    }
    content = content.to(torch::kFloat32);
    style = style.to(torch::kFloat32);
    Tensor wavelet_result = wavelet_reconstruction(content, style);
    Tensor hsv_result = hsv_saturation_histogram_match(content, style);
    Tensor content_sat = saturation_map(content);
    Tensor style_sat = saturation_map(style);
    Tensor wavelet_sat = saturation_map(wavelet_result);
    Tensor sat_diff = content_sat - style_sat;
    double thr = 0.15, sharp = 5.0;
    Tensor w = torch::sigmoid(sharp * (sat_diff - thr));
    Tensor wave_over = ((wavelet_sat - style_sat) > (thr * 0.5)).to(torch::kFloat32);
    w = w * wave_over;
    w = w.clamp(0.0, 1.0);
    return wavelet_result * (1.0 - w) + hsv_result * w;
}

torch::Tensor apply_color_correction(const std::string& method, const Tensor& content, const Tensor& style) {
    if (method == "none" || method.empty()) return content;
    if (method == "lab") return lab_color_transfer(content, style, 0.8);
    if (method == "wavelet") return wavelet_reconstruction(content, style);
    if (method == "wavelet_adaptive") return wavelet_adaptive_color_correction(content, style);
    if (method == "hsv") return hsv_saturation_histogram_match(content, style);
    if (method == "adain") return adaptive_instance_normalization(content, style);
    SEEDVR_LOG(LogLevel::Warn, "unknown color_correction '" + method + "', falling back to wavelet");
    return wavelet_reconstruction(content, style);
}

} // namespace seedvr2
