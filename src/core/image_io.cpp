// Image loading / saving (OpenCV) and conversion helpers.
#include "core/image_io.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace seedvr2 {

static cv::Mat bgr_to_rgb_tensor(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static torch::Tensor mat_to_thwc(const cv::Mat& rgb_u8) {
    // HWC uint8 -> float [H,W,3] in [0,1]
    cv::Mat f;
    rgb_u8.convertTo(f, CV_32FC3, 1.0 / 255.0);
    auto t = torch::from_blob(f.data, { f.rows, f.cols, 3 }, torch::kFloat32).clone();
    return t;
}

torch::Tensor decode_image(const std::vector<unsigned char>& bytes) {
    require(!bytes.empty(), "empty image payload");
    cv::Mat raw(1, (int)bytes.size(), CV_8U, (void*)bytes.data());
    cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
    require(!bgr.empty(), "failed to decode image bytes");
    return mat_to_thwc(bgr_to_rgb_tensor(bgr));
}

static cv::Mat thwc_to_bgr_u8(const torch::Tensor& img01) {
    auto t = img01.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    require(t.dim() == 3 && t.size(2) == 3, "expected [H,W,3] tensor");
    int H = (int)t.size(0), W = (int)t.size(1);
    cv::Mat rgb(H, W, CV_32FC3, (void*)t.data_ptr<float>());
    cv::Mat rgb8;
    // round(x*255)
    rgb.convertTo(rgb8, CV_8UC3, 255.0);
    cv::Mat bgr;
    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

std::vector<unsigned char> encode_image(const torch::Tensor& img01, bool png, int quality) {
    cv::Mat bgr = thwc_to_bgr_u8(img01);
    std::vector<unsigned char> out;
    std::vector<int> params;
    std::string ext;
    if (png) {
        ext = ".png";
        params = { cv::IMWRITE_PNG_COMPRESSION, 6 };
    } else {
        ext = ".jpg";
        params = { cv::IMWRITE_JPEG_QUALITY, quality };
    }
    require(cv::imencode(ext, bgr, out, params), "failed to encode image");
    return out;
}

// ---------------------------------------------------------------------------
// Bicubic resize with antialias (matches torchvision's bicubic kernel, which is
// a cubic convolution with a=-0.75 and a support of 2.0, widened by the scale
// factor when downscaling).
// ---------------------------------------------------------------------------
static inline double cubic_a(double x, double a) {
    x = std::fabs(x);
    const double pi_x = 3.14159265358979323846 * x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

static std::vector<double> build_weights(int in_size, int out_size, bool antialias,
                                         std::vector<int>& starts, int& taps) {
    // torchvision "bicubic" a = -0.75, support 2.0
    const double scale = (double)in_size / (double)out_size;
    const bool down = scale > 1.0;
    const double eff = (antialias && down) ? scale : 1.0;
    const double support = 2.0 * eff;
    scale; // silence
    const int in_stride = (int)std::ceil(support);
    taps = in_stride * 2;
    std::vector<double> w((size_t)out_size * taps, 0.0);
    starts.resize(out_size);
    for (int o = 0; o < out_size; ++o) {
        double center = ((double)o + 0.5) * (double)in_size / (double)out_size;
        double s = center - support;
        int start = (int)std::floor(s);
        starts[o] = start;
        double total = 0.0;
        for (int k = 0; k < taps; ++k) {
            int idx = start + k;
            double d = (idx + 0.5 - center) / eff;
            double ww = cubic_a(d, -0.75);
            w[(size_t)o * taps + k] = ww;
            total += ww;
        }
        double inv = (std::fabs(total) > 1e-12) ? 1.0 / total : 0.0;
        for (int k = 0; k < taps; ++k) w[(size_t)o * taps + k] *= inv;
    }
    return w;
}

torch::Tensor resize_bicubic(const torch::Tensor& src_thwc, int dst_w, int dst_h) {
    // src: [T, H, W, C] float
    auto src = src_thwc.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    int T = (int)src.size(0), H = (int)src.size(1), W = (int)src.size(2), C = (int)src.size(3);
    if (H == dst_h && W == dst_w) return src;

    std::vector<int> hs, ws;
    int ht, wt;
    auto wy = build_weights(H, dst_h, true, hs, ht);
    auto wx = build_weights(W, dst_w, true, ws, wt);

    torch::Tensor out = torch::empty({ T, dst_h, dst_w, C }, torch::kFloat32);
    const float* in = src.data_ptr<float>();
    float* o = out.data_ptr<float>();
    std::vector<double> tmp((size_t)T * H * dst_w * C, 0.0);

    // horizontal pass: [T,H,W,C] -> [T,H,dst_w,C]
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < dst_w; ++x) {
                int start = ws[x];
                for (int c = 0; c < C; ++c) {
                    double acc = 0.0;
                    for (int k = 0; k < wt; ++k) {
                        int xi = start + k;
                        if (xi < 0) xi = 0; else if (xi >= W) xi = W - 1;
                        acc += wx[(size_t)x * wt + k] * (double)in[(((size_t)t * H + y) * W + xi) * C + c];
                    }
                    tmp[(((size_t)t * H + y) * dst_w + x) * C + c] = acc;
                }
            }

    // vertical pass: [T,H,dst_w,C] -> [T,dst_h,dst_w,C]
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < dst_h; ++y) {
            int start = hs[y];
            for (int x = 0; x < dst_w; ++x)
                for (int c = 0; c < C; ++c) {
                    double acc = 0.0;
                    for (int k = 0; k < ht; ++k) {
                        int yi = start + k;
                        if (yi < 0) yi = 0; else if (yi >= H) yi = H - 1;
                        acc += wy[(size_t)y * ht + k] * tmp[(((size_t)t * H + yi) * dst_w + x) * C + c];
                    }
                    o[(((size_t)t * dst_h + y) * dst_w + x) * C + c] = (float)acc;
                }
        }
    return out;
}

} // namespace seedvr2
