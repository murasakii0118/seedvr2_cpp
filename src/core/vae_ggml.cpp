// Causal 3D video VAE (SeedVR2 / s8_c16_t4) implemented with ggml.
//
// Weights come from a quantised GGUF produced by vae_convert.cpp:
//   * conv3d kernels are stored as {Cin, Cout, K} (K = kT*kH*kW, tap order
//     (kt*kH + kh)*kW + kw), i.e. one {Cin, Cout} matmul matrix per tap, which
//     keeps Q8_0 blocks inside the contraction dim.
//   * activations are {C, W, H} tensors: ne0 = channel (what ggml_mul_mat
//     contracts) and ne1 = the row (the image WIDTH) so the flat spatial index
//     enumerates (h, w) with w fastest - the same order as an im2col row.
//   * ggml_mul_mat never broadcasts the batch dims, so a 3D convolution is
//     kT*kH*kW matmuls against shifted 2D views of the zero padded input.  The
//     output keeps the padded row length and is returned as a strided view of
//     the valid region (crop), so no extra copies are needed.
//   * GroupNorm = two tiny "sum per group" matmuls against 0/1 indicator
//     matrices, then a per-channel affine.
//   * the decoder's pixel-shuffle upsample writes its sub-planes into strided
//     views of one buffer with GGML_OP_CPY.
// Only T == 1 (single image) is supported, which is what the pipeline feeds.
#include "core/vae_ggml.h"
#include "core/common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "ggml.h"


namespace seedvr2 {

namespace {

struct VaeCtx {
    ggml_context* ctx = nullptr;
    GgufModel* m = nullptr;
    ggml_cgraph* graph_ = nullptr;
    std::vector<std::pair<ggml_tensor*, std::vector<char>>> pending;

    ggml_tensor* W(const std::string& n) { return m->get(n); }
    ggml_tensor* f32(ggml_tensor* t) {
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    }
    ggml_tensor* silu(ggml_tensor* x) { return ggml_silu(ctx, x); }
    void input(ggml_tensor* t, const void* data, size_t bytes) {
        pending.emplace_back(t, std::vector<char>((const char*)data, (const char*)data + bytes));
    }
    bool has(const std::string& n) const { return m->has(n); }

    // ---- weight shape helpers (conv kernels are {Cin, Cout, K}) ----
    int64_t cin(const std::string& c) { return W(c + ".weight")->ne[0]; }
    int64_t cout(const std::string& c) { return W(c + ".weight")->ne[1]; }
    void kdims(const std::string& c, int& kT, int& kH, int& kW) {
        const int64_t k = W(c + ".weight")->ne[2];
        if (k == 27) { kT = 3; kH = 3; kW = 3; }
        else if (k == 9) { kT = 1; kH = 3; kW = 3; }
        else if (k == 1) { kT = 1; kH = 1; kW = 1; }
        else throw Error("unexpected kernel size " + std::to_string(k) + " for " + c);
    }

    // ---- convolution core -------------------------------------------------
    // x: {Cin, W, H} (the *physical* frame).  extraR/extraB are the extra
    // right/bottom padding used by the 2D downsampler (which is what makes its
    // logical input one pixel larger).  Returns a strided view {Cout, Wo, Ho} of
    // the valid output region - no copies.
    ggml_tensor* conv_core(const std::string& conv, ggml_tensor* x, int64_t Hi, int64_t Wi,
                           int padH, int padW, int extraB, int extraR, int sH, int sW) {
        int kT, kH, kW;
        kdims(conv, kT, kH, kW);
        const int64_t Cin = cin(conv), Cout = cout(conv);
        ggml_tensor* base = ggml_pad_ext(ctx, x, 0, 0,
                                         (int)padW, (int)(padW + extraR + kW),
                                         (int)padH, (int)(padH + extraB + kH), 0, 0);
        const int64_t Wb = Wi + 2 * padW + extraR + kW;      // padded row length
        const int64_t Hb = Hi + 2 * padH + extraB + kH;
        const int64_t Hp = Hi + 2 * padH + extraB;           // padded output rows
        const int64_t Ho = (Hi + extraB + 2 * padH - kH) / sH + 1;
        const int64_t Wo = (Wi + extraR + 2 * padW - kW) / sW + 1;
        ggml_tensor* base2 = ggml_reshape_2d(ctx, base, Cin, Wb * Hb);
        ggml_tensor* y = nullptr;
        for (int kh = 0; kh < kH; ++kh)
            for (int kw = 0; kw < kW; ++kw) {
                const size_t shift = (size_t)kh * Wb + kw;               // in spatial units
                const size_t nb_flat = (size_t)Cin * sizeof(float);        // one flat spatial step
                ggml_tensor* xv = ggml_view_2d(ctx, base2, Cin, Wb * Hp, nb_flat, shift * nb_flat);
                for (int kt = 0; kt < kT; ++kt) {
                    // all temporal taps read frame 0 (T == 1)
                    ggml_tensor* w = W(conv + ".weight");                // {Cin, Cout, K}
                    ggml_tensor* wt = ggml_view_2d(ctx, w, Cin, Cout, w->nb[1],
                                                   (size_t)((kt * kH + kh) * kW + kw) * w->nb[2]);
                    ggml_tensor* t = ggml_mul_mat(ctx, wt, xv);          // {Cout, Wb*Hp}
                    y = y ? ggml_add(ctx, y, t) : t;
                }
            }
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, f32(W(conv + ".bias")), Cout, 1));
        // The padded base carries exactly the reference's left/top padding
        // (padH/padW is the same-conv padding, 0 for the downsampler), so the
        // first valid output sample is the flat index 0.  Cropping at
        // padH*Wb+padW shifted every 3x3 "same" conv by one pixel against the
        // reference (verified layer by layer in Python).
        const size_t off = 0;
        // Return a *contiguous* cropped result: the crop view is strided (rows are
        // Wb wide) and some backends (Vulkan) mishandle strided inputs for a few
        // ops (pad/clamp).  Downstream ops then always see contiguous tensors.
        return ggml_cont(ctx, ggml_view_3d(ctx, y, Cout, Wo, Ho,
                            (size_t)sW * Cout * sizeof(float),
                            (size_t)sH * Wb * Cout * sizeof(float), off));
    }

    // plain 3x3/1x1 "same" convolution
    ggml_tensor* conv3d(const std::string& conv, ggml_tensor* x, int64_t Hi, int64_t Wi,
                        int padH, int padW) {
        return conv_core(conv, x, Hi, Wi, padH, padW, 0, 0, 1, 1);
    }
    // 2D downsampler: pad the right/bottom by one, then a 3x3 stride-2 conv
    ggml_tensor* conv_down(const std::string& conv, ggml_tensor* x, int64_t Hi, int64_t Wi) {
        return conv_core(conv, x, Hi, Wi, 0, 0, 1, 1, 2, 2);
    }
    ggml_tensor* conv1x1(const std::string& conv, ggml_tensor* x, int64_t Hi, int64_t Wi) {
        return conv_core(conv, x, Hi, Wi, 0, 0, 0, 0, 1, 1);
    }

    // ---- GroupNorm via 0/1 indicator matmuls ------------------------------
    // x: {C, W, H} (may be a strided view) -> {C, W, H}
    ggml_tensor* group_norm(const std::string& norm, ggml_tensor* x, int64_t C, int64_t Wi, int64_t Hi,
                            int64_t groups, float eps) {
        const int64_t HW = Wi * Hi;
        const int64_t gc = C / groups;
        ggml_tensor* xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
        ggml_tensor* x2 = ggml_reshape_2d(ctx, xc, C, HW);
        ggml_tensor* ind = indicator(C, groups);                       // {C, groups}
        ggml_tensor* s1 = ggml_mul_mat(ctx, ind, x2);                  // {groups, HW}
        ggml_tensor* s2 = ggml_mul_mat(ctx, ind, ggml_sqr(ctx, x2));
        ggml_tensor* t1 = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, s1, 1, 0, 2, 3)));
        ggml_tensor* t2 = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, s2, 1, 0, 2, 3)));
        const float n = (float)(gc * HW);
        ggml_tensor* mu = ggml_scale(ctx, t1, 1.0f / n);               // {1, groups}
        ggml_tensor* e2 = ggml_scale(ctx, t2, 1.0f / n);
        ggml_tensor* var = ggml_sub(ctx, e2, ggml_mul(ctx, mu, mu));
        ggml_tensor* sd = ggml_sqrt(ctx, ggml_add(ctx, var, const_1xG(groups, eps)));
        ggml_tensor* inv = ggml_div(ctx, const_1xG(groups, 1.0f), sd);
        ggml_tensor* ind2 = indicator_t(groups, C);                    // {groups, C}
        ggml_tensor* mur = ggml_mul_mat(ctx, ind2, ggml_cont(ctx, ggml_permute(ctx, mu, 1, 0, 2, 3)));   // {C,1}
        ggml_tensor* invr = ggml_mul_mat(ctx, ind2, ggml_cont(ctx, ggml_permute(ctx, inv, 1, 0, 2, 3)));
        ggml_tensor* y = ggml_mul(ctx, ggml_sub(ctx, x2, mur), invr);
        ggml_tensor* gw = ggml_reshape_2d(ctx, f32(W(norm + ".weight")), C, 1);
        ggml_tensor* gb = ggml_reshape_2d(ctx, f32(W(norm + ".bias")), C, 1);
        y = ggml_add(ctx, ggml_mul(ctx, y, gw), gb);
        return ggml_reshape_3d(ctx, y, C, Wi, Hi);
    }

    // ---- host side constant tensors ----
    ggml_tensor* indicator(int64_t C, int64_t G) {
        if (!ind_cg_ || ind_C_ != C || ind_G_ != G) {
            ind_C_ = C; ind_G_ = G;
            std::vector<float> v((size_t)C * G, 0.f);
            for (int64_t g = 0; g < G; ++g)
                for (int64_t c = g * (C / G); c < (g + 1) * (C / G); ++c) v[(size_t)g * C + c] = 1.f;
            ind_cg_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, G);
            input(ind_cg_, v.data(), v.size() * 4);
            std::fill(v.begin(), v.end(), 0.f);
            for (int64_t g = 0; g < G; ++g)
                for (int64_t c = g * (C / G); c < (g + 1) * (C / G); ++c) v[(size_t)c * G + g] = 1.f;
            ind_gc_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, G, C);
            input(ind_gc_, v.data(), v.size() * 4);
        }
        return ind_cg_;
    }
    ggml_tensor* indicator_t(int64_t G, int64_t C) { indicator(C, G); return ind_gc_; }
    ggml_tensor* const_1xG(int64_t G, float val) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, G, 1, 1);
        std::vector<float> v((size_t)G, val);
        input(t, v.data(), v.size() * 4);
        return t;
    }
    ggml_tensor* ind_cg_ = nullptr, *ind_gc_ = nullptr;
    int64_t ind_C_ = -1, ind_G_ = -1;

    // ---- linear op (attention projections) ----
    ggml_tensor* linear(const std::string& base, ggml_tensor* x, int64_t Cout) {
        ggml_tensor* y = ggml_mul_mat(ctx, W(base + ".weight"), x);
        return ggml_add(ctx, y, ggml_reshape_2d(ctx, f32(W(base + ".bias")), Cout, 1));
    }

    ggml_tensor* resnet(const std::string& p, ggml_tensor* x, int64_t Cin, int64_t Cmid,
                        int64_t Hi, int64_t Wi, int64_t groups, float eps) {
        ggml_tensor* h = conv3d(p + ".conv1", silu(group_norm(p + ".norm1", x, Cin, Wi, Hi, groups, eps)),
                                Hi, Wi, 1, 1);
        h = conv3d(p + ".conv2", silu(group_norm(p + ".norm2", h, Cmid, Wi, Hi, groups, eps)), Hi, Wi, 1, 1);
        ggml_tensor* res = x;
        if (has(p + ".conv_shortcut.weight")) res = conv1x1(p + ".conv_shortcut", x, Hi, Wi);
        return ggml_add(ctx, res, h);
    }

    // ---- mid-block spatial attention (single head, D = C) ----
    ggml_tensor* attention(const std::string& p, ggml_tensor* x, int64_t C, int64_t Hi, int64_t Wi,
                           int64_t groups, float eps) {
        ggml_tensor* xn = group_norm(p + ".group_norm", x, C, Wi, Hi, groups, eps);
        ggml_tensor* f = ggml_reshape_2d(ctx, ggml_cont(ctx, xn), C, Wi * Hi);   // {C, HW}
        ggml_tensor* q = linear(p + ".to_q", f, C);
        ggml_tensor* k = linear(p + ".to_k", f, C);
        ggml_tensor* v = linear(p + ".to_v", f, C);
        ggml_tensor* s = ggml_mul_mat(ctx, k, q);                    // {keys, queries}
        s = ggml_scale(ctx, s, (float)(1.0 / std::sqrt((double)C)));
        s = ggml_soft_max(ctx, s);                                   // softmax over keys (ne0)
        // out[c,q] = sum_k V[c,k]*S[k,q]: ggml_mul_mat contracts ne0, so V needs
        // its sequence dim on ne0
        ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));   // {HW, C}
        ggml_tensor* o = linear(p + ".to_out.0", ggml_mul_mat(ctx, vT, s), C);
        return ggml_add(ctx, ggml_reshape_3d(ctx, o, C, Wi, Hi), x);
    }

    // ---- decoder pixel-shuffle upsample ----
    ggml_tensor* upsample(const std::string& p, ggml_tensor* x, int64_t Cin, int64_t Hi, int64_t Wi) {
        ggml_tensor* S = conv1x1(p + ".upscale_conv", x, Hi, Wi);    // {Cin*R, Wi, Hi} (cropped view)
        const int64_t R = S->ne[0] / Cin;                            // 4 or 8 = x*y*z
        // The rearrange is "b (x y z c) f h w -> b c (f z) (h x) (w y)", so one
        // output channel block is (x_i*2 + y_i)*z_ + z_i (c innermost) - note the
        // *z_* factor: blocks 2/3 of the decoder have z_ == 1 (R == 4), where
        // multiplying by 2 instead of z_ reads the wrong planes entirely.
        // remove_head() keeps frame 0, hence z_i == 0 for the temporal-up case.
        const int64_t z_ = R / 4;                                    // temporal ratio
        const int zuse = 0;
        const int64_t H2 = Hi * 2, W2 = Wi * 2;
        // Build the interleave with contiguous tensors + concat instead of
        // writing the four sub-planes into strided views of T (the CPU backend
        // copies strided destinations reliably, Vulkan does not):
        //   {Cin,2(y),Wi,Hi} -merge-> W index = y + 2w
        //   {Cin,2Wi,2(x),Hi} -merge-> H index = x + 2h
        auto plane = [&](int x2, int y) {
            const size_t off = (size_t)(((x2 * 2 + y) * z_ + zuse)) * Cin * sizeof(float);
            return ggml_cont(ctx, ggml_view_3d(ctx, S, Cin, Wi, Hi, S->nb[1], S->nb[2], off));
        };
        ggml_tensor* rows = nullptr;                       // {Cin, 2Wi, 2(x), Hi}
        for (int x2 = 0; x2 < 2; ++x2) {
            ggml_tensor* a0 = ggml_reshape_4d(ctx, plane(x2, 0), Cin, 1, Wi, Hi);
            ggml_tensor* a1 = ggml_reshape_4d(ctx, plane(x2, 1), Cin, 1, Wi, Hi);
            ggml_tensor* aw = ggml_concat(ctx, a0, a1, 1);             // {Cin,2,Wi,Hi}
            aw = ggml_reshape_4d(ctx, ggml_reshape_3d(ctx, aw, Cin, 2 * Wi, Hi), Cin, 2 * Wi, 1, Hi);
            rows = rows ? ggml_concat(ctx, rows, aw, 2) : aw;
        }
        ggml_tensor* T = ggml_reshape_3d(ctx, rows, Cin, W2, H2);
        return conv3d(p + ".conv", T, H2, W2, 1, 1);
    }

    // ---- encoder / decoder ----
    int64_t attn_dim(const std::string& p) { return W(p + ".to_q.weight")->ne[0]; }

    ggml_tensor* encoder(ggml_tensor* x, int64_t Hi, int64_t Wi, int groups, float eps, int64_t& outC) {
        ggml_tensor* h = conv3d("encoder.conv_in", x, Hi, Wi, 1, 1);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                std::string p = "encoder.down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j);
                h = resnet(p, h, cin(p + ".conv1"), cout(p + ".conv1"), Hi, Wi, groups, eps);
            }
            std::string d = "encoder.down_blocks." + std::to_string(i) + ".downsamplers.0.conv";
            if (has(d + ".weight")) {
                h = conv_down(d, h, Hi, Wi);
                Hi = (Hi + 1 - 3) / 2 + 1; Wi = (Wi + 1 - 3) / 2 + 1;
            }
        }
        std::string m0 = "encoder.mid_block.resnets.0";
        h = resnet(m0, h, cin(m0 + ".conv1"), cout(m0 + ".conv1"), Hi, Wi, groups, eps);
        std::string a0 = "encoder.mid_block.attentions.0";
        h = attention(a0, h, attn_dim(a0), Hi, Wi, groups, eps);
        std::string m1 = "encoder.mid_block.resnets.1";
        h = resnet(m1, h, cin(m1 + ".conv1"), cout(m1 + ".conv1"), Hi, Wi, groups, eps);
        h = silu(group_norm("encoder.conv_norm_out", h, cout(m1 + ".conv1"), Wi, Hi, groups, eps));
        h = conv3d("encoder.conv_out", h, Hi, Wi, 1, 1);
        outC = cout("encoder.conv_out");
        return h;
    }

    ggml_tensor* decoder(ggml_tensor* z, int64_t Hi, int64_t Wi, int groups, float eps) {
        ggml_tensor* h = conv3d("decoder.conv_in", z, Hi, Wi, 1, 1);
        std::string m0 = "decoder.mid_block.resnets.0";
        h = resnet(m0, h, cin(m0 + ".conv1"), cout(m0 + ".conv1"), Hi, Wi, groups, eps);
        std::string a0 = "decoder.mid_block.attentions.0";
        h = attention(a0, h, attn_dim(a0), Hi, Wi, groups, eps);
        std::string m1 = "decoder.mid_block.resnets.1";
        h = resnet(m1, h, cin(m1 + ".conv1"), cout(m1 + ".conv1"), Hi, Wi, groups, eps);
        int64_t C = cout(m1 + ".conv1");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::string p = "decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j);
                h = resnet(p, h, cin(p + ".conv1"), cout(p + ".conv1"), Hi, Wi, groups, eps);
                C = cout(p + ".conv1");
            }
            std::string u = "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0";
            if (has(u + ".conv.weight")) {
                h = upsample(u, h, cin(u + ".upscale_conv"), Hi, Wi);
                Hi *= 2; Wi *= 2;
                C = cout(u + ".conv");
            }
        }
        h = silu(group_norm("decoder.conv_norm_out", h, C, Wi, Hi, groups, eps));
        ggml_tensor* o = conv3d("decoder.conv_out", h, Hi, Wi, 1, 1);
        return o;
    }
};

struct Runner {
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t alloc = nullptr;
    std::vector<std::pair<ggml_tensor*, std::vector<char>>> pending;
    ~Runner() { if (alloc) ggml_gallocr_free(alloc); if (ctx) ggml_free(ctx); }

    // NOTE: deliberately *not* ggml_backend_sched.  With a scheduler the graph
    // gets split whenever an op lands on a different backend, and every split
    // boundary copies its inputs (the "split inputs capacity" messages); with
    // the leaf tensors (indicators, constants) in host memory that ping-pongs
    // per layer and dominates the runtime.  The gallocr keeps every node in the
    // device buffer, and the whole VAE graph is backend-friendly now
    // (contiguous operands, no strided writes).
    bool run(ggml_cgraph* gf) {
        if (!ggml_gallocr_alloc_graph(alloc, gf)) return false;
        for (auto& p : pending) ggml_backend_tensor_set(p.first, p.second.data(), 0, p.second.size());
        return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
};

// ---------------------------------------------------------------------------
// Graph runners - one ggml graph per call, shared by the untiled and the tiled
// paths.  The tiling itself (below) is a faithful port of the Python
// VideoAutoencoderKL.tiled_encode / tiled_decode: spatial tiles + a raised
// cosine feather blend in latent space (encode) / output space (decode).
// ---------------------------------------------------------------------------
constexpr double kPi = 3.14159265358979323846;

// encoder forward up to the quant_conv output (2 * latent_channels channels),
// channel-first: [1, 2C, 1, h, w]
torch::Tensor run_encoder_h(GgufModel& model, const torch::Tensor& x_in) {
    torch::Tensor x = x_in.to(torch::kFloat32).contiguous();
    require(x.dim() == 5 && x.size(0) == 1 && x.size(1) == 3 && x.size(2) == 1,
            "VAE(ggml) encode expects [1,3,1,H,W]");
    const int64_t H = x.size(3), W = x.size(4);

    ggml_init_params ip{};
    ip.mem_size = 1024ull * 1024 * 1024;
    ip.no_alloc = true;
    Runner R;
    R.ctx = ggml_init(ip);
    require(R.ctx != nullptr, "ggml_init failed (VAE)");
    ggml_cgraph* gf = ggml_new_graph_custom(R.ctx, 1 << 20, false);

    VaeCtx C;
    C.ctx = R.ctx; C.m = &model; C.graph_ = gf;

    // ggml tensors are channel-fastest: {C,W,H} puts (c,w,h) at
    // c + C*w + C*W*h - exactly the memory order of a torch [H,W,C] buffer.
    torch::Tensor xp = x.permute({ 0, 3, 4, 1, 2 }).contiguous();       // [1,H,W,3,1]
    ggml_tensor* in = ggml_new_tensor_3d(R.ctx, GGML_TYPE_F32, 3, W, H);
    C.input(in, xp.data_ptr<float>(), (size_t)3 * H * W * 4);

    int64_t Cz = 0;
    ggml_tensor* h = C.encoder(in, H, W, 32, 1e-6f, Cz);
    const int64_t Ho = h->ne[2], Wo = h->ne[1];       // {Cz, Wo, Ho} contiguous
    ggml_build_forward_expand(gf, h);

    R.backend = model.backend();
    R.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(R.backend));
    R.pending.swap(C.pending);
    require(R.run(gf), "VAE encode graph failed");

    torch::Tensor hwc = torch::empty({ Ho, Wo, Cz }, torch::kFloat32);
    ggml_backend_tensor_get(h, hwc.data_ptr<float>(), 0, (size_t)Cz * Wo * Ho * 4);
    return hwc.permute({ 2, 0, 1 }).unsqueeze(0).unsqueeze(2).contiguous();   // [1,Cz,1,Ho,Wo]
}

torch::Tensor run_decoder(GgufModel& model, const torch::Tensor& z_in) {
    torch::Tensor z = z_in.to(torch::kFloat32).contiguous();
    require(z.dim() == 5 && z.size(0) == 1 && z.size(2) == 1, "VAE(ggml) decode expects [1,C,1,h,w]");
    const int64_t C0 = z.size(1), H = z.size(3), W = z.size(4);

    ggml_init_params ip{};
    ip.mem_size = 1024ull * 1024 * 1024;
    ip.no_alloc = true;
    Runner R;
    R.ctx = ggml_init(ip);
    require(R.ctx != nullptr, "ggml_init failed (VAE)");
    ggml_cgraph* gf = ggml_new_graph_custom(R.ctx, 1 << 20, false);

    VaeCtx C;
    C.ctx = R.ctx; C.m = &model; C.graph_ = gf;

    torch::Tensor zp = z.permute({ 0, 3, 4, 1, 2 }).contiguous();       // [1,H,W,C0,1]
    ggml_tensor* in = ggml_new_tensor_3d(R.ctx, GGML_TYPE_F32, C0, W, H);
    C.input(in, zp.data_ptr<float>(), (size_t)C0 * H * W * 4);

    ggml_tensor* h = C.decoder(in, H, W, 32, 1e-6f);
    const int64_t Co = h->ne[0], Wo = h->ne[1], Ho = h->ne[2];
    ggml_tensor* out = ggml_cont(R.ctx, h);
    ggml_build_forward_expand(gf, out);

    R.backend = model.backend();
    R.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(R.backend));
    R.pending.swap(C.pending);
    require(R.run(gf), "VAE decode graph failed");

    torch::Tensor hwc = torch::empty({ Ho, Wo, Co }, torch::kFloat32);
    ggml_backend_tensor_get(out, hwc.data_ptr<float>(), 0, (size_t)Co * Wo * Ho * 4);
    return hwc.permute({ 2, 0, 1 }).unsqueeze(0).unsqueeze(2).contiguous();   // [1,Co,1,Ho,Wo]
}

// 0.5 - 0.5*cos(t*pi), t = linspace(0,1,n) - the Python feather ramp
torch::Tensor make_ramp(int64_t n, const torch::TensorOptions& opt) {
    torch::Tensor t = torch::linspace(0.0, 1.0, n, opt);
    return 0.5 - 0.5 * torch::cos(t * kPi);
}

// Posterior: split h into (mean, logvar), clamp, then sample with torch's RNG
// (same noise as the reference / libtorch path at a given seed).
torch::Tensor posterior_sample(const torch::Tensor& h, double seed) {
    const int64_t C = h.size(1) / 2;
    torch::Tensor mean = h.narrow(1, 0, C);
    torch::Tensor logvar = h.narrow(1, C, C);
    logvar = logvar.clamp(-30.0, 20.0);
    torch::Tensor std = (logvar * 0.5).exp();
    torch::manual_seed((int64_t)seed);
    torch::Tensor noise = torch::randn(mean.sizes(), mean.options());
    return (mean + std * noise).contiguous();
}

// Python VideoAutoencoderKL.tiled_encode: tile the input in output space, run
// the full encoder on each tile, feather-blend the (2C) encoder outputs in
// latent space, then build the posterior once on the assembled h.
torch::Tensor tiled_encode_h(GgufModel& model, const torch::Tensor& x, int tile_px, int tile_ov) {
    const int s = 8;                                  // spatial downsample factor
    const int64_t H = x.size(3), W = x.size(4);
    int64_t lth = std::max<int64_t>(1, tile_px / s);
    int64_t ltw = std::max<int64_t>(1, tile_px / s);
    if (H <= tile_px && W <= tile_px) return run_encoder_h(model, x);

    int64_t lov_h = std::max<int64_t>(0, std::min<int64_t>(tile_ov / s, lth - 1));
    int64_t lov_w = std::max<int64_t>(0, std::min<int64_t>(tile_ov / s, ltw - 1));
    int64_t sh = std::max<int64_t>(1, lth - lov_h);
    int64_t sw = std::max<int64_t>(1, ltw - lov_w);
    int64_t HlT = (H + s - 1) / s, WlT = (W + s - 1) / s;

    torch::Tensor ramp_h, ramp_w;
    if (lov_h > 0) ramp_h = make_ramp(lov_h, x.options());
    if (lov_w > 0) ramp_w = make_ramp(lov_w, x.options());

    torch::Tensor result, count;
    int64_t ntiles = 0;
    for (int64_t y = 0; y < HlT; y += sh) {
        int64_t yl = std::min(y + lth, HlT);
        for (int64_t xx = 0; xx < WlT; xx += sw) {
            int64_t xl = std::min(xx + ltw, WlT);
            // skip a tile that lies fully inside the previous tile's overlap
            if ((y > 0 && (yl - y) <= lov_h) || (xx > 0 && (xl - xx) <= lov_w)) continue;
            int64_t y0 = y * s, x0 = xx * s;
            int64_t y1 = std::min(yl * s, H), x1 = std::min(xl * s, W);
            torch::Tensor tile = x.narrow(3, y0, y1 - y0).narrow(4, x0, x1 - x0).contiguous();
            torch::Tensor enc = run_encoder_h(model, tile);        // [1,2C,1,Ho,Wo]
            ++ntiles;
            if (!result.defined()) {
                result = torch::zeros({ 1, enc.size(1), 1, HlT, WlT }, enc.options());
                count = torch::zeros({ 1, 1, 1, HlT, WlT }, enc.options());
            }
            int64_t eff_h = std::min({ yl - y, enc.size(3), HlT - y });
            int64_t eff_w = std::min({ xl - xx, enc.size(4), WlT - xx });
            enc = enc.narrow(1, 0, result.size(1)).narrow(3, 0, eff_h).narrow(4, 0, eff_w).contiguous();

            torch::Tensor wh = torch::ones({ eff_h }, enc.options());
            torch::Tensor ww = torch::ones({ eff_w }, enc.options());
            int64_t ovh = std::max<int64_t>(0, std::min<int64_t>(lov_h, eff_h - 1));
            int64_t ovw = std::max<int64_t>(0, std::min<int64_t>(lov_w, eff_w - 1));
            if (ovh > 0) {
                if (y > 0) wh.narrow(0, 0, ovh).copy_(ramp_h.narrow(0, 0, ovh));
                if (yl < HlT) wh.narrow(0, eff_h - ovh, ovh).copy_(1.0 - ramp_h.narrow(0, 0, ovh));
            }
            if (ovw > 0) {
                if (xx > 0) ww.narrow(0, 0, ovw).copy_(ramp_w.narrow(0, 0, ovw));
                if (xl < WlT) ww.narrow(0, eff_w - ovw, ovw).copy_(1.0 - ramp_w.narrow(0, 0, ovw));
            }
            torch::Tensor wh5 = wh.view({ 1, 1, 1, eff_h, 1 });
            torch::Tensor ww5 = ww.view({ 1, 1, 1, 1, eff_w });
            enc.mul_(wh5).mul_(ww5);
            result.narrow(3, y, eff_h).narrow(4, xx, eff_w).add_(enc);
            count.narrow(3, y, eff_h).narrow(4, xx, eff_w).add_(wh5 * ww5);
        }
    }
    result.div_(count.clamp_min(1e-6));
    SEEDVR_LOG(LogLevel::Info, "VAE tiled encode: " + std::to_string(ntiles) + " tiles (size " +
                                   std::to_string(tile_px) + ", overlap " + std::to_string(tile_ov) + ")");
    return result;
}

// Python VideoAutoencoderKL.tiled_decode: tile the latent, run the full decoder
// on each tile, feather-blend in output space.
torch::Tensor tiled_decode(GgufModel& model, const torch::Tensor& z, int tile_px, int tile_ov) {
    const int s = 8;
    const int64_t H = z.size(3), W = z.size(4);
    int64_t lth = std::max<int64_t>(1, tile_px / s);
    int64_t ltw = std::max<int64_t>(1, tile_px / s);
    if (H <= lth && W <= ltw) return run_decoder(model, z);

    int64_t lov_h = std::max<int64_t>(0, std::min<int64_t>(tile_ov / s, lth - 1));
    int64_t lov_w = std::max<int64_t>(0, std::min<int64_t>(tile_ov / s, ltw - 1));
    int64_t sh = std::max<int64_t>(1, lth - lov_h);
    int64_t sw = std::max<int64_t>(1, ltw - lov_w);

    torch::Tensor ramp_h, ramp_w;
    if (tile_ov > 0) { ramp_h = make_ramp(tile_ov, z.options()); ramp_w = ramp_h; }

    torch::Tensor result, count;
    int64_t ntiles = 0;
    for (int64_t y = 0; y < H; y += sh) {
        int64_t yl = std::min(y + lth, H);
        for (int64_t xx = 0; xx < W; xx += sw) {
            int64_t xl = std::min(xx + ltw, W);
            if ((y > 0 && (yl - y) <= lov_h) || (xx > 0 && (xl - xx) <= lov_w)) continue;
            torch::Tensor tile = z.narrow(3, y, yl - y).narrow(4, xx, xl - xx).contiguous();
            torch::Tensor dec = run_decoder(model, tile);          // [1,3,1,ho,wo]
            ++ntiles;
            if (!result.defined()) {
                result = torch::zeros({ 1, dec.size(1), 1, H * s, W * s }, dec.options());
                count = torch::zeros({ 1, 1, 1, H * s, W * s }, dec.options());
            }
            int64_t y0 = y * s, x0 = xx * s;
            int64_t h_out = (yl - y) * s, w_out = (xl - xx) * s;
            require(dec.size(3) == h_out && dec.size(4) == w_out, "VAE decode tile size mismatch");
            int64_t ovh = std::max<int64_t>(0, std::min<int64_t>(tile_ov, h_out - 1));
            int64_t ovw = std::max<int64_t>(0, std::min<int64_t>(tile_ov, w_out - 1));
            torch::Tensor wh = torch::ones({ h_out }, dec.options());
            torch::Tensor ww = torch::ones({ w_out }, dec.options());
            if (ovh > 0) {
                if (y > 0) wh.narrow(0, 0, ovh).copy_(ramp_h.narrow(0, 0, ovh));
                if (yl < H) wh.narrow(0, h_out - ovh, ovh).copy_(1.0 - ramp_h.narrow(0, 0, ovh));
            }
            if (ovw > 0) {
                if (xx > 0) ww.narrow(0, 0, ovw).copy_(ramp_w.narrow(0, 0, ovw));
                if (xl < W) ww.narrow(0, w_out - ovw, ovw).copy_(1.0 - ramp_w.narrow(0, 0, ovw));
            }
            torch::Tensor wh5 = wh.view({ 1, 1, 1, h_out, 1 });
            torch::Tensor ww5 = ww.view({ 1, 1, 1, 1, w_out });
            dec.mul_(wh5).mul_(ww5);
            result.narrow(3, y0, h_out).narrow(4, x0, w_out).add_(dec);
            count.narrow(3, y0, h_out).narrow(4, x0, w_out).add_(wh5 * ww5);
        }
    }
    result.div_(count.clamp_min(1e-6));
    SEEDVR_LOG(LogLevel::Info, "VAE tiled decode: " + std::to_string(ntiles) + " tiles (size " +
                                   std::to_string(tile_px) + ", overlap " + std::to_string(tile_ov) + ")");
    return result;
}

} // namespace

struct VaeGgml::Impl {
    GgufModel model;
    std::string info;
};

VaeGgml::VaeGgml() : impl_(new Impl()) {}
VaeGgml::~VaeGgml() { delete impl_; }
const std::string& VaeGgml::info() const { return impl_->info; }

void VaeGgml::free() {
    impl_->model.free_all();
    loaded_ = false;
}

void VaeGgml::load(const std::string& gguf_path, const std::string& device) {
    // upcast F16 weights to F32: else the CUDA backend converts the activations to F16 for every
    // mul_mat against F16 weights, and this VAE does one mul_mat per conv tap (thousands per forward)
    impl_->model.load(gguf_path, device, /*upcast_f16=*/true);
    impl_->info = std::string("ggml/") + impl_->model.backend_name() + " f32";
    loaded_ = true;
    SEEDVR_LOG(LogLevel::Info, "VAE(ggml) loaded: " + impl_->info + " | " + gguf_path);
}

torch::Tensor VaeGgml::encode(const torch::Tensor& x_in, double sample_seed,
                              bool tiled, int tile_size, int tile_overlap) {
    require(loaded_, "VAE(ggml) not loaded");
    torch::Tensor h = tiled ? tiled_encode_h(impl_->model, x_in, tile_size, tile_overlap)
                            : run_encoder_h(impl_->model, x_in);
    return posterior_sample(h, sample_seed);
}

torch::Tensor VaeGgml::decode(const torch::Tensor& z_in,
                              bool tiled, int tile_size, int tile_overlap) {
    require(loaded_, "VAE(ggml) not loaded");
    return tiled ? tiled_decode(impl_->model, z_in, tile_size, tile_overlap)
                 : run_decoder(impl_->model, z_in);
}

} // namespace seedvr2
