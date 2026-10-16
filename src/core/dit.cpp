// NaDiT 3B inference on ggml (CUDA).
//
// Layout conventions
//   * activations are F32, ne0 = feature dim, ne1.. = token dims
//   * GGUF weights have ne0 = in_features, ne1 = out_features, therefore
//     y = W x  ==  ggml_mul_mat(W, x)
//
#include "core/dit.h"
#include "core/common.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace seedvr2 {

// ---------------------------------------------------------------------------
static double pyround(double x) { return std::nearbyint(x); }

struct Win { int t0, t1, h0, h1, w0, w1; };

static std::vector<Win> make_720p_windows(int t, int h, int w, bool shifted) {
    // window = (4,3,3), window_method = 720pwin / 720pswin by size
    const int nw0 = 4, nw1 = 3, nw2 = 3;
    std::vector<Win> out;
    double scale = std::sqrt((45.0 * 80.0) / ((double)h * (double)w));
    int resized_h = (int)pyround(h * scale);
    int resized_w = (int)pyround(w * scale);
    int wh = std::max(1, (int)std::ceil((double)resized_h / nw1));
    int ww = std::max(1, (int)std::ceil((double)resized_w / nw2));
    int wt = std::max(1, (int)std::ceil((double)std::min(t, 30) / nw0));

    if (!shifted) {
        int nt = (int)std::ceil((double)t / wt);
        int nh = (int)std::ceil((double)h / wh);
        int nwi = (int)std::ceil((double)w / ww);
        for (int iw = 0; iw < nwi; ++iw)
            for (int ih = 0; ih < nh; ++ih)
                for (int it = 0; it < nt; ++it) {
                    int w0 = iw * ww, w1 = std::min((iw + 1) * ww, w);
                    int h0 = ih * wh, h1 = std::min((ih + 1) * wh, h);
                    int t0 = it * wt, t1 = std::min((it + 1) * wt, t);
                    if (w1 > w0 && h1 > h0 && t1 > t0) out.push_back({ t0, t1, h0, h1, w0, w1 });
                }
    } else {
        double st = (wt < t) ? 0.5 : 0.0;
        double sh = (wh < h) ? 0.5 : 0.0;
        double sw = (ww < w) ? 0.5 : 0.0;
        int nt = (int)std::ceil((t - st) / wt);
        int nh = (int)std::ceil((h - sh) / wh);
        int nwi = (int)std::ceil((w - sw) / ww);
        nt = (st > 0) ? nt + 1 : 1;
        nh = (sh > 0) ? nh + 1 : 1;
        nwi = (sw > 0) ? nwi + 1 : 1;
        for (int iw = 0; iw < nwi; ++iw)
            for (int ih = 0; ih < nh; ++ih)
                for (int it = 0; it < nt; ++it) {
                    int w0 = std::max((int)((iw - sw) * ww), 0);
                    int w1 = std::min((int)((iw - sw + 1) * ww), w);
                    int h0 = std::max((int)((ih - sh) * wh), 0);
                    int h1 = std::min((int)((ih - sh + 1) * wh), h);
                    int t0 = std::max((int)((it - st) * wt), 0);
                    int t1 = std::min((int)((it - st + 1) * wt), t);
                    if (w1 > w0 && h1 > h0 && t1 > t0) out.push_back({ t0, t1, h0, h1, w0, w1 });
                }
    }
    return out;
}

// ---------------------------------------------------------------------------
static std::vector<float> fetch_f32(GgufModel& m, const std::string& name) {
    ggml_tensor* t = m.get(name);
    int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw((size_t)n);
        ggml_backend_tensor_get(t, raw.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) out[(size_t)i] = ggml_fp16_to_fp32(raw[(size_t)i]);
    } else {
        throw Error("fetch_f32: unsupported tensor type for " + name);
    }
    return out;
}

// ---------------------------------------------------------------------------
void Dit::free() {
    if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
    model_.free_all();
    embW1.clear(); embB1.clear(); embW2.clear(); embB2.clear(); embW3.clear(); embB3.clear();
    ropeFreq_.clear();
    loaded_ = false;
}

void Dit::load(const std::string& gguf_path, const std::string& device) {
    if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
    model_.load(gguf_path, device);
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));
    require(galloc_ != nullptr, "gallocr init failed");

    // time-embedding MLP weights -> host (F32)
    embW1 = fetch_f32(model_, "emb_in.proj_in.weight");   embB1 = fetch_f32(model_, "emb_in.proj_in.bias");
    embW2 = fetch_f32(model_, "emb_in.proj_hid.weight");  embB2 = fetch_f32(model_, "emb_in.proj_hid.bias");
    embW3 = fetch_f32(model_, "emb_in.proj_out.weight");  embB3 = fetch_f32(model_, "emb_in.proj_out.bias");
    embHid = cfg_.vid_dim;
    ropeFreq_ = fetch_f32(model_, "blocks.0.attn.rope.rope.freqs");

    loaded_ = true;
    SEEDVR_LOG(LogLevel::Info, std::string("DiT ready on ") + model_.backend_name());
}

// proj_in(256->2560) -> silu -> proj_hid(2560->2560) -> silu -> proj_out(2560->15360)
void Dit::compute_time_embedding(float t, std::vector<float>& emb) const {
    const int V = cfg_.vid_dim, E = cfg_.emb_dim, SIN = 256;
    std::vector<float> x(SIN);
    for (int i = 0; i < 128; ++i) {
        double exponent = -std::log(10000.0) * (double)i / 128.0;
        double ang = (double)t * std::exp(exponent);
        x[i] = (float)std::sin(ang);
        x[128 + i] = (float)std::cos(ang);
    }
    // proj_in: [V, SIN] x [SIN] -> [V]
    std::vector<float> h1(V);
    for (int o = 0; o < V; ++o) {
        const float* w = &embW1[(size_t)o * SIN];
        float s = embB1[o];
        for (int i = 0; i < SIN; ++i) s += w[i] * x[i];
        h1[o] = s / (1.0f + std::exp(-s));   // silu
    }
    // proj_hid: [V, V]
    std::vector<float> h2(V);
    for (int o = 0; o < V; ++o) {
        const float* w = &embW2[(size_t)o * V];
        float s = embB2[o];
        for (int i = 0; i < V; ++i) s += w[i] * h1[i];
        h2[o] = s / (1.0f + std::exp(-s));   // silu
    }
    // proj_out: [E, V]
    emb.assign((size_t)E, 0.0f);
    for (int o = 0; o < E; ++o) {
        const float* w = &embW3[(size_t)o * V];
        float s = embB3[o];
        for (int i = 0; i < V; ++i) s += w[i] * h2[i];
        emb[o] = s;
    }
    {
        double mn = 1e30, mx = -1e30, mean = 0.0, var = 0.0;
        for (int i = 0; i < E; ++i) { double v = emb[i]; mn = std::min(mn, v); mx = std::max(mx, v); mean += v; }
        mean /= E;
        for (int i = 0; i < E; ++i) { double d = emb[i] - mean; var += d * d; }
        var /= E;
        if (verbose_) if (verbose_) if (verbose_) SEEDVR_LOG(LogLevel::Info, "emb(t=" + std::to_string(t) + "): mean=" + std::to_string(mean) +
                                       " std=" + std::to_string(std::sqrt(var)) +
                                       " min=" + std::to_string(mn) + " max=" + std::to_string(mx));
    }
}

// ---------------------------------------------------------------------------
namespace {

struct Builder {
    ggml_context* ctx = nullptr;

    // GGUF scalars/vectors are F16; ggml's elementwise kernels need matching
    // element sizes, so materialise them as F32 before use.
    ggml_tensor* f32(ggml_tensor* t) {
        if (t->type == GGML_TYPE_F32) return t;
        return ggml_cast(ctx, t, GGML_TYPE_F32);
    }

    ggml_tensor* lin(GgufModel& m, ggml_tensor* x, const std::string& w,
                     const std::string& b = "") {
        ggml_tensor* W = m.get(w);
        ggml_tensor* y = ggml_mul_mat(ctx, W, x);
        if (!b.empty() && m.has(b)) {
            ggml_tensor* B = f32(m.get(b));
            y = ggml_add(ctx, y, ggml_reshape_2d(ctx, B, B->ne[0], 1));
        }
        return y;
    }

    ggml_tensor* qk_prep(GgufModel& m, ggml_tensor* x, const std::string& wname, float eps) {
        ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
        ggml_tensor* W = f32(m.get(wname));
        return ggml_mul(ctx, ggml_cont(ctx, y), ggml_reshape_2d(ctx, W, W->ne[0], 1));
    }

    // x:{HD,NH,L}  cos/sin:{1,NP,1,L} -> rotated {HD,NH,L}
    //
    // Mirrors rotary_embedding_torch.apply_rotary_emb(): the leading 2*NP head
    // dims are rotated, pair p (dims 2p/2p+1, adjacent) uses cos/sin[p]; every
    // remaining head dim is passed through untouched.
    ggml_tensor* rope(ggml_tensor* x, ggml_tensor* cosT, ggml_tensor* sinT,
                      int64_t NH, int64_t L) {
        const int64_t HD = x->ne[0];
        const int64_t NP = cosT->ne[1];
        const int64_t RD = 2 * NP;
        const size_t nb1 = x->nb[1], nb2 = x->nb[2];
        ggml_tensor* xrot = ggml_cont(ctx, ggml_view_3d(ctx, x, RD, NH, L, nb1, nb2, 0));
        ggml_tensor* xr = ggml_reshape_4d(ctx, xrot, 2, NP, NH, L);
        const size_t s0 = 2 * sizeof(float);
        const size_t s1 = s0 * NP;
        const size_t s2 = s0 * NP * NH;
        ggml_tensor* e0 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1, NP, NH, L, s0, s1, s2, 0));
        ggml_tensor* e1 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1, NP, NH, L, s0, s1, s2, sizeof(float)));
        ggml_tensor* tpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, NP, NH, L);
        ggml_tensor* c = ggml_repeat(ctx, cosT, tpl);
        ggml_tensor* s = ggml_repeat(ctx, sinT, tpl);
        ggml_tensor* oe = ggml_sub(ctx, ggml_mul(ctx, e0, c), ggml_mul(ctx, e1, s));
        ggml_tensor* oo = ggml_add(ctx, ggml_mul(ctx, e0, s), ggml_mul(ctx, e1, c));
        ggml_tensor* rot = ggml_concat(ctx, ggml_cont(ctx, oe), ggml_cont(ctx, oo), 0);
        ggml_tensor* rotR = ggml_reshape_3d(ctx, ggml_cont(ctx, rot), RD, NH, L);
        if (HD == RD) return rotR;
        ggml_tensor* rest = ggml_cont(ctx, ggml_view_3d(ctx, x, HD - RD, NH, L, nb1, nb2, (size_t)RD * sizeof(float)));
        return ggml_concat(ctx, rotR, rest, 0);
    }
};

struct Pending { ggml_tensor* t; std::vector<char> data; };

static std::string nes(const ggml_tensor* t) {
    char b[128];
    std::snprintf(b, sizeof(b), "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1],
                  (long long)t->ne[2], (long long)t->ne[3]);
    return std::string(b);
}

} // namespace

// ---------------------------------------------------------------------------
torch::Tensor Dit::forward(const float* vid_in, int T, int H, int W,
                           const float* txt, int Lt, float t) {
    const DitConfig& C = cfg_;
    const int V = C.vid_dim, HD = C.head_dim, NH = C.heads;
    const int Hp = H / C.patch_h, Wp = W / C.patch_w;
    const int Lp = T * Hp * Wp;
    const int PATCH = C.patch_t * C.patch_h * C.patch_w;
    const int PIN = C.vid_in_ch * PATCH;
    const int POUT = C.vid_out_ch * PATCH;

    // ---------------- patchify (host) ----------------
    std::vector<float> tok((size_t)Lp * PIN);
    for (int tt = 0; tt < T; ++tt)
        for (int hp = 0; hp < Hp; ++hp)
            for (int wp = 0; wp < Wp; ++wp) {
                int tokidx = (tt * Hp + hp) * Wp + wp;
                float* dst = tok.data() + (size_t)tokidx * PIN;
                for (int hh = 0; hh < C.patch_h; ++hh)
                    for (int ww = 0; ww < C.patch_w; ++ww)
                        for (int cc = 0; cc < C.vid_in_ch; ++cc) {
                            size_t src = (((size_t)(tt * H + hp * C.patch_h + hh) * W) +
                                          (wp * C.patch_w + ww)) * C.vid_in_ch + cc;
                            dst[(hh * C.patch_w + ww) * C.vid_in_ch + cc] = vid_in[src];
                        }
            }

    // ---------------- time embedding (host) ----------------
    std::vector<float> embCpu;
    compute_time_embedding(t, embCpu);

    // ---- debug dump of the exact DiT inputs (SEEDVR2_DUMP=<dir>) ----
    const char* dumpDir = std::getenv("SEEDVR2_DUMP");
    const bool dumpAll = (std::getenv("SEEDVR2_DUMP_ALL") != nullptr);
    std::vector<std::pair<std::string, ggml_tensor*>> dumpOut;
    auto writeBin = [&](const std::string& nm, const void* p, size_t bytes) {
        std::ofstream f(std::string(dumpDir) + "/" + nm, std::ios::binary);
        f.write((const char*)p, (std::streamsize)bytes);
    };

    // ================= time embedding =================

    // emb layout is (d, l=2, g=3) -> flat index d*6 + l*3 + g
    auto extractA = [&](int l, int g) {
        std::vector<float> o(V);
        for (int d = 0; d < V; ++d) o[d] = embCpu[(size_t)d * 6 + l * 3 + g];
        return o;
    };
    std::vector<float> attnShift = extractA(0, 0), attnScale = extractA(0, 1), attnGate = extractA(0, 2);
    std::vector<float> mlpShift = extractA(1, 0),  mlpScale = extractA(1, 1),  mlpGate = extractA(1, 2);
    if (std::getenv("SEEDVR2_VERBOSE")) {
        auto rep = [&](const char* nm, const std::vector<float>& v) {
            double mean = 0, mn = 1e30, mx = -1e30;
            for (float x : v) { mean += x; mn = std::min(mn, (double)x); mx = std::max(mx, (double)x); }
            mean /= (double)v.size();
            double var = 0; for (float x : v) { double d = x - mean; var += d * d; } var /= (double)v.size();
            SEEDVR_LOG(LogLevel::Debug, std::string("    A.") + nm + " mean=" + std::to_string(mean) +
                                           " std=" + std::to_string(std::sqrt(var)) +
                                           " min=" + std::to_string(mn) + " max=" + std::to_string(mx));
        };
        rep("attnShift", attnShift); rep("attnScale", attnScale); rep("attnGate", attnGate);
        rep("mlpShift", mlpShift);   rep("mlpScale", mlpScale);   rep("mlpGate", mlpGate);
    }
    std::vector<float> outShift = attnShift, outScale = attnScale;  // final ada: layer slot idx=0

    // ================= main graph =================
    ggml_init_params ip{};
    ip.mem_size = 512ull * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 65536, false);
    Builder B{ ctx };

    // Debug dumps for diffing against the python reference.  Every captured
    // tensor is copied (GGML_OP_CPY) into its own freshly created leaf tensor,
    // so the allocator can never recycle a buffer before we read it back.
    std::vector<std::pair<std::string, ggml_tensor*>> dumpB0;
    auto DumpInto = [&](std::vector<std::pair<std::string, ggml_tensor*>>& list,
                        const std::string& nm, ggml_tensor* t) {
        ggml_tensor* leaf = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, t->ne);
        ggml_set_name(leaf, nm.c_str());
        list.push_back({ nm, leaf });
        ggml_build_forward_expand(gf, ggml_cpy(ctx, t, leaf));
    };
    auto D0 = [&](const std::string& nm, ggml_tensor* t) {
        if (dumpAll) DumpInto(dumpB0, nm, t);
    };

    std::vector<Pending> pending;
    auto addInput = [&](ggml_tensor* x, const void* p, size_t nb) {
        Pending pd; pd.t = x; pd.data.assign((const char*)p, (const char*)p + nb);
        pending.push_back(std::move(pd));
        return x;
    };

    ggml_tensor* vidTok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PIN, Lp);
    ggml_tensor* txtTok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C.txt_in_dim, Lt);
    addInput(vidTok, tok.data(), (size_t)Lp * PIN * sizeof(float));
    addInput(txtTok, txt, (size_t)Lt * C.txt_in_dim * sizeof(float));
    {
        double mean = 0, mn = 1e30, mx = -1e30;
        for (float v : tok) { mean += v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
        mean /= (double)tok.size();
        double var = 0; for (float v : tok) { double d = v - mean; var += d * d; } var /= (double)tok.size();
        double tmean = 0, tmn = 1e30, tmx = -1e30;
        for (int i = 0; i < Lt * C.txt_in_dim; ++i) { tmean += txt[i]; tmn = std::min(tmn, (double)txt[i]); tmx = std::max(tmx, (double)txt[i]); }
        tmean /= (double)(Lt * C.txt_in_dim);
        SEEDVR_LOG(LogLevel::Info,
                   "tokens: vid mean=" + std::to_string(mean) + " std=" + std::to_string(std::sqrt(var)) +
                   " min=" + std::to_string(mn) + " max=" + std::to_string(mx) +
                   " | txt mean=" + std::to_string(tmean) + " min=" + std::to_string(tmn) + " max=" + std::to_string(tmx));
    }

    if (dumpDir) {
        writeBin("vid_tokens.f32", tok.data(), tok.size() * sizeof(float));
        writeBin("txt_tokens.f32", txt, (size_t)Lt * C.txt_in_dim * sizeof(float));
        writeBin("emb.f32", embCpu.data(), embCpu.size() * sizeof(float));
        std::ofstream m(std::string(dumpDir) + "/meta.txt");
        m << "T " << T << "\nH " << H << "\nW " << W << "\nLp " << Lp << "\nPIN " << PIN
          << "\nLt " << Lt << "\ntxtdim " << C.txt_in_dim << "\nt " << t
          << "\nHp " << Hp << "\nWp " << Wp << "\nPOUT " << POUT << "\n";
    }

    auto mkA = [&](const char* nm, std::vector<float>& v) {
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, 1);
        ggml_set_name(x, nm);
        addInput(x, v.data(), v.size() * sizeof(float));
        return x;
    };
    ggml_tensor* aAS = mkA("a_attn_shift", attnShift);
    ggml_tensor* aAC = mkA("a_attn_scale", attnScale);
    ggml_tensor* aAG = mkA("a_attn_gate", attnGate);
    ggml_tensor* aMS = mkA("a_mlp_shift", mlpShift);
    ggml_tensor* aMC = mkA("a_mlp_scale", mlpScale);
    ggml_tensor* aMG = mkA("a_mlp_gate", mlpGate);
    ggml_tensor* aOS = mkA("a_out_shift", outShift);
    ggml_tensor* aOC = mkA("a_out_scale", outScale);

    ggml_tensor* vid = B.lin(model_, vidTok, "vid_in.proj.weight", "vid_in.proj.bias");
    ggml_tensor* txtT = B.lin(model_, txtTok, "txt_in.weight", "txt_in.bias");
    ggml_tensor* probe0 = vid;                       // probe: after vid_in.proj
    D0("b0.vidproj", vid);
    D0("b0.txtproj", txtT);
    ggml_tensor* probe1 = nullptr;                    // probe: after block 0
    ggml_tensor* probe2 = nullptr;                    // probe: block0 attn proj_out
    ggml_tensor* probe3 = nullptr;                    // probe: block0 mlp swiglu
    ggml_tensor* probe4 = nullptr;                    // probe: block0 mlp gate pre-silu
    ggml_tensor* probe5 = nullptr;                    // probe: block0 attn ada gate
    ggml_tensor* probe6 = nullptr;                    // probe: block0 attn ada gate (A part)
    ggml_tensor* probe7 = nullptr;                    // probe: block0 attn adaOut result (x*gate)
    ggml_tensor* probe8 = nullptr;                    // probe: block0 attn proj_out (x of adaOut)

    // cache the rope frequency buffer (identical for every layer)
    std::vector<float> ropeFreq = ropeFreq_;

    for (int b = 0; b < C.num_layers; ++b) {
        const bool shared = (b >= C.mm_layers);
        const bool is_last = (b == C.num_layers - 1);
        const std::string p = "blocks." + std::to_string(b) + ".";
        const std::string vb = shared ? "all" : "vid";
        const std::string tb = shared ? "all" : "txt";

        // --- norm + ada in ---
        ggml_tensor* vn = ggml_rms_norm(ctx, vid, C.norm_eps);
        ggml_tensor* tn = ggml_rms_norm(ctx, txtT, C.norm_eps);
        if (b == 0) { D0("b0.vn", vn); D0("b0.tn", tn); }

        auto adaIn = [&](ggml_tensor* x, const char* lname, const std::string& br) {
            std::string base = p + "ada." + br + "." + lname;
            ggml_tensor* sh = B.f32(model_.get(base + "_shift"));
            ggml_tensor* sc = B.f32(model_.get(base + "_scale"));
            ggml_tensor* shA = (std::string(lname) == "attn") ? aAS : aMS;
            ggml_tensor* scA = (std::string(lname) == "attn") ? aAC : aMC;
            ggml_tensor* scale = ggml_add(ctx, ggml_reshape_2d(ctx, sc, V, 1), scA);
            ggml_tensor* shift = ggml_add(ctx, ggml_reshape_2d(ctx, sh, V, 1), shA);
            return ggml_add(ctx, ggml_mul(ctx, x, scale), shift);
        };
        auto adaOut = [&](ggml_tensor* x, const char* lname, const std::string& br) {
            std::string base = p + "ada." + br + "." + lname;
            ggml_tensor* gt = B.f32(model_.get(base + "_gate"));
            ggml_tensor* gtA = (std::string(lname) == "attn") ? aAG : aMG;
            ggml_tensor* gate = ggml_add(ctx, ggml_reshape_2d(ctx, gt, V, 1), gtA);
            if (b == 0 && std::string(lname) == "attn" && br == vb) probe5 = gate;
            if (b == 0 && std::string(lname) == "attn" && br == vb) probe6 = gtA;
            ggml_tensor* r = ggml_mul(ctx, x, gate);
            if (b == 0 && std::string(lname) == "attn" && br == vb) probe7 = r;
            if (b == 0 && std::string(lname) == "attn" && br == vb) probe8 = x;
            return r;
        };

        ggml_tensor* vq = adaIn(vn, "attn", vb);
        // the last block's ada module is vid_only, so its text stream is left
        // untouched (see NaMMSRTransformerBlock / MMModule.forward).
        ggml_tensor* tq = is_last ? tn : adaIn(tn, "attn", tb);
        if (b == 0) { D0("b0.vq", vq); D0("b0.tq", tq); }

        // --- qkv projection ---
        ggml_tensor* vqkv = B.lin(model_, vq, p + "attn.proj_qkv." + vb + ".weight");
        ggml_tensor* tqkv = B.lin(model_, tq, p + "attn.proj_qkv." + tb + ".weight");
        if (b == 0) { D0("b0.vqkv", vqkv); D0("b0.tqkv", tqkv); }

        auto split_qkv = [&](ggml_tensor* qkv, int64_t L) {
            // qkv: {3*V, L} with the per-token feature layout (o, h, d), d fastest
            // (matches the reference's rearrange("l (o h d) -> l o h d")).
            ggml_tensor* q4 = ggml_reshape_4d(ctx, qkv, HD, NH, 3, L);   // {HD, NH, o, L}
            // Each o slice is {HD, NH, L}: the h stride is nb[1] and the token
            // stride is nb[3] (nb[2] is the *o* stride - using it here would
            // treat q/k/v of one token as three separate tokens).
            ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, q4, HD, NH, L, q4->nb[1], q4->nb[3], 0));
            ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, q4, HD, NH, L, q4->nb[1], q4->nb[3], (size_t)V * sizeof(float)));
            ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, q4, HD, NH, L, q4->nb[1], q4->nb[3], (size_t)(2 * V) * sizeof(float)));
            return std::make_tuple(q, k, v);
        };
        auto [vq_, vk_, vv_] = split_qkv(vqkv, Lp);
        auto [tq_, tk_, tv_] = split_qkv(tqkv, Lt);

        // --- qk norm ---
        if (b == 0) {
            D0("b0.rawq", ggml_cont(ctx, vq_));                 // materialised q slice
            D0("b0.rmsq", ggml_rms_norm(ctx, vq_, C.norm_eps));  // pure rms norm
        }
        ggml_tensor* qv = B.qk_prep(model_, vq_, p + "attn.norm_q." + vb + ".weight", C.norm_eps);
        ggml_tensor* kv = B.qk_prep(model_, vk_, p + "attn.norm_k." + vb + ".weight", C.norm_eps);
        ggml_tensor* qt = B.qk_prep(model_, tq_, p + "attn.norm_q." + tb + ".weight", C.norm_eps);
        ggml_tensor* kt = B.qk_prep(model_, tk_, p + "attn.norm_k." + tb + ".weight", C.norm_eps);
        ggml_tensor* vt = ggml_cont(ctx, tv_);
        if (b == 0) { D0("b0.qv", qv); D0("b0.kv", kv); D0("b0.qt", qt); D0("b0.kt", kt); D0("b0.vt", vt); }

        // --- window partition ---
        std::vector<Win> wins = make_720p_windows(T, Hp, Wp, (b % 2) == 1);
        const int nwin = (int)wins.size();
        int WL = 0;
        for (auto& w : wins) WL = std::max(WL, (w.t1 - w.t0) * (w.h1 - w.h0) * (w.w1 - w.w0));
        const int S = WL + Lt;
        const int LR = WL * nwin;

        std::vector<int32_t> idx((size_t)LR, 0);
        for (int wi = 0; wi < nwin; ++wi) {
            const Win& Wn = wins[wi];
            int wt = Wn.t1 - Wn.t0, wh = Wn.h1 - Wn.h0, ww = Wn.w1 - Wn.w0;
            int loc = 0;
            for (int lt = 0; lt < wt; ++lt)
                for (int lh = 0; lh < wh; ++lh)
                    for (int lw = 0; lw < ww; ++lw, ++loc)
                        idx[(size_t)wi * WL + loc] =
                            ((Wn.t0 + lt) * Hp + (Wn.h0 + lh)) * Wp + (Wn.w0 + lw);
        }
        std::vector<int32_t> rev((size_t)Lp, 0);
        {
            std::vector<char> seen((size_t)Lp, 0);
            for (size_t j = 0; j < idx.size(); ++j)
                if (!seen[idx[j]]) { rev[idx[j]] = (int32_t)j; seen[idx[j]] = 1; }
        }

        // rope freqs
        //
        // rotary_embedding_torch.get_axial_freqs() concatenates one 2*NF block
        // per axis along the head dim, so pair p (dims 2p/2p+1) belongs to
        // axis p/NF and uses frequency ropeFreq[p%NF] scaled by that axis'
        // position.  Axis 0 (temporal) is offset by the text length, matching
        // seedvr2's `vid_freqs[l : l + f, :h, :w]`.
        const int NF = (int)ropeFreq.size();     // 21
        const int NP = 3 * NF;                   // 63 pairs -> 126 rotated dims
        std::vector<float> cosV((size_t)NP * LR), sinV((size_t)NP * LR);
        for (int wi = 0; wi < nwin; ++wi) {
            const Win& Wn = wins[wi];
            int wt = Wn.t1 - Wn.t0, wh = Wn.h1 - Wn.h0, ww = Wn.w1 - Wn.w0;
            int loc = 0;
            for (int lt = 0; lt < wt; ++lt)
                for (int lh = 0; lh < wh; ++lh)
                    for (int lw = 0; lw < ww; ++lw, ++loc) {
                        size_t li = (size_t)wi * WL + loc;
                        for (int pp = 0; pp < NP; ++pp) {
                            int axis = pp / NF;         // 0 = t, 1 = h, 2 = w
                            int fi = pp % NF;
                            int pos = (axis == 0) ? (Lt + lt) : (axis == 1 ? lh : lw);
                            double ang = (double)pos * (double)ropeFreq[fi];
                            cosV[(size_t)pp + NP * li] = (float)std::cos(ang);
                            sinV[(size_t)pp + NP * li] = (float)std::sin(ang);
                        }
                    }
        }
        // text: a single axis, replicated over all three blocks -> every pair
        // uses ropeFreq[p%NF] with the text token index as position.
        std::vector<float> cosT((size_t)NP * Lt), sinT((size_t)NP * Lt);
        for (int j = 0; j < Lt; ++j)
            for (int pp = 0; pp < NP; ++pp) {
                double ang = (double)j * (double)ropeFreq[pp % NF];
                cosT[(size_t)pp + NP * j] = (float)std::cos(ang);
                sinT[(size_t)pp + NP * j] = (float)std::sin(ang);
            }

        std::vector<ggml_fp16_t> mask((size_t)S * S * nwin);
        {
            ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int wi = 0; wi < nwin; ++wi) {
                const Win& Wn = wins[wi];
                int wl = (Wn.t1 - Wn.t0) * (Wn.h1 - Wn.h0) * (Wn.w1 - Wn.w0);
                for (int qi = 0; qi < S; ++qi)
                    for (int kj = 0; kj < S; ++kj)
                        mask[((size_t)wi * S + qi) * S + kj] =
                            ((kj < wl) || (kj >= WL)) ? zero : ninf;
            }
        }
        std::vector<int32_t> idxT((size_t)Lt * nwin);
        for (int wi = 0; wi < nwin; ++wi)
            for (int j = 0; j < Lt; ++j) idxT[(size_t)wi * Lt + j] = j;

        ggml_tensor* gcosV = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, NP, 1, LR);
        ggml_tensor* gsinV = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, NP, 1, LR);
        ggml_tensor* gcosT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, NP, 1, Lt);
        ggml_tensor* gsinT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, NP, 1, Lt);
        ggml_tensor* gmask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, nwin);
        ggml_tensor* gidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, LR);
        ggml_tensor* grev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Lp);
        ggml_tensor* gidxT = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)Lt * nwin);
        addInput(gcosV, cosV.data(), cosV.size() * sizeof(float));
        addInput(gsinV, sinV.data(), sinV.size() * sizeof(float));
        addInput(gcosT, cosT.data(), cosT.size() * sizeof(float));
        addInput(gsinT, sinT.data(), sinT.size() * sizeof(float));
        addInput(gmask, mask.data(), mask.size() * sizeof(ggml_fp16_t));
        addInput(gidx, idx.data(), idx.size() * sizeof(int32_t));
        addInput(grev, rev.data(), rev.size() * sizeof(int32_t));
        addInput(gidxT, idxT.data(), idxT.size() * sizeof(int32_t));

        // --- vid: gather to window layout, then rope with local coords ---
        auto gather_vid = [&](ggml_tensor* x) {
            if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    gather_vid in " + nes(x));
            ggml_tensor* xf = ggml_reshape_2d(ctx, x, V, x->ne[2]);
            return ggml_get_rows(ctx, ggml_cont(ctx, xf), gidx);      // {V, LR}
        };
        ggml_tensor* qv_w = ggml_reshape_3d(ctx, gather_vid(qv), HD, NH, LR);
        ggml_tensor* kv_w = ggml_reshape_3d(ctx, gather_vid(kv), HD, NH, LR);
        ggml_tensor* vv_w = ggml_reshape_3d(ctx, gather_vid(vv_), HD, NH, LR);
        if (b == 0) D0("b0.vv_w", vv_w);
        qv_w = B.rope(qv_w, gcosV, gsinV, NH, LR);
        kv_w = B.rope(kv_w, gcosV, gsinV, NH, LR);
        if (b == 0) { D0("b0.qv_w", qv_w); D0("b0.kv_w", kv_w); }

        // --- text: rope on the (un-repeated) tokens, then repeat per window ---
        ggml_tensor* qt_r = B.rope(qt, gcosT, gsinT, NH, Lt);
        ggml_tensor* kt_r = B.rope(kt, gcosT, gsinT, NH, Lt);
        if (b == 0) { D0("b0.qt_r", qt_r); D0("b0.kt_r", kt_r); }
        auto repeat_txt = [&](ggml_tensor* x) {
            if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    repeat_txt in " + nes(x));
            ggml_tensor* xf = ggml_reshape_2d(ctx, x, V, Lt);
            return ggml_get_rows(ctx, ggml_cont(ctx, xf), gidxT);     // {V, Lt*nwin}
        };
        ggml_tensor* qt_g = repeat_txt(qt_r);
        ggml_tensor* kt_g = repeat_txt(kt_r);
        ggml_tensor* vt_g = repeat_txt(vt);

        // --- concatenate [window vid | text] -> {HD, S, NH, nwin} ---
        //
        // ggml's flash_attn_ext wants q/k/v as {head_dim, n_tokens, n_heads,
        // n_batch} (queries indexed by ne1, heads by ne2) and returns
        // {head_dim, n_heads, n_tokens, n_batch}.
        auto to_tokens = [&](ggml_tensor* x, int64_t L) {
            ggml_tensor* x4 = ggml_reshape_4d(ctx, x, HD, NH, L, nwin);   // {HD,NH,L,w}
            return ggml_cont(ctx, ggml_permute(ctx, x4, 0, 2, 1, 3));     // {HD,L,NH,w}
        };
        ggml_tensor* Q  = ggml_concat(ctx, to_tokens(qv_w, WL), to_tokens(qt_g, Lt), 1);
        ggml_tensor* K  = ggml_concat(ctx, to_tokens(kv_w, WL), to_tokens(kt_g, Lt), 1);
        ggml_tensor* Vv = ggml_concat(ctx, to_tokens(vv_w, WL), to_tokens(vt_g, Lt), 1);
        if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    Q " + nes(Q));

        // The mask is {n_kv, n_q, 1, nwin}.
        // ggml's CUDA flash attention keeps Q in F32 and K/V in F16.
        ggml_tensor* Qh = ggml_cont(ctx, Q);
        ggml_tensor* Kh = ggml_cast(ctx, ggml_cont(ctx, K), GGML_TYPE_F16);
        ggml_tensor* Vh = ggml_cast(ctx, ggml_cont(ctx, Vv), GGML_TYPE_F16);
        float aScale = 1.0f / std::sqrt((float)HD);
        ggml_tensor* attn = ggml_cont(ctx, ggml_flash_attn_ext(ctx, Qh, Kh, Vh, gmask, aScale, 0.0f, 0.0f));
        if (b == 0) D0("b0.attn", attn);
        if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    attn " + nes(attn));
        // attn: {HD, NH, S, nwin}

        // --- vid output: reverse the window partition ---
        ggml_tensor* attn_vid = ggml_view_4d(ctx, attn, HD, NH, WL, nwin,
                                             attn->nb[1], attn->nb[2], attn->nb[3], 0);
        ggml_tensor* av = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_vid), V, LR);  // {V, w*WL+loc}
        if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    av " + nes(av) + " target V=" + std::to_string(V) + " LR=" + std::to_string(LR));
        av = ggml_get_rows(ctx, av, grev);                                          // {V, Lp}
        av = ggml_reshape_2d(ctx, ggml_cont(ctx, av), V, Lp);
        if (b == 0) D0("b0.av", av);

        // --- text output: average the per-window copies ---
        ggml_tensor* attn_txt = ggml_view_4d(ctx, attn, HD, NH, Lt, nwin,
                                             attn->nb[1], attn->nb[2], attn->nb[3],
                                             (size_t)WL * attn->nb[2]);
        ggml_tensor* at = ggml_reshape_3d(ctx, ggml_cont(ctx, attn_txt), V, Lt, nwin);
        if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    at(3d) " + nes(at));
        // average the per-window text copies: bring nwin to ne0 so sum_rows can
        // reduce over it (ggml_permute takes, per source axis, its result axis).
        at = ggml_cont(ctx, ggml_permute(ctx, at, 1, 2, 0, 3));                    // {nwin,V,Lt,1}
        at = ggml_scale(ctx, ggml_sum_rows(ctx, at), 1.0f / (float)nwin);
        if (verbose_) SEEDVR_LOG(LogLevel::Debug, "    at(avg) " + nes(at));
        at = ggml_reshape_2d(ctx, ggml_cont(ctx, at), V, Lt);
        if (b == 0) D0("b0.at", at);

        ggml_tensor* av2 = B.lin(model_, av, p + "attn.proj_out." + vb + ".weight",
                                 p + "attn.proj_out." + vb + ".bias");
        if (b == 0) probe2 = av2;
        if (b == 0) D0("b0.av2", av2);
        vid = ggml_add(ctx, adaOut(av2, "attn", vb), vid);

        if (!is_last) {
            ggml_tensor* at2 = B.lin(model_, at, p + "attn.proj_out." + tb + ".weight",
                                     p + "attn.proj_out." + tb + ".bias");
            if (b == 0) D0("b0.at2", at2);
            txtT = ggml_add(ctx, adaOut(at2, "attn", tb), txtT);
        }

        // --- mlp ---
        ggml_tensor* vm = adaIn(ggml_rms_norm(ctx, vid, C.norm_eps), "mlp", vb);
        ggml_tensor* g = B.lin(model_, vm, p + "mlp." + vb + ".proj_in_gate.weight");
        ggml_tensor* u = B.lin(model_, vm, p + "mlp." + vb + ".proj_in.weight");
        ggml_tensor* h = ggml_mul(ctx, ggml_silu(ctx, g), u);
        if (b == 0) probe3 = h;
        if (b == 0) probe4 = g;
        h = B.lin(model_, h, p + "mlp." + vb + ".proj_out.weight");
        if (b == 0) D0("b0.vmlp", h);
        vid = ggml_add(ctx, adaOut(h, "mlp", vb), vid);

        if (!is_last) {
            ggml_tensor* tm = adaIn(ggml_rms_norm(ctx, txtT, C.norm_eps), "mlp", tb);
            ggml_tensor* g2 = B.lin(model_, tm, p + "mlp." + tb + ".proj_in_gate.weight");
            ggml_tensor* u2 = B.lin(model_, tm, p + "mlp." + tb + ".proj_in.weight");
            ggml_tensor* h2 = ggml_mul(ctx, ggml_silu(ctx, g2), u2);
            h2 = B.lin(model_, h2, p + "mlp." + tb + ".proj_out.weight");
            txtT = ggml_add(ctx, adaOut(h2, "mlp", tb), txtT);
        }
        if (b == 0) probe1 = vid;
        // keep a materialised copy of every block's stream when dumping
        if (dumpAll) {
            DumpInto(dumpOut, "blk" + std::to_string(b) + ".vid", vid);
            DumpInto(dumpOut, "blk" + std::to_string(b) + ".txt", txtT);
        }
    }

    // --- final norm + ada + output projection ---
    vid = ggml_rms_norm(ctx, vid, C.norm_eps);
    {
        ggml_tensor* w = B.f32(model_.get("vid_out_norm.weight"));
        vid = ggml_mul(ctx, ggml_cont(ctx, vid), ggml_reshape_2d(ctx, w, V, 1));
    }
    {
        ggml_tensor* sc = B.f32(model_.get("vid_out_ada.out_scale"));
        ggml_tensor* sh = B.f32(model_.get("vid_out_ada.out_shift"));
        ggml_tensor* scale = ggml_add(ctx, ggml_reshape_2d(ctx, sc, V, 1), aOC);
        ggml_tensor* shift = ggml_add(ctx, ggml_reshape_2d(ctx, sh, V, 1), aOS);
        vid = ggml_add(ctx, ggml_mul(ctx, vid, scale), shift);
    }
    ggml_tensor* outTokens = ggml_cont(ctx, B.lin(model_, vid, "vid_out.proj.weight", "vid_out.proj.bias"));
    ggml_set_name(outTokens, "vid_out");
    ggml_build_forward_expand(gf, outTokens);
    ggml_tensor* probe = nullptr;
    if (const char* pv = std::getenv("SEEDVR2_PROBE")) {
        int k = std::atoi(pv);
        probe = (k == 1) ? probe1 : (k == 2) ? probe2 : (k == 3) ? probe3 : (k == 4) ? probe4
              : (k == 5) ? probe5 : (k == 6) ? probe6 : (k == 7) ? probe7 : (k == 8) ? probe8 : probe0;
        if (probe) ggml_build_forward_expand(gf, probe);
    }

    // ---- execute ----
    if (const char* dbg = std::getenv("SEEDVR2_VERIFY_GRAPH")) {
        (void)dbg;
        int nn = ggml_graph_n_nodes(gf);
        for (int i = 0; i < nn; ++i) {
            ggml_tensor* nd = ggml_graph_node(gf, i);
            for (int s = 0; s < 2; ++s) {
                ggml_tensor* a = nd->src[s];
                if (!a) continue;
                size_t es = ggml_type_size(a->type);
                if (es && (a->nb[0] % es) != 0) {
                    SEEDVR_LOG(LogLevel::Warn,
                               std::string("op ") + ggml_op_name(nd->op) + " src" + std::to_string(s) +
                                   " '" + a->name + "' nb0=" + std::to_string((long long)a->nb[0]) +
                                   " etype=" + std::to_string((long long)a->type) + " ne=" + nes(a));
                }
            }
            if (!ggml_backend_supports_op(model_.backend(), nd)) {
                SEEDVR_LOG(LogLevel::Warn, std::string("UNSUPPORTED op '") + ggml_op_name(nd->op) +
                                               "' type=" + std::to_string((long long)nd->type) +
                                               " src0type=" + std::to_string(nd->src[0] ? (long long)nd->src[0]->type : -1) +
                                               " src1type=" + std::to_string(nd->src[1] ? (long long)nd->src[1]->type : -1) +
                                               " src2type=" + std::to_string(nd->src[2] ? (long long)nd->src[2]->type : -1) +
                                               " name='" + nd->name + "'");
            }
        }
    }
    require(ggml_gallocr_alloc_graph(galloc_, gf), "alloc graph failed");
    for (auto& pd : pending)
        ggml_backend_tensor_set(pd.t, pd.data.data(), 0, pd.data.size());
    {
        enum ggml_status st = ggml_backend_graph_compute(model_.backend(), gf);
        if (st != GGML_STATUS_SUCCESS) {
            // report the first node the backend cannot handle to ease debugging
            std::string bad;
            int nn = ggml_graph_n_nodes(gf);
            for (int i = 0; i < nn && bad.empty(); ++i) {
                ggml_tensor* nd = ggml_graph_node(gf, i);
                if (!ggml_backend_supports_op(model_.backend(), nd))
                    bad = std::string(ggml_op_name(nd->op)) + " (" + (nd->name[0] ? nd->name : "?") + ")";
            }
            throw Error("ggml graph compute failed (status " + std::to_string((int)st) +
                        (bad.empty() ? std::string() : (", unsupported op: " + bad)) + ")");
        }
    }

    std::vector<float> outCpu((size_t)POUT * Lp);
    ggml_backend_tensor_get(outTokens, outCpu.data(), 0, outCpu.size() * sizeof(float));
    if (probe) {
        int64_t n = ggml_nelements(probe);
        std::vector<float> pb((size_t)n);
        ggml_backend_tensor_get(probe, pb.data(), 0, (size_t)n * sizeof(float));
        double mean = 0, mn = 1e30, mx = -1e30;
        for (float v : pb) { mean += v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
        mean /= (double)n;
        double var = 0; for (float v : pb) { double d = v - mean; var += d * d; } var /= (double)n;
        SEEDVR_LOG(LogLevel::Info, "PROBE k=" + std::to_string(*std::getenv("SEEDVR2_PROBE") - '0') +
                                       " ne=" + nes(probe) + " mean=" + std::to_string(mean) +
                                       " std=" + std::to_string(std::sqrt(var)) + " min=" + std::to_string(mn) +
                                       " max=" + std::to_string(mx));
    }
    if (dumpDir) {
        for (auto& d : dumpB0) {
            int64_t n = ggml_nelements(d.second);
            std::vector<float> buf((size_t)n);
            ggml_backend_tensor_get(d.second, buf.data(), 0, (size_t)n * sizeof(float));
            writeBin(d.first + ".f32", buf.data(), buf.size() * sizeof(float));
        }
        for (auto& d : dumpOut) {
            int64_t n = ggml_nelements(d.second);
            std::vector<float> buf((size_t)n);
            ggml_backend_tensor_get(d.second, buf.data(), 0, (size_t)n * sizeof(float));
            writeBin(d.first + ".f32", buf.data(), buf.size() * sizeof(float));
        }
        writeBin("out_tokens.f32", outCpu.data(), outCpu.size() * sizeof(float));
    }

    ggml_gallocr_free(galloc_);
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));
    ggml_free(ctx);

    // ---- depatchify ----
    // outTokens is a ggml tensor {POUT, Lp}: ne0 (the feature index) is the
    // fastest varying axis, so the element for token `tokidx` feature `f` lives
    // at tokidx*POUT + f (NOT f*Lp + tokidx).
    const int OH = Hp * C.patch_h, OW = Wp * C.patch_w;
    torch::Tensor result = torch::empty({ T, OH, OW, C.vid_out_ch }, torch::kFloat32);
    float* rp = result.data_ptr<float>();
    for (int tt = 0; tt < T; ++tt)
        for (int hp = 0; hp < Hp; ++hp)
            for (int wp = 0; wp < Wp; ++wp) {
                int tokidx = (tt * Hp + hp) * Wp + wp;
                for (int hh = 0; hh < C.patch_h; ++hh)
                    for (int ww = 0; ww < C.patch_w; ++ww)
                        for (int cc = 0; cc < C.vid_out_ch; ++cc) {
                            float v = outCpu[(size_t)tokidx * POUT +
                                             (size_t)((hh * C.patch_w + ww) * C.vid_out_ch + cc)];
                            int yy = hp * C.patch_h + hh, xx = wp * C.patch_w + ww;
                            rp[(((size_t)tt * OH + yy) * OW + xx) * C.vid_out_ch + cc] = v;
                        }
            }
    return result;
}

} // namespace seedvr2
