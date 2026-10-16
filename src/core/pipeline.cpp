#include "core/pipeline.h"
#include "core/image_io.h"
#include "core/transforms.h"
#include "core/color_fix.h"
#include "core/schedule.h"

#include <torch/torch.h>
#include <fstream>
#include <thread>
#include <chrono>

namespace seedvr2 {

static constexpr int   PATCH_T = 1, PATCH_H = 2, PATCH_W = 2;
static constexpr int   LATENT_CH = 16;
static constexpr float VAE_SCALE = 0.9152f;

// wall-clock helper for the per-phase logs
struct PhaseTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};
static void log_ms(const char* what, double ms) {
    SEEDVR_LOG(LogLevel::Info, std::string(what) + ": " + std::to_string((int)(ms + 0.5)) + " ms");
}

void Engine::init(const EngineConfig& cfg, Progress& prog) {
    init_locked(cfg, prog);
}

void Engine::reinit(const std::string& dit_device, const std::string& vae_device, Progress& prog) {
    std::lock_guard<std::mutex> lock(mu_);
    EngineConfig c = cfg_;
    c.dit_device = dit_device;
    c.vae_device = vae_device;
    ready_ = false;
    SEEDVR_LOG(LogLevel::Info, "switching backends: DiT -> " + resolve_device(dit_device) +
                                   ", VAE -> " + resolve_device(vae_device));
    dit_.free();
    vae_g_.free();
    init_locked(c, prog);
}

void Engine::ensure_vae() {
    if (vae_g_.ready()) return;
    try {
        vae_g_.load(cfg_.vae_path, cfg_.vae_device);
    } catch (const std::exception& e) {
        if (cfg_.vae_device == "CPU") throw;
        SEEDVR_LOG(LogLevel::Warn, std::string("VAE load on '") + cfg_.vae_device + "' failed (" +
                                       e.what() + ") - falling back to the CPU backend");
        cfg_.vae_device = "CPU";
        vae_g_.load(cfg_.vae_path, "CPU");
    }
}

void Engine::ensure_dit() {
    if (!dit_.loaded()) dit_.load(cfg_.gguf_path, cfg_.dit_device);
}

void Engine::release_models() {
    if (!cfg_.staged_load) return;
    if (dit_.loaded()) dit_.free();
    if (vae_g_.ready()) vae_g_.free();
}

void Engine::init_locked(const EngineConfig& cfg, Progress& prog) {
    cfg_ = cfg;
    dit_dev_ = resolve_device(cfg.dit_device);
    vae_dev_ = resolve_device(cfg.vae_device);
    ready_ = false;
    prog.set_message("loading DiT (ggml)...");
    {
        PhaseTimer t;
        dit_.load(cfg.gguf_path, cfg.dit_device);
        log_ms("load DiT", t.ms());
    }
    // the VAE is always the quantised ggml model (.gguf, no libtorch)
    {
        PhaseTimer t;
        prog.set_message("loading VAE (ggml)...");
        ensure_vae();
        vae_dev_ = resolve_device(cfg_.vae_device);
        log_ms("load VAE", t.ms());
    }
    prog.set_message("loading text embeddings...");
    {
        PhaseTimer t;
        text_ = load_text_embeddings("cpu");
        log_ms("load text embeddings", t.ms());
    }

    int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
    torch::set_num_threads(nthreads);

    ready_ = true;
    info_ = std::string("DiT: ") + dit_.backend_name() +
            " | VAE: " + vae_g_.info() +
            " | " + (cfg.staged_load ? "staged" : "resident") +
            " | threads = " + std::to_string(nthreads);
    SEEDVR_LOG(LogLevel::Info, info_);
    // only one model stays resident on the device (see EngineConfig::staged_load)
    release_models();
    if (cfg.staged_load) SEEDVR_LOG(LogLevel::Info, "staged loading: models unloaded, will be loaded per phase");
}

std::vector<unsigned char> Engine::vae_roundtrip(const std::vector<unsigned char>& input,
                                            const UpscaleParams& params, Progress& prog) {
    std::lock_guard<std::mutex> lock(mu_);
    require(ready_, "engine not initialised");
    torch::Tensor img = decode_image(input);
    torch::Tensor video = img.unsqueeze(0);
    TransformResult tr = prepare_video(video, params.resolution, params.max_resolution);
    torch::Tensor x_in = tr.cthw.unsqueeze(0).to(torch::kFloat32);
    prog.set_message("vae encode");
    ensure_vae();
    PhaseTimer t_e;
    torch::Tensor lat = vae_encode(x_in, (double)params.seed + 1000000.0, params);
    log_ms("vae encode", t_e.ms());
    {
        auto m = lat.mean(); auto s = lat.std();
        SEEDVR_LOG(LogLevel::Info, "latent stats: mean=" + std::to_string(m.item<float>()) +
                                       " std=" + std::to_string(s.item<float>()));
    }
    prog.set_message("vae decode");
    PhaseTimer t_d;
    torch::Tensor dec = vae_decode(lat, params);
    log_ms("vae decode", t_d.ms());
    SEEDVR_LOG(LogLevel::Info, "vae decode out: [" + std::to_string(dec.size(0)) + "," + std::to_string(dec.size(1)) +
                                   "," + std::to_string(dec.size(2)) + "," + std::to_string(dec.size(3)) + "," +
                                   std::to_string(dec.size(4)) + "]");
    if (const char* dd = std::getenv("SEEDVR2_DUMP")) {
        auto save = [&](const std::string& nm, const torch::Tensor& t) {
            torch::Tensor c = t.to(torch::kFloat32).contiguous();
            std::ofstream f(std::string(dd) + "/" + nm, std::ios::binary);
            f.write((const char*)c.data_ptr<float>(), (std::streamsize)(c.numel() * 4));
        };
        save("rt_lat.f32", lat);
        save("rt_dec.f32", dec);
        SEEDVR_LOG(LogLevel::Info, "dumped rt_lat/rt_dec to " + std::string(dd));
    }
    torch::Tensor sample = dec.select(2, 0);
    sample = sample.narrow(2, 0, tr.true_h).narrow(3, 0, tr.true_w);
    sample = torch::clamp(sample, -1.0, 1.0).mul(0.5).add(0.5);
    torch::Tensor out01 = sample.squeeze(0).permute({ 1, 2, 0 }).contiguous();
    prog.set_message("done");
    return encode_image(out01, true);
}

// ---------------------------------------------------------------------------
// The pipeline is split into three stages so that a whole batch can run
//     all VAE encodes -> all DiT passes -> all VAE decodes
// with the device holding only the model of the current stage ("latent cache").
// ---------------------------------------------------------------------------
struct Engine::Stage {
    TransformResult tr;              // decoded + resized input
    torch::Tensor latent;            // [T,h,w,16] conditioning (scaled)
    torch::Tensor cond;              // [T,h,w,17]
    torch::Tensor x_t;               // [T,h,w,16] start noise
    torch::Tensor x0;                // [T,h,w,16] DiT output
    int T = 1, lat_h = 0, lat_w = 0;
};

void Engine::stage_encode(const std::vector<unsigned char>& input, const UpscaleParams& params,
                          Progress& prog, Stage& st) {
    prog.set_message("pre-processing image");
    const int vs = 8;                            // spatial downsample factor
    torch::Tensor img = decode_image(input);     // [H,W,3] in [0,1]
    torch::Tensor video = img.unsqueeze(0);      // [T,H,W,3]
    st.tr = prepare_video(video, params.resolution, params.max_resolution);
    st.lat_h = st.tr.padded_h / vs;
    st.lat_w = st.tr.padded_w / vs;
    SEEDVR_LOG(LogLevel::Info, "input " + std::to_string((int)img.size(1)) + "x" +
                                   std::to_string((int)img.size(0)) +
                                   " -> target " + std::to_string(st.tr.true_w) + "x" +
                                   std::to_string(st.tr.true_h) + " (padded " +
                                   std::to_string(st.tr.padded_w) + "x" + std::to_string(st.tr.padded_h) + ")");

    prog.set_message("VAE encoding");
    ensure_vae();
    PhaseTimer t;
    torch::Tensor x_in = st.tr.cthw.unsqueeze(0).to(torch::kFloat32);   // [1,3,T,H,W]
    torch::Tensor lat = vae_encode(x_in, (double)params.seed + 1000000.0, params);
    log_ms("vae encode", t.ms());
    st.latent = lat.permute({ 0, 2, 3, 4, 1 }).squeeze(0) * VAE_SCALE;  // [T,h,w,16]
    {
        auto m = st.latent.mean(); auto s = st.latent.std();
        SEEDVR_LOG(LogLevel::Info, "latent stats: mean=" + std::to_string(m.item<float>()) +
                                       " std=" + std::to_string(s.item<float>()));
    }
    // condition: the blurred latent plus a "task = sr" flag channel
    st.cond = torch::zeros({ st.T, st.lat_h, st.lat_w, LATENT_CH + 1 }, st.latent.options());
    st.cond.slice(3, 0, LATENT_CH).copy_(st.latent);
    st.cond.slice(3, LATENT_CH, LATENT_CH + 1).fill_(1.0);
}

void Engine::stage_dit(const UpscaleParams& params, Progress& prog, Stage& st) {
    prog.phase = 2;
    LerpSchedule sched; sched.T = 1000.0;
    int steps = std::max(1, params.steps);
    SamplingTimesteps ts = uniform_trailing_timesteps(sched.T, steps, 1.0);
    EulerSampler sampler(sched, ts);

    torch::manual_seed((int64_t)params.seed);
    torch::Tensor x_t = torch::randn({ st.T, st.lat_h, st.lat_w, LATENT_CH }, st.latent.options());

    bool  hasTOverride = false;
    float tOverride = 0.0f;
    if (const char* tv = std::getenv("SEEDVR2_T")) {
        tOverride = (float)std::atof(tv);
        hasTOverride = true;
        ts.timesteps = torch::tensor({ (double)tOverride }, torch::kFloat32);
        sampler = EulerSampler(sched, ts);
    }
    const bool xtIsLatent = (std::getenv("SEEDVR2_XT_LATENT") != nullptr);
    if (xtIsLatent) x_t = st.latent.clone();
    if (hasTOverride || xtIsLatent)
        SEEDVR_LOG(LogLevel::Info, "DEBUG: t=" + std::to_string(tOverride) +
                                       " xt=" + (xtIsLatent ? "latent" : "noise"));

    const float* posTxt = text_.pos.contiguous().data_ptr<float>();
    const float* negTxt = text_.neg.contiguous().data_ptr<float>();
    const int Lt = (int)text_.pos.size(0);
    const bool useCfg = std::abs(params.cfg_scale - 1.0) > 1e-6;
    torch::Tensor last_pred_;

    auto run_dit = [&](const torch::Tensor& x, float t, const float* txt) {
        torch::Tensor vin = torch::cat({ x, st.cond }, -1).contiguous();   // [T,h,w,33]
        auto out = dit_.forward(vin.data_ptr<float>(), st.T, st.lat_h, st.lat_w, txt, Lt, t);
        last_pred_ = out;
        if (std::getenv("SEEDVR2_CORR")) {
            auto corr = [](const torch::Tensor& p, const torch::Tensor& q) {
                torch::Tensor a = p.reshape({ -1 }).to(torch::kFloat64);
                torch::Tensor b = q.reshape({ -1 }).to(torch::kFloat64);
                double am = a.mean().item<double>(), bm = b.mean().item<double>();
                double d = a.std().item<double>() * b.std().item<double>();
                return ((a - am) * (b - bm)).mean().item<double>() / std::max(1e-12, d);
            };
            SEEDVR_LOG(LogLevel::Info,
                       "  pred: std=" + std::to_string(out.std().item<double>()) +
                       "  corr(pred,x_t)=" + std::to_string(corr(out, x)) +
                       "  corr(pred,cond)=" + std::to_string(corr(out, st.cond.slice(3, 0, LATENT_CH))) +
                       "  corr(x_t,cond)=" + std::to_string(corr(x, st.cond.slice(3, 0, LATENT_CH))) +
                       "  spatialStd(pred)=" + std::to_string(out.reshape({ -1, LATENT_CH }).std(0).mean().item<double>()));
        }
        if (hasTOverride || xtIsLatent) {
            auto m = out.mean(); auto s = out.std();
            SEEDVR_LOG(LogLevel::Info, "DEBUG pred: mean=" + std::to_string(m.item<float>()) +
                                           " std=" + std::to_string(s.item<float>()) +
                                           " absmax=" + std::to_string(out.abs().max().item<float>()));
        }
        return out;
    };

    prog.set_message("DiT inference");
    ensure_dit();
    PhaseTimer t_dit;
    torch::Tensor x0 = sampler.sample(
        x_t,
        [&](const torch::Tensor& x, const torch::Tensor& t, int i) -> torch::Tensor {
            float tv = hasTOverride ? tOverride : t.item<float>();
            prog.set_message(std::string("DiT step ") + (i == 0 ? "1" : "2"));
            torch::Tensor pos = run_dit(x, tv, posTxt);
            if (!useCfg) return pos;
            torch::Tensor neg = run_dit(x, tv, negTxt);
            double scale = params.cfg_scale;
            if (((double)(i + 1) / (double)ts.size()) > 1.0) scale = 1.0;
            return classifier_free_guidance(pos, neg, scale, params.cfg_rescale);
        },
        [&](int cur, int total) {
            prog.item = cur; prog.total = total;
        });
    log_ms("DiT", t_dit.ms());
    release_models();            // staged: unload the DiT weights before decoding

    {
        // sanity: an SR model must keep the denoised latent strongly correlated with
        // the conditioning (blurred) latent; a near-zero correlation means the
        // conditioning path through the DiT is broken.
        torch::Tensor a = x0.reshape({ -1 }).to(torch::kFloat64);
        torch::Tensor b = st.cond.slice(3, 0, LATENT_CH).reshape({ -1 }).to(torch::kFloat64);
        torch::Tensor p = x_t.reshape({ -1 }).to(torch::kFloat64);
        double am = a.mean().item<double>(), bm = b.mean().item<double>(), pm = p.mean().item<double>();
        double bs = b.std().item<double>();
        double c1 = ((a - am) * (b - bm)).mean().item<double>() / std::max(1e-12, a.std().item<double>() * bs);
        double c2 = ((p - pm) * (b - bm)).mean().item<double>() / std::max(1e-12, p.std().item<double>() * bs);
        SEEDVR_LOG(LogLevel::Info, "corr(denoised_x0, cond)=" + std::to_string(c1) +
                                       "   corr(noise_x_t, cond)=" + std::to_string(c2));
    }
    st.x_t = x_t;
    st.x0 = x0;
}

std::vector<unsigned char> Engine::stage_decode(const UpscaleParams& params, Progress& prog, Stage& st) {
    prog.phase = 3;
    prog.set_message("VAE decoding");
    {
        auto m = st.x0.mean(); auto s = st.x0.std();
        auto mn = st.x0.min(); auto mx = st.x0.max();
        SEEDVR_LOG(LogLevel::Info, "DiT output stats: mean=" + std::to_string(m.item<float>()) +
                                       " std=" + std::to_string(s.item<float>()) +
                                       " min=" + std::to_string(mn.item<float>()) +
                                       " max=" + std::to_string(mx.item<float>()));
    }
    if (const char* dd = std::getenv("SEEDVR2_DUMP")) {
        auto save = [&](const std::string& nm, const torch::Tensor& t) {
            torch::Tensor c = t.to(torch::kFloat32).contiguous();
            std::ofstream f(std::string(dd) + "/" + nm, std::ios::binary);
            f.write((const char*)c.data_ptr<float>(), (std::streamsize)(c.numel() * 4));
        };
        save("cpp_xt.f32", st.x_t);
        save("cpp_x0.f32", st.x0);
        SEEDVR_LOG(LogLevel::Info, "dumped cpp_xt/cpp_x0 to " + std::string(dd));
    }
    torch::Tensor upscaled = st.x0 / VAE_SCALE;                       // [T,h,w,16]
    torch::Tensor z = upscaled.permute({ 3, 0, 1, 2 }).unsqueeze(0).contiguous();  // [1,16,T,h,w]
    ensure_vae();
    PhaseTimer t_dec;
    torch::Tensor dec = vae_decode(z, params);                                // [1,3,T,H,W]
    log_ms("vae decode", t_dec.ms());
    SEEDVR_LOG(LogLevel::Info, "vae decode out: [" + std::to_string(dec.size(0)) + "," + std::to_string(dec.size(1)) +
                                   "," + std::to_string(dec.size(2)) + "," + std::to_string(dec.size(3)) + "," +
                                   std::to_string(dec.size(4)) + "]  target " + std::to_string(st.tr.true_w) + "x" +
                                   std::to_string(st.tr.true_h));
    torch::Tensor sample = dec.select(2, 0);                          // [1,3,H,W]  (T == 1)
    sample = sample.narrow(2, 0, st.tr.true_h).narrow(3, 0, st.tr.true_w);

    prog.phase = 4;
    prog.set_message("colour correction: " + params.color_correction);
    torch::Tensor style = st.tr.cthw.permute({ 1, 0, 2, 3 }).contiguous();  // [T,C,H,W]
    style = style.narrow(2, 0, st.tr.true_h).narrow(3, 0, st.tr.true_w);
    torch::Tensor corrected = apply_color_correction(params.color_correction, sample, style);
    corrected = torch::clamp(corrected, -1.0, 1.0).mul(0.5).add(0.5);    // [-1,1] -> [0,1]
    torch::Tensor out01 = corrected.squeeze(0).permute({ 1, 2, 0 }).contiguous();  // [H,W,3]
    prog.phase = 0;
    prog.percent = 100;
    prog.set_message("done");
    return encode_image(out01, /*png=*/true);
}

std::vector<unsigned char> Engine::upscale(const std::vector<unsigned char>& input,
                                           const UpscaleParams& params, Progress& prog) {
    std::lock_guard<std::mutex> lock(mu_);
    require(ready_, "engine not initialised");
    prog.phase = 1;
    Stage st;
    stage_encode(input, params, prog, st);
    release_models();
    stage_dit(params, prog, st);
    return stage_decode(params, prog, st);
}

std::vector<std::vector<unsigned char>> Engine::upscale_batch(
        const std::vector<std::vector<unsigned char>>& inputs,
        const std::vector<UpscaleParams>& params, Progress& prog) {
    std::lock_guard<std::mutex> lock(mu_);
    require(ready_, "engine not initialised");
    const int n = (int)inputs.size();
    std::vector<Stage> st((size_t)n);
    std::vector<std::vector<unsigned char>> out((size_t)n);
    prog.total = n * 3;
    prog.item = 0;
    PhaseTimer t_all;

    // ---- phase 1/3: VAE encode, VAE stays on the device ----
    prog.phase = 1;
    for (int i = 0; i < n && !prog.cancel; ++i) {
        prog.item = i;
        prog.percent = (int)(100.0 * i / std::max(1, (int)prog.total.load()));
        prog.set_message("latent cache 1/3 encode " + std::to_string(i + 1) + "/" + std::to_string(n));
        stage_encode(inputs[(size_t)i], params[(size_t)i], prog, st[(size_t)i]);
    }
    release_models();

    // ---- phase 2/3: DiT, DiT stays on the device ----
    prog.phase = 2;
    if (!prog.cancel) {
        ensure_dit();
        for (int i = 0; i < n && !prog.cancel; ++i) {
            prog.item = n + i;
            prog.percent = (int)(100.0 * (n + i) / std::max(1, (int)prog.total.load()));
            prog.set_message("latent cache 2/3 DiT " + std::to_string(i + 1) + "/" + std::to_string(n));
            stage_dit(params[(size_t)i], prog, st[(size_t)i]);
        }
        release_models();
    }

    // ---- phase 3/3: VAE decode (+ colour correction) ----
    prog.phase = 3;
    for (int i = 0; i < n && !prog.cancel; ++i) {
        prog.item = 2 * n + i;
        prog.percent = (int)(100.0 * (2 * n + i) / std::max(1, (int)prog.total.load()));
        prog.set_message("latent cache 3/3 decode " + std::to_string(i + 1) + "/" + std::to_string(n));
        out[(size_t)i] = stage_decode(params[(size_t)i], prog, st[(size_t)i]);
    }
    release_models();
    log_ms("latent-cache batch (encode+DiT+decode)", t_all.ms());
    prog.phase = 0;
    prog.percent = 100;
    prog.set_message(prog.cancel ? "cancelled" : "done");
    return out;
}

} // namespace seedvr2
