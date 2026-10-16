#include "server/server.h"
#include "frontend_assets.h"
#include "icon_assets.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <map>
#include <thread>

namespace seedvr2 {

static double qd(const httplib::Request& req, const char* key, double def) {
    if (!req.has_param(key)) return def;
    auto v = req.get_param_value(key);
    if (v.empty()) return def;
    try { return std::stod(v); } catch (...) { return def; }
}
static int qi(const httplib::Request& req, const char* key, int def) {
    return (int)qd(req, key, (double)def);
}
static std::string qs(const httplib::Request& req, const char* key, const std::string& def) {
    if (!req.has_param(key)) return def;
    return req.get_param_value(key);
}

// VAE spatial-tiling options (ggml VAE).  Mirrors the Python CLI defaults
// (tile 1024 px, overlap 128 px, output space) and its overlap<tile sanity.
static void apply_tiling(UpscaleParams& p, const httplib::Request& req) {
    p.vae_encode_tiled = qi(req, "vae_encode_tiled", 0) != 0;
    p.vae_decode_tiled = qi(req, "vae_decode_tiled", 0) != 0;
    p.vae_tile_size    = qi(req, "vae_tile_size", 1024);
    p.vae_tile_overlap = qi(req, "vae_tile_overlap", 128);
    if (p.vae_tile_size < 8) p.vae_tile_size = 8;
    if (p.vae_tile_overlap < 0) p.vae_tile_overlap = 0;
    if (p.vae_tile_overlap >= p.vae_tile_size) p.vae_tile_overlap = p.vae_tile_size - 8;
}

// results of the last latent-cache batch (fetched one by one by the UI)
static std::mutex g_res_mu;
static std::map<std::string, std::vector<unsigned char>> g_results;
static int g_res_seq = 0;

Server::Server(Engine& engine, Progress& progress, ServerConfig cfg)
    : engine_(engine), progress_(progress), cfg_(cfg) {}

void Server::run() {
    httplib::Server svr;
    svr.set_payload_max_length(1024ull * 1024 * 1024);
    svr.set_read_timeout(3600, 0);
    svr.set_write_timeout(3600, 0);
    svr.set_idle_interval(0, 200000);
    svr.new_task_queue = [] { return new httplib::ThreadPool(8); };

    // ---------------- UI ----------------
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(kFrontendHtml, "text/html; charset=utf-8");
    });
    svr.Get("/index.html", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(kFrontendHtml, "text/html; charset=utf-8");
    });
    svr.Get("/favicon.ico", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(reinterpret_cast<const char*>(kAppIcon), (size_t)kAppIconSize, "image/x-icon");
    });

    // ---------------- status ----------------
    svr.Get("/api/status", [&](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        j["ready"]   = engine_.ready();
        j["info"]    = engine_.info();
        j["running"] = progress_.running.load();
        j["phase"]   = progress_.phase.load();
        j["item"]    = progress_.item.load();
        j["total"]   = progress_.total.load();
        j["percent"] = progress_.percent.load();
        j["message"] = progress_.get_message();
        j["dit_device"] = engine_.dit_device();
        j["vae_device"] = engine_.vae_device();
        res.set_content(j.dump(), "application/json");
    });

    // ---------------- compute backends ----------------
    svr.Get("/api/devices", [&](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        nlohmann::json devs = nlohmann::json::array();
        nlohmann::json vae_devs = nlohmann::json::array();
        for (const auto& d : list_devices()) {
            nlohmann::json jd = { { "name", d.name }, { "type", d.type }, { "desc", d.description } };
            devs.push_back(jd);
            // The ggml Vulkan backend currently mis-computes one of the VAE ops
            // (DiT is fine), so the VAE is offered on CPU + CUDA only.
            if (d.type == "CPU" || d.name.rfind("CUDA", 0) == 0) vae_devs.push_back(jd);
        }
        j["devices"] = devs;
        j["vae_devices"] = vae_devs;
        j["dit"] = engine_.dit_device();
        j["vae"] = engine_.vae_device();
        res.set_content(j.dump(), "application/json");
    });

    // switch the DiT / VAE compute backend and reload the models in the background
    svr.Post("/api/backend", [&](const httplib::Request& req, httplib::Response& res) {
        if (progress_.running.load()) {
            res.status = 409;
            res.set_content("{\"error\":\"a job is already running\"}", "application/json");
            return;
        }
        const std::string dit = qs(req, "dit", engine_.dit_device());
        const std::string vae = qs(req, "vae", engine_.vae_device());
        const std::string rd = resolve_device(dit), rv = resolve_device(vae);
        if (rd.empty() || rv.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"unknown device\"}", "application/json");
            return;
        }
        res.set_content("{\"ok\":true,\"dit\":\"" + rd + "\",\"vae\":\"" + rv + "\"}",
                        "application/json");
        // persist the choice so the next start uses the same backends
        try {
            std::string bj = cfg_.root.empty() ? std::string("backend.json") : (cfg_.root + "/backend.json");
            std::ofstream bf(bj, std::ios::trunc);
            bf << nlohmann::json{ { "dit", rd }, { "vae", rv } }.dump(2) << "\n";
        } catch (const std::exception& e) {
            SEEDVR_LOG(LogLevel::Warn, std::string("could not write backend.json: ") + e.what());
        }
        std::thread([&, dit, vae] {
            try {
                engine_.reinit(dit, vae, progress_);
            } catch (const std::exception& e) {
                SEEDVR_LOG(LogLevel::Error, std::string("backend switch failed: ") + e.what());
            }
        }).detach();
    });

    svr.Get("/api/log", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(Logger::instance().text(), "text/plain; charset=utf-8");
    });

    svr.Post("/api/clearLog", [&](const httplib::Request&, httplib::Response& res) {
        Logger::instance().clear();
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Post("/api/cancel", [&](const httplib::Request&, httplib::Response& res) {
        progress_.cancel = true;
        res.set_content("{\"ok\":true}", "application/json");
    });

    // ---------------- inference ----------------
    svr.Post("/api/upscale", [&](const httplib::Request& req, httplib::Response& res) {
        if (!engine_.ready()) {
            res.status = 503;
            res.set_content("{\"error\":\"engine not ready\"}", "application/json");
            return;
        }
        if (progress_.running.exchange(true)) {
            res.status = 409;
            res.set_content("{\"error\":\"a job is already running\"}", "application/json");
            return;
        }
        UpscaleParams p;
        p.resolution         = qi(req, "resolution", 1080);
        p.max_resolution     = qi(req, "max_resolution", 0);
        p.seed               = qi(req, "seed", 42);
        p.steps              = qi(req, "steps", 1);
        p.cfg_scale          = qd(req, "cfg_scale", 1.0);
        p.cfg_rescale        = qd(req, "cfg_rescale", 0.0);
        p.color_correction   = qs(req, "color_correction", "lab");
        apply_tiling(p, req);

        progress_.cancel = false;
        progress_.done = false;
        progress_.failed = false;

        try {
            std::vector<unsigned char> body(req.body.begin(), req.body.end());
            SEEDVR_LOG(LogLevel::Info, "job start: " + std::to_string(body.size()) + " bytes, res=" +
                                           std::to_string(p.resolution) + ", seed=" + std::to_string(p.seed) +
                                           ", steps=" + std::to_string(p.steps) + ", cc=" + p.color_correction);
            std::vector<unsigned char> png = engine_.upscale(body, p, progress_);
            res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
            res.set_header("X-Seedvr-Ok", "1");
            progress_.done = true;
            SEEDVR_LOG(LogLevel::Info, "job done: " + std::to_string(png.size()) + " bytes");
        } catch (const std::exception& e) {
            progress_.failed = true;
            progress_.set_message(std::string("error: ") + e.what());
            SEEDVR_LOG(LogLevel::Error, std::string("job failed: ") + e.what());
            res.status = 500;
            nlohmann::json j; j["error"] = e.what();
            res.set_content(j.dump(), "application/json");
        }
        progress_.running = false;
    });

    // ---------------- batch with latent cache ----------------
    // multipart body (每个文件字段名都是 "files") + 参数在 query string。流程：
    // 全部 VAE 编码 -> 全部 DiT 推理 -> 全部 VAE 解码，每个模型整个批次只上传一次。
    svr.Post("/api/upscaleBatch", [&](const httplib::Request& req, httplib::Response& res) {
        if (!engine_.ready()) {
            res.status = 503;
            res.set_content("{\"error\":\"engine not ready\"}", "application/json");
            return;
        }
        if (progress_.running.exchange(true)) {
            res.status = 409;
            res.set_content("{\"error\":\"a job is already running\"}", "application/json");
            return;
        }
        UpscaleParams base;
        base.resolution         = qi(req, "resolution", 1080);
        base.max_resolution     = qi(req, "max_resolution", 0);
        base.seed               = qi(req, "seed", 42);
        base.steps              = qi(req, "steps", 1);
        base.cfg_scale          = qd(req, "cfg_scale", 1.0);
        base.cfg_rescale        = qd(req, "cfg_rescale", 0.0);
        base.color_correction   = qs(req, "color_correction", "lab");
        apply_tiling(base, req);
        // optional per-item shortest edge: "resolutions=512,768,512"
        std::vector<int> resolutions;
        {
            std::string rl = qs(req, "resolutions", "");
            size_t p = 0;
            while (p < rl.size()) {
                size_t q = rl.find(',', p);
                std::string tok = rl.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (!tok.empty()) { try { resolutions.push_back(std::stoi(tok)); } catch (...) {} }
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }

        std::vector<std::vector<unsigned char>> inputs;
        std::vector<std::string> names;
        for (const auto& kv : req.form.files) {
            if (kv.first != "files") continue;
            const auto& fd = kv.second;
            inputs.emplace_back(fd.content.begin(), fd.content.end());
            names.push_back(fd.filename.empty() ? ("image" + std::to_string(inputs.size()) + ".png")
                                                : fd.filename);
        }
        if (inputs.empty()) {
            progress_.running = false;
            res.status = 400;
            res.set_content("{\"error\":\"no files\"}", "application/json");
            return;
        }
        std::vector<UpscaleParams> params(inputs.size(), base);
        for (size_t i = 0; i < params.size() && i < resolutions.size(); ++i)
            if (resolutions[i] > 0) params[i].resolution = resolutions[i];

        progress_.cancel = false;
        progress_.done = false;
        progress_.failed = false;
        SEEDVR_LOG(LogLevel::Info, "latent-cache batch start: " + std::to_string(inputs.size()) +
                                       " images, res=" + std::to_string(base.resolution) +
                                       ", seed=" + std::to_string(base.seed));
        try {
            auto outs = engine_.upscale_batch(inputs, params, progress_);
            nlohmann::json j;
            j["items"] = nlohmann::json::array();
            {
                std::lock_guard<std::mutex> lk(g_res_mu);
                g_results.clear();
                for (size_t i = 0; i < outs.size(); ++i) {
                    std::string id;
                    bool ok = !outs[i].empty();
                    if (ok) {
                        id = "r" + std::to_string(g_res_seq++);
                        g_results[id] = std::move(outs[i]);
                    }
                    j["items"].push_back({ { "id", id }, { "name", names[i] }, { "ok", ok } });
                }
            }
            res.set_content(j.dump(), "application/json");
            progress_.done = true;
            SEEDVR_LOG(LogLevel::Info, "latent-cache batch done");
        } catch (const std::exception& e) {
            progress_.failed = true;
            progress_.set_message(std::string("error: ") + e.what());
            SEEDVR_LOG(LogLevel::Error, std::string("latent-cache batch failed: ") + e.what());
            res.status = 500;
            nlohmann::json j;
            j["error"] = e.what();
            res.set_content(j.dump(), "application/json");
        }
        progress_.running = false;
    });

    svr.Get("/api/result", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string id = qs(req, "id", "");
        std::lock_guard<std::mutex> lk(g_res_mu);
        auto it = g_results.find(id);
        if (id.empty() || it == g_results.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown result id\"}", "application/json");
            return;
        }
        res.set_content(reinterpret_cast<const char*>(it->second.data()), it->second.size(), "image/png");
    });

    if (cfg_.open_browser) {
        std::string url = "http://" + cfg_.host + ":" + std::to_string(cfg_.port) + "/";
#ifdef _WIN32
        std::string cmd = "start \"\" \"" + url + "\"";
        std::system(cmd.c_str());
        (void)url;
#endif
    }

    SEEDVR_LOG(LogLevel::Info, "listening on http://" + cfg_.host + ":" + std::to_string(cfg_.port) + "/");
    svr.listen(cfg_.host, cfg_.port);
}

} // namespace seedvr2
