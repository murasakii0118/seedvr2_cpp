// SeedVR2 C++ - entry point
#include "core/common.h"
#include "core/pipeline.h"
#include "server/server.h"

#include <filesystem>
#include <fstream>
#include "nlohmann/json.hpp"
#include <windows.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <thread>
#include <string>
#include <cstring>

using namespace seedvr2;
namespace fs = std::filesystem;

static std::string exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    return fs::path(std::wstring(buf, n)).parent_path().string();
}

int main(int argc, char** argv) {
    ServerConfig srvCfg;
    EngineConfig engCfg;
    // set when --dit-device/--vae-device are given (then backend.json is ignored)
    bool cliDeviceOverride = false;
    std::string repoRoot;
    std::string cliIn, cliOut;
    UpscaleParams cliParams;
    bool cliMode = false;
    bool cliRoundtrip = false;
    bool cliSelfTest = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--port") srvCfg.port = std::stoi(next());
        else if (a == "--host") srvCfg.host = next();
        else if (a == "--device") engCfg.dit_device = "CUDA" + next();   // legacy: CUDA index
        else if (a == "--dit-device") { engCfg.dit_device = next(); cliDeviceOverride = true; }
        else if (a == "--gguf") engCfg.gguf_path = next();
        else if (a == "--vae") engCfg.vae_path = next();
        else if (a == "--vae-device") { engCfg.vae_device = next(); cliDeviceOverride = true; }
        else if (a == "--no-staged-load") engCfg.staged_load = false;
        else if (a == "--list-devices") {
            for (const auto& d : seedvr2::list_devices())
                std::printf("%-10s %-6s %s\n", d.name.c_str(), d.type.c_str(), d.description.c_str());
            return 0;
        }
        else if (a == "--root") repoRoot = next();
        else if (a == "--no-browser") srvCfg.open_browser = false;
        else if (a == "--input") { cliIn = next(); cliMode = true; }
        else if (a == "--output") cliOut = next();
        else if (a == "--resolution") cliParams.resolution = std::stoi(next());
        else if (a == "--maxres") cliParams.max_resolution = std::stoi(next());
        else if (a == "--seed") cliParams.seed = std::stoi(next());
        else if (a == "--steps") cliParams.steps = std::stoi(next());
        else if (a == "--cfg") cliParams.cfg_scale = std::stod(next());
        else if (a == "--cc") cliParams.color_correction = next();
        else if (a == "--vae-encode-tiled") cliParams.vae_encode_tiled = true;
        else if (a == "--vae-decode-tiled") cliParams.vae_decode_tiled = true;
        else if (a == "--vae-tiled") { cliParams.vae_encode_tiled = cliParams.vae_decode_tiled = true; }
        else if (a == "--vae-tile-size") cliParams.vae_tile_size = std::stoi(next());
        else if (a == "--vae-tile-overlap") cliParams.vae_tile_overlap = std::stoi(next());
        else if (a == "--selftest") { cliSelfTest = true; }
        else if (a == "--vae-roundtrip") { cliRoundtrip = true; }
        else if (a == "--help" || a == "-h") {
            std::printf(
                "seedvr2_cpp [options]\n"
                "  --root DIR      repository root (default: parent of the executable, or CWD)\n"
                "  --gguf PATH     DiT gguf model\n"
                "  --vae PATH      quantised VAE .gguf\n"
                "  --vae-device D  device for the ggml VAE: CUDA0 (default) / CPU / Vulkan0\n"
                "  --dit-device D  DiT backend: CUDA0 / Vulkan0 / CPU (or cuda|vulkan|cpu)\n"
                "  --device N      shorthand for --dit-device CUDA N\n"
                "  --list-devices  print the available ggml devices and exit\n"
                "  --no-staged-load keep DiT and VAE resident at the same time\n"
                "  --host ADDR     bind address (default 127.0.0.1)\n"
                "  --port N        port (default 8188)\n"
                "  --no-browser    do not open the browser\n"
                "  --input FILE    run a single inference and exit (CLI mode)\n"
                "  --vae-roundtrip with --input: VAE encode+decode only (no DiT)\n"
                "  --output FILE   output image path (CLI mode)\n"
                "  --resolution N  target shortest edge (default 1080)\n"
                "  --maxres N      maximum edge, 0 = unlimited\n"
                "  --seed N --steps N --cfg F --cc NAME\n"
                "  --vae-encode-tiled          VAE encode in spatial tiles (high res, less VRAM)\n"
                "  --vae-decode-tiled          VAE decode in spatial tiles (high res, less VRAM)\n"
                "  --vae-tiled                 shorthand for both encode+decode tiling\n"
                "  --vae-tile-size N           tile size in output pixels (default 1024)\n"
                "  --vae-tile-overlap N        tile overlap in output pixels (default 128)\n");
            return 0;
        }
    }

    // VAE tiling sanity (mirrors the Python CLI check: overlap < tile size)
    if (cliParams.vae_tile_size < 8) cliParams.vae_tile_size = 8;
    if (cliParams.vae_tile_overlap < 0) cliParams.vae_tile_overlap = 0;
    if (cliParams.vae_tile_overlap >= cliParams.vae_tile_size) {
        SEEDVR_LOG(LogLevel::Warn, "VAE tile overlap >= tile size; clamping overlap to size-8");
        cliParams.vae_tile_overlap = cliParams.vae_tile_size - 8;
    }

    // resolve the repository root: search upwards for models/ and seedvr2_videoupscaler/
    if (repoRoot.empty()) {
        std::vector<fs::path> cands = { fs::current_path(), exe_dir() };
        for (auto& c : cands) {
            fs::path d = c;
            for (int up = 0; up < 4 && !d.empty(); ++up) {
                if (fs::exists(d / "models") && fs::exists(d / "seedvr2_videoupscaler")) {
                    repoRoot = d.string();
                    break;
                }
                d = d.parent_path();
            }
            if (!repoRoot.empty()) break;
        }
    }
    require(!repoRoot.empty(), "cannot locate repository root (expected ./models and ./seedvr2_videoupscaler). Use --root.");
    srvCfg.root = repoRoot;

    fs::path root(repoRoot);
    // remember the backends picked in the UI (written by /api/backend)
    if (!cliDeviceOverride) {
        std::ifstream bf((root / "backend.json").string());
        if (bf) {
            try {
                auto j = nlohmann::json::parse(bf);
                if (j.contains("dit") && j["dit"].is_string()) engCfg.dit_device = j["dit"];
                if (j.contains("vae") && j["vae"].is_string()) engCfg.vae_device = j["vae"];
                SEEDVR_LOG(LogLevel::Info, "backend.json: DiT=" + engCfg.dit_device + " VAE=" + engCfg.vae_device);
            } catch (const std::exception&) {
                SEEDVR_LOG(LogLevel::Warn, "ignoring malformed backend.json");
            }
        }
    }
    if (engCfg.gguf_path.empty()) engCfg.gguf_path = (root / "models" / "seedvr2_ema_3b-Q8_0.gguf").string();
    if (engCfg.vae_path.empty()) {
        // default: the fp16 ggml VAE
        engCfg.vae_path = (root / "models" / "ema_vae_fp16.gguf").string();
    }

    SEEDVR_LOG(LogLevel::Info, "repo root: " + repoRoot);
    SEEDVR_LOG(LogLevel::Info, "gguf: " + engCfg.gguf_path);
    SEEDVR_LOG(LogLevel::Info, "vae : " + engCfg.vae_path);

    Progress progress;
    Engine engine;

    // ---------------- CLI mode: one image, then exit ----------------
    if (cliMode) {
        try {
            engine.init(engCfg, progress);
            std::ifstream in(cliIn, std::ios::binary);
            require(in.good(), "cannot open input: " + cliIn);
            std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                             std::istreambuf_iterator<char>());
            SEEDVR_LOG(LogLevel::Info, "CLI input " + cliIn + " (" + std::to_string(bytes.size()) +
                                           " bytes), res=" + std::to_string(cliParams.resolution));
            auto png = cliRoundtrip ? engine.vae_roundtrip(bytes, cliParams, progress)
                                    : engine.upscale(bytes, cliParams, progress);
            if (cliOut.empty()) {
                std::string base = fs::path(cliIn).stem().string() + "_seedvr2.png";
                cliOut = (fs::current_path() / base).string();
            }
            std::ofstream o(cliOut, std::ios::binary);
            o.write(reinterpret_cast<const char*>(png.data()), (std::streamsize)png.size());
            o.close();
            SEEDVR_LOG(LogLevel::Info, "wrote " + cliOut + " (" + std::to_string(png.size()) + " bytes)");
        } catch (const std::exception& e) {
            SEEDVR_LOG(LogLevel::Error, std::string("CLI failed: ") + e.what());
            return 2;
        }
        return 0;
    }

    // ---------------- ggml self-test (broadcast elementwise via gallocr) ----------------
    if (cliSelfTest) {
        ggml_backend_load_all();
        ggml_backend_t be = ggml_backend_cuda_init(0);
        if (!be) be = ggml_backend_cpu_init();
        ggml_init_params ip{}; ip.mem_size = 4 * 1024 * 1024; ip.no_alloc = true;
        ggml_context* c = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 3);   // {4,3}
        ggml_tensor* g = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 1);   // {4,1} broadcast
        ggml_tensor* y = ggml_mul(c, x, g);
        ggml_tensor* z = ggml_add(c, y, ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 1));
        ggml_tensor* s = ggml_silu(c, x);
        ggml_build_forward_expand(gf, z);
        ggml_build_forward_expand(gf, s);
        ggml_gallocr_t gal = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(gal, gf);
        float xv[12] = { 1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400 };
        float gv[4] = { 2, 3, 4, 5 };
        float bv[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        ggml_backend_tensor_set(x, xv, 0, sizeof xv);
        ggml_backend_tensor_set(g, gv, 0, sizeof gv);
        ggml_backend_tensor_set(z->src[1], bv, 0, sizeof bv);
        ggml_backend_graph_compute(be, gf);
        float out[12] = { 0 }, so[12] = { 0 };
        ggml_backend_tensor_get(z, out, 0, sizeof out);
        ggml_backend_tensor_get(s, so, 0, sizeof so);
        std::printf("selftest mul+add: ");
        for (int i = 0; i < 12; ++i) std::printf("%g ", out[i]);
        std::printf("\nselftest silu   : ");
        for (int i = 0; i < 12; ++i) std::printf("%g ", so[i]);
        std::printf("\nexpected mul+add: 2.5 6.5 12.5 20.5 20.5 60.5 120.5 200.5 200.5 600.5 1200.5 2000.5\n");
        std::printf("expected silu   : 0.7311 0.8808 0.9526 0.9820 0.99995 1 1 1 1 1 1 1\n");
        return 0;
    }

    // load the models in the background so the UI is reachable immediately
    std::thread loader([&]() {
        try {
            engine.init(engCfg, progress);
            progress.set_message("ready");
        } catch (const std::exception& e) {
            progress.failed = true;
            progress.set_message(std::string("init failed: ") + e.what());
            SEEDVR_LOG(LogLevel::Error, std::string("init failed: ") + e.what());
        }
    });
    loader.detach();

    Server server(engine, progress, srvCfg);
    server.run();
    return 0;
}
