// SeedVR2 C++ - shared definitions
#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <algorithm>

namespace seedvr2 {

// ---------------------------------------------------------------------------
// Runtime error type
// ---------------------------------------------------------------------------
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& m) : std::runtime_error(m) {}
};

inline void require(bool cond, const std::string& msg) {
    if (!cond) throw Error(msg);
}

// ---------------------------------------------------------------------------
// Logging (console + in-memory ring buffer surfaced to the web UI)
// ---------------------------------------------------------------------------
enum class LogLevel { Debug, Info, Warn, Error };

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void log(LogLevel lvl, const std::string& msg) {
        static const char* tag[] = { "DEBUG", "INFO ", "WARN ", "ERROR" };
        std::string line = std::string("[") + tag[(int)lvl] + "] " + msg + "\n";
        std::lock_guard<std::mutex> lk(mu_);
        std::fputs(line.c_str(), stdout);
        std::fflush(stdout);
        buf_ += line;
        if (buf_.size() > 512 * 1024) buf_.erase(0, buf_.size() - 512 * 1024);
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        buf_.clear();
    }
    std::string text() const {
        std::lock_guard<std::mutex> lk(mu_);
        return buf_;
    }

private:
    mutable std::mutex mu_;
    std::string buf_;
};

#define SEEDVR_LOG(lvl, msg) ::seedvr2::Logger::instance().log((lvl), (msg))

// ---------------------------------------------------------------------------
// Progress reporting for the web UI
// ---------------------------------------------------------------------------
struct Progress {
    std::atomic<int>    phase{0};          // 0 idle, 1..4 pipeline phases
    std::atomic<int>    item{0};
    std::atomic<int>    total{0};
    std::atomic<int>    percent{0};
    std::atomic<bool>   running{false};
    std::atomic<bool>   cancel{false};
    std::atomic<bool>   done{false};
    std::atomic<bool>   failed{false};

    std::mutex          msg_mu;
    std::string         message;

    void set_message(const std::string& m) {
        std::lock_guard<std::mutex> lk(msg_mu);
        message = m;
    }
    std::string get_message() {
        std::lock_guard<std::mutex> lk(msg_mu);
        return message;
    }
};

// ---------------------------------------------------------------------------
// Inference parameters (exposed in the web UI)
// ---------------------------------------------------------------------------
struct UpscaleParams {
    int         resolution      = 1080;   // target shortest edge
    int         max_resolution  = 0;      // 0 = unlimited
    int         seed            = 42;
    int         steps           = 1;      // distilled model -> 1 step
    double      cfg_scale       = 1.0;    // forced to 1.0 for the distilled model
    double      cfg_rescale     = 0.0;
    std::string color_correction = "lab"; // lab|wavelet|wavelet_adaptive|hsv|adain|none

    // VAE spatial tiling (ggml VAE only).  Mirrors the Python
    // --vae_encode_tiled / --vae_decode_tiled options: high resolutions are
    // processed tile by tile to cut peak VRAM.  tile_size / tile_overlap are in
    // *output-space* pixels (defaults match the Python CLI).
    bool        vae_encode_tiled = false;
    bool        vae_decode_tiled = false;
    int         vae_tile_size    = 1024;
    int         vae_tile_overlap = 128;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
inline int round_half_away(double v) { return (int)std::floor(v + 0.5); }

} // namespace seedvr2
