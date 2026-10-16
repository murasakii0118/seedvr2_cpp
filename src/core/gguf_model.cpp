#include "core/gguf_model.h"
#include "core/common.h"

#include <algorithm>
#include <cctype>

#include "ggml-cpu.h"
#include "gguf.h"

namespace seedvr2 {

static bool g_backend_loaded = false;

static void load_backends_once() {
    if (!g_backend_loaded) {
        ggml_backend_load_all();
        g_backend_loaded = true;
    }
}

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static const char* dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

std::vector<DeviceInfo> list_devices() {
    load_backends_once();
    std::vector<DeviceInfo> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        DeviceInfo di;
        di.name = ggml_backend_dev_name(d);
        di.type = dev_type_name(ggml_backend_dev_type(d));
        const char* desc = ggml_backend_dev_description(d);
        di.description = desc ? desc : "";
        if (di.type == "CPU" || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_ACCEL)
            out.push_back(di);
    }
    return out;
}

std::string resolve_device(const std::string& want) {
    load_backends_once();
    const std::string w = lower(want);
    std::string first_gpu, first_cpu;

    // exact (case-insensitive) name match first
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const std::string nm = ggml_backend_dev_name(d);
        const std::string ln = lower(nm);
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU && first_gpu.empty()) first_gpu = nm;
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU && first_cpu.empty()) first_cpu = nm;
        if (!w.empty() && ln == w) return nm;
    }
    // backend keyword ("vulkan", "cuda", "cpu") -> first matching device
    if (!w.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const std::string ln = lower(ggml_backend_dev_name(d));
            if (ln.compare(0, w.size(), w) == 0) return ggml_backend_dev_name(d);
        }
    }
    if (w.empty()) return first_gpu.empty() ? first_cpu : first_gpu;
    return "";
}

void GgufModel::load(const std::string& path, const std::string& device, bool upcast_f16) {
    load_backends_once();
    free_all();

    const std::string dev_name = resolve_device(device);
    require(!dev_name.empty(),
            "no ggml device matches '" + device + "' (available: " +
                [&] { std::string s; for (auto& d : list_devices()) s += d.name + " "; return s; }() + ")");

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (dev_name == ggml_backend_dev_name(d)) { dev = d; break; }
    }
    require(dev != nullptr, "ggml device disappeared: " + dev_name);
    backend_ = ggml_backend_dev_init(dev, nullptr);
    require(backend_ != nullptr, "failed to initialise backend device " + dev_name);

    // ---- parse gguf (data stays in host memory for now) ----
    ggml_context* src_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = &src_ctx;
    gguf_context* gctx = gguf_init_from_file(path.c_str(), params);
    require(gctx != nullptr, "failed to open GGUF: " + path);

    const int64_t n = gguf_get_n_tensors(gctx);
    SEEDVR_LOG(LogLevel::Info, "GGUF: " + std::to_string(n) + " tensors from " + path);

    // ---- create destination context + tensors ----
    ggml_init_params ip{};
    ip.mem_size = (size_t)ggml_tensor_overhead() * (size_t)(n + 8) + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ctx_data_ = ggml_init(ip);
    require(ctx_data_ != nullptr, "ggml_init failed");

    std::vector<std::string> names;
    names.reserve((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        const char* nm = gguf_get_tensor_name(gctx, i);
        ggml_tensor* src = ggml_get_tensor(src_ctx, nm);
        require(src != nullptr, std::string("missing src tensor ") + nm);
        ggml_tensor* dst = ggml_new_tensor(ctx_data_,
                                           (upcast_f16 && src->type == GGML_TYPE_F16) ? GGML_TYPE_F32 : src->type,
                                           GGML_MAX_DIMS, src->ne);
        ggml_set_name(dst, nm);
        names.push_back(nm);
    }

    // ---- allocate on the chosen device ----
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_data_, backend_);
    require(buf_ != nullptr, "failed to allocate model buffer on " + dev_name);

    // ---- copy data ----
    for (const auto& nm : names) {
        ggml_tensor* s = ggml_get_tensor(src_ctx, nm.c_str());
        ggml_tensor* d = ggml_get_tensor(ctx_data_, nm.c_str());
        if (s->type == GGML_TYPE_F16 && d->type == GGML_TYPE_F32) {
            std::vector<float> tmp((size_t)ggml_nelements(s));
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)s->data, tmp.data(), (int64_t)tmp.size());
            ggml_backend_tensor_set(d, tmp.data(), 0, tmp.size() * sizeof(float));
        } else {
            size_t nb = ggml_nbytes(s);
            ggml_backend_tensor_set(d, s->data, 0, nb);
        }
        tensors_[nm] = d;
    }

    gguf_free(gctx);
    ggml_free(src_ctx);
    SEEDVR_LOG(LogLevel::Info, "GGUF weights uploaded to " + backend_name());
}

ggml_tensor* GgufModel::get(const std::string& name) const {
    auto it = tensors_.find(name);
    require(it != tensors_.end(), "missing tensor: " + name);
    return it->second;
}

bool GgufModel::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::string GgufModel::backend_name() const {
    if (!backend_) return "(none)";
    return ggml_backend_name(backend_);
}

void GgufModel::free_all() {
    if (buf_) { ggml_backend_buffer_free(buf_); buf_ = nullptr; }
    if (ctx_data_) { ggml_free(ctx_data_); ctx_data_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
    tensors_.clear();
}

} // namespace seedvr2
