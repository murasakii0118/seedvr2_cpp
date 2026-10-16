// GGUF weight store living on a ggml backend device.
//
// Backends (CUDA / Vulkan / CPU) are linked statically into the executable, so
// the device is chosen at *runtime* by name ("CUDA0", "Vulkan0", "CPU", ...)
// through ggml's device registry.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace seedvr2 {

struct DeviceInfo {
    std::string name;          // canonical ggml device name, e.g. "CUDA0"
    std::string type;          // "GPU" / "ACCEL" / "CPU"
    std::string description;   // human readable, from the backend
};

// every device the linked-in backends registered
std::vector<DeviceInfo> list_devices();

// resolve a user supplied string to a canonical device name.  Accepts an exact
// name ("Vulkan0"), a backend keyword ("vulkan", "cuda", "cpu") or an empty
// string (= first GPU, falling back to CPU).  Returns "" when nothing matches.
std::string resolve_device(const std::string& want);

class GgufModel {
public:
    // upcast_f16: convert F16 weights to F32 while uploading.  The CUDA backend
    // otherwise converts the *activations* to F16 on every mul_mat against F16
    // weights - with the VAE's tap-per-mul_mat scheme that is thousands of
    // conversions per forward and dominates the runtime.
    void load(const std::string& path, const std::string& device, bool upcast_f16 = false);
    ggml_tensor* get(const std::string& name) const;
    bool has(const std::string& name) const;
    ggml_backend_t backend() const { return backend_; }
    std::string backend_name() const;
    void free_all();

private:
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    ggml_context* ctx_data_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};

} // namespace seedvr2
