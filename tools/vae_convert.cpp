// seedvr2_vae_convert - standalone SeedVR2 VAE (.safetensors) -> quantised GGUF converter.
//
// This tool is intentionally independent of the main application: it only needs
// ggml (for quantisation) and nlohmann/json (for the safetensors header).  It
// builds a tiny CPU-only binary, so converting a model never drags in CUDA /
// Vulkan / libtorch / OpenCV.
//
// Layout notes (the whole point of this converter):
//   * torch stores conv3d weights as [Cout, Cin, kT, kH, kW].  ggml's matmul
//     wants the *contraction* dim on ne0, so every conv kernel is transposed to
//     {Cin, Cout, K} with K = kT*kH*kW taps, tap index k = (kt*kH + kh)*kW + kw.
//     Each tap is then a {Cin, Cout} plane at offset k*Cin*Cout, i.e. a plain
//     ggml_view_2d away - which lets the whole 3D convolution run as K quantised
//     ggml_mul_mat calls.
//   * Q8_0 blocks are 32 elements along ne0 (= Cin), so quantisation is exactly
//     the usual per-row scheme and stays valid per tap.
//   * tensors that cannot be quantised (Cin % 32 != 0, vectors, biases) are
//     stored as F16.
//
// NOTE: the {Cin, Cout, K} layout and the tap order are paired with the reader in
// the main program (src/core/vae_ggml.cpp, conv_core/kdims).  Changing one side
// without the other yields "looks plausible but is wrong" results.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "nlohmann/json.hpp"

namespace {

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

void info(const std::string& msg) { std::printf("%s\n", msg.c_str()); }

// ---------------------------------------------------------------------------
// minimal safetensors reader (header json + raw data blob, no torch involved)
// ---------------------------------------------------------------------------
struct StTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    const char* data = nullptr;
    size_t nbytes = 0;
};

std::vector<char> read_safetensors(const std::string& path, std::vector<StTensor>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) fail("cannot open " + path);
    f.seekg(0, std::ios::end);
    const size_t total = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(total);
    f.read(buf.data(), (std::streamsize)total);
    if ((size_t)f.gcount() != total) fail("short read " + path);

    uint64_t hlen = 0;
    std::memcpy(&hlen, buf.data(), 8);
    if (hlen + 8 > total) fail("bad safetensors header size");
    auto meta = nlohmann::json::parse(std::string(buf.data() + 8, (size_t)hlen));
    const size_t data_off = 8 + (size_t)hlen;

    for (auto it = meta.begin(); it != meta.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        StTensor t;
        t.name = it.key();
        t.dtype = it.value()["dtype"].get<std::string>();
        for (auto& d : it.value()["shape"]) t.shape.push_back(d.get<int64_t>());
        uint64_t b0 = it.value()["data_offsets"][0].get<uint64_t>();
        uint64_t b1 = it.value()["data_offsets"][1].get<uint64_t>();
        if (data_off + b1 > total) fail("tensor out of range: " + t.name);
        t.data = buf.data() + data_off + b0;
        t.nbytes = (size_t)(b1 - b0);
        out.push_back(t);
    }
    return buf;
}

void to_f32(const StTensor& t, std::vector<float>& out) {
    const size_t n = t.nbytes / (t.dtype == "F16" ? 2 : 4);
    out.resize(n);
    if (t.dtype == "F16") {
        const ggml_fp16_t* p = reinterpret_cast<const ggml_fp16_t*>(t.data);
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(p[i]);
    } else if (t.dtype == "F32") {
        std::memcpy(out.data(), t.data, n * 4);
    } else if (t.dtype == "BF16") {
        const ggml_bf16_t* p = reinterpret_cast<const ggml_bf16_t*>(t.data);
        for (size_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(p[i]);
    } else {
        fail("unsupported safetensors dtype " + t.dtype + " for " + t.name);
    }
}

// ---------------------------------------------------------------------------
// GGUF writer
// ---------------------------------------------------------------------------
struct GgufOut {
    std::string name;
    std::vector<int64_t> ne;         // ggml order: ne0 first
    uint32_t type = 0;               // ggml type id
    std::vector<uint8_t> bytes;      // serialised tensor data
};

// Q8_0 = one fp16 scale + 32 int8 quants
constexpr size_t Q8_0_BLOCK_BYTES = 2 + 32;

void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.insert(v.end(), (uint8_t*)&x, (uint8_t*)&x + 4);
}
void put_u64(std::vector<uint8_t>& v, uint64_t x) {
    v.insert(v.end(), (uint8_t*)&x, (uint8_t*)&x + 8);
}
void put_str(std::vector<uint8_t>& v, const std::string& s) {
    put_u64(v, s.size());
    v.insert(v.end(), s.begin(), s.end());
}

void write_gguf(const std::string& path, const std::vector<GgufOut>& tensors) {
    std::vector<uint8_t> hdr;
    put_u32(hdr, 0x46554747 /*GGUF*/);
    put_u32(hdr, 3);
    put_u64(hdr, tensors.size());
    put_u64(hdr, 2);                                   // kv count
    put_str(hdr, "general.architecture");              // kv 1
    put_u32(hdr, 8 /*str*/);
    put_str(hdr, "seedvr2-vae");
    put_str(hdr, "general.alignment");                 // kv 2
    put_u32(hdr, 4 /*u32*/);
    put_u32(hdr, 32);

    size_t off = 0;
    for (auto& t : tensors) {
        put_str(hdr, t.name);
        put_u32(hdr, (uint32_t)t.ne.size());
        for (int64_t d : t.ne) put_u64(hdr, (uint64_t)d);
        put_u32(hdr, t.type);
        put_u64(hdr, off);
        size_t row = (size_t)(t.ne[0]);
        size_t block = 0;
        switch (t.type) {
            case GGML_TYPE_Q8_0: block = row / 32 * Q8_0_BLOCK_BYTES; break;
            case GGML_TYPE_F16:  block = row * 2; break;
            default:             block = row * 4; break;
        }
        size_t rows = 1;
        for (size_t i = 1; i < t.ne.size(); ++i) rows *= (size_t)t.ne[i];
        if (block * rows != t.bytes.size()) fail("size mismatch for " + t.name);
        off += t.bytes.size();
        off = (off + 31) / 32 * 32;      // keep every tensor 32B aligned
    }

    std::ofstream f(path, std::ios::binary);
    if (!f.good()) fail("cannot write " + path);
    f.write((const char*)hdr.data(), (std::streamsize)hdr.size());
    size_t pos = hdr.size();
    std::vector<char> zeros(64, 0);
    for (auto& t : tensors) {
        size_t pad = (32 - (pos % 32)) % 32;
        if (pad) { f.write(zeros.data(), (std::streamsize)pad); pos += pad; }
        f.write((const char*)t.bytes.data(), (std::streamsize)t.bytes.size());
        pos += t.bytes.size();
    }
    info("wrote " + path + " (" + std::to_string(pos) + " bytes, " +
         std::to_string(tensors.size()) + " tensors)");
}

void quantize_rows(const std::vector<float>& src, int64_t n_per_row, bool quantize,
                   std::vector<uint8_t>& out, uint32_t& type) {
    const int64_t nrows = (int64_t)(src.size()) / n_per_row;
    if (quantize && n_per_row % 32 == 0) {
        type = GGML_TYPE_Q8_0;
        out.resize((size_t)nrows * (size_t)(n_per_row / 32) * Q8_0_BLOCK_BYTES);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), out.data(), 0, nrows, n_per_row, nullptr);
    } else {
        type = GGML_TYPE_F16;
        out.resize(src.size() * 2);
        for (size_t i = 0; i < src.size(); ++i)
            reinterpret_cast<ggml_fp16_t*>(out.data())[i] = ggml_fp32_to_fp16(src[i]);
    }
}

size_t convert_vae_to_gguf(const std::string& safetensors, const std::string& out_gguf,
                           const std::string& quant) {
    const bool do_quant = (quant != "f16" && quant != "none");
    std::vector<StTensor> ts;
    std::vector<char> blob = read_safetensors(safetensors, ts);
    (void)blob;
    info("read " + std::to_string(ts.size()) + " tensors from " + safetensors);

    std::vector<GgufOut> outs;
    size_t n_quant = 0, n_f16 = 0;
    std::vector<float> f32;

    for (const auto& t : ts) {
        GgufOut o;
        o.name = t.name;
        const bool is_conv3d = t.shape.size() == 5 && t.name.size() > 7 &&
                               t.name.compare(t.name.size() - 7, 7, ".weight") == 0;
        const bool is_linear = t.shape.size() == 2 && t.name.size() > 7 &&
                               t.name.compare(t.name.size() - 7, 7, ".weight") == 0;
        if (is_conv3d) {
            // [Cout,Cin,kT,kH,kW] -> {Cin, Cout, K}
            const int64_t Cout = t.shape[0], Cin = t.shape[1], K = t.shape[2] * t.shape[3] * t.shape[4];
            to_f32(t, f32);
            std::vector<float> dst((size_t)Cin * Cout * K);
            for (int64_t k = 0; k < K; ++k)
                for (int64_t co = 0; co < Cout; ++co)
                    for (int64_t ci = 0; ci < Cin; ++ci)
                        dst[(size_t)k * Cin * Cout + (size_t)co * Cin + ci] =
                            f32[(size_t)((co * Cin + ci) * K + k)];
            quantize_rows(dst, Cin, do_quant, o.bytes, o.type);
            o.ne = { Cin, Cout, K };
            (o.type == GGML_TYPE_Q8_0) ? ++n_quant : ++n_f16;
        } else if (is_linear) {
            // [out,in] -> {in,out}
            const int64_t out_f = t.shape[0], in_f = t.shape[1];
            to_f32(t, f32);
            std::vector<float> dst((size_t)in_f * out_f);
            for (int64_t oo = 0; oo < out_f; ++oo)
                for (int64_t ii = 0; ii < in_f; ++ii)
                    dst[(size_t)ii + (size_t)in_f * oo] = f32[(size_t)oo * in_f + ii];
            quantize_rows(dst, in_f, do_quant, o.bytes, o.type);
            o.ne = { in_f, out_f };
            (o.type == GGML_TYPE_Q8_0) ? ++n_quant : ++n_f16;
        } else {
            to_f32(t, f32);
            quantize_rows(f32, 1, false, o.bytes, o.type);   // always F16
            o.ne.assign(t.shape.begin(), t.shape.end());
            ++n_f16;
        }
        outs.push_back(std::move(o));
    }

    info("vae tensors: " + std::to_string(outs.size()) + " (q8_0=" + std::to_string(n_quant) +
         " f16=" + std::to_string(n_f16) + ")");
    write_gguf(out_gguf, outs);
    return outs.size();
}

void usage(const char* exe) {
    std::printf(
        "seedvr2_vae_convert - convert a SeedVR2 VAE .safetensors into a quantised GGUF\n"
        "\n"
        "usage: %s <input.safetensors> <output.gguf> [--quant f16|q8_0]\n"
        "\n"
        "  f16   (default) every kernel is stored as F16 - lossless and recommended;\n"
        "        the main program upcasts F16 weights to F32 on load.\n"
        "  q8_0  quantise conv/linear kernels whose contraction dim is a multiple of 32\n"
        "        to Q8_0 (everything else stays F16).  Lossy (~4 dB end-to-end).\n"
        "\n"
        "example:\n"
        "  %s models/ema_vae_fp16.safetensors models/ema_vae_fp16.gguf --quant f16\n",
        exe, exe);
}

} // namespace

int main(int argc, char** argv) {
    std::string in, out, quant = "f16";
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        if (a == "--quant") { if (i + 1 >= argc) { usage(argv[0]); return 1; } quant = argv[++i]; }
        else if (a.rfind("--quant=", 0) == 0) quant = a.substr(8);
        else pos.push_back(a);
    }
    if (pos.size() != 2) { usage(argv[0]); return 1; }
    if (quant != "f16" && quant != "q8_0" && quant != "none") {
        std::fprintf(stderr, "error: unknown --quant '%s' (expected f16 or q8_0)\n", quant.c_str());
        return 1;
    }
    in = pos[0];
    out = pos[1];

    try {
        const size_t n = convert_vae_to_gguf(in, out, quant);
        info("done: " + std::to_string(n) + " tensors -> " + out);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
