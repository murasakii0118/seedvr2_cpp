# SeedVR2 C++

把 [SeedVR2](https://github.com/ByteDance-Seed/SeedVR2)（视频/图像超分）的推理部分重写为 C++：
DiT 与 VAE 全部走 [ggml](https://github.com/ggml-org/ggml)（CUDA / Vulkan / CPU 三种后端静态编入同一个 exe），
自带 cpp-httplib Web UI，MSVC 编译，产物放在 `release/`。

当前状态：单图 / 批量图像放大可用，DiT 与 VAE 均与 Python 参考实现数值对齐。
视频（T > 1）暂不支持。

---

## 目录

| 路径 | 说明 |
|---|---|
| `src/` | 主程序（DiT、VAE、管线、HTTP 服务、Web UI） |
| `src/core/` | 核心：`dit.cpp`（ggml DiT）、`vae_ggml.cpp`（ggml VAE + tiling）、`pipeline.cpp`、`gguf_model.cpp`、前后处理等 |
| `src/server/` | cpp-httplib 服务与 REST API |
| `src/frontend/index.html` | 单文件 Web UI（构建时内嵌进 exe） |
| `tools/` | 独立小工具：`seedvr2_vae_convert`（VAE safetensors → GGUF 量化） |
| `models/` | 模型（需自行准备，见下） |
| `release/` | 构建产物（两个 exe + 运行时 DLL） |
| `3rd/` | 第三方依赖（ggml / cpp-httplib / opencv / libtorch / nlohmann） |
| `seedvr2_videoupscaler/` | Python 参考实现（用于数值对拍，运行时**不需要**） |
| `introduction.md` | 开发记录：踩过的坑、性能分析、tiling/量化细节 |

---

## 依赖

### 构建期

| 依赖 | 版本 | 说明 |
|---|---|---|
| Windows | 10/11 x64 | 目前在 MSVC 上验证 |
| Visual Studio 2022 | v17.x | 需要 C++ 桌面工作负载 |
| CUDA Toolkit | 13.3 | 提供 `nvcc`；`GGML_CUDA` 需要 |
| Vulkan SDK | 1.4.x（可选） | 仅当需要 Vulkan 后端时；须设置 `VULKAN_SDK` 环境变量 |
| CMake | ≥ 3.20 | 目前用 3.31 验证 |

`3rd/` 下已随仓库放置：`ggml/`、`cpp-httplib/`、`opencv4120/`、`libtorch/`、`nlohmann/`。
Vulkan 的 shader 由 `glslc` 在构建期编译并静态编入 exe；运行期只依赖驱动自带的 `vulkan-1.dll`。

### 运行期

- 一张支持 CUDA 的 NVIDIA 显卡（Vulkan / CPU 可用于 DiT；VAE 目前只支持 CUDA / CPU）。
- 与 exe 同目录的 `opencv_world4120.dll`、libtorch 的若干 DLL（构建时自动拷贝到 `release/`），
  以及驱动提供的 `vulkan-1.dll`。

### 模型（需自行准备）

| 文件 | 说明 |
|---|---|
| `models/seedvr2_ema_3b-Q8_0.gguf` | 3B DiT（Q8_0） |
| `models/ema_vae_fp16.gguf` | VAE（fp16，由 `tools/` 里的转换工具生成） |
| `models/ema_vae_fp16.safetensors` | VAE 原始权重（**仅**转换时需要，运行不需要） |

DiT 的 GGUF 与 Python 参考实现的模型对应；VAE 的 safetensors 来自参考工程。
`pos_emb` / `neg_emb` 词向量已硬编码进 exe，不需要外部文件。

---

## 构建

### 1. 主程序

```bat
:: 配置（把 <CUDA> 换成你的 CUDA Toolkit 路径，或用环境变量 CUDA_PATH）
cmake -S src -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_CUDA_COMPILER="<CUDA>/bin/nvcc.exe"

:: 编译（Release，已开启 /O2 等优化且不生成调试符号）
cmake --build build --config Release -j
```

产物：`release/seedvr2_cpp.exe`（以及自动拷贝的 opencv / libtorch 运行时 DLL）。

可选配置项：

- `-DVULKAN_SDK_DIR=<path>`：未设置 `VULKAN_SDK` 环境变量时用它定位 Vulkan SDK；
- `-DCMAKE_CUDA_ARCHITECTURES=89`（默认）：按显卡算力调整（40 系 = 89）。

### 2. VAE 转换工具（独立、仅 CPU）

```bat
cmake -S tools -B build_tools -G "Visual Studio 17 2022" -A x64
cmake --build build_tools --config Release -j
```

产物：`release/seedvr2_vae_convert.exe`（约 0.6 MB，不依赖 CUDA / Vulkan / OpenCV / libtorch）。

---

## 生成 VAE 的 GGUF

```bat
release\seedvr2_vae_convert.exe models\ema_vae_fp16.safetensors models\ema_vae_fp16.gguf --quant f16
```

- `--quant f16`（默认，推荐）：全部权重存 F16，**无损**（源本身就是 fp16），加载时程序会转成 F32。
- `--quant q8_0`：卷积 / 线性层量化到 Q8_0（要求收缩维是 32 的倍数），其余保持 F16；有损，端到端约低 4 dB。

细节与注意事项见 `introduction.md`「VAE → GGUF 转换」。

---

## 使用

### Web UI

```bat
release\seedvr2_cpp.exe --root .
```

浏览器打开 `http://127.0.0.1:8188/`。可拖入多张图片批处理，勾选「启用 latent 缓存」走三段式批处理，
顶栏可切换 DiT / VAE 的计算后端（切换会重新加载模型）。

常用参数（也可用命令行传入）：

```bat
release\seedvr2_cpp.exe --root . --input in.jpg --output out.png --resolution 1080 --seed 42 --cc lab
```

| 参数 | 说明 |
|---|---|
| `--root DIR` | 仓库根目录（默认从 exe 位置 / 当前目录向上查找 `models`） |
| `--gguf PATH` / `--vae PATH` | DiT / VAE 模型路径 |
| `--dit-device D` / `--vae-device D` | 计算后端：`CUDA0` / `Vulkan0` / `CPU`（VAE 不支持 Vulkan） |
| `--list-devices` | 列出可用 ggml 设备并退出 |
| `--no-staged-load` | 让 DiT 与 VAE 同时常驻（默认分阶段加载以省显存） |
| `--no-browser` / `--host ADDR` / `--port N` | 服务相关 |
| `--input FILE` / `--output FILE` | 命令行单图模式 |
| `--resolution N` / `--maxres N` | 目标短边 / 最大边长（0 = 不限） |
| `--seed N` / `--steps N` / `--cfg F` / `--cc NAME` | 采样与色彩校正（`lab` / `wavelet` / `wavelet_adaptive` / `hsv` / `adain` / `none`） |
| `--vae-roundtrip` | 只跑 VAE encode+decode（诊断用） |
| `--vae-tiled` / `--vae-encode-tiled` / `--vae-decode-tiled` | 开启 VAE 空间分块（高分辨率省显存） |
| `--vae-tile-size N` / `--vae-tile-overlap N` | 分块尺寸 / 重叠（输出空间像素，默认 1024 / 128） |

### REST API

`GET /`、`GET /favicon.ico`、`GET /api/status`、`GET /api/devices`、`GET /api/log`、
`POST /api/backend?dit=..&vae=..`、`POST /api/cancel`、`POST /api/clearLog`、
`POST /api/upscale`（body = 图片字节，参数在 query）、
`POST /api/upscaleBatch`（multipart，字段名 `files` + query 参数）、`GET /api/result?id=..`。

---

## 已知限制

- **Vulkan 上 VAE 结果错误**（编码器即出现 1e6 级数值），因此 `GET /api/devices` 的 VAE 只提供 CPU / CUDA；
  DiT 在 Vulkan 上完全正确，且通常比 CUDA 更快。
- 视频 / 多帧（T > 1）尚未支持，当前只处理单帧图像。
- 管线中的前后处理仍以 libtorch 张量为容器（正在进行去 libtorch 的工作，见 `introduction.md`）。

---

## 许可证

本项目使用 **Apache License 2.0**，见根目录 [`LICENSE`](LICENSE)。

随仓库包含的第三方组件保留各自许可证：

| 组件 | 许可证 |
|---|---|
| ggml | MIT |
| cpp-httplib | MIT |
| nlohmann/json | MIT |
| OpenCV | Apache-2.0 |
| libtorch / PyTorch | BSD-3-Clause |

Python 参考实现 `seedvr2_videoupscaler/` 来自 ByteDance（Apache-2.0）。
