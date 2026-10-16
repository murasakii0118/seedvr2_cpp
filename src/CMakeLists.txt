##############################################################################
# SeedVR2 C++  -  ggml (CUDA/Vulkan/CPU) DiT + ggml VAE  +  cpp-httplib UI
#
# Build (MSVC + CUDA):
#   cmake -S src -B build -G "Visual Studio 17 2022" -A x64 ^
#         -DCMAKE_CUDA_COMPILER="<CUDA>/bin/nvcc.exe"
#   cmake --build build --config Release -j
#
# Output binary is copied to <repo>/release
##############################################################################
cmake_minimum_required(VERSION 3.20)

project(seedvr2_cpp CXX C)

enable_language(RC)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# CUDA 13 dropped <sm_75; build only for architectures we care about.
# (RTX 40xx = sm_89).  Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "" FORCE)
endif()

# MSVC needs UTF-8 source encoding for the embedded HTML / CJK comments.
# IMPORTANT: must NOT be applied to CUDA sources (nvcc chokes on /MP and friends).
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:C,CXX>:/MP>
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4267>
    )
endif()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
get_filename_component(REPO_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(THIRD_PARTY "${REPO_ROOT}/3rd")
set(RELEASE_DIR "${REPO_ROOT}/release")

message(STATUS "REPO_ROOT   = ${REPO_ROOT}")
message(STATUS "THIRD_PARTY = ${THIRD_PARTY}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RELEASE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${RELEASE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${RELEASE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${RELEASE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# ggml  (CUDA + Vulkan + CPU backends, all linked statically into the exe)
# ---------------------------------------------------------------------------
set(GGML_CUDA           ON  CACHE BOOL "" FORCE)
set(GGML_VULKAN         ON  CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_STATIC         ON  CACHE BOOL "" FORCE)
# keep the x86 SIMD kernels (AVX/AVX2/BMI2) enabled even with GGML_NATIVE=OFF so
# the CPU backend uses the AVX2 accelerated paths instead of the scalar fallback
set(GGML_AVX            ON  CACHE BOOL "" FORCE)
set(GGML_AVX2           ON  CACHE BOOL "" FORCE)
set(GGML_BMI2           ON  CACHE BOOL "" FORCE)

# Vulkan SDK: honour the VULKAN_SDK environment variable, or an explicitly
# passed -DVULKAN_SDK_DIR=<path>.  No absolute path is baked into this file.
# (The ggml Vulkan backend is compiled+linked statically into the exe; only the
#  OS Vulkan loader vulkan-1.dll - shipped with the GPU driver - is dynamic.)
if (NOT DEFINED ENV{VULKAN_SDK} AND DEFINED VULKAN_SDK_DIR AND EXISTS "${VULKAN_SDK_DIR}")
    set(ENV{VULKAN_SDK} "${VULKAN_SDK_DIR}")
endif()
if (DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
    message(STATUS "VULKAN_SDK  = $ENV{VULKAN_SDK}")
else()
    message(WARNING "VULKAN_SDK is not set - the Vulkan backend may fail to configure")
endif()

add_subdirectory("${THIRD_PARTY}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# libtorch (CPU)
# ---------------------------------------------------------------------------
set(Torch_DIR "${THIRD_PARTY}/libtorch/share/cmake/Torch" CACHE PATH "" FORCE)
find_package(Torch REQUIRED)

# ---------------------------------------------------------------------------
# OpenCV  (opencv_world)
# ---------------------------------------------------------------------------
set(OpenCV_DIR "${THIRD_PARTY}/opencv4120" CACHE PATH "" FORCE)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version = ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# cpp-httplib
# ---------------------------------------------------------------------------
set(HTTPLIB_DIR "${THIRD_PARTY}/cpp-httplib")

# ---------------------------------------------------------------------------
# Embedded frontend (single-file HTML -> C++ header)
# ---------------------------------------------------------------------------
set(FRONTEND_HTML "${CMAKE_CURRENT_SOURCE_DIR}/frontend/index.html")
set(FRONTEND_HDR  "${CMAKE_BINARY_DIR}/generated/frontend_assets.h")

# re-run the configure step (which regenerates the embedded header below) whenever
# the frontend changes, otherwise edits to index.html would be silently ignored
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FRONTEND_HTML}")

file(READ "${FRONTEND_HTML}" FRONTEND_RAW)

# escape for a C string literal
string(REPLACE "\\" "\\\\"  FRONTEND_RAW "${FRONTEND_RAW}")
string(REPLACE "\"" "\\\""  FRONTEND_RAW "${FRONTEND_RAW}")
string(REPLACE "\r" ""      FRONTEND_RAW "${FRONTEND_RAW}")
string(REPLACE "\n" "\\n\"\n\"" FRONTEND_RAW "${FRONTEND_RAW}")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(WRITE "${FRONTEND_HDR}"
"// Auto-generated from frontend/index.html - do not edit
#pragma once
static const char* const kFrontendHtml =
\"${FRONTEND_RAW}\";
")

# ---------------------------------------------------------------------------
# Embedded application icon (served as /favicon.ico)
# ---------------------------------------------------------------------------
set(APP_ICON "${REPO_ROOT}/03.ico")
file(SIZE "${APP_ICON}" APP_ICON_SIZE)
file(READ "${APP_ICON}" APP_ICON_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," APP_ICON_BYTES "${APP_ICON_HEX}")
file(WRITE "${CMAKE_BINARY_DIR}/generated/icon_assets.h"
"// Auto-generated from 03.ico - do not edit
#pragma once
static const unsigned int kAppIconSize = ${APP_ICON_SIZE};
static const unsigned char kAppIcon[] = {
${APP_ICON_BYTES}
};
")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
set(APP_SOURCES
    main.cpp
    core/image_io.cpp
    core/transforms.cpp
    core/color_fix.cpp
    core/schedule.cpp
    core/text_emb.cpp
    core/vae_ggml.cpp
    core/dit.cpp
    core/gguf_model.cpp
    core/pipeline.cpp
    server/server.cpp
)

# application icon resource (path injected at configure time)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/app.rc.in" "${CMAKE_BINARY_DIR}/generated/app.rc" @ONLY)

add_executable(seedvr2_cpp ${APP_SOURCES} "${HTTPLIB_DIR}/httplib.cpp"
               "${CMAKE_BINARY_DIR}/generated/app.rc")

# Release: optimise, and emit no debug information - neither the executable nor
# a .pdb carries symbols.  (Applied per target so the ggml sub-libraries are
# left untouched.)
if(MSVC)
    set_target_properties(seedvr2_cpp PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT "")
    target_compile_options(seedvr2_cpp PRIVATE $<$<CONFIG:Release>:/Oi /Gy /Gw>)
    target_link_options(seedvr2_cpp PRIVATE $<$<CONFIG:Release>:/INCREMENTAL:NO /OPT:REF /OPT:ICF>)
endif()

target_include_directories(seedvr2_cpp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/generated"
    "${THIRD_PARTY}"
    "${HTTPLIB_DIR}"
    "${THIRD_PARTY}/ggml/include"
    "${THIRD_PARTY}/opencv4120/include"
)

target_link_libraries(seedvr2_cpp PRIVATE
    ggml
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

target_compile_definitions(seedvr2_cpp PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    GGML_USE_CUDA
)

# copy libtorch / opencv runtime DLLs next to the exe so it can run out of the box
file(GLOB TORCH_DLLS "${THIRD_PARTY}/libtorch/lib/*.dll")
add_custom_command(TARGET seedvr2_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t "${RELEASE_DIR}"
            "${THIRD_PARTY}/opencv4120/x64/vc16/bin/opencv_world4120.dll"
            ${TORCH_DLLS}
    VERBATIM)
