##############################################################################
# seedvr2_vae_convert - standalone VAE (.safetensors) -> quantised GGUF tool.
#
#   cmake -S tools -B build_tools -G "Visual Studio 17 2022" -A x64
#   cmake --build build_tools --config Release -j
#   -> <repo>/release/seedvr2_vae_convert.exe
#
# CPU only: this tool needs ggml for ggml_quantize_chunk and the fp16 helpers and
# nlohmann/json for the safetensors header.  No CUDA / Vulkan / libtorch / OpenCV.
##############################################################################
cmake_minimum_required(VERSION 3.20)

project(seedvr2_vae_convert CXX C)

enable_language(RC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Paths (everything is derived from the repository root - no absolute paths)
# ---------------------------------------------------------------------------
get_filename_component(REPO_ROOT   "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(THIRD_PARTY "${REPO_ROOT}/3rd")
set(RELEASE_DIR "${REPO_ROOT}/release")
set(APP_ICON    "${REPO_ROOT}/03.ico")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         "${RELEASE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${RELEASE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${RELEASE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         "${RELEASE_DIR}")

if(MSVC)
    add_compile_options(/utf-8 /bigobj /MP /wd4244 /wd4267)
endif()

# ---------------------------------------------------------------------------
# ggml (CPU only)
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_STATIC         ON  CACHE BOOL "" FORCE)
set(GGML_AVX            ON  CACHE BOOL "" FORCE)
set(GGML_AVX2           ON  CACHE BOOL "" FORCE)
set(GGML_BMI2           ON  CACHE BOOL "" FORCE)

add_subdirectory("${THIRD_PARTY}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Application icon (shared with the main program)
# ---------------------------------------------------------------------------
configure_file("${REPO_ROOT}/src/app.rc.in" "${CMAKE_BINARY_DIR}/generated/app.rc" @ONLY)

add_executable(seedvr2_vae_convert
    vae_convert.cpp
    "${CMAKE_BINARY_DIR}/generated/app.rc"
)

if(MSVC)
    set_target_properties(seedvr2_vae_convert PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT "")
    target_compile_options(seedvr2_vae_convert PRIVATE $<$<CONFIG:Release>:/Oi /Gy /Gw>)
    target_link_options(seedvr2_vae_convert PRIVATE $<$<CONFIG:Release>:/INCREMENTAL:NO /OPT:REF /OPT:ICF>)
endif()

target_include_directories(seedvr2_vae_convert PRIVATE
    "${THIRD_PARTY}"
    "${THIRD_PARTY}/ggml/include"
)

target_link_libraries(seedvr2_vae_convert PRIVATE ggml)

target_compile_definitions(seedvr2_vae_convert PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
)
