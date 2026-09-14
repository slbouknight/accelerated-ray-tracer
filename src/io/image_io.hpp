#pragma once

#include "../core/cuda_compat.hpp"

#include <cuda_runtime.h>
#include <string>
#include <vector>

// A device-resident image, passed to kernels by value. Only a view: it does not
// own the pixels, so copying it is free and the lifetime is managed by whoever
// called load_image_to_device.
struct DeviceImage {
    const unsigned char* data = nullptr;
    int width  = 0;
    int height = 0;
    int bpp    = 3;
    RT_HD inline bool valid() const {
        return data && width > 0 && height > 0 && bpp >= 3;
    }
};

// Declarations are visible to both compiler passes; the definitions below are
// host-only, which keeps stb_image and the CUDA runtime calls out of the device
// pass entirely.
__host__ DeviceImage load_image_to_device(const char* path);
__host__ void        free_device_image(DeviceImage& img);
__host__ void        set_asset_search_root(const char* exe_path);

#if !defined(__CUDA_ARCH__)
  #include <cstdio>
  #include <iostream>
  #include "../../external/stb_image.h"

  namespace rt_io {

  inline std::string& asset_root() { static std::string r; return r; }

  inline bool file_exists(const std::string& p) {
      if (std::FILE* f = std::fopen(p.c_str(), "rb")) { std::fclose(f); return true; }
      return false;
  }

  // Resolve a scene's relative asset path ("textures/earthmap.jpg") against the
  // current directory first, then against the executable's directory and its
  // parents. Without this the renderer only works when launched from exactly
  // the right cwd, which made it awkward to drive from scripts and from CTest.
  inline std::string resolve_asset(const char* path) {
      if (file_exists(path)) return path;
      std::string base = asset_root();
      for (int up = 0; up < 4 && !base.empty(); ++up) {
          const std::string candidate = base + "/" + path;
          if (file_exists(candidate)) return candidate;
          const size_t slash = base.find_last_of("/\\");
          if (slash == std::string::npos) break;
          base = base.substr(0, slash);
      }
      return path;   // let the caller report the original name in its error
  }

  } // namespace rt_io

  // Call once at startup with argv[0] so relative asset paths can be resolved
  // against the executable's location as well as the working directory.
  inline void set_asset_search_root(const char* exe_path) {
      const std::string s = exe_path ? exe_path : "";
      const size_t slash = s.find_last_of("/\\");
      rt_io::asset_root() = (slash == std::string::npos) ? "." : s.substr(0, slash);
  }

  inline DeviceImage load_image_to_device(const char* path) {
      const std::string resolved = rt_io::resolve_asset(path);

      int w = 0, h = 0, n = 0;
      unsigned char* h_pixels = stbi_load(resolved.c_str(), &w, &h, &n, 3);
      if (!h_pixels) {
          std::cerr << "stbi_load failed for '" << path << "': "
                    << stbi_failure_reason() << "\n";
          return {};
      }

      const size_t bytes = static_cast<size_t>(w) * h * 3;
      unsigned char* d_pixels = nullptr;
      if (cudaMalloc(&d_pixels, bytes) != cudaSuccess) {
          std::cerr << "cudaMalloc failed for texture '" << path << "'\n";
          stbi_image_free(h_pixels);
          return {};
      }
      if (cudaMemcpy(d_pixels, h_pixels, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
          std::cerr << "cudaMemcpy failed for texture '" << path << "'\n";
          cudaFree(d_pixels);
          stbi_image_free(h_pixels);
          return {};
      }
      stbi_image_free(h_pixels);

      return DeviceImage{ d_pixels, w, h, 3 };
  }

  inline void free_device_image(DeviceImage& img) {
      if (img.data) cudaFree(const_cast<unsigned char*>(img.data));
      img = DeviceImage{};
  }
#endif
