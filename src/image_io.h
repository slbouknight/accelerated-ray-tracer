#pragma once
#include <cuda_runtime.h>

struct DeviceImage 
{
    const unsigned char* data = nullptr;
    int width = 0, height = 0, bpp = 3;
    __host__ __device__ bool valid() const { return data && width>0 && height>0 && bpp>=3; }
};

#if !defined(__CUDA_ARCH__)
  #include "../external/stb_image.h"

  inline DeviceImage load_image_to_device(const char* path, unsigned char** d_pixels_out=nullptr) 
  {
      int w=0,h=0,n=0;
      unsigned char* h_pixels = stbi_load(path, &w, &h, &n, 3);
      if (!h_pixels) return {};

      size_t bytes = size_t(w)*h*3;
      unsigned char* d_pixels = nullptr;
      cudaMalloc(&d_pixels, bytes);
      cudaMemcpy(d_pixels, h_pixels, bytes, cudaMemcpyHostToDevice);
      stbi_image_free(h_pixels);

      if (d_pixels_out) *d_pixels_out = d_pixels;
      return DeviceImage{ d_pixels, w, h, 3 };
  }

  inline void free_device_image(DeviceImage& img) 
  {
      if (img.data) cudaFree(const_cast<unsigned char*>(img.data));
      img = DeviceImage{};
  }
#endif
