cmake_minimum_required(VERSION 3.25)
project(AcceleratedRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

add_executable(rayTracer src/main.cu)

# Prefer dynamic (shared) cudart for VS Debug; 'Hybrid' is fine too.
set(CMAKE_CUDA_RUNTIME_LIBRARY Hybrid)
set_property(TARGET rayTracer PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)

target_compile_options(rayTracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

target_link_libraries(rayTracer PRIVATE CUDA::cudart)
