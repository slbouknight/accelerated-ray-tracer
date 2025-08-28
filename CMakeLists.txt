cmake_minimum_required(VERSION 3.25)
project(AcceleratedRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

add_executable(rayTracer src/main.cu)
target_compile_options(rayTracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20199>
)

# Gather all .png and .jpg files in the textures directory
file(GLOB TEXTURE_FILES
    "${CMAKE_SOURCE_DIR}/textures/*.png"
    "${CMAKE_SOURCE_DIR}/textures/*.jpg"
)

# Copy each texture to the build directory (where exe runs from)
foreach(TEXTURE_FILE ${TEXTURE_FILES})
    get_filename_component(FILENAME ${TEXTURE_FILE} NAME)
    configure_file(${TEXTURE_FILE}
                   ${CMAKE_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

# Prefer dynamic (shared) cudart for VS Debug; 'Hybrid' is fine too.
set(CMAKE_CUDA_RUNTIME_LIBRARY Hybrid)
set_property(TARGET rayTracer PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)

target_compile_options(rayTracer PRIVATE
  # CUDA flags
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -diag-suppress=20199              # silence CURAND "unrecognized #pragma"
    -Xcompiler=/wd4312                # silence stb_image pointer-cast warnings
    -Xcompiler=/wd4311
    -Xcompiler=/wd4996
  >
  # Host C++ flags (for any .cpp/.cu host pass compiled by MSVC)
  $<$<COMPILE_LANGUAGE:CXX>:
    /wd4312 /wd4311 /wd4996
  >
)

target_link_libraries(rayTracer PRIVATE CUDA::cudart)
