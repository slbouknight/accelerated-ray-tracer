cmake_minimum_required(VERSION 3.25)
project(AcceleratedRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Single-config generators (Ninja, Makefiles) default to an *empty* build type,
# which means no optimisation at all. Multi-config generators (Visual Studio)
# ignore this and pick the config at build time via --config.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  message(STATUS "No CMAKE_BUILD_TYPE set; defaulting to Release.")
endif()

# "native" asks nvcc to detect the installed GPU's compute capability. The old
# hard-coded 89 (Ada) meant anything newer ran through PTX JIT at startup, and
# anything older would not run at all. Override on the command line if you need
# a portable fat binary, e.g. -DRT_CUDA_ARCH="75;86;89;120".
set(RT_CUDA_ARCH "native" CACHE STRING "CMAKE_CUDA_ARCHITECTURES value")
set(CMAKE_CUDA_ARCHITECTURES ${RT_CUDA_ARCH})

find_package(CUDAToolkit REQUIRED)

enable_testing()

# ---------------------------------------------------------------------------
# Shared compile settings
# ---------------------------------------------------------------------------
add_library(rt_flags INTERFACE)
target_compile_options(rt_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -diag-suppress=20199              # silence CURAND "unrecognized #pragma"
    # -lineinfo keeps source correlation for Nsight Compute without the
    # catastrophic slowdown of -G (device debug). Safe in optimised builds.
    $<$<NOT:$<CONFIG:Debug>>:-lineinfo>
  >
)
if(MSVC)
  target_compile_options(rt_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4312 -Xcompiler=/wd4311 -Xcompiler=/wd4996>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4312 /wd4311 /wd4996>
  )
endif()
target_link_libraries(rt_flags INTERFACE CUDA::cudart)

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
add_executable(rayTracer src/main.cu)
target_link_libraries(rayTracer PRIVATE rt_flags)
set_property(TARGET rayTracer PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)

# The renderer opens textures by the relative path "textures/<name>.jpg", so a
# copy has to sit next to the executable. A POST_BUILD copy (rather than
# configure_file) uses a generator expression for the output directory, so it
# lands in the right place for every configuration of a multi-config generator
# -- build/bin/Debug and build/bin/Release both get one.
add_custom_command(TARGET rayTracer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/textures" "$<TARGET_FILE_DIR:rayTracer>/textures"
  COMMENT "Copying textures next to rayTracer"
  VERBATIM)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(test_math tests/test_math.cu)
target_link_libraries(test_math PRIVATE rt_flags)
set_property(TARGET test_math PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)

add_executable(test_device tests/test_device.cu)
target_link_libraries(test_device PRIVATE rt_flags)
set_property(TARGET test_device PROPERTY CUDA_RUNTIME_LIBRARY Hybrid)

add_test(NAME math   COMMAND test_math)
add_test(NAME device COMMAND test_device)

# test_device returns 77 when no GPU is present, so CTest reports "skipped"
# rather than a hard failure on a CI box without a device.
set_tests_properties(device PROPERTIES SKIP_RETURN_CODE 77)

# Golden-image regression: renders every scene small and diffs against a
# committed reference. Slower than the unit tests (tens of seconds), so it gets
# a longer timeout and its own label for `ctest -L unit`.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME golden
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_golden.py
                   --exe $<TARGET_FILE:rayTracer>)
  set_tests_properties(golden PROPERTIES TIMEOUT 900 DEPENDS device)
else()
  message(STATUS "Python3 not found; skipping the golden-image test.")
endif()

set_tests_properties(math device PROPERTIES LABELS unit)
