cmake_minimum_required(VERSION 3.28)

# Must precede project(): with no build type CMake adds no -O flag and GCC defaults to -O0.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
endif()

project(raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 120 = compute capability 12.0 (RTX 50-series). "native" works on CUDA 12.6+.
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(raytracer
    src/main.cpp
    src/render_gpu.cu
    src/render_gpu.h
    src/scene.h
    src/geom_gpu.cuh)

find_package(SDL2 REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(raytracer PRIVATE SDL2::SDL2 CUDA::cudart)

# The COMPILE_LANGUAGE guards are required: nvcc rejects -march=native and -ffast-math.
target_compile_options(raytracer PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native -ffast-math>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>
    # Prints registers per thread and spill counts; a nonzero stack frame means
    # something fell out to local memory.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
