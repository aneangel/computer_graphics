cmake_minimum_required(VERSION 3.28)
project(raytracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer
    src/main.cpp)

find_package(SDL2 REQUIRED)
target_link_libraries(raytracer PRIVATE SDL2::SDL2)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raytracer PRIVATE OpenMP::OpenMP_CXX)
endif()
