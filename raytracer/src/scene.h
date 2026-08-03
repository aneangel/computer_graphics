#ifndef SCENE_H
#define SCENE_H

// Scene types shared by host and device. All POD, so they can be copied
// straight into __constant__ memory with cudaMemcpyToSymbol.

#include "geom_gpu.cuh"

// Constant memory is a fixed bank, so these arrays are sized at compile time.
constexpr int MAX_SPHERES = 32;
constexpr int MAX_LIGHTS = 8;

struct Material {
    float3 diffuse_color;
    float diffuse_albedo;
    float specular_albedo;
    float reflectivity;  // 0 = matte, 1 = perfect mirror
    float specular_exponent;
};

inline Material make_material(const float3 color, const float kd, const float ks, const float kr,
    const float exponent) {
    return Material{color, kd, ks, kr, exponent};
}

struct Sphere {
    float3 center;
    float radius;
    Material material;
};

// radius 0 is a point light (hard shadows); above 0 gives soft shadows.
struct Light {
    float3 position;
    float intensity;
    float radius;
};

// Finite axis-aligned checkerboard floor at constant y.
struct Plane {
    float y;
    float half_extent_x;
    float z_min;
    float z_max;
    float checker_scale;
    Material material;
};

// fov_scale and aspect are precomputed here so the kernel does not redo them per pixel.
struct Camera {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;
    float fov_scale;
    float aspect;
};

inline Camera make_camera(const float3 position, const float3 target, const float3 world_up,
    const float fov, const int width, const int height) {
    Camera c;
    c.position = position;
    c.forward = normalize(target - position);
    c.right = normalize(cross(c.forward, world_up));
    c.up = cross(c.right, c.forward);
    c.fov_scale = tanf(fov * 0.5f);
    c.aspect = static_cast<float>(width) / static_cast<float>(height);
    return c;
}

// Primary ray direction. px/py are continuous, so passing x + jitter anti-aliases.
GPU_FN float3 camera_ray(const Camera &c, const float px, const float py, const int width,
    const int height) {
    const float sx = (2.0f * px / static_cast<float>(width) - 1.0f) * c.fov_scale * c.aspect;
    const float sy = -(2.0f * py / static_cast<float>(height) - 1.0f) * c.fov_scale;
    return normalize(c.right * sx + c.up * sy + c.forward);
}

#endif  // SCENE_H
