#ifndef GEOM_GPU_CUH
#define GEOM_GPU_CUH

// Vector math, RNG, sampling and colour conversion for host and device.
// NOTE: operator*(float3, float3) is component-wise, not a dot product.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// __host__ __device__ under nvcc, plain inline otherwise.
#ifdef __CUDACC__
#define GPU_FN __host__ __device__ __forceinline__
#else
#define GPU_FN inline
#endif

// ------------------------------------------------------------------ float3 --

GPU_FN float3 operator-(const float3 &v) { return make_float3(-v.x, -v.y, -v.z); }

GPU_FN float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
GPU_FN float3 operator-(const float3 &a, const float3 &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
GPU_FN float3 operator*(const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
GPU_FN float3 operator/(const float3 &a, const float3 &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }

GPU_FN float3 operator+(const float3 &v, const float s) { return make_float3(v.x + s, v.y + s, v.z + s); }
GPU_FN float3 operator-(const float3 &v, const float s) { return make_float3(v.x - s, v.y - s, v.z - s); }
GPU_FN float3 operator*(const float3 &v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }
GPU_FN float3 operator*(const float s, const float3 &v) { return v * s; }
GPU_FN float3 operator/(const float3 &v, const float s) { return v * (1.0f / s); }

GPU_FN float3 &operator+=(float3 &a, const float3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
GPU_FN float3 &operator-=(float3 &a, const float3 &b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
GPU_FN float3 &operator*=(float3 &a, const float3 &b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
GPU_FN float3 &operator*=(float3 &a, const float s) { a.x *= s; a.y *= s; a.z *= s; return a; }

GPU_FN float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

GPU_FN float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

GPU_FN float length_squared(const float3 &v) { return dot(v, v); }
GPU_FN float length(const float3 &v) { return sqrtf(dot(v, v)); }

GPU_FN float3 normalize(const float3 &v) {
#ifdef __CUDA_ARCH__
    return v * rsqrtf(dot(v, v));  // single instruction on device
#else
    return v * (1.0f / sqrtf(dot(v, v)));
#endif
}

GPU_FN float3 fminf(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

GPU_FN float3 fmaxf(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

GPU_FN float max_component(const float3 &v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
GPU_FN float3 lerp(const float3 &a, const float3 &b, const float t) { return a + (b - a) * t; }

// ------------------------------------------------------------------ float4 --
// 16-byte aligned, so warp-wide reads and writes coalesce into whole transactions.

GPU_FN float4 to_float4(const float3 &v, const float w) { return make_float4(v.x, v.y, v.z, w); }
GPU_FN float3 xyz(const float4 &v) { return make_float3(v.x, v.y, v.z); }

GPU_FN float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// -------------------------------------------------------------------- math --

// Clamp to [0,1].
GPU_FN float saturate(const float x) {
#ifdef __CUDA_ARCH__
    return __saturatef(x);
#else
    return fminf(fmaxf(x, 0.0f), 1.0f);
#endif
}

GPU_FN float clampf(const float x, const float lo, const float hi) { return fminf(fmaxf(x, lo), hi); }

// Reduced-precision pow; fine for specular exponents and gamma curves.
GPU_FN float fast_pow(const float x, const float y) {
#ifdef __CUDA_ARCH__
    return __powf(x, y);
#else
    return powf(x, y);
#endif
}

// Mirror direction. Unit in, unit out.
GPU_FN float3 reflect(const float3 &i, const float3 &n) { return i - n * (2.0f * dot(i, n)); }

// Snell's law. eta_ratio is n_from / n_into; assumes i is normalized.
GPU_FN float3 refract(const float3 &i, const float3 &n, const float eta_ratio) {
    const float cos_theta = fminf(dot(-i, n), 1.0f);
    const float3 out_perp = (i + n * cos_theta) * eta_ratio;
    const float3 out_par = n * -sqrtf(fabsf(1.0f - length_squared(out_perp)));
    return out_perp + out_par;
}

// --------------------------------------------------------------------- rng --
// Per-thread hash-based stream. rand() cannot be used from a kernel.

struct RNG {
    uint32_t state;
};

GPU_FN uint32_t pcg_hash(const uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Seed from pixel and frame so successive frames draw different samples.
GPU_FN RNG make_rng(const uint32_t pixel_index, const uint32_t frame) {
    RNG r;
    r.state = pcg_hash(pixel_index ^ pcg_hash(frame + 1u));
    return r;
}

// Uniform in [0,1).
GPU_FN float rand_float(RNG &r) {
    r.state = pcg_hash(r.state);
    return static_cast<float>(r.state) * 2.3283064365386963e-10f;
}

// ---------------------------------------------------------------- sampling --

// Uniform on the unit sphere. Analytic, not rejection-based, to avoid warp divergence.
GPU_FN float3 random_unit_vector(RNG &r) {
    const float z = 1.0f - 2.0f * rand_float(r);
    const float rad = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 6.28318530718f * rand_float(r);
    float s, c;
    sincosf(phi, &s, &c);
    return make_float3(rad * c, rad * s, z);
}

// Cosine-weighted direction around n, for Lambertian bounces.
GPU_FN float3 cosine_hemisphere(const float3 &n, RNG &r) {
    const float3 d = n + random_unit_vector(r);
    return length_squared(d) < 1e-8f ? n : normalize(d);
}

// Random point on a spherical light's surface. radius 0 returns the centre.
GPU_FN float3 sample_sphere_light(const float3 &center, const float radius, RNG &r) {
    return center + random_unit_vector(r) * radius;
}

// ------------------------------------------------------------------- color --

GPU_FN float3 tonemap_reinhard(const float3 &c) { return c / (c + 1.0f); }

// Narkowicz ACES fit.
GPU_FN float3 tonemap_aces(const float3 &x) {
    const float3 num = x * (x * 2.51f + 0.03f);
    const float3 den = x * (x * 2.43f + 0.59f) + 0.14f;
    const float3 v = num / den;
    return make_float3(saturate(v.x), saturate(v.y), saturate(v.z));
}

// Linear to display gamma. Apply last, after any tone mapping.
GPU_FN float3 linear_to_srgb(const float3 &c) {
    const float g = 1.0f / 2.2f;
    return make_float3(fast_pow(saturate(c.x), g), fast_pow(saturate(c.y), g), fast_pow(saturate(c.z), g));
}

// Pack to SDL_PIXELFORMAT_ARGB8888.
GPU_FN uint32_t pack_argb8888(const float3 &c) {
    const uint32_t r = static_cast<uint32_t>(saturate(c.x) * 255.0f + 0.5f);
    const uint32_t g = static_cast<uint32_t>(saturate(c.y) * 255.0f + 0.5f);
    const uint32_t b = static_cast<uint32_t>(saturate(c.z) * 255.0f + 0.5f);
    return 0xFF000000u | (r << 16) | (g << 8) | b;
}

#endif  // GEOM_GPU_CUH
