#include <cstdio>
#include <cstdlib>
#include "render_gpu.h"

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                  \
                cudaGetErrorString(err_));                                                      \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

constexpr int MAX_DEPTH = 4;
constexpr float MIN_THROUGHPUT = 1e-3f;
constexpr float SURFACE_EPS = 1e-3f;

// 8x8 puts neighbouring pixels in the same warp, which keeps ray paths coherent.
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;

// Constant memory: every thread in a warp reads the same address, which is
// broadcast in one cycle.
__constant__ Sphere c_spheres[MAX_SPHERES];
__constant__ Light c_lights[MAX_LIGHTS];
__constant__ Plane c_floor;
__constant__ int c_sphere_count;
__constant__ int c_light_count;

float4 *d_accum = nullptr;
uint32_t *d_pixels = nullptr;
uint32_t *h_pixels = nullptr;
int g_width = 0;
int g_height = 0;
cudaEvent_t ev_start, ev_stop;
float g_kernel_ms = 0.0f;

struct Hit {
    float3 point;
    float3 normal;
    Material material;
    float t;
};

// Analytic ray-sphere test. t is the distance along dir, which is normalized.
__device__ bool sphere_hit(const Sphere &s, const float3 &origin, const float3 &dir, float &t) {
    const float3 L = s.center - origin;
    const float tca = dot(L, dir);
    const float r2 = s.radius * s.radius;
    const float d2 = dot(L, L) - tca * tca;
    if (d2 > r2) return false;
    const float thc = sqrtf(r2 - d2);
    t = tca - thc;                // front of the sphere
    if (t < 0.0f) t = tca + thc;  // back, only when the origin is inside
    return t >= 0.0f;
}

// Closest-hit query. Records an index and resolves the surface once at the end.
__device__ bool scene_intersect(const float3 &origin, const float3 &dir, Hit &hit) {
    float best = 1e30f;
    int best_sphere = -1;

    for (int i = 0; i < c_sphere_count; ++i) {
        float t;
        if (sphere_hit(c_spheres[i], origin, dir, t) && t < best) {
            best = t;
            best_sphere = i;
        }
    }

    bool hit_floor = false;
    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (c_floor.y - origin.y) / dir.y; d > 0.0f && d < best) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < c_floor.half_extent_x && pt.z > c_floor.z_min && pt.z < c_floor.z_max) {
                best = d;
                hit_floor = true;
            }
        }
    }

    if (best_sphere < 0 && !hit_floor) return false;

    hit.t = best;
    hit.point = origin + dir * best;

    if (hit_floor) {
        hit.normal = make_float3(0.0f, 1.0f, 0.0f);
        hit.material = c_floor.material;
        const int cx = static_cast<int>(floorf(hit.point.x * c_floor.checker_scale));
        const int cz = static_cast<int>(floorf(hit.point.z * c_floor.checker_scale));
        const float3 tint = ((cx + cz) & 1) ? make_float3(1.0f, 1.0f, 1.0f) : make_float3(1.0f, 0.7f, 0.3f);
        hit.material.diffuse_color = tint * 0.3f;
    } else {
        hit.normal = normalize(hit.point - c_spheres[best_sphere].center);
        hit.material = c_spheres[best_sphere].material;
    }
    return true;
}

// Any-hit query for shadow rays: returns on the first occluder found.
__device__ bool scene_occluded(const float3 &origin, const float3 &dir, const float max_dist) {
    for (int i = 0; i < c_sphere_count; ++i) {
        float t;
        if (sphere_hit(c_spheres[i], origin, dir, t) && t < max_dist) return true;
    }

    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (c_floor.y - origin.y) / dir.y; d > 0.0f && d < max_dist) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < c_floor.half_extent_x && pt.z > c_floor.z_min && pt.z < c_floor.z_max)
                return true;
        }
    }
    return false;
}

// Vertical gradient background.
__device__ float3 sky(const float3 &dir) {
    const float t = saturate(dir.y * 0.5f + 0.5f);
    return lerp(make_float3(0.35f, 0.52f, 0.70f), make_float3(0.10f, 0.26f, 0.60f), t);
}

// Nudges a new ray off the surface it started on to avoid self-intersection.
__device__ float3 offset_origin(const float3 &point, const float3 &normal, const float3 &dir) {
    return point + normal * (dot(dir, normal) < 0.0f ? -SURFACE_EPS : SURFACE_EPS);
}

// Iterative path loop. `throughput` is the running product of reflectivities, so
// the loop can stop once further bounces cannot affect the result.
__device__ float3 trace(float3 origin, float3 dir, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float throughput = 1.0f;

    for (int depth = 0;; ++depth) {
        Hit hit;
        if (depth > MAX_DEPTH || !scene_intersect(origin, dir, hit)) {
            radiance += sky(dir) * throughput;
            break;
        }

        float diffuse = 0.0f;
        float specular = 0.0f;
        for (int i = 0; i < c_light_count; ++i) {
            // Sampling a random point on the light is what softens the shadow.
            const float3 lp = sample_sphere_light(c_lights[i].position, c_lights[i].radius, rng);
            const float3 to_light = lp - hit.point;
            const float light_distance = length(to_light);
            const float3 light_dir = to_light * (1.0f / light_distance);

            if (scene_occluded(offset_origin(hit.point, hit.normal, light_dir), light_dir,
                light_distance))
                continue;

            diffuse += c_lights[i].intensity * fmaxf(0.0f, dot(light_dir, hit.normal));
            specular += fast_pow(fmaxf(0.0f, dot(reflect(light_dir, hit.normal), dir)),
                            hit.material.specular_exponent) * c_lights[i].intensity;
        }

        const float3 direct = hit.material.diffuse_color * (diffuse * hit.material.diffuse_albedo) +
            make_float3(1.0f, 1.0f, 1.0f) * (specular * hit.material.specular_albedo);
        radiance += direct * throughput;

        throughput *= hit.material.reflectivity;
        if (throughput < MIN_THROUGHPUT) break;

        dir = reflect(dir, hit.normal);
        origin = offset_origin(hit.point, hit.normal, dir);
    }
    return radiance;
}

// One thread per pixel: jitter, trace, accumulate, convert, pack.
__global__ void render_kernel(float4 *accum, uint32_t *out, const int width, const int height,
    const Camera cam, const int frame, const int spp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;  // the grid rounds up to whole blocks

    const int idx = y * width + x;
    RNG rng = make_rng(idx, frame);

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < spp; ++s) {
        const float jx = rand_float(rng);
        const float jy = rand_float(rng);
        sum += trace(cam.position,
            camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height), rng);
    }
    sum = sum * (1.0f / static_cast<float>(spp));

    // Running average across frames. .w holds the sample count.
    const float4 acc = frame == 0 ? to_float4(sum, 1.0f) : accum[idx] + to_float4(sum, 1.0f);
    accum[idx] = acc;

    // Gamma only. This scene has little range above 1.0, so a tone map would
    // just flatten contrast; reinstate one if the lighting starts to clip.
    const float3 average = xyz(acc) * (1.0f / acc.w);
    out[idx] = pack_argb8888(linear_to_srgb(average));
}

}  // namespace

namespace gpu {

void init(const int width, const int height, const std::vector<Sphere> &spheres,
    const std::vector<Light> &lights, const Plane &floor) {
    g_width = width;
    g_height = height;

    const int sphere_count = static_cast<int>(spheres.size());
    const int light_count = static_cast<int>(lights.size());
    if (sphere_count > MAX_SPHERES || light_count > MAX_LIGHTS) {
        std::fprintf(stderr, "scene exceeds constant memory capacity\n");
        std::exit(1);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_spheres, spheres.data(), sphere_count * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_lights, lights.data(), light_count * sizeof(Light)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_floor, &floor, sizeof(Plane)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sphere_count, &sphere_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_light_count, &light_count, sizeof(int)));

    const size_t pixel_count = static_cast<size_t>(width) * height;
    CUDA_CHECK(cudaMalloc(&d_accum, pixel_count * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_pixels, pixel_count * sizeof(uint32_t)));
    // Pinned host memory: the DMA engine reads it directly, roughly doubling
    // device-to-host transfer speed.
    CUDA_CHECK(cudaHostAlloc(&h_pixels, pixel_count * sizeof(uint32_t), cudaHostAllocDefault));

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
}

const uint32_t *render(const Camera &camera, const int frame, const int samples_per_frame) {
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((g_width + BLOCK_X - 1) / BLOCK_X, (g_height + BLOCK_Y - 1) / BLOCK_Y);

    CUDA_CHECK(cudaEventRecord(ev_start));
    render_kernel<<<grid, block>>>(d_accum, d_pixels, g_width, g_height, camera, frame,
        samples_per_frame);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_pixels, d_pixels,
        static_cast<size_t>(g_width) * g_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&g_kernel_ms, ev_start, ev_stop));

    return h_pixels;
}

float last_kernel_ms() { return g_kernel_ms; }

void shutdown() {
    cudaFree(d_accum);
    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

}  // namespace gpu
