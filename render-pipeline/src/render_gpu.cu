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

// Traversal depth limit, held in registers. log2(triangles) with huge margin.
constexpr int BVH_STACK = 64;
constexpr float RAY_MISS = 1e30f;

// Constant memory broadcasts to a whole warp in one cycle.
__constant__ MeshView c_mesh;
__constant__ Light c_lights[MAX_LIGHTS];
__constant__ Plane c_floor;
__constant__ int c_light_count;

// Device-owned mesh storage. Freed in shutdown().
float3 *d_positions = nullptr;
float3 *d_normals = nullptr;
int3 *d_indices = nullptr;
BVHNode *d_nodes = nullptr;
int *d_tri_index = nullptr;
// Fixed for the run, so update_geometry() need not be told them again.
size_t g_vertex_bytes = 0;
size_t g_order_bytes = 0;

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

// Moller-Trumbore. u and v come out as barycentric weights for free, which is
// what lets the same numbers blend the three corner normals for smooth shading.
__device__ bool triangle_hit(const float3 &a, const float3 &b, const float3 &c, const float3 &origin,
    const float3 &dir, float &t, float &u, float &v) {
    const float3 e1 = b - a;
    const float3 e2 = c - a;
    const float3 p = cross(dir, e2);
    const float det = dot(e1, p);

    // Magnitude not sign, which keeps the triangle double-sided.
    if (fabsf(det) < 1e-8f) return false;

    const float inv_det = 1.0f / det;
    const float3 tv = origin - a;

    u = dot(tv, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    const float3 q = cross(tv, e1);
    v = dot(dir, q) * inv_det;
    // u + v > 1 puts the point past the edge joining b and c.
    if (v < 0.0f || u + v > 1.0f) return false;

    t = dot(e2, q) * inv_det;
    return t > 0.0f;
}

// Slab test, returning entry distance so the caller can order the children. A
// zero direction component gives NaN, which fminf and fmaxf discard by design.
__device__ float ray_box(const float3 &bmin, const float3 &bmax, const float3 &origin,
    const float3 &inv_dir, const float max_dist) {
    const float tx1 = (bmin.x - origin.x) * inv_dir.x, tx2 = (bmax.x - origin.x) * inv_dir.x;
    float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);
    const float ty1 = (bmin.y - origin.y) * inv_dir.y, ty2 = (bmax.y - origin.y) * inv_dir.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    const float tz1 = (bmin.z - origin.z) * inv_dir.z, tz2 = (bmax.z - origin.z) * inv_dir.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
    // tmax > 0 rejects boxes entirely behind the ray.
    return (tmax >= tmin && tmin < max_dist && tmax > 0.0f) ? fmaxf(tmin, 0.0f) : RAY_MISS;
}

// Nearest child first, far child only if it could still hold something closer.
// That ordering, not the boxes alone, is most of why a BVH is fast.
__device__ bool mesh_intersect(const float3 &origin, const float3 &dir, float &best, int &best_tri,
    float &best_u, float &best_v) {
    const float3 inv_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int stack[BVH_STACK];
    int sp = 0;
    int node = 0;
    bool found = false;

    while (true) {
        const BVHNode n = c_mesh.nodes[node];

        if (n.tri_count > 0) {
            for (int i = 0; i < n.tri_count; ++i) {
                const int t = c_mesh.tri_index[n.left_first + i];
                const int3 f = c_mesh.indices[t];
                float tt, u, v;
                if (triangle_hit(c_mesh.positions[f.x], c_mesh.positions[f.y],
                        c_mesh.positions[f.z], origin, dir, tt, u, v) &&
                    tt < best) {
                    best = tt;
                    best_tri = t;
                    best_u = u;
                    best_v = v;
                    found = true;
                }
            }
            if (sp == 0) break;
            node = stack[--sp];
            continue;
        }

        int near_idx = n.left_first;
        int far_idx = n.left_first + 1;
        float near_d = ray_box(c_mesh.nodes[near_idx].bmin, c_mesh.nodes[near_idx].bmax, origin,
            inv_dir, best);
        float far_d = ray_box(c_mesh.nodes[far_idx].bmin, c_mesh.nodes[far_idx].bmax, origin,
            inv_dir, best);

        if (near_d > far_d) {
            const int ti = near_idx;
            near_idx = far_idx;
            far_idx = ti;
            const float td = near_d;
            near_d = far_d;
            far_d = td;
        }

        if (near_d == RAY_MISS) {
            if (sp == 0) break;
            node = stack[--sp];
        } else {
            node = near_idx;
            if (far_d != RAY_MISS) stack[sp++] = far_idx;
        }
    }
    return found;
}

// Closest-hit query. Records an index and resolves the surface once at the end.
__device__ bool scene_intersect(const float3 &origin, const float3 &dir, const bool smooth_normals,
    Hit &hit) {
    float best = 1e30f;
    int best_tri = -1;
    float best_u = 0.0f, best_v = 0.0f;

    mesh_intersect(origin, dir, best, best_tri, best_u, best_v);

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

    if (best_tri < 0 && !hit_floor) return false;

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
        const int3 f = c_mesh.indices[best_tri];
        if (smooth_normals) {
            // Weighted average of the corner normals; the silhouette stays faceted.
            const float w = 1.0f - best_u - best_v;
            hit.normal = normalize(c_mesh.normals[f.x] * w + c_mesh.normals[f.y] * best_u +
                c_mesh.normals[f.z] * best_v);
        } else {
            hit.normal = normalize(cross(c_mesh.positions[f.y] - c_mesh.positions[f.x],
                c_mesh.positions[f.z] - c_mesh.positions[f.x]));
        }
        // Double-sided hits can face away from the ray; flipping keeps them lit.
        if (dot(hit.normal, dir) > 0.0f) hit.normal = -hit.normal;
        hit.material = c_mesh.material;
    }
    return true;
}

// Any-hit walk. The first occluder ends the query, so children are not sorted.
__device__ bool mesh_occluded(const float3 &origin, const float3 &dir, const float max_dist) {
    const float3 inv_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int stack[BVH_STACK];
    int sp = 0;
    int node = 0;

    while (true) {
        const BVHNode n = c_mesh.nodes[node];

        if (n.tri_count > 0) {
            for (int i = 0; i < n.tri_count; ++i) {
                const int t = c_mesh.tri_index[n.left_first + i];
                const int3 f = c_mesh.indices[t];
                float tt, u, v;
                if (triangle_hit(c_mesh.positions[f.x], c_mesh.positions[f.y],
                        c_mesh.positions[f.z], origin, dir, tt, u, v) &&
                    tt < max_dist)
                    return true;
            }
            if (sp == 0) break;
            node = stack[--sp];
            continue;
        }

        const int l = n.left_first, r = n.left_first + 1;
        const bool want_l = ray_box(c_mesh.nodes[l].bmin, c_mesh.nodes[l].bmax, origin, inv_dir,
                                max_dist) != RAY_MISS;
        const bool want_r = ray_box(c_mesh.nodes[r].bmin, c_mesh.nodes[r].bmax, origin, inv_dir,
                                max_dist) != RAY_MISS;

        if (want_l) {
            node = l;
            if (want_r) stack[sp++] = r;
        } else if (want_r) {
            node = r;
        } else {
            if (sp == 0) break;
            node = stack[--sp];
        }
    }
    return false;
}

// Any-hit query for shadow rays: returns on the first occluder found.
__device__ bool scene_occluded(const float3 &origin, const float3 &dir, const float max_dist) {
    if (mesh_occluded(origin, dir, max_dist)) return true;

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

// throughput is the running product of reflectivities, so the loop can stop
// once further bounces cannot change the result.
__device__ float3 trace(float3 origin, float3 dir, const bool smooth_normals, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float throughput = 1.0f;

    for (int depth = 0;; ++depth) {
        Hit hit;
        if (depth > MAX_DEPTH || !scene_intersect(origin, dir, smooth_normals, hit)) {
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
    const Camera cam, const int frame, const int spp, const bool smooth_normals) {
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
            camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height),
            smooth_normals, rng);
    }
    sum = sum * (1.0f / static_cast<float>(spp));

    // Running average across frames; .w holds the sample count.
    const float4 acc = frame == 0 ? to_float4(sum, 1.0f) : accum[idx] + to_float4(sum, 1.0f);
    accum[idx] = acc;

    // Gamma only; this scene has little range above 1.0 to tone map.
    const float3 average = xyz(acc) * (1.0f / acc.w);
    out[idx] = pack_argb8888(linear_to_srgb(average));
}

}  // namespace

namespace gpu {

void init(const int width, const int height, const Mesh &mesh, const BVH &bvh,
    const Material &mesh_material, const std::vector<Light> &lights, const Plane &floor) {
    g_width = width;
    g_height = height;

    const int light_count = static_cast<int>(lights.size());
    if (light_count > MAX_LIGHTS) {
        std::fprintf(stderr, "scene exceeds constant memory capacity\n");
        std::exit(1);
    }
    if (mesh.triangle_count() == 0) {
        std::fprintf(stderr, "mesh has no triangles\n");
        std::exit(1);
    }

    // The mesh goes to global memory, then a descriptor naming it goes to
    // constant memory. Only the descriptor is small enough for the bank.
    const size_t vertex_bytes = mesh.vertex_count() * sizeof(float3);
    const size_t index_bytes = mesh.triangle_count() * sizeof(int3);
    CUDA_CHECK(cudaMalloc(&d_positions, vertex_bytes));
    CUDA_CHECK(cudaMalloc(&d_normals, vertex_bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, index_bytes));
    CUDA_CHECK(cudaMemcpy(d_positions, mesh.positions.data(), vertex_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, mesh.normals.data(), vertex_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, mesh.indices.data(), index_bytes, cudaMemcpyHostToDevice));

    // Built by the caller. Its structure is fixed from here on.
    // Sized for the 2n-1 maximum so a later rebuild, which may land on a
    // different node count, still fits without reallocating.
    const size_t max_nodes = static_cast<size_t>(2 * mesh.triangle_count() - 1);
    const size_t order_bytes = bvh.tri_index.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_nodes, max_nodes * sizeof(BVHNode)));
    CUDA_CHECK(cudaMalloc(&d_tri_index, order_bytes));
    CUDA_CHECK(cudaMemcpy(d_nodes, bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri_index, bvh.tri_index.data(), order_bytes, cudaMemcpyHostToDevice));
    std::printf("bvh: %zu nodes, %.1f KB\n", bvh.nodes.size(),
        bvh.nodes.size() * sizeof(BVHNode) / 1024.0);
    g_vertex_bytes = vertex_bytes;
    g_order_bytes = order_bytes;

    MeshView view{};
    view.positions = d_positions;
    view.normals = d_normals;
    view.indices = d_indices;
    view.nodes = d_nodes;
    view.tri_index = d_tri_index;
    view.triangle_count = static_cast<int>(mesh.triangle_count());
    view.material = mesh_material;

    CUDA_CHECK(cudaMemcpyToSymbol(c_mesh, &view, sizeof(MeshView)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_lights, lights.data(), light_count * sizeof(Light)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_floor, &floor, sizeof(Plane)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_light_count, &light_count, sizeof(int)));

    const size_t pixel_count = static_cast<size_t>(width) * height;
    CUDA_CHECK(cudaMalloc(&d_accum, pixel_count * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_pixels, pixel_count * sizeof(uint32_t)));
    // Pinned so the DMA engine reads it directly, roughly doubling copy speed.
    CUDA_CHECK(cudaHostAlloc(&h_pixels, pixel_count * sizeof(uint32_t), cudaHostAllocDefault));

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
}

void update_geometry(const Mesh &mesh, const BVH &bvh) {
    // Sizes never change, so the constant-memory descriptor stays valid.
    CUDA_CHECK(cudaMemcpy(d_positions, mesh.positions.data(), g_vertex_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, mesh.normals.data(), g_vertex_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode),
        cudaMemcpyHostToDevice));
    // A rebuild reorders triangles, a refit does not; sending it always costs
    // four bytes per triangle and saves the caller having to say which it is.
    CUDA_CHECK(cudaMemcpy(d_tri_index, bvh.tri_index.data(), g_order_bytes,
        cudaMemcpyHostToDevice));
}

const uint32_t *render(const Camera &camera, const int frame, const int samples_per_frame,
    const bool smooth_normals) {
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((g_width + BLOCK_X - 1) / BLOCK_X, (g_height + BLOCK_Y - 1) / BLOCK_Y);

    CUDA_CHECK(cudaEventRecord(ev_start));
    render_kernel<<<grid, block>>>(d_accum, d_pixels, g_width, g_height, camera, frame,
        samples_per_frame, smooth_normals);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_pixels, d_pixels,
        static_cast<size_t>(g_width) * g_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&g_kernel_ms, ev_start, ev_stop));

    return h_pixels;
}

float last_kernel_ms() { return g_kernel_ms; }

void shutdown() {
    cudaFree(d_positions);
    cudaFree(d_normals);
    cudaFree(d_indices);
    cudaFree(d_nodes);
    cudaFree(d_tri_index);
    cudaFree(d_accum);
    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

}  // namespace gpu
