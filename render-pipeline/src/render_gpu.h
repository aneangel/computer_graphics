#ifndef RENDER_GPU_H
#define RENDER_GPU_H

// Host-side interface to the CUDA renderer. Callers never see a kernel launch.

#include <cstdint>
#include <vector>
#include "bvh.h"
#include "mesh.h"
#include "scene.h"

namespace gpu {

// Uploads the scene and allocates the frame buffers.
void init(int width, int height, const Mesh &mesh, const BVH &bvh, const Material &mesh_material,
    const std::vector<Light> &lights, const Plane &floor);

// Rewrites vertex positions, normals and BVH bounds in place. Call once per
// animation frame, before render().
void update_geometry(const Mesh &mesh, const BVH &bvh);

// Traces one frame into pinned host memory, valid until the next call. Pass
// frame 0 to restart the accumulator, then increment to keep averaging in.
const uint32_t *render(const Camera &camera, int frame, int samples_per_frame, bool smooth_normals);

// GPU time of the last render(), excluding the host copy.
float last_kernel_ms();

void shutdown();

}  // namespace gpu

#endif  // RENDER_GPU_H
