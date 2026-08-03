#ifndef RENDER_GPU_H
#define RENDER_GPU_H

// Host-side interface to the CUDA renderer. Callers never see a kernel launch.

#include <cstdint>
#include <vector>
#include "scene.h"

namespace gpu {

// Uploads the scene to constant memory and allocates the frame buffers.
void init(int width, int height, const std::vector<Sphere> &spheres, const std::vector<Light> &lights,
    const Plane &floor);

// Traces one frame. Returns width*height packed ARGB8888 pixels in pinned host
// memory, valid until the next call.
//
// frame: accumulation index. Pass 0 to restart (the camera moved), then increment
//        while it holds still and samples will keep averaging in.
// samples_per_frame: rays per pixel this frame. 1 is fastest; higher values
//        anti-alias while the camera is moving.
const uint32_t *render(const Camera &camera, int frame, int samples_per_frame);

// GPU time of the last render(), from CUDA events. Excludes the host copy.
float last_kernel_ms();

void shutdown();

}  // namespace gpu

#endif  // RENDER_GPU_H
