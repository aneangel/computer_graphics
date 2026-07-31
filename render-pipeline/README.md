# Render Pipeline

A modular real-time rendering pipeline demonstrating forward and deferred rendering techniques in OpenGL.

## Features

**Geometry & Shading**
- Forward and deferred rendering passes (switchable at runtime)
- PBR material system (metallic/roughness workflow)
- Normal and parallax occlusion mapping

**Lighting**
- Directional, point, and spot lights
- Shadow mapping with PCF soft shadows
- Image-based lighting (IBL) via HDR environment maps

**Post-processing**
- Screen-space ambient occlusion (SSAO)
- Bloom (dual-pass Kawase blur)
- HDR tone mapping (Reinhard, ACES)
- FXAA

**Performance**
- Instanced rendering
- Frustum culling
- Render state batching

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./pipeline
```

## Dependencies

- OpenGL 4.6
- GLFW 3.3
- GLM
- stb_image (included)
- Dear ImGui (included, for runtime controls)

## Controls

| Input | Action |
|---|---|
| WASD | Move camera |
| Mouse | Look |
| `Tab` | Toggle ImGui panel |
| `1` / `2` | Forward / Deferred mode |
| `P` | Toggle post-processing |
