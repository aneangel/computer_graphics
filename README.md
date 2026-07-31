# Graphics Showcase

A collection of computer graphics projects spanning rendering, geometry processing, and real-time pipelines — built in C++/OpenGL.

---

## Projects

### [Raytracer](./raytracer)
A CPU-based raytracer implementing physically based rendering concepts from scratch.

**Features**
- Recursive ray-object intersection (spheres, planes, triangles)
- Phong and physically based shading models
- Reflection, refraction, and shadows
- Anti-aliasing via supersampling
- BVH acceleration structure

**Stack:** C++, no external dependencies

---

### [3D Viewer](./3d-viewer)
An interactive viewer for loading and inspecting `.obj` and `.ply` mesh files.

**Features**
- `.obj` and `.ply` parser (ASCII + binary PLY)
- Orbit/pan/zoom camera controls
- Wireframe, flat, and smooth shading modes
- Normal visualization and mesh statistics

**Stack:** C++, OpenGL, GLFW, GLM

---

### [Render Pipeline](./render-pipeline)
A modular forward and deferred rendering pipeline demonstrating core real-time graphics techniques.

**Features**
- Forward and deferred rendering passes
- Shadow mapping (PCF soft shadows)
- Normal mapping and PBR material system
- Post-processing: bloom, tone mapping, SSAO
- Instanced rendering

**Stack:** C++, OpenGL 4.6, GLFW, GLM, stb_image

---

## Building

Each project is self-contained with its own build instructions. See the `README.md` inside each subdirectory.

**Common dependencies**
```
cmake >= 3.20
g++ >= 11 or clang >= 13
OpenGL (for 3d-viewer and render-pipeline)
```

---

## Structure

```
graphics-showcase/
├── raytracer/          # CPU raytracer
│   ├── src/
│   ├── scenes/         # example scene configs
│   └── output/         # rendered images
├── 3d-viewer/          # .obj/.ply mesh viewer
│   ├── src/
│   ├── shaders/
│   └── models/         # sample meshes
└── render-pipeline/    # real-time render pipeline
    ├── src/
    ├── shaders/
    └── assets/
```
