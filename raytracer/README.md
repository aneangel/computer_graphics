# Raytracer

A CPU-based raytracer built from scratch implementing core physically based rendering concepts.

## Features

- Ray-object intersection: spheres, planes, triangles, meshes
- Shading: Phong, Blinn-Phong, and PBR (Cook-Torrance)
- Reflection and refraction (Fresnel via Schlick approximation)
- Soft shadows via area lights
- Anti-aliasing: stratified supersampling
- BVH (Bounding Volume Hierarchy) acceleration structure
- Multithreaded tile rendering

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./raytracer ../scenes/cornell_box.json
```

## Output

Renders are written to `output/` as `.ppm` files.
