# 3D Viewer

An interactive 3D mesh viewer supporting `.obj` and `.ply` file formats.

## Features

- **Format support:** Wavefront `.obj` (with `.mtl` materials) and `.ply` (ASCII and binary)
- **Camera:** Arcball orbit, pan, zoom with mouse
- **Shading modes:** Wireframe, flat, smooth (toggleable)
- **Overlays:** Vertex normals, face normals, bounding box
- **Mesh stats:** vertex/face/edge count, surface area, bounding box dimensions

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./viewer ../models/bunny.ply
```

## Controls

| Input | Action |
|---|---|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll | Zoom |
| `W` | Toggle wireframe |
| `N` | Toggle normals |
| `F` | Toggle flat/smooth shading |
| `R` | Reset camera |

## Dependencies

- OpenGL 4.1+
- GLFW 3.3
- GLM
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) (header-only, included)
- [happly](https://github.com/nmwsharp/happly) (PLY parser, header-only, included)
