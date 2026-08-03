# Raytracer

A real-time raytracer that runs on your GPU.

![Screenshot](docs/screenshot.png)

## What this is

Most 3D graphics you see in games is made by *rasterizing*. This does the opposite. For every pixel on
screen it fires a ray out into the world and follows it until it hits something,
then works out what color that surface should be by tracing more rays toward each
light.

That is a lot of rays. Doing that fast enough to move around in real time is the whole point of the project. It runs as a
CUDA program on the GPU, where thousands of pixels get traced at the same time.

## What it can do

- **Mirrors** — reflective surfaces that bounce rays up to five times, so mirrors
  show up inside other mirrors
- **Shadows** — from multiple lights at once, with optional soft edges
- **Anti-aliasing** — no jagged staircase edges on the spheres
- **A picture that improves while you sit still** — see [below](#the-image-gets-better-if-you-wait)
- **Save a screenshot** with `P`

## Requirements (at least what I used for this project)

- An NVIDIA GPU (RTX 50-series out of the box — see the note below for others)
- CUDA Toolkit 12 or newer
- SDL2
- CMake 3.28+

On Ubuntu or Debian:

```bash
sudo apt install libsdl2-dev cmake build-essential
```

CUDA comes from [NVIDIA's installer](https://developer.nvidia.com/cuda-downloads).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Using a different GPU?** Open `CMakeLists.txt` and change this line to match your
card, or just set it to `native` to auto-detect:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)   # 120 = RTX 50-series
```

## Run

```bash
./build/raytracer
```

### Controls

| Key | What it does |
|---|---|
| Drag left mouse | Look around |
| `W` `A` `S` `D` | Move around |
| `Q` / `E` | Move down / up |
| `Shift` | Move faster |
| `P` | Save a screenshot as `screenshot.ppm` |
| `Esc` | Quit |

The window title shows your framerate, how long the GPU spent on the last frame,
and how many samples have built up in the current image.

## The image gets better if you wait

This is the part worth knowing about.

A raytracer normally faces a trade-off: firing more rays per pixel makes the
picture cleaner but slows everything down. This one sidesteps that.

So:

- **While you're moving**, you get one quick sample. Maximum speed, and any
  roughness is hidden by the motion anyway.
- **The moment you stop**, samples start stacking up. After a second or so the
  image has averaged hundreds of them and looks genuinely smooth.

It's the same trick modern real-time raytracing in games uses.

It also makes normally-expensive effects cheap. Soft shadows, for example, would
usually mean firing dozens of rays at different points on a light.
## Changing the scene

Everything lives at the top of `src/main.cpp`, and it's plain data — no scene file
format to learn.

**Add a sphere.** Position, radius, material:

```cpp
{make_float3(0.0f, 1.0f, -20.0f), 2.5f, ivory},
```

**Make a material.** Color, then how matte it is, how shiny, how mirror-like
(`0` = no reflection, `1` = perfect mirror), and how tight the highlight is:

```cpp
const Material jade = make_material(make_float3(0.2f, 0.5f, 0.4f), 0.7f, 0.2f, 0.15f, 80.0f);
```

Keep color values between 0 and 1, above that a surface would be reflecting more
light than lands on it, and it'll just clip to white. To make something brighter,
turn up a light instead.

**Turn on soft shadows.** The last number on each light is its physical size.
It's at `0`, which gives crisp shadows. Give it some size and the shadow edges
go soft:

```cpp
{make_float3(-20.0f, 20.0f, 20.0f), 1.5f, 3.0f},   // position, brightness, radius
```

**Change the resolution.** Near the top of `main()`:

```cpp
constexpr int width  = 3840;   // 4K
constexpr int height = 2160;
```

## Speed

Time the GPU spends tracing one frame, on an RTX 5090:

| Resolution | Time per frame |
|---|---|
| 1280×960 | 0.07 ms |
| 1920×1080 | 0.10 ms |
| 3840×2160 (4K) | 0.28 ms |

For comparison, an earlier CPU version of this same renderer took about 23 ms at
640×480. 4K here has 27× as many pixels and still finishes roughly 80× quicker.

Tracing is no longer the slow part, copying the finished image back for display
now takes longer than drawing it.

## Project layout

| File | What's in it |
|---|---|
| `src/main.cpp` | Window, keyboard and mouse, camera, scene setup |
| `src/scene.h` | The scene building blocks: spheres, lights, materials, camera |
| `src/geom_gpu.cuh` | Vector maths, random numbers, colour handling |
| `src/render_gpu.cu` | The actual raytracing, running on the GPU |
| `src/render_gpu.h` | The handful of functions `main.cpp` calls to render a frame |

## Credits

The scene and the original CPU raytracer it grew out of were inspired by
[ssloy/tinyraytracer](https://github.com/ssloy/tinyraytracer).

And thank you to one of many [resources for raytracing](https://github.com/RayTracing/raytracing.github.io/tree/release)
