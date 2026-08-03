#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <SDL2/SDL.h>
#include "anim.h"
#include "bvh.h"
#include "mesh.h"
#include "render_gpu.h"
#include "scene.h"

namespace {
    // With a BVH the cost is driven by pixels, not by mesh size.
    constexpr int width = 1280;
    constexpr int height = 720;
    constexpr float fov = M_PI / 2;

    // An animated frame cannot reuse the last frame's samples, so this is the
    // whole anti-aliasing budget during playback.
    constexpr int samples_per_frame = 4;
    // Offline renders are not racing anything, so they can afford far more.
    constexpr int offline_samples = 64;

    const float3 world_up = make_float3(0, 1, 0);
    const float3 mesh_center = make_float3(0.0f, 0.0f, -16.0f);
    constexpr float mesh_radius = 3.0f;

    float3 forward_from_angles(const float yaw, const float pitch) {
        return make_float3(cosf(pitch) * sinf(yaw), sinf(pitch), -cosf(pitch) * cosf(yaw));
    }

    // OBJ carries no unit convention, so imported models arrive any size.
    void fit_into(Mesh &m, const float3 &center, const float radius) {
        float3 lo = m.positions[0], hi = m.positions[0];
        for (const float3 &p : m.positions) {
            lo = fminf(lo, p);
            hi = fmaxf(hi, p);
        }
        const float3 mid = (lo + hi) * 0.5f;
        const float extent = fmaxf(1e-6f, max_component(hi - lo) * 0.5f);
        const float scale = radius / extent;
        // transform() scales about the origin, so the offset has to be scaled too.
        transform(m, scale, center - mid * scale);
    }

    void save_ppm(const uint32_t *pixels, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        // Otherwise a missing directory reports success having written nothing.
        if (!ofs) {
            std::cerr << "cannot write " << filename << " (does the directory exist?)\n";
            std::exit(1);
        }
        ofs << "P6\n" << width << " " << height << "\n255\n";
        std::vector<char> rgb(static_cast<size_t>(width) * height * 3);
        for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
            rgb[i * 3 + 0] = static_cast<char>(pixels[i] >> 16 & 0xFFu);
            rgb[i * 3 + 1] = static_cast<char>(pixels[i] >> 8 & 0xFFu);
            rgb[i * 3 + 2] = static_cast<char>(pixels[i] & 0xFFu);
        }
        ofs.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }

    // Refitting costs ~4 ms against ~220 ms to rebuild but yields a tree ~8x
    // slower to traverse, so it only wins while few samples are drawn from it.
    void advance_to(MeshSequence &seq, const int index, Mesh &mesh, BVH &bvh, const bool rebuild) {
        seq.read_frame(index, mesh);  // also recomputes normals
        if (rebuild)
            bvh = build_bvh(mesh);
        else
            refit_bvh(bvh, mesh);
        gpu::update_geometry(mesh, bvh);
    }

    // Frames to numbered files, no window. Nothing here is racing a display.
    int render_offline(MeshSequence &seq, Mesh &mesh, BVH &bvh, const std::string &out_dir) {
        const Camera camera = make_camera(make_float3(0, 0, 0), make_float3(0, 0, -1), world_up,
            fov, width, height);

        for (int f = 0; f < seq.frame_count(); ++f) {
            // Offline: many samples per frame, so a better tree repays its build.
            advance_to(seq, f, mesh, bvh, true);

            // Geometry is frozen, so the accumulator converges as it does when
            // the camera holds still.
            const uint32_t *pixels = nullptr;
            for (int s = 0; s < offline_samples; ++s) pixels = gpu::render(camera, s, 1, true);

            char name[512];
            std::snprintf(name, sizeof(name), "%s/frame_%04d.ppm", out_dir.c_str(), f);
            save_ppm(pixels, name);
            std::printf("\rframe %d/%d", f + 1, seq.frame_count());
            std::fflush(stdout);
        }
        std::printf("\nwrote %d frames to %s at %.0f fps\n", seq.frame_count(), out_dir.c_str(),
            seq.fps());
        gpu::shutdown();
        return 0;
    }
}

int main(int argc, char **argv) {
    const std::string input = argc > 1 ? argv[1] : "";
    const bool animated = has_sequence_extension(input);

    Mesh mesh;
    BVH bvh;
    MeshSequence sequence;

    if (animated) {
        sequence.open(input, mesh);
        std::printf("%s: %d frames at %.0f fps\n", input.c_str(), sequence.frame_count(),
            sequence.fps());
    } else if (!input.empty()) {
        mesh = load_obj(input);
        fit_into(mesh, mesh_center, mesh_radius);
        std::printf("loaded %s\n", input.c_str());
    } else {
        mesh = make_uv_sphere(mesh_center, mesh_radius, 48, 24);
    }
    std::printf("%zu vertices, %zu triangles\n", mesh.vertex_count(), mesh.triangle_count());

    // Built against the first frame; advance_to() keeps it current.
    bvh = build_bvh(mesh);

    // colour, diffuse albedo, specular albedo, reflectivity, specular exponent
    const Material ivory = make_material(make_float3(0.4f, 0.4f, 0.3f), 0.6f, 0.3f, 0.1f, 50.0f);

    // Last field is the light's radius: 0 gives hard shadows, above 0 gives soft ones.
    std::vector<Light> lights = {
        {make_float3(-20.0f, 20.0f, 20.0f), 1.5f, 0.0f},
        {make_float3(30.0f, 50.0f, -25.0f), 1.8f, 0.0f},
        {make_float3(30.0f, 50.0f, 30.0f), 1.7f, 0.0f},
    };

    Plane floor{};
    floor.y = -4.0f;
    floor.half_extent_x = 10.0f;
    floor.z_min = -30.0f;
    floor.z_max = -10.0f;
    floor.checker_scale = 0.5f;
    floor.material = make_material(make_float3(1.0f, 1.0f, 1.0f), 1.0f, 0.0f, 0.0f, 1.0f);

    gpu::init(width, height, mesh, bvh, ivory, lights, floor);

    // A second argument names an output directory and switches to offline mode.
    if (animated && argc > 2) return render_offline(sequence, mesh, bvh, argv[2]);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("render-pipeline", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_RESIZABLE);
    // No vsync flag: it would cap the loop at the monitor refresh and hide the real speed.
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_RenderSetLogicalSize(renderer, width, height);

    float3 position = make_float3(0, 0, 0);
    float yaw = 0, pitch = 0;
    bool looking = false, running = true, smooth_normals = true, playing = animated;
    int frame = 0;
    int anim_frame = 0;
    float anim_clock = 0.0f;
    const uint32_t *pixels = nullptr;
    uint64_t prev = SDL_GetPerformanceCounter();

    std::cout << "drag left mouse to look, WASD to move, Q/E down/up, shift to sprint,\n"
                 "N toggles smooth normals, space pauses playback, P to save, Esc to quit\n";

    while (running) {
        const uint64_t now = SDL_GetPerformanceCounter();
        const float dt = std::max(1e-4f, std::min(0.1f,
            static_cast<float>(now - prev) / static_cast<float>(SDL_GetPerformanceFrequency())));
        prev = now;

        const float3 prev_position = position;
        const float prev_yaw = yaw, prev_pitch = pitch;
        const bool prev_smooth = smooth_normals;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) looking = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT) looking = false;
                    break;
                case SDL_MOUSEMOTION:
                    if (looking) {
                        yaw += static_cast<float>(event.motion.xrel) * 0.004f;
                        // Must stay short of +/-pi/2, where the camera basis degenerates.
                        pitch = std::max(-1.55f, std::min(1.55f,
                            pitch - static_cast<float>(event.motion.yrel) * 0.004f));
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
                    if (event.key.keysym.scancode == SDL_SCANCODE_N) smooth_normals = !smooth_normals;
                    if (event.key.keysym.scancode == SDL_SCANCODE_SPACE && animated)
                        playing = !playing;
                    if (event.key.keysym.scancode == SDL_SCANCODE_P && pixels) {
                        save_ppm(pixels, "./screenshot.ppm");
                        std::cout << "wrote ./screenshot.ppm (" << frame + 1 << " samples/pixel)\n";
                    }
                    break;
                default:
                    break;
            }
        }

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);
        const float speed = (keys[SDL_SCANCODE_LSHIFT] ? 24.f : 8.f) * dt;
        const float3 forward = forward_from_angles(yaw, pitch);
        const float3 right = normalize(cross(forward, world_up));
        if (keys[SDL_SCANCODE_W]) position += forward * speed;
        if (keys[SDL_SCANCODE_S]) position -= forward * speed;
        if (keys[SDL_SCANCODE_A]) position -= right * speed;
        if (keys[SDL_SCANCODE_D]) position += right * speed;
        if (keys[SDL_SCANCODE_E]) position += world_up * speed;
        if (keys[SDL_SCANCODE_Q]) position -= world_up * speed;

        // Playback follows the sequence's frame rate, not the render rate.
        bool stepped = false;
        if (playing) {
            anim_clock += dt;
            const float step = 1.0f / sequence.fps();
            while (anim_clock >= step) {
                anim_clock -= step;
                anim_frame = (anim_frame + 1) % sequence.frame_count();
                stepped = true;
            }
            // Interactive: few samples per frame, so the cheap tree wins.
            if (stepped) advance_to(sequence, anim_frame, mesh, bvh, false);
        }

        // Any change to the picture restarts the average; pausing lets it settle.
        const bool moved = position.x != prev_position.x || position.y != prev_position.y ||
            position.z != prev_position.z || yaw != prev_yaw || pitch != prev_pitch ||
            smooth_normals != prev_smooth || stepped;
        frame = moved ? 0 : frame + 1;

        const Camera camera = make_camera(position, position + forward_from_angles(yaw, pitch),
            world_up, fov, width, height);

        pixels = gpu::render(camera, frame, samples_per_frame, smooth_normals);

        SDL_UpdateTexture(texture, nullptr, pixels, width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        char title[256];
        if (animated)
            snprintf(title, sizeof(title),
                "render-pipeline  %zu tris  frame %d/%d %s  %.1f fps  kernel %.2f ms",
                mesh.triangle_count(), anim_frame + 1, sequence.frame_count(),
                playing ? "playing" : "paused", 1.f / dt, gpu::last_kernel_ms());
        else
            snprintf(title, sizeof(title),
                "render-pipeline  %zu tris  %s  %.1f fps  kernel %.2f ms  %d spp",
                mesh.triangle_count(), smooth_normals ? "smooth" : "flat", 1.f / dt,
                gpu::last_kernel_ms(), (frame + 1) * samples_per_frame);
        SDL_SetWindowTitle(window, title);
    }

    gpu::shutdown();
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
