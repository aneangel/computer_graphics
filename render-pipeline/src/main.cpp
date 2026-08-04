#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <SDL2/SDL.h>
#include "anim.h"
#include "bvh.h"
#include "mesh.h"
#include "render_gpu.h"
#include "scene.h"

namespace {
    // Output size. With a BVH the cost follows pixels, not triangles.
    constexpr int width = 1280;
    constexpr int height = 720;
    // Vertical field of view. Narrow enough that a wide scene still reads.
    constexpr float fov = M_PI / 3;

    // Samples per pixel during playback, where each frame starts from nothing.
    constexpr int samples_per_frame = 4;
    // Samples per pixel offline, where there is time to spare.
    constexpr int offline_samples = 64;

    const float3 world_up = make_float3(0, 1, 0);
    const float3 mesh_center = make_float3(0.0f, 0.0f, -16.0f);
    constexpr float mesh_radius = 3.0f;

    // View direction from yaw and pitch.
    float3 forward_from_angles(const float yaw, const float pitch) {
        return make_float3(cosf(pitch) * sinf(yaw), sinf(pitch), -cosf(pitch) * cosf(yaw));
    }

    // Scales and centres a loaded model, which arrives in no particular units.
    void fit_into(Mesh &m, const float3 &center, const float radius) {
        float3 lo = m.positions[0], hi = m.positions[0];
        for (const float3 &p : m.positions) {
            lo = fminf(lo, p);
            hi = fmaxf(hi, p);
        }
        const float3 mid = (lo + hi) * 0.5f;
        const float extent = fmaxf(1e-6f, max_component(hi - lo) * 0.5f);
        const float scale = radius / extent;
        // transform() scales about the origin, so the offset scales too.
        transform(m, scale, center - mid * scale);
    }

    // Camera as a point to look at and an angle to look from, so it can orbit.
    struct View {
        float3 centre;
        float distance;
        float elevation;
        float azimuth;
    };

    // The camera `turn` radians around the view.
    Camera camera_at(const View &v, const float turn) {
        const float a = v.azimuth + turn;
        const float3 offset = make_float3(sinf(a) * cosf(v.elevation), sinf(v.elevation),
            cosf(a) * cosf(v.elevation));
        return make_camera(v.centre + offset * v.distance, v.centre, world_up, fov, width, height);
    }

    // Picks a view that fits the given bounds in frame.
    View frame_camera(const float3 &lo, const float3 &hi) {
        const float3 centre = (lo + hi) * 0.5f;
        const float3 extent = hi - lo;
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float tan_half = tanf(fov * 0.5f);

        // Fits the box rather than a sphere round it, which wastes less picture.
        const float for_width = extent.x * 0.5f / (tan_half * aspect);
        const float for_height = extent.y * 0.5f / tan_half;
        // Plus half the depth, so the camera clears the near end of the scene.
        const float distance = fmaxf(for_width, for_height) * 1.1f + extent.z * 0.5f;

        // Raised a little, which shows how things are arranged in depth.
        return View{centre, fmaxf(1e-3f, distance), atanf(0.38f), 0.0f};
    }

    // Writes one frame as a binary PPM.
    void save_ppm(const uint32_t *pixels, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        // Without this a missing directory reports success having written nothing.
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

    // Moves to one frame of a sequence. Rebuilding gives a faster tree than refitting.
    void advance_to(MeshSequence &seq, const int index, Mesh &mesh, BVH &bvh, const bool rebuild) {
        seq.read_frame(index, mesh);  // also recomputes normals
        if (rebuild)
            bvh = build_bvh(mesh);
        else
            refit_bvh(bvh, mesh);
        gpu::update_geometry(mesh, bvh);
    }

    // True if the path names a video rather than a mesh or a sequence.
    bool has_video_extension(const std::string &path) {
        for (const char *ext : {".mp4", ".mov", ".avi", ".mkv", ".webm"}) {
            const size_t n = std::string(ext).size();
            if (path.size() > n && path.compare(path.size() - n, n, ext) == 0) return true;
        }
        return false;
    }

    // Directory the binary lives in, so the tools beside it are always findable.
    std::string exe_dir() {
        char buf[4096];
        const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n <= 0) return ".";
        buf[n] = '\0';
        const std::string path(buf);
        return path.substr(0, path.find_last_of('/'));
    }

    // Runs the reconstruction over a video and returns the sequence it wrote.
    std::string convert_video(const std::string &video) {
        const std::string root = exe_dir() + "/..";
        const std::string name = video.substr(video.find_last_of('/') + 1);
        const std::string stem = name.substr(0, name.find_last_of('.'));
        const std::string out = root + "/.cache/" + stem + ".msq";

        std::ifstream cached(out, std::ios::binary);
        if (cached.good()) {
            std::printf("cached: %s\n", out.c_str());
            return out;
        }

        // A subprocess, because the estimator is a large Python stack.
        const std::string cmd = "python3 '" + root + "/tools/video_to_scene.py' '" + video +
            "' --out '" + out + "' --work '" + root + "/.cache/" + stem + ".work'";
        std::printf("reconstructing %s\n", video.c_str());
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "reconstruction failed. see tools/video_to_scene.py --help\n";
            std::exit(1);
        }
        return out;
    }

    // Frames to numbered files, no window needed.
    int render_offline(MeshSequence &seq, Mesh &mesh, BVH &bvh, const std::string &out_dir,
        const View &view, const bool animated, const float turns, const int samples) {
        // A still model has no length of its own, so an orbit is what gives it one.
        const int frames = animated ? seq.frame_count() : (turns > 0.0f ? 120 : 1);
        for (int f = 0; f < frames; ++f) {
            // Many samples per frame here, so a rebuilt tree repays its cost.
            if (animated) advance_to(seq, f, mesh, bvh, true);

            const float turn = frames > 1
                ? turns * 2.0f * static_cast<float>(M_PI) * static_cast<float>(f) /
                    static_cast<float>(frames)
                : 0.0f;
            const Camera camera = camera_at(view, turn);

            // Nothing moves within a frame, so the average keeps converging.
            const uint32_t *pixels = nullptr;
            for (int s = 0; s < samples; ++s) pixels = gpu::render(camera, s, 1, true);

            char name[512];
            std::snprintf(name, sizeof(name), "%s/frame_%04d.ppm", out_dir.c_str(), f);
            save_ppm(pixels, name);
            std::printf("\rframe %d/%d", f + 1, frames);
            std::fflush(stdout);
        }
        std::printf("\nwrote %d frames to %s at %.0f fps\n", frames, out_dir.c_str(),
            animated ? seq.fps() : 0.0f);
        gpu::shutdown();
        return 0;
    }
}

int main(int argc, char **argv) {
    std::string input = argc > 1 ? argv[1] : "";
    // A video becomes a sequence first, then is handled like any other.
    if (has_video_extension(input)) input = convert_video(input);
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

    // A scene gives a colour per actor; how it shades is decided here.
    std::vector<Material> materials{ivory};
    std::vector<int> tri_material;
    if (animated && sequence.objects().size() > 1) {
        materials.clear();
        for (const SequenceObject &o : sequence.objects()) {
            Material m = ivory;
            m.diffuse_color = o.color;
            materials.push_back(m);
        }
        tri_material = sequence.triangle_materials();
        std::printf("%zu objects\n", materials.size());
    }

    // Last field is the light's radius: 0 gives hard shadows, above 0 soft ones.
    std::vector<Light> lights = {
        {make_float3(-20.0f, 20.0f, 20.0f), 1.5f, 0.0f},
        {make_float3(30.0f, 50.0f, -25.0f), 1.8f, 0.0f},
        {make_float3(30.0f, 50.0f, 30.0f), 1.7f, 0.0f},
    };

    // Whole sequence, since the floor has to cover wherever the actors get to.
    float3 lo = mesh.positions[0], hi = mesh.positions[0];
    if (animated) {
        sequence.bounds(lo, hi);
    } else {
        for (const float3 &p : mesh.positions) {
            lo = fminf(lo, p);
            hi = fmaxf(hi, p);
        }
    }
    const float margin = fmaxf(2.0f, 0.12f * max_component(hi - lo));

    // Floor sized to the scene. Its height is fixed, and is where the tools put feet.
    Plane floor{};
    floor.y = -4.0f;
    floor.half_extent_x = fmaxf(fabsf(lo.x), fabsf(hi.x)) + margin;
    floor.z_min = lo.z - margin;
    floor.z_max = fminf(-1.0f, hi.z + margin);  // never in front of the camera
    floor.checker_scale = 0.5f;
    floor.material = make_material(make_float3(1.0f, 1.0f, 1.0f), 1.0f, 0.0f, 0.0f, 1.0f);

    gpu::init(width, height, mesh, bvh, materials, tri_material, lights, floor);

    const View view = frame_camera(lo, hi);

    // Second argument is an output directory, third orbits, fourth sets quality.
    if (argc > 2) {
        const float turns = argc > 3 ? strtof(argv[3], nullptr) : 0.0f;
        const int samples = argc > 4 ? atoi(argv[4]) : offline_samples;
        return render_offline(sequence, mesh, bvh, argv[2], view, animated, turns,
            std::max(1, samples));
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("render-pipeline", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_RESIZABLE);
    // No vsync: it would cap the loop at the refresh rate and hide the real speed.
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_RenderSetLogicalSize(renderer, width, height);

    // Starts from the same framing as the offline path, then the controls take over.
    const Camera start = camera_at(view, 0.0f);
    float3 position = start.position;
    float yaw = atan2f(start.forward.x, -start.forward.z);
    float pitch = asinf(std::max(-1.0f, std::min(1.0f, start.forward.y)));
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
                        // Stays short of a quarter turn, where the basis degenerates.
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
            // Few samples per frame here, so the cheaper refit wins.
            if (stepped) advance_to(sequence, anim_frame, mesh, bvh, false);
        }

        // Any change restarts the average, so pausing lets the image settle.
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
