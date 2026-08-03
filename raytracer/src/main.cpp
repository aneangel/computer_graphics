#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <SDL2/SDL.h>
#include "render_gpu.h"
#include "scene.h"

namespace {
    float3 forward_from_angles(const float yaw, const float pitch) {
        return make_float3(cosf(pitch) * sinf(yaw), sinf(pitch), -cosf(pitch) * cosf(yaw));
    }

    void save_ppm(const uint32_t *pixels, const int width, const int height, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        std::vector<char> rgb(static_cast<size_t>(width) * height * 3);
        for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
            rgb[i * 3 + 0] = static_cast<char>(pixels[i] >> 16 & 0xFFu);
            rgb[i * 3 + 1] = static_cast<char>(pixels[i] >> 8 & 0xFFu);
            rgb[i * 3 + 2] = static_cast<char>(pixels[i] & 0xFFu);
        }
        ofs.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }
}

int main(int, char **) {
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr float fov = M_PI / 2;
    // Rays per pixel each frame. Raise to anti-alias while the camera is moving.
    constexpr int samples_per_frame = 1;
    const float3 world_up = make_float3(0, 1, 0);

    // colour, diffuse albedo, specular albedo, reflectivity, specular exponent
    const Material ivory = make_material(make_float3(0.4f, 0.4f, 0.3f), 0.6f, 0.3f, 0.1f, 50.0f);
    const Material red_rubber = make_material(make_float3(0.3f, 0.1f, 0.1f), 0.9f, 0.1f, 0.0f, 10.0f);
    const Material mirror = make_material(make_float3(1.0f, 1.0f, 1.0f), 0.0f, 10.0f, 0.8f, 1425.0f);
    const Material purple = make_material(make_float3(2.0, 1.0, 4.0), 0.6, 0.3, 0.1, 50.0f);

    std::vector<Sphere> spheres = {
        {make_float3(-3.0f, 0.0f, -16.0f), 2.0f, ivory},
        {make_float3(-1.0f, -1.5f, -12.0f), 2.0f, mirror},
        {make_float3(1.5f, -0.5f, -18.0f), 3.0f, red_rubber},
        {make_float3(7.0f, 5.0f, -18.0f), 4.0f, mirror},
        {make_float3(0.4, 0.5, -25.0f), 3.0f, purple},
    };

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

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, SDL_WINDOW_RESIZABLE);
    // No vsync flag: it would cap the loop at the monitor refresh and hide the real speed.
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_RenderSetLogicalSize(renderer, width, height);

    gpu::init(width, height, spheres, lights, floor);

    float3 position = make_float3(0, 0, 0);
    float yaw = 0, pitch = 0;
    bool looking = false, running = true;
    int frame = 0;
    const uint32_t *pixels = nullptr;
    uint64_t prev = SDL_GetPerformanceCounter();

    std::cout << "drag left mouse to look, WASD to move, Q/E down/up, shift to sprint, P to save, Esc to quit\n";

    while (running) {
        const uint64_t now = SDL_GetPerformanceCounter();
        const float dt = std::max(1e-4f, std::min(0.1f,
            static_cast<float>(now - prev) / static_cast<float>(SDL_GetPerformanceFrequency())));
        prev = now;

        const float3 prev_position = position;
        const float prev_yaw = yaw, prev_pitch = pitch;

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
                    if (event.key.keysym.scancode == SDL_SCANCODE_P && pixels) {
                        save_ppm(pixels, width, height, "./screenshot.ppm");
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

        // Any camera change invalidates the accumulated samples, so the average restarts.
        const bool moved = position.x != prev_position.x || position.y != prev_position.y ||
            position.z != prev_position.z || yaw != prev_yaw || pitch != prev_pitch;
        frame = moved ? 0 : frame + 1;

        const Camera camera = make_camera(position, position + forward_from_angles(yaw, pitch),
            world_up, fov, width, height);

        pixels = gpu::render(camera, frame, samples_per_frame);

        SDL_UpdateTexture(texture, nullptr, pixels, width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        char title[160];
        snprintf(title, sizeof(title), "raytracer  %dx%d  %.1f fps  kernel %.2f ms  %d spp  pos %.1f %.1f %.1f",
            width, height, 1.f / dt, gpu::last_kernel_ms(), (frame + 1) * samples_per_frame,
            position.x, position.y, position.z);
        SDL_SetWindowTitle(window, title);
    }

    gpu::shutdown();
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
