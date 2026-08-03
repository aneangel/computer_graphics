#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <SDL2/SDL.h>
#include "geom.h"

namespace {
    struct Light {
        Light(const Vec3f &p, const float &i) : position(p), intensity(i) {}
        Vec3f position;
        float intensity;
    };
}

namespace {
    struct Material {
        explicit Material(const Vec3f &a, const Vec3f &color, const float &spec) : albedo(a), diffuse_color(color),
        specular_exponent(spec) {}
        Material() : albedo(1, 0, 0), diffuse_color(), specular_exponent() {}

        Vec3f albedo;
        Vec3f diffuse_color;
        float specular_exponent;
    };
}

namespace {
    struct Sphere {
        Vec3f center;
        float radius;
        Material material;

        Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}

        bool ray_intersect(const Vec3f &origin, const Vec3f &dir, float &t0) const {
            const Vec3f L = center - origin;
            const float tca = L * dir;
            const float d2 = L*L - tca*tca;

            if (d2 > radius*radius) return false;
            const float thc = sqrtf(radius*radius - d2);
            t0 = tca - thc;
            const float t1 = tca + thc;
            if (t0 < 0) t0 = t1;
            if (t0 < 0) return false;
            return true;
        }
    };
}

namespace {
    struct Camera {
        Camera(const Vec3f &pos, const Vec3f &target, const Vec3f &world_up, const float &f)
            : position(pos), fov(f) {
            forward = (target - pos).normalize();
            right = cross(forward, world_up).normalize();
            up = cross(right, forward);
        }

        Vec3f ray_direction(const float &x, const float &y) const {
            return (right * x + up * y + forward).normalize();
        }

        Vec3f position;
        Vec3f right;
        Vec3f up;
        Vec3f forward;
        float fov;
    };
}

static Vec3f reflect(const Vec3f &I, const Vec3f &normal) {
    return I - normal * (2.f * (I * normal));
}

static bool scene_intersect(const Vec3f &origin, const Vec3f &dir, const std::vector<Sphere> &spheres,
    Vec3f &hit, Vec3f &normal, Material &material) {
    float spheres_dist = std::numeric_limits<float>::max();
    for (const auto & sphere : spheres) {
        if (float dist_i; sphere.ray_intersect(origin, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = origin + dir * dist_i;
            normal = (hit - sphere.center).normalize();
            material = sphere.material;
        }
    }

    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.y) > 1e-3) {
        float d = -(origin.y + 4) / dir.y;
        Vec3f pt = origin + dir * d;
        if (d > 0 && fabs(pt.x) < 10 && pt.z < -10 && pt.z > -30 && d < spheres_dist) {
            checkerboard_dist = d;
            hit = pt;
            normal = {0, 1, 0};
            material.diffuse_color = (int(.5 * hit.x + 1000) + int(.5 * hit.z)) & 1 ? Vec3f(1, 1, 1) : Vec3f(1, .7, .3);
            material.diffuse_color = material.diffuse_color * .3;
        }
    }

    return std::min(spheres_dist, checkerboard_dist) < 1000;
}

static Vec3f cast_ray(const Vec3f &origin, const Vec3f &dir, const std::vector<Sphere> &spheres,
    const std::vector<Light> &lights, const size_t depth = 0) {
    Vec3f point, normal;
    Material material;

    if (depth > 4 || !scene_intersect(origin, dir, spheres, point, normal, material)) {
        return {0.5, 0.7, 0.8};
    }

    const Vec3f reflect_dir = reflect(dir, normal).normalize();
    const Vec3f reflect_origin = reflect_dir * normal < 0 ? point - normal * 1e-3 : point + normal * 1e-3;
    const Vec3f reflect_color = cast_ray(reflect_origin, reflect_dir, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i = 0; i < lights.size(); ++i) {
        Vec3f light_dir = (lights[i].position - point).normalize();
        const float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_origin = light_dir * normal < 0 ? point - normal * 1e-3 : point + normal * 1e-3;
        Vec3f shadow_normal;
        Material tmpmaterial;

        if (Vec3f shadow_pt; scene_intersect(shadow_origin, light_dir, spheres, shadow_pt, shadow_normal,
            tmpmaterial) && (shadow_pt - shadow_origin).norm() < light_distance) continue;

        diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir * normal);
        specular_light_intensity += powf(std::max(0.f, reflect(light_dir, normal) * dir),
            material.specular_exponent) * lights[i].intensity;
    }

    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.) *
        specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2];
}

static Vec3f forward_from_angles(const float &yaw, const float &pitch) {
    return {cosf(pitch) * sinf(yaw), sinf(pitch), -cosf(pitch) * cosf(yaw)};
}

static void trace_frame(const std::vector<Sphere> &spheres, const std::vector<Light> &lights,
    const Camera &camera, std::vector<Vec3f> &framebuffer, const int width, const int height) {
#pragma omp parallel for
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float x = (2 * (0.5 + i) / static_cast<float>(width) - 1) * tan(camera.fov / 2.) * width / static_cast<float>(height);
            float y = -(2*(j + 0.5)/static_cast<float>(height) - 1)*tan(camera.fov/2.);
            Vec3f dir = camera.ray_direction(x, y);
            framebuffer[i+j*width] = cast_ray(camera.position, dir, spheres, lights);
        }
    }
}

static void tone_map(std::vector<Vec3f> &framebuffer) {
    for (auto &c : framebuffer) {
        if (const float max = std::max(c[0], std::max(c[1], c[2])); max > 1) c = c * (1.f / max);
    }
}

static void blit(std::vector<Vec3f> &framebuffer, std::vector<uint32_t> &pixels) {
    tone_map(framebuffer);
    for (size_t i = 0; i < framebuffer.size(); ++i) {
        const Vec3f &c = framebuffer[i];
        const auto r = static_cast<uint32_t>(255 * std::max(0.f, std::min(1.f, c[0])));
        const auto g = static_cast<uint32_t>(255 * std::max(0.f, std::min(1.f, c[1])));
        const auto b = static_cast<uint32_t>(255 * std::max(0.f, std::min(1.f, c[2])));
        pixels[i] = 0xFF000000u | (r << 16) | (g << 8) | b;
    }
}

static void save_ppm(std::vector<Vec3f> &framebuffer, const int width, const int height,
    const std::string &filename) {
    tone_map(framebuffer);
    std::ofstream ofs;
    ofs.open(filename);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < static_cast<size_t>(height)*width; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main(int, char **) {
    const Material ivory(Vec3f(0.6, 0.3, 0.1), {0.4, 0.4, 0.3}, 50.);
    const Material red_rubber(Vec3f(0.9, 0.1, 0.0), {0.3, 0.1, 0.1}, 10.);
    const Material mirror({0.0, 10.0, 0.8}, {1.0, 1.0, 1.0}, 1425.);

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere({-3, 0, -16}, 2, ivory));
    spheres.push_back(Sphere({-1.0, -1.5, -12}, 2, mirror));
    spheres.push_back(Sphere({1.5, -0.5, -18}, 3, red_rubber));
    spheres.push_back(Sphere({7, 5, -18}, 4, mirror));

    std::vector<Light> lights;
    lights.push_back(Light({-20, 20, 20}, 1.5));
    lights.push_back(Light({30, 50, -25}, 1.8));
    lights.push_back(Light({30, 50, 30}, 1.7));

    constexpr int width = 640;
    constexpr int height = 480;
    constexpr int window_scale = 2;
    constexpr float fov = M_PI / 2;
    const Vec3f world_up = {0, 1, 0};

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width * window_scale, height * window_scale, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_RenderSetLogicalSize(renderer, width, height);

    std::vector<Vec3f> framebuffer(width * height);
    std::vector<uint32_t> pixels(width * height);

    Vec3f position = {0, 0, 0};
    float yaw = 0, pitch = 0;
    bool looking = false, running = true;
    uint64_t prev = SDL_GetPerformanceCounter();

    std::cout << "drag left mouse to look, WASD to move, Q/E down/up, shift to sprint, P to save, Esc to quit\n";

    while (running) {
        const uint64_t now = SDL_GetPerformanceCounter();
        const float dt = std::max(1e-4f, std::min(0.1f,
            static_cast<float>(now - prev) / static_cast<float>(SDL_GetPerformanceFrequency())));
        prev = now;

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
                        pitch = std::max(-1.55f, std::min(1.55f,
                            pitch - static_cast<float>(event.motion.yrel) * 0.004f));
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
                    if (event.key.keysym.scancode == SDL_SCANCODE_P) {
                        save_ppm(framebuffer, width, height, "./screenshot.ppm");
                        std::cout << "wrote ./screenshot.ppm\n";
                    }
                    break;
                default:
                    break;
            }
        }

        const Camera camera(position, position + forward_from_angles(yaw, pitch), world_up, fov);

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);
        const float speed = (keys[SDL_SCANCODE_LSHIFT] ? 24.f : 8.f) * dt;
        if (keys[SDL_SCANCODE_W]) position = position + camera.forward * speed;
        if (keys[SDL_SCANCODE_S]) position = position - camera.forward * speed;
        if (keys[SDL_SCANCODE_A]) position = position - camera.right * speed;
        if (keys[SDL_SCANCODE_D]) position = position + camera.right * speed;
        if (keys[SDL_SCANCODE_E]) position = position + world_up * speed;
        if (keys[SDL_SCANCODE_Q]) position = position - world_up * speed;

        trace_frame(spheres, lights, camera, framebuffer, width, height);
        blit(framebuffer, pixels);

        SDL_UpdateTexture(texture, nullptr, pixels.data(), width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        char title[96];
        snprintf(title, sizeof(title), "raytracer  %dx%d  %.1f fps  pos %.1f %.1f %.1f",
            width, height, 1.f / dt, position.x, position.y, position.z);
        SDL_SetWindowTitle(window, title);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}