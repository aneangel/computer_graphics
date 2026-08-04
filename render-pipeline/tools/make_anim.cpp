// Writes a test mesh sequence, meeting the renderer only at the .msq format.
//
//   make-anim out.msq [model.obj] [frames]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "anim.h"
#include "mesh.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: make-anim out.msq [model.obj] [frames]\n");
        return 1;
    }
    const std::string out = argv[1];
    const int frame_count = argc > 3 ? std::atoi(argv[3]) : 120;
    constexpr float fps = 30.0f;

    const float3 center = make_float3(0.0f, 0.0f, -16.0f);
    constexpr float radius = 3.0f;

    Mesh base;
    if (argc > 2) {
        base = load_obj(argv[2]);
        float3 lo = base.positions[0], hi = base.positions[0];
        for (const float3 &p : base.positions) {
            lo = fminf(lo, p);
            hi = fmaxf(hi, p);
        }
        const float3 mid = (lo + hi) * 0.5f;
        const float extent = fmaxf(1e-6f, max_component(hi - lo) * 0.5f);
        const float s = radius / extent;
        transform(base, s, center - mid * s);
    } else {
        base = make_uv_sphere(center, radius, 96, 48);
    }

    // Not rigid on purpose, so the normals stop describing it and are rebuilt.
    std::vector<std::vector<float3>> frames;
    frames.reserve(frame_count);
    for (int f = 0; f < frame_count; ++f) {
        const float t = static_cast<float>(f) / static_cast<float>(frame_count);
        std::vector<float3> positions(base.vertex_count());
        for (size_t i = 0; i < base.vertex_count(); ++i) {
            const float3 &p = base.positions[i];
            const float phase = (p.y - center.y) * 2.2f - t * 6.28318530718f;
            positions[i] = p + base.normals[i] * (0.28f * radius * sinf(phase));
        }
        frames.push_back(std::move(positions));
    }

    write_sequence(out, base, frames, fps);
    std::printf("%s: %zu verts, %zu tris, %d frames at %.0f fps (%.1f MB)\n", out.c_str(),
        base.vertex_count(), base.triangle_count(), frame_count, fps,
        static_cast<double>(frame_count) * base.vertex_count() * sizeof(float3) / (1024.0 * 1024.0));
    return 0;
}
