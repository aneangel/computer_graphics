#ifndef ANIM_H
#define ANIM_H

// Mesh sequence: one fixed topology, many frames of vertex positions.
// Normals are not stored, they are recomputed from the faces on load.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "mesh.h"

struct SequenceHeader {
    char magic[4];
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t frame_count;
    float fps;
};

// Which run of triangles belongs to which actor, and what colour it is.
struct SequenceObject {
    uint32_t first_triangle;
    uint32_t triangle_count;
    float3 color;
};

// Both are read straight off disk, so their layout is the file format.
static_assert(sizeof(SequenceHeader) == 20, "sequence header layout changed");
static_assert(sizeof(SequenceObject) == 20, "sequence object layout changed");

inline bool has_sequence_extension(const std::string &path) {
    return path.size() > 4 && path.compare(path.size() - 4, 4, ".msq") == 0;
}

inline void write_sequence(const std::string &path, const Mesh &topology,
    const std::vector<std::vector<float3>> &frames, const float fps) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }

    SequenceHeader h{};
    h.magic[0] = 'M';
    h.magic[1] = 'S';
    h.magic[2] = 'Q';
    h.magic[3] = '1';
    h.vertex_count = static_cast<uint32_t>(topology.vertex_count());
    h.triangle_count = static_cast<uint32_t>(topology.triangle_count());
    h.frame_count = static_cast<uint32_t>(frames.size());
    h.fps = fps;

    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(topology.indices.data()),
        static_cast<std::streamsize>(topology.indices.size() * sizeof(int3)));
    for (const std::vector<float3> &f : frames)
        out.write(reinterpret_cast<const char *>(f.data()),
            static_cast<std::streamsize>(f.size() * sizeof(float3)));
}

// Seeks per frame, so a long sequence never has to fit in memory at once.
class MeshSequence {
public:
    void open(const std::string &path, Mesh &mesh) {
        in_.open(path, std::ios::binary);
        if (!in_) {
            std::fprintf(stderr, "cannot open %s\n", path.c_str());
            std::exit(1);
        }
        in_.read(reinterpret_cast<char *>(&h_), sizeof(h_));
        const std::string magic(h_.magic, 4);
        if (magic != "MSQ1" && magic != "MSQ2") {
            std::fprintf(stderr, "%s is not a mesh sequence\n", path.c_str());
            std::exit(1);
        }

        if (magic == "MSQ2") {
            uint32_t count = 0;
            in_.read(reinterpret_cast<char *>(&count), sizeof(count));
            objects_.resize(count);
            in_.read(reinterpret_cast<char *>(objects_.data()),
                static_cast<std::streamsize>(count * sizeof(SequenceObject)));
        } else {
            // MSQ1 has no table, so the whole mesh is one object.
            objects_.push_back(SequenceObject{0, h_.triangle_count,
                make_float3(0.4f, 0.4f, 0.3f)});
        }

        mesh.indices.resize(h_.triangle_count);
        in_.read(reinterpret_cast<char *>(mesh.indices.data()),
            static_cast<std::streamsize>(h_.triangle_count * sizeof(int3)));
        frames_begin_ = in_.tellg();

        mesh.positions.resize(h_.vertex_count);
        mesh.normals.resize(h_.vertex_count);
        read_frame(0, mesh);
    }

    // Moves the vertices and rebuilds the normals. Connectivity never changes.
    void read_frame(const int index, Mesh &mesh) {
        const std::streamoff stride =
            static_cast<std::streamoff>(h_.vertex_count) * static_cast<std::streamoff>(sizeof(float3));
        in_.seekg(frames_begin_ + stride * index);
        in_.read(reinterpret_cast<char *>(mesh.positions.data()),
            static_cast<std::streamsize>(h_.vertex_count * sizeof(float3)));
        recompute_normals(mesh);
    }

    int frame_count() const { return static_cast<int>(h_.frame_count); }
    float fps() const { return h_.fps; }
    const std::vector<SequenceObject> &objects() const { return objects_; }

    // Bounds over every frame, since actors move away from where they start.
    void bounds(float3 &lo, float3 &hi) {
        std::vector<float3> frame(h_.vertex_count);
        lo = make_float3(1e30f, 1e30f, 1e30f);
        hi = make_float3(-1e30f, -1e30f, -1e30f);
        const std::streamoff stride = static_cast<std::streamoff>(h_.vertex_count) *
            static_cast<std::streamoff>(sizeof(float3));
        for (uint32_t f = 0; f < h_.frame_count; ++f) {
            in_.seekg(frames_begin_ + stride * static_cast<std::streamoff>(f));
            in_.read(reinterpret_cast<char *>(frame.data()),
                static_cast<std::streamsize>(h_.vertex_count * sizeof(float3)));
            for (const float3 &p : frame) {
                lo = fminf(lo, p);
                hi = fmaxf(hi, p);
            }
        }
    }

    // One material index per triangle, in the mesh's own triangle order.
    std::vector<int> triangle_materials() const {
        std::vector<int> out(h_.triangle_count, 0);
        for (size_t i = 0; i < objects_.size(); ++i) {
            const SequenceObject &o = objects_[i];
            const uint32_t end = std::min(o.first_triangle + o.triangle_count, h_.triangle_count);
            for (uint32_t t = o.first_triangle; t < end; ++t) out[t] = static_cast<int>(i);
        }
        return out;
    }

private:
    std::ifstream in_;
    SequenceHeader h_{};
    std::vector<SequenceObject> objects_;
    std::streampos frames_begin_{};
};

#endif  // ANIM_H
