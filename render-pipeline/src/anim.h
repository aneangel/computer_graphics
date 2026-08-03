#ifndef ANIM_H
#define ANIM_H

// Mesh sequence: one fixed topology, many frames of vertex positions. The file
// layout is documented in the README. Normals are not stored, they are
// recomputed from the faces on load.

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
        if (std::string(h_.magic, 4) != "MSQ1") {
            std::fprintf(stderr, "%s is not a mesh sequence\n", path.c_str());
            std::exit(1);
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

private:
    std::ifstream in_;
    SequenceHeader h_{};
    std::streampos frames_begin_{};
};

#endif  // ANIM_H
