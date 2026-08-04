#ifndef MESH_H
#define MESH_H

// Host side triangle mesh, generated or read from an OBJ.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "geom_gpu.cuh"

// Indexed triangles, so a vertex normal is shared by every face meeting there.
struct Mesh {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<int3> indices;

    size_t vertex_count() const { return positions.size(); }
    size_t triangle_count() const { return indices.size(); }
};

// Area weighted vertex normals. The raw cross product already carries the area.
inline void recompute_normals(Mesh &m) {
    m.normals.assign(m.vertex_count(), make_float3(0.0f, 0.0f, 0.0f));
    for (const int3 &f : m.indices) {
        const float3 &a = m.positions[f.x];
        const float3 &b = m.positions[f.y];
        const float3 &c = m.positions[f.z];
        const float3 face = cross(b - a, c - a);
        m.normals[f.x] += face;
        m.normals[f.y] += face;
        m.normals[f.z] += face;
    }
    for (float3 &n : m.normals)
        n = length_squared(n) > 1e-20f ? normalize(n) : make_float3(0.0f, 1.0f, 0.0f);
}

// Uniform scale then translate. Normals are unaffected by both.
inline void transform(Mesh &m, const float scale, const float3 &translate) {
    for (float3 &p : m.positions) p = p * scale + translate;
}

// Sphere on a latitude and longitude grid, slices around and stacks pole to pole.
inline Mesh make_uv_sphere(const float3 &center, const float radius, const int slices,
    const int stacks) {
    Mesh m;
    // The seam column is duplicated so its two sides can differ later.
    for (int i = 0; i <= stacks; ++i) {
        const float theta = 3.14159265359f * static_cast<float>(i) / static_cast<float>(stacks);
        const float st = sinf(theta), ct = cosf(theta);
        for (int j = 0; j <= slices; ++j) {
            const float phi = 6.28318530718f * static_cast<float>(j) / static_cast<float>(slices);
            // On a sphere the outward normal is the unit offset from the center.
            const float3 n = make_float3(st * cosf(phi), ct, st * sinf(phi));
            m.normals.push_back(n);
            m.positions.push_back(center + n * radius);
        }
    }

    const int row = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const int a = i * row + j;  // this ring
            const int b = a + row;      // next ring down
            // The poles collapse to one row of triangles instead of two.
            if (i != 0) m.indices.push_back(make_int3(a, b, a + 1));
            if (i != stacks - 1) m.indices.push_back(make_int3(a + 1, b, b + 1));
        }
    }
    return m;
}

// Minimal OBJ reader: v, vn and f. Everything else is skipped.
inline Mesh load_obj(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }

    Mesh m;
    std::vector<float3> file_normals;
    // OBJ indexes positions and normals separately, so they are paired here.
    std::vector<int> normal_of_vertex;
    std::string line;

    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;

        if (tag == "v") {
            float x, y, z;
            ls >> x >> y >> z;
            m.positions.push_back(make_float3(x, y, z));
            normal_of_vertex.push_back(-1);
        } else if (tag == "vn") {
            float x, y, z;
            ls >> x >> y >> z;
            file_normals.push_back(make_float3(x, y, z));
        } else if (tag == "f") {
            std::vector<int> corner;
            std::string token;
            while (ls >> token) {
                // Turning slashes into spaces lets one parse handle every face form.
                for (char &ch : token)
                    if (ch == '/') ch = ' ';
                std::istringstream ts(token);
                int vi = 0, ti = 0, ni = 0;
                ts >> vi;
                ts >> ti;
                ts >> ni;
                // OBJ counts from 1, and negative means relative to the end.
                const int v = vi > 0 ? vi - 1 : static_cast<int>(m.positions.size()) + vi;
                if (ni != 0 && v >= 0 && v < static_cast<int>(normal_of_vertex.size()))
                    normal_of_vertex[v] =
                        ni > 0 ? ni - 1 : static_cast<int>(file_normals.size()) + ni;
                corner.push_back(v);
            }
            // Polygons are fanned from the first corner.
            for (size_t k = 2; k < corner.size(); ++k)
                m.indices.push_back(make_int3(corner[0], corner[k - 1], corner[k]));
        }
    }

    // A partial set of normals would shade inconsistently, so it is all or none.
    bool complete = !file_normals.empty();
    for (const int n : normal_of_vertex)
        if (n < 0) complete = false;

    if (complete) {
        m.normals.resize(m.vertex_count());
        for (size_t i = 0; i < m.vertex_count(); ++i) m.normals[i] = file_normals[normal_of_vertex[i]];
    } else {
        recompute_normals(m);
    }
    return m;
}

#endif  // MESH_H
