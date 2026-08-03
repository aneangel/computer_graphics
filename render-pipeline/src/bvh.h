#ifndef BVH_H
#define BVH_H

// Bounding volume hierarchy: a binary tree of boxes, so a ray that misses a box
// skips every triangle beneath it. Turns per-ray cost from linear in the
// triangle count into logarithmic. Built on the host, traversed on the device.

#include <algorithm>
#include <vector>
#include "geom_gpu.cuh"
#include "mesh.h"

// 32 bytes, two aligned loads. tri_count == 0 marks an interior node, and
// left_first is then the left child; the right child is always the next slot.
struct BVHNode {
    float3 bmin;
    int left_first;
    float3 bmax;
    int tri_count;
};

struct BVH {
    std::vector<BVHNode> nodes;
    // Triangles are reordered as the tree builds, so each leaf owns a
    // contiguous run of this array. Values index mesh.indices.
    std::vector<int> tri_index;
};

namespace bvh_detail {

// Candidate splits are drawn from bins, which makes the build linear in the
// triangle count for a tree that is nearly as good as testing every plane.
constexpr int BIN_COUNT = 12;
constexpr int LEAF_SIZE = 2;
constexpr float EMPTY = 1e30f;

struct AABB {
    float3 lo = make_float3(EMPTY, EMPTY, EMPTY);
    float3 hi = make_float3(-EMPTY, -EMPTY, -EMPTY);

    void grow(const float3 &p) {
        lo = fminf(lo, p);
        hi = fmaxf(hi, p);
    }

    void grow(const AABB &b) {
        if (b.hi.x < b.lo.x) return;  // an empty box must not move the bounds
        grow(b.lo);
        grow(b.hi);
    }

    // Half the true surface area; the constant cancels in every comparison.
    float area() const {
        const float3 d = hi - lo;
        if (d.x < 0.0f) return 0.0f;
        return d.x * d.y + d.y * d.z + d.z * d.x;
    }
};

inline float axis_of(const float3 &v, const int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

inline void set_bounds(BVH &bvh, const int node_idx, const int first, const int count,
    const std::vector<AABB> &tri_box) {
    AABB b;
    for (int i = 0; i < count; ++i) b.grow(tri_box[bvh.tri_index[first + i]]);
    bvh.nodes[node_idx].bmin = b.lo;
    bvh.nodes[node_idx].bmax = b.hi;
    bvh.nodes[node_idx].left_first = first;
    bvh.nodes[node_idx].tri_count = count;
}

inline void subdivide(BVH &bvh, const int node_idx, const std::vector<float3> &centroid,
    const std::vector<AABB> &tri_box) {
    const int first = bvh.nodes[node_idx].left_first;
    const int count = bvh.nodes[node_idx].tri_count;
    if (count <= LEAF_SIZE) return;

    // Split on centroids, not triangle boxes: overlapping triangles still have
    // distinct centres, so this keeps making progress where boxes would stall.
    AABB cbox;
    for (int i = 0; i < count; ++i) cbox.grow(centroid[bvh.tri_index[first + i]]);

    int best_axis = -1, best_bin = -1;
    float best_cost = EMPTY;

    for (int axis = 0; axis < 3; ++axis) {
        const float lo = axis_of(cbox.lo, axis);
        const float extent = axis_of(cbox.hi, axis) - lo;
        if (extent < 1e-12f) continue;  // every centroid at the same coordinate
        const float scale = static_cast<float>(BIN_COUNT) / extent;

        AABB bin_box[BIN_COUNT];
        int bin_count[BIN_COUNT] = {0};
        for (int i = 0; i < count; ++i) {
            const int t = bvh.tri_index[first + i];
            const int b = std::min(BIN_COUNT - 1,
                static_cast<int>((axis_of(centroid[t], axis) - lo) * scale));
            bin_count[b]++;
            bin_box[b].grow(tri_box[t]);
        }

        // One sweep from each end gives every plane both sides without
        // rescanning the triangles per candidate.
        float left_area[BIN_COUNT - 1], right_area[BIN_COUNT - 1];
        int left_n[BIN_COUNT - 1], right_n[BIN_COUNT - 1];

        AABB acc;
        int n = 0;
        for (int i = 0; i < BIN_COUNT - 1; ++i) {
            acc.grow(bin_box[i]);
            n += bin_count[i];
            left_area[i] = acc.area();
            left_n[i] = n;
        }
        acc = AABB();
        n = 0;
        for (int i = BIN_COUNT - 1; i > 0; --i) {
            acc.grow(bin_box[i]);
            n += bin_count[i];
            right_area[i - 1] = acc.area();
            right_n[i - 1] = n;
        }

        for (int i = 0; i < BIN_COUNT - 1; ++i) {
            // Surface area heuristic: a ray's chance of entering a box scales
            // with its area, so this is the expected triangle tests per ray.
            const float cost = left_area[i] * static_cast<float>(left_n[i]) +
                right_area[i] * static_cast<float>(right_n[i]);
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_bin = i;
            }
        }
    }

    if (best_axis < 0) return;

    // If no split beats testing the triangles directly, stay a leaf.
    AABB self;
    self.lo = bvh.nodes[node_idx].bmin;
    self.hi = bvh.nodes[node_idx].bmax;
    if (best_cost >= static_cast<float>(count) * self.area()) return;

    const float lo = axis_of(cbox.lo, best_axis);
    const float scale = static_cast<float>(BIN_COUNT) /
        (axis_of(cbox.hi, best_axis) - lo);
    const auto begin = bvh.tri_index.begin() + first;
    const auto mid = std::partition(begin, begin + count, [&](const int t) {
        const int b = std::min(BIN_COUNT - 1,
            static_cast<int>((axis_of(centroid[t], best_axis) - lo) * scale));
        return b <= best_bin;
    });

    const int left_count = static_cast<int>(mid - begin);
    if (left_count == 0 || left_count == count) return;  // nothing moved

    const int left_idx = static_cast<int>(bvh.nodes.size());
    bvh.nodes.push_back(BVHNode{});
    bvh.nodes.push_back(BVHNode{});
    set_bounds(bvh, left_idx, first, left_count, tri_box);
    set_bounds(bvh, left_idx + 1, first + left_count, count - left_count, tri_box);

    bvh.nodes[node_idx].left_first = left_idx;
    bvh.nodes[node_idx].tri_count = 0;  // only now does it stop being a leaf

    subdivide(bvh, left_idx, centroid, tri_box);
    subdivide(bvh, left_idx + 1, centroid, tri_box);
}

}  // namespace bvh_detail

inline BVH build_bvh(const Mesh &mesh) {
    using namespace bvh_detail;
    const int tri_count = static_cast<int>(mesh.triangle_count());

    BVH bvh;
    if (tri_count == 0) return bvh;

    bvh.tri_index.resize(tri_count);
    std::vector<float3> centroid(tri_count);
    std::vector<AABB> tri_box(tri_count);

    for (int i = 0; i < tri_count; ++i) {
        const int3 f = mesh.indices[i];
        const float3 &a = mesh.positions[f.x];
        const float3 &b = mesh.positions[f.y];
        const float3 &c = mesh.positions[f.z];
        AABB box;
        box.grow(a);
        box.grow(b);
        box.grow(c);
        tri_box[i] = box;
        centroid[i] = (a + b + c) * (1.0f / 3.0f);
        bvh.tri_index[i] = i;
    }

    // At most 2n-1 nodes, reserved so the vector never reallocates while the
    // recursion holds indices into it.
    bvh.nodes.reserve(static_cast<size_t>(2 * tri_count - 1));
    bvh.nodes.push_back(BVHNode{});
    set_bounds(bvh, 0, 0, tri_count, tri_box);
    subdivide(bvh, 0, centroid, tri_box);
    return bvh;
}

// Recomputes bounds against moved vertices, keeping the tree's structure. Runs
// backwards because children are pushed after their parent, so one reverse pass
// visits both children first.
inline void refit_bvh(BVH &bvh, const Mesh &mesh) {
    using namespace bvh_detail;
    for (int i = static_cast<int>(bvh.nodes.size()) - 1; i >= 0; --i) {
        BVHNode &n = bvh.nodes[i];
        AABB b;
        if (n.tri_count > 0) {
            for (int k = 0; k < n.tri_count; ++k) {
                const int3 f = mesh.indices[bvh.tri_index[n.left_first + k]];
                b.grow(mesh.positions[f.x]);
                b.grow(mesh.positions[f.y]);
                b.grow(mesh.positions[f.z]);
            }
        } else {
            const BVHNode &l = bvh.nodes[n.left_first];
            const BVHNode &r = bvh.nodes[n.left_first + 1];
            b.lo = fminf(l.bmin, r.bmin);
            b.hi = fmaxf(l.bmax, r.bmax);
        }
        n.bmin = b.lo;
        n.bmax = b.hi;
    }
}

#endif  // BVH_H
