"""Bake an animated, skinned glTF character into a mesh sequence.

glTF carries a mesh, skeleton, weights and animation in one file and needs no
registration, so the pipeline can be exercised without the research body models.
The deformation is linear blend skinning, the same operation SMPL applies.

A known-good file to test with (CC-BY, no registration):

    curl -sLO https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb
    python tools/gltf_to_msq.py CesiumMan.glb --out walk.msq
    mkdir frames && ./build/render-pipeline walk.msq frames/

Scope: .glb, LINEAR interpolation, non-sparse accessors, one skinned primitive.
Anything outside that raises rather than quietly producing a mangled body.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_to_msq import place_in_scene  # noqa: E402
from msq import write_msq  # noqa: E402

# glTF componentType -> numpy dtype, and type -> component count.
DTYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class Gltf:
    """Just enough of the format to read geometry, skins and animation."""

    def __init__(self, path):
        blob = open(path, "rb").read()
        magic, _version, total = struct.unpack("<III", blob[:12])
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a .glb (bad magic)")

        chunks, at = {}, 12
        while at < total:
            length, kind = struct.unpack("<II", blob[at : at + 8])
            chunks[kind] = blob[at + 8 : at + 8 + length]
            at += 8 + length

        self.json = json.loads(chunks[0x4E4F534A])  # 'JSON'
        self.bin = chunks.get(0x004E4942, b"")      # 'BIN'

    def accessor(self, index):
        """Read accessor `index` as (count, components), respecting byteStride."""
        acc = self.json["accessors"][index]
        if "sparse" in acc:
            raise NotImplementedError("sparse accessors are not supported")

        dtype = np.dtype(DTYPES[acc["componentType"]])
        ncomp = COUNTS[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:  # spec allows an all-zero accessor
            return np.zeros((count, ncomp), dtype)

        view = self.json["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        element = dtype.itemsize * ncomp
        stride = view.get("byteStride", element)

        if stride == element:
            flat = np.frombuffer(self.bin, dtype=dtype, count=count * ncomp, offset=base)
        else:
            # Interleaved: gather each element out of the stride.
            span = stride * (count - 1) + element
            raw = np.frombuffer(self.bin, dtype=np.uint8, count=span, offset=base)
            picks = np.arange(count)[:, None] * stride + np.arange(element)[None, :]
            flat = np.ascontiguousarray(raw[picks]).view(dtype)

        return flat.reshape(count, ncomp)


def quat_to_matrix(q):
    """Rotation matrix from a glTF quaternion, which is stored (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(a, b, t):
    """Shortest-arc quaternion interpolation.

    Interpolating the four numbers directly cuts the corner and changes speed
    through the arc, which shows on a limb swinging through a large angle.
    """
    dot = float(np.dot(a, b))
    if dot < 0.0:  # the far way round is the same rotation; take the near one
        b, dot = -b, -dot
    if dot > 0.9995:  # nearly parallel, slerp is numerically unhappy
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b) / s


def local_matrix(node, override):
    """A node's own transform, with any animated channels substituted in."""
    if "matrix" in node and not override:
        # glTF stores matrices column-major; numpy reads row-major.
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T

    t = override.get("translation", node.get("translation", [0, 0, 0]))
    r = override.get("rotation", node.get("rotation", [0, 0, 0, 1]))
    s = override.get("scale", node.get("scale", [1, 1, 1]))

    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(np.asarray(r, np.float64)) * np.asarray(s, np.float64)
    m[:3, 3] = t
    return m


def sample_animation(g, animation, time):
    """Evaluate every channel at `time`. Returns {node index: {path: value}}."""
    out = {}
    for channel in animation["channels"]:
        target = channel["target"]
        if "node" not in target:
            continue
        sampler = animation["samplers"][channel["sampler"]]
        if sampler.get("interpolation", "LINEAR") != "LINEAR":
            raise NotImplementedError(f"{sampler['interpolation']} interpolation")

        times = g.accessor(sampler["input"])[:, 0]
        values = g.accessor(sampler["output"])

        # Locate the segment containing `time`, clamping outside the curve.
        i = int(np.searchsorted(times, time, side="right")) - 1
        i = max(0, min(i, len(times) - 2)) if len(times) > 1 else 0
        if len(times) == 1:
            value = values[0]
        else:
            span = times[i + 1] - times[i]
            u = 0.0 if span <= 0 else float(np.clip((time - times[i]) / span, 0.0, 1.0))
            path = target["path"]
            if path == "rotation":
                value = slerp(values[i].astype(np.float64), values[i + 1].astype(np.float64), u)
            else:
                value = values[i] + u * (values[i + 1] - values[i])

        out.setdefault(target["node"], {})[target["path"]] = np.asarray(value, np.float64)
    return out


def global_transforms(g, overrides):
    """World matrix of every node, composed down the scene hierarchy."""
    nodes = g.json["nodes"]
    parent = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i

    cache = {}

    def resolve(i):
        if i not in cache:
            local = local_matrix(nodes[i], overrides.get(i, {}))
            cache[i] = local if i not in parent else resolve(parent[i]) @ local
        return cache[i]

    return [resolve(i) for i in range(len(nodes))]


def skinned_mesh_node(g):
    """The one node carrying both a mesh and a skin."""
    for i, node in enumerate(g.json["nodes"]):
        if "mesh" in node and "skin" in node:
            return i
    raise ValueError("no skinned mesh in this file")


def bake(path, fps=30.0, max_frames=0):
    """Evaluate the animation. Returns (vertices, faces, duration)."""
    g = Gltf(path)
    node_index = skinned_mesh_node(g)
    node = g.json["nodes"][node_index]
    skin = g.json["skins"][node["skin"]]

    primitives = g.json["meshes"][node["mesh"]]["primitives"]
    if len(primitives) != 1:
        raise NotImplementedError(f"{len(primitives)} primitives; expected 1")
    prim = primitives[0]
    if prim.get("mode", 4) != 4:
        raise NotImplementedError("only triangle primitives are supported")

    rest = g.accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    joints = g.accessor(prim["attributes"]["JOINTS_0"]).astype(np.int64)
    weights = g.accessor(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    faces = g.accessor(prim["indices"])[:, 0].reshape(-1, 3).astype(np.int32)

    # Unnormalised weights inflate or shrink the body without looking wrong.
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    # Column-major again, one 4x4 per joint.
    ibm = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)

    animation = g.json["animations"][0]
    duration = max(
        float(g.accessor(s["input"])[:, 0].max()) for s in animation["samplers"]
    )
    frames = max(1, int(round(duration * fps)))
    if max_frames:
        frames = min(frames, max_frames)

    homogeneous = np.concatenate([rest, np.ones((len(rest), 1))], axis=1)
    out = np.empty((frames, len(rest), 3), np.float32)

    for f in range(frames):
        world = global_transforms(g, sample_animation(g, animation, f / fps))
        # Undo the mesh placement, apply the joint, undo the bind pose.
        inverse_mesh = np.linalg.inv(world[node_index])
        matrices = np.stack([
            inverse_mesh @ world[j] @ ibm[k] for k, j in enumerate(skin["joints"])
        ])

        # Each influence transforms the vertex alone, then they mix by weight.
        posed = np.zeros((len(rest), 3))
        for slot in range(joints.shape[1]):
            m = matrices[joints[:, slot]]                        # (V, 4, 4)
            moved = np.einsum("vab,vb->va", m, homogeneous)[:, :3]
            posed += weights[:, slot : slot + 1] * moved
        out[f] = posed

    return out, faces, duration


def verify_bind_pose(path):
    """At the bind pose every skin matrix must be identity and the mesh must
    reproduce itself. That one check covers accessor strides, the column-major
    convention, quaternion order and the joint indirection at once, and guards
    against a plausible looking body in the wrong shape.
    """
    g = Gltf(path)
    index = skinned_mesh_node(g)
    node = g.json["nodes"][index]
    skin = g.json["skins"][node["skin"]]
    prim = g.json["meshes"][node["mesh"]]["primitives"][0]

    rest = g.accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    joints = g.accessor(prim["attributes"]["JOINTS_0"]).astype(np.int64)
    weights = g.accessor(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    ibm = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)

    world = global_transforms(g, {})
    inverse_mesh = np.linalg.inv(world[index])
    matrices = np.stack([
        inverse_mesh @ world[j] @ ibm[k] for k, j in enumerate(skin["joints"])
    ])

    homogeneous = np.concatenate([rest, np.ones((len(rest), 1))], axis=1)
    posed = np.zeros((len(rest), 3))
    for slot in range(joints.shape[1]):
        posed += weights[:, slot : slot + 1] * np.einsum(
            "vab,vb->va", matrices[joints[:, slot]], homogeneous
        )[:, :3]

    identity_error = float(np.abs(matrices - np.eye(4)).max())
    rest_error = float(np.abs(posed - rest).max())
    scale = float(np.abs(rest).max())
    ok = identity_error < 1e-5 and rest_error < 1e-5 * max(scale, 1.0)

    print(f"  bind pose: skin matrices differ from identity by {identity_error:.2e}")
    print(f"  bind pose: reskinned mesh differs from rest by {rest_error:.2e}")
    print(f"  {'pass' if ok else 'FAIL'}")
    return ok


def detect_up(frame):
    """Guess the up axis from one frame's proportions.

    glTF declares Y-up and plenty of files ignore it; CesiumMan is Z-up. One
    frame, not the whole sequence, because travel becomes the longest axis over
    a clip. Weak for a wide T-pose, so it is reported and can be overridden.
    """
    extent = frame.max(0) - frame.min(0)
    return int(np.argmax(extent)), extent


def to_y_up(verts, axis):
    """Permute so `axis` becomes +Y, which is what the renderer expects."""
    if axis == 1:
        return verts
    if axis == 2:  # Z-up
        return np.stack([verts[..., 0], verts[..., 2], -verts[..., 1]], axis=-1)
    return np.stack([verts[..., 1], -verts[..., 0], verts[..., 2]], axis=-1)  # X-up


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gltf", help="a .glb file with a skinned, animated mesh")
    ap.add_argument("--out", default="gltf.msq")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--loops", type=int, default=1, help="repeat the clip this many times")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--verify", action="store_true", help="run the bind-pose check and exit")
    args = ap.parse_args()

    if args.verify:
        print(args.gltf)
        sys.exit(0 if verify_bind_pose(args.gltf) else 1)

    verts, faces, duration = bake(args.gltf, args.fps, args.max_frames)
    print(f"{args.gltf}: {verts.shape[1]} vertices, {len(faces)} triangles")
    print(f"  {duration:.2f} s of animation -> {verts.shape[0]} frames at {args.fps:g} fps")

    guess, extent = detect_up(verts[0])
    axis = guess if args.up == "auto" else "xyz".index(args.up)
    print(f"  frame extents  x {extent[0]:.2f}  y {extent[1]:.2f}  z {extent[2]:.2f}"
          f"  -> up is {'xyz'[axis]}" + ("" if args.up == "auto" else " (forced)"))

    if args.loops > 1:
        verts = np.concatenate([verts] * args.loops, axis=0)

    placed = place_in_scene(to_y_up(verts, axis), up="y")
    lo, hi = placed.reshape(-1, 3).min(0), placed.reshape(-1, 3).max(0)
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")

    write_msq(args.out, faces, placed, args.fps)
    print(f"wrote {args.out} ({verts.shape[0]} frames)")


if __name__ == "__main__":
    main()
