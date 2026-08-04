"""Read and write mesh sequences (.msq), the format the renderer consumes.

The layout is documented in the README. Run directly to inspect a file:

    python tools/msq.py some.msq
"""

import struct
import sys

import numpy as np

MAGIC = b"MSQ1"
# MSQ2 adds an object table saying which run of triangles belongs to whom.
MAGIC_OBJECTS = b"MSQ2"
_HEADER = "<4sIIIf"
_HEADER_SIZE = struct.calcsize(_HEADER)
_OBJECT = "<II3f"
_OBJECT_SIZE = struct.calcsize(_OBJECT)


def write_msq(path, faces, verts, fps=30.0, objects=None):
    """Write faces (F,3), positions (T,V,3) and optional objects to `path`."""
    faces = np.ascontiguousarray(faces, dtype="<i4")
    verts = np.ascontiguousarray(verts, dtype="<f4")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"verts must be (T, V, 3), got {verts.shape}")
    if faces.max() >= verts.shape[1]:
        raise ValueError(
            f"face index {faces.max()} out of range for {verts.shape[1]} vertices"
        )

    if objects is not None:
        covered = sorted((int(a), int(n)) for a, n, _ in objects)
        at = 0
        for first, count in covered:
            # A gap or an overlap surfaces as a shading bug far from here.
            if first != at:
                raise ValueError(f"objects must tile the triangles; gap or overlap at {first}")
            at += count
        if at != faces.shape[0]:
            raise ValueError(f"objects cover {at} triangles, the mesh has {faces.shape[0]}")

    frame_count, vertex_count, _ = verts.shape
    with open(path, "wb") as f:
        magic = MAGIC if objects is None else MAGIC_OBJECTS
        f.write(struct.pack(_HEADER, magic, vertex_count, faces.shape[0], frame_count, fps))
        if objects is not None:
            f.write(struct.pack("<I", len(objects)))
            for first, count, colour in objects:
                f.write(struct.pack(_OBJECT, int(first), int(count), *[float(c) for c in colour]))
        f.write(faces.tobytes())
        f.write(verts.tobytes())


def read_msq(path):
    """Return (faces, verts, fps, objects), matching write_msq."""
    with open(path, "rb") as f:
        magic, vertex_count, triangle_count, frame_count, fps = struct.unpack(
            _HEADER, f.read(_HEADER_SIZE)
        )
        if magic not in (MAGIC, MAGIC_OBJECTS):
            raise ValueError(f"{path} is not a mesh sequence (magic {magic!r})")

        objects = None
        if magic == MAGIC_OBJECTS:
            (count,) = struct.unpack("<I", f.read(4))
            objects = []
            for _ in range(count):
                first, n, r, g, b = struct.unpack(_OBJECT, f.read(_OBJECT_SIZE))
                objects.append((first, n, (r, g, b)))

        faces = np.frombuffer(f.read(triangle_count * 3 * 4), dtype="<i4")
        verts = np.frombuffer(f.read(frame_count * vertex_count * 3 * 4), dtype="<f4")

    return (
        faces.reshape(triangle_count, 3),
        verts.reshape(frame_count, vertex_count, 3),
        fps,
        objects,
    )


def selftest(path):
    """Rewriting a file the C++ side wrote must reproduce it byte for byte."""
    faces, verts, fps, objects = read_msq(path)
    out = path + ".roundtrip"
    write_msq(out, faces, verts, fps, objects)
    same = open(path, "rb").read() == open(out, "rb").read()
    print(f"round-trip {'identical' if same else 'MISMATCH'}: {out}")
    return same


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    faces, verts, fps, objects = read_msq(sys.argv[1])
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"{sys.argv[1]}")
    print(f"  {verts.shape[1]} vertices, {faces.shape[0]} triangles")
    print(f"  {verts.shape[0]} frames at {fps:g} fps")
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")
    if objects:
        print(f"  {len(objects)} objects")
        for i, (first, count, colour) in enumerate(objects):
            rgb = " ".join(f"{c:.2f}" for c in colour)
            print(f"    {i:2d}  triangles {first:6d}..{first + count:<6d}  colour {rgb}")
