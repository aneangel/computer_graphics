"""Read and write mesh sequences (.msq), the format the renderer consumes.

The layout is documented in the README. Run directly to inspect a file:

    python tools/msq.py some.msq
"""

import struct
import sys

import numpy as np

MAGIC = b"MSQ1"
_HEADER = "<4sIIIf"
_HEADER_SIZE = struct.calcsize(_HEADER)


def write_msq(path, faces, verts, fps=30.0):
    """Write faces (F,3) and per-frame positions (T,V,3) to `path`."""
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

    frame_count, vertex_count, _ = verts.shape
    with open(path, "wb") as f:
        f.write(struct.pack(_HEADER, MAGIC, vertex_count, faces.shape[0], frame_count, fps))
        f.write(faces.tobytes())
        f.write(verts.tobytes())


def read_msq(path):
    """Return (faces, verts, fps), matching write_msq."""
    with open(path, "rb") as f:
        magic, vertex_count, triangle_count, frame_count, fps = struct.unpack(
            _HEADER, f.read(_HEADER_SIZE)
        )
        if magic != MAGIC:
            raise ValueError(f"{path} is not a mesh sequence (magic {magic!r})")
        faces = np.frombuffer(f.read(triangle_count * 3 * 4), dtype="<i4")
        verts = np.frombuffer(f.read(frame_count * vertex_count * 3 * 4), dtype="<f4")

    return (
        faces.reshape(triangle_count, 3),
        verts.reshape(frame_count, vertex_count, 3),
        fps,
    )


def selftest(path):
    """Rewriting a file the C++ side wrote must reproduce it byte for byte."""
    faces, verts, fps = read_msq(path)
    out = path + ".roundtrip"
    write_msq(out, faces, verts, fps)
    same = open(path, "rb").read() == open(out, "rb").read()
    print(f"round-trip {'identical' if same else 'MISMATCH'}: {out}")
    return same


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    faces, verts, fps = read_msq(sys.argv[1])
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"{sys.argv[1]}")
    print(f"  {verts.shape[1]} vertices, {faces.shape[0]} triangles")
    print(f"  {verts.shape[0]} frames at {fps:g} fps")
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")
