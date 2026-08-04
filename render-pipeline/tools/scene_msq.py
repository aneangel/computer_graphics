"""Merge several mesh sequences, and a ball, into one scene the renderer opens.

Everything lands in one vertex array behind one triangle list, with an object
table saying which run of triangles is whose.

    python tools/scene_msq.py a.msq b.msq --out scene.msq \
        --colour 0.10,0.25,0.80 --colour 0.90,0.90,0.90

    python tools/scene_msq.py players.msq --ball ball.json --out scene.msq

Sequences written one at a time were each centred on the same spot, so merging
them unaltered stands everybody inside everybody else. Use --offset for those.

The ball file is JSON: {"radius": 0.4, "centres": [[x, y, z], ...]}, one centre
per frame, in the same units as the sequences.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msq import read_msq, write_msq  # noqa: E402

BODY_COLOURS = [
    (0.10, 0.25, 0.80),  # blue kit
    (0.85, 0.85, 0.88),  # white kit
    (0.80, 0.15, 0.15),  # red kit
    (0.15, 0.55, 0.20),  # green kit
    (0.90, 0.75, 0.10),  # keeper
    (0.45, 0.20, 0.60),
]
BALL_COLOUR = (0.95, 0.95, 0.95)


def uv_sphere(radius, slices=32, stacks=16):
    """Sphere on a latitude and longitude grid, matching the renderer's own."""
    theta = (np.pi * np.arange(stacks + 1) / stacks)[:, None]
    phi = (2.0 * np.pi * np.arange(slices + 1) / slices)[None, :]
    st, ct = np.sin(theta), np.cos(theta)
    normals = np.stack([
        st * np.cos(phi),
        np.broadcast_to(ct, (stacks + 1, slices + 1)),
        st * np.sin(phi),
    ], axis=-1).reshape(-1, 3)

    row = slices + 1
    faces = []
    for i in range(stacks):
        for j in range(slices):
            a, b = i * row + j, i * row + j + row
            if i != 0:
                faces.append((a, b, a + 1))
            if i != stacks - 1:
                faces.append((a + 1, b, b + 1))

    return (normals * radius).astype(np.float32), np.array(faces, np.int32)


def load_part(path, offset):
    """One sequence as (verts, faces, objects), shifted by `offset`."""
    faces, verts, fps, objects = read_msq(path)
    verts = verts + np.asarray(offset, np.float32)
    return verts.astype(np.float32), faces, objects, fps


def ball_part(path, frames):
    """The ball as a sphere flown along a stored trajectory."""
    with open(path) as f:
        spec = json.load(f)
    verts, faces = uv_sphere(float(spec.get("radius", 0.4)))

    centres = np.asarray(spec["centres"], np.float32)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise SystemExit(f"{path}: centres must be a list of [x, y, z], got {centres.shape}")
    if len(centres) < frames:
        # Holds the last position, rather than looping back across the pitch.
        centres = np.concatenate([centres, np.repeat(centres[-1:], frames - len(centres), 0)])

    return (verts[None, :, :] + centres[:frames, None, :]).astype(np.float32), faces


def merge(parts):
    """Concatenate parts into one topology, truncated to the shortest."""
    frames = min(v.shape[0] for v, _, _ in parts)

    out_faces, out_verts, objects = [], [], []
    vertex_at, triangle_at = 0, 0
    for verts, faces, colours in parts:
        out_verts.append(verts[:frames])
        out_faces.append(faces + vertex_at)
        for first, count, colour in colours:
            objects.append((triangle_at + first, count, colour))
        vertex_at += verts.shape[1]
        triangle_at += len(faces)

    return (np.concatenate(out_faces), np.concatenate(out_verts, axis=1), objects, frames)


def parse_triple(text, what):
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise SystemExit(f"{what} needs three numbers, got {text!r}")
    return tuple(float(p) for p in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sequences", nargs="*", help="the .msq files to merge")
    ap.add_argument("--out", default="scene.msq")
    ap.add_argument("--colour", action="append", default=[], metavar="R,G,B",
                    help="one per sequence, in order; the rest come from a default kit list")
    ap.add_argument("--offset", action="append", default=[], metavar="X,Y,Z",
                    help="one per sequence, for actors that were recovered separately "
                         "and so share no frame to be placed in")
    ap.add_argument("--ball", help="a trajectory JSON to fly a sphere along")
    ap.add_argument("--ball-colour", default=None, metavar="R,G,B")
    ap.add_argument("--fps", type=float, default=0.0, help="0 keeps the first sequence's rate")
    args = ap.parse_args()

    if not args.sequences and not args.ball:
        ap.error("give at least one sequence, or a --ball")
    if args.colour and len(args.colour) > len(args.sequences):
        ap.error(f"{len(args.colour)} colours for {len(args.sequences)} sequences")
    if args.offset and len(args.offset) > len(args.sequences):
        ap.error(f"{len(args.offset)} offsets for {len(args.sequences)} sequences")

    parts, fps = [], args.fps
    for i, path in enumerate(args.sequences):
        offset = parse_triple(args.offset[i], "--offset") if i < len(args.offset) else (0, 0, 0)
        verts, faces, objects, part_fps = load_part(path, offset)
        if not fps:
            fps = part_fps

        if i < len(args.colour):
            colour = parse_triple(args.colour[i], "--colour")
        else:
            colour = BODY_COLOURS[i % len(BODY_COLOURS)]

        # A sequence that already names its actors keeps them.
        if objects and len(objects) > 1:
            parts.append((verts, faces, objects))
        else:
            parts.append((verts, faces, [(0, len(faces), colour)]))
        print(f"  {os.path.basename(path):<28} {verts.shape[0]:4d} frames  "
              f"{verts.shape[1]:6d} verts  {len(faces):6d} tris")

    if args.ball:
        # Sized against the actors, so the ball never decides the scene length.
        frames = min(v.shape[0] for v, _, _ in parts) if parts else None
        verts, faces = ball_part(args.ball, frames or 10 ** 9)
        colour = parse_triple(args.ball_colour, "--ball-colour") if args.ball_colour \
            else BALL_COLOUR
        parts.append((verts, faces, [(0, len(faces), colour)]))
        print(f"  {'ball':<28} {verts.shape[0]:4d} frames  "
              f"{verts.shape[1]:6d} verts  {len(faces):6d} tris")

    faces, verts, objects, frames = merge(parts)
    write_msq(args.out, faces, verts, fps or 30.0, objects)

    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"\nwrote {args.out}")
    print(f"  {len(objects)} objects, {verts.shape[1]} vertices, {len(faces)} triangles")
    print(f"  {frames} frames at {fps or 30.0:g} fps")
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")


if __name__ == "__main__":
    main()
