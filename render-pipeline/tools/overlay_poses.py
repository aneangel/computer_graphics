"""Draw the estimated bodies back onto the footage they came from.

video_to_scene.py takes the world-space parameters and builds a scene you can
walk around. This takes the camera-space ones and puts them back where they
started, which is how you see whether the estimate is actually right.

    python tools/overlay_poses.py .cache/poses/gem/actors.json --out overlay.mp4

The whole of it is one equation. GEM writes pose as joint rotations, so the
vertices have to be built before anything can be drawn:

    X = M(beta, theta)          SMPL-X, rotations down the kinematic tree
    x = K X / X_z               perspective divide, into pixels

X must come from `body_params_incam`. The global parameters have the camera
motion taken out of them, so they are right for a scene and wrong for this: they
would sit still while the shot panned away from them.

Triangles are painted back to front rather than z-buffered. A body does not
intersect itself, so the order is the whole of the hidden surface problem here.
"""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import warp as wp

# Nothing drawn yet. Depth is the high half, so this loses every atomic minimum.
EMPTY_KEY = (1 << 62) - 1
_EMPTY = wp.constant(wp.int64(EMPTY_KEY))

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))
SMPLX_DIR = os.path.join(GEM_ROOT, "inputs", "checkpoints", "body_models")

# Distinct enough to tell apart once they overlap, and to read against grass.
COLORS = [
    (255, 92, 92), (92, 160, 255), (108, 224, 128), (255, 200, 64),
    (216, 122, 255), (96, 232, 232), (255, 148, 96), (200, 200, 200),
]

# Runs inside GEM's venv, the only place smplx and the model files are.
_VERTS_SCRIPT = r"""
import sys, json, numpy as np, torch, smplx

spec = json.loads(sys.stdin.read())
model_path, out_path = sys.argv[1:3]

def take(body, key, width, frames):
    if key not in body:
        return torch.zeros(frames, width)
    v = body[key].reshape(frames, -1).float()
    return v[:, :width] if v.shape[1] >= width else torch.nn.functional.pad(
        v, (0, width - v.shape[1]))

out = {}
for actor in spec["actors"]:
    saved = torch.load(actor["params"], map_location="cpu", weights_only=False)
    # Camera space, not world. This is the whole reason the file is read twice.
    body = saved["body_params_incam"]
    frames = len(body["body_pose"])

    # One shape for the whole track. It is the same person throughout.
    betas = body["betas"].reshape(frames, -1).float()[:, :10].mean(0, keepdim=True)

    model = smplx.create(model_path, model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=frames)
    with torch.no_grad():
        posed = model(betas=betas.expand(frames, -1),
                      global_orient=take(body, "global_orient", 3, frames),
                      body_pose=take(body, "body_pose", 63, frames),
                      transl=take(body, "transl", 3, frames))

    out[str(actor["track"])] = {
        "verts": posed.vertices.numpy().astype(np.float32),
        # GEM stores K per frame; it only varies if it estimated a zoom.
        "K": saved["K_fullimg"].numpy().astype(np.float32),
    }

np.savez(out_path, faces=model.faces.astype(np.int32),
         **{f"verts_{k}": v["verts"] for k, v in out.items()},
         **{f"K_{k}": v["K"] for k, v in out.items()})
print("@@" + json.dumps(sorted(out)))
"""


def build_vertices(actors, cache):
    """Turn each actor's rotations into vertices, once, inside GEM's venv."""
    python = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"needs smplx; GEM's venv not found at {GEM_ROOT}")

    spec = {"actors": [{"track": a["track"], "params": a["params"]} for a in actors]}
    proc = subprocess.run([python, "-c", _VERTS_SCRIPT, SMPLX_DIR, cache],
                          input=json.dumps(spec), capture_output=True, text=True)
    if not os.path.exists(cache):
        raise SystemExit(f"could not pose the bodies:\n{proc.stderr[-1200:]}")

    # Read out in full: indexing an npz decompresses the whole array again.
    with np.load(cache) as handle:
        return {name: handle[name] for name in handle.files}


@wp.kernel
def _raster(verts: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.vec3i),
            fx: float, fy: float, cx: float, cy: float, width: int, height: int,
            depth_id: wp.array(dtype=wp.int64)):
    """One thread per triangle, writing the nearest one to each pixel it covers.

    Depth and triangle are packed into a single integer, depth in the high half,
    so one atomic minimum settles both which triangle won and how far away it
    was. Painting back to front cannot be done this way round, since the order
    is the algorithm; a depth buffer has no order to preserve.
    """
    t = wp.tid()
    f = tris[t]
    a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
    if a[2] <= 1.0e-4 or b[2] <= 1.0e-4 or c[2] <= 1.0e-4:
        return

    ax, ay = a[0] / a[2] * fx + cx, a[1] / a[2] * fy + cy
    bx, by = b[0] / b[2] * fx + cx, b[1] / b[2] * fy + cy
    ux, uy = c[0] / c[2] * fx + cx, c[1] / c[2] * fy + cy

    lo_x = wp.max(0, int(wp.floor(wp.min(wp.min(ax, bx), ux))))
    hi_x = wp.min(width - 1, int(wp.ceil(wp.max(wp.max(ax, bx), ux))))
    lo_y = wp.max(0, int(wp.floor(wp.min(wp.min(ay, by), uy))))
    hi_y = wp.min(height - 1, int(wp.ceil(wp.max(wp.max(ay, by), uy))))
    if lo_x > hi_x or lo_y > hi_y:
        return

    area = (bx - ax) * (uy - ay) - (by - ay) * (ux - ax)
    if wp.abs(area) < 1.0e-9:
        return
    # Winding is not fixed after projection, so the tests are signed to match.
    turn = wp.where(area > 0.0, 1.0, -1.0)

    for y in range(lo_y, hi_y + 1):
        for x in range(lo_x, hi_x + 1):
            px, py = float(x) + 0.5, float(y) + 0.5
            e0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * turn
            e1 = ((ux - bx) * (py - by) - (uy - by) * (px - bx)) * turn
            e2 = ((ax - ux) * (py - uy) - (ay - uy) * (px - ux)) * turn
            if e0 < 0.0 or e1 < 0.0 or e2 < 0.0:
                continue

            z = (e1 * a[2] + e2 * b[2] + e0 * c[2]) / wp.abs(area)
            # A tenth of a millimeter of depth per step, finer than any body needs.
            key = (wp.int64(wp.clamp(z * 10000.0, 0.0, 2.0e9)) << wp.int64(32)) \
                | wp.int64(t)
            wp.atomic_min(depth_id, y * width + x, key)


@wp.kernel
def _resolve(depth_id: wp.array(dtype=wp.int64), color: wp.array(dtype=wp.vec3),
             alpha: float, frame: wp.array(dtype=wp.vec3)):
    """Blend the winning triangle's color over the footage, one thread a pixel."""
    p = wp.tid()
    key = depth_id[p]
    if key == _EMPTY:
        return
    c = color[int(key & wp.int64(0xFFFFFFFF))]
    frame[p] = frame[p] * (1.0 - alpha) + c * alpha


def project(verts, K):
    """Camera space to pixels. The perspective divide is where depth is lost."""
    # Behind the camera or on the plane: no image point exists, so park it.
    z = np.maximum(verts[:, 2], 1e-4)
    x = verts[:, 0] / z * K[0, 0] + K[0, 2]
    y = verts[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([x, y], axis=1), z


def shade(verts, faces, color, light=(0.3, -0.8, -0.5)):
    """Flat Lambertian per triangle, so the form reads without a lighting rig."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)

    d = np.array(light, dtype=np.float32)
    d = d / np.linalg.norm(d)
    # Two-sided, since which way a triangle faces after projection is not fixed.
    lit = 0.35 + 0.65 * np.abs(n @ d)
    return np.clip(np.array(color, dtype=np.float32)[None, :] * lit[:, None], 0, 255)


class Rasterizer:
    """Everybody in one depth buffer, so the near player hides the far one.

    Drawing each actor in turn cannot get that right: whoever is drawn last wins
    the pixel whether or not they are in front. One buffer for the whole frame
    settles it by depth, which is also the only way to give a thread per triangle
    instead of a python call per triangle.
    """

    def __init__(self, width, height, faces, actors, verts_each):
        wp.init()
        self.width, self.height = width, height
        self.pixels = width * height
        self.verts_each = verts_each
        self.faces_each = len(faces)

        # One triangle list for everybody, uploaded once since topology is fixed.
        stacked = np.concatenate([faces + i * verts_each for i in range(actors)])
        self.tris = wp.array(stacked.astype(np.int32), dtype=wp.vec3i)

        self.verts = wp.zeros(actors * verts_each, dtype=wp.vec3)
        self.color = wp.zeros(actors * self.faces_each, dtype=wp.vec3)
        self.depth_id = wp.zeros(self.pixels, dtype=wp.int64)
        self.frame = wp.zeros(self.pixels, dtype=wp.vec3)

    def draw(self, image, packed_verts, packed_colors, K, alpha):
        """One frame: upload, rasterize, resolve. Returns the blended image."""
        self.verts.assign(np.ascontiguousarray(packed_verts, np.float32))
        self.color.assign(np.ascontiguousarray(packed_colors, np.float32))
        self.depth_id.fill_(EMPTY_KEY)
        self.frame.assign(np.ascontiguousarray(image.reshape(-1, 3), np.float32))

        wp.launch(_raster, dim=len(self.tris),
                  inputs=[self.verts, self.tris,
                          float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                          self.width, self.height, self.depth_id])
        wp.launch(_resolve, dim=self.pixels,
                  inputs=[self.depth_id, self.color, float(alpha), self.frame])

        out = self.frame.numpy().reshape(self.height, self.width, 3)
        return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("actors", help="a run_gem.py actors.json")
    ap.add_argument("--out", default="overlay.mp4")
    ap.add_argument("--alpha", type=float, default=0.75, help="how solid the bodies are")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many frames")
    ap.add_argument("--by-team", action="store_true",
                    help="color by kit rather than one color each")
    args = ap.parse_args()

    import cv2

    with open(args.actors) as f:
        manifest = json.load(f)

    work = tempfile.mkdtemp(prefix="overlay.")
    posed = build_vertices(manifest["actors"], os.path.join(work, "verts.npz"))
    faces = posed["faces"]
    tracks = [a["track"] for a in manifest["actors"]]
    print(f"{len(tracks)} actors, {faces.shape[0]} triangles each")

    cap = cv2.VideoCapture(manifest["clip"])
    frames_dir = os.path.join(work, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    count = len(manifest["actors"])
    verts_each = posed[f"verts_{tracks[0]}"].shape[1]
    grid = Rasterizer(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      faces, count, verts_each)

    # Room for everybody. An actor not on yet is parked behind the camera.
    scratch = np.zeros((count * verts_each, 3), np.float32)
    tints = np.zeros((count * len(faces), 3), np.float32)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.limit and n >= args.limit):
            break

        scratch[:] = 0.0
        scratch[:, 2] = -1.0
        K = None
        for slot, actor in enumerate(manifest["actors"]):
            track = actor["track"]
            # By kit where one is known, so the two sides read apart at a glance.
            tone = actor["team"] if args.by_team and "team" in actor else slot
            verts = posed[f"verts_{track}"]
            # Each actor starts on its own frame, so it waits until it walks on.
            i = n - actor.get("first", 0)
            if not 0 <= i < len(verts):
                continue

            K = posed[f"K_{track}"][min(i, len(posed[f"K_{track}"]) - 1)]
            lo = slot * verts_each
            scratch[lo:lo + verts_each] = verts[i]
            tints[slot * len(faces):(slot + 1) * len(faces)] = shade(
                verts[i], faces, COLORS[tone % len(COLORS)])

        if K is not None:
            frame = grid.draw(frame, scratch, tints, K, args.alpha)

        cv2.imwrite(os.path.join(frames_dir, f"f_{n:05d}.png"), frame)
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames", flush=True)
    cap.release()

    if not n:
        raise SystemExit(f"no frames read from {manifest['clip']}")

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(frames_dir, "f_%05d.png"), "-c:v", "libx264",
                    "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", args.out],
                   check=True)
    print(f"wrote {args.out}  ({n} frames)")


if __name__ == "__main__":
    main()
