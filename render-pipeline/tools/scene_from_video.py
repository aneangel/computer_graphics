"""Turn estimated actors and a tracked ball into one scene the renderer opens.

Actors are placed in camera space, the one frame every estimator run shares.
The ball is placed from its own apparent size. The scene is scaled once, over
everybody together, so their spacing survives.

    python tools/run_gem.py tracks/clip.json --out gem/clip
    python tools/scene_from_video.py gem/clip/actors.json --out scene.msq
    ./build/render-pipeline scene.msq
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_to_msq import rotation_between  # noqa: E402
from msq import write_msq  # noqa: E402
from scene_msq import BODY_COLOURS, BALL_COLOUR, merge, uv_sphere  # noqa: E402

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))

# Radius of a size 5 football, which is what turns apparent size into distance.
BALL_RADIUS_M = 0.11

# Runs inside GEM's venv, posing every actor in one go.
_POSE_SCRIPT = r"""
import sys, json, numpy as np, torch, smplx

model_path, out_path = sys.argv[1:3]
paths = json.loads(sys.stdin.read())

posed, faces, K = [], None, None
for params_path in paths:
    saved = torch.load(params_path, map_location="cpu", weights_only=False)
    # Same camera for every actor, read here because the file is already open.
    K = saved["K_fullimg"][0].numpy().astype(np.float32)
    # incam, not global, since only the camera is shared between separate runs.
    body = saved["body_params_incam"]
    frames = len(body["body_pose"])

    def take(key, width):
        if key not in body:
            return torch.zeros(frames, width)
        v = body[key].reshape(frames, -1).float()
        return v[:, :width] if v.shape[1] >= width else torch.nn.functional.pad(
            v, (0, width - v.shape[1]))

    # GEM emits SMPL-X: body_pose is 21 joints, 63 values.
    betas = body["betas"].reshape(frames, -1).float()[:, :10].mean(0, keepdim=True)
    model = smplx.create(model_path, model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=frames)
    with torch.no_grad():
        out = model(betas=betas.expand(frames, -1),
                    global_orient=take("global_orient", 3),
                    body_pose=take("body_pose", 63),
                    transl=take("transl", 3))
    posed.append(out.vertices.numpy().astype(np.float32))
    faces = model.faces.astype(np.int32)

np.savez(out_path, faces=faces, count=len(posed), K=K,
         **{f"verts{i}": v for i, v in enumerate(posed)})
print(f"posed {len(posed)} actors, {posed[0].shape[0]} frames")
"""


def pose_actors(params_paths, model_path):
    """Run the body model over every actor. Returns (verts per actor, faces, K)."""
    python = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"GEM's venv not found at {GEM_ROOT}")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        out_path = tmp.name
    try:
        proc = subprocess.run(
            [python, "-c", _POSE_SCRIPT, os.path.expanduser(model_path), out_path],
            input=json.dumps(params_paths), capture_output=True, text=True)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise SystemExit(f"posing failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
        data = np.load(out_path)
        return ([data[f"verts{i}"] for i in range(int(data["count"]))],
                data["faces"], data["K"])
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def camera_to_world(points):
    """Camera space to the renderer's frame, a half turn about x."""
    out = np.array(points, np.float32, copy=True)
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def ball_observations(boxes, seen, K):
    """Ball position in camera space, from its apparent size and pixel."""
    boxes, seen = np.asarray(boxes, np.float64), np.asarray(seen, bool)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = (boxes[:, 0] + boxes[:, 2]) * 0.5
    v = (boxes[:, 1] + boxes[:, 3]) * 0.5
    # Averaged over both spans, since motion blur stretches one of them.
    radius = (boxes[:, 2] - boxes[:, 0] + boxes[:, 3] - boxes[:, 1]) * 0.25

    with np.errstate(divide="ignore", invalid="ignore"):
        z = fx * BALL_RADIUS_M / radius
    good = seen & np.isfinite(z) & (radius > 1.0)

    points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    return points, good, np.stack([u, v], axis=1)


def anchor_to_bodies(points, good, uv, actors, K, pixel_radius=45.0):
    """Where the ball sits over a body in frame, take the body's depth instead.

    A body's depth is far better known than a ball's, so the body wins. Returns
    the corrected points and which frames were anchored.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    anchored = np.zeros(len(points), bool)
    out = points.copy()

    for t in np.flatnonzero(good):
        best = None
        for verts in actors:
            # Every twentieth vertex is plenty to place a ball against a body.
            body = verts[t][::20]
            depth = body[:, 2]
            ahead = depth > 1e-3
            if not ahead.any():
                continue
            u = body[ahead, 0] * fx / depth[ahead] + cx
            v = body[ahead, 1] * fy / depth[ahead] + cy
            far = np.hypot(u - uv[t, 0], v - uv[t, 1])
            near = int(np.argmin(far))
            if far[near] < pixel_radius and (best is None or depth[ahead][near] < best):
                best = float(depth[ahead][near])

        if best is None:
            continue
        # Moved along its own ray, since the direction was never in doubt.
        anchored[t] = True
        out[t] = [(uv[t, 0] - cx) * best / fx, (uv[t, 1] - cy) * best / fy, best]

    return out, anchored


def fit_trajectory(points, good, weight_depth=0.02, smooth=40.0):
    """Smooth path through sparse observations, as one least squares system.

    Fits the measurements and penalises sudden changes of velocity, which is
    what carries the trajectory across the gaps. Depth is weighted low.
    """
    n = len(points)
    if good.sum() < 2:
        return None

    # Second difference rows, saying velocity did not change abruptly.
    rows = max(0, n - 2)
    A = np.zeros((n + rows, n))
    b = np.zeros((n + rows, 3))

    weights = np.where(good, 1.0, 0.0)
    A[np.arange(n), np.arange(n)] = weights
    b[:n] = points * weights[:, None]

    for i in range(rows):
        A[n + i, i] = smooth
        A[n + i, i + 1] = -2.0 * smooth
        A[n + i, i + 2] = smooth

    out = np.zeros((n, 3))
    for axis in range(3):
        # Per frame weight, so a depth taken from a body can outrank a guessed one.
        trust = np.asarray(weight_depth if axis == 2 else 1.0, np.float64)
        trust = np.broadcast_to(trust, (n,)).copy()
        Aa, ba = A.copy(), b[:, axis].copy()
        Aa[:n] *= trust[:, None]
        ba[:n] *= trust
        out[:, axis] = np.linalg.lstsq(Aa, ba, rcond=None)[0]
    return out


def limit_speed(verts, fps, max_speed=12.0):
    """Cap how far a body may travel between frames, and rebuild its path.

    Catches a slipped box taking the recovered depth with it. The cap is a
    sprinter's top speed, so only impossible motion is touched. Returns the
    corrected vertices and how far the worst frame moved.
    """
    centre = verts.mean(axis=1)
    steps = np.diff(centre, axis=0)
    distance = np.linalg.norm(steps, axis=1)

    cap = max_speed / max(fps, 1e-6)
    over = distance > cap
    if not over.any():
        return verts, 0.0

    # Offending steps scaled back to the cap, keeping their direction.
    scaled = steps.copy()
    scaled[over] *= (cap / distance[over])[:, None]
    corrected = np.concatenate([np.zeros((1, 3)), np.cumsum(scaled, axis=0)])

    # Anchored on the median, since frame zero is as likely as any to be the bad one.
    corrected += np.median(centre, axis=0) - np.median(corrected, axis=0)

    shift = (corrected - centre).astype(np.float32)
    return verts + shift[:, None, :], float(np.abs(shift).max())


def ground_plane(actors, quantile=0.15):
    """The plane the actors are standing on, fitted to every foot contact."""
    contacts = []
    for verts in actors:
        low = verts[..., 1].argmin(axis=1)
        contacts.append(verts[np.arange(len(verts)), low])
    contacts = np.concatenate(contacts)

    # Only the lowest of them, since a raised foot is touching nothing.
    keep = contacts[contacts[:, 1] <= np.quantile(contacts[:, 1], quantile)]
    if len(keep) < 8:
        return np.array([0.0, 1.0, 0.0]), float(contacts[:, 1].min())

    centred = keep - keep.mean(axis=0)
    _u, singular, basis = np.linalg.svd(centred, full_matrices=False)
    spread = singular / np.sqrt(len(keep))
    # A line or a patch of contacts fixes no plane, so decline rather than guess.
    if spread[1] < 0.05 * spread[0] or spread[1] < 0.05:
        return np.array([0.0, 1.0, 0.0]), float(np.quantile(contacts[:, 1], 0.02))

    normal = basis[-1]
    if normal[1] < 0.0:
        normal = -normal
    return normal, float(keep.mean(axis=0) @ normal)


def level_scene(actors, ball, normal, max_tilt=25.0):
    """Stand the whole scene up, one rotation for everybody, no turn about vertical."""
    tilt = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    if tilt > max_tilt or tilt < 1e-3:
        # Past this the fit found something that is not a floor, so leave it alone.
        return actors, ball, tilt if tilt > max_tilt else 0.0

    R = rotation_between(normal, np.array([0.0, 1.0, 0.0])).astype(np.float32)
    return ([v @ R.T for v in actors],
            None if ball is None else ball @ R.T, tilt)


def rest_on_ground(ball, floor, radius):
    """Lift the ball out of the floor, then smooth away the corner that leaves."""
    if ball is None:
        return None
    lifted = ball.copy()
    below = lifted[:, 1] < floor + radius
    if not below.any():
        return lifted
    lifted[below, 1] = floor + radius

    good = np.ones(len(lifted), bool)
    smoothed = fit_trajectory(lifted, good, weight_depth=1.0, smooth=2.0)
    return lifted if smoothed is None else smoothed.astype(np.float32)


def place_scene(actors, ball, height=6.0, centre_x=0.0, nearest_z=-11.0, floor_y=-4.0):
    """Fit the scene into the renderer's box, one transform for everybody.

    Scale comes off the actors, whose real size is the one thing known for free.
    """
    per_actor = [float(np.median(v[..., 1].max(axis=1) - v[..., 1].min(axis=1)))
                 for v in actors]
    scale = height / max(float(np.median(per_actor)), 1e-6)

    actors = [v * scale for v in actors]
    ball = None if ball is None else ball * scale

    flat = np.concatenate([v.reshape(-1, 3) for v in actors])
    lo, hi = flat.min(0), flat.max(0)
    # Pushed back until the nearest body clears the camera, not centred on a point.
    ground = float(np.quantile(flat[:, 1], 0.002))
    offset = np.array([centre_x - (lo[0] + hi[0]) * 0.5,
                       floor_y - ground,
                       nearest_z - hi[2]], np.float32)

    return ([(v + offset).astype(np.float32) for v in actors],
            None if ball is None else (ball + offset).astype(np.float32),
            scale, floor_y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("actors", help="a run_gem.py actors.json")
    ap.add_argument("--out", default="scene.msq")
    ap.add_argument("--model-path", default=os.environ.get("SMPLX_MODEL_PATH", "~/models"))
    ap.add_argument("--fps", type=float, default=0.0, help="0 reads it off the clip")
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--contact-radius", type=float, default=45.0,
                    help="pixels between ball and body counting as touching")
    ap.add_argument("--max-speed", type=float, default=12.0,
                    help="metres a second a body may travel, above a sprinter's best")
    args = ap.parse_args()

    with open(args.actors) as f:
        scene = json.load(f)

    fps = args.fps
    if not fps:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", scene["clip"]],
            capture_output=True, text=True).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den or 1) if num else 30.0

    posed, faces, K = pose_actors([a["params"] for a in scene["actors"]], args.model_path)
    print(f"{len(posed)} actors, {posed[0].shape[0]} frames, {len(faces)} triangles each")

    # Unprojected with the estimator's own intrinsics, so it shares their space.
    ball_world = None
    ball_points = None
    if not args.no_ball and any(scene["ball"]["seen"]):
        ball_points, ball_good, ball_uv = ball_observations(
            scene["ball"]["boxes"], scene["ball"]["seen"], K)
        print(f"ball: {int(ball_good.sum())} observations over {len(ball_good)} frames, "
              f"depth from size {ball_points[ball_good][:, 2].min():.1f}.."
              f"{ball_points[ball_good][:, 2].max():.1f} m")

    for i, verts in enumerate(posed):
        posed[i], moved = limit_speed(verts, fps, args.max_speed)
        if moved > 0.05:
            print(f"  actor {scene['actors'][i]['track']}: pulled back {moved:.1f} m of "
                  f"travel no runner could have made")

    # After the bodies are corrected, since the ball is about to follow them.
    ball_world = None
    if ball_points is not None:
        anchored_points, anchored = anchor_to_bodies(
            ball_points, ball_good, ball_uv, posed, K, args.contact_radius)
        if anchored.any():
            moved = np.abs(anchored_points[anchored, 2] - ball_points[anchored, 2])
            print(f"  ball: {int(anchored.sum())} of {int(ball_good.sum())} sightings were "
                  f"over a player, moving them a median {np.median(moved):.1f} m in depth")
        # A depth off a body is trusted, one off a radius is worth a fiftieth.
        trust = np.where(anchored, 1.0, 0.02)
        fitted = fit_trajectory(anchored_points, ball_good, weight_depth=trust)
        if fitted is not None:
            ball_world = camera_to_world(fitted)

    actors_world = [camera_to_world(v) for v in posed]

    normal, _ = ground_plane(actors_world)
    actors_world, ball_world, tilt = level_scene(actors_world, ball_world, normal)
    print(f"levelled {tilt:.1f} deg of camera tilt, read off {len(actors_world)} actors' feet")

    # Before scaling, so the clearance is a real ball radius.
    if ball_world is not None:
        _, floor_m = ground_plane(actors_world)
        ball_world = rest_on_ground(ball_world, floor_m, BALL_RADIUS_M)

    actors_world, ball_world, scale, floor_y = place_scene(actors_world, ball_world)
    print(f"scene scale {scale:.2f} (one metre is {scale:.2f} units)")

    parts = []
    for i, verts in enumerate(actors_world):
        parts.append((verts, faces, [(0, len(faces), BODY_COLOURS[i % len(BODY_COLOURS)])]))
        lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
        print(f"  actor {scene['actors'][i]['track']}: x {lo[0]:6.1f}..{hi[0]:5.1f}  "
              f"z {lo[2]:6.1f}..{hi[2]:5.1f}")

    if ball_world is not None:
        sphere, ball_faces = uv_sphere(BALL_RADIUS_M * scale)
        parts.append((sphere[None, :, :] + ball_world[:, None, :], ball_faces,
                      [(0, len(ball_faces), BALL_COLOUR)]))
        print(f"  ball: radius {BALL_RADIUS_M * scale:.2f} units")

    merged_faces, verts, objects, frames = merge(parts)
    write_msq(args.out, merged_faces, verts, fps, objects)

    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"\nwrote {args.out}")
    print(f"  {len(objects)} objects, {verts.shape[1]} vertices, {len(merged_faces)} triangles")
    print(f"  {frames} frames at {fps:g} fps")
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")


if __name__ == "__main__":
    main()
