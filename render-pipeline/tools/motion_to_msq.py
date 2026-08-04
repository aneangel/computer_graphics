"""Turn SMPL and SMPL-X motion parameters into a mesh sequence the renderer reads.

Runs the body model forward, then writes a .msq. Nothing downstream knows what
SMPL is.

    python tools/motion_to_msq.py kungfu.zip --clip Kung_Fu_Flying_Kick_1_clip3 \
        --model-path ~/models --out kick.msq

Reads AMASS .npz and Motion-X (T, 322) arrays. Needs `pip install smplx` and the
model files, which are behind a form at https://smpl-x.is.tue.mpg.de
"""

import argparse
import fnmatch
import io
import sys
import zipfile

import numpy as np
import numpy.lib.format as npy_format

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from msq import write_msq  # noqa: E402

# Motion-X packs one row per frame, and these must sum to the row width.
MOTIONX_FIELDS = [
    ("root_orient", 3),
    ("pose_body", 63),
    ("pose_hand", 90),
    ("pose_jaw", 3),
    ("face_expr", 50),
    ("face_shape", 100),
    ("trans", 3),
    ("betas", 10),
]
MOTIONX_WIDTH = sum(n for _, n in MOTIONX_FIELDS)  # 322


def load_motion(path):
    """Sniff the layout and return a dict of arrays plus the model type."""
    raw = np.load(path, allow_pickle=True)

    # Motion-X is a single (T, 322) block, either bare or under one key.
    array = None
    if isinstance(raw, np.ndarray):
        array = raw
    elif hasattr(raw, "files") and "poses" not in raw.files:
        for key in raw.files:
            if raw[key].ndim == 2 and raw[key].shape[1] == MOTIONX_WIDTH:
                array = raw[key]
                break

    if array is not None:
        if array.ndim != 2 or array.shape[1] != MOTIONX_WIDTH:
            raise ValueError(
                f"expected a (T, {MOTIONX_WIDTH}) Motion-X array, got {array.shape}"
            )
        out, at = {}, 0
        for name, width in MOTIONX_FIELDS:
            out[name] = array[:, at : at + width].astype(np.float32)
            at += width
        # betas repeat every frame but describe a fixed body.
        out["betas"] = out["betas"][0]
        out["model_type"] = "smplx"
        out["fps"] = 30.0  # Motion-X standardises everything to 30
        return out

    # AMASS: separate arrays, SMPL+H pose layout.
    if "poses" not in raw.files:
        raise ValueError(f"{path}: not a recognised AMASS or Motion-X file")
    poses = raw["poses"].astype(np.float32)
    if poses.shape[1] < 72:
        raise ValueError(f"expected at least 72 pose values per frame, got {poses.shape[1]}")

    fps = 30.0
    for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate"):
        if key in raw.files:
            fps = float(raw[key])
            break

    return {
        "root_orient": poses[:, :3],
        # SMPL has 23 joints past the root, and AMASS stores hands after those.
        "pose_body": poses[:, 3:72],
        "trans": raw["trans"].astype(np.float32),
        "betas": raw["betas"].astype(np.float32)[:10],
        "gender": str(raw["gender"]) if "gender" in raw.files else "neutral",
        "model_type": "smpl",
        "fps": fps,
    }


def clip_shape(handle):
    """Shape of a .npy from its header, without decompressing the array."""
    version = npy_format.read_magic(handle)
    reader = (npy_format.read_array_header_1_0 if version == (1, 0)
              else npy_format.read_array_header_2_0)
    shape, _fortran, _dtype = reader(handle)
    return shape


def list_archive(path, pattern=None):
    """Print every clip in a Motion-X subset zip, optionally filtered."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".npy"))
        if pattern:
            lowered = f"*{pattern.lower()}*"
            names = [n for n in names if fnmatch.fnmatch(n.lower(), lowered)]

        total = 0.0
        for name in names:
            with archive.open(name) as handle:
                shape = clip_shape(handle)
            seconds = shape[0] / 30.0
            total += seconds
            label = name.rsplit("/", 1)[-1][: -len(".npy")]
            print(f"  {shape[0]:5d} frames  {seconds:6.1f}s  {label}")

        print(f"{len(names)} clips, {total / 60.0:.1f} minutes"
              + (f" matching {pattern!r}" if pattern else ""))
    return names


def open_clip(path, clip):
    """Read one clip straight out of a subset zip, no extraction needed."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".npy")]
        exact = [n for n in names if n.rsplit("/", 1)[-1][: -len(".npy")] == clip]
        matches = exact or [n for n in names if clip.lower() in n.lower()]
        if not matches:
            raise SystemExit(f"no clip matching {clip!r} in {path}; try --list")
        if len(matches) > 1 and not exact:
            listing = "\n  ".join(m.rsplit("/", 1)[-1] for m in matches[:8])
            raise SystemExit(f"{clip!r} matches {len(matches)} clips:\n  {listing}\n  ...")
        print(f"clip: {matches[0]}")
        return io.BytesIO(archive.read(matches[0]))


def detect_up(joints, pelvis=0, head=15):
    """Up axis and sign, from the mean pelvis to head vector."""
    direction = (joints[:, head] - joints[:, pelvis]).mean(axis=0)
    axis = int(np.argmax(np.abs(direction)))
    return axis, float(np.sign(direction[axis]))


# SMPL-X body joints, in the order the model emits them.
J_PELVIS, J_LHIP, J_RHIP = 0, 1, 2
J_LFOOT, J_RFOOT, J_HEAD = 10, 11, 15


def rotation_between(a, b):
    """Shortest rotation taking unit vector `a` onto unit vector `b`."""
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel, so any perpendicular axis will do for the half turn.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = np.cross(a, perp)
        perp /= np.linalg.norm(perp)
        K = np.array([[0, -perp[2], perp[1]], [perp[2], 0, -perp[0]], [-perp[1], perp[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def ground_alignment(joints, max_tilt=50.0, min_spread=0.05):
    """Rotation standing the body level and facing the camera, plus tilt and yaw."""
    feet = joints[:, [J_LFOOT, J_RFOOT]].reshape(-1, 3)
    low = feet[feet[:, 1] <= np.quantile(feet[:, 1], 0.25)]

    normal = None
    if len(low) >= 3:
        # The normal is the direction of least spread in the contact points.
        centred = low - low.mean(axis=0)
        _u, singular, basis = np.linalg.svd(centred, full_matrices=False)
        spread = singular / np.sqrt(len(low))  # metres, independent of frame count
        # A line of contacts fixes no plane, and a tiny patch of them is noise.
        if spread[1] > 0.05 * spread[0] and spread[1] > min_spread:
            normal = basis[-1]
            if normal[1] < 0.0:
                normal = -normal

    if normal is None:
        # Nothing to level against, and posture is no substitute, so leave it alone.
        return _facing_only(joints)

    tilt = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    if tilt > max_tilt:
        return np.eye(3), tilt, 0.0

    level = rotation_between(normal, np.array([0.0, 1.0, 0.0]))

    return _apply_facing(joints, level, tilt)


def _facing_only(joints):
    """Turn the subject to face the camera without touching the tilt."""
    return _apply_facing(joints, np.eye(3), 0.0)


def _apply_facing(joints, level, tilt):
    """Compose `level` with a turn about the vertical so the body faces +Z."""
    # The hip axis crossed with up gives forward, averaged over the whole clip.
    hips = joints[:, J_RHIP] - joints[:, J_LHIP]
    hips = hips @ level.T
    hips /= np.maximum(np.linalg.norm(hips, axis=1, keepdims=True), 1e-9)
    forward = np.cross(hips, np.array([0.0, 1.0, 0.0]))
    mean_forward = forward.mean(axis=0)
    mean_forward[1] = 0.0
    norm = float(np.linalg.norm(mean_forward))
    if norm < 1e-6:
        return level, tilt, 0.0
    mean_forward /= norm

    # Turn about the vertical until the body faces the camera.
    yaw = float(np.arctan2(mean_forward[0], mean_forward[2]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return spin @ level, tilt, float(np.degrees(yaw))


def orient_to_y_up(verts, axis, sign):
    """Quarter turn fixing which axis is up. Lean is ground_alignment's job."""
    x, y, z = verts[..., 0], verts[..., 1], verts[..., 2]
    return {
        (0, 1.0): lambda: np.stack([-y, x, z], -1),
        (0, -1.0): lambda: np.stack([y, -x, z], -1),
        (1, 1.0): lambda: verts,
        (1, -1.0): lambda: np.stack([x, -y, -z], -1),
        (2, 1.0): lambda: np.stack([x, z, -y], -1),
        (2, -1.0): lambda: np.stack([x, -z, y], -1),
    }[(axis, sign)]()


def to_vertices(motion, model_path, chunk=128):
    """Run the body model forward. Face and jaw are skipped as too small to see."""
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"{exc.name} is not installed.\n"
            "  pip install smplx torch\n"
            "The body model files are separate, and gated behind a registration\n"
            "form: https://smpl-x.is.tue.mpg.de"
        ) from exc

    kind = motion["model_type"]
    frames = len(motion["root_orient"])
    gender = motion.get("gender", "neutral")
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"

    model = smplx.create(
        model_path,
        model_type=kind,
        gender=gender,
        use_pca=False,  # Motion-X stores full axis-angle hand poses, not PCA
        batch_size=chunk,
    )

    betas = torch.from_numpy(np.asarray(motion["betas"], dtype=np.float32))[None]
    betas = betas[:, : model.num_betas].expand(chunk, -1)

    out, joints = [], []
    for start in range(0, frames, chunk):
        stop = min(start + chunk, frames)

        def slab(name):
            # Fixed batch size, so the last chunk is padded then trimmed.
            a = torch.from_numpy(motion[name][start:stop])
            if a.shape[0] < chunk:
                a = torch.cat([a, a[-1:].repeat(chunk - a.shape[0], 1)], dim=0)
            return a

        kwargs = {
            "betas": betas,
            "global_orient": slab("root_orient"),
            "body_pose": slab("pose_body"),
            "transl": slab("trans"),
        }
        if kind == "smplx":
            hands = slab("pose_hand")
            kwargs["left_hand_pose"] = hands[:, :45]
            kwargs["right_hand_pose"] = hands[:, 45:]

        with torch.no_grad():
            result = model(**kwargs)
        out.append(result.vertices.numpy()[: stop - start])
        joints.append(result.joints.numpy()[: stop - start])

    return (np.concatenate(out, axis=0), model.faces.astype(np.int32),
            np.concatenate(joints, axis=0))


def place_in_scene(verts, up="z", height=6.0, centre=(0.0, -16.0), floor_y=-4.0):
    """Fit the sequence into the default scene, once over every frame."""
    if up == "z":
        # Body models are usually authored z up, and the renderer is y up.
        verts = np.stack([verts[..., 0], verts[..., 2], -verts[..., 1]], axis=-1)
    elif up != "y":
        raise ValueError("up must be 'z' or 'y'")

    # Median height of one frame, since a jump sweeps more space than a body fills.
    per_frame = np.median(verts[..., 1].max(axis=1) - verts[..., 1].min(axis=1))
    scale = height / max(float(per_frame), 1e-6)

    verts = verts * scale
    flat = verts.reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    mid = (lo + hi) * 0.5
    # A low quantile, so one bad frame cannot lift the clip off the ground.
    ground = float(np.quantile(flat[:, 1], 0.002))
    offset = np.array([centre[0] - mid[0], floor_y - ground, centre[1] - mid[2]], np.float32)
    return (verts + offset).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("motion", help="a Motion-X subset .zip, or a single .npy / AMASS .npz")
    ap.add_argument("--clip", help="clip name inside a subset zip")
    ap.add_argument("--list", nargs="?", const="", metavar="PATTERN",
                    help="list the clips in a subset zip and exit")
    ap.add_argument("--model-path", help="directory holding the SMPL(-X) models")
    ap.add_argument("--out", default="motion.msq")
    ap.add_argument("--up", default="auto",
                    choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
                    help="up axis of the source motion; auto reads it off the skeleton")
    ap.add_argument("--max-frames", type=int, default=0, help="0 for all")
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="skip levelling and facing; keep the capture's own frame")
    args = ap.parse_args()

    if args.list is not None:
        list_archive(args.motion, args.list or None)
        return
    if not args.model_path:
        ap.error("--model-path is required (or use --list to browse a subset)")

    if zipfile.is_zipfile(args.motion):
        if not args.clip:
            ap.error(f"{args.motion} is a subset archive; pass --clip NAME or --list")
        source = open_clip(args.motion, args.clip)
    else:
        source = args.motion

    motion = load_motion(source)
    if args.max_frames:
        for key in ("root_orient", "pose_body", "pose_hand", "trans"):
            if key in motion and isinstance(motion[key], np.ndarray) and motion[key].ndim == 2:
                motion[key] = motion[key][: args.max_frames]

    print(f"{motion['model_type']}, {len(motion['root_orient'])} frames "
          f"at {motion['fps']:g} fps")

    verts, faces, joints = to_vertices(motion, args.model_path)

    guess = detect_up(joints)
    if args.up == "auto":
        axis, sign = guess
    else:
        axis, sign = "xyz".index(args.up.lstrip("-")), -1.0 if args.up.startswith("-") else 1.0
    print(f"  up axis: {'-' if sign < 0 else '+'}{'xyz'[axis]}"
          + ("" if args.up == "auto" else " (forced)")
          + f"   [measured from pelvis->head: {'-' if guess[1] < 0 else '+'}{'xyz'[guess[0]]}]")

    verts = orient_to_y_up(verts, axis, sign)
    joints = orient_to_y_up(joints, axis, sign)

    if args.align:
        rotation, tilt, yaw = ground_alignment(joints)
        print(f"  levelled {tilt:.1f} deg of tilt, turned {yaw:.0f} deg to face the camera")
        verts = verts @ rotation.T

    verts = place_in_scene(verts, up="y")

    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"  {verts.shape[1]} vertices, {len(faces)} triangles")
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")
    print("  (a standing figure should span about 6 units in y, feet near -4)")

    write_msq(args.out, faces, verts, motion["fps"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
