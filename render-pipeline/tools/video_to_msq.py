"""Estimate one person's motion from a video and write a mesh sequence.

The single subject path. For everybody in the video, plus the ball, use
video_to_scene.py instead.

    python tools/video_to_msq.py clip.mp4 --out clip.msq

GEM lives in its own virtualenv, so the heavy work runs as a subprocess. Point
GEM_ROOT at the clone if it is not in ~/GENMO.

GEM is under the NVIDIA OneWay Noncommercial License. See docs/INSTALL.md in the
clone for setup.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_to_msq import detect_up, ground_alignment, orient_to_y_up, place_in_scene  # noqa: E402
from msq import write_msq  # noqa: E402

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# Runs inside GEM's venv, turning its saved parameters into vertices.
_POSE_SCRIPT = r"""
import sys, numpy as np, torch, smplx

params_path, model_path, out_path = sys.argv[1:4]
saved = torch.load(params_path, map_location="cpu", weights_only=False)

# Global is world space. incam would slide about as the shot moves.
body = saved.get("body_params_global") or saved["body_params_incam"]
frames = len(body["body_pose"])

def take(key, width):
    if key not in body:
        return torch.zeros(frames, width)
    v = body[key].reshape(frames, -1).float()
    return v[:, :width] if v.shape[1] >= width else torch.nn.functional.pad(
        v, (0, width - v.shape[1]))

# GEM emits SMPL-X: body_pose is 21 joints, 63 values.
pose = take("body_pose", 63)
betas = body["betas"].reshape(frames, -1).float()[:, :10].mean(0, keepdim=True)

model = smplx.create(model_path, model_type="smplx", gender="neutral",
                     use_pca=False, batch_size=frames)
with torch.no_grad():
    out = model(betas=betas.expand(frames, -1),
                global_orient=take("global_orient", 3),
                body_pose=pose,
                transl=take("transl", 3))

np.savez(out_path, verts=out.vertices.numpy().astype(np.float32),
         faces=model.faces.astype(np.int32), joints=out.joints.numpy().astype(np.float32))
print(f"posed {frames} frames, {out.vertices.shape[1]} vertices")
"""


def source_fps(video, default=30.0):
    """Frame rate of the video, so the sequence plays back at the right speed."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video],
            capture_output=True, text=True, check=True).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception:
        return default


def gem_python():
    """The interpreter inside GEM's virtualenv."""
    path = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(path):
        raise SystemExit(
            f"GEM not found at {GEM_ROOT}.\n"
            "Clone https://github.com/NVlabs/GENMO and follow docs/INSTALL.md,\n"
            "or set GEM_ROOT to the clone."
        )
    return path


def estimate_motion(video, static_cam=False):
    """Run GEM over the video. Returns the path to its saved parameters."""
    name = os.path.splitext(os.path.basename(video))[0]
    saved = os.path.join(GEM_ROOT, "outputs", "demo_smpl", name, "smpl_params.pt")
    if os.path.exists(saved):
        print(f"reusing {saved}")
        return saved

    # demo_smpl_hpe.py is the pose only entry point, and needs no text prompt.
    cmd = [gem_python(), "scripts/demo/demo_smpl_hpe.py",
           "--video", os.path.abspath(video),
           "--ckpt_path", "inputs/pretrained/gem_smpl.ckpt",
           "--no_render"]
    if static_cam:
        cmd.append("--static_cam")

    print(f"estimating motion from {os.path.basename(video)}, this takes a while")
    subprocess.run(cmd, cwd=GEM_ROOT, check=True)
    if not os.path.exists(saved):
        raise SystemExit(f"GEM finished but wrote no parameters at {saved}")
    return saved


def pose_body(params_path, model_path):
    """Turn saved parameters into per frame vertices. Returns (verts, faces, joints)."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        out_path = tmp.name
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as script:
            script.write(_POSE_SCRIPT)
            script_path = script.name
        subprocess.run([gem_python(), script_path, params_path,
                        os.path.expanduser(model_path), out_path], check=True)
        data = np.load(out_path)
        return data["verts"], data["faces"], data["joints"]
    finally:
        for p in (out_path, script_path):
            if os.path.exists(p):
                os.unlink(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-path", default=os.environ.get("SMPLX_MODEL_PATH", "~/models"))
    ap.add_argument("-s", "--static-cam", action="store_true",
                    help="the camera does not move, which makes the trajectory easier")
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="keep GEM's own frame instead of leveling the floor")
    args = ap.parse_args()

    if not args.video.lower().endswith(VIDEO_EXTENSIONS):
        raise SystemExit(f"{args.video} does not look like a video")

    out = args.out or os.path.splitext(args.video)[0] + ".msq"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    verts, faces, joints = pose_body(
        estimate_motion(args.video, args.static_cam), args.model_path)
    print(f"  {verts.shape[0]} frames, {verts.shape[1]} vertices, {len(faces)} triangles")

    axis, sign = detect_up(joints)
    verts, joints = orient_to_y_up(verts, axis, sign), orient_to_y_up(joints, axis, sign)
    print(f"  up axis: {'-' if sign < 0 else '+'}{'xyz'[axis]}")

    if args.align:
        rotation, tilt, yaw = ground_alignment(joints)
        print(f"  leveled {tilt:.1f} deg of tilt, turned {yaw:.0f} deg to face the camera")
        verts = verts @ rotation.T

    verts = place_in_scene(verts, up="y")
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    print(f"  bounds  min {lo.round(2)}  max {hi.round(2)}")

    # Playback must match the source, which for broadcast is usually 25 not 30.
    write_msq(out, faces, verts, source_fps(args.video))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
