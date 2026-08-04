"""Checks for motion_to_msq that do not need the licensed body models.

    python tools/test_motion_to_msq.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_to_msq import (MOTIONX_WIDTH, J_LFOOT, J_RFOOT, ground_alignment,  # noqa: E402
                           load_motion, place_in_scene, rotation_between)
from msq import read_msq, write_msq  # noqa: E402

TMP = tempfile.mkdtemp()
failures = []


def check(name, condition, detail=""):
    print(f"  {'pass' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


print("Motion-X layout")
frames = 7
block = np.zeros((frames, MOTIONX_WIDTH), np.float32)
# Distinct value per field, so a mis-sliced offset shows up.
for value, (start, width) in enumerate(
    [(0, 3), (3, 63), (66, 90), (156, 3), (159, 50), (209, 100), (309, 3), (312, 10)], start=1
):
    block[:, start : start + width] = value
path = os.path.join(TMP, "motionx.npy")
np.save(path, block)

m = load_motion(path)
check("model type", m["model_type"] == "smplx", m["model_type"])
check("fps", m["fps"] == 30.0)
check("root_orient", m["root_orient"].shape == (frames, 3) and np.all(m["root_orient"] == 1))
check("pose_body", m["pose_body"].shape == (frames, 63) and np.all(m["pose_body"] == 2))
check("pose_hand", m["pose_hand"].shape == (frames, 90) and np.all(m["pose_hand"] == 3))
check("trans", m["trans"].shape == (frames, 3) and np.all(m["trans"] == 7))
check("betas collapsed to one row", m["betas"].shape == (10,) and np.all(m["betas"] == 8))

print("AMASS layout")
poses = np.zeros((frames, 156), np.float32)
poses[:, :3] = 1.0
poses[:, 3:72] = 2.0
poses[:, 72:] = 9.0  # hands, which SMPL should drop
path = os.path.join(TMP, "amass.npz")
np.savez(path, poses=poses, betas=np.arange(16, dtype=np.float32),
         trans=np.full((frames, 3), 5.0, np.float32), gender="male", mocap_framerate=120.0)

m = load_motion(path)
check("model type", m["model_type"] == "smpl", m["model_type"])
check("fps from file", m["fps"] == 120.0)
check("gender", m["gender"] == "male")
check("body pose is 69 wide", m["pose_body"].shape == (frames, 69))
check("hand poses dropped", np.all(m["pose_body"] == 2))
check("betas trimmed to 10", m["betas"].shape == (10,))

print("axis conversion")
# One point one unit along each axis, z-up. Expect (x, z, -y).
pts = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float32)
out = place_in_scene(pts, up="z", height=1.0, centre=(0.0, 0.0), floor_y=0.0)
direction = out[0, 2] - out[0, 0]  # was +z, should now point up in y
check("z becomes up", direction[1] > 0, f"delta {direction.round(2)}")

print("scene fit")
# A 2-unit-tall box that also drifts sideways over three frames.
box = np.array([[-0.5, -1.0, -0.5], [0.5, -1.0, 0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, -0.5]], np.float32)
seq = np.stack([box + np.array([d, 0, 0], np.float32) for d in (0.0, 1.0, 2.0)])
out = place_in_scene(seq, up="y", height=6.0, centre=(0.0, -16.0), floor_y=-4.0)

check("scaled to requested height",
      abs((out[..., 1].max() - out[..., 1].min()) - 6.0) < 1e-4,
      f"{out[..., 1].max() - out[..., 1].min():.3f}")
check("feet land on the floor", abs(out[..., 1].min() - (-4.0)) < 1e-4, f"{out[..., 1].min():.3f}")
check("centred on z", abs(out[..., 2].mean() - (-16.0)) < 1e-4, f"{out[..., 2].mean():.3f}")

# Drift of 1.0 per frame at scale 3.0 must survive as 3.0.
drift = out[1, :, 0].mean() - out[0, :, 0].mean()
check("motion preserved, not fitted away", abs(drift - 3.0) < 1e-4, f"drift {drift:.3f}")

print("rotation_between")
rng = np.random.default_rng(1)
worst_map, worst_det = 0.0, 0.0
for _ in range(200):
    a, b = rng.normal(size=3), rng.normal(size=3)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    R = rotation_between(a, b)
    worst_map = max(worst_map, float(np.abs(R @ a - b).max()))
    worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
check("maps a onto b", worst_map < 1e-9, f"max err {worst_map:.1e}")
check("stays a rotation", worst_det < 1e-9, f"max |det-1| {worst_det:.1e}")
R = rotation_between(np.array([0., 1., 0.]), np.array([0., -1., 0.]))
check("handles the half turn", np.allclose(R @ [0, 1, 0], [0, -1, 0], atol=1e-9))

print("ground_alignment recovers a known tilt and yaw")
# Path covers two dimensions on purpose; a line of contacts fits no plane.
def standing(frames=60):
    t = np.linspace(0, 2 * np.pi, frames)[:, None]
    x, z = 0.6 * np.cos(t), 0.6 * np.sin(t)
    one = np.ones_like(x)
    j = np.zeros((frames, 16, 3))
    j[:, 0] = np.c_[x, 1.00 * one, z]                      # pelvis
    j[:, 15] = np.c_[x, 1.60 * one, z]                     # head
    j[:, 1] = np.c_[x - 0.10, 0.95 * one, z]               # left hip
    j[:, 2] = np.c_[x + 0.10, 0.95 * one, z]               # right hip
    j[:, J_LFOOT] = np.c_[x - 0.10, 0.05 * one, z]
    j[:, J_RFOOT] = np.c_[x + 0.10, 0.05 * one, z]
    return j

def spin(angle, axis):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0: return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == 1: return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

base = standing()
for tilt_deg, yaw_deg in ((0, 0), (25, 0), (12, 40), (30, -110)):
    j = base @ spin(np.radians(yaw_deg), 1).T @ spin(np.radians(tilt_deg), 2).T
    R, tilt, yaw = ground_alignment(j)
    fixed = j @ R.T
    feet = fixed[:, [J_LFOOT, J_RFOOT]].reshape(-1, 3)
    sel = feet[feet[:, 1] <= np.quantile(feet[:, 1], 0.25)]
    normal = np.linalg.svd(sel - sel.mean(0), full_matrices=False)[2][-1]
    residual = np.degrees(np.arccos(np.clip(abs(normal[1]), -1, 1)))
    forward = np.cross((fixed[:, 2] - fixed[:, 1]), [0, 1, 0]).mean(0)
    facing = np.degrees(np.arctan2(forward[0], forward[2]))
    check(f"tilt {tilt_deg:>4}deg yaw {yaw_deg:>4}deg -> level and facing camera",
          residual < 0.5 and abs(facing) < 1.0,
          f"residual {residual:.2f}deg, facing {facing:+.1f}deg")

print("collinear contacts fall back instead of inventing a plane")
line = standing()
line[:, [J_LFOOT, J_RFOOT], 2] = 0.0        # every footfall on one line
line[:, [J_LFOOT, J_RFOOT], 1] = 0.05
R, tilt, _ = ground_alignment(line)
check("upright figure stays upright", abs(tilt) < 1.0, f"tilt {tilt:.2f}deg")

print("a subject standing on one spot is not levelled from posture")
# Bent double with feet planted, where levelling from either cue stands him up.
bent = standing(40)
bent[:, [J_LFOOT, J_RFOOT], 0] = np.array([-0.10, 0.10])       # feet still
bent[:, [J_LFOOT, J_RFOOT], 2] = 0.0
bent[:, [J_LFOOT, J_RFOOT], 1] = 0.05
bent[:, 15] = np.c_[np.zeros(40), np.full(40, 1.15), np.full(40, 0.55)]   # head forward

def lean(j):
    v = j[:, 15] - j[:, 0]
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(v[:, 1], -1, 1))).mean()

R, tilt, _ = ground_alignment(bent)
before, after = lean(bent), lean(bent @ R.T)
check("declines to level", abs(tilt) < 1e-9, f"tilt {tilt:.1f}deg")
check("the bend survives", abs(after - before) < 0.5, f"{before:.0f}deg -> {after:.0f}deg")

print("implausible geometry is refused rather than guessed")
absurd = base.copy()
absurd[:, [J_LFOOT, J_RFOOT], 1] = np.linspace(0, 8, len(absurd))[:, None]  # feet up a wall
R, tilt, _ = ground_alignment(absurd)
check("falls back to identity", np.allclose(R, np.eye(3)), f"estimated tilt {tilt:.0f}deg")

print("writer handoff")
faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
path = os.path.join(TMP, "box.msq")
write_msq(path, faces, out, 30.0)
back_faces, back_verts, fps, objects = read_msq(path)
check("faces survive", np.array_equal(back_faces, faces))
check("positions survive", np.allclose(back_verts, out, atol=1e-6))
check("fps survives", fps == 30.0)
check("a file without objects reports none", objects is None)

print("object table")
scene = os.path.join(TMP, "scene.msq")
colours = [(0.0, 0.2, 0.8), (0.9, 0.9, 0.9)]
write_msq(scene, faces, out, 30.0, [(0, 1, colours[0]), (1, 1, colours[1])])
_, _, _, back = read_msq(scene)
check("objects survive", back is not None and len(back) == 2, f"got {back}")
check("spans survive", [(o[0], o[1]) for o in back] == [(0, 1), (1, 1)])
check("colours survive", np.allclose([o[2] for o in back], colours, atol=1e-6))

print("rejects bad input")
for name, fn in [
    ("out-of-range face index",
     lambda: write_msq(os.path.join(TMP, "x.msq"), np.array([[0, 1, 99]], np.int32), out, 30.0)),
    ("wrong vertex rank",
     lambda: write_msq(os.path.join(TMP, "x.msq"), faces, np.zeros((4, 3), np.float32), 30.0)),
    # A gap here is a triangle with no colour, seen far from the tool that wrote it.
    ("objects leaving a gap",
     lambda: write_msq(os.path.join(TMP, "x.msq"), faces, out, 30.0,
                       [(0, 1, (1, 1, 1))])),
    ("objects overlapping",
     lambda: write_msq(os.path.join(TMP, "x.msq"), faces, out, 30.0,
                       [(0, 2, (1, 1, 1)), (1, 1, (1, 1, 1))])),
]:
    try:
        fn()
        check(name, False, "no error raised")
    except ValueError:
        check(name, True)

print()
if failures:
    sys.exit(f"{len(failures)} failed: {', '.join(failures)}")
print("all checks passed")
