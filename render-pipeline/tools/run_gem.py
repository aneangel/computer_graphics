"""Estimate motion for every tracked person in a clip, not just the biggest one.

GEM keeps only the largest track, but it caches those boxes and reads them back,
so writing that file first points it at whoever we like. Nothing in GEM changes.

    python tools/track_people.py clip.mp4 --out tracks/clip.json
    python tools/run_gem.py tracks/clip.json --out gem/clip --max-actors 4

The clip is trimmed to the window where the chosen actors are all present, since
a scene has a fixed vertex count and nobody in it can stop existing.

GEM is under the NVIDIA OneWay Noncommercial License.
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))

# Uses GEM's own conversion, so a change to how it frames a crop cannot silently break this.
_WRITE_BBX = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
from gem.utils.geo_transform import get_bbx_xys_from_xyxy

boxes = torch.tensor(json.loads(sys.stdin.read()), dtype=torch.float32)
torch.save(get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2), sys.argv[2])
print(f"wrote {tuple(get_bbx_xys_from_xyxy(boxes).shape)} boxes")
"""


# Runs under GEM's interpreter, every actor at once, models loaded only once.
_ESTIMATE_ALL = r"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

root, batch, hmr2_ckpt, gem_ckpt, static = sys.argv[1:6]
batch, static = int(batch), static == "1"
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts", "demo"))
os.chdir(root)

from demo_smpl_hpe import assemble_data
from demo_utils import CocoPoseExtractor, get_camera_static, load_model
from gem.utils.hmr2_extractor import HMR2FeatureExtractor, _get_batch
from gem.utils.video_io_utils import get_video_lwh, read_video_np

jobs = json.loads(sys.stdin.read())

t = time.time()
pose = CocoPoseExtractor(device="cuda")
hmr2 = HMR2FeatureExtractor(hmr2_ckpt, device="cuda")
model = load_model(gem_ckpt)
print(f"@@loaded {time.time() - t:.1f}", flush=True)

for job in jobs:
    if os.path.exists(job["params"]):
        print(f"@@done {job['track']} cached", flush=True)
        continue

    t = time.time()
    clip = job["clip"]
    L, W, H = get_video_lwh(clip)
    bbx = torch.load(job["bbx"], map_location="cpu")

    # Full precision: these heatmaps go through numpy, which has no bfloat16.
    frames = read_video_np(clip)
    kp2d = pose.extract(frames, bbx, batch_size=batch)
    del frames

    imgs, _ = _get_batch(clip, bbx)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(imgs), batch):
            out.append(hmr2.model({"img": imgs[i:i + batch].cuda()}).float().cpu())
    f_imgseq = torch.cat(out).clone()

    data = assemble_data(
        kp2d=kp2d, bbx_xys=bbx,
        **dict(zip(("R_w2c", "cam_angvel", "cam_tvel", "K_fullimg"),
                   get_camera_static(L, W, H))),
        f_imgseq=f_imgseq,
        has_img_mask=torch.ones(f_imgseq.shape[0], dtype=torch.bool),
        static_cam=static)

    with torch.no_grad():
        pred = model.predict(data, static_cam=static)

    # The same file the demo writes, so nothing downstream can tell the difference.
    saved = {}
    for group in ("body_params_incam", "body_params_global"):
        if group in pred:
            saved[group] = {k: v.cpu() for k, v in pred[group].items()}
    if "K_fullimg" in pred:
        saved["K_fullimg"] = pred["K_fullimg"].cpu()

    os.makedirs(os.path.dirname(job["params"]), exist_ok=True)
    torch.save(saved, job["params"])
    print(f"@@done {job['track']} {time.time() - t:.1f}", flush=True)
"""


# Kept for --no-precompute, which runs GEM's own command line tool per actor.
_FEATURES_SCRIPT = r"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

root = sys.argv[1]
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts", "demo"))
os.chdir(root)

from gem.utils.hmr2_extractor import HMR2FeatureExtractor, _get_batch
from gem.utils.video_io_utils import read_video_np
from demo_utils import CocoPoseExtractor

jobs = json.loads(sys.stdin.read())
batch = int(sys.argv[2])
hmr2_ckpt = sys.argv[3]

# The whole point. Loading these costs ten seconds and was being paid per actor.
t = time.time()
pose = CocoPoseExtractor(device="cuda")
hmr2 = HMR2FeatureExtractor(hmr2_ckpt, device="cuda")
print(f"@@loaded {time.time() - t:.1f}", flush=True)

for job in jobs:
    want_kp = not os.path.exists(job["vitpose"])
    want_ft = not os.path.exists(job["features"])
    if not (want_kp or want_ft):
        continue

    bbx = torch.load(job["bbx"], map_location="cpu")
    t = time.time()

    if want_kp:
        frames = read_video_np(job["clip"])
        # Full precision: these heatmaps go through numpy, which has no bfloat16.
        kp2d = pose.extract(frames, bbx, batch_size=batch)
        torch.save(kp2d.float(), job["vitpose"])
        del frames

    if want_ft:
        imgs, _ = _get_batch(job["clip"], bbx)
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(imgs), batch):
                out.append(hmr2.model({"img": imgs[i:i + batch].cuda()}).float().cpu())
        # GEM reads these as float32, so the cast back happens here not there.
        torch.save(torch.cat(out).clone(), job["features"])

    print(f"@@done {job['track']} {time.time() - t:.1f}", flush=True)
"""


def estimate_all(ready, args):
    """Every actor through one process, with the three models loaded once.

    Returns the parameter file for each, or None if the run failed and the
    per-actor path should be used instead.
    """
    jobs = []
    for t, clip, root in ready:
        stem = os.path.splitext(os.path.basename(clip))[0]
        jobs.append({"track": t["id"], "clip": os.path.abspath(clip),
                     "bbx": os.path.join(root, stem, "preprocess", "bbx.pt"),
                     "params": os.path.join(root, stem, "smpl_params.pt")})

    hmr2 = os.path.join(GEM_ROOT, "inputs", "checkpoints", "hmr2",
                        "epoch=10-step=25000.ckpt")
    gem = os.path.join(GEM_ROOT, "inputs", "pretrained", "gem_smpl.ckpt")
    if not (os.path.exists(hmr2) and os.path.exists(gem)):
        print("  checkpoints not where expected; falling back to one run each")
        return None

    # Sharded: a few processes, several actors each, to overlap and share the load.
    shards = max(1, min(args.jobs, len(jobs)))
    groups = [jobs[i::shards] for i in range(shards)]
    groups = [g for g in groups if g]

    print(f"  estimating {len(jobs)} actors over {len(groups)} processes")

    def run(group):
        return subprocess.run(
            [gem_python(), "-c", _ESTIMATE_ALL, GEM_ROOT, str(args.batch), hmr2, gem,
             "1" if args.static_cam else "0"],
            input=json.dumps(group), capture_output=True, text=True)

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        runs = list(pool.map(run, groups))

    proc = runs[-1]
    for done in runs:
        for line in done.stdout.splitlines():
            if line.startswith("@@"):
                print(f"    {line[2:]}")

    missing = [j for j in jobs if not os.path.exists(j["params"])]
    if missing:
        why = next((r.stderr for r in runs if r.returncode), proc.stderr)
        print(f"  {len(missing)} of {len(jobs)} did not finish, so each will be "
              f"run on its own:\n{why[-700:]}")
        return None
    return [j["params"] for j in jobs]


def precompute(ready, args):
    """Run ViTPose and HMR2 for every actor in one process, models loaded once.

    GEM caches both of these beside the boxes, and skips the stage when it finds
    them. It has no way to share the weights between actors though: each one is
    a separate run, and each spends ten seconds loading the same two backbones
    before it looks at a frame. Doing them all here pays that once.
    """
    jobs = []
    for t, clip, root in ready:
        stem = os.path.splitext(os.path.basename(clip))[0]
        pre = os.path.join(root, stem, "preprocess")
        jobs.append({"track": t["id"], "clip": os.path.abspath(clip),
                     "bbx": os.path.join(pre, "bbx.pt"),
                     "vitpose": os.path.join(pre, "vitpose.pt"),
                     "features": os.path.join(pre, "vit_features.pt")})

    hmr2 = os.path.join(GEM_ROOT, "inputs", "checkpoints", "hmr2",
                        "epoch=10-step=25000.ckpt")
    if not os.path.exists(hmr2):
        print("  no HMR2 checkpoint; leaving the features to GEM")
        return

    print(f"  precomputing keypoints and features for {len(jobs)} actors")
    proc = subprocess.run(
        [gem_python(), "-c", _FEATURES_SCRIPT, GEM_ROOT, str(args.batch), hmr2],
        input=json.dumps(jobs), capture_output=True, text=True)

    marks = [line[2:] for line in proc.stdout.splitlines() if line.startswith("@@")]
    for line in marks:
        print(f"    {line}")

    # Count completions: loading and then failing still prints one mark.
    finished = sum(1 for line in marks if line.startswith("done"))
    if finished < len(jobs):
        print(f"  precomputed {finished} of {len(jobs)}; GEM will redo the rest:\n"
              f"{proc.stderr[-600:]}")


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


def quality(track):
    """How much use a track will be, as one number. Height matters most."""
    span = track["last"] - track["first"] + 1
    size = min(1.0, max(0.0, (track["height"] - 0.15) / 0.10))
    return size * (track["coverage"] ** 2) * min(1.0, span / 60.0)


def choose_alone(tracks, max_actors, min_frames, min_coverage):
    """Pick actors on their own merit, each over the frames it is actually there.

    A scene needs everyone present at once. An overlay does not, and insisting on
    it throws away most of the clip: five actors in a 300 frame shot shared only
    46 frames. Here each actor is estimated over its own span instead.

    Coverage is what keeps that honest. A track carried mostly by interpolation
    reaches further but is guessing where the player was, and the estimator has
    no way to tell a guessed crop from a seen one.
    """
    ranked = [t for t in sorted(tracks, key=quality, reverse=True)
              if t["last"] - t["first"] + 1 >= min_frames and t["coverage"] >= min_coverage]
    return ranked[:max_actors]


def choose_actors(tracks, max_actors, min_frames):
    """Pick actors that coexist, greedily, and the window over which they all do."""
    ranked = sorted(tracks, key=quality, reverse=True)

    chosen, first, last = [], None, None
    for t in ranked:
        if len(chosen) >= max_actors:
            break
        lo = t["first"] if first is None else max(first, t["first"])
        hi = t["last"] if last is None else min(last, t["last"])
        if hi - lo + 1 < min_frames:
            continue
        chosen, first, last = chosen + [t], lo, hi

    return chosen, first, last


def trim(video, out, first, last):
    """Cut frames first to last inclusive, counted so they line up with the tracks."""
    if os.path.exists(out):
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
         "-vf", f"select='between(n\\,{first}\\,{last})',setpts=N/FRAME_RATE/TB",
         "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out],
        check=True)
    return out


def write_bbx(boxes, path):
    """Convert dense boxes to the center and size file GEM reads back."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    proc = subprocess.run([gem_python(), "-c", _WRITE_BBX, GEM_ROOT, path],
                          input=json.dumps(boxes), capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(path):
        raise SystemExit(f"could not write {path}:\n{proc.stderr[-800:]}")


def estimate(video, output_root, static_cam):
    """Run GEM against boxes already placed in its cache. Returns the params."""
    stem = os.path.splitext(os.path.basename(video))[0]
    saved = os.path.join(output_root, stem, "smpl_params.pt")
    if os.path.exists(saved):
        print(f"    reusing {saved}")
        return saved

    cmd = [gem_python(), "scripts/demo/demo_smpl_hpe.py",
           "--video", os.path.abspath(video),
           "--output_root", os.path.abspath(output_root),
           "--ckpt_path", "inputs/pretrained/gem_smpl.ckpt",
           "--no_render"]
    if static_cam:
        cmd.append("--static_cam")

    proc = subprocess.run(cmd, cwd=GEM_ROOT, capture_output=True, text=True)
    if not os.path.exists(saved):
        raise SystemExit(f"GEM wrote no parameters at {saved}:\n{proc.stdout[-1500:]}")
    return saved


def independent(data, video, out_root, args):
    """Estimate each actor over its own frames and write the overlay manifest."""
    chosen = choose_alone(data["tracks"], args.max_actors, args.min_frames,
                          args.min_coverage)
    if not chosen:
        raise SystemExit(f"no track in {os.path.basename(video)} lasts "
                         f"{args.min_frames} frames and is seen on "
                         f"{args.min_coverage:.0%} of them; try a longer clip")

    print(f"{os.path.basename(video)}: {len(chosen)} actors, each on its own frames")

    def prepare(t):
        """Everything before the estimator: one clip and one box file per actor."""
        first, last = t["first"], t["last"]
        root = os.path.join(out_root, f"actor_{t['id']:02d}")
        # Each actor gets its own trimmed clip, so the directory has to exist first.
        os.makedirs(root, exist_ok=True)
        clip = trim(video, os.path.join(root, "window.mp4"), first, last)
        stem = os.path.splitext(os.path.basename(clip))[0]
        write_bbx(t["boxes"][first:last + 1],
                  os.path.join(root, stem, "preprocess", "bbx.pt"))
        print(f"  actor {t['id']}: frames {first}-{last} "
              f"({last - first + 1}), height {t['height']:.3f}")
        return t, clip, root

    ready = [prepare(t) for t in chosen]

    # One process for everybody, three models loaded once. Per actor is the fallback.
    done = None if args.no_precompute else estimate_all(ready, args)

    if done is None:
        # Half of a run leaves the GPU idle, so several fill each other's gaps.
        if args.jobs > 1:
            print(f"  estimating {args.jobs} at a time")
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                done = list(pool.map(
                    lambda r: estimate(r[1], r[2], args.static_cam), ready))
        else:
            done = [estimate(clip, root, args.static_cam) for _, clip, root in ready]

    actors = [{
        "track": t["id"], "height": t["height"], "coverage": t["coverage"],
        # Carried through so an overlay can put the two sides in two colors.
        "team": t.get("team", 0),
        # Where this actor's first estimated frame sits in the source clip.
        "first": t["first"], "last": t["last"],
        "params": params,
    } for (t, _, _), params in zip(ready, done)]

    manifest = os.path.join(out_root, "actors.json")
    with open(manifest, "w") as f:
        json.dump({
            "video": video, "clip": os.path.abspath(video),
            "independent": True, "frames": data["frames"],
            "width": data["width"], "height": data["height"],
            "actors": actors, "ball": data["ball"],
        }, f)
    print(f"wrote {manifest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tracks", help="a track_people.py JSON")
    ap.add_argument("--out", default=None, help="where per-actor results go")
    ap.add_argument("--max-actors", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=45,
                    help="shortest shared window worth estimating over")
    ap.add_argument("-s", "--static-cam", action="store_true",
                    help="the camera does not move, which makes the trajectory easier")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="estimator runs at once; each wants about 7 GB and leaves "
                         "the GPU idle half the time, which the others fill. Four "
                         "of them fit a 32 GB card and run 1.8x faster than one")
    ap.add_argument("--batch", type=int, default=64,
                    help="frames per forward pass when precomputing; 16 was set "
                         "for smaller cards and leaves this one mostly idle")
    ap.add_argument("--no-precompute", action="store_true",
                    help="let each GEM run extract its own keypoints and features, "
                         "reloading both backbones every time")
    ap.add_argument("--min-coverage", type=float, default=0.7,
                    help="fraction of its own frames a track must actually have "
                         "been seen on, rather than interpolated across")
    ap.add_argument("-i", "--independent", action="store_true",
                    help="estimate each actor over its own frames, for overlaying "
                         "rather than for building a scene")
    args = ap.parse_args()

    with open(args.tracks) as f:
        data = json.load(f)
    video = data["video"]
    out_root = args.out or os.path.splitext(args.tracks)[0] + ".gem"
    os.makedirs(out_root, exist_ok=True)

    if args.independent:
        independent(data, video, out_root, args)
        return

    chosen, first, last = choose_actors(data["tracks"], args.max_actors, args.min_frames)
    if not chosen:
        raise SystemExit(
            f"no {args.min_frames}-frame window in {os.path.basename(video)} holds two "
            "actors at once; try a longer clip or --min-frames")

    frames = last - first + 1
    print(f"{os.path.basename(video)}: {len(chosen)} actors over frames "
          f"{first}-{last} ({frames} of {data['frames']})")
    for t in chosen:
        print(f"  track {t['id']}: height {t['height']:.3f}  coverage {t['coverage']:.2f}  "
              f"quality {quality(t):.3f}")

    clip = trim(video, os.path.join(out_root, "window.mp4"), first, last)

    actors = []
    for t in chosen:
        root = os.path.join(out_root, f"actor_{t['id']:02d}")
        stem = os.path.splitext(os.path.basename(clip))[0]
        print(f"  actor {t['id']}: estimating over {frames} frames")
        write_bbx(t["boxes"][first:last + 1],
                  os.path.join(root, stem, "preprocess", "bbx.pt"))
        actors.append({
            "track": t["id"],
            "height": t["height"],
            "coverage": t["coverage"],
            "params": estimate(clip, root, args.static_cam),
        })

    manifest = os.path.join(out_root, "actors.json")
    with open(manifest, "w") as f:
        json.dump({
            "video": video, "clip": os.path.abspath(clip),
            "first": first, "last": last, "frames": frames,
            "width": data["width"], "height": data["height"],
            "actors": actors,
            # Sliced to the same window, so ball and bodies share an index.
            "ball": {
                "boxes": data["ball"]["boxes"][first:last + 1],
                "seen": data["ball"]["seen"][first:last + 1],
            },
        }, f)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
