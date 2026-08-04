"""Estimate motion for every tracked person in a clip, not just the biggest one.

GEM recovers one person per run and keeps only the largest track. It caches
those boxes to preprocess/bbx.pt and reads them back, so writing that file first
points it at whoever we like. Nothing in GEM is modified.

    python tools/track_people.py clip.mp4 --out tracks/clip.json
    python tools/run_gem.py tracks/clip.json --out gem/clip --max-actors 4

The clip is trimmed to the window where the chosen actors are all present, since
a scene has a fixed vertex count and nobody in it can simply stop existing.

GEM is under the NVIDIA OneWay Noncommercial License. See docs/INSTALL.md in the
clone for setup.
"""

import argparse
import json
import os
import subprocess
import sys

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


def choose_actors(tracks, max_actors, min_frames):
    """Pick actors that coexist, and the window over which they all do.

    Greedy on quality, keeping each only while the shared window stays usable.
    Picking the best first and intersecting after collapses to nothing.
    """
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
    """Cut frames first to last inclusive, counted rather than timed.

    Frame numbers line up with the tracks. Timestamps drift by a frame and put
    every box on the wrong picture.
    """
    if os.path.exists(out):
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
         "-vf", f"select='between(n\\,{first}\\,{last})',setpts=N/FRAME_RATE/TB",
         "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out],
        check=True)
    return out


def write_bbx(boxes, path):
    """Convert dense boxes to the centre and size file GEM reads back."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tracks", help="a track_people.py JSON")
    ap.add_argument("--out", default=None, help="where per-actor results go")
    ap.add_argument("--max-actors", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=45,
                    help="shortest shared window worth estimating over")
    ap.add_argument("-s", "--static-cam", action="store_true",
                    help="the camera does not move, which makes the trajectory easier")
    args = ap.parse_args()

    with open(args.tracks) as f:
        data = json.load(f)
    video = data["video"]
    out_root = args.out or os.path.splitext(args.tracks)[0] + ".gem"
    os.makedirs(out_root, exist_ok=True)

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
