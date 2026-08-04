"""Score video clips by how well a single subject estimator will do on them.

Superseded on the main path by scan_video.py, which watches whole matches.
This still ranks a directory of clips that are already cut.

    python tools/rank_clips.py clips/ --top 10 --copy samples/

  size      how tall the largest person is, as a fraction of frame height
  framing   whether the whole body fits inside the frame, feet and head
  solo      how much the largest person dominates the next largest
  presence  the fraction of sampled frames that person appears in

Needs ultralytics and a YOLO model, both already in GEM's virtualenv.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))

# Runs under GEM's interpreter, which is where ultralytics lives.
_DETECT_SCRIPT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ultralytics import YOLO

weights, samples = sys.argv[1], int(sys.argv[2])
clips = json.loads(sys.stdin.read())
model = YOLO(weights)
out = {}

for path in clips:
    # A handful of evenly spaced frames, since framing barely moves within a cut.
    try:
        res = model(path, stream=True, vid_stride=max(1, samples), classes=[0],
                    conf=0.5, verbose=False)
        heights, ratios, contained, seen, total = [], [], [], 0, 0
        for r in res:
            total += 1
            h_img = r.orig_shape[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
            if len(boxes) == 0:
                continue
            seen += 1
            hs = boxes[:, 3] - boxes[:, 1]
            order = np.argsort(-hs)
            top = boxes[order[0]]
            heights.append(float(hs[order[0]] / h_img))
            ratios.append(float(hs[order[1]] / hs[order[0]]) if len(hs) > 1 else 0.0)
            # Margin at both ends means head and feet are inside the frame.
            contained.append(float(top[1] > 0.02 * h_img and top[3] < 0.98 * h_img))
        out[path] = {
            "size": float(np.median(heights)) if heights else 0.0,
            "second": float(np.median(ratios)) if ratios else 0.0,
            "framing": float(np.mean(contained)) if contained else 0.0,
            "presence": seen / total if total else 0.0,
        }
    except Exception as exc:
        out[path] = {"error": str(exc)[:80]}

print("@@" + json.dumps(out))
"""


def score(m):
    """One number per clip. Any weak dimension drags the whole thing down."""
    if "error" in m:
        return 0.0
    # A band, not a maximum. Too small has no pixels, too large is cropped.
    h = m["size"]
    size = min(1.0, max(0.0, (h - 0.18) / 0.15)) * min(1.0, max(0.0, (0.95 - h) / 0.15))
    # A second person nearly as tall means overlap and a confused tracker.
    solo = max(0.0, 1.0 - m["second"] / 0.8)
    # Framing counts twice, since a body cropped at the ankles has no ground contact.
    return size * solo * (m["framing"] ** 2) * m["presence"]


def detect(clips, samples):
    python = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"needs ultralytics; GEM's venv not found at {GEM_ROOT}")
    weights = os.path.join(GEM_ROOT, "yolov8x.pt")

    proc = subprocess.run([python, "-c", _DETECT_SCRIPT, weights, str(samples)],
                          input=json.dumps(clips), capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit(f"detection failed:\n{proc.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", help="directory of clips")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--copy", default=None, help="copy the winners here")
    ap.add_argument("--stride", type=int, default=15, help="detect every Nth frame")
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument("--cache", default=None, help="reuse detections from this json")
    args = ap.parse_args()

    clips = sorted(os.path.join(args.clips, f) for f in os.listdir(args.clips)
                   if f.lower().endswith(".mp4"))
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    # Detection is kept apart from scoring, so the weighting can be retuned free.
    if args.cache and os.path.exists(args.cache):
        measured = json.load(open(args.cache))
        print(f"reusing detections for {len(measured)} clips", file=sys.stderr)
    else:
        print(f"detecting in {len(clips)} clips ...", file=sys.stderr)
        measured = detect(clips, args.stride)
        if args.cache:
            json.dump(measured, open(args.cache, "w"))
    ranked = sorted(((score(m), p, m) for p, m in measured.items()), reverse=True)

    print(f"{'score':>6}  {'size':>5} {'solo':>5} {'frame':>5} {'seen':>5}  clip")
    for s, path, m in ranked[: args.top]:
        if "error" in m:
            continue
        print(f"{s:6.3f}  {m['size']:5.2f} {1 - m['second'] / 0.8:5.2f} "
              f"{m['framing']:5.2f} {m['presence']:5.2f}  {os.path.basename(path)}")

    kept = [(s, p) for s, p, _ in ranked[: args.top] if s >= args.min_score]
    print(f"\n{len(kept)} of {len(clips)} clips scored at least {args.min_score}")

    if args.copy and kept:
        os.makedirs(args.copy, exist_ok=True)
        for rank, (s, path) in enumerate(kept):
            dest = os.path.join(args.copy, f"soccer_{rank:02d}.mp4")
            shutil.copy(path, dest)
            print(f"  {os.path.basename(dest)}  <- {os.path.basename(path)}  ({s:.3f})")


if __name__ == "__main__":
    main()
