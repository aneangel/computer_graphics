"""Follow every person and the ball through a clip, one box per frame.

The detector answers each frame on its own. Deciding which box is which person,
and what happened on the frames it found nothing, is what this adds.

    python tools/track_people.py clip.mp4 --out tracks/clip.json

Writes dense per track boxes as JSON, which run_gem.py feeds to the estimator.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))

# Runs under GEM's interpreter, where ultralytics lives. Every frame, not every Nth.
_DETECT_SCRIPT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ultralytics import YOLO

weights, video, imgsz, conf = sys.argv[1:5]
model = YOLO(weights)

people, balls, width, height = [], [], 0, 0
for r in model(video, stream=True, classes=[0, 32], conf=float(conf),
               imgsz=int(imgsz), verbose=False):
    height, width = r.orig_shape
    b = r.boxes
    cls = b.cls.cpu().numpy() if b is not None else np.empty(0)
    xyxy = b.xyxy.cpu().numpy() if b is not None else np.empty((0, 4))
    score = b.conf.cpu().numpy() if b is not None else np.empty(0)
    people.append(np.column_stack([xyxy[cls == 0], score[cls == 0]]).round(2).tolist())
    balls.append(np.column_stack([xyxy[cls == 32], score[cls == 32]]).round(2).tolist())

print("@@" + json.dumps({"width": width, "height": height,
                         "people": people, "balls": balls}))
"""


def detect(video, imgsz, conf):
    """Every person and ball box in every frame. The one external step."""
    python = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"needs ultralytics; GEM's venv not found at {GEM_ROOT}")
    weights = os.path.join(GEM_ROOT, "yolov8x.pt")

    proc = subprocess.run(
        [python, "-c", _DETECT_SCRIPT, weights, os.path.abspath(video), str(imgsz), str(conf)],
        capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit(f"detection failed on {video}:\n{proc.stderr[-800:]}")


class Track:
    """One person, carried forward by a constant velocity filter."""

    _F = np.eye(6, dtype=np.float64)
    _F[0, 4] = _F[1, 5] = 1.0
    _H = np.zeros((4, 6))
    _H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0

    def __init__(self, track_id, box, frame):
        cx, cy, w, h = to_cxcywh(box)
        self.id = track_id
        self.x = np.array([cx, cy, w, h, 0.0, 0.0])
        # Position is trusted at birth, velocity is not, hence the huge variance.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e3, 1e3])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0])
        self.R = np.diag([4.0, 4.0, 16.0, 16.0])

        self.first = frame
        self.last = frame
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.boxes = {frame: np.asarray(box[:4], np.float64)}

    def predict(self):
        self.x = Track._F @ self.x
        self.P = Track._F @ self.P @ Track._F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.x[:4].copy()

    def update(self, box, frame):
        z = np.array(to_cxcywh(box))
        y = z - Track._H @ self.x
        S = Track._H @ self.P @ Track._H.T + self.R
        K = self.P @ Track._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ Track._H) @ self.P

        self.last = frame
        self.hits += 1
        self.misses = 0
        self.boxes[frame] = np.asarray(box[:4], np.float64)

    def coast(self):
        self.misses += 1


def to_cxcywh(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1.0, x2 - x1), max(1.0, y2 - y1)


def to_xyxy(state):
    cx, cy, w, h = state[:4]
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])


def iou_matrix(predicted, detections):
    """IoU of every prediction against every detection."""
    if not len(predicted) or not len(detections):
        return np.zeros((len(predicted), len(detections)))
    a = np.asarray(predicted, np.float64)[:, None, :]
    d = np.asarray(detections, np.float64)[None, :, :4]

    w = np.clip(np.minimum(a[..., 2], d[..., 2]) - np.maximum(a[..., 0], d[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], d[..., 3]) - np.maximum(a[..., 1], d[..., 1]), 0, None)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    return inter / np.maximum(area_a + area_d - inter, 1e-9)


def track_people(frames, args):
    """Follow every person across the clip. Returns the finished tracks."""
    live, done, next_id = [], [], 0

    for f, detections in enumerate(frames):
        predicted = [to_xyxy(t.predict()) for t in live]
        cost = 1.0 - iou_matrix(predicted, detections)

        matched, used_d = {}, set()
        if len(live) and len(detections):
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                # The gate does the rejecting. The solver always pairs something.
                if cost[r, c] <= 1.0 - args.min_iou:
                    matched[r] = c
                    used_d.add(c)

        for i, t in enumerate(live):
            if i in matched:
                t.update(detections[matched[i]], f)
                if t.hits >= args.min_hits:
                    t.confirmed = True
            else:
                t.coast()

        for j, d in enumerate(detections):
            if j not in used_d:
                live.append(Track(next_id, d, f))
                next_id += 1

        keep = []
        for t in live:
            if t.misses > args.max_age:
                # A track that never got confirmed was a flicker, not a person.
                if t.confirmed:
                    done.append(t)
            else:
                keep.append(t)
        live = keep

    done.extend(t for t in live if t.confirmed)
    return done


def track_ball(frames, args):
    """Follow the ball, matched by predicted position rather than overlap."""
    x = None  # [cx, cy, vx, vy]
    missed = 0
    boxes, seen = [], []

    for detections in frames:
        prediction = None if x is None else x[:2] + x[2:]

        pick = None
        if detections:
            d = np.asarray(detections, np.float64)
            centres = np.column_stack([(d[:, 0] + d[:, 2]) * 0.5, (d[:, 1] + d[:, 3]) * 0.5])
            if prediction is None:
                pick = int(np.argmax(d[:, 4]))  # nothing to predict from yet
            else:
                far = np.linalg.norm(centres - prediction, axis=1)
                near = int(np.argmin(far))
                if far[near] <= args.ball_gate:
                    pick = near

        if pick is None:
            missed += 1
            if x is not None:
                # Dropped after a few frames, or a stale prediction rejects every sighting.
                x = None if missed > args.ball_max_age else \
                    np.concatenate([x[:2] + x[2:], x[2:] * 0.9])
            boxes.append(boxes[-1] if boxes else [0.0, 0.0, 0.0, 0.0])
            seen.append(0)
            continue

        box = np.asarray(detections[pick][:4], np.float64)
        centre = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
        velocity = np.zeros(2) if x is None else centre - x[:2]
        x = np.concatenate([centre, velocity])
        missed = 0
        boxes.append(box.round(2).tolist())
        seen.append(1)

    return boxes, seen


def densify(track, frames, width, height):
    """One box for every frame, interpolating gaps and holding the ends."""
    known = sorted(track.boxes)
    stack = np.stack([track.boxes[f] for f in known])

    grid = np.arange(frames)
    dense = np.stack([np.interp(grid, known, stack[:, k]) for k in range(4)], axis=1)

    seen = np.zeros(frames, np.int8)
    seen[known] = 1
    dense[:, [0, 2]] = dense[:, [0, 2]].clip(0, width - 1)
    dense[:, [1, 3]] = dense[:, [1, 3]].clip(0, height - 1)
    return dense, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="the JSON to write")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2,
                    help="low on purpose, so occluded players are not dropped")
    ap.add_argument("--min-iou", type=float, default=0.2,
                    help="overlap with the prediction below which a match is refused")
    ap.add_argument("--min-hits", type=int, default=3,
                    help="detections in a row before a track counts as a person")
    ap.add_argument("--max-age", type=int, default=15,
                    help="frames a track may coast unseen before it is closed")
    ap.add_argument("--min-length", type=int, default=25,
                    help="drop tracks seen for fewer frames than this")
    ap.add_argument("--ball-gate", type=float, default=140.0,
                    help="pixels the ball may move in one frame and still be itself")
    ap.add_argument("--ball-max-age", type=int, default=3,
                    help="frames the ball may coast before the track is dropped")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.video)[0] + ".tracks.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    detected = detect(args.video, args.imgsz, args.conf)
    people, balls = detected["people"], detected["balls"]
    width, height, frames = detected["width"], detected["height"], len(people)
    print(f"{os.path.basename(args.video)}: {frames} frames at {width}x{height}")

    tracks = [t for t in track_people(people, args) if t.hits >= args.min_length]
    tracks.sort(key=lambda t: -t.hits)

    ball_boxes, ball_seen = track_ball(balls, args)

    out_tracks = []
    for rank, t in enumerate(tracks):
        dense, seen = densify(t, frames, width, height)
        heights = (dense[:, 3] - dense[:, 1]) / height
        # Against the track's own span, so a late arrival is not scored as missing.
        span = t.last - t.first + 1
        out_tracks.append({
            "id": rank,
            "first": int(t.first), "last": int(t.last),
            "hits": int(t.hits), "coverage": round(t.hits / span, 3),
            "height": round(float(np.median(heights[seen == 1])), 4),
            "boxes": dense.round(2).tolist(),
            "seen": seen.tolist(),
        })
        print(f"  track {rank}: frames {t.first:4d}-{t.last:<4d} "
              f"seen {t.hits:4d}/{span:<4d} ({100 * t.hits / span:3.0f}% of span)  "
              f"median height {np.median(heights[seen == 1]):.3f}")

    # Both numbers, so a sparse ball can be blamed on the right stage.
    raw = sum(1 for d in balls if d)
    print(f"  ball: detected in {raw}/{frames} frames, tracked in {sum(ball_seen)}")

    with open(out, "w") as f:
        json.dump({
            "video": os.path.abspath(args.video),
            "width": width, "height": height, "frames": frames,
            "tracks": out_tracks,
            "ball": {"boxes": ball_boxes, "seen": ball_seen},
        }, f)
    print(f"wrote {out}  ({len(out_tracks)} tracks)")


if __name__ == "__main__":
    main()
