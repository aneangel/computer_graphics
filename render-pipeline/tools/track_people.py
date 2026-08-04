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

# Shirt color tells two crossing players apart, the ground tells a player from a fan.
def describe(img, boxes):
    H, W = img.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)

        # The shirt, clear of the head above and the shorts below.
        a = img[int(max(0, y1 + 0.18 * h)):int(min(H, y1 + 0.50 * h)),
                int(max(0, x1 + 0.25 * w)):int(min(W, x1 + 0.75 * w))]
        px = a.reshape(-1, 3).astype(np.int16)
        # A box is wider than the man in it, so grass would skew a green shirt.
        if px.size:
            grassy = (px[:, 1] > px[:, 0] + 15) & (px[:, 1] > px[:, 2] + 15)
            px = px[~grassy] if (~grassy).sum() >= 8 else px
        color = np.median(px, axis=0)[::-1] if px.size else np.zeros(3)

        # A band just under the feet. Their own shoes would be in the way higher up.
        g = img[int(min(H - 1, y2 + 0.01 * h)):int(min(H, y2 + 0.07 * h + 3)),
                int(max(0, x1 + 0.30 * w)):int(min(W, x1 + 0.70 * w))].astype(np.int16)
        grass = float(np.mean((g[..., 1] > g[..., 0] + 15) &
                              (g[..., 1] > g[..., 2] + 15))) if g.size else 0.0

        out.append([*np.asarray(color, float), grass])
    return np.asarray(out, float).reshape(len(boxes), 4)


people, balls, width, height = [], [], 0, 0
for r in model(video, stream=True, classes=[0, 32], conf=float(conf),
               imgsz=int(imgsz), verbose=False):
    height, width = r.orig_shape
    b = r.boxes
    cls = b.cls.cpu().numpy() if b is not None else np.empty(0)
    xyxy = b.xyxy.cpu().numpy() if b is not None else np.empty((0, 4))
    score = b.conf.cpu().numpy() if b is not None else np.empty(0)
    # Trailing columns, so anything reading box[:4] or box[4] is unaffected.
    who = xyxy[cls == 0]
    people.append(np.column_stack(
        [who, score[cls == 0], describe(r.orig_img, who)]).round(2).tolist())
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
        # Kept as a running mean. One frame's shirt can be half in shadow.
        self.color = color_of(box)
        self.grass = [grass_of(box)]

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

        c = color_of(box)
        if c is not None:
            # Slow, so a frame where somebody crosses in front does not rewrite it.
            self.color = c if self.color is None else 0.9 * self.color + 0.1 * c
        self.grass.append(grass_of(box))

    def coast(self):
        self.misses += 1


def color_of(box):
    """The shirt color the detector measured, if this build wrote one."""
    return np.asarray(box[5:8], np.float64) if len(box) >= 8 else None


def grass_of(box):
    """How much of the ground under this box is pitch."""
    return float(box[8]) if len(box) >= 9 else 1.0


def chroma(rgb):
    """Color with the brightness divided out, as the two opponent axes.

    Under floodlights the same shirt varies far more in brightness than in hue.
    Clustering raw RGB sorts the players by how well lit they were, which is not
    a thing anyone is wearing.
    """
    v = np.asarray(rgb, np.float64)
    total = max(float(v.sum()), 1e-6)
    r, g, b = v / total
    return np.array([r - g, g - b])


def color_cost(tracks, detections):
    """Shirt distance, 0 to 1, between every track and every detection."""
    have = [t.color for t in tracks]
    if not len(tracks) or not len(detections) or any(c is None for c in have):
        return np.zeros((len(tracks), len(detections)))
    d = [color_of(x) for x in detections]
    if any(c is None for c in d):
        return np.zeros((len(tracks), len(detections)))

    a = np.stack([chroma(c) for c in have])[:, None, :]
    b = np.stack([chroma(c) for c in d])[None, :, :]
    # Scaled so the other side's shirt lands near 1, comparable to IoU.
    return np.clip(np.linalg.norm(a - b, axis=2) / 0.35, 0.0, 1.0)


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


def rejoin(lost, detections, taken, frame, args):
    """Match what is left over against players who went missing earlier.

    Without this a player who is blocked from view for longer than `max_age`
    comes back as a new person, and one player becomes several short tracks that
    are each too brief to estimate from.
    """
    free = [j for j in range(len(detections)) if j not in taken]
    if not lost or not free:
        return {}

    # Where each lost track would be now if it had kept going.
    predicted = np.stack([to_xyxy(t.x) for t in lost])
    centers = np.column_stack([(predicted[:, 0] + predicted[:, 2]) * 0.5,
                               (predicted[:, 1] + predicted[:, 3]) * 0.5])
    d = np.asarray([detections[j][:4] for j in free], np.float64)
    seen = np.column_stack([(d[:, 0] + d[:, 2]) * 0.5, (d[:, 1] + d[:, 3]) * 0.5])

    away = np.linalg.norm(centers[:, None, :] - seen[None, :, :], axis=2)
    # Allowed to have drifted further the longer it has been gone.
    reach = np.array([[args.reid_gate * (1.0 + t.misses / 25.0)] for t in lost])
    shirt = color_cost(lost, [detections[j] for j in free])

    cost = away / np.maximum(reach, 1e-6) + shirt / max(args.reid_color, 1e-6)
    rows, cols = linear_sum_assignment(cost)

    out = {}
    for r, c in zip(rows, cols):
        # Both halves must hold: near enough to be reachable, and the right shirt.
        if away[r, c] <= reach[r, 0] and shirt[r, c] <= args.reid_color:
            out[r] = free[c]
    return out


def track_people(frames, args):
    """Follow every person across the clip. Returns the finished tracks."""
    live, lost, done, next_id = [], [], [], 0

    for f, detections in enumerate(frames):
        predicted = [to_xyxy(t.predict()) for t in live]
        overlap = iou_matrix(predicted, detections)
        # Shape only says which pairings are possible; the shirt says which is right.
        cost = (1.0 - overlap) + args.color_weight * color_cost(live, detections)
        cost[overlap < args.min_iou] = 1e6

        matched, used_d = {}, set()
        if len(live) and len(detections):
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                # The gate does the rejecting. The solver always pairs something.
                if cost[r, c] < 1e6:
                    matched[r] = c
                    used_d.add(c)

        for i, t in enumerate(live):
            if i in matched:
                t.update(detections[matched[i]], f)
                if t.hits >= args.min_hits:
                    t.confirmed = True
            else:
                t.coast()

        for t in lost:
            t.predict()
            t.coast()
        for i, j in rejoin(lost, detections, used_d, f, args).items():
            lost[i].update(detections[j], f)
            lost[i].misses = 0
            live.append(lost[i])
            used_d.add(j)
        lost = [t for t in lost if t.misses <= args.reid_age and t not in live]

        for j, d in enumerate(detections):
            if j not in used_d:
                live.append(Track(next_id, d, f))
                next_id += 1

        keep = []
        for t in live:
            if t.misses > args.max_age:
                # A track that never got confirmed was a flicker, not a person.
                if t.confirmed:
                    lost.append(t)
            else:
                keep.append(t)
        live = keep

    done.extend(t for t in live + lost if t.confirmed)
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
            centers = np.column_stack([(d[:, 0] + d[:, 2]) * 0.5, (d[:, 1] + d[:, 3]) * 0.5])
            if prediction is None:
                pick = int(np.argmax(d[:, 4]))  # nothing to predict from yet
            else:
                far = np.linalg.norm(centers - prediction, axis=1)
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
        center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
        velocity = np.zeros(2) if x is None else center - x[:2]
        x = np.concatenate([center, velocity])
        missed = 0
        boxes.append(box.round(2).tolist())
        seen.append(1)

    return boxes, seen


def densify(track, frames, width, height):
    """One box for every frame, interpolating gaps and holding the ends.

    Center and height are interpolated, and the width is rebuilt from the shape
    this person usually is. Sliding the four corners independently lets a box
    turn into a sliver, and a body fitted to a sliver comes back the wrong size
    in the wrong place. Half a player is still a whole player's shape.
    """
    known = sorted(track.boxes)
    stack = np.stack([track.boxes[f] for f in known])

    cx = (stack[:, 0] + stack[:, 2]) * 0.5
    cy = (stack[:, 1] + stack[:, 3]) * 0.5
    w = np.maximum(stack[:, 2] - stack[:, 0], 1.0)
    h = np.maximum(stack[:, 3] - stack[:, 1], 1.0)

    # Taken from the frames where the detector saw enough of them to be believed.
    aspect = float(np.median(w / h))

    grid = np.arange(frames)
    mid_x = np.interp(grid, known, cx)
    mid_y = np.interp(grid, known, cy)
    tall = np.interp(grid, known, h)
    wide = np.interp(grid, known, w)

    seen = np.zeros(frames, np.int8)
    seen[known] = 1
    # Only where nothing was seen. An observed box, however odd, is a measurement.
    wide = np.where(seen == 1, wide, tall * aspect)

    dense = np.stack([mid_x - wide * 0.5, mid_y - tall * 0.5,
                      mid_x + wide * 0.5, mid_y + tall * 0.5], axis=1)
    dense[:, [0, 2]] = dense[:, [0, 2]].clip(0, width - 1)
    dense[:, [1, 3]] = dense[:, [1, 3]].clip(0, height - 1)
    return dense, seen


def assign_teams(tracks, teams):
    """Group the tracks by shirt, biggest group first.

    A match has two of nearly everything and one of a few things. Clustering the
    shirts finds the two crowds; what is left over is the referee, the keepers,
    and whoever is standing on the touchline in a coat.
    """
    from scipy.cluster.vq import kmeans2

    usable = [t for t in tracks if t.color is not None]
    if len(usable) < teams:
        return {id(t): 0 for t in tracks}

    book = np.stack([chroma(t.color) for t in usable])
    # Fixed seed, since a report that reorders between runs cannot be compared.
    centers, label = kmeans2(book, teams, minit="++", seed=0)

    # Renumbered so cluster 0 is the largest, which is one of the two sides.
    order = np.argsort(-np.bincount(label, minlength=teams))
    rank = {int(old): new for new, old in enumerate(order)}
    return {id(t): rank[int(label[i])] for i, t in enumerate(usable)}


def on_pitch(track, floor):
    """Whether this one stood on grass, over the frames it was actually seen."""
    return float(np.median(track.grass)) >= floor if track.grass else True


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
    ap.add_argument("--color-weight", type=float, default=0.6,
                    help="how much the shirt counts against overlap when two "
                         "players cross and the boxes alone are ambiguous")
    ap.add_argument("--reid-age", type=int, default=90,
                    help="frames a vanished player is still remembered for")
    ap.add_argument("--reid-gate", type=float, default=90.0,
                    help="pixels from where a vanished player was predicted to be")
    ap.add_argument("--reid-color", type=float, default=0.16,
                    help="shirt distance, 0 to 1, allowed when rejoining a track")
    ap.add_argument("--players-only", action="store_true",
                    help="keep only those on the pitch: both sides, the keepers "
                         "and the officials, but not the crowd or the touchline")
    ap.add_argument("--teams", type=int, default=5,
                    help="shirts to cluster into: two sides, two keepers, officials")
    ap.add_argument("--keep-teams", type=int, default=0,
                    help="with --players-only, keep only this many of the biggest "
                         "kits; 0 keeps every kit that is on the pitch")
    ap.add_argument("--min-grass", type=float, default=0.25,
                    help="with --players-only, pitch under the feet to count as on it")
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

    team = assign_teams(tracks, args.teams)
    if args.players_only:
        before = len(tracks)
        # Keepers and officials wear neither kit, so being on the pitch is the test.
        kept = [t for t in tracks
                if on_pitch(t, args.min_grass)
                and (not args.keep_teams or team.get(id(t), 0) < args.keep_teams)]
        off = sum(1 for t in tracks if not on_pitch(t, args.min_grass))
        print(f"  on the pitch: {before} tracks -> {len(kept)} "
              f"({off} off it, {before - len(kept) - off} in an excluded kit)")
        tracks = kept

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
            "team": int(team.get(id(t), 0)),
            "shirt": [round(float(v)) for v in t.color] if t.color is not None else None,
            "grass": round(float(np.median(t.grass)), 3) if t.grass else None,
        })
        print(f"  track {rank}: frames {t.first:4d}-{t.last:<4d} "
              f"seen {t.hits:4d}/{span:<4d} ({100 * t.hits / span:3.0f}% of span)  "
              f"median height {np.median(heights[seen == 1]):.3f}  "
              f"kit {team.get(id(t), 0)}")

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
