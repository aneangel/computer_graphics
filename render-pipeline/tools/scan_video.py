"""Sweep a match once and write down where the shots are and what is in them.

Detection is the expensive half, so it runs once here and everything downstream
reads the manifest.

    python tools/scan_video.py ~/Downloads/archive --out scans/
    python tools/scan_video.py scans/ --report --top 20

Cuts come from two signals. The bodies say a cut happened, since they jump to an
unrelated arrangement. The scene score says which frame it happened on.

Needs ultralytics and a YOLO model, both already in GEM's virtualenv.
"""

import argparse
import json
import os
import re
import subprocess
import sys

GEM_ROOT = os.path.expanduser(os.environ.get("GEM_ROOT", "~/GENMO"))
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# metadata=print puts the timestamp and the score on consecutive lines.
SCENE_RE = re.compile(r"pts_time:([0-9.]+)\s+lavfi\.scene_score=([0-9.]+)")

# Runs under GEM's interpreter. Only raw measurements come back, so retuning is free.
_DETECT_SCRIPT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ultralytics import YOLO

weights, video, stride, imgsz, conf = sys.argv[1:6]
model = YOLO(weights)

# Class 0 is person, 32 is sports ball. Low confidence keeps occluded players.
results = model(video, stream=True, vid_stride=int(stride), classes=[0, 32],
                conf=float(conf), imgsz=int(imgsz), verbose=False)

samples, width, height = [], 0, 0
for i, r in enumerate(results):
    height, width = r.orig_shape
    boxes = r.boxes
    cls = boxes.cls.cpu().numpy() if boxes is not None else np.empty(0)
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4))
    people, balls = xyxy[cls == 0], xyxy[cls == 32]

    if len(people):
        people = people[np.argsort(-(people[:, 3] - people[:, 1]))]  # tallest first

    # Every height, since the count is a measurement. Boxes are capped, they only feed cuts.
    heights = [round(float(v) / height, 4) for v in (people[:, 3] - people[:, 1])[:64]]

    # Grass separates a pitch shot from the crowd or a graphic. orig_img is BGR.
    px = r.orig_img[::8, ::8].astype(np.int16)
    green = float(np.mean((px[..., 1] > px[..., 0] + 15) & (px[..., 1] > px[..., 2] + 15)))

    samples.append({
        "n": int(len(people)),
        "h": heights,
        "p": [[round(float(v), 1) for v in b] for b in people[:12]],
        "b": [[round(float(v), 1) for v in b] for b in balls[:2]],
        "g": round(green, 3),
    })

print("@@" + json.dumps({"width": width, "height": height, "samples": samples}))
"""


def probe(video):
    """Frame rate and duration, which turn sample indices into timestamps."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate:format=duration", "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True).stdout.split()
    if len(out) < 2:
        raise SystemExit(f"cannot probe {video}")
    num, _, den = out[0].partition("/")
    return float(num) / float(den or 1), float(out[1])


def scene_scores(video):
    """Per frame change score as (time, score), thresholded later not here."""
    proc = subprocess.run(
        ["ffmpeg", "-i", video, "-filter:v", "select='gt(scene,0)',metadata=print:file=-",
         "-f", "null", "-"], capture_output=True, text=True)
    return [(float(t), float(s)) for t, s in SCENE_RE.findall(proc.stdout)]


def detect(video, stride, imgsz, conf):
    """Run the detector over every `stride`-th frame. Returns the raw sweep."""
    python = os.path.join(GEM_ROOT, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"needs ultralytics; GEM's venv not found at {GEM_ROOT}")
    weights = os.path.join(GEM_ROOT, "yolov8x.pt")

    proc = subprocess.run(
        [python, "-c", _DETECT_SCRIPT, weights, os.path.abspath(video),
         str(stride), str(imgsz), str(conf)], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit(f"detection failed on {video}:\n{proc.stderr[-800:]}")


def displacement(a, b, width, height):
    """Median nearest center distance between two samples, as a frame fraction."""
    if not a or not b:
        return 1.0 if (a or b) else 0.0

    def centers(boxes):
        return [((x[0] + x[2]) * 0.5 / width, (x[1] + x[3]) * 0.5 / height) for x in boxes]

    ca, cb = centers(a), centers(b)

    def nearest(one, many):
        out = []
        for x in one:
            out.append(min(((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5 for y in many))
        return out

    both = sorted(nearest(ca, cb) + nearest(cb, ca))
    return both[len(both) // 2]


def cut_times(scores, samples, fps, stride, args, width, height, gap=0.4):
    """Timestamps where the camera cut, plus a tally of which signal found each."""
    marks = [(t, "scene") for t, s in scores if s > args.threshold]

    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        moved = displacement(a["p"], b["p"], width, height) > args.max_shift
        # Grass entering or leaving covers the case with nobody in shot to measure.
        recolored = abs(a["g"] - b["g"]) > args.green_jump
        if not (moved or recolored):
            continue
        # The boxes give the stride, the peak scene score inside it gives the frame.
        lo, hi = (i - 1) * stride / fps, i * stride / fps
        window = [(s, t) for t, s in scores if lo <= t <= hi]
        marks.append((max(window)[1] if window else hi, "boxes"))

    # A dissolve scores high for several frames, and both signals may find the same cut.
    merged, sources = [], []
    for t, source in sorted(marks):
        if merged and t - merged[-1] <= gap:
            # Already have it. Recording the agreement is how the thresholds get judged.
            if sources[-1] != source:
                sources[-1] = "both"
            continue
        merged.append(t)
        sources.append(source)

    tally = {k: sources.count(k) for k in ("scene", "boxes", "both")}
    return merged, tally


def summarize(window, width, height, min_height=0.06):
    """Reduce the samples inside one shot to the numbers that decide its fate."""
    import statistics

    def median(values, default=0.0):
        return statistics.median(values) if values else default

    counts, medians, tallest, framed, greens, balls = [], [], [], [], [], []
    for s in window:
        # Below this it is a spectator, and counting them populates every wide shot.
        tall = [v for v in s["h"] if v >= min_height]
        counts.append(len(tall))
        greens.append(s["g"])
        balls.append(1.0 if s["b"] else 0.0)
        if not tall:
            continue
        medians.append(median(tall))
        tallest.append(max(tall))
        # Only the tallest boxes are stored, and only those can reach a frame edge.
        boxes = [b for b in s["p"] if (b[3] - b[1]) / height >= min_height]
        if boxes:
            inside = [b for b in boxes if b[1] > 0.02 * height and b[3] < 0.98 * height]
            framed.append(len(inside) / len(boxes))

    # Spread of the tallest body across the shot, recorded but not scored on.
    spread = (max(tallest) - min(tallest)) if len(tallest) > 1 else 0.0

    return {
        "people": median(counts),
        "height": median(medians),
        "tallest": median(tallest),
        "spread": round(spread, 3),
        "contained": median(framed),
        "green": median(greens),
        "ball": sum(balls) / len(balls) if balls else 0.0,
        # A second body wandering through for a moment is not a shot of two players.
        "presence": sum(1 for c in counts if c >= 2) / len(counts) if counts else 0.0,
        "samples": len(window),
    }


def classify(m):
    """Which of the handful of shots broadcast football is made of."""
    if m["people"] < 1:
        return "empty"
    # Low on purpose, since a good low camera shot can be mostly stands behind.
    if m["green"] < 0.18:
        return "crowd"  # stands, dugout or a graphic: no pitch in view
    if m["tallest"] > 0.75:
        return "close"  # cropped at the waist, so there is no ground contact
    if m["tallest"] < 0.15:
        return "wide"  # the tactical camera, forty pixels a player
    return "medium"


def score(m):
    """One number per shot. Any weak dimension drags the whole thing down."""
    if m["kind"] != "medium":
        return 0.0

    # A band, not a maximum, read off the median player rather than the tallest.
    h = m["height"]
    size = min(1.0, max(0.0, (h - 0.12) / 0.06)) * min(1.0, max(0.0, (0.70 - h) / 0.15))

    # Several bodies is the point now. Past about six they occlude each other.
    n = m["people"]
    crowd = min(1.0, max(0.0, n - 1.0)) * min(1.0, max(0.0, (9.0 - n) / 3.0))

    # Framing counts twice, since a cropped body has no ground contact.
    return size * crowd * (m["contained"] ** 2) * m["presence"]


def analyze(data, args):
    """Cut the sweep into shots and judge them, without touching video."""
    samples, width, height = data["samples"], data["width"], data["height"]
    fps, total, stride = data["fps"], data["duration"], data["stride"]
    scores = [(t, s) for t, s in data["scene"]]

    cuts, tally = cut_times(scores, samples, fps, stride, args, width, height)

    shots = []
    for start, stop in zip([0.0] + cuts, cuts + [total]):
        # Trimmed at each end, where a dissolve or a camera whip often sits.
        start, stop = start + 0.2, stop - 0.2
        if stop - start < args.min:
            continue
        lo = int(start * fps / stride)
        hi = max(lo + 1, int(stop * fps / stride))
        m = summarize(samples[lo:hi], width, height)
        if not m["samples"]:
            continue
        m["kind"] = classify(m)
        m["score"] = round(score(m), 4)
        m["start"], m["end"] = round(start, 2), round(stop, 2)
        shots.append(m)

    data["cuts"], data["shots"] = tally, shots
    return data


def sweep(video, args):
    """Read one match. The expensive half, and the only half that touches video."""
    fps, total = probe(video)
    detected = detect(video, args.stride, args.imgsz, args.conf)
    if not detected["samples"]:
        raise SystemExit(f"no frames read from {video}")

    return {
        "video": os.path.abspath(video),
        "fps": fps, "duration": round(total, 2),
        "width": detected["width"], "height": detected["height"],
        "stride": args.stride, "imgsz": args.imgsz, "conf": args.conf,
        # Kept whole and unthresholded, so the cache survives any retuning.
        "scene": [[t, s] for t, s in scene_scores(video)],
        "samples": detected["samples"],
    }


def report(scans, top):
    """Print the best shots across every manifest, best first."""
    rows = []
    for path in scans:
        data = json.load(open(path))
        for s in data["shots"]:
            rows.append((s["score"], os.path.basename(data["video"]), s))
    rows.sort(key=lambda r: -r[0])

    kinds = {}
    for _, _, s in rows:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print("shots by kind: " + "  ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"\n{'score':>6} {'people':>6} {'size':>5} {'frame':>5} {'ball':>5} "
          f"{'span':>13}  clip")
    for s_score, name, s in rows[:top]:
        print(f"{s_score:6.3f} {s['people']:6.1f} {s['height']:5.2f} "
              f"{s['contained']:5.2f} {s['ball']:5.2f} "
              f"{s['start']:6.1f}-{s['end']:6.1f}  {name}")
    print(f"\n{sum(1 for r in rows if r[0] > 0):d} usable of {len(rows)} shots")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="a video, a directory of them, or a scan directory")
    ap.add_argument("--out", default="scans", help="where the manifests go")
    ap.add_argument("--report", action="store_true", help="print scanned manifests and exit")
    ap.add_argument("--rescore", action="store_true",
                    help="recut and rescore existing manifests without touching video")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5, help="detect every Nth frame")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="detector input size, wasteful below the source width")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="scene score counting as a cut, well under the usual 0.3")
    ap.add_argument("--max-shift", type=float, default=0.30,
                    help="frame fraction people may move between samples, play sits near 0.02")
    ap.add_argument("--green-jump", type=float, default=0.15,
                    help="change in the fraction of grass in frame counting as a cut")
    ap.add_argument("--min", type=float, default=2.0, help="drop shots shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many videos")
    ap.add_argument("--force", action="store_true", help="rescan videos already scanned")
    args = ap.parse_args()

    if args.report or args.rescore:
        root = args.input if os.path.isdir(args.input) else args.out
        scans = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".json"))
        if not scans:
            raise SystemExit(f"no manifests in {root}")
        if args.rescore:
            for path in scans:
                with open(path) as f:
                    data = json.load(f)
                with open(path, "w") as f:
                    json.dump(analyze(data, args), f)
            print(f"recut and rescored {len(scans)} manifests", file=sys.stderr)
        report(scans, args.top)
        return

    if os.path.isdir(args.input):
        videos = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                        if f.lower().endswith(VIDEO_EXTENSIONS))
    else:
        videos = [args.input]
    if not videos:
        raise SystemExit(f"no videos in {args.input}")
    if args.limit:
        videos = videos[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    for video in videos:
        stem = os.path.splitext(os.path.basename(video))[0]
        path = os.path.join(args.out, stem + ".json")
        if os.path.exists(path) and not args.force:
            print(f"{stem:<12} cached", file=sys.stderr)
            continue

        data = analyze(sweep(video, args), args)
        with open(path, "w") as f:
            json.dump(data, f)

        usable = [s for s in data["shots"] if s["score"] > 0]
        kinds = {}
        for s in data["shots"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        c = data["cuts"]
        print(f"{stem:<12} {data['duration']:6.0f}s  "
              f"cuts {c['scene'] + c['boxes'] + c['both']:3d} "
              f"(scene {c['scene']}, boxes {c['boxes']}, both {c['both']})  "
              f"{len(data['shots']):3d} shots  "
              + "  ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
              + f"  -> {len(usable)} usable")


if __name__ == "__main__":
    main()
