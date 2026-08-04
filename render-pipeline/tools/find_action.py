"""Find the windows inside a scan where the football is actually being played.

scan_video.py judges whether a shot is *filmed* well: right body size, nobody
cropped, enough players. A wall standing still scores the same as a counter
attack. This reads the same manifests and asks the other question, whether
anyone is moving and whether the ball is there to move with.

    python tools/find_action.py scans/ --top 20
    python tools/find_action.py scans/ --top 6 --cut clips/

Nothing here opens a video. The player and ball boxes were already written down
every `stride` frames, and motion is the difference between consecutive ones.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_video import classify, displacement, summarize  # noqa: E402


def center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def ball_flow(window, width, height):
    """How far the ball travels between samples, when it is on screen at all."""
    moved = []
    for a, b in zip(window, window[1:]):
        if not a["b"] or not b["b"]:
            continue
        ca, cb = center(a["b"][0]), center(b["b"][0])
        moved.append((((ca[0] - cb[0]) / width) ** 2 + ((ca[1] - cb[1]) / height) ** 2) ** 0.5)
    return statistics.median(moved) if moved else 0.0


def motion(window, width, height):
    """The per-sample body flow through one window, and what could spoil it."""
    flow = [displacement(a["p"], b["p"], width, height)
            for a, b in zip(window, window[1:])]
    if not flow:
        return None

    # displacement returns 1.0 when one side is empty, reading as violent motion.
    torn = max(flow) > 0.30

    return {
        "flow": round(statistics.median(flow), 4),
        # The busiest moment in the window: a sprint or a shot, not the average.
        "surge": round(sorted(flow)[int(0.9 * (len(flow) - 1))], 4),
        "ball": round(sum(1.0 for s in window if s["b"]) / len(window), 3),
        "ball_flow": round(ball_flow(window, width, height), 4),
        "torn": torn,
    }


def framing_score(m):
    """How well the window is filmed, on the bands broadcast football lands in.

    scan_video.score answers this too, but its bands were set for a few players
    filling the frame. The main camera here gives a median body 0.10 of frame
    height and fifteen of them, which that score reads as unusable.
    """
    # A band, not a maximum: under 0.06 a player is 40 pixels, over 0.6 they crop.
    h = m["height"]
    size = min(1.0, max(0.0, (h - 0.055) / 0.035)) * min(1.0, max(0.0, (0.60 - h) / 0.20))

    # Several bodies is the point. Past twenty they occlude each other.
    n = m["people"]
    crowd = min(1.0, max(0.0, n - 1.0)) * min(1.0, max(0.0, (26.0 - n) / 6.0))

    # Counts twice, since a body cropped at the ankles has no ground contact.
    return size * crowd * (m["contained"] ** 2)


def action_score(m):
    """One number per window. Any weak dimension drags the whole thing down."""
    if m["torn"]:
        return 0.0

    # A band: below this they stand about, above it the camera is whipping.
    f = m["flow"]
    pace = min(1.0, max(0.0, (f - 0.008) / 0.010)) * min(1.0, max(0.0, (0.075 - f) / 0.025))

    # A ball on screen is what separates play from a restart or a replay caption.
    ball = 0.35 + 0.65 * m["ball"]

    # Gentle: a burst on steady pace is a break, a burst alone is one bad sample.
    surge = min(1.0, 0.7 + m["surge"] / 0.08)

    return pace * ball * surge


def windows(data, length, step, min_people):
    """Every `length`-second window inside a shot, scored on both halves."""
    width, height = data["width"], data["height"]
    fps, stride = data["fps"], data["stride"]
    per_sample = stride / fps

    out = []
    for shot in data["shots"]:
        if shot["end"] - shot["start"] < length:
            continue
        span = int(round(length / per_sample))
        hop = max(1, int(round(step / per_sample)))

        lo = int(shot["start"] * fps / stride)
        hi = int(shot["end"] * fps / stride)
        for i in range(lo, hi - span + 1, hop):
            window = data["samples"][i:i + span]
            if len(window) < span:
                break

            m = summarize(window, width, height)
            if not m["samples"] or m["people"] < min_people:
                continue
            m["kind"] = classify(m)

            moved = motion(window, width, height)
            if moved is None:
                continue
            m.update(moved)

            m["framing"] = round(framing_score(m), 4)
            m["action"] = round(action_score(m), 4)
            # Both halves must hold: a still frame and a blur are each useless.
            m["score"] = round(m["framing"] * m["action"], 4)
            m["start"] = round(i * per_sample, 2)
            m["end"] = round(m["start"] + length, 2)
            out.append(m)
    return out


def best_per_video(rows, apart):
    """Drop windows that overlap a better one, so one shot cannot fill the table."""
    kept = []
    for row in sorted(rows, key=lambda r: -r[0]):
        s, video, m = row
        if any(v == video and abs(m["start"] - k["start"]) < apart for _, v, k in kept):
            continue
        kept.append(row)
    return kept


def cut(video, start, length, path):
    """Write one window out, re-encoded since a stream copy only cuts on keyframes."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                    "-i", video, "-t", f"{length:.2f}", "-an", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", path], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scans", help="a directory of scan_video.py manifests")
    ap.add_argument("--length", type=float, default=10.0, help="window length in seconds")
    ap.add_argument("--step", type=float, default=1.0, help="how far the window slides")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-people", type=float, default=2.0)
    ap.add_argument("--apart", type=float, default=10.0,
                    help="keep only the best window within this many seconds")
    ap.add_argument("--cut", default=None, help="write the winners here as mp4")
    args = ap.parse_args()

    files = sorted(os.path.join(args.scans, f) for f in os.listdir(args.scans)
                   if f.endswith(".json"))
    if not files:
        raise SystemExit(f"no manifests in {args.scans}")

    rows = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for m in windows(data, args.length, args.step, args.min_people):
            rows.append((m["score"], data["video"], m))

    kept = best_per_video(rows, args.apart)[: args.top]
    if not kept:
        raise SystemExit("no window scored above zero; loosen --min-people or --length")

    print(f"{'score':>6} {'act':>5} {'frm':>5} {'flow':>6} {'surge':>6} {'ball':>5} "
          f"{'ppl':>4} {'size':>5} {'kind':>6}  clip")
    for s, video, m in kept:
        print(f"{s:6.3f} {m['action']:5.2f} {m['framing']:5.2f} {m['flow']:6.4f} "
              f"{m['surge']:6.4f} {m['ball']:5.2f} {m['people']:4.1f} {m['height']:5.2f} "
              f"{m['kind']:>6}  {os.path.basename(video)} {m['start']:.1f}-{m['end']:.1f}")

    print(f"\n{len(kept)} shown of {sum(1 for r in rows if r[0] > 0)} windows scoring above zero")

    if args.cut:
        os.makedirs(args.cut, exist_ok=True)
        for rank, (s, video, m) in enumerate(kept):
            stem = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(args.cut, f"{rank:02d}_{stem}_{m['start']:.0f}.mp4")
            cut(video, m["start"], args.length, path)
            print(f"  {os.path.basename(path):<28} {s:.3f}")


if __name__ == "__main__":
    main()
