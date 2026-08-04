"""Cut a long video into short segments at its shot boundaries.

An estimator reading across a camera cut sees a person teleport, so this only
ever cuts where the footage already does.

    python tools/clip_video.py match.mp4 --out-dir clips/

Cuts found from the picture alone are weak on grass. With --scan they come from
scan_video.py, which watched the people instead.

    python tools/clip_video.py --scan scans/ --out-dir clips/ --limit 40
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCENE_RE = re.compile(r"pts_time:([0-9.]+)")


def duration(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def shot_boundaries(video, threshold):
    """Timestamps where the picture changes enough to be a camera cut."""
    proc = subprocess.run(
        ["ffmpeg", "-i", video, "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"], capture_output=True, text=True)
    return sorted({float(t) for t in SCENE_RE.findall(proc.stderr)})


def segments(video, threshold, minimum, maximum):
    """Runs of continuous footage, split down to at most `maximum` seconds."""
    total = duration(video)
    marks = [0.0] + shot_boundaries(video, threshold) + [total]

    out = []
    for start, stop in zip(marks, marks[1:]):
        # Trimmed at each end, where a dissolve or a camera whip often sits.
        start, stop = start + 0.2, stop - 0.2
        length = stop - start
        if length < minimum:
            continue
        pieces = max(1, int(length // maximum) + (1 if length % maximum >= minimum else 0))
        step = length / pieces
        out.extend((start + i * step, step) for i in range(pieces))
    return out


def cut(video, start, length, path, fps):
    """Write one segment, unless it is already there."""
    if os.path.exists(path):
        return
    # Encoded again, since a stream copy can only cut on keyframes.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
           "-i", video, "-t", f"{length:.2f}", "-an"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


def extract(video, out_dir, spans, fps, limit):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    written = []
    for i, (start, length) in enumerate(spans):
        if limit and len(written) >= limit:
            break
        path = os.path.join(out_dir, f"{stem}_{i:03d}.mp4")
        cut(video, start, length, path, fps)
        written.append(path)
    return written


def scanned(scan_dir, min_score):
    """Shots a scan already found and judged, best first."""
    if not os.path.isdir(scan_dir):
        raise SystemExit(f"{scan_dir} is not a directory of manifests")
    out = []
    for name in sorted(os.listdir(scan_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(scan_dir, name)) as f:
            data = json.load(f)
        for i, s in enumerate(data["shots"]):
            if s["score"] >= min_score:
                out.append((s["score"], data["video"], i, s["start"], s["end"] - s["start"]))
    out.sort(reverse=True)
    if not out:
        raise SystemExit(f"no shot in {scan_dir} scored {min_score} or better")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", nargs="?", help="a video, or a directory of them")
    ap.add_argument("--scan", help="a directory of scan_video.py manifests to cut from, "
                                   "instead of finding the shots here")
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="with --scan, the lowest shot score worth cutting")
    ap.add_argument("--out-dir", default="clips")
    ap.add_argument("--min", type=float, default=2.0, help="drop segments shorter than this")
    ap.add_argument("--max", type=float, default=6.0, help="split segments longer than this")
    ap.add_argument("--threshold", type=float, default=0.3, help="scene change sensitivity, 0 to 1")
    ap.add_argument("--fps", type=float, default=None, help="resample, otherwise keep the source rate")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many clips, 0 for all")
    args = ap.parse_args()

    if args.scan:
        shots = scanned(args.scan, args.min_score)
        os.makedirs(args.out_dir, exist_ok=True)
        written = []
        for score, video, index, start, length in shots:
            if args.limit and len(written) >= args.limit:
                break
            stem = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(args.out_dir, f"{stem}_{index:03d}.mp4")
            cut(video, start, length, path, args.fps)
            written.append(path)
            print(f"  {os.path.basename(path):<20} {length:5.1f}s  score {score:.3f}")
        print(f"\n{len(written)} clips in {args.out_dir}, "
              f"{sum(s[4] for s in shots[:len(written)]) / 60:.1f} minutes")
        return

    if not args.input:
        ap.error("give a video or a directory, or --scan a directory of manifests")

    if os.path.isdir(args.input):
        videos = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    else:
        videos = [args.input]
    if not videos:
        raise SystemExit(f"no videos in {args.input}")

    total = []
    for video in videos:
        if args.limit and len(total) >= args.limit:
            break
        spans = segments(video, args.threshold, args.min, args.max)
        remaining = args.limit - len(total) if args.limit else 0
        written = extract(video, args.out_dir, spans, args.fps, remaining)
        total.extend(written)
        print(f"{os.path.basename(video):<14} {duration(video):6.0f}s "
              f"-> {len(spans):3d} shots, wrote {len(written)}")

    seconds = sum(duration(p) for p in total)
    print(f"\n{len(total)} clips in {args.out_dir}, {seconds / 60:.1f} minutes")


if __name__ == "__main__":
    main()
