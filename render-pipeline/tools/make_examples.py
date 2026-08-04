"""Build one video showing each example: the footage, then what came out of it.

Every entry plays twice, source then reconstruction, with the camera circling
so the result can be seen from more than one side.

    python tools/make_examples.py --out examples.mp4

Models have no footage and only get the orbit. They are here so a change that
breaks them shows up in the same file.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msq import read_msq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH, HEIGHT, FPS = 1280, 720, 30

# name, source footage, what the renderer opens
EXAMPLES = [
    ("Four players and a ball", "samples/soccer_four_players.mp4",
     ".cache/soccer_four_players.msq"),
    ("Four players, wider", "samples/soccer_four_wide.mp4", ".cache/soccer_four_wide.msq"),
    ("Keeper diving", "samples/soccer_keeper_dive.mp4", ".cache/soccer_keeper_dive.msq"),
    ("Running", "samples/soccer_player_run.mp4", ".cache/soccer_player_run.msq"),
    ("Flying kick", "samples/kungfu_kick.mp4", ".cache/kungfu_kick.msq"),
    ("Stanford Dragon", None, "~/Downloads/xyzrgb_dragon.obj"),
    ("Happy Buddha", None, "~/Downloads/happy.obj"),
]


def label(text, corner):
    """A drawtext filter, with the characters ffmpeg reads as syntax escaped."""
    safe = text.replace("\\", "").replace(":", "\\:").replace("'", "")
    x = "40" if corner == "left" else f"w-tw-40"
    return (f"drawtext=fontfile={FONT}:text='{safe}':fontsize=30:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=14:x={x}:y=h-th-40")


def encode(inputs, out, filters, rate=None):
    """One ffmpeg run, always to the same codec so the parts can be joined."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if rate:
        cmd += ["-framerate", str(rate)]
    cmd += inputs
    cmd += ["-vf", ",".join(filters), "-r", str(FPS), "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", out]
    subprocess.run(cmd, check=True)


def source_clip(video, name, out, seconds):
    """The footage, letterboxed to the render's shape so nothing jumps."""
    encode(["-t", str(seconds), "-i", video], out, [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
        label(name, "left"), label("source", "right"),
    ])


def render_clip(scene, name, tag, out, work, turns, samples, seconds):
    """What the renderer draws, with the camera going once around it."""
    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)

    subprocess.run([os.path.join(ROOT, "build", "render-pipeline"),
                    os.path.expanduser(scene), frames, str(turns), str(samples)],
                   check=True, stdout=subprocess.DEVNULL)

    # Sequences play at their own rate. A model has none, so it takes the output rate.
    rate = FPS
    if scene.endswith(".msq"):
        rate = read_msq(os.path.expanduser(scene))[2]

    filters = [label(name, "left"), label(tag, "right")]
    encode(["-i", os.path.join(frames, "frame_%04d.ppm")], out, filters, rate=rate)

    if seconds:
        # Trimmed after the fact, so the orbit still completes a full turn.
        trimmed = out + ".trim.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-t", str(seconds),
                        "-c", "copy", trimmed], check=True)
        os.replace(trimmed, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="examples.mp4")
    ap.add_argument("--turns", type=float, default=1.0, help="times round the subject")
    ap.add_argument("--samples", type=int, default=24, help="rays per pixel per frame")
    ap.add_argument("--seconds", type=float, default=6.0, help="longest either half may run")
    ap.add_argument("--keep", action="store_true", help="leave the working files behind")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="examples.")
    parts = []
    try:
        for i, (name, video, scene) in enumerate(EXAMPLES):
            path = os.path.expanduser(scene)
            if not os.path.exists(path):
                print(f"  skipping {name}: no {scene}")
                continue

            recovered = bool(video) and os.path.exists(os.path.join(ROOT, video))
            if recovered:
                clip = os.path.join(work, f"{i:02d}a.mp4")
                print(f"  {name}: footage")
                source_clip(os.path.join(ROOT, video), name, clip, args.seconds)
                parts.append(clip)

            clip = os.path.join(work, f"{i:02d}b.mp4")
            print(f"  {name}: rendering")
            # A model was loaded, not recovered, so it is not called a reconstruction.
            render_clip(scene, name, "reconstruction" if recovered else "model", clip,
                        os.path.join(work, f"{i:02d}"), args.turns, args.samples, args.seconds)
            parts.append(clip)

        if not parts:
            raise SystemExit("nothing to show; run the pipeline on something first")

        listing = os.path.join(work, "parts.txt")
        with open(listing, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        # Joined without encoding again, safe because every part used the same encode.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listing, "-c", "copy", args.out], check=True)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    seconds = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         args.out], capture_output=True, text=True).stdout.strip()
    print(f"\nwrote {args.out}  ({len(parts)} parts, {float(seconds or 0):.0f}s)")


if __name__ == "__main__":
    main()
