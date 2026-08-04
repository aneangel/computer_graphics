"""Turn a video into a scene: everybody in it, and the ball, ready to render.

    python tools/video_to_scene.py clip.mp4 --out clip.msq
    ./build/render-pipeline clip.mp4

    track_people.py     who is in it, and where, on every frame
    run_gem.py          what each of them was doing, one estimator run each
    scene_from_video.py all of it in one frame, with the ball, as a sequence

Each stage is skipped if its output already exists, so a failure halfway through
costs only the stage that failed. The estimator runs once per person, which is
what --max-actors is for.
"""

import argparse
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))


def stage(name, script, arguments, produces):
    """Run one tool, unless what it makes is already on disk."""
    if produces and os.path.exists(produces):
        print(f"[{name}] cached: {produces}")
        return produces

    print(f"[{name}] {' '.join(str(a) for a in arguments)}")
    result = subprocess.run([sys.executable, os.path.join(TOOLS, script)] +
                            [str(a) for a in arguments])
    if result.returncode != 0 or (produces and not os.path.exists(produces)):
        raise SystemExit(f"[{name}] failed")
    return produces


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="the .msq to write")
    ap.add_argument("--work", default=None, help="where the intermediate files go")
    ap.add_argument("--max-actors", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=45)
    ap.add_argument("--model-path", default=os.environ.get("SMPLX_MODEL_PATH", "~/models"))
    ap.add_argument("-s", "--static-cam", action="store_true")
    ap.add_argument("--no-ball", action="store_true")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out = args.out or os.path.splitext(args.video)[0] + ".msq"
    work = args.work or os.path.join(os.path.dirname(os.path.abspath(out)), stem + ".work")
    os.makedirs(work, exist_ok=True)

    tracks = os.path.join(work, "tracks.json")
    stage("track", "track_people.py", [args.video, "--out", tracks], tracks)

    gem = os.path.join(work, "gem")
    actors = os.path.join(gem, "actors.json")
    stage("estimate", "run_gem.py",
          [tracks, "--out", gem, "--max-actors", args.max_actors,
           "--min-frames", args.min_frames] + (["--static-cam"] if args.static_cam else []),
          actors)

    stage("assemble", "scene_from_video.py",
          [actors, "--out", out, "--model-path", args.model_path]
          + (["--no-ball"] if args.no_ball else []),
          out)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
