#!/usr/bin/env bash
# One command from motion data to the 3D viewer.
#
#   tools/play.sh SOURCE [CLIP]              open the viewer
#   tools/play.sh SOURCE [CLIP] --render OUT.mp4    write a video instead
#
#   SOURCE  a Motion-X subset .zip (then CLIP names the clip inside it),
#           a single .npy / AMASS .npz, or an animated .glb
#
#   SMPLX_MODEL_PATH   where the SMPL-X models live, for the SMPL paths.
#                      Overridden by --model-path.
#
# Examples:
#   tools/play.sh CesiumMan.glb
#   tools/play.sh ~/Downloads/kungfu.zip Kung_Fu_Flying_Kick_1_clip3
#   tools/play.sh ~/Downloads/kungfu.zip Tai_Chi_Kungfu_7_clip1 --render taichi.mp4
#
# Converted sequences are cached under .cache/, so reopening a clip is instant.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RENDERER=$ROOT/build/render-pipeline
CACHE=$ROOT/.cache
MODEL_PATH=${SMPLX_MODEL_PATH:-}
RENDER_TO=""

if [ $# -lt 1 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

SOURCE=$1; shift
CLIP=""
[ $# -gt 0 ] && case "$1" in --*) ;; *) CLIP=$1; shift ;; esac

while [ $# -gt 0 ]; do
    case "$1" in
        --render)      RENDER_TO=$2; shift 2 ;;
        --model-path)  MODEL_PATH=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -x "$RENDERER" ]; then
    echo "renderer not built. run:" >&2
    echo "  cmake -S $ROOT -B $ROOT/build -DCMAKE_BUILD_TYPE=Release && cmake --build $ROOT/build" >&2
    exit 1
fi
[ -e "$SOURCE" ] || { echo "no such file: $SOURCE" >&2; exit 1; }

# Cache key: the source and clip names, so different clips coexist.
mkdir -p "$CACHE"
KEY=$(basename "$SOURCE")
KEY=${KEY//[^A-Za-z0-9._-]/_}${CLIP:+__$CLIP}
MSQ=$CACHE/$KEY.msq

# Rebuild only if the source is newer, or nothing is cached yet.
if [ ! -f "$MSQ" ] || [ "$SOURCE" -nt "$MSQ" ]; then
    case "$SOURCE" in
        *.glb|*.gltf)
            python "$ROOT/tools/gltf_to_msq.py" "$SOURCE" --out "$MSQ"
            ;;
        *.zip)
            # Motion-X ships video and motion under one name, so check the contents.
            # Matched with case, since grep -q would SIGPIPE the producer under pipefail.
            LISTING=$(unzip -l "$SOURCE" 2>/dev/null || true)
            case "$LISTING" in
                *.npy*) ;;
                *)
                    echo "$SOURCE holds no .npy motion files." >&2
                    case "$LISTING" in
                        *.mp4*)
                            echo "It looks like the video/ archive -- those are the source" >&2
                            echo "clips, not motion. You want the same subset from" >&2
                            echo "motion/motion_generation/smplx322/." >&2
                            ;;
                    esac
                    exit 1
                    ;;
            esac
            [ -n "$CLIP" ] || {
                echo "$SOURCE is a subset archive; name a clip, or browse it with:" >&2
                echo "  python $ROOT/tools/motion_to_msq.py '$SOURCE' --list" >&2
                exit 1
            }
            [ -n "$MODEL_PATH" ] || { echo "set SMPLX_MODEL_PATH or pass --model-path" >&2; exit 1; }
            python "$ROOT/tools/motion_to_msq.py" "$SOURCE" --clip "$CLIP" \
                --model-path "$MODEL_PATH" --out "$MSQ"
            ;;
        *.npy|*.npz)
            [ -n "$MODEL_PATH" ] || { echo "set SMPLX_MODEL_PATH or pass --model-path" >&2; exit 1; }
            python "$ROOT/tools/motion_to_msq.py" "$SOURCE" \
                --model-path "$MODEL_PATH" --out "$MSQ"
            ;;
        *)
            echo "don't know how to read $SOURCE (expected .glb, .zip, .npy or .npz)" >&2
            exit 1
            ;;
    esac
else
    echo "cached: $MSQ"
fi

if [ -z "$RENDER_TO" ]; then
    exec "$RENDERER" "$MSQ"
fi

# Encode at the sequence's own rate so playback matches the capture.
FRAMES=$(mktemp -d)
trap 'rm -rf "$FRAMES"' EXIT
"$RENDERER" "$MSQ" "$FRAMES"

FPS=$(python -c "
import sys; sys.path.insert(0, '$ROOT/tools')
from msq import read_msq
print(read_msq('$MSQ')[2])
")
ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%04d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$RENDER_TO"
echo "wrote $RENDER_TO"
