#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/plannersim"
RUNS="${RUNS:-15}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-30}"
RESULTS_FILE="$SCRIPT_DIR/results.csv"

PARTICLE_COUNTS="500 3000 7500 12500 20000 25000"

TOTAL=0
DONE=0

if [[ -f "$RESULTS_FILE" ]]; then
    HEADER="$(head -n 1 "$RESULTS_FILE")"
    if [[ "$HEADER" != *"run_time_s"* || "$HEADER" != *"timed_out"* ]]; then
        TS="$(date +%Y%m%d_%H%M%S)"
        LEGACY="$SCRIPT_DIR/results_legacy_$TS.csv"
        mv "$RESULTS_FILE" "$LEGACY"
        echo "Archived legacy schema results to $LEGACY"
    fi
fi

for n in $PARTICLE_COUNTS; do
    for scenario in 1 2 3; do
        for alg in 0 1; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

for n in $PARTICLE_COUNTS; do
    for scenario in 1 2 3; do
        for alg in 0 1; do
            DONE=$((DONE + 1))
            ALG_NAME="A*"
            [ "$alg" -eq 1 ] && ALG_NAME="RRT*"
            case $scenario in
                1) SCENE_NAME="particles_only" ;;
                2) SCENE_NAME="static_obstacles" ;;
                3) SCENE_NAME="dynamic_obstacles" ;;
            esac
            echo "[$DONE/$TOTAL] particles=$n  scenario=$SCENE_NAME  algorithm=$ALG_NAME  runs=$RUNS"
            "$BINARY" "$n" "$scenario" "$alg" "$RUNS" --headless --max-run-seconds "$MAX_RUN_SECONDS"
        done
    done
done

echo "Benchmark complete. Results in $RESULTS_FILE"
