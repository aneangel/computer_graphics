# How to Run the Demonstration

## Overview

This project contains three simulations built around a 3D particle/fluid system:

- **`physicssim`** — Basic SPH fluid simulation. Particles interact via smoothed-particle hydrodynamics inside a bounded box.
- **`dynamicpaths`** — Fluid simulation with dynamic pathfinding overlaid.
- **`plannersim`** — Full robot path-planning simulation. A robot particle navigates through a fluid environment toward a goal using either **A\*** or **RRT\*** path planning, with optional static or dynamic obstacles. Metrics are recorded to `results.csv` for analysis.

---

## Building the Simulations

To build all simulations, run:

```sh
make
```
or
```sh
make all
```

This will generate three executables:
- `physicssim`
- `dynamicpaths`
- `plannersim` (shell launcher) + `plannersim.bin` (CUDA-enabled binary)

To build them individually:

```sh
make physicssim
make dynamicpaths
make plannersim
```

To clean up all object files and executables:

```sh
make clean
```

---

## plannersim

`plannersim` simulates a robot navigating through a 3D particle fluid toward a goal. It uses either A* or RRT* to plan a path through the particle density field, replanning as conditions change. Each completed run logs metrics (time to goal, path length, success, etc.) to `results.csv`.

### Usage

```sh
./plannersim [particles] [scenario] [algorithm] [runs] [--headless]
```

Optional flag for fair benchmarking with explicit failed runs:

```sh
./plannersim ... --max-run-seconds <seconds>
```

| Argument      | Description                                                                 | Default |
|---------------|-----------------------------------------------------------------------------|---------|
| `particles`   | Number of fluid particles                                                   | 1500    |
| `scenario`    | `1` = Particles only, `2` = Static obstacles, `3` = Dynamic obstacles       | 3       |
| `algorithm`   | `0` = A\*, `1` = RRT\*                                                      | 0       |
| `runs`        | Number of runs to complete before exiting (0 = run indefinitely)            | 0       |
| `--headless`  | Run without a window (for benchmarking); can appear anywhere in the args    | off     |
| `--max-run-seconds` | Max simulated seconds allowed per run before logging timeout failure | off     |

**Examples:**

```sh
# Interactive: 1500 particles, dynamic obstacles, A*
./plannersim

# Interactive: 3000 particles, static obstacles, RRT*
./plannersim 3000 2 1

# Headless benchmark: 5 runs, particles only, A*
./plannersim 1000 1 0 5 --headless
```

### Interactive Controls

| Key / Input         | Action                                      |
|---------------------|---------------------------------------------|
| **Space**           | Agitate particles (apply random impulses)   |
| **1**               | Switch to Scenario 1 (Particles Only)       |
| **2**               | Switch to Scenario 2 (Static Obstacles)     |
| **3**               | Switch to Scenario 3 (Dynamic Obstacles)    |
| **P**               | Toggle path planner between A\* and RRT\*   |
| **R**               | Reset the simulation                        |
| **W / S**           | Tilt camera up / down                       |
| **A / D**           | Rotate camera left / right                  |
| **+ / -**           | Zoom in / out                               |
| **Q / ESC**         | Quit                                        |
| **Left-click drag** | Orbit camera                                |

### Benchmarking

The included `benchmark.sh` script sweeps over particle counts (500–25000), all three scenarios, and both algorithms, and runs a fixed number of attempts per condition with an explicit per-run timeout so failed runs are logged (reduces survivorship bias):

```sh
./benchmark.sh
```

Environment overrides:

```sh
RUNS=30 MAX_RUN_SECONDS=180 ./benchmark.sh
```

After benchmarking, use the analysis script to generate plots and summary statistics:

```sh
python analyse.py
```

`results.csv` includes both conditional and unconditional fields, including:
`success`, `time_to_goal_s` (success-only), `run_time_s` (all runs), and `timed_out`.

---

## Other Simulations

### physicssim

Basic SPH fluid simulation with no pathfinding. Press **Space** to agitate particles.

### dynamicpaths

SPH fluid simulation with dynamic path overlays. Press **Space** to agitate particles.