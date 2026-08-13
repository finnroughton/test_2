# test-project — RL drone control + layered command system

Reinforcement learning for quadrotor control (`gym-pybullet-drones` + stable-baselines3 PPO),
plus a layered system that turns a spoken command into a trajectory and flies it.

> **Continuing this work — including in a new Claude session or on another machine — start
> with [HANDOFF.md](HANDOFF.md).** It contains the findings, the traps, and the reasoning
> behind the current state. Reading it first will save you from re-deriving several things
> that took a long time to learn.

**Repo:** https://github.com/finnroughton/test_2

## Setup

```bash
git clone https://github.com/finnroughton/test_2.git
cd test_2

# PyBullet is built from source (not a wheel on all platforms)
git clone https://github.com/bulletphysics/bullet3.git bullet3

uv sync
```

## Try the current best model

```bash
# follow a walking person, with the GUI
uv run python commands.py --model models/05-cruise-BEST.zip --command follow-walking \
  --lead-time 0.75 --gui

# reliability by altitude band — success/crash rates, not mean reward
uv run python evaluate.py --env combined-3d --model models/05-cruise-BEST.zip \
  --episodes 40 --xy-range 1.0 --sweep
```

## Layout

| path | what |
|---|---|
| `drone_sim.py` | training — environments, reward shaping, curriculum flags |
| `evaluate.py` | reliability measurement (success/crash rates, altitude envelope) |
| `commands.py` | command → trajectory → setpoints, plus operator-clearance safety checks |
| `make_airframe.py` | generate a URDF for an arbitrary quadrotor |
| `models/` | key checkpoints; `05-cruise-BEST.zip` is current best |
| `gym-pybullet-drones/` | vendored, with uv/setuptools compatibility fixes |
| `bullet3/` | PyBullet source — cloned separately, gitignored |
| `results/` | training run outputs — gitignored |

## Where this is headed

The control layer trained here is already solved on real hardware by PX4/ArduPilot, and the
simulated 27 g Crazyflie cannot carry a lidar. The unsolved, valuable layers are language →
safe constrained trajectory, semantic scene understanding, and runtime safety enforcement.

Build roadmap: https://claude.ai/code/artifact/f362577b-1e4f-4b9e-92ba-4ac88b8e52cd
