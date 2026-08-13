# Handoff — read this first

You are picking up an RL drone-control project mid-stream. This file is written for a fresh
Claude session with no memory of the work. Read it before running anything.

**If you are a chat or Cowork session** (no repo access): read this file and the decisions log
at the bottom, and do not answer questions about measured performance from memory — those
numbers come from `evaluate.py` and are easy to guess wrong. Ask for a real run instead.

---

## What this is

A reinforcement-learning stack for quadrotor control, built on `gym-pybullet-drones` +
stable-baselines3 (PPO), plus a layered command system: spoken intent → trajectory → setpoints
→ trained policy.

**The simulated aircraft is a Crazyflie 2.X: 27 g, 12.6 cm across.** Every trained policy is
specific to it. This matters — see "Strategic finding" below.

## Files

| file | what it is |
|---|---|
| `drone_sim.py` | Training. All environments, reward shaping, curriculum flags, CLI. |
| `evaluate.py` | **Reliability measurement.** Success/crash *rates*, not mean reward. `--sweep` maps the altitude envelope. |
| `commands.py` | Layer 1/2: trajectory vocabulary (`Hover`/`Orbit`/`FollowBehind`), subject models, setpoint executor, lead compensation, operator-clearance safety measurement. |
| `make_airframe.py` | Generates a URDF for an arbitrary quadrotor (mass / wheelbase / thrust:weight). |
| `models/` | Key checkpoints. `05-cruise-BEST.zip` is the current best. |
| `test_sim.py` | Original PyBullet demo, unrelated to the RL work. |

`results/` is gitignored (18 MB of run dirs); the checkpoints worth keeping are copied into
`models/`.

## Setup

```bash
git clone https://github.com/bulletphysics/bullet3.git bullet3   # pybullet built from source
uv sync
```

## Current best model

`models/05-cruise-BEST.zip` — trained through the full curriculum ending with cruise phase 1.

```bash
# watch it follow a walking person (GUI)
uv run python commands.py --model models/05-cruise-BEST.zip --command follow-walking --lead-time 0.75 --gui

# reliability by altitude band
uv run python evaluate.py --env combined-3d --model models/05-cruise-BEST.zip --episodes 40 --xy-range 1.0 --sweep
```

### Measured performance

| capability | result |
|---|---|
| hold a point, 0.5–2.5 m | 98–100 % success, ~3 mm |
| above 2.5 m | 70–85 % — **not trustworthy** |
| follow a walker (0.25 m/s) | ~5 cm |
| orbit a walker | ~9 cm |
| cruise, all 8 directions | 8/8 at 0.5 m/s |
| cruise forward | works to ~2.5 m/s, crashes at 3.0 |
| operator clearance (follow-cam) | +0.64 to +0.96 m, never below 2.1 m altitude |

---

## Hard-won findings — do not re-derive these

1. **Mean reward is a bad metric.** It hid a policy that was flawless below 3.4 m and
   crashed 100 % of the time above it. Always judge with `evaluate.py`, which reports rates.

2. **Setpoint discontinuities are what actually crash this drone.** Three separate bugs, all
   the same family, all initially misread as policy limitations:
   - lead compensation applied as a step at t=0 (faked *every* high-speed crash)
   - "behind me" flipping 180° instantly when the subject reversed (**struck the operator**)
   - a minimum standoff that demanded a 1.7 m/s whip-around at reversals

3. **Safety penalties were dead code.** They sat inside `max(0, pos_reward - penalties)`, and
   `pos_reward ≈ 0` more than ~0.5 m from target — so during any long approach the whole group
   clamped to zero and the tilt penalty had no gradient. They now sit outside the clamp.

4. **Thrust saturation caused the altitude crashes.** Climbing to a tall target pinned all four
   motors at max, leaving ~zero differential authority for attitude; roll diverged and it
   tipped. Fixed with a headroom penalty (`SAT_THRESHOLD`). Took the 3.5–4 m band from 0 % to
   65 % success.

5. **Gentle curricula work; aggressive ones backfire.** Widening the cruise speed range twice
   (0.4–1.2, then 0.2–0.8) both produced *strictly worse* models than the 0.2–0.6 original —
   which generalises to 1.0 m/s better than models trained at 1.0 m/s. Discarded both.

6. **The drone crabs.** Yaw is pinned at 0 by the reward, so it never turns to face travel;
   "fly backward" is a separate learned skill from "fly forward", and the failures are
   left/right asymmetric. `CruiseYawAviary` (yaw-toward-travel) was training when the session
   ended — **result unknown, evaluate before trusting it.**

7. **Wind is unresolved.** Three wind models tried (piecewise, two Ornstein–Uhlenbeck variants);
   all collapsed to a flat near-zero reward. Best fallback is the wind-free model.

8. **The 23° tilt "crash" is a training threshold, not a real crash.** Some flagged crashes
   bank hard and stay airborne; others invert and hit the ground. These are different events
   and the current metric conflates them.

---

## Strategic finding — the most important thing here

**The control layer being trained is already solved on real hardware.** PX4/ArduPilot do
position hold and waypoints better and more reliably than anything trainable here, and
obstacle-avoiding follow-me ships as a product (Skydio). Meanwhile the simulated 27 g
Crazyflie cannot carry lidar, and neither can the HS175D (215 g) the owner has — a single
Livox Mid-360 is 265 g.

The layers that are *not* solved, and where this project's value lies:
language → safe constrained trajectory, semantic scene understanding, and **runtime safety
guarantees** — an independent monitor that clamps altitude, clearance, speed and acceleration
regardless of what the model asks for.

Full roadmap: https://claude.ai/code/artifact/f362577b-1e4f-4b9e-92ba-4ac88b8e52cd

**Recommended next step is Phase 0 of that roadmap: move to PX4 SITL + Gazebo and re-point
`commands.py` at MAVSDK.** That runs the real flight firmware, so nothing learned there has to
be re-learned on hardware. Continuing to tune the Crazyflie policy has low expected value.

---

## Operator safety constraints (non-negotiable)

The owner is **1.8 m tall**. `commands.py` models them as a cylinder 1.8 m tall, 0.75 m wide
and measures clearance to it.

- Follow-cam defaults: **2.5 m altitude**, 0.8–1.2 m behind direction of travel, never directly
  overhead (a drop from overhead lands on them).
- The original demo defaults — 1.5 m altitude, 0.8 m orbit radius — flew **through the
  operator's head**. Do not reintroduce them.
- Stated priority: **never crash > accuracy.** 30 cm off in open space is fine; 3 cm into a
  wall is not. Judge changes on crash rate and clearance, not tracking error.

---

## Open threads

| thread | state |
|---|---|
| `CruiseYawAviary` (yaw-toward-travel) | trained, **never evaluated** — measure before use |
| Wind robustness | unresolved after three approaches |
| Obstacle avoidance in trajectory following | `ObstacleAvoidanceAviary` exists standalone; `commands.py` flies an obstacle-free env |
| Clearance to walls/obstacles | not measured at all — only clearance to the operator |
| Crash metric | conflates "banked past 23°" with "hit the ground"; should be split |

## Working across surfaces (Claude Code / chat / Cowork)

The work is deliberately split across surfaces. Split by **what tools a task needs**, not by
subject matter:

| task | surface | why |
|---|---|---|
| run training, edit code, read logs | Claude Code | needs the repo and a shell |
| source parts, track cost, compare sensors | chat / Cowork | needs the web, not the repo |
| debug a physical build, read a diagram | chat / Cowork | needs images |
| architecture decisions | wherever the binding constraint is | cross-cutting; see below |

**The failure mode to guard against.** The most consequential findings in this project have all
been cross-domain — a physical fact arriving mid-simulation and invalidating the technical
direction. Three real examples:

- "I'm 1.8 m tall" (a build fact) revealed the demo trajectories were flying through the
  operator's head at 1.5 m.
- "My drone is an HS175D" (a procurement fact) revealed the simulated airframe was 27 g, the
  owned one 215 g, and the job needs 2–5 kg — which is what exposed that the control layer
  being trained is the layer to *buy*.
- "Nobody sprints while crab-walking" (domain reasoning) reframed an entire test matrix that
  was being optimised against the wrong requirement.

None of these would surface in a session scoped strictly to one domain. So: when a decision
depends on a constraint from the *other* side, pull that constraint in rather than deciding
without it. And when a chat session needs a performance number, get it measured — do not let it
be recalled.

**Keep this file current.** It is the only shared state between surfaces. Write decisions *and
their reasons* here, not just code changes; a fresh session can read the code but cannot
reconstruct why something was chosen.

## Decisions log

Newest last. Record the call, the reason, and what would reverse it.

| date | decision | reason | what would reverse it |
|---|---|---|---|
| 2026-08-12 | Judge models by success/crash rate, not mean reward | Mean reward hid a policy flawless below 3.4 m and crashing 100 % above it | Nothing — this was strictly an improvement |
| 2026-08-13 | Discarded both cruise speed-ramp attempts (0.4–1.2, 0.2–0.8) | Both produced strictly worse models than the 0.2–0.6 original, which generalises to 1.0 m/s better than models trained there | Evidence that a slower ramp or different LR schedule behaves differently |
| 2026-08-13 | Follow-cam geometry: 2.5 m altitude, behind travel, never overhead | Operator is 1.8 m; earlier defaults flew through their head. Overhead is the worst place to fail from | A tested runtime monitor enforcing clearance independently |
| 2026-08-13 | Priority is never-crash over tracking accuracy | Owner's stated requirement: 30 cm off in open space is fine, 3 cm into a wall is not | Changes only if the mission changes |
| 2026-08-13 | Stop investing in the RL control layer; buy it (PX4/ArduPilot) | Position control is solved and more reliable off the shelf; obstacle-avoiding follow-me ships as a product. The simulated 27 g airframe cannot carry the sensor payload anyway | Discovering PX4 cannot meet a specific requirement the RL policy can |
| 2026-08-13 | Next step is Phase 0: PX4 SITL + Gazebo, re-point `commands.py` at MAVSDK | SITL runs the real firmware, so nothing learned there needs re-learning on hardware | A decision to buy a Skydio-class platform and build on its SDK instead |

## Useful commands

```bash
# train (curriculum flags: --xy-range --min-height --max-height --min/max-cruise-speed)
uv run python drone_sim.py --env combined-3d --timesteps 3000000 --n-envs 16 \
  --eval-freq 3000 --n-eval-episodes 6 --target-reward 0 \
  --warm-start models/05-cruise-BEST.zip --learning-rate 1e-4

# reliability, and operator-clearance safety
uv run python evaluate.py --env combined-3d --model models/05-cruise-BEST.zip --episodes 200
uv run python commands.py --model models/05-cruise-BEST.zip --command orbit-walking --gui

# generate a URDF for a real payload-carrying airframe
uv run python make_airframe.py --name carrier --mass 2.5 --wheelbase 0.45 --thrust2weight 2.2
```

Notes: `--target-reward 0` disables early stopping (the inherited threshold fires on lucky
3-episode samples). `--n-envs 16` with parallel eval gives ~3500 steps/s on an 8-core M3.
