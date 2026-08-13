"""Reliability evaluation for trained drone policies.

Mean episode reward -- the metric training optimizes and reports -- is a bad proxy for
trustworthiness. A policy that flies perfectly 80% of the time and cartwheels into the ground
the other 20% can score the same average as one that is unspectacular but never crashes, and
only the second one is safe to hand a command to. This reports the things you'd actually want
to know before trusting it: how often does it arrive, how often does it crash, and when it
arrives, how close does it park?

Run as:
    uv run python evaluate.py --env combined-3d --model results/<run-dir>/best_model.zip
"""
import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from drone_sim import ENVS, OBS_TYPE, ACT_TYPE

# Episode is a success only if it survived the full duration AND finished parked on target:
# both the final position error and the average error over the last SETTLE_STEPS control steps
# must be within TOLERANCE. Requiring the settled average too (not just the final sample) means
# a drone that happens to fly through the target on the last frame doesn't count as arrived.
#
# TOLERANCE is 0.30m rather than a tighter figure on purpose. The operator's requirement is
# "never crash"; positional accuracy is explicitly secondary, and 30cm off in open space is
# fine while 3cm off into a wall is not. So station-keeping precision is graded loosely here
# and crash rate is what the report leads with -- accuracy that matters is clearance from
# things you can hit, which is an obstacle-distance question, not a setpoint-error question.
TOLERANCE = 0.30   # meters
SETTLE_STEPS = 30  # control steps (~1s at 30Hz)


def classify(env, errors):
    """Why did this episode end? Mirrors the truncation checks in the env classes."""
    state = env._getDroneStateVector(0)
    if hasattr(env, "_collided") and env._collided():
        return "collision"
    if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
        return "crash_tilt"
    max_height = getattr(env, "MAX_HEIGHT", 2.0)
    if abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > max_height + 1.0:
        return "out_of_bounds"
    # Survived to the time limit -- did it actually arrive and stay?
    settled = np.mean(errors[-SETTLE_STEPS:]) if len(errors) >= SETTLE_STEPS else np.inf
    if errors[-1] <= TOLERANCE and settled <= TOLERANCE:
        return "success"
    return "timeout_off_target"


def evaluate(env_cls, model_path, n_episodes, env_kwargs):
    model = PPO.load(model_path)
    env = env_cls(obs=OBS_TYPE, act=ACT_TYPE, **env_kwargs)

    outcomes = Counter()
    final_errors = []
    success_errors = []

    for ep in range(n_episodes):
        # Targets are drawn from the global numpy RNG inside reset(), so seeding per-episode
        # here makes every model see the identical sequence of targets -- comparisons between
        # checkpoints are then like-for-like rather than luck-of-the-draw.
        np.random.seed(10_000 + ep)
        obs, _ = env.reset()
        errors = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            state = env._getDroneStateVector(0)
            errors.append(float(np.linalg.norm(env.TARGET_POS - state[0:3])))
            if terminated or truncated:
                break
        outcome = classify(env, errors)
        outcomes[outcome] += 1
        final_errors.append(errors[-1])
        if outcome == "success":
            success_errors.append(errors[-1])

    env.close()

    print(f"\n=== {n_episodes} episodes: {model_path} ===")
    crashes = outcomes["crash_tilt"] + outcomes["out_of_bounds"] + outcomes["collision"]
    print(f"  SUCCESS (arrived & parked):  {outcomes['success'] / n_episodes:6.1%}  ({outcomes['success']})")
    print(f"  CRASHED (any cause):         {crashes / n_episodes:6.1%}  ({crashes})")
    print(f"      - excessive tilt:        {outcomes['crash_tilt'] / n_episodes:6.1%}  ({outcomes['crash_tilt']})")
    print(f"      - flew out of bounds:    {outcomes['out_of_bounds'] / n_episodes:6.1%}  ({outcomes['out_of_bounds']})")
    print(f"      - hit obstacle:          {outcomes['collision'] / n_episodes:6.1%}  ({outcomes['collision']})")
    print(f"  SURVIVED but off-target:     {outcomes['timeout_off_target'] / n_episodes:6.1%}  ({outcomes['timeout_off_target']})")
    print(f"\n  Final position error over all episodes (m):")
    fe = np.array(final_errors)
    print(f"      median {np.median(fe):.3f}   p90 {np.percentile(fe, 90):.3f}   worst {fe.max():.3f}")
    if success_errors:
        se = np.array(success_errors)
        print(f"  Final position error on successful episodes (m):")
        print(f"      median {np.median(se):.3f}   p90 {np.percentile(se, 90):.3f}   worst {se.max():.3f}")
    return outcomes


def sweep(env_cls, model_path, episodes_per_band, xy_range, bands):
    """Success rate per altitude band -- i.e. the operating envelope.

    A single aggregate success rate averages over easy and hard commands together, which hides
    exactly the thing you need to know before trusting the drone with a command: not "is it
    reliable", but "reliable *up to where*". Reporting per band gives a defensible altitude
    ceiling to either fly within or keep working on.
    """
    model = PPO.load(model_path)
    print(f"\n=== operating envelope: {model_path} ===")
    print(f"    (xy range +-{xy_range}m, {episodes_per_band} episodes per band)\n")
    print(f"    {'altitude band':<18} {'success':>9} {'crash':>8}   {'median err':>11}")
    for lo, hi in bands:
        env = env_cls(obs=OBS_TYPE, act=ACT_TYPE, xy_range=xy_range, min_height=lo, max_height=hi)
        outcomes = Counter()
        errs = []
        for ep in range(episodes_per_band):
            np.random.seed(20_000 + ep)
            obs, _ = env.reset()
            errors = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                s = env._getDroneStateVector(0)
                errors.append(float(np.linalg.norm(env.TARGET_POS - s[0:3])))
                if terminated or truncated:
                    break
            outcomes[classify(env, errors)] += 1
            errs.append(errors[-1])
        env.close()
        n = episodes_per_band
        crashes = outcomes["crash_tilt"] + outcomes["out_of_bounds"] + outcomes["collision"]
        flag = "  <-- unreliable" if outcomes["success"] / n < 0.95 else ""
        print(f"    {f'{lo:.1f} - {hi:.1f} m':<18} {outcomes['success'] / n:>8.0%} {crashes / n:>8.0%}   {np.median(errs):>10.3f}m{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure how reliably a policy arrives without crashing")
    parser.add_argument("--env", choices=list(ENVS), required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--xy-range", type=float, default=None)
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--max-height", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="report success rate per altitude band (operating envelope) instead of one aggregate number")
    args = parser.parse_args()

    env_kwargs = {}
    if args.xy_range is not None:
        env_kwargs["xy_range"] = args.xy_range
    if args.min_height is not None:
        env_kwargs["min_height"] = args.min_height
    if args.max_height is not None:
        env_kwargs["max_height"] = args.max_height

    if args.sweep:
        bands = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]
        sweep(ENVS[args.env], args.model, args.episodes, args.xy_range if args.xy_range else 1.0, bands)
    else:
        evaluate(ENVS[args.env], args.model, args.episodes, env_kwargs)
