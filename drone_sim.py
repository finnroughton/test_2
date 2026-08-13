"""Train a PPO agent to hover a quadcopter using gym-pybullet-drones' HoverAviary.

Train:
    uv run python drone_sim.py --env windy --timesteps 2000000 --warm-start results/<run-dir>/best_model.zip

Watch a trained model fly (GUI):
    uv run python drone_sim.py --env windy --replay results/<run-dir>/best_model.zip
"""
import argparse
import time
from datetime import datetime
import os

import gymnasium as gym
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync

OBS_TYPE = ObservationType.KIN
ACT_TYPE = ActionType.RPM


class PrecisionHoverAviary(HoverAviary):
    """HoverAviary with a reward that sharply punishes drift, demands stillness, and holds heading.

    HoverAviary's original reward, max(0, 2 - err**4), is deceptively forgiving: at err=0.5m
    it's still 1.85/2 (93%), so PPO happily settles half a meter off target and calls it a day.

    A first attempt tightened this with a hard-clipped quadratic (2 - 15*err**2, clipped at 0),
    but that clips to exactly zero anywhere past ~0.37m -- and the drone starts on the ground,
    ~1m from the target, so it got zero reward signal for the entire climb and PPO collapsed to
    a "give up immediately" policy (reward stuck at 0.00 for 500k+ steps). A smooth Gaussian-
    shaped term instead decays continuously with no dead zone: still near-max next to the
    target, sharply lower by ~0.3-0.5m (real precision pressure), but never literally zero, so
    there's always a gradient to climb even from the starting position.

    Position/velocity/angular-velocity alone still let yaw drift freely (nothing constrains
    heading), so a tight-position hover could end up 30+ degrees off heading -- fine in open
    air, but that's wasted control margin in tight quarters or gusty wind. A yaw penalty pulls
    heading back toward 0. Using (1 - cos(yaw)) instead of yaw**2 avoids the +-pi wraparound
    discontinuity.

    Under wind (see WindyPrecisionHoverAviary), this policy learned to settle ~0.3-0.4m below
    the target altitude and just hold there rather than fight back up -- not because the wind
    (capped at ~11% of the drone's weight, against a 2.25 thrust-to-weight ratio) is actually
    strong enough to prevent recovery, but because POS_REWARD_STEEPNESS=4.0 still scored a
    0.34m error at ~63% of max reward: a comfortable local optimum, not a capability limit.
    Steepening to 8.0 (~40% at that error), then 15.0 (~14%) plus an entropy bonus, still
    didn't move it after a combined 2.5M+ fine-tuning steps -- reward stayed flat. The problem:
    all of this only ever scores *how close* the drone is, never *whether it's closing the
    gap*. A step that starts climbing back and a step that's given up both look almost
    identical to a purely position-based reward if neither has reached the target yet, so
    there was no direct signal that the recovery *behavior itself* is good, only a distant
    payoff for actually arriving. PROGRESS_WEIGHT adds dense potential-based shaping: a direct
    per-step bonus for reducing distance to target (and penalty for increasing it), on top of
    the existing terms, so "closing the gap" is reinforced immediately rather than only once
    it's already close.

    On RandomHeightHoverAviary (target height randomized 0.5-4m/episode), that same progress
    bonus backfired: with PROGRESS_WEIGHT=10 and a weak ANG_PENALTY_WEIGHT=0.02, sustaining a
    fast climb toward a distant target paid off every step regardless of the cost of the
    angular velocity it was accumulating, and reward kept improving right up until the roll
    oscillation it was building diverged and crossed the 0.4 rad tilt-truncation limit --
    observed directly in a replay: roll went -7.6d -> -11.4d -> -15.1d -> -21.2d -> -26.5d
    (crash) over 6 steps while angular velocity grew -0.83 -> -0.49 -> -0.81 -> -2.13 -> -3.38
    rad/s, sustaining ~1.2 m/s climb the whole time. Tripling ANG_PENALTY_WEIGHT and adding
    TILT_PENALTY_WEIGHT (on roll/pitch angle itself, not just their rate) should make that
    growing wobble costly before it becomes destructive; halving PROGRESS_WEIGHT makes sprinting
    less overwhelmingly attractive relative to those penalties.
    """

    TARGET_REWARD = 380.0
    POS_REWARD_STEEPNESS = 15.0
    PROGRESS_WEIGHT = 5.0
    ANG_PENALTY_WEIGHT = 0.15
    TILT_PENALTY_WEIGHT = 3.0
    # Leave ~25% of the command band free for attitude control rather than spending all of it
    # on climbing; see the thrust-headroom note in _computeReward.
    SAT_THRESHOLD = 0.75
    SAT_PENALTY_WEIGHT = 8.0
    # How far from the origin an episode is allowed to wander before it's truncated. This is a
    # training-episode heuristic, not a limit of the policy: the observation is purely
    # target-relative, so the policy is translation-invariant in xy and flies identically at
    # x=5m as at x=0. Trajectory following over a moving subject (see commands.py) legitimately
    # ranges further than a training episode ever does, and raises this.
    XY_BOUND = 1.5

    def reset(self, seed=None, options=None):
        obs, info = self._fastReset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self._prev_pos_err = np.linalg.norm(self.TARGET_POS - state[0:3])
        return obs, info

    def _fastReset(self, seed=None, options=None):
        """Equivalent to BaseAviary.reset(), but skips the expensive p.resetSimulation() +
        URDF reload on every episode.

        BaseAviary.reset() unconditionally wipes the whole physics world and reloads the
        ground plane and drone from URDF files on disk -- expensive next to a normal physics
        step, and paid on every episode boundary. Under SubprocVecEnv (synchronous: a batch of
        N parallel envs can't advance until all N finish their step), whichever env happens to
        be mid-reset becomes the straggler the other N-1 wait on. With RandomHeightHoverAviary,
        episode length varies a lot (crashes end early, full episodes now run 12s), so reset
        timing is uncorrelated across envs and that straggler effect happens often and
        unpredictably -- observed as ~385-860 steps/sec swings across otherwise-identical
        windows of the same training run.

        After the very first reset (which still has to create the bodies), this repositions
        the existing plane/drone bodies instead of reloading them -- physically identical
        (INIT_XYZS/INIT_RPYS are constant across resets already, even in the slow path), just
        without the disk I/O and URDF (re)parsing. Skips re-issuing p.setGravity() etc. since
        those aren't touched by anything except p.resetSimulation(), which this path never
        calls, so they're still in effect; also skips the OBSTACLES/USER_DEBUG-gated calls
        (_addObstacles, _showDroneLocalAxes). BaseRLAviary actually hardcodes obstacles=True
        unconditionally for every env in this file, so _addObstacles() (the library's default,
        or ObstacleAvoidanceAviary's override) already ran once on that first slow reset --
        skipping it here on fast resets is correct precisely because those obstacles are
        static and were already created, not because OBSTACLES is off. Skipping the debug-axis
        call only loses cosmetic per-reset redraw during a GUI replay, not anything physical.

        Must be kept in sync with BaseAviary._housekeeping() if the vendored library changes.
        """
        if getattr(self, "DRONE_IDS", None) is None:
            return super().reset(seed=seed, options=options)

        gym.Env.reset(self, seed=seed, options=options)
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        for i in range(self.NUM_DRONES):
            p.resetBasePositionAndOrientation(
                self.DRONE_IDS[i],
                self.INIT_XYZS[i, :],
                p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                physicsClientId=self.CLIENT,
            )
            p.resetBaseVelocity(
                self.DRONE_IDS[i],
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self.CLIENT,
            )
        self._updateAndStoreKinematicInformation()
        return self._computeObs(), self._computeInfo()

    def _yawPenalty(self, state):
        """Hold heading at zero. Overridden by YawToTravelMixin to face the direction of travel."""
        return 0.3 * (1 - np.cos(state[9]))

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos_err = np.linalg.norm(self.TARGET_POS - state[0:3])
        roll, pitch, yaw = state[7], state[8], state[9]
        vel = state[10:13]
        ang_v = state[13:16]
        pos_reward = 2 * np.exp(-self.POS_REWARD_STEEPNESS * pos_err ** 2)
        vel_penalty = 0.1 * np.dot(vel, vel)
        ang_penalty = self.ANG_PENALTY_WEIGHT * np.dot(ang_v, ang_v)
        tilt_penalty = self.TILT_PENALTY_WEIGHT * (roll ** 2 + pitch ** 2)
        yaw_penalty = self._yawPenalty(state)
        progress_bonus = self.PROGRESS_WEIGHT * (self._prev_pos_err - pos_err)
        self._prev_pos_err = pos_err

        # Thrust-headroom penalty. BaseRLAviary maps a normalized action to
        # HOVER_RPM * (1 + 0.05*action) -- so the entire control range is +-5% of hover RPM,
        # and *attitude* control has to come out of that same narrow band by running some
        # rotors harder than others. Climbing to a tall target consumes all of it: measured on
        # a failing 3.83m climb, all four motor commands sat at 0.98-1.00 with a spread of
        # ~0.018 between them, i.e. essentially zero authority left to correct roll. Roll then
        # drifted 0.6 -> 4.4 -> 6.8 -> -8 -> 23 degrees and truncated, mid-climb, every time
        # (20/20 on targets 3.5-4.0m), while an otherwise identical 1.83m flight came off
        # saturation by step 36 and parked to 1cm. Penalizing commands above SAT_THRESHOLD
        # makes keeping steering authority worth more than arriving a second sooner.
        act = np.asarray(self.action_buffer[-1])[0]
        saturation = max(0.0, float(np.max(act)) - self.SAT_THRESHOLD)
        sat_penalty = self.SAT_PENALTY_WEIGHT * saturation ** 2

        # Safety penalties are applied *outside* the max(0, ...) clamp, unlike the task-shaping
        # terms. Inside it they were dead weight exactly when they mattered most: pos_reward is
        # ~0 whenever the drone is more than ~0.5m from target, so during a multi-metre climb
        # the clamped group was already negative and pinned at 0, making the tilt/angular
        # penalties literally have no gradient for the whole approach. (This is why raising
        # TILT_PENALTY_WEIGHT earlier changed nothing about the climb crashes.) Outside the
        # clamp they stay live for the entire flight, so reward can go negative when the drone
        # is flying dangerously -- which is the intended signal.
        task_reward = max(0, pos_reward - vel_penalty - ang_penalty - yaw_penalty)
        return task_reward + progress_bonus - tilt_penalty - sat_penalty


class WindyPrecisionHoverAviary(PrecisionHoverAviary):
    """PrecisionHoverAviary with random light wind gusts, for a policy robust to disturbance.

    Wind is a piecewise-constant horizontal-biased force applied to the drone body every
    physics substep (240Hz), resampled every 1-2s to a new random direction and magnitude
    (0 to WIND_MAX_FORCE) -- gusts, not a steady headwind. WIND_MAX_FORCE is ~11% of the
    drone's ~0.26N weight, enough to meaningfully push the drone off station without being
    unrecoverable ("light gusts", not a storm).

    Wind stays off until the drone first gets within CALM_RADIUS of the target -- on the
    ground, and during the climb-out that follows, it has the least thrust margin to correct
    a sideways shove, so gusting from step 0 just knocked it over before it ever got a chance
    to stabilize (observed: episodes crashing in ~25-30 steps, reward stuck near 0 for 500k+
    steps even warm-started from a solid no-wind hover policy). Once it's proven it can reach
    a stable hover, gusts kick in and stay on for the rest of the episode -- so it learns
    liftoff undisturbed, then has to hold/adapt through wind mid-flight.
    """

    WIND_MAX_FORCE = 0.03  # N
    WIND_MIN_STEPS = 240   # physics substeps (240Hz) a gust holds for: 240-480 -> 1.0-2.0s
    WIND_MAX_STEPS = 480
    CALM_RADIUS = 0.3      # meters; wind arms once the drone first gets this close to target

    def reset(self, seed=None, options=None):
        self._wind_force = np.zeros(3)
        self._wind_timer = 0
        self._wind_armed = False
        return super().reset(seed=seed, options=options)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if not self._wind_armed:
            pos_err = np.linalg.norm(self.TARGET_POS - self.pos[nth_drone, :])
            if pos_err < self.CALM_RADIUS:
                self._wind_armed = True
            else:
                return  # still on the ground / climbing out -- no wind yet
        if self._wind_timer <= 0:
            direction = np.random.uniform(-1, 1, 3)
            direction[2] *= 0.3  # wind is mostly horizontal
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.zeros(3)
            magnitude = np.random.uniform(0, self.WIND_MAX_FORCE)
            self._wind_force = direction * magnitude
            self._wind_timer = np.random.randint(self.WIND_MIN_STEPS, self.WIND_MAX_STEPS)
        else:
            self._wind_timer -= 1
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=self._wind_force.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )


class RandomHeightHoverAviary(PrecisionHoverAviary):
    """PrecisionHoverAviary where the target hover height is randomized every episode.

    The raw observation is just the drone's own kinematic state (position, orientation,
    velocity, angular velocity) -- it never encoded the target at all, because the target was
    always the same fixed [0,0,1] baked into training. Randomizing the height without changing
    the observation would make the task unlearnable: the same observation (e.g. sitting at
    z=1.0) would require opposite actions (stay vs. keep climbing) depending on an invisible
    target, which a memoryless policy can't condition on. So _computeObs substitutes height
    *error* (target - current, matching the reward's convention) for absolute height -- the
    policy sees "how far off am I" instead of "where am I", which means the same signal is
    meaningful regardless of what the target happens to be that episode.

    HoverAviary's built-in truncation also hard-codes a 2.0m crash ceiling (state[2] > 2.0),
    which would auto-truncate every episode with a target above that before it ever got
    close -- overridden here to scale with MAX_HEIGHT instead.

    HoverAviary.EPISODE_LEN_SEC=8 was tuned for a fixed 1m target reachable in a couple of
    seconds. With targets up to 4m away, 8s pressures the policy to sustain a fast, near-max
    climb rate the whole way -- which is exactly what drove the roll instability seen in
    training (see PrecisionHoverAviary docstring). Extending it gives room to climb at a safer
    pace without running out the clock.
    """

    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EPISODE_LEN_SEC = 12

    def reset(self, seed=None, options=None):
        self.TARGET_POS = np.array([0, 0, np.random.uniform(self.MIN_HEIGHT, self.MAX_HEIGHT)])
        return super().reset(seed=seed, options=options)

    def _observationSpace(self):
        space = super()._observationSpace()
        space.low[:, 2] = -np.inf  # height error can be negative now (below target)
        return space

    def _computeObs(self):
        obs = super()._computeObs()
        obs[:, 0:3] -= self.TARGET_POS
        return obs

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.XY_BOUND or abs(state[1]) > self.XY_BOUND or state[2] > self.MAX_HEIGHT + 1.0
             or abs(state[7]) > .4 or abs(state[8]) > .4
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False


class HorizontalMoveAviary(PrecisionHoverAviary):
    """PrecisionHoverAviary where the target x,y position is randomized every episode.

    Height stays fixed at FIXED_HEIGHT (same target height PrecisionHoverAviary always used)
    -- this isolates lateral control as its own skill before combining it with the vertical
    range mastered in RandomHeightHoverAviary. Unlike that class, no observation-space fix is
    needed: x,y already had [-inf, inf] bounds (only z was floored at 0), and x,y=0 was always
    both the typical converged position AND the target in every prior task, so target-relative
    x,y and absolute x,y were already numerically identical in everything trained so far.

    XY_RANGE=1.0 keeps every possible target at least 0.5m inside HoverAviary's existing
    +-1.5m truncation bound, leaving margin for overshoot. No EPISODE_LEN_SEC extension needed
    either: worst-case start-to-target distance (~1.7m diagonal) is close to the original
    fixed-target task's 1m, nowhere near RandomHeightHoverAviary's up-to-4m climbs that
    actually required more time.

    z itself never needs the RandomHeightHoverAviary-style unbounded observation fix (height
    is fixed here, so z stays physically non-negative) -- but _observationSpace() still loosens
    that bound to match, purely so PPO.load() can warm-start from a RandomHeightHoverAviary
    checkpoint: SB3 checks declared observation Box bounds match exactly on load and refuses
    otherwise, even though the actual values would be fine either way (declared bounds aren't
    used for clipping during training).
    """

    XY_RANGE = 1.0
    FIXED_HEIGHT = 1.0

    def reset(self, seed=None, options=None):
        x = np.random.uniform(-self.XY_RANGE, self.XY_RANGE)
        y = np.random.uniform(-self.XY_RANGE, self.XY_RANGE)
        self.TARGET_POS = np.array([x, y, self.FIXED_HEIGHT])
        return super().reset(seed=seed, options=options)

    def _observationSpace(self):
        space = super()._observationSpace()
        space.low[:, 2] = -np.inf
        return space

    def _computeObs(self):
        obs = super()._computeObs()
        obs[:, 0:3] -= self.TARGET_POS
        return obs


class Combined3DMoveAviary(PrecisionHoverAviary):
    """PrecisionHoverAviary where the full 3D target position (x, y, and z) is randomized
    every episode -- combines RandomHeightHoverAviary's vertical range with
    HorizontalMoveAviary's lateral range into one task, rather than either in isolation.

    Not implemented via multiple inheritance from those two classes: both override reset()/
    _computeObs()/_observationSpace() as complete alternative bodies (not incremental hooks),
    so composing them through MRO would mean carefully merging two full reset() paths rather
    than cleanly extending one -- more error-prone than just writing the combined version
    directly, especially with RandomHeightHoverAviary's non-default EPISODE_LEN_SEC/truncation
    also needing to carry over.

    After 2M+ steps of consolidation, training-time eval reward kept oscillating between
    ~550-600 (low variance) and ~190-390 (huge variance, +-250+) rather than converging --
    across n_eval_episodes=3 with an unseeded random target each episode, that swing is almost
    exactly what "2 easy targets + 1 hard one scores ~0" produces arithmetically. So the
    likely story isn't a genuinely unstable policy, it's a specific hard region of the target
    space (probably far xy *and* extreme z at once) still failing while the rest of the space
    already works. EPISODE_LEN_SEC=12 was sized for RandomHeightHoverAviary's height-only
    travel; the hardest combined cases need to execute large tilt and sustained climb at the
    same time, which is a harder control problem than either alone even at similar raw
    distance, so it plausibly needs more settling time, not just more raw travel time.
    """

    XY_RANGE = 1.0
    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 4.0

    def __init__(self, *args, xy_range=None, min_height=None, max_height=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.EPISODE_LEN_SEC = 16
        # Overridable per-instance so a curriculum can start with an easy, narrow target range
        # and widen it in later phases, rather than uniformly sampling the full range from the
        # start -- 6M+ steps against the full range left variance stuck around +-300-430
        # (see class docstring), consistent with a persistent capability gap on hard target
        # combinations (far xy *and* extreme z at once) rather than noise that more time fixes.
        if xy_range is not None:
            self.XY_RANGE = xy_range
        if min_height is not None:
            self.MIN_HEIGHT = min_height
        if max_height is not None:
            self.MAX_HEIGHT = max_height

    def _sampleTarget(self):
        """This episode's target. Split out from reset() so subclasses can constrain it."""
        x = np.random.uniform(-self.XY_RANGE, self.XY_RANGE)
        y = np.random.uniform(-self.XY_RANGE, self.XY_RANGE)
        z = np.random.uniform(self.MIN_HEIGHT, self.MAX_HEIGHT)
        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        self.TARGET_POS = self._sampleTarget()
        return super().reset(seed=seed, options=options)

    def _observationSpace(self):
        space = super()._observationSpace()
        space.low[:, 2] = -np.inf
        return space

    def _computeObs(self):
        obs = super()._computeObs()
        obs[:, 0:3] -= self.TARGET_POS
        return obs

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.XY_BOUND or abs(state[1]) > self.XY_BOUND or state[2] > self.MAX_HEIGHT + 1.0
             or abs(state[7]) > .4 or abs(state[8]) > .4
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False


class WindyCombined3DMoveAviary(Combined3DMoveAviary):
    """Combined3DMoveAviary with random light wind gusts -- the "real-world variables" stage:
    randomized full 3D target position AND wind disturbance together, rather than either alone.

    Wind mechanics are identical to WindyPrecisionHoverAviary (piecewise gust force, arms once
    the drone first gets within CALM_RADIUS of its *current* target so liftoff/initial travel
    stays undisturbed) -- duplicated rather than shared via multiple inheritance, same reasoning
    as Combined3DMoveAviary's own docstring: both would-be parent classes override reset() as
    complete alternative bodies, so composing them through MRO is more error-prone than writing
    the combination directly.

    First attempt at the OU redesign used WIND_MEAN_FORCE=0.015 (half the cap) as a *sustained*
    per-episode bias -- replaying the trained checkpoint showed why that stalled flat: not
    crashes, but a slow spiral (yaw drifting past 40 degrees, roll/pitch building to 15-18 over
    many seconds) under a wind that never let up for the rest of a 16s episode. That's a
    fundamentally harder, more sustained-torque regime than either the original bursty model
    (effectively zero-mean over time, since it resampled to a fresh random direction) or what
    the source article actually described (gusts with a randomized *characteristic* direction,
    not a strong constant push). Dropping the mean to a small fraction of the cap keeps the
    per-episode direction randomization (still prevents memorizing one fixed trim correction)
    while making the wind genuinely gusty/fluctuating rather than a steady headwind to fight
    indefinitely.
    """

    WIND_MAX_FORCE = 0.03   # N; hard safety cap, never exceeded regardless of OU excursion
    WIND_MEAN_FORCE = 0.005 # N; magnitude of the per-episode mean wind the OU process reverts to
    WIND_THETA = 0.4        # /s; mean-reversion rate -> ~2.5s correlation time constant (1/theta)
    WIND_SIGMA = 0.0134     # noise scale; tuned so stationary std per axis is ~WIND_MAX_FORCE/2
    CALM_RADIUS = 0.3       # meters; wind arms once the drone first gets this close to target

    # Originally a piecewise-constant force that jumped to a new random direction/magnitude
    # every 1-2s -- a sudden step change, not realistic turbulence. Training on it stalled flat
    # near-zero reward for 75%+ of a 3M-step budget even with an entropy bonus: the collapse
    # this produces (converging to near-zero action / "just hover and eat the crash risk" as
    # the least-bad option under an unpredictable jolt) matches a documented failure mode for
    # exactly this kind of abrupt disturbance (see northlakelabs.com "When your drone learns to
    # fight the wind"). That writeup uses an Ornstein-Uhlenbeck process instead: wind that
    # reverts toward a per-episode-randomized mean with exponentially-correlated noise, so it
    # ramps and decays smoothly rather than jumping -- the policy gets a continuous signal to
    # react to instead of a shock. Also randomizing the mean direction per episode (rather than
    # reverting toward zero) prevents the policy from learning one fixed trim correction instead
    # of genuine reactive control.

    def reset(self, seed=None, options=None):
        self._wind_force = np.zeros(3)
        self._wind_armed = False
        direction = np.random.uniform(-1, 1, 3)
        direction[2] *= 0.3  # wind is mostly horizontal
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros(3)
        self._wind_mean = direction * self.WIND_MEAN_FORCE
        return super().reset(seed=seed, options=options)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if not self._wind_armed:
            pos_err = np.linalg.norm(self.TARGET_POS - self.pos[nth_drone, :])
            if pos_err < self.CALM_RADIUS:
                self._wind_armed = True
            else:
                return  # still climbing/traveling to target -- no wind yet
        dt = 1.0 / self.PYB_FREQ
        noise = np.random.randn(3)
        noise[2] *= 0.3  # keep the turbulence itself mostly horizontal too
        self._wind_force += self.WIND_THETA * (self._wind_mean - self._wind_force) * dt \
            + self.WIND_SIGMA * np.sqrt(dt) * noise
        norm = np.linalg.norm(self._wind_force)
        if norm > self.WIND_MAX_FORCE:
            self._wind_force *= self.WIND_MAX_FORCE / norm
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=self._wind_force.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )


class ObstacleAvoidanceAviary(Combined3DMoveAviary):
    """Combined3DMoveAviary with a static obstacle the drone must fly around to reach its
    randomized 3D target, rather than only needing to hold position/attitude.

    The obstacle is repositioned every episode to sit directly on the straight line from the
    drone's spawn point to that episode's target, at OBSTACLE_PATH_FRACTION of the way along.
    A first version parked it at one fixed spot instead, which meant that with targets drawn
    uniformly around the arena the direct path only actually passed near it in roughly half of
    episodes -- the rest taught nothing about avoidance and the policy could score well while
    barely learning to dodge. Placing it on the path guarantees every episode is an avoidance
    episode.

    Deriving the obstacle position from the target (rather than randomizing it independently)
    is deliberate: it keeps the obstacle fully determined by information the policy already
    observes -- it sees target-relative position, and obstacle == fraction * target_xy -- so no
    extra observation channel is needed and warm-starting from a Combined3DMoveAviary
    checkpoint still works. The tradeoff is honest: this teaches "something blocks the middle
    of my route," not general obstacle perception. Genuine perception (independently randomized
    obstacles + relative-position observations, which would change the observation shape and
    require training from scratch) is the real next step beyond this.

    Collision truncates the episode (treated like a crash) rather than only adding a reward
    penalty on contact, consistent with how excessive tilt already truncates rather than just
    being penalized -- a real collision would plausibly damage/crash the drone.

    Uses createCollisionShape/createVisualShape/createMultiBody directly instead of loading a
    URDF, since it wants a specific radius/height matched to the flight volume rather than
    whatever a pre-made asset happens to be sized like.
    """

    OBSTACLE_RADIUS = 0.12
    OBSTACLE_HEIGHT = 3.0
    OBSTACLE_PATH_FRACTION = 0.55
    # The obstacle needs room at both ends of the route. Too close to spawn and it lands on the
    # drone at t=0; too close to the target and the target ends up *inside* the cylinder, which
    # makes the episode literally impossible to complete -- silently poisoning training with
    # unwinnable episodes rather than teaching avoidance. MIN_TARGET_XY_DIST guarantees the
    # route is long enough for both clearances to fit.
    OBSTACLE_MIN_DIST = 0.35
    OBSTACLE_TARGET_CLEARANCE = 0.30
    MIN_TARGET_XY_DIST = 0.70

    def _sampleTarget(self):
        target = super()._sampleTarget()
        xy = target[:2]
        dist = np.linalg.norm(xy)
        if dist < self.MIN_TARGET_XY_DIST:
            if dist < 1e-6:
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                direction = xy / dist
            target[:2] = direction * self.MIN_TARGET_XY_DIST
        return target

    def __init__(self, *args, **kwargs):
        # BaseRLAviary already hardcodes obstacles=True unconditionally for every env in this
        # file (HoverAviary.__init__ doesn't even accept an `obstacles` kwarg to override it),
        # so self.OBSTACLES is already True and _addObstacles() -- resolved to *this* class's
        # override via normal polymorphism -- gets called automatically. No need to pass it.
        #
        # BaseAviary.__init__ actually calls _housekeeping() (and so _addObstacles()) directly
        # during construction, before any explicit .reset() ever runs -- so _obstacle_id gets
        # set correctly as a side effect of super().__init__() below. Setting it to None here
        # afterward would silently wipe that out; _fastReset's "first reset -> full/slow path"
        # logic never actually triggers via .reset() at all for this reason (DRONE_IDS is
        # already set by the time __init__ returns), which is fine since the real one-time
        # setup already happened here.
        self._obstacle_id = None
        super().__init__(*args, **kwargs)

    def _addObstacles(self):
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self.OBSTACLE_RADIUS, height=self.OBSTACLE_HEIGHT, physicsClientId=self.CLIENT
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=self.OBSTACLE_RADIUS, length=self.OBSTACLE_HEIGHT, rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=self.CLIENT
        )
        self._obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, -10],  # parked out of the way; reset() moves it onto the route
            physicsClientId=self.CLIENT,
        )

    def _obstacleXY(self):
        """Point on the spawn->target line where the obstacle sits, clear of the spawn point."""
        target_xy = np.array(self.TARGET_POS[:2], dtype=float)
        dist = np.linalg.norm(target_xy)
        if dist < 1e-6:
            # Target is directly overhead: no meaningful "route" direction, so pick one at
            # random rather than dividing by ~0.
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            direction = target_xy / dist
        along = dist * self.OBSTACLE_PATH_FRACTION
        along = max(along, self.OBSTACLE_MIN_DIST)
        along = min(along, max(dist - self.OBSTACLE_TARGET_CLEARANCE, self.OBSTACLE_MIN_DIST))
        return direction * along

    def reset(self, seed=None, options=None):
        # Runs after super().reset() because Combined3DMoveAviary.reset() is what draws this
        # episode's TARGET_POS, and the obstacle position is derived from it. The body is
        # static (mass 0), so repositioning it is just a teleport -- no physics settling needed.
        out = super().reset(seed=seed, options=options)
        if self._obstacle_id is not None:
            xy = self._obstacleXY()
            p.resetBasePositionAndOrientation(
                self._obstacle_id,
                [xy[0], xy[1], self.OBSTACLE_HEIGHT / 2],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )
        return out

    def _collided(self):
        if self._obstacle_id is None:
            return False
        return len(p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=self._obstacle_id, physicsClientId=self.CLIENT)) > 0

    def _computeTruncated(self):
        if self._collided():
            return True
        return super()._computeTruncated()


class YawToTravelMixin:
    """Reward the drone for turning to face where it is going, instead of pinning heading to 0.

    Every earlier task pinned yaw at zero, so the drone crabbed: to reach a target behind it,
    it translated backwards rather than turning around. That made cruise direction-dependent --
    measured at 0.75 m/s sustained, forward and one sideways direction tracked fine while
    backward and a diagonal crashed outright, and orbits died at exactly the point in the
    circle where travel goes backwards. Training more cruise fixed some directions but cost
    precision everywhere, and raising the trained speed range made it strictly worse.

    Facing the direction of travel makes every cruise "forward", which is already the strongest
    direction by a wide margin -- so this is meant to dissolve the eight-direction problem
    rather than keep training around it, and it is how a real drone flies.

    Heading is taken from the bearing to the target rather than from the velocity vector: the
    target is where the drone is *about* to go (lead compensation puts it further ahead still),
    while velocity is where it is already going, which would lag and could chase its own tail
    when nearly stationary. Inside YAW_DEADBAND the bearing is ill-conditioned -- a drone parked
    on its target has no meaningful direction to face -- so the penalty is switched off there
    and the drone is free to hold whatever heading it arrived with.
    """

    YAW_DEADBAND = 0.20  # metres of horizontal error below which "which way am I going" is meaningless

    def _yawPenalty(self, state):
        to_target = self.TARGET_POS[:2] - state[0:2]
        dist = np.linalg.norm(to_target)
        if dist < self.YAW_DEADBAND:
            return 0.0
        desired_yaw = np.arctan2(to_target[1], to_target[0])
        # (1 - cos) of the *error* rather than of yaw itself, so it stays continuous across the
        # +-pi wrap in either the drone's heading or the bearing.
        return 0.3 * (1 - np.cos(state[9] - desired_yaw))


class CruiseAviary(Combined3DMoveAviary):
    """Target holds still briefly, then translates at a constant velocity for the rest of the
    episode -- the drone has to fly *with* it, not fly to it and park.

    Every task before this one was "go to a static point and stop", and the resulting policy
    could not cruise: driven by commands.py with a setpoint moving in a straight line at
    0.75 m/s it crashed within ~13m, at any lead-compensation setting, while tight orbits at
    the same speed were fine. Sustained one-directional travel simply never appeared in
    training -- the furthest it was ever asked to fly was ~1.4m, and it always decelerated to a
    stop at the end. This task makes the steady-state cruise itself the thing being rewarded.

    The episode deliberately mirrors how commands.py actually drives the policy: a stationary
    hold first (long enough to launch from the ground and settle -- the policy already does
    that part well) and only then a moving setpoint. Training on the deployment pattern rather
    than a tidier abstraction avoids teaching a skill that has to be re-bridged later.

    Note this does *not* add target velocity to the observation, which is the textbook fix for
    tracking lag -- the policy still only sees relative position, so it must infer motion from
    how the error evolves and will retain some lag. That is a deliberate scope choice: lag is
    already handled well in layer 2 by lead compensation (1-4cm tracking), whereas *crashing*
    is not recoverable at any layer, so the crash is what is worth spending training on. Adding
    a velocity channel changes the observation shape and forfeits every existing checkpoint.
    """

    HOLD_SEC = 4.0          # stationary hold before the setpoint starts moving
    MIN_CRUISE_SPEED = 0.2
    MAX_CRUISE_SPEED = 1.0  # m/s; curriculum knob -- raise once the low end is reliable
    XY_BOUND = 60.0         # a 1 m/s cruise for 20s covers 20m; see the note on XY_BOUND above

    def __init__(self, *args, min_cruise_speed=None, max_cruise_speed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.EPISODE_LEN_SEC = 20
        if min_cruise_speed is not None:
            self.MIN_CRUISE_SPEED = min_cruise_speed
        if max_cruise_speed is not None:
            self.MAX_CRUISE_SPEED = max_cruise_speed

    def reset(self, seed=None, options=None):
        speed = np.random.uniform(self.MIN_CRUISE_SPEED, self.MAX_CRUISE_SPEED)
        heading = np.random.uniform(0, 2 * np.pi)
        self._cruise_vel = np.array([np.cos(heading), np.sin(heading), 0.0]) * speed
        out = super().reset(seed=seed, options=options)
        self._cruise_origin = self.TARGET_POS.copy()
        return out

    def _advanceTarget(self):
        elapsed = self.step_counter / self.PYB_FREQ
        moving_for = max(0.0, elapsed - self.HOLD_SEC)
        self.TARGET_POS = self._cruise_origin + self._cruise_vel * moving_for

    def step(self, action):
        # Advance before super().step() so the physics, observation and reward for this step
        # all use the same target -- BaseAviary computes obs/reward at the end of step().
        self._advanceTarget()
        return super().step(action)


class CruiseYawAviary(YawToTravelMixin, CruiseAviary):
    """CruiseAviary, but the drone turns to face where it is going. See YawToTravelMixin."""


ENVS = {
    "hover": HoverAviary,
    "precision": PrecisionHoverAviary,
    "windy": WindyPrecisionHoverAviary,
    "random-height": RandomHeightHoverAviary,
    "horizontal": HorizontalMoveAviary,
    "combined-3d": Combined3DMoveAviary,
    "windy-combined-3d": WindyCombined3DMoveAviary,
    "obstacle": ObstacleAvoidanceAviary,
    "cruise": CruiseAviary,
    "cruise-yaw": CruiseYawAviary,
}


def train(env_cls, timesteps: int, n_envs: int, output_folder: str, warm_start: str = None, learning_rate: float = 3e-4, ent_coef: float = 0.0, eval_freq: int = 2000, n_eval_episodes: int = 5, target_reward: float = -1.0, env_kwargs: dict = None) -> str:
    run_dir = os.path.join(output_folder, "hover-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(run_dir, exist_ok=True)

    env_kwargs = dict(obs=OBS_TYPE, act=ACT_TYPE, **(env_kwargs or {}))
    train_env = make_vec_env(
        env_cls,
        env_kwargs=env_kwargs,
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )
    # A single serial eval env means n_eval_episodes costs that many episodes' worth of
    # wall-clock time, one after another -- with n_eval_episodes=6 and 16s episodes, that's
    # ~2.7-3.7x slower overall throughput than a cheap 3-episode eval, for a signal that's
    # only better because it averages over more episodes. Running one eval worker per episode
    # in parallel keeps that same signal (still n_eval_episodes samples) but collapses the
    # wall-clock cost from N serial episodes down to ~1 episode's worth, since they all run
    # simultaneously -- eval frequency/episode count no longer have to trade off against speed.
    eval_env = make_vec_env(
        env_cls,
        env_kwargs=env_kwargs,
        n_envs=n_eval_episodes,
        seed=1,
        vec_env_cls=SubprocVecEnv if n_eval_episodes > 1 else None,
    )

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    if warm_start:
        # A fresh, unfamiliar wind disturbance makes the first rollouts look very different
        # from what the warm-started policy was trained on. At the default LR (3e-4) that
        # first gradient step can be large enough to wipe out the pretrained behavior
        # entirely (observed: episode length collapsed 12 -> 3 steps within 3 eval cycles).
        # A much smaller LR lets it adapt gradually instead of overwriting what it knows.
        model = PPO.load(warm_start, env=train_env, device="cpu", learning_rate=learning_rate, ent_coef=ent_coef)
        print(f"[INFO] Warm-started from {warm_start} at lr={learning_rate}, ent_coef={ent_coef}")
    else:
        model = PPO("MlpPolicy", train_env, verbose=1, device="cpu", learning_rate=learning_rate, ent_coef=ent_coef)

    # target_reward=-1.0 (default) means "use the env class's own threshold"; pass 0 to
    # disable early stopping entirely. Needed because with only n_eval_episodes=3 (kept low to
    # avoid burning wall-clock on eval -- see eval_freq/n_eval_episodes above), a single lucky
    # sample can cross an inherited threshold long before the policy is actually consistent
    # (observed on Combined3DMoveAviary: stopped after just 2 evals on a 453.75 sample, while
    # a full 10-episode replay right after still showed 449 +/- 156 -- high variance, not
    # genuine convergence). A harder/more varied task needs either a higher bar or none at all
    # while consolidating.
    if target_reward < 0:
        target_reward = getattr(env_cls, "TARGET_REWARD", None)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1) if target_reward else None
    # eval_freq is "calls to the callback" (one per rollout step across all n_envs at once),
    # so actual env-steps between evals = eval_freq * n_envs. Eval itself runs n_eval_episodes
    # serially in a single non-parallel env -- with the old eval_freq=2000/n_eval_episodes=5,
    # that was costing a real fraction of wall-clock (5 full episodes, no parallelism, every
    # 16k steps) for something that doesn't train the policy at all, just measures it. Wider
    # spacing and fewer episodes trades eval resolution for more actual training throughput.
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, log_interval=100, reset_num_timesteps=warm_start is None)

    final_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_path)

    with np.load(os.path.join(run_dir, "evaluations.npz")) as data:
        ts, results = data["timesteps"], data["results"][:, 0]
        for t, r in zip(ts, results):
            print(f"{t},{r:.2f}")

    train_env.close()
    eval_env.close()
    print(f"[INFO] Run saved to {run_dir}")
    return run_dir


def replay(env_cls, model_path: str, env_kwargs: dict = None, speed: float = 1.0):
    env_kwargs = env_kwargs or {}
    model = PPO.load(model_path)
    env = env_cls(gui=True, obs=OBS_TYPE, act=ACT_TYPE, **env_kwargs)

    mean_reward, std_reward = evaluate_policy(model, Monitor(env_cls(obs=OBS_TYPE, act=ACT_TYPE, **env_kwargs)), n_eval_episodes=10)
    print(f"[INFO] Mean reward over 10 eval episodes: {mean_reward:.2f} +- {std_reward:.2f}")

    obs, _ = env.reset(seed=42)
    start = time.time()
    for i in range((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        # sync() paces playback to CTRL_TIMESTEP (real-time/1x) by sleeping; dividing the
        # timestep by speed makes it think less wall-clock time should have passed per step,
        # so it sleeps less and plays back faster (speed=0 or negative would be nonsensical --
        # not guarded against since this is a local dev script, not user-facing).
        sync(i, start, env.CTRL_TIMESTEP / speed)
        if terminated or truncated:
            obs, _ = env.reset(seed=42)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/replay a PPO hover policy")
    parser.add_argument("--env", choices=list(ENVS), default="windy", help="which task/env to use")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments for training")
    parser.add_argument("--output-folder", type=str, default="results")
    parser.add_argument("--warm-start", type=str, default=None, help="path to a saved model .zip to continue training from")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate (lower this when warm-starting to avoid catastrophic forgetting)")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy bonus (SB3 default is 0.0); raise this if training has plateaued around an entrenched habit and needs more exploration to escape it")
    parser.add_argument("--eval-freq", type=int, default=2000, help="callback calls between evals (actual env-steps between evals = this * n-envs); raise to spend less wall-clock on evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="episodes per eval, run serially in a single non-parallel env; lower to spend less wall-clock on evaluation")
    parser.add_argument("--target-reward", type=float, default=-1.0, help="early-stop reward threshold; defaults to the env class's own TARGET_REWARD, pass 0 to disable early stopping and always use the full --timesteps budget")
    parser.add_argument("--xy-range", type=float, default=None, help="Combined3DMoveAviary/HorizontalMoveAviary: override the target x,y range (meters) -- narrow this for an easy curriculum phase, defaults to the class's own XY_RANGE")
    parser.add_argument("--min-height", type=float, default=None, help="Combined3DMoveAviary/RandomHeightHoverAviary: override the target min height (meters), defaults to the class's own MIN_HEIGHT")
    parser.add_argument("--max-height", type=float, default=None, help="Combined3DMoveAviary/RandomHeightHoverAviary: override the target max height (meters), defaults to the class's own MAX_HEIGHT")
    parser.add_argument("--min-cruise-speed", type=float, default=None, help="CruiseAviary: slowest setpoint cruise speed (m/s)")
    parser.add_argument("--max-cruise-speed", type=float, default=None, help="CruiseAviary: fastest setpoint cruise speed (m/s) -- the curriculum knob for cruise")
    parser.add_argument("--replay", type=str, default=None, help="path to a saved model .zip to watch fly instead of training")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="replay playback speed multiplier (2.0 = 2x real-time, default is real-time)")
    args = parser.parse_args()

    env_cls = ENVS[args.env]
    curriculum_kwargs = {}
    if args.xy_range is not None:
        curriculum_kwargs["xy_range"] = args.xy_range
    if args.min_height is not None:
        curriculum_kwargs["min_height"] = args.min_height
    if args.max_height is not None:
        curriculum_kwargs["max_height"] = args.max_height
    if args.min_cruise_speed is not None:
        curriculum_kwargs["min_cruise_speed"] = args.min_cruise_speed
    if args.max_cruise_speed is not None:
        curriculum_kwargs["max_cruise_speed"] = args.max_cruise_speed

    if args.replay:
        replay(env_cls, args.replay, env_kwargs=curriculum_kwargs, speed=args.replay_speed)
    else:
        train(env_cls, args.timesteps, args.n_envs, args.output_folder, warm_start=args.warm_start, learning_rate=args.learning_rate, ent_coef=args.ent_coef, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes, target_reward=args.target_reward, env_kwargs=curriculum_kwargs)
