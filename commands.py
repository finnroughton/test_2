"""Layer 1/2 of the stack: turn a high-level command into a stream of setpoints, and fly it.

The split:

    "orbit me"  --(an LLM, i.e. Claude at conversation time)-->  TrajectorySpec
    TrajectorySpec  --(this file)-->  desired position at time t
    desired position  --(the trained RL policy)-->  motor commands

Only the middle layer belongs in code. Natural language is open-ended and parsing it with
regexes would be a toy; a language model already does that translation well and can ask a
clarifying question when a command is ambiguous ("orbit at what radius?"). So the durable
artifact here is the *trajectory vocabulary* an interpreter emits into, plus an executor that
runs one against the policy and reports how well it actually tracked.

Run as:
    uv run python commands.py --model results/<run-dir>/best_model.zip --command orbit
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

import time

import pybullet as p
from gym_pybullet_drones.utils.utils import sync

from drone_sim import Combined3DMoveAviary, OBS_TYPE, ACT_TYPE


#: Seconds over which lead compensation is eased in at the start of tracking. Applying the full
#: lead immediately is a position step proportional to speed, which pitches the drone over hard
#: enough to truncate at anything above walking pace -- see the note in fly().
LEAD_RAMP_SEC = 2.0


class Trajectory:
    """A desired position as a function of time. Subclasses are the command vocabulary."""

    #: seconds the drone is given to fly from spawn to the trajectory's start point before
    #: tracking is scored -- the policy was trained to launch from the ground, so the approach
    #: is a normal static-target flight and only what follows is genuinely trajectory tracking.
    approach_time = 6.0

    def at(self, t):
        raise NotImplementedError

    def start(self):
        return self.at(0.0)


class Hover(Trajectory):
    """'hold at 2 metres' -- the degenerate trajectory, useful as a tracking control case."""

    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def at(self, t):
        return self.position


class Subject:
    """Where the person being followed is at time t. 'Me', in 'orbit me'.

    In a real system this is a live position estimate (GPS, a tracked tag, vision). Here it's
    a scripted path so the tracking behaviour can be measured repeatably.
    """

    def at(self, t):
        raise NotImplementedError


class Stationary(Subject):
    def __init__(self, position=(0.0, 0.0)):
        self.position = np.asarray(position, dtype=float)

    def at(self, t):
        return self.position


class Walking(Subject):
    """Person walking a straight line at a constant speed.

    Real walking is ~1.4 m/s, which is fast relative to this drone -- it has a ~0.75s control
    lag and the orbit adds its own tangential speed on top of the walking speed, so the two
    compose into something much harder than either alone. Default is deliberately a slow
    stroll; `speed` is the knob worth sweeping to find where following breaks down.
    """

    def __init__(self, speed=0.25, heading=0.0, start=(0.0, 0.0)):
        self.speed = speed
        self.direction = np.array([np.cos(heading), np.sin(heading)])
        self.start = np.asarray(start, dtype=float)

    def at(self, t):
        return self.start + self.direction * (self.speed * t)


class Pacing(Subject):
    """Person walking back and forth -- includes direction reversals, which a straight-line
    walk never tests. Reversals are where a follower is most likely to overshoot.

    Sinusoidal rather than a triangle wave. A triangle wave reverses instantaneously, which
    means infinite acceleration at each turnaround -- no person does that, and it is not a fair
    test of the drone: it demands the setpoint teleport from full speed one way to full speed
    the other. That version crashed the follower every time (and lead compensation made it
    worse, since looking 0.75s ahead across a reversal points the drone the wrong way entirely).
    A sinusoid decelerates into the turn and accelerates out of it, like walking does.
    `speed` is the peak speed, reached mid-stride.
    """

    def __init__(self, speed=0.25, extent=1.0, heading=0.0, start=(0.0, 0.0)):
        self.speed = speed
        self.extent = extent
        self.direction = np.array([np.cos(heading), np.sin(heading)])
        self.start = np.asarray(start, dtype=float)
        # Peak speed of extent*sin(wt) is extent*w, so w = speed/extent gives the asked-for peak.
        self.omega = speed / extent

    def at(self, t):
        offset = self.extent * np.sin(self.omega * t)
        return self.start + self.direction * offset


class Orbit(Trajectory):
    """'orbit me' -- circle a subject at fixed radius and altitude, following them as they move.

    `period` is seconds per revolution; smaller means faster tangential speed and a harder
    tracking problem, since the setpoint runs away from the drone continuously. When the
    subject also moves, the drone's required ground speed is the vector sum of the orbital
    tangential velocity and the subject's velocity -- so a moving orbit is strictly harder than
    a stationary one, and worst at the point in the circle where the two align.
    """

    def __init__(self, subject=None, radius=0.8, altitude=1.5, period=12.0):
        if subject is None:
            subject = Stationary()
        elif not isinstance(subject, Subject):
            subject = Stationary(subject)
        self.subject = subject
        self.radius = radius
        self.altitude = altitude
        self.period = period

    def at(self, t):
        center = self.subject.at(t)
        angle = 2 * np.pi * (t / self.period)
        return np.array([
            center[0] + self.radius * np.cos(angle),
            center[1] + self.radius * np.sin(angle),
            self.altitude,
        ])


class FollowBehind(Trajectory):
    """'follow me' -- hold station off the subject rather than circling them.

    By default this is the follow-cam position: above head height and behind the direction of
    travel. Both parts are safety, not framing. The operator is 1.8m tall, so anything at or
    below that is at eye level and a 27g quadrotor with exposed props becomes an injury rather
    than a lost drone; and keeping the standoff *behind* the direction of travel means a loss
    of control drops the drone behind the operator rather than onto them. SAFE_ALTITUDE also
    sits just inside the measured reliable altitude band (98-100% success up to 2.5m), so the
    safe choice and the dependable choice happen to coincide.

    `behind` is measured along the subject's own direction of travel rather than a fixed world
    offset, so "behind me" stays behind when the operator turns around. When the subject is
    stationary the travel direction is undefined and the last known one is reused.
    """

    SAFE_ALTITUDE = 2.5
    SAFE_BEHIND = 1.2

    def __init__(self, subject=None, offset=None, altitude=None, behind=None):
        self.subject = subject if subject is not None else Stationary()
        self.altitude = self.SAFE_ALTITUDE if altitude is None else altitude
        self.behind = self.SAFE_BEHIND if behind is None else behind
        # A fixed world-frame offset is still available for tests that need a specific
        # geometry; passing it disables travel-relative placement.
        self.offset = None if offset is None else np.asarray(offset, dtype=float)
        self._last_dir = np.array([1.0, 0.0])

    #: Finite-difference half-window for estimating travel direction. Deliberately wide: a
    #: narrow window reverses the direction almost instantly when the operator turns around,
    #: which swings the standoff point through 180 degrees and teleports the setpoint 2*behind
    #: metres across. Measured with a 0.05s window that produced a loss of control and a strike
    #: on the operator during pacing. A wide window turns the reversal into a gradual swing.
    DIR_WINDOW = 1.0
    #: Speed at which the standoff reaches its full `behind` distance.
    FULL_STANDOFF_SPEED = 0.35
    #: Horizontal standoff never goes below this, so the drone is never directly overhead.
    MIN_STANDOFF = 0.0

    def _travelState(self, t):
        ahead = np.asarray(self.subject.at(t + self.DIR_WINDOW))[:2]
        back = np.asarray(self.subject.at(max(0.0, t - self.DIR_WINDOW)))[:2]
        delta = ahead - back
        norm = np.linalg.norm(delta)
        speed = norm / (2 * self.DIR_WINDOW)
        if norm > 1e-4:
            self._last_dir = delta / norm
        return self._last_dir, speed

    def at(self, t):
        center = np.asarray(self.subject.at(t))
        if self.offset is not None:
            return np.array([center[0] + self.offset[0], center[1] + self.offset[1], self.altitude])
        direction, speed = self._travelState(t)
        # Standoff shrinks as the operator slows and grows back as they pick up speed. Through
        # a direction reversal the operator is momentarily near-stationary, so the drone eases
        # in toward overhead and back out on the new side instead of whipping around them --
        # smooth, and it keeps the drone above head height throughout rather than swinging it
        # through eye level.
        # Never let the standoff collapse to zero, even at a standstill: hovering directly
        # overhead reads as fine on a clearance metric (there is still 0.5m of air above the
        # operator's head) but it is the worst place to fail from, because a drop lands on
        # them. Holding a floor keeps every failure a drop *near* the operator rather than
        # *onto* them.
        scale = min(1.0, speed / self.FULL_STANDOFF_SPEED)
        standoff = max(self.MIN_STANDOFF, self.behind * scale)
        pos = center[:2] - direction * standoff
        return np.array([pos[0], pos[1], self.altitude])


def fly(model_path, trajectory, duration=20.0, gui=False, env_kwargs=None, lead_time=0.0, quiet=False, model=None, stop_on_crash=True):
    """Fly a trajectory with the trained policy; report how closely it tracked.

    The env is only a vehicle for the physics and the policy's observation format here -- its
    reward is irrelevant, and TARGET_POS is overwritten every control step with the setpoint
    the trajectory asks for, which is precisely the interface layer 2 would use in a real
    system.
    """
    env_kwargs = env_kwargs or {}
    if model is None:
        model = PPO.load(model_path)
    env = Combined3DMoveAviary(gui=gui, obs=OBS_TYPE, act=ACT_TYPE, **env_kwargs)
    env.EPISODE_LEN_SEC = duration + trajectory.approach_time + 5
    # Following a walking subject legitimately ranges past the training arena; the policy is
    # translation-invariant in xy (it only ever sees target-relative position) so this is a
    # bookkeeping limit, not a safety one. Sized to cover the subject's walk plus orbit radius.
    env.XY_BOUND = 50.0

    obs, _ = env.reset()
    dt = 1.0 / env.CTRL_FREQ

    # Visual-only markers so a GUI run is actually readable: without them the window shows a
    # drone moving for no visible reason. Collision-free (createMultiBody with no collision
    # shape) so they cannot perturb the physics being measured.
    markers = {}
    if gui:
        subject = getattr(trajectory, "subject", None)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.1, 0.9, 0.2, 0.9], physicsClientId=env.CLIENT)
        markers["setpoint"] = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, -10], physicsClientId=env.CLIENT
        )
        if subject is not None:
            # A life-sized body, not a dot: 1.8m tall and 0.75m across. A small marker made
            # eye-level fly-bys look harmless on screen when they were anything but -- the
            # whole point of drawing the operator is to see whether the drone is flying through
            # the space a person occupies.
            body = p.createVisualShape(
                p.GEOM_CYLINDER, radius=PERSON_RADIUS, length=PERSON_HEIGHT,
                rgbaColor=[0.2, 0.4, 1.0, 0.55], physicsClientId=env.CLIENT,
            )
            markers["subject"] = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=body, basePosition=[0, 0, -10], physicsClientId=env.CLIENT
            )
    approach_steps = int(trajectory.approach_time * env.CTRL_FREQ)
    total_steps = approach_steps + int(duration * env.CTRL_FREQ)

    tracking_errors = []
    crashed = False
    start_wall = time.time()
    for i in range(total_steps):
        # Phase 1 holds the trajectory's start point still so the drone can climb to it from
        # the ground; phase 2 starts advancing the setpoint and scores the tracking.
        if i < approach_steps:
            env.TARGET_POS = trajectory.start()
            scored_target = env.TARGET_POS
        else:
            t = (i - approach_steps) * dt
            # Lead compensation. The policy is a position regulator: it steers toward wherever
            # the setpoint currently is, so against a moving setpoint it settles into a steady
            # trail of roughly its own settling time (measured ~0.75s, near-constant across
            # orbit speeds -- error/speed came out 0.79/0.75/0.67s at three different speeds).
            # Because layer 2 knows the trajectory analytically, it can simply hand the policy
            # where the setpoint *will* be, and let the lag cancel the lead. This is a layer-2
            # fix for a layer-3 limitation -- no retraining needed.
            #
            # The lead is ramped in rather than applied from the first tracking step. Applied
            # as a step it teleports the setpoint lead_time*speed metres forward instantly --
            # 1.05m at 1.4 m/s -- and the drone pitches over 20 degrees chasing it and
            # truncates within 0.2s. That single line was the cause of every high-speed
            # "crash" measured before it was fixed: with the step removed the same policy
            # cruises at 2.0 m/s fine. Ramping keeps the benefit without the transient.
            eased = lead_time * min(1.0, t / LEAD_RAMP_SEC) if LEAD_RAMP_SEC > 0 else lead_time
            env.TARGET_POS = trajectory.at(t + eased)
            scored_target = trajectory.at(t)

        if markers:
            t_now = max(0.0, (i - approach_steps) * dt)
            p.resetBasePositionAndOrientation(
                markers["setpoint"], list(scored_target), [0, 0, 0, 1], physicsClientId=env.CLIENT
            )
            if "subject" in markers:
                c = trajectory.subject.at(t_now)
                p.resetBasePositionAndOrientation(
                    markers["subject"], [c[0], c[1], PERSON_HEIGHT / 2], [0, 0, 0, 1], physicsClientId=env.CLIENT
                )

        obs = env._computeObs()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if gui:
            # Keep the camera on the drone. The default camera is fixed at the origin, so any
            # trajectory that travels -- which is all the interesting ones -- flies out of
            # frame within seconds and the run is unwatchable.
            drone_pos = env._getDroneStateVector(0)[0:3]
            p.resetDebugVisualizerCamera(
                cameraDistance=2.2, cameraYaw=50, cameraPitch=-28,
                cameraTargetPosition=list(drone_pos), physicsClientId=env.CLIENT,
            )
            sync(i, start_wall, env.CTRL_TIMESTEP)

        state = env._getDroneStateVector(0)
        if i >= approach_steps and not crashed:
            # Scored against the *true* setpoint, never the lead point -- otherwise lead
            # compensation would flatter itself by grading against its own fudge factor.
            # Stops accumulating once control is lost so post-crash tumbling, which is only
            # simulated for visualisation, cannot pollute the tracking statistics.
            tracking_errors.append(float(np.linalg.norm(scored_target - state[0:3])))
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            if not crashed and not quiet:
                print(f"  [{(i - approach_steps) * dt:.2f}s] LOST CONTROL: roll {np.degrees(state[7]):+.0f} deg, "
                      f"pitch {np.degrees(state[8]):+.0f} deg, speed {np.linalg.norm(state[10:13]):.2f} m/s")
            crashed = True
            # Training truncates the episode here, so this is where measurement stops. For a
            # GUI run it is worth continuing anyway: the interesting part -- whether it tumbles
            # into the ground or recovers -- happens *after* the threshold, and stopping at the
            # threshold makes every failure look identical.
            if stop_on_crash:
                break
    env.close()

    if not quiet:
        name = type(trajectory).__name__
        print(f"\n=== {name}: {len(tracking_errors) / env.CTRL_FREQ:.1f}s tracked ===")
        if crashed:
            print("  CRASHED mid-trajectory")
        if tracking_errors:
            e = np.array(tracking_errors)
            print(f"  tracking error   median {np.median(e):.3f}m   p90 {np.percentile(e, 90):.3f}m   worst {e.max():.3f}m")
    return tracking_errors, crashed


#: The operator is 1.8m tall. Model them as a vertical cylinder from the ground to head height:
#: anything the drone does inside this volume is a strike on a person, not a near miss, and a
#: 27g quadrotor with exposed props at eye level is an injury rather than a lost drone. Radius
#: covers shoulder width plus an arm.
PERSON_HEIGHT = 1.8
PERSON_RADIUS = 0.375  # 0.75m across, shoulder to shoulder


def clearance_to_person(drone_pos, person_xy):
    """Shortest distance from the drone to the operator's body cylinder. Negative = strike."""
    dxy = np.linalg.norm(np.asarray(drone_pos[:2]) - np.asarray(person_xy[:2]))
    horizontal_gap = dxy - PERSON_RADIUS
    vertical_gap = drone_pos[2] - PERSON_HEIGHT
    if horizontal_gap <= 0 and vertical_gap <= 0:
        # Inside the cylinder's footprint and below head height: this is a strike, and the
        # signed depth is how far inside.
        return max(horizontal_gap, vertical_gap)
    if horizontal_gap <= 0:
        return vertical_gap          # directly overhead, clearance is purely height
    if vertical_gap <= 0:
        return horizontal_gap        # at body height, clearance is purely standoff
    return float(np.hypot(horizontal_gap, vertical_gap))  # clear of the rim, diagonal distance


def safety_report(model_path, trajectory, duration=25.0, lead_time=0.75, label=""):
    """Fly a trajectory and report the worst-case clearance to the operator.

    Tracking error says how well the drone held its commanded point. It says nothing about
    whether that commanded point was somewhere safe -- a trajectory can be tracked to 3cm and
    still be routed straight through the operator's head. This measures the thing that
    actually matters for standing next to it.
    """
    subject = getattr(trajectory, "subject", None)
    model = PPO.load(model_path)
    env = Combined3DMoveAviary(obs=OBS_TYPE, act=ACT_TYPE)
    env.EPISODE_LEN_SEC = duration + trajectory.approach_time + 5
    env.XY_BOUND = 80.0

    obs, _ = env.reset()
    dt = 1.0 / env.CTRL_FREQ
    approach_steps = int(trajectory.approach_time * env.CTRL_FREQ)
    worst = np.inf
    min_alt = np.inf
    crashed = False
    for i in range(approach_steps + int(duration * env.CTRL_FREQ)):
        t = max(0.0, (i - approach_steps) * dt)
        if i < approach_steps:
            env.TARGET_POS = trajectory.start()
        else:
            eased = lead_time * min(1.0, t / LEAD_RAMP_SEC)
            env.TARGET_POS = trajectory.at(t + eased)
        obs = env._computeObs()
        action, _ = model.predict(obs, deterministic=True)
        env.step(action)
        s = env._getDroneStateVector(0)
        # Only score the tracking phase. The drone launches from the ground at the origin, so
        # including the climb-out would report a strike for every trajectory purely because
        # takeoff passes through ground level -- true, but it is a separate procedure (the
        # operator simply must not stand on the launch point) and it would mask whether the
        # commanded trajectory itself is safe to stand next to.
        if i >= approach_steps:
            person_xy = subject.at(t)[:2] if subject is not None else np.zeros(2)
            worst = min(worst, clearance_to_person(s[0:3], person_xy))
            min_alt = min(min_alt, s[2])
        if abs(s[7]) > 0.4 or abs(s[8]) > 0.4:
            crashed = True
    env.close()
    verdict = "STRIKES OPERATOR" if worst <= 0 else ("tight" if worst < 0.5 else "safe")
    print(f"  {label:<34} min clearance {worst:+.2f}m   min alt {min_alt:.2f}m   "
          f"{'lost control  ' if crashed else ''}{verdict}")
    return worst


def feasibility_sweep(model_path, speeds, accels, altitude=1.5, duration=18.0, lead_time=0.75):
    """Map which (setpoint speed, setpoint acceleration) combinations the policy can track.

    Testing named commands one at a time answers "does *this* command work"; layer 2 needs the
    general question, "which commands are safe to generate at all". Speed and acceleration are
    the right axes because a circular setpoint lets them be set independently -- for radius R
    and period T, speed v = 2*pi*R/T and centripetal acceleration a = v^2/R, so asking for a
    given (v, a) just means R = v^2/a and T = 2*pi*v/a.

    The output is the envelope layer 2 should clamp trajectories to, which is what actually
    delivers "it won't crash on me": not a better policy, but a planner that declines to ask
    for flight the policy can't deliver.
    """
    print(f"\n=== trajectory feasibility envelope: {model_path} ===")
    print(f"    (lead {lead_time}s, {duration:.0f}s per cell; 'ok' shows median tracking error)\n")
    header = "    accel \\ speed  " + "".join(f"{v:>9.2f}" for v in speeds)
    print(header)
    for a in accels:
        row = f"    {a:>6.2f} m/s2   "
        for v in speeds:
            radius = v * v / a
            period = 2 * np.pi * v / a
            # Offset the circle so its t=0 point sits directly above the spawn point. Centred
            # on the origin instead, a low-acceleration cell implies a large radius (v=0.75,
            # a=0.15 gives R=3.75m) and the drone would have to *approach* a start point ~4m
            # away -- far outside the +-1.4m it was ever trained to fly to -- so those cells
            # measured approach failure and reported it as a tracking limit.
            traj = Orbit(subject=Stationary((-radius, 0.0)), radius=radius, altitude=altitude, period=period)
            errors, crashed = fly(model_path, traj, duration=duration, lead_time=lead_time, quiet=True)
            if crashed:
                row += f"{'CRASH':>9}"
            else:
                row += f"{np.median(errors):>9.3f}"
        print(row)
    print("\n    (values are median tracking error in metres)")


COMMANDS = {
    # What an interpreter emits into. Values are the structured form of a spoken command.
    "hover": lambda: Hover((0.0, 0.0, 1.5)),
    "orbit": lambda: Orbit(radius=0.8, altitude=1.5, period=12.0),
    "orbit-slow": lambda: Orbit(radius=0.8, altitude=1.5, period=24.0),
    "orbit-fast": lambda: Orbit(radius=0.8, altitude=1.5, period=6.0),
    # "orbit me while I walk"
    "orbit-walking": lambda: Orbit(subject=Walking(speed=0.25), radius=0.8, altitude=1.5, period=12.0),
    "orbit-walking-fast": lambda: Orbit(subject=Walking(speed=0.6), radius=0.8, altitude=1.5, period=12.0),
    # "orbit me while I pace back and forth" -- includes direction reversals
    "orbit-pacing": lambda: Orbit(subject=Pacing(speed=0.35, extent=1.0), radius=0.8, altitude=1.5, period=12.0),
    # "follow me" -- hold station behind the subject instead of circling
    "follow-walking": lambda: FollowBehind(subject=Walking(speed=0.25)),
    "follow-walking-fast": lambda: FollowBehind(subject=Walking(speed=0.6)),
    "follow-pacing": lambda: FollowBehind(subject=Pacing(speed=0.35, extent=1.0)),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly a high-level command with a trained policy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--command", choices=list(COMMANDS), default="orbit")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--lead-time", type=float, default=0.0, help="seconds to look ahead on the trajectory, cancelling the policy's tracking lag (~0.75 works well)")
    parser.add_argument("--sweep", action="store_true", help="map the (speed, acceleration) envelope the policy can track, instead of flying one command")
    args = parser.parse_args()

    if args.sweep:
        feasibility_sweep(
            args.model,
            speeds=[0.25, 0.50, 0.75, 1.00, 1.25],
            accels=[0.15, 0.30, 0.50, 0.80, 1.20],
            lead_time=args.lead_time if args.lead_time else 0.75,
        )
    else:
        fly(args.model, COMMANDS[args.command](), duration=args.duration, gui=args.gui, lead_time=args.lead_time)
