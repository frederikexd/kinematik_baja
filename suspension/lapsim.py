# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Quasi-steady-state lap-time simulator — the number that actually wins.

WHY THIS MODULE IS THE KEYSTONE
-------------------------------
Everything else in KinematiK produces *intermediate* quantities: camber gain,
roll-centre height, a balance index, a max-lateral-g figure. Those are the right
things to compute, but they are not the language the design event, the team's
own decision-making, or the results sheet speaks. The currency that wins FSAE
dynamic events is **seconds**. A funded team (USC, etc.) runs a lap simulator so
that every geometry and setup change is judged by "how much faster does this make
the car round a lap?" — not by whether a curve looks nicer.

This module closes that final gap. It consumes the EXACT vehicle-dynamics + tire
stack the rest of the tool already builds (`VehicleDynamics`, the Pacejka tire,
the solved kinematics) and turns it into:

    - predicted event times for the three timed dynamic events
      (skidpad, 75 m acceleration, autocross/endurance lap),
    - a per-distance speed / lateral-g / longitudinal-g trace, and
    - a g-g-V envelope: the combined-acceleration capability the car has at each
      speed, which is the picture engineers use to see *where* on the track the
      car is grip-limited, power-limited, or brake-limited.

That converts "this setup is +0.03 in balance index" into "this setup is -0.4 s
on the endurance lap" — which is the sentence that moves you up the order and the
sentence a design judge wants to hear.

METHOD (and its honest limits)
------------------------------
Quasi-steady-state (QSS) point-mass simulation. The track is a sequence of
segments, each a straight or a constant-radius arc. For every segment we compute:

  1. the maximum cornering speed the tires allow on that arc, from the SAME
     load-sensitive Pacejka grip the rest of the tool uses (so a geometry/tire
     change that helps grip immediately shows up as lap time);
  2. a forward pass (traction- and power-limited acceleration out of each corner),
  3. a backward pass (braking into each corner),
  taking the minimum speed profile that satisfies all three — the standard QSS
  construction. Time is the integral of ds / v.

QSS captures the things that dominate an FSAE lap — corner-speed limits, the
accel/brake trade between corners, power and downforce — and is what most teams'
in-house sims actually are. It deliberately does NOT model transient yaw response,
combined-slip friction-circle usage during the brake/turn overlap, tire thermal
state, or driver line optimisation. The first two of those now have a companion
model — see `transient.py`, the explicit high-frequency time-step DAE solver,
which integrates yaw/sideslip, pitch/dive and kerb response millisecond by
millisecond. QSS remains the right tool for the lap-time number; the transient
solver is the right tool for the unsteady behaviour QSS assumes away. Tire
thermal state and a closed-loop racing line are still flagged in the UI so the
number is never oversold. A QSS lap time is a strong *relative*
comparator (setup A vs setup B) and a reasonable absolute estimate; treat the
ranking as trustworthy and the absolute seconds as ±a few percent.

EVERYTHING IS DEFENSIVE
-----------------------
A lap is thousands of evaluations; one bad geometry, a non-converging corner, a
divide-by-zero at v=0, or a NaN from a pathological tire fit must never take down
the session. Every public entry point here catches its own failures, substitutes
a safe physical default for the offending point, records a structured warning,
and keeps going. The functions return a result object carrying a `.warnings`
list; they never raise to the caller. The UI surfaces those warnings instead of
crashing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dynamics import VehicleDynamics, VehicleParams


# --------------------------------------------------------------------------- #
#  Aerodynamic & drivetrain parameters (point-mass level)
# --------------------------------------------------------------------------- #
@dataclass
class LapSimParams:
    """
    Vehicle parameters the lap sim needs ON TOP of what VehicleDynamics already
    carries. All have FSAE-representative defaults so the sim runs out of the box;
    every one is a knob a team can set from measured/spec data.

    Aero uses the standard 1/2 rho C v^2 form. Downforce is added to the vertical
    load the tires see, so more wing => higher corner speed AND more drag — the
    real trade. Set cl_a = cd_a = 0 to model a wingless car.
    """
    # Drivetrain / power
    power_w: float = 60_000.0        # peak power at the wheels, W (~80 hp FSAE cap region)
    drivetrain_eff: float = 0.90     # tractive efficiency multiplier (0..1)
    mass: float = 280.0              # total mass incl. driver, kg (mirrors VehicleParams)
    # Tractive grip: fraction of available lateral mu usable longitudinally for
    # accel out of a corner (RWD traction limit, simple friction-circle proxy).
    drive_grip_frac: float = 1.0
    # Braking
    brake_g: float = 1.6             # max decel the brakes+tires can sustain, g
    # Aerodynamics (force = 0.5 * rho * coeff_area * v^2)
    rho: float = 1.225               # air density, kg/m^3
    cl_a: float = 2.5                # downforce coefficient * frontal area, m^2 (ClA)
    cd_a: float = 1.2                # drag coefficient * frontal area, m^2 (CdA)
    # Rolling / driveline drag as an equivalent constant decel, g
    rolling_g: float = 0.015
    g: float = 9.81

    # Hard physical clamps so a wild UI entry can't poison the sim. These are
    # generous bounds, not tuning values; they only catch nonsense.
    V_MIN: float = 0.5               # m/s, floor speed to avoid divide-by-zero
    V_MAX: float = 60.0              # m/s (~216 km/h) — well above any FSAE car


# --------------------------------------------------------------------------- #
#  Track description
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One piece of track: a straight (radius=None/inf) or constant-radius arc."""
    length: float                     # m, arc/segment length along the path
    radius: float | None = None       # m, corner radius (None or <=0 => straight)
    name: str = ""

    def is_corner(self) -> bool:
        return self.radius is not None and np.isfinite(self.radius) and self.radius > 0


@dataclass
class Track:
    name: str
    segments: list[Segment]
    closed: bool = True               # closed circuit (lap) vs open (accel run)
    laps: int = 1                     # event laps (endurance multiplies a lap time)

    def total_length(self) -> float:
        return float(sum(max(s.length, 0.0) for s in self.segments))


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class LapResult:
    track_name: str
    ok: bool
    lap_time: float                   # s, single lap
    event_time: float                 # s, lap_time * laps (or single run for accel)
    avg_speed: float                  # m/s
    top_speed: float                  # m/s
    distance: np.ndarray              # m, cumulative
    speed: np.ndarray                 # m/s at each sample
    lat_g: np.ndarray                 # lateral g used at each sample
    long_g: np.ndarray                # longitudinal g (accel +, brake -) at each sample
    limit: list[str]                  # 'corner'/'accel'/'brake'/'power'/'straight' per sample
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @staticmethod
    def failed(track_name: str, warnings: list[str]) -> "LapResult":
        """A safe, inert result used when the sim cannot run at all."""
        z = np.zeros(1)
        return LapResult(
            track_name=track_name, ok=False, lap_time=float("nan"),
            event_time=float("nan"), avg_speed=0.0, top_speed=0.0,
            distance=z.copy(), speed=z.copy(), lat_g=z.copy(), long_g=z.copy(),
            limit=["n/a"], warnings=list(warnings), meta={},
        )


# --------------------------------------------------------------------------- #
#  The simulator
# --------------------------------------------------------------------------- #
class LapSimulator:
    """
    Quasi-steady-state point-mass lap sim wrapped around a VehicleDynamics object.

    It reuses VehicleDynamics for the load-sensitive cornering grip (so the same
    tire/geometry the rest of KinematiK shows is what drives lap time) and adds a
    point-mass longitudinal model (power, drag, downforce, braking) on top.

    Robustness contract: NOTHING in the public API raises. `simulate()` always
    returns a LapResult; on total failure it returns LapResult.failed(...) with
    the reason in `.warnings`. Per-point numerical problems degrade to a safe
    local default and add a warning rather than aborting the lap.
    """

    def __init__(self, veh: VehicleDynamics, params: LapSimParams | None = None,
                 ds: float = 1.0):
        self.veh = veh
        self.p = params or LapSimParams()
        # keep mass consistent with the vehicle model if it carries one
        try:
            if getattr(veh, "p", None) is not None and np.isfinite(veh.p.mass):
                self.p.mass = float(veh.p.mass)
        except Exception:
            pass
        self.ds = max(float(ds), 0.25)      # spatial step, m (clamped sane)
        self.warnings: list[str] = []
        self._mu_cache: dict[int, float] = {}

    # ---- internal: warn without ever raising ---------------------------- #
    def _warn(self, msg: str):
        if msg not in self.warnings:        # de-dup so a per-point fault floods once
            self.warnings.append(msg)

    # ---- effective grip used for cornering (load-sensitive) ------------- #
    def _corner_speed(self, radius: float) -> float:
        """
        Max steady speed on a corner of given radius, balancing required lateral
        acceleration against available grip INCLUDING the speed-dependent
        downforce that raises that grip. Solved by a short fixed-point iteration:
        v depends on grip, grip depends on load, load depends on v (aero).

        Returns a safe, finite speed; never raises. On any failure returns V_MIN.
        """
        p = self.p
        try:
            if not (np.isfinite(radius) and radius > 0):
                return p.V_MAX
            W = p.mass * p.g
            v = math.sqrt(max(0.6 * W / p.mass * radius, 1.0))  # rough seed (~0.6g)
            for _ in range(12):
                # downforce at current v adds to vertical load -> more grip
                Fz_aero = 0.5 * p.rho * p.cl_a * v * v
                mu = self._grip_mu(W + Fz_aero)
                if not np.isfinite(mu) or mu <= 0:
                    mu = 1.2                       # safe physical default
                    self._warn("Cornering grip returned a non-physical value at "
                               "some corner; substituted mu=1.2 there.")
                # available lateral accel = mu*(W+DF)/m ; required = v^2/R
                a_lat = mu * (W + Fz_aero) / p.mass
                v_new = math.sqrt(max(a_lat * radius, 1.0))
                if abs(v_new - v) < 0.05:
                    v = v_new
                    break
                v = 0.5 * (v + v_new)
            return float(min(max(v, p.V_MIN), p.V_MAX))
        except Exception as e:
            self._warn(f"Corner-speed solve failed on a segment ({type(e).__name__}); "
                       "used floor speed there.")
            return p.V_MIN

    def _grip_mu(self, Fz_total: float) -> float:
        """
        Effective lateral grip coefficient (lateral force capacity / vertical load)
        for the whole car at a given total vertical load, from the vehicle's tire
        model. Cached on rounded load. Falls back to a constant if anything fails.
        """
        key = int(max(Fz_total, 0.0) / 50.0)        # 50 N buckets
        if key in self._mu_cache:
            return self._mu_cache[key]
        try:
            # split load across 4 tires, ask the tire for per-tire capacity, sum.
            Fz_corner = max(Fz_total, 1.0) / 4.0
            tire = getattr(self.veh, "tire", None)
            if tire is not None:
                # use the camber the front axle actually runs, like the grip tab
                try:
                    gamma = self.veh._axle_camber_rad("front")
                except Exception:
                    gamma = 0.0
                Fy_total = 4.0 * tire.peak_force(Fz_corner, gamma)
                mu = Fy_total / max(Fz_total, 1.0)
            else:
                # legacy linear placeholder, mirrors dynamics.py
                vp = self.veh.p
                mu = vp.mu_peak - vp.tire_load_sens * Fz_corner
            mu = float(mu)
            if not np.isfinite(mu) or mu <= 0:
                raise ValueError("non-physical mu")
            mu = min(mu, 3.0)                        # clamp absurd values
        except Exception:
            mu = 1.4
            self._warn("Tire grip evaluation failed somewhere; used mu=1.4 fallback.")
        self._mu_cache[key] = mu
        return mu

    # ---- longitudinal capability --------------------------------------- #
    def _tractive_accel(self, v: float, mu: float) -> float:
        """Net forward accel (m/s^2) available at speed v: power & traction limited,
        minus drag, downforce-aware traction. Always finite."""
        p = self.p
        try:
            v = max(v, p.V_MIN)
            # power-limited tractive force
            F_power = p.power_w * p.drivetrain_eff / v
            # traction-limited force (rear/grip limited): mu * (W + downforce) * frac
            Fz_aero = 0.5 * p.rho * p.cl_a * v * v
            F_grip = p.drive_grip_frac * mu * (p.mass * p.g + Fz_aero)
            F_drive = min(F_power, F_grip)
            # resistances
            F_drag = 0.5 * p.rho * p.cd_a * v * v
            F_roll = p.rolling_g * p.mass * p.g
            a = (F_drive - F_drag - F_roll) / p.mass
            return float(a if np.isfinite(a) else 0.0)
        except Exception:
            self._warn("Tractive-accel evaluation failed at a point; used 0 there.")
            return 0.0

    def _braking_decel(self, v: float) -> float:
        """Max decel (positive m/s^2) at speed v, aided by aero drag+downforce."""
        p = self.p
        try:
            v = max(v, p.V_MIN)
            Fz_aero = 0.5 * p.rho * p.cl_a * v * v
            # tire braking limited by brake_g scaled by downforce-augmented load
            a_tire = p.brake_g * p.g * (1.0 + Fz_aero / max(p.mass * p.g, 1.0))
            F_drag = 0.5 * p.rho * p.cd_a * v * v
            a = a_tire + F_drag / p.mass + p.rolling_g * p.g
            return float(a if np.isfinite(a) and a > 0 else p.brake_g * p.g)
        except Exception:
            self._warn("Braking evaluation failed at a point; used nominal brake_g.")
            return p.brake_g * p.g

    # ---- the QSS construction ------------------------------------------ #
    def simulate(self, track: Track) -> LapResult:
        """
        Run the quasi-steady-state lap. Returns a LapResult always.

        Construction:
          - discretise the track into ds-length samples, each tagged with the local
            corner radius (inf on straights);
          - cap each sample at its corner-limited speed;
          - forward pass: limit accel to tractive capability;
          - backward pass: limit decel so we can brake to each cap (wrapping for a
            closed lap so the start-of-lap speed is consistent);
          - integrate dt = ds / v_avg over the lap.
        """
        self.warnings = []
        self._mu_cache.clear()
        try:
            samples = self._discretise(track)
            if samples is None or len(samples[0]) < 2:
                return LapResult.failed(
                    track.name,
                    self.warnings + ["Track had no usable length; nothing to simulate."])
            dist, radius = samples
            n = len(dist)
            p = self.p

            # 1) corner-limited speed cap at each sample
            v_cap = np.empty(n)
            for i in range(n):
                v_cap[i] = self._corner_speed(radius[i])
            v_cap = np.clip(np.nan_to_num(v_cap, nan=p.V_MIN,
                                          posinf=p.V_MAX, neginf=p.V_MIN),
                            p.V_MIN, p.V_MAX)

            # mu at each sample (for tractive limit) — reuse corner mu where cornering
            mu_local = np.array([
                self._grip_mu(p.mass * p.g + 0.5 * p.rho * p.cl_a * v * v)
                for v in v_cap])

            # 2) forward pass (acceleration limited)
            v_fwd = v_cap.copy()
            # seed start speed: for a closed lap start at the slowest corner so the
            # wrap is self-consistent; for an open run (accel) start from rest.
            if track.closed:
                v_fwd[0] = v_cap.min()
            else:
                v_fwd[0] = p.V_MIN
            for i in range(1, n):
                ds = max(dist[i] - dist[i - 1], 1e-6)
                a = self._tractive_accel(v_fwd[i - 1], mu_local[i - 1])
                v_next = math.sqrt(max(v_fwd[i - 1] ** 2 + 2.0 * a * ds, 0.0))
                v_fwd[i] = min(v_next, v_cap[i])

            # 3) backward pass (braking limited), wrap once for closed laps
            v_bwd = v_fwd.copy()
            rng = range(n - 2, -1, -1)
            for _wrap in range(2 if track.closed else 1):
                for i in rng:
                    ds = max(dist[i + 1] - dist[i], 1e-6)
                    a = self._braking_decel(v_bwd[i + 1])
                    v_prev = math.sqrt(max(v_bwd[i + 1] ** 2 + 2.0 * a * ds, 0.0))
                    v_bwd[i] = min(v_bwd[i], v_prev, v_cap[i])
                if track.closed:
                    # couple lap end to lap start
                    ds = max(dist[1] - dist[0], 1e-6)
                    a = self._braking_decel(v_bwd[0])
                    wrap_v = math.sqrt(max(v_bwd[0] ** 2 + 2.0 * a * ds, 0.0))
                    v_bwd[-1] = min(v_bwd[-1], wrap_v)

            v = np.clip(v_bwd, p.V_MIN, p.V_MAX)

            # 4) integrate time and assemble traces
            lap_time = 0.0
            long_g = np.zeros(n)
            lat_g = np.zeros(n)
            limit = ["straight"] * n
            for i in range(n):
                if i > 0:
                    ds = max(dist[i] - dist[i - 1], 1e-6)
                    v_avg = max(0.5 * (v[i] + v[i - 1]), p.V_MIN)
                    lap_time += ds / v_avg
                    long_g[i] = ((v[i] ** 2 - v[i - 1] ** 2) / (2.0 * ds)) / p.g
                # lateral g actually used here
                if np.isfinite(radius[i]) and radius[i] > 0:
                    lat_g[i] = (v[i] ** 2 / radius[i]) / p.g
                # classify what's limiting this sample
                if abs(v[i] - v_cap[i]) < 0.15 and lat_g[i] > 0.2:
                    limit[i] = "corner"
                elif long_g[i] > 0.05:
                    a_av = self._tractive_accel(v[i], mu_local[i])
                    F_power = p.power_w * p.drivetrain_eff / max(v[i], p.V_MIN)
                    Fz_aero = 0.5 * p.rho * p.cl_a * v[i] * v[i]
                    F_grip = p.drive_grip_frac * mu_local[i] * (p.mass * p.g + Fz_aero)
                    limit[i] = "power" if F_power < F_grip else "accel"
                elif long_g[i] < -0.05:
                    limit[i] = "brake"
                else:
                    limit[i] = "straight"

            if not np.isfinite(lap_time) or lap_time <= 0:
                return LapResult.failed(
                    track.name,
                    self.warnings + ["Lap-time integral was non-finite; "
                                     "geometry or parameters may be degenerate."])

            laps = max(int(getattr(track, "laps", 1) or 1), 1)
            event_time = lap_time * laps
            total_len = dist[-1] - dist[0]
            avg_speed = total_len / lap_time if lap_time > 0 else 0.0

            # flag an implausibly slow lap (e.g. zero power / no grip) rather than
            # reporting a huge time as if it were a real result.
            if avg_speed < 1.0 or lap_time > 5.0 * max(total_len, 1.0):
                self._warn("Predicted pace is implausibly slow — check power, mass, "
                           "and aero inputs; the car barely accelerates with these "
                           "values. Lap time shown is not trustworthy.")

            try:
                grip_name = (self.veh.grip_model_name()
                             if hasattr(self.veh, "grip_model_name") else "unknown")
            except Exception:
                grip_name = "unknown"

            return LapResult(
                track_name=track.name, ok=True,
                lap_time=float(lap_time), event_time=float(event_time),
                avg_speed=float(avg_speed), top_speed=float(np.max(v)),
                distance=dist, speed=v, lat_g=lat_g, long_g=long_g, limit=limit,
                warnings=list(self.warnings),
                meta=dict(n_samples=n, track_length=float(total_len),
                          laps=laps, grip_model=grip_name),
            )
        except Exception as e:
            # absolute backstop — a lap must never crash the session
            return LapResult.failed(
                track.name,
                self.warnings + [f"Lap simulation failed entirely "
                                 f"({type(e).__name__}: {e}); returned a safe empty "
                                 f"result so the app keeps running."])

    # ---- discretisation ------------------------------------------------- #
    def _discretise(self, track: Track):
        """Turn segments into per-sample (cumulative distance, local radius) arrays.
        Returns (dist, radius) or None if the track has no length."""
        try:
            dlist, rlist = [0.0], [float("inf")]
            d = 0.0
            for seg in track.segments:
                L = max(float(getattr(seg, "length", 0.0)), 0.0)
                if L <= 0:
                    continue
                r = seg.radius if seg.is_corner() else float("inf")
                nstep = max(int(round(L / self.ds)), 1)
                step = L / nstep
                for _ in range(nstep):
                    d += step
                    dlist.append(d)
                    rlist.append(r)
            if len(dlist) < 2:
                return None
            return np.array(dlist), np.array(rlist)
        except Exception as e:
            self._warn(f"Track discretisation failed ({type(e).__name__}); "
                       "no samples produced.")
            return None

    # ---- g-g-V envelope ------------------------------------------------- #
    def gg_v_envelope(self, speeds=None):
        """
        Combined-acceleration envelope vs speed: for a set of speeds, the max
        lateral g (pure cornering), max forward g (pure accel) and max braking g
        (pure brake) the car can make. This is the picture engineers read to see
        the car's capability at each speed. Returns a dict of arrays; never raises.
        """
        p = self.p
        try:
            if speeds is None:
                speeds = np.linspace(5.0, min(p.V_MAX, 40.0), 24)
            speeds = np.asarray(speeds, float)
            lat, acc, brk = [], [], []
            for v in speeds:
                v = float(max(v, p.V_MIN))
                Fz_aero = 0.5 * p.rho * p.cl_a * v * v
                mu = self._grip_mu(p.mass * p.g + Fz_aero)
                lat.append(mu * (p.mass * p.g + Fz_aero) / p.mass / p.g)
                acc.append(max(self._tractive_accel(v, mu) / p.g, 0.0))
                brk.append(self._braking_decel(v) / p.g)
            return dict(speed=speeds,
                        lat_g=np.array(lat),
                        accel_g=np.array(acc),
                        brake_g=np.array(brk),
                        warnings=list(self.warnings))
        except Exception as e:
            self._warn(f"g-g-V envelope failed ({type(e).__name__}); empty envelope.")
            z = np.zeros(1)
            return dict(speed=z, lat_g=z, accel_g=z, brake_g=z,
                        warnings=list(self.warnings))


# --------------------------------------------------------------------------- #
#  Standard FSAE event tracks
# --------------------------------------------------------------------------- #
def skidpad_track() -> Track:
    """
    FSAE skidpad: two 18.25 m-diameter circles in a figure-of-eight; the timed
    result is one full circle on each side. Modelled here as the timed constant-
    radius circle (R = 9.125 m), which is what sets the time. The event score is
    driven by this single steady cornering speed — exactly the load-sensitive grip
    KinematiK computes — so it's the cleanest possible check of the grip model.
    """
    R = 9.125
    circumference = 2.0 * math.pi * R
    return Track(
        name="Skidpad (timed circle)",
        segments=[Segment(length=circumference, radius=R, name="skidpad circle")],
        closed=True, laps=1,
    )


def acceleration_track() -> Track:
    """FSAE acceleration: a 75 m straight-line sprint from a standing start."""
    return Track(
        name="Acceleration (75 m)",
        segments=[Segment(length=75.0, radius=None, name="75 m straight")],
        closed=False, laps=1,
    )


def autocross_track(laps: int = 1) -> Track:
    """
    A representative ~800 m autocross/endurance lap built from FSAE-legal features:
    the rules cap straights ~60 m, constant-radius corners 9–45 m radius, hairpins
    down to ~4.5 m inside radius, and slaloms. This is a *generic* layout standing
    in for the real event map (which changes yearly) — good for relative setup
    comparison and a realistic absolute lap. Swap in your own segment list to model
    a specific course. Endurance = the same lap repeated; pass laps>1.
    """
    S = Segment
    segs = [
        S(60.0, None, "main straight"),
        S(14.0, 9.0, "turn 1 (medium)"),
        S(25.0, None, "short chute"),
        S(20.0, 18.0, "sweeper"),
        S(30.0, None, "back straight"),
        S(7.0, 4.5, "hairpin"),
        S(18.0, None, "exit chute"),
        S(12.0, 8.0, "turn 4"),
        S(10.0, None, "link"),
        S(16.0, 12.0, "turn 5"),
        S(40.0, None, "slalom straight"),
        S(6.0, 6.0, "slalom 1"),
        S(8.0, 6.0, "slalom 2"),
        S(6.0, 6.0, "slalom 3"),
        S(28.0, None, "run to constant"),
        S(35.0, 22.0, "constant-radius arc"),
        S(20.0, None, "approach"),
        S(8.0, 5.0, "tight left"),
        S(45.0, None, "back chute"),
        S(15.0, 10.0, "turn 9"),
        S(20.0, None, "final straight"),
        S(7.0, 4.5, "final hairpin"),
        S(30.0, None, "start/finish straight"),
    ]
    return Track(name="Autocross / Endurance lap (generic)",
                 segments=segs, closed=True, laps=max(int(laps), 1))


def standard_events() -> dict:
    """The three timed dynamic events as ready-to-run tracks."""
    return {
        "skidpad": skidpad_track(),
        "acceleration": acceleration_track(),
        "autocross": autocross_track(laps=1),
    }


# --------------------------------------------------------------------------- #
#  One-call convenience: simulate all standard events for a vehicle
# --------------------------------------------------------------------------- #
def simulate_events(veh: VehicleDynamics, params: LapSimParams | None = None,
                    endurance_laps: int = 1) -> dict:
    """
    Run all three standard FSAE timed events for a given vehicle model and return
    a dict of LapResult. Never raises; a failed event yields a LapResult.failed
    carrying its warning, so the caller can render the others normally.
    """
    out: dict[str, LapResult] = {}
    sim = LapSimulator(veh, params=params)
    try:
        out["skidpad"] = sim.simulate(skidpad_track())
    except Exception as e:
        out["skidpad"] = LapResult.failed("Skidpad", [f"skidpad failed: {e}"])
    try:
        out["acceleration"] = sim.simulate(acceleration_track())
    except Exception as e:
        out["acceleration"] = LapResult.failed("Acceleration", [f"accel failed: {e}"])
    try:
        out["autocross"] = sim.simulate(autocross_track(laps=max(int(endurance_laps), 1)))
    except Exception as e:
        out["autocross"] = LapResult.failed("Autocross", [f"autocross failed: {e}"])
    return out


# --------------------------------------------------------------------------- #
#  FSAE points models (so a lap-time delta becomes a POINTS delta)
# --------------------------------------------------------------------------- #
def event_points(event: str, your_time: float,
                 best_time: float | None = None,
                 max_time: float | None = None) -> float:
    """
    Approximate FSAE Rules points for a timed dynamic event given your time and a
    reference best (Tmin) and worst-scoring (Tmax) time. Uses the standard FSAE
    formulae shape (acceleration/skidpad on a ratio, autocross/endurance similar)
    so a setup that's 0.3 s faster turns into an estimated points gain — the thing
    that actually decides standings. Returns a safe 0.0 on bad input; never raises.

    These are the published-form scoring curves; the exact Tmax/Tmin come from the
    event each year, so pass them if you have them. Defaults use typical multiples.
    """
    try:
        if not np.isfinite(your_time) or your_time <= 0:
            return 0.0
        ev = event.lower()
        if ev.startswith("accel"):
            tmin = best_time or your_time
            tmax = max_time or 1.5 * tmin
            pts_max, pts_min = 100.0, 4.5
        elif ev.startswith("skid"):
            tmin = best_time or your_time
            tmax = max_time or 1.25 * tmin
            pts_max, pts_min = 75.0, 3.5
        else:  # autocross / endurance family
            tmin = best_time or your_time
            tmax = max_time or 1.45 * tmin
            pts_max, pts_min = 125.0, 6.5
        if tmax <= tmin:
            return pts_max
        # standard shape: full points at Tmin, min points at/after Tmax
        frac = (tmax - your_time) / (tmax - tmin)
        frac = min(max(frac, 0.0), 1.0)
        return float(pts_min + (pts_max - pts_min) * frac)
    except Exception:
        return 0.0
