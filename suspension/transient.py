# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Explicit, high-frequency transient time-step solver — the unsteady half of the lap.

WHY THIS MODULE EXISTS
----------------------
`lapsim.py` is quasi-steady-state (QSS): it assumes the car sits at a balanced
equilibrium at every point on the racing line, solves a speed profile from the
grip envelope, and integrates ds/v. That is the right tool for the question
"how fast can this setup get round a lap?" and it is honest about what it throws
away — it says so in its own docstring: no transient yaw, no friction-circle
overlap during brake/turn, no tyre lag, no pitch/dive dynamics.

This module computes exactly the things QSS discards. It is an explicit,
fixed-step transient integrator that advances the car's full dynamic state
millisecond by millisecond, so it can show the behaviour QSS cannot:

    * turn-in lag and yaw overshoot (the relaxation-length tyre lag finally used),
    * SNAP-OVERSTEER onset and recovery (the rear transiently exceeding grip,
      the yaw rate diverging, and a countersteer pulling it back),
    * PITCH and DIVE oscillation through a brake-to-throttle transition (the
      sprung mass rocking on the springs, the digressive damper settling it),
    * KERB / CURB strikes (the unsprung mass hopping at ~15-20 Hz, the contact
      load spiking and momentarily dropping to zero — wheel lift),
    * the chaotic, unsettled load history of a car BEFORE it reaches the steady
      state QSS assumes it is always already in.

It is built on the SAME primitives the rest of KinematiK already trusts and
tests as standalone parts — the fitted `PacejkaLateral` tyre, the
`CombinedSlipTire` friction ellipse, the `relaxation_length` / lag law, and the
bilinear-digressive `DamperCurve`. Those modules say, in their own docstrings,
that they are "the building block for the transient model on the roadmap." This
is that model.

THE DAE, AND HOW IT IS SOLVED
-----------------------------
The car is an index-1 differential-algebraic system: a set of ordinary
differential equations for the dynamic states (the velocities, the ride
heights, the lagged slip angles) coupled to a set of ALGEBRAIC equations that
must hold exactly at every instant — the tyre vertical loads, the slip-angle
definitions, the Magic-Formula force law, and the friction-ellipse coupling.

We realise it in MINIMAL COORDINATES: the constraints that would otherwise need
Lagrange multipliers (the suspension kinematics) are pre-reduced to wheel-rate
springs and motion-ratioed dampers, so the remaining algebraic block is an
EXPLICIT function of the state — no implicit constraint solve per step. The
integrator therefore evaluates the algebraic block (loads -> slips -> forces ->
accelerations) and then advances the differential block with classical
explicit Runge-Kutta (RK4) at a fixed millisecond step, with optional
sub-stepping for the stiff tyre-vertical mode. This is the standard, real-time-
capable way to integrate an automotive index-1 DAE; it enforces the algebraic
constraints exactly every step while staying fully explicit.

Honest scope (same contract as the rest of the repo): this models the
DOMINANT transient modes — planar yaw/sideslip, heave/pitch/roll of the sprung
mass, four unsprung vertical hops, and lateral tyre relaxation. It does NOT
spin up four wheel-speed states for full longitudinal slip-ratio dynamics
(longitudinal force is demanded and friction-ellipse-limited, which captures the
load-transfer transient that drives pitch/dive); it does not model tyre thermal
state; and it does not solve the closed kinematic loops with multipliers. Those
are flagged, not faked.

EVERYTHING IS DEFENSIVE
-----------------------
A transient run is tens of thousands of force evaluations; one NaN from a
pathological tyre or a wild input must never take down the session. Every public
entry point catches its own failures, clamps the state to physical bounds each
step, and returns a result object carrying a `.warnings` list rather than
raising. A blown-up integration is reported as a flagged result, not a crash.

Sign conventions (right-handed, SAE-ish, documented once):
    x : body-forward (+),   y : body-left (+),   z : up (+)
    u : longitudinal velocity,  v : lateral velocity,  r : yaw rate (+ = left turn)
    pitch  theta : + = nose UP   (braking drives it negative -> nose dives)
    roll   phi   : + = body leans LEFT (left side down)
    suspension deflection : + = bump (compression)
    damper shaft velocity : + = bump (matches damper.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dynamics import VehicleDynamics, VehicleParams
from .tiremodel import (PacejkaLateral, CombinedSlipTire, default_tire,
                        default_combined_tire, relaxation_length)
from .damper import DamperCurve, default_damper


# Corner index convention used everywhere in this module.
FL, FR, RL, RR = 0, 1, 2, 3
CORNER_NAMES = ("FL", "FR", "RL", "RR")


# --------------------------------------------------------------------------- #
#  Parameters the transient model needs ON TOP of VehicleParams
# --------------------------------------------------------------------------- #
@dataclass
class TransientParams:
    """
    Everything the unsteady model needs that the steady model didn't. All have
    FSAE-representative defaults so the solver runs out of the box; every one is a
    knob a team sets from measured/spec data. Build from a VehicleParams with
    `from_vehicle` so mass/geometry stay consistent with the rest of the tool.
    """
    # Masses (kg)
    mass: float = 280.0                 # total incl. driver
    unsprung_mass_corner: float = 11.0  # per corner (wheel+upright+brake+arm share)
    # Inertias (kg·m²) about the CG.  Defaults are FSAE-representative; override
    # from a CAD inertia tensor when you have one.
    izz: float = 0.0                    # yaw; 0 => auto = mass * a * b
    ixx: float = 35.0                   # roll
    iyy: float = 90.0                   # pitch
    # Geometry (m) — mirror VehicleParams (converted from mm)
    wheelbase: float = 1.55
    track_front: float = 1.20
    track_rear: float = 1.18
    cg_height: float = 0.30
    weight_dist_front: float = 0.47
    # Roll / pitch axis heights (m, above ground) — the inertial moment arms.
    # Default to small positive roll-centre heights (typical FSAE) and a pitch
    # centre near ground.  When a VehicleDynamics with geometry is supplied these
    # are replaced by the solved roll-centre heights.
    roll_axis_height: float = 0.05
    pitch_axis_height: float = 0.0
    # Wheel-rate springs (N/m at the wheel) and ARB (N·m/rad of body roll).
    # When use_spring_rates is set on the vehicle these are derived from the
    # spring rate through the motion ratio; otherwise from these directly.
    k_wheel_front: float = 35_000.0     # ~35 N/mm wheel rate
    k_wheel_rear: float = 35_000.0
    arb_front: float = 0.0              # N·m per rad of body roll, at the axle
    arb_rear: float = 0.0
    motion_ratio_front: float = 1.0     # spring/wheel; folds the damper to the wheel
    motion_ratio_rear: float = 1.0
    # Tyre vertical (radial) stiffness & damping
    k_tire: float = 130_000.0           # N/m (~130 N/mm, typical FSAE radial)
    c_tire: float = 500.0               # N·s/m, small structural damping
    # Aerodynamics (force = 0.5*rho*coeff_area*v²)
    rho: float = 1.225
    cl_a: float = 2.5                   # downforce ClA, m²
    cd_a: float = 1.2                   # drag CdA, m²
    aero_balance_front: float = 0.45    # fraction of downforce on the front axle
    # Drivetrain / brakes
    power_w: float = 60_000.0
    drivetrain_eff: float = 0.90
    drive_bias_rear: float = 1.0        # 1.0 = RWD
    brake_bias_front: float = 0.62
    brake_g_max: float = 1.6            # tyre+brake decel ceiling, g
    rolling_g: float = 0.015
    g: float = 9.81
    # Numerics
    dt: float = 1.0e-3                  # base step, s (millisecond)
    substeps: int = 1                   # extra RK4 sub-steps per logged step
    u_min: float = 0.3                  # m/s floor to avoid divide-by-zero
    # Hard physical clamps (catch nonsense, not tuning)
    V_MAX: float = 65.0

    @property
    def a(self) -> float:               # CG -> front axle (m)
        return (1.0 - self.weight_dist_front) * self.wheelbase

    @property
    def b(self) -> float:               # CG -> rear axle (m)
        return self.weight_dist_front * self.wheelbase

    @property
    def m_unsprung_total(self) -> float:
        return 4.0 * self.unsprung_mass_corner

    @property
    def m_sprung(self) -> float:
        return max(self.mass - self.m_unsprung_total, 1.0)

    def izz_eff(self) -> float:
        return self.izz if self.izz > 0 else self.mass * self.a * self.b

    # Per-corner longitudinal (x, forward+) and lateral (y, left+) positions (m).
    def corner_xy(self):
        a, b = self.a, self.b
        tf, tr = self.track_front / 2.0, self.track_rear / 2.0
        # FL, FR, RL, RR
        x = np.array([a, a, -b, -b])
        y = np.array([tf, -tf, tr, -tr])
        return x, y

    @staticmethod
    def from_vehicle(veh: VehicleDynamics | None,
                     base: "TransientParams | None" = None) -> "TransientParams":
        """
        Build TransientParams consistent with a VehicleDynamics/VehicleParams.
        Pulls mass, geometry, weight distribution and — when geometry+spring-rate
        mode are present — the wheel rates, ARBs, motion ratios and solved roll-
        centre heights, so the transient model runs on the SAME setup the rest of
        the tool shows. Falls back to defaults for anything unavailable. Never
        raises.
        """
        tp = base or TransientParams()
        if veh is None:
            return tp
        try:
            vp: VehicleParams = veh.p
            tp.mass = float(vp.mass)
            tp.wheelbase = float(vp.wheelbase) / 1000.0
            tp.track_front = float(vp.track_front) / 1000.0
            tp.track_rear = float(vp.track_rear) / 1000.0
            tp.cg_height = float(vp.cg_height) / 1000.0
            tp.weight_dist_front = float(vp.weight_dist_front)
            tp.g = float(getattr(vp, "g", 9.81))
        except Exception:
            return tp
        # Motion ratios from real rocker geometry when available.
        try:
            mf, mr = veh.motion_ratios()
            if np.isfinite(mf) and mf > 0:
                tp.motion_ratio_front = float(mf)
            if np.isfinite(mr) and mr > 0:
                tp.motion_ratio_rear = float(mr)
        except Exception:
            pass
        # Wheel rates: k_wheel = k_spring * MR² when spring-rate mode is on.
        try:
            if getattr(vp, "use_spring_rates", False):
                tp.k_wheel_front = float(vp.spring_rate_front) * 1000.0 * tp.motion_ratio_front ** 2
                tp.k_wheel_rear = float(vp.spring_rate_rear) * 1000.0 * tp.motion_ratio_rear ** 2
        except Exception:
            pass
        # Solved roll-centre heights (m) as the roll moment arm reference.
        try:
            rc_f = veh.roll_center_height(veh.front_kin, vp.track_front) if veh.front_kin else None
            rc_r = veh.roll_center_height(veh.rear_kin, vp.track_rear) if veh.rear_kin else None
            heights = [h for h in (rc_f, rc_r) if h is not None and np.isfinite(h)]
            if heights:
                tp.roll_axis_height = float(np.mean(heights)) / 1000.0
        except Exception:
            pass
        return tp


# --------------------------------------------------------------------------- #
#  Driver inputs and road (curb) profile
# --------------------------------------------------------------------------- #
@dataclass
class DriverInput:
    """
    Time-varying control inputs. Each is a callable t(s) -> value, so you can
    prescribe an open-loop manoeuvre (step steer, brake-then-throttle) or, for the
    closed-loop lap, feed controller outputs through `set_runtime`.

        steer    : road-wheel steer angle at the FRONT wheels, rad (+ = left)
        throttle : [0,1] fraction of available tractive force
        brake    : [0,1] fraction of max braking
    """
    steer: Callable[[float], float] = lambda t: 0.0
    throttle: Callable[[float], float] = lambda t: 0.0
    brake: Callable[[float], float] = lambda t: 0.0

    def sample(self, t: float, state: dict | None = None):
        # each control may be f(t) OR f(t, state) — the latter enables closed-loop
        # control (e.g. a countersteer that reacts to sideslip). Try the richer
        # signature first, fall back to time-only.
        def _call(fn, default):
            try:
                try:
                    return float(fn(t, state))
                except TypeError:
                    return float(fn(t))
            except Exception:
                return default
        d = _call(self.steer, 0.0)
        thr = min(max(_call(self.throttle, 0.0), 0.0), 1.0)
        brk = min(max(_call(self.brake, 0.0), 0.0), 1.0)
        return d, thr, brk


@dataclass
class RoadInput:
    """
    Per-corner vertical road height z_road(t) (m) and its rate, for kerb/curb
    strikes and bumps. Default flat. `z` is a callable t -> (4,) array (FL,FR,RL,RR).
    """
    z: Callable[[float], np.ndarray] = lambda t: np.zeros(4)
    zdot: Callable[[float], np.ndarray] | None = None

    def sample(self, t: float):
        try:
            zr = np.asarray(self.z(t), float).reshape(4)
        except Exception:
            zr = np.zeros(4)
        if self.zdot is not None:
            try:
                zd = np.asarray(self.zdot(t), float).reshape(4)
            except Exception:
                zd = np.zeros(4)
        else:
            # finite-difference the road if no analytic rate supplied
            h = 1e-4
            try:
                zd = (np.asarray(self.z(t + h), float).reshape(4) - zr) / h
            except Exception:
                zd = np.zeros(4)
        return zr, zd


# --------------------------------------------------------------------------- #
#  State vector layout (24 states)
# --------------------------------------------------------------------------- #
#   0  u        longitudinal velocity (body), m/s
#   1  v        lateral velocity (body), m/s
#   2  r        yaw rate, rad/s
#   3  X        global x, m
#   4  Y        global y, m
#   5  psi      global heading, rad
#   6  z_s      sprung heave (from static), m
#   7  zd_s     heave velocity, m/s
#   8  phi      roll, rad
#   9  phid     roll rate, rad/s
#  10  theta    pitch, rad
#  11  thetad   pitch rate, rad/s
#  12-15 z_u[4] unsprung vertical (from static), m
#  16-19 zd_u[4] unsprung vertical velocity, m/s
#  20-23 alpha_lag[4]  lagged slip angle, rad
N_STATES = 24
IU, IV, IR, IX, IY, IPSI = 0, 1, 2, 3, 4, 5
IZS, IZDS, IPHI, IPHID, ITH, ITHD = 6, 7, 8, 9, 10, 11
IZU = slice(12, 16)
IZDU = slice(16, 20)
IAL = slice(20, 24)


@dataclass
class TransientResult:
    """Time-domain history of a transient run. Arrays are length n_steps+1."""
    ok: bool
    t: np.ndarray
    # planar
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    beta: np.ndarray          # body sideslip angle, rad
    ax: np.ndarray            # longitudinal accel, g
    ay: np.ndarray            # lateral accel, g
    X: np.ndarray
    Y: np.ndarray
    psi: np.ndarray
    # ride
    heave: np.ndarray         # m
    pitch: np.ndarray         # rad
    roll: np.ndarray          # rad
    # per-corner (n x 4)
    Fz: np.ndarray            # contact vertical load, N
    Fy: np.ndarray            # lateral force, N
    Fx: np.ndarray            # longitudinal force, N
    alpha: np.ndarray         # lagged slip angle, rad
    susp_vel: np.ndarray      # suspension (wheel) velocity, m/s (+bump)
    # inputs (logged)
    steer: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @staticmethod
    def failed(warnings: list[str]) -> "TransientResult":
        z = np.zeros(1)
        z4 = np.zeros((1, 4))
        return TransientResult(
            ok=False, t=z.copy(), u=z.copy(), v=z.copy(), r=z.copy(),
            beta=z.copy(), ax=z.copy(), ay=z.copy(), X=z.copy(), Y=z.copy(),
            psi=z.copy(), heave=z.copy(), pitch=z.copy(), roll=z.copy(),
            Fz=z4.copy(), Fy=z4.copy(), Fx=z4.copy(), alpha=z4.copy(),
            susp_vel=z4.copy(), steer=z.copy(), throttle=z.copy(),
            brake=z.copy(), warnings=list(warnings), meta={})

    def summary(self) -> dict:
        """A few headline transient metrics, computed defensively."""
        out = {}
        try:
            out["peak_yaw_rate_deg_s"] = float(np.degrees(np.max(np.abs(self.r))))
            out["peak_sideslip_deg"] = float(np.degrees(np.max(np.abs(self.beta))))
            out["peak_pitch_deg"] = float(np.degrees(np.max(np.abs(self.pitch))))
            out["peak_roll_deg"] = float(np.degrees(np.max(np.abs(self.roll))))
            out["max_Fz_N"] = float(np.max(self.Fz))
            out["min_Fz_N"] = float(np.min(self.Fz))
            out["wheel_lift"] = bool(np.any(self.Fz < 1.0))
            out["peak_ay_g"] = float(np.max(np.abs(self.ay)))
            out["peak_ax_g"] = float(np.max(np.abs(self.ax)))
        except Exception:
            pass
        return out


# --------------------------------------------------------------------------- #
#  The solver
# --------------------------------------------------------------------------- #
class TransientSolver:
    """
    Explicit fixed-step (RK4) transient integrator for the 24-state vehicle DAE.

    Robustness contract (same as lapsim): NOTHING in the public API raises.
    `run()` always returns a TransientResult; on blow-up it returns a flagged
    result with the reason in `.warnings`. Per-step numerical faults clamp to a
    safe physical state and add a warning rather than aborting.
    """

    def __init__(self, veh: VehicleDynamics | None = None,
                 params: TransientParams | None = None,
                 tire: CombinedSlipTire | None = None,
                 damper: DamperCurve | None = None):
        self.veh = veh
        self.p = TransientParams.from_vehicle(veh, params) if params is None \
            else params
        # tyre: prefer a combined-slip wrapper so corner entry/exit couples
        if tire is not None:
            self.tire = tire
        elif veh is not None and getattr(veh, "tire", None) is not None:
            self.tire = default_combined_tire(veh.tire)
        else:
            self.tire = default_combined_tire(default_tire())
        self.damper = damper or default_damper()
        self.warnings: list[str] = []
        self._peak_cache: dict = {}
        # camber (rad) per axle from the vehicle, fixed during a run
        self._cam_f = self._axle_cam("front")
        self._cam_r = self._axle_cam("rear")

    # ---- helpers -------------------------------------------------------- #
    def _warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _axle_cam(self, axle: str) -> float:
        try:
            if self.veh is not None:
                return float(self.veh._axle_camber_rad(axle))
        except Exception:
            pass
        return math.radians(1.5)

    def _peak_force(self, fz: float, axle_front: bool) -> float:
        """
        Cached peak lateral force (N) at load `fz` for an axle, memoised on a
        coarse load bucket. peak_force() runs a 121-point slip sweep internally, so
        without this cache the inner DAE loop would do millions of Pacejka calls a
        run. Camber is fixed per axle during a run, so (bucket, axle) keys it. The
        |camber| magnitude is what sets peak grip, so the antisymmetric L/R sign
        used for the force itself doesn't change the peak — one value per axle.
        """
        bucket = int(max(fz, 0.0) / 25.0)        # 25 N buckets
        key = (bucket, axle_front)
        v = self._peak_cache.get(key)
        if v is not None:
            return v
        gamma = self._cam_f if axle_front else self._cam_r
        try:
            v = float(self.tire.lateral.peak_force(max(fz, 1.0), abs(gamma)))
            if not np.isfinite(v) or v <= 0:
                v = 1.2 * max(fz, 1.0)
        except Exception:
            v = 1.2 * max(fz, 1.0)
            self._warn("Tyre peak-force evaluation failed; used mu=1.2 fallback.")
        self._peak_cache[key] = v
        return v

    def static_corner_loads(self) -> np.ndarray:
        """Static vertical load per corner (N), FL,FR,RL,RR."""
        p = self.p
        W = p.mass * p.g
        Wf = W * p.weight_dist_front
        Wr = W - Wf
        return np.array([Wf / 2, Wf / 2, Wr / 2, Wr / 2])

    def initial_state(self, u0: float = 10.0) -> np.ndarray:
        """State at static ride height, travelling straight at u0."""
        y = np.zeros(N_STATES)
        y[IU] = max(float(u0), self.p.u_min)
        return y

    # ---- the algebraic block: loads, slips, tyre forces ----------------- #
    def algebraic(self, t: float, y: np.ndarray,
                  driver: DriverInput, road: RoadInput) -> dict:
        """
        Evaluate every algebraic quantity the DAE constrains at the current state:
        per-corner vertical load, slip angle, and tyre Fx/Fy. Returns a dict; this
        is also what the logger records. Pure (no mutation of y). Never raises —
        clamps to safe values and flags instead.
        """
        p = self.p
        x_i, y_i = p.corner_xy()
        u = y[IU]; v = y[IV]; r = y[IR]
        z_s = y[IZS]; zd_s = y[IZDS]
        phi = y[IPHI]; phid = y[IPHID]
        th = y[ITH]; thd = y[ITHD]
        z_u = y[IZU]; zd_u = y[IZDU]
        al = y[IAL]
        beta = math.atan2(v, max(abs(u), p.u_min))
        d_steer, thr, brk = driver.sample(t, dict(t=t, u=u, v=v, r=r, beta=beta))
        z_road, zd_road = road.sample(t)

        # --- sprung-corner vertical position & velocity (small angle) ---
        zc = z_s + x_i * th - y_i * phi
        zcd = zd_s + x_i * thd - y_i * phid

        # --- suspension deflection (+bump) and wheel velocity ---
        # delta = how much the wheel has risen toward the body vs static
        delta = z_u - zc
        delta_vel = zd_u - zcd            # +bump (wheel approaching body)

        # --- spring + damper + ARB force at each wheel (N, + pushes body up) --
        k_wheel = np.array([p.k_wheel_front, p.k_wheel_front,
                            p.k_wheel_rear, p.k_wheel_rear])
        mr = np.array([p.motion_ratio_front, p.motion_ratio_front,
                       p.motion_ratio_rear, p.motion_ratio_rear])
        F_spring = k_wheel * delta
        # damper: shaft velocity = wheel velocity * MR; force at wheel = F_shaft*MR
        try:
            shaft_v = delta_vel * mr
            F_damp = self.damper.force(shaft_v) * mr
            F_damp = np.nan_to_num(np.asarray(F_damp, float), nan=0.0)
        except Exception:
            F_damp = np.zeros(4)
            self._warn("Damper force evaluation failed; used zero damping there.")
        # anti-roll bar: opposes the body roll (couples L/R of each axle)
        # body roll phi (+left down) -> ARB adds restoring at each wheel
        arb_axle = np.array([p.arb_front, p.arb_front, p.arb_rear, p.arb_rear])
        # convert N·m/rad of roll to a wheel force via half-track lever
        tw = np.array([p.track_front, p.track_front, p.track_rear, p.track_rear])
        # +left-down roll pushes left wheels up, right wheels down (sign via y_i)
        F_arb = arb_axle * phi / np.maximum(tw / 2.0, 1e-3) * np.sign(y_i)
        F_susp = F_spring + F_damp + F_arb     # on sprung mass (+up); reaction on wheel

        # --- tyre vertical (radial) force at the contact patch ---
        # static + tyre spring on (road - wheel) + tyre damping + aero downforce
        Fz_static = self.static_corner_loads()
        aero_DF = 0.5 * p.rho * p.cl_a * u * u
        DF_front = aero_DF * p.aero_balance_front / 2.0
        DF_rear = aero_DF * (1.0 - p.aero_balance_front) / 2.0
        aero_corner = np.array([DF_front, DF_front, DF_rear, DF_rear])
        F_tire = (Fz_static
                  + p.k_tire * (z_road - z_u)
                  + p.c_tire * (zd_road - zd_u)
                  + aero_corner)
        Fz = np.maximum(F_tire, 0.0)           # tyre can't pull the road

        # --- slip angles per corner (with lateral relaxation already in al) ---
        vx_w = u - r * y_i
        vy_w = v + r * x_i
        steer_arr = np.array([d_steer, d_steer, 0.0, 0.0])
        # slip angle, SAE-consistent with this tyre's Fy sign (positive alpha
        # gives negative Fy here), so alpha = atan2(v_y, v_x) - steer makes a
        # left steer (delta>0) produce a left (+y) force and a left (+) yaw.
        vx_safe = np.where(np.abs(vx_w) < p.u_min, p.u_min * np.sign(vx_w + 1e-9), vx_w)
        alpha_target = np.arctan2(vy_w, vx_safe) - steer_arr
        alpha_target = np.nan_to_num(alpha_target, nan=0.0)
        # The LAGGED slip is the integrated state `al`; we use it for the force.
        # Inclination angle is ANTISYMMETRIC left/right: negative static camber
        # leans both tops inboard, so the camber-thrust (odd) terms point toward
        # the centreline on each side and CANCEL in a straight line, while the
        # grip-magnitude (even) term is preserved. Passing |cam| to all four would
        # synthesise a spurious straight-line side force and yaw — the sign(y_i)
        # factor is what removes it.
        cam = np.array([self._cam_f, self._cam_f,
                        self._cam_r, self._cam_r]) * np.sign(y_i)

        # --- longitudinal force demand (drive + brake), per corner ---
        # power/traction-limited drive, split rear by drive_bias_rear
        u_eff = max(u, p.u_min)
        F_power = p.power_w * p.drivetrain_eff / u_eff
        F_drive_total = thr * F_power
        drive_split = np.array([(1 - p.drive_bias_rear) / 2.0,
                                (1 - p.drive_bias_rear) / 2.0,
                                p.drive_bias_rear / 2.0,
                                p.drive_bias_rear / 2.0])
        Fx_drive = F_drive_total * drive_split
        # brake: total = brk * brake_g_max * m * g, split by brake bias
        F_brake_total = brk * p.brake_g_max * p.mass * p.g
        brake_split = np.array([p.brake_bias_front / 2.0, p.brake_bias_front / 2.0,
                                (1 - p.brake_bias_front) / 2.0,
                                (1 - p.brake_bias_front) / 2.0])
        Fx_brake = -F_brake_total * brake_split
        Fx_demand = Fx_drive + Fx_brake

        # --- tyre forces via Pacejka + friction-ellipse coupling ---
        # peak forces come from the per-axle cache (one slip-sweep per load bucket),
        # and the ellipse coupling is done inline so the inner loop never triggers
        # another 121-point mu sweep.
        ell_kx = getattr(self.tire, "ell_kx", 2.0)
        ell_ky = getattr(self.tire, "ell_ky", 2.0)
        mu_x_ratio = getattr(self.tire, "mu_x_ratio", 1.05)
        Fy = np.zeros(4)
        Fx = np.zeros(4)
        for i in range(4):
            fz = float(Fz[i])
            if fz <= 1.0:
                continue
            front = i in (FL, FR)
            try:
                fy_pure = float(self.tire.lateral.fy(al[i], fz, cam[i]))
            except Exception:
                fy_pure = 0.0
                self._warn("Tyre lateral force failed at a corner; used 0 there.")
            fy_max = self._peak_force(fz, front)
            fx_max = mu_x_ratio * fy_max
            # clamp the longitudinal demand to the ellipse given the lateral use,
            # then reduce the lateral to what's left — friction-circle coupling.
            if fy_max > 1e-6:
                use_y = min(abs(fy_pure) / fy_max, 1.0)
                fx_avail = fx_max * max(1.0 - use_y ** ell_ky, 0.0) ** (1.0 / ell_kx)
            else:
                fx_avail = fx_max
            fx = float(np.clip(Fx_demand[i], -fx_avail, fx_avail))
            if fx_max > 1e-6:
                use_x = min(abs(fx) / fx_max, 1.0)
                fy_avail = fy_max * max(1.0 - use_x ** ell_kx, 0.0) ** (1.0 / ell_ky)
            else:
                fy_avail = abs(fy_pure)
            fy = float(np.clip(fy_pure, -fy_avail, fy_avail))
            Fx[i] = fx
            Fy[i] = fy

        return dict(
            x_i=x_i, y_i=y_i, u=u, v=v, r=r,
            delta=delta, delta_vel=delta_vel,
            F_susp=F_susp, F_spring=F_spring, F_damp=F_damp,
            Fz=Fz, Fy=Fy, Fx=Fx,
            alpha_target=alpha_target, alpha_lag=al.copy(),
            steer=d_steer, throttle=thr, brake=brk,
            z_road=z_road, zd_road=zd_road,
        )

    # ---- the differential block: state derivatives ---------------------- #
    def derivatives(self, t: float, y: np.ndarray,
                    driver: DriverInput, road: RoadInput) -> np.ndarray:
        p = self.p
        A = self.algebraic(t, y, driver, road)
        x_i, y_i = A["x_i"], A["y_i"]
        u = A["u"]; v = A["v"]; r = A["r"]
        Fz = A["Fz"]; Fy = A["Fy"]; Fx = A["Fx"]
        F_susp = A["F_susp"]
        z_u = y[IZU]
        dy = np.zeros(N_STATES)

        m = p.mass
        m_s = p.m_sprung
        m_u = p.unsprung_mass_corner

        # --- planar handling ---
        F_drag = 0.5 * p.rho * p.cd_a * u * u
        F_roll = p.rolling_g * m * p.g
        sumFx = float(np.sum(Fx)) - F_drag - np.sign(u) * F_roll
        sumFy = float(np.sum(Fy))
        Mz = float(np.sum(x_i * Fy - y_i * Fx))     # yaw moment

        du = sumFx / m + v * r
        dv = sumFy / m - u * r
        dr = Mz / p.izz_eff()

        ax = sumFx / m            # m/s² longitudinal (body)
        ay = sumFy / m            # m/s² lateral (body)

        dy[IU] = du
        dy[IV] = dv
        dy[IR] = dr
        # global pose
        psi = y[IPSI]
        dy[IX] = u * math.cos(psi) - v * math.sin(psi)
        dy[IY] = u * math.sin(psi) + v * math.cos(psi)
        dy[IPSI] = r

        # --- sprung-mass ride (heave/pitch/roll) ---
        # suspension forces on sprung mass: +up. Sum and moments about CG.
        Fz_susp = float(np.sum(F_susp))
        # heave: springs/dampers + gravity already in static => F_susp is delta-force
        # (since spring uses delta from static, the static balance cancels). Heave
        # acceleration from net suspension force.
        zdd_s = Fz_susp / m_s
        # pitch moment: from suspension forces at x_i, plus inertial (longitudinal)
        # weight transfer: accel forward (ax>0) pitches nose UP (+theta).
        M_pitch_susp = float(np.sum(x_i * F_susp))
        M_pitch_inertia = m_s * ax * p.cg_height        # +ax -> +theta (nose up)
        thdd = (M_pitch_susp + M_pitch_inertia) / p.iyy
        # roll moment: suspension forces at -y_i, plus inertial lateral transfer.
        # ay>0 is a leftward acceleration (a left turn); the sprung CG above the
        # roll axis swings OUTWARD (to the right), leaning the body right (phi<0
        # in the +left-down convention) and loading the outer (right) tyres.
        M_roll_susp = float(np.sum(-y_i * F_susp))
        h_roll = max(p.cg_height - p.roll_axis_height, 0.0)
        M_roll_inertia = -m_s * ay * h_roll
        phidd = (M_roll_susp + M_roll_inertia) / p.ixx

        dy[IZS] = y[IZDS]
        dy[IZDS] = zdd_s
        dy[IPHI] = y[IPHID]
        dy[IPHID] = phidd
        dy[ITH] = y[ITHD]
        dy[ITHD] = thdd

        # --- unsprung vertical: tyre force up, suspension reaction down ---
        # m_u * zdd_u = Fz(tyre) - Fz_static - F_susp_reaction
        # F_susp acts +up on sprung => -F_susp on unsprung. The tyre contact force
        # already includes the static term; subtract static so this is delta-dyn.
        Fz_static = self.static_corner_loads()
        zdd_u = (Fz - Fz_static - F_susp) / m_u
        dy[IZU] = y[IZDU]
        dy[IZDU] = zdd_u

        # --- lateral tyre relaxation (the transient slip lag) ---
        # d(alpha_lag)/dt = (|Vx|/sigma) * (alpha_target - alpha_lag)
        alpha_target = A["alpha_target"]
        al = y[IAL]
        for i in range(4):
            fz = max(float(Fz[i]), 1.0)
            try:
                sigma = relaxation_length(fz)
            except Exception:
                sigma = 0.4
            sigma = max(sigma, 1e-3)
            Vx = max(abs(u), p.u_min)
            dy[IAL.start + i] = (Vx / sigma) * (alpha_target[i] - al[i])

        # store derived accels for logging via a side channel
        self._last_ax = ax / p.g
        self._last_ay = ay / p.g
        return dy

    # ---- one RK4 step --------------------------------------------------- #
    def _rk4_step(self, t, y, h, driver, road):
        k1 = self.derivatives(t, y, driver, road)
        k2 = self.derivatives(t + 0.5 * h, y + 0.5 * h * k1, driver, road)
        k3 = self.derivatives(t + 0.5 * h, y + 0.5 * h * k2, driver, road)
        k4 = self.derivatives(t + h, y + h * k3, driver, road)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _clamp(self, y: np.ndarray) -> np.ndarray:
        p = self.p
        y[IU] = float(np.clip(y[IU], 0.0, p.V_MAX))
        y[IV] = float(np.clip(y[IV], -p.V_MAX, p.V_MAX))
        y[IR] = float(np.clip(y[IR], -20.0, 20.0))           # rad/s, generous
        # ride travel clamps (±0.2 m is well beyond any FSAE travel)
        y[IZS] = float(np.clip(y[IZS], -0.2, 0.2))
        y[IZU] = np.clip(y[IZU], -0.2, 0.2)
        y[IPHI] = float(np.clip(y[IPHI], -0.6, 0.6))
        y[ITH] = float(np.clip(y[ITH], -0.6, 0.6))
        y[IAL] = np.clip(y[IAL], -0.6, 0.6)                  # slip angle rad
        return y

    # ---- run a manoeuvre ------------------------------------------------ #
    def run(self, t_end: float, driver: DriverInput | None = None,
            road: RoadInput | None = None, y0: np.ndarray | None = None,
            u0: float = 10.0) -> TransientResult:
        """
        Integrate from 0 to t_end. Returns a TransientResult always. Logs every
        base step (dt); `substeps` adds internal RK4 sub-steps for stiffness
        without bloating the trace.
        """
        self.warnings = []
        self._peak_cache.clear()
        self._last_ax = 0.0
        self._last_ay = 0.0
        driver = driver or DriverInput()
        road = road or RoadInput()
        p = self.p
        try:
            y = (y0.copy() if y0 is not None else self.initial_state(u0))
            dt = max(float(p.dt), 1e-5)
            nsub = max(int(p.substeps), 1)
            hsub = dt / nsub
            n = max(int(round(float(t_end) / dt)), 1)

            # pre-allocate logs
            T = np.zeros(n + 1)
            U = np.zeros(n + 1); V = np.zeros(n + 1); R = np.zeros(n + 1)
            BETA = np.zeros(n + 1); AX = np.zeros(n + 1); AY = np.zeros(n + 1)
            GX = np.zeros(n + 1); GY = np.zeros(n + 1); PSI = np.zeros(n + 1)
            HEAVE = np.zeros(n + 1); PITCH = np.zeros(n + 1); ROLL = np.zeros(n + 1)
            FZ = np.zeros((n + 1, 4)); FY = np.zeros((n + 1, 4))
            FX = np.zeros((n + 1, 4)); AL = np.zeros((n + 1, 4))
            SV = np.zeros((n + 1, 4))
            ST = np.zeros(n + 1); TH = np.zeros(n + 1); BR = np.zeros(n + 1)

            def log(idx, tt, yy):
                A = self.algebraic(tt, yy, driver, road)
                T[idx] = tt
                U[idx] = yy[IU]; V[idx] = yy[IV]; R[idx] = yy[IR]
                BETA[idx] = math.atan2(yy[IV], max(abs(yy[IU]), p.u_min))
                AX[idx] = self._last_ax; AY[idx] = self._last_ay
                GX[idx] = yy[IX]; GY[idx] = yy[IY]; PSI[idx] = yy[IPSI]
                HEAVE[idx] = yy[IZS]; PITCH[idx] = yy[ITH]; ROLL[idx] = yy[IPHI]
                FZ[idx] = A["Fz"]; FY[idx] = A["Fy"]; FX[idx] = A["Fx"]
                AL[idx] = yy[IAL]; SV[idx] = A["delta_vel"]
                ST[idx] = A["steer"]; TH[idx] = A["throttle"]; BR[idx] = A["brake"]

            # prime the derived-accel side channel and log t=0
            self.derivatives(0.0, y, driver, road)
            log(0, 0.0, y)

            blew_up = False
            for k in range(1, n + 1):
                tt = (k - 1) * dt
                for _ in range(nsub):
                    y = self._rk4_step(tt, y, hsub, driver, road)
                    tt += hsub
                    if not np.all(np.isfinite(y)):
                        blew_up = True
                        break
                    y = self._clamp(y)
                if blew_up:
                    self._warn("Integration produced a non-finite state and was "
                               "stopped early; trace truncated at the blow-up. "
                               "Try a smaller dt or check the inputs.")
                    # truncate logs
                    T = T[:k]; U = U[:k]; V = V[:k]; R = R[:k]; BETA = BETA[:k]
                    AX = AX[:k]; AY = AY[:k]; GX = GX[:k]; GY = GY[:k]; PSI = PSI[:k]
                    HEAVE = HEAVE[:k]; PITCH = PITCH[:k]; ROLL = ROLL[:k]
                    FZ = FZ[:k]; FY = FY[:k]; FX = FX[:k]; AL = AL[:k]; SV = SV[:k]
                    ST = ST[:k]; TH = TH[:k]; BR = BR[:k]
                    break
                # refresh derived accels for this logged point
                self.derivatives(k * dt, y, driver, road)
                log(k, k * dt, y)

            return TransientResult(
                ok=not blew_up, t=T, u=U, v=V, r=R, beta=BETA, ax=AX, ay=AY,
                X=GX, Y=GY, psi=PSI, heave=HEAVE, pitch=PITCH, roll=ROLL,
                Fz=FZ, Fy=FY, Fx=FX, alpha=AL, susp_vel=SV,
                steer=ST, throttle=TH, brake=BR,
                warnings=list(self.warnings),
                meta=dict(dt=dt, substeps=nsub, n_steps=len(T),
                          tire=self.tire.status() if hasattr(self.tire, "status") else "n/a",
                          damper=self.damper.status() if hasattr(self.damper, "status") else "n/a",
                          izz=p.izz_eff(), m_sprung=p.m_sprung),
            )
        except Exception as e:
            return TransientResult.failed(
                self.warnings + [f"Transient run failed entirely "
                                 f"({type(e).__name__}: {e}); returned a safe "
                                 f"empty result so the app keeps running."])


# --------------------------------------------------------------------------- #
#  Manoeuvre harness — the named transient events
# --------------------------------------------------------------------------- #
#  Each builder returns (DriverInput, RoadInput, t_end, u0, label). `run_maneuver`
#  wires one to a solver and returns the TransientResult. These are the scenarios
#  the QSS sim structurally cannot show, set up so the transient content is the
#  headline of the trace.
# --------------------------------------------------------------------------- #
def step_steer_maneuver(steer_deg: float = 4.0, u0: float = 18.0,
                        t_step: float = 0.3, ramp: float = 0.05,
                        t_end: float = 3.0):
    """
    Classic step-steer (J-turn): hold speed, snap the wheel to `steer_deg` over a
    short `ramp`, hold. Reveals turn-in lag, yaw-rate overshoot, and the settle to
    steady cornering — the transient response QSS replaces with an instantaneous
    balanced corner.
    """
    d = math.radians(steer_deg)

    def steer(t):
        if t < t_step:
            return 0.0
        return d * min((t - t_step) / max(ramp, 1e-3), 1.0)

    drv = DriverInput(steer=steer, throttle=lambda t: 0.12)  # light throttle to hold speed
    return drv, RoadInput(), t_end, u0, f"Step steer {steer_deg:.1f}°"


def snap_oversteer_maneuver(steer_deg: float = 3.8, u0: float = 16.0,
                            lift_time: float = 1.0, brake_stab: float = 0.45,
                            stab_dur: float = 0.35, recover: bool = True,
                            counter_gain: float = 4.0, counter_max_deg: float = 14.0,
                            t_end: float = 4.0):
    """
    Provoke trailing-throttle / lift-off snap oversteer from a STABLE corner and
    CATCH it with a reactive countersteer.

    The base corner (`steer_deg` at `u0`) is chosen sub-limit so the car is in
    settled equilibrium under power — this is the state QSS would report. At
    `lift_time` the throttle snaps shut and a brief brake stab (`brake_stab` for
    `stab_dur`) pitches the car forward; that forward weight transfer unloads the
    rear axle, which loses grip and steps out — the snap. A hand-timed open-loop
    countersteer can't track a slide, so the recovery is a state-feedback
    countersteer: once sideslip exceeds a small threshold the driver subtracts
    opposite lock proportional to (sideslip + a yaw-rate lead term), saturating at
    `counter_max_deg`. The trace then shows the departure from equilibrium and the
    catch back toward it — the unsteady excursion a quasi-steady model assumes
    away. Set recover=False to watch the uncaught spin for contrast.
    """
    d = math.radians(steer_deg)
    cmax = math.radians(counter_max_deg)
    beta_thresh = math.radians(2.0)
    t_stab_end = lift_time + stab_dur

    def steer(t, state=None):
        base = d * min(max((t - 0.2) / 0.05, 0.0), 1.0)
        if not recover or state is None or t < lift_time:
            return base
        beta = state.get("beta", 0.0)
        r = state.get("r", 0.0)
        slide = beta + 0.2 * r           # yaw-rate lead so the catch is early
        if abs(slide) <= beta_thresh:
            return base
        corr = -counter_gain * (slide - math.copysign(beta_thresh, slide))
        corr = max(-cmax, min(cmax, corr))
        return base + corr

    def throttle(t):
        return 0.18 if t < lift_time else 0.0

    def brake(t):
        return brake_stab if lift_time <= t < t_stab_end else 0.0

    drv = DriverInput(steer=steer, throttle=throttle, brake=brake)
    return drv, RoadInput(), t_end, u0, "Snap-oversteer + recovery"


def brake_to_throttle_maneuver(u0: float = 25.0, brake_time: float = 0.4,
                               brake_dur: float = 0.6, t_end: float = 2.5):
    """
    Straight-line brake-to-throttle transition: coast, brake hard, then snap to
    full throttle. The sprung mass pitches forward under braking (dive) and rocks
    back under power (squat); the digressive damper controls the oscillation. QSS
    has no pitch DOF at all, so this transient is invisible to it.
    """
    t_off = brake_time + brake_dur

    def brake(t):
        return 1.0 if brake_time <= t < t_off else 0.0

    def throttle(t):
        return 1.0 if t >= t_off else 0.0

    drv = DriverInput(steer=lambda t: 0.0, brake=brake, throttle=throttle)
    return drv, RoadInput(), t_end, u0, "Brake → throttle pitch/dive"


def curb_strike_maneuver(u0: float = 20.0, curb_h: float = 0.025,
                         t_hit: float = 0.3, wheels=("FL", "RL"),
                         curb_len_m: float = 0.5, t_end: float = 1.2):
    """
    Drive over a kerb: a half-sine vertical bump of height `curb_h` (m) and length
    `curb_len_m` (m) under the chosen `wheels`, encountered at `t_hit`. Excites the
    high-frequency unsprung (wheel-hop) mode — the contact load spikes then can
    drop to zero (wheel lift). This is the millisecond-scale event the whole
    point of a high-frequency transient solver is to resolve; a QSS point mass
    has no unsprung mass and cannot represent it.
    """
    idx = [{"FL": FL, "FR": FR, "RL": RL, "RR": RR}[w] for w in wheels]
    mask = np.zeros(4)
    for i in idx:
        mask[i] = 1.0
    dur = max(curb_len_m / max(u0, 0.1), 1e-3)   # time over the kerb at this speed

    def z(t):
        if t_hit <= t < t_hit + dur:
            phase = (t - t_hit) / dur
            return mask * curb_h * math.sin(math.pi * phase)
        return np.zeros(4)

    road = RoadInput(z=z)
    drv = DriverInput(throttle=lambda t: 0.1)
    return drv, road, t_end, u0, f"Kerb strike ({'+'.join(wheels)})"


def run_maneuver(veh: VehicleDynamics | None, kind: str = "step_steer",
                 params: TransientParams | None = None,
                 tire: CombinedSlipTire | None = None,
                 damper: DamperCurve | None = None, **kw) -> TransientResult:
    """
    Build and run one named manoeuvre. `kind` in
    {step_steer, snap_oversteer, brake_to_throttle, curb_strike}. Extra kwargs are
    forwarded to the builder. Never raises; returns a (possibly flagged) result.
    """
    builders = {
        "step_steer": step_steer_maneuver,
        "snap_oversteer": snap_oversteer_maneuver,
        "brake_to_throttle": brake_to_throttle_maneuver,
        "curb_strike": curb_strike_maneuver,
    }
    try:
        builder = builders.get(kind)
        if builder is None:
            return TransientResult.failed([f"Unknown manoeuvre '{kind}'. "
                                           f"Options: {sorted(builders)}."])
        drv, road, t_end, u0, label = builder(**kw)
        sim = TransientSolver(veh, params=params, tire=tire, damper=damper)
        res = sim.run(t_end, driver=drv, road=road, u0=u0)
        res.meta["maneuver"] = label
        return res
    except Exception as e:
        return TransientResult.failed([f"Manoeuvre '{kind}' failed "
                                       f"({type(e).__name__}: {e})."])


# --------------------------------------------------------------------------- #
#  Transient vs steady-state: the "before it settles" comparison
# --------------------------------------------------------------------------- #
@dataclass
class SettlingResult:
    ok: bool
    steady_ay_g: float          # transient steady-state lateral g reached
    qss_max_ay_g: float         # QSS max lateral g for the same car (reference)
    peak_ay_g: float            # transient peak (overshoot) lateral g
    overshoot_pct: float        # (peak-steady)/steady * 100
    rise_time_s: float          # time to first reach 90% of steady ay
    settle_time_s: float        # time to stay within 5% of steady ay
    result: TransientResult
    warnings: list[str] = field(default_factory=list)


def transient_vs_qss_corner(veh: VehicleDynamics,
                            target_ay_g: float | None = None,
                            u0: float = 18.0, t_end: float = 3.0,
                            params: TransientParams | None = None) -> SettlingResult:
    """
    Turn into a steady corner and measure HOW the car gets to steady state, which
    is precisely the information QSS discards. Picks a steer that asymptotes near
    `target_ay_g` (default: 70% of the QSS limit, a comfortably attainable corner),
    runs the transient solver, and reports rise time, yaw/lat-g overshoot, and
    settling time, alongside the QSS steady-state reference for the same car.

    The point it makes: QSS reports a single number (the steady corner); the
    transient solver shows the car overshooting it, oscillating, and only then
    settling onto it — the unsettled phase a quasi-steady model assumes away.
    Never raises.
    """
    try:
        try:
            qss_max = float(veh.max_lateral_g())
        except Exception:
            qss_max = 1.4
        tgt = target_ay_g if target_ay_g is not None else 0.7 * qss_max

        sim = TransientSolver(veh, params=params)
        # find a steer that gives ~tgt steady ay by a short bisection on a 1.5 s run
        def steady_ay(steer_deg):
            drv = DriverInput(steer=(lambda t, d=math.radians(steer_deg):
                                     d * min(max((t - 0.2) / 0.05, 0.0), 1.0)),
                              throttle=lambda t: 0.12)
            r = sim.run(1.5, driver=drv, u0=u0)
            if not r.ok or len(r.ay) < 5:
                return 0.0, r
            return float(np.mean(np.abs(r.ay[-50:]))), r

        lo, hi = 0.5, 8.0
        best = None
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            ay_mid, _ = steady_ay(mid)
            if ay_mid < tgt:
                lo = mid
            else:
                hi = mid
            best = mid
        steer_deg = best or 3.0

        # final, longer run at the chosen steer
        drv = DriverInput(steer=(lambda t, d=math.radians(steer_deg):
                                 d * min(max((t - 0.2) / 0.05, 0.0), 1.0)),
                          throttle=lambda t: 0.12)
        res = sim.run(t_end, driver=drv, u0=u0)
        ay = np.abs(res.ay)
        steady = float(np.mean(ay[-100:])) if len(ay) > 100 else float(np.mean(ay))
        peak = float(np.max(ay)) if len(ay) else 0.0
        overshoot = 100.0 * (peak - steady) / steady if steady > 1e-6 else 0.0

        # rise time to 90% of steady
        rise = float("nan")
        if steady > 1e-6:
            above = np.where(ay >= 0.9 * steady)[0]
            if len(above):
                rise = float(res.t[above[0]] - 0.2)
        # settle time: last time it leaves the ±5% band
        settle = float("nan")
        if steady > 1e-6:
            outside = np.where(np.abs(ay - steady) > 0.05 * steady)[0]
            if len(outside):
                settle = float(res.t[outside[-1]] - 0.2)
            else:
                settle = 0.0

        return SettlingResult(
            ok=res.ok, steady_ay_g=steady, qss_max_ay_g=qss_max,
            peak_ay_g=peak, overshoot_pct=overshoot,
            rise_time_s=rise, settle_time_s=settle,
            result=res, warnings=list(res.warnings),
        )
    except Exception as e:
        return SettlingResult(
            ok=False, steady_ay_g=0.0, qss_max_ay_g=0.0, peak_ay_g=0.0,
            overshoot_pct=0.0, rise_time_s=float("nan"),
            settle_time_s=float("nan"), result=TransientResult.failed([str(e)]),
            warnings=[f"transient_vs_qss_corner failed ({type(e).__name__}: {e})"])
