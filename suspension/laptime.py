# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Lap-time simulation — turn the grip envelope into the only number that wins:
seconds.

Everything else in KinematiK reports steady-state grip at a single operating
point. But competition is decided by *lap time*, which is a transient, track-
dependent integral of that grip envelope. A better-funded team buys that integral
empirically by testing fresh rubber all season. An underfunded team that can only
run ONE tire set has to predict it instead — and predicting it well, before the
build is frozen, is the single highest-leverage thing the software can do.

This module is a quasi-steady-state (QSS) point-mass lap simulator. It takes the
*same* `VehicleDynamics` object the rest of the tool already builds — so every
geometry / setup / tire change you make upstream flows straight through to a
predicted time — and runs it around:

    * the FSAE skidpad (a fixed-radius circle: the cleanest possible validation
      case, with a closed-form steady-state answer), and
    * a parameterisable autocross / track defined as a sequence of segments
      (straights + constant-radius corners), solved with a standard three-pass
      QSS method: per-corner limit speeds, a forward acceleration pass, a backward
      braking pass, then integrate dt = ds / v.

QSS is deliberate. A full transient model needs tyre relaxation, yaw inertia, and
combined-slip data we don't have on one tyre set; QSS needs only the lateral grip
envelope we already trust and a defensible longitudinal model, and on FSAE-scale
tracks it lands within a few percent — accurate enough to RANK setups, which is
what actually moves you up the results sheet. It pairs with the setup optimiser:
optimise for max grip, then confirm the change is worth seconds here.

DESIGN RULE FOR THIS MODULE: never let one bad data point kill a session. Every
public function is wrapped so that if a calculation can't complete it returns a
safe, clearly-flagged default (with a `warning` string) instead of raising. A lap
sim is run interactively on geometry the user is actively dragging around — a
non-convergent linkage or a degenerate track must surface a warning in the UI, not
crash the app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from .dynamics import VehicleDynamics, VehicleParams


# --------------------------------------------------------------------------- #
#  Powertrain / longitudinal envelope
# --------------------------------------------------------------------------- #
@dataclass
class MotorMap:
    """
    A real motor torque/power curve, replacing the flat power cap.

    An electric FSAE motor is not a constant-power source: below its base speed it
    is torque-limited (a roughly flat torque plateau), and above it, field weakening
    gives an approximately constant-power region that then tails off. Modelling that
    shape changes corner-exit acceleration noticeably versus assuming P=const, which
    is what `Powertrain.power_w()/v` does.

    Supply the curve as motor-shaft (torque Nm, speed rpm) points — from your motor's
    datasheet or a dyno pull. They are linearly interpolated and combined with the
    final drive ratio and loaded wheel radius to give wheel tractive force vs road
    speed:

        omega_motor [rad/s] = v / r_wheel * final_drive
        rpm_motor           = omega_motor * 60 / (2*pi)
        T_wheel             = T_motor(rpm) * final_drive * drivetrain_eff
        F_wheel             = T_wheel / r_wheel

    Honesty note: if you don't have a curve, DON'T fabricate one — leave the motor
    map off and the model falls back to the documented flat-power approximation,
    clearly the cruder of the two. `from_peak()` builds a *representative* curve from
    just peak torque/power/redline for when that's all the datasheet gives you; it is
    labelled representative, not measured.
    """
    rpm: List[float]                 # motor speed sample points, rev/min
    torque_nm: List[float]           # motor shaft torque at each rpm
    final_drive: float = 3.5         # motor:wheel reduction (sprocket / gearbox)
    wheel_radius_m: float = 0.20     # loaded tyre radius
    source: str = "user"             # "user" | "datasheet" | "representative"

    def __post_init__(self):
        r = np.asarray(self.rpm, float)
        t = np.asarray(self.torque_nm, float)
        order = np.argsort(r)
        self._rpm = r[order]
        self._t = t[order]

    @staticmethod
    def from_peak(peak_torque_nm: float, peak_power_kw: float,
                  redline_rpm: float, final_drive: float = 3.5,
                  wheel_radius_m: float = 0.20) -> "MotorMap":
        """
        Build a REPRESENTATIVE curve from the three numbers a datasheet always
        gives: a flat torque plateau up to the base speed where it would exceed
        peak power, then a constant-power hyperbola (T = P/omega) to redline.
        Flagged source='representative' so the UI can say so — it is a shape, not
        a measurement.
        """
        peak_torque_nm = max(peak_torque_nm, 1.0)
        peak_power_w = max(peak_power_kw, 0.1) * 1000.0
        redline_rpm = max(redline_rpm, 1000.0)
        # base speed where flat torque first hits peak power: P = T * omega
        omega_base = peak_power_w / peak_torque_nm           # rad/s
        rpm_base = omega_base * 60.0 / (2.0 * math.pi)
        rpm_base = min(rpm_base, redline_rpm * 0.95)
        pts_rpm, pts_t = [0.0, rpm_base], [peak_torque_nm, peak_torque_nm]
        # constant-power tail
        for frac in (0.6, 0.75, 0.9, 1.0):
            rp = rpm_base + frac * (redline_rpm - rpm_base)
            omega = rp * 2.0 * math.pi / 60.0
            pts_rpm.append(rp)
            pts_t.append(peak_power_w / max(omega, 1e-3))
        return MotorMap(rpm=pts_rpm, torque_nm=pts_t, final_drive=final_drive,
                        wheel_radius_m=wheel_radius_m, source="representative")

    def wheel_force(self, v_ms: float, drivetrain_eff: float = 0.90) -> float:
        """Tractive force available at the contact patch at road speed v (m/s)."""
        v = max(v_ms, 0.05)
        r = max(self.wheel_radius_m, 0.05)
        rpm = (v / r) * self.final_drive * 60.0 / (2.0 * math.pi)
        # clamp to the mapped range; beyond redline the motor makes no more torque
        if rpm <= self._rpm[0]:
            t_motor = float(self._t[0])
        elif rpm >= self._rpm[-1]:
            t_motor = 0.0                    # past redline: no drive
        else:
            t_motor = float(np.interp(rpm, self._rpm, self._t))
        t_wheel = t_motor * self.final_drive * max(drivetrain_eff, 0.05)
        return t_wheel / r

    def as_dict(self):
        return dict(rpm=list(self._rpm), torque_nm=list(self._t),
                    final_drive=self.final_drive, wheel_radius_m=self.wheel_radius_m,
                    source=self.source)


@dataclass
class Powertrain:
    """
    A deliberately simple longitudinal model — enough to make the straights and
    the corner-exit acceleration realistic. All fields have FSAE-EV-representative
    defaults; override with your own numbers on the Lap Sim tab.

    Tractive force is the lesser of (a) what the tyres can put down — mu * rear (or
    all-wheel) vertical load — and (b) what the motor delivers at the current speed.
    The motor side is EITHER a flat power cap (F = P/v, capped by max_tractive_n) —
    the cruder default — OR, if `motor_map` is set, a real torque/speed curve. The
    map is strictly better when you have the curve; the flat cap is the honest
    fallback when you don't.
    """
    power_kw: float = 80.0           # peak electric power at the wheels, kW
    max_tractive_n: float = 2600.0   # low-speed torque/traction cap, N (flat model)
    drivetrain_eff: float = 0.90     # wheel power / battery power
    cda: float = 1.10                # drag area Cd*A, m^2
    cla: float = 2.60                # downforce area Cl*A, m^2 (aero pkg; 0 if none)
    rho: float = 1.20                # air density, kg/m^3
    crr: float = 0.018               # rolling resistance coefficient
    drive: str = "rwd"               # "rwd" or "awd" — which axle loads cap traction
    brake_g_cap: float = 1.8         # mechanical brake ceiling (g), grip-limited below
    motor_map: Optional[MotorMap] = None  # real curve; None => flat power cap
    combined_tire: object = None     # optional tiremodel.CombinedSlipTire for the
                                     # friction-ellipse coupling (Fx can exceed Fy
                                     # peak, tunable exponents). None => symmetric circle.

    def power_w(self) -> float:
        return max(self.power_kw, 0.0) * 1000.0 * max(self.drivetrain_eff, 0.05)

    def tractive_force(self, v_ms: float) -> float:
        """
        Motor-side tractive force ceiling at road speed v (before the tyre grip
        limit is applied by the caller). Uses the motor map when present, else the
        flat power-cap approximation.
        """
        v = max(v_ms, 0.05)
        if self.motor_map is not None:
            return self.motor_map.wheel_force(v, self.drivetrain_eff)
        return min(self.power_w() / v, self.max_tractive_n)

    def uses_real_motor_map(self) -> bool:
        return self.motor_map is not None


# --------------------------------------------------------------------------- #
#  Track description
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One piece of track. A straight has radius=None; a corner has radius_m>0."""
    length_m: float
    radius_m: Optional[float] = None   # None / <=0 => straight

    @property
    def is_corner(self) -> bool:
        return self.radius_m is not None and self.radius_m > 0.0


@dataclass
class Track:
    name: str
    segments: List[Segment] = field(default_factory=list)
    ds: float = 1.0                    # integration step, m

    def total_length(self) -> float:
        return float(sum(max(s.length_m, 0.0) for s in self.segments))


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class LapResult:
    lap_time_s: float
    avg_speed_ms: float
    top_speed_ms: float
    min_speed_ms: float
    distance_m: float
    # per-station traces (for plotting); always finite, may be empty on failure
    s: list = field(default_factory=list)
    v: list = field(default_factory=list)
    ok: bool = True
    warning: str = ""

    def as_summary(self) -> dict:
        return dict(lap_time_s=round(self.lap_time_s, 3),
                    avg_speed_ms=round(self.avg_speed_ms, 2),
                    top_speed_ms=round(self.top_speed_ms, 2),
                    min_speed_ms=round(self.min_speed_ms, 2),
                    distance_m=round(self.distance_m, 1),
                    ok=self.ok, warning=self.warning)


def _safe_lap(distance=0.0, warning="calculation unavailable") -> LapResult:
    """A finite, non-crashing placeholder result with a surfaced warning."""
    return LapResult(lap_time_s=float("nan"), avg_speed_ms=0.0, top_speed_ms=0.0,
                     min_speed_ms=0.0, distance_m=float(distance), s=[], v=[],
                     ok=False, warning=warning)


# --------------------------------------------------------------------------- #
#  Core grip lookups (wrapped so a bad geometry never throws)
# --------------------------------------------------------------------------- #
def _max_lat_g(veh: VehicleDynamics) -> float:
    """Steady-state lateral g from the live dynamics model, guarded."""
    try:
        g = float(veh.max_lateral_g())
        if not math.isfinite(g) or g <= 0.0:
            return 1.4  # safe representative fallback, flagged by caller
        return g
    except Exception:
        return 1.4


def _corner_limit_speed(veh: VehicleDynamics, radius_m: float, pt: Powertrain,
                        max_lat_g: float) -> float:
    """
    Max speed through a constant-radius corner. With aero downforce, grip grows
    with speed, so the limit is the fixed point of:
        v^2 / R = a_lat(v) = max_lat_g * g * (1 + downforce/weight)
    Solve directly (downforce ∝ v^2 makes this closed-form).
    """
    if radius_m <= 0.0 or not math.isfinite(radius_m):
        return float("inf")  # treat as straight
    g = 9.81
    m = max(veh.p.mass, 1.0)
    W = m * g
    a0 = max_lat_g * g                      # grip accel at zero aero, m/s^2
    # downforce accel coefficient: F_down = 0.5*rho*ClA*v^2 ; extra a_lat = mu*F/m
    # approximate mu ~ max_lat_g (grip already includes load sensitivity envelope)
    k = 0.5 * pt.rho * max(pt.cla, 0.0) / m  # downforce / v^2 per unit mass
    # v^2/R = a0 + max_lat_g * k * v^2   ->  v^2 (1/R - mu*k) = a0
    denom = (1.0 / radius_m) - max_lat_g * k
    if denom <= 1e-9:
        # aero would (unphysically) let speed run away; cap at no-aero solution
        v2 = a0 * radius_m
    else:
        v2 = a0 / denom
    v2 = max(v2, 0.0)
    return math.sqrt(v2)


def _accel_long(veh: VehicleDynamics, v: float, pt: Powertrain,
                max_lat_g: float, lat_used_g: float) -> float:
    """
    Available longitudinal acceleration (m/s^2) at speed v, accounting for the
    grip already spent cornering (friction-circle coupling) plus power, drag and
    rolling resistance. Used in the forward pass.
    """
    g = 9.81
    m = max(veh.p.mass, 1.0)
    v = max(v, 0.1)
    # vertical load with aero
    F_down = 0.5 * pt.rho * max(pt.cla, 0.0) * v * v
    # longitudinal grip ceiling (driven axle share)
    axle_frac = 1.0 if pt.drive == "awd" else (1.0 - veh.p.weight_dist_front)
    N_drive = (m * g + F_down) * (axle_frac if pt.drive != "awd" else 1.0)
    mu = max(max_lat_g, 0.3)
    F_grip = mu * N_drive
    # friction circle: subtract lateral usage. If a combined-slip tyre is supplied,
    # use its (possibly asymmetric, calibrated) ellipse; else the symmetric circle.
    frac_lat = min(lat_used_g / max(max_lat_g, 1e-6), 1.0)
    ct = getattr(pt, "combined_tire", None)
    if ct is not None:
        try:
            ky = max(ct.ell_ky, 1e-3); kx = max(ct.ell_kx, 1e-3)
            long_grip_frac = max(1.0 - frac_lat ** ky, 0.0) ** (1.0 / kx)
            F_grip *= ct.mu_x_ratio * long_grip_frac
        except Exception:
            long_grip_frac = math.sqrt(max(1.0 - frac_lat * frac_lat, 0.0))
            F_grip *= long_grip_frac
    else:
        long_grip_frac = math.sqrt(max(1.0 - frac_lat * frac_lat, 0.0))
        F_grip *= long_grip_frac
    # motor-side tractive ceiling: real torque/speed map if supplied, else flat cap
    F_motor = pt.tractive_force(v)
    F_drive = min(F_grip, F_motor)
    # resistances
    F_drag = 0.5 * pt.rho * pt.cda * v * v
    F_roll = pt.crr * (m * g + F_down)
    a = (F_drive - F_drag - F_roll) / m
    return a


def _decel_long(veh: VehicleDynamics, v: float, pt: Powertrain,
                max_lat_g: float, lat_used_g: float) -> float:
    """
    Available braking deceleration (m/s^2, positive number) at speed v under the
    same friction-circle coupling. Drag and downforce *help* braking.
    """
    g = 9.81
    m = max(veh.p.mass, 1.0)
    v = max(v, 0.1)
    F_down = 0.5 * pt.rho * max(pt.cla, 0.0) * v * v
    mu = max(max_lat_g, 0.3)
    F_grip = mu * (m * g + F_down)          # all four tyres brake
    frac_lat = min(lat_used_g / max(max_lat_g, 1e-6), 1.0)
    long_grip_frac = math.sqrt(max(1.0 - frac_lat * frac_lat, 0.0))
    F_grip *= long_grip_frac
    F_brake = min(F_grip, pt.brake_g_cap * m * g)
    F_drag = 0.5 * pt.rho * pt.cda * v * v
    a = (F_brake + F_drag) / m
    return a


# --------------------------------------------------------------------------- #
#  Standing-start acceleration (the 75 m event)
# --------------------------------------------------------------------------- #
def acceleration_time(veh: VehicleDynamics, pt: Optional[Powertrain] = None,
                      distance_m: float = 75.0, dv_steps: int = 4000) -> LapResult:
    """
    Time for a standing-start straight-line acceleration over `distance_m`.

    `simulate_lap` is built for CLOSED laps — it seeds the forward pass from the
    corner ceiling and closes the loop, so it does not model a run from rest and
    returns ~0 on a bare straight. This helper does the honest thing: integrate
    the same longitudinal model (`_accel_long`, full longitudinal grip since no
    lateral is used) forward from v=0 until `distance_m` is covered.

    Never raises; returns a safe flagged LapResult on any failure.
    """
    try:
        pt = pt or Powertrain()
        if not (math.isfinite(distance_m) and distance_m > 0):
            return _safe_lap(warning="acceleration distance invalid; check inputs")
        max_lat_g = _max_lat_g(veh)          # only used as the grip ceiling
        s = 0.0
        v = 0.1                               # tiny seed to avoid F_power = P/0
        t = 0.0
        ds = distance_m / max(int(dv_steps), 50)
        s_trace, v_trace = [0.0], [0.0]
        guard = 0
        max_iter = int(dv_steps) * 4 + 1000
        while s < distance_m and guard < max_iter:
            guard += 1
            a = _accel_long(veh, v, pt, max_lat_g, lat_used_g=0.0)
            if not math.isfinite(a):
                a = 0.0
            v_next = math.sqrt(max(v * v + 2.0 * a * ds, 1e-4))
            v_mean = max(0.5 * (v + v_next), 0.05)
            t += ds / v_mean
            s += ds
            v = v_next
            s_trace.append(min(s, distance_m))
            v_trace.append(v)
        if guard >= max_iter:
            return _safe_lap(warning="acceleration sim did not reach the line — "
                                     "car barely accelerates; check power/grip inputs")
        warning = ("grip fell back to default 1.4 g — verify tire/geometry"
                   if max_lat_g == 1.4 else "")
        return LapResult(lap_time_s=float(t),
                         avg_speed_ms=float(distance_m / t) if t > 0 else 0.0,
                         top_speed_ms=float(v), min_speed_ms=0.0,
                         distance_m=float(distance_m),
                         s=s_trace, v=v_trace, ok=True, warning=warning)
    except Exception as e:
        return _safe_lap(warning=f"acceleration sim failed safely: {e}")


# --------------------------------------------------------------------------- #
#  Skidpad — the clean closed-form case
# --------------------------------------------------------------------------- #
# FSAE skidpad: two 15.25 m centreline-diameter circles in a figure-8; the timed
# run is one full lap of one circle. Standard path radius ~ 9.125 m (8.5 m inner
# circle radius + ~0.625 m to the tyre centreline track). We use the commonly
# cited timed-circle radius and circumference.
SKIDPAD_RADIUS_M = 9.125
SKIDPAD_CIRCUMFERENCE_M = 2.0 * math.pi * SKIDPAD_RADIUS_M


def skidpad_time(veh: VehicleDynamics, pt: Optional[Powertrain] = None,
                 radius_m: float = SKIDPAD_RADIUS_M) -> LapResult:
    """
    Predicted FSAE skidpad time for one timed circle, from the live grip model.
    Steady-state and closed-form: v = sqrt(a_lat * R), t = circumference / v.
    This is the cleanest possible check that the whole grip stack is sane — you
    can sanity-check it by hand and against your own skidpad runs.
    """
    try:
        pt = pt or Powertrain()
        if radius_m <= 0 or not math.isfinite(radius_m):
            return _safe_lap(warning="skidpad radius invalid; check inputs")
        max_lat_g = _max_lat_g(veh)
        v = _corner_limit_speed(veh, radius_m, pt, max_lat_g)
        if not math.isfinite(v) or v <= 0.0:
            return _safe_lap(warning="grip model returned no usable speed; "
                                     "using safe default — check geometry/tire")
        circ = 2.0 * math.pi * radius_m
        t = circ / v
        return LapResult(lap_time_s=t, avg_speed_ms=v, top_speed_ms=v,
                         min_speed_ms=v, distance_m=circ,
                         s=[0.0, circ], v=[v, v], ok=True,
                         warning="" if max_lat_g != 1.4 else
                                 "grip fell back to default 1.4 g — verify tire/geometry")
    except Exception as e:                       # never crash the session
        return _safe_lap(warning=f"skidpad sim failed safely: {e}")


# --------------------------------------------------------------------------- #
#  General QSS lap over an arbitrary track
# --------------------------------------------------------------------------- #
def simulate_lap(veh: VehicleDynamics, track: Track,
                 pt: Optional[Powertrain] = None) -> LapResult:
    """
    Quasi-steady-state lap time over `track`. Three passes:
      1. corner limit speed at every station (vertical asymptote of grip),
      2. forward pass: cap acceleration out of corners by available traction,
      3. backward pass: cap entry speed by available braking,
    then integrate dt = ds / v_mean. Returns a LapResult with traces; on any
    failure returns a flagged safe default rather than raising.
    """
    try:
        pt = pt or Powertrain()
        if not track.segments or track.total_length() <= 0.0:
            return _safe_lap(warning="track is empty; add at least one segment")

        ds = track.ds if track.ds and track.ds > 0 else 1.0
        max_lat_g = _max_lat_g(veh)

        # Build station list: position s, local radius (inf for straight)
        s_pts, radii = [], []
        s_cur = 0.0
        for seg in track.segments:
            L = max(seg.length_m, 0.0)
            r = seg.radius_m if seg.is_corner else float("inf")
            n = max(int(round(L / ds)), 1)
            for _ in range(n):
                s_pts.append(s_cur)
                radii.append(r)
                s_cur += L / n
        s_pts.append(s_cur)
        radii.append(radii[-1] if radii else float("inf"))
        N = len(s_pts)
        if N < 2:
            return _safe_lap(warning="track too short to integrate")

        # Pass 1: cornering speed ceiling at each station
        v_ceiling = np.empty(N)
        for i in range(N):
            r = radii[i]
            if math.isinf(r):
                v_ceiling[i] = 1e6   # straight: no cornering limit (capped later)
            else:
                v_ceiling[i] = _corner_limit_speed(veh, r, pt, max_lat_g)
        # clamp non-finite
        v_ceiling = np.where(np.isfinite(v_ceiling), v_ceiling, 1e6)
        v_ceiling = np.clip(v_ceiling, 0.0, 1e6)

        # Pass 2: forward (acceleration-limited), closed loop -> seed from ceiling
        v_fwd = v_ceiling.copy()
        # iterate twice for the closed lap so the start speed is consistent
        for _ in range(2):
            for i in range(1, N):
                ds_i = max(s_pts[i] - s_pts[i - 1], 1e-3)
                v0 = v_fwd[i - 1]
                lat_used = (v0 * v0 / radii[i - 1] / 9.81) if math.isfinite(radii[i - 1]) else 0.0
                a = _accel_long(veh, v0, pt, max_lat_g, lat_used)
                v_next = math.sqrt(max(v0 * v0 + 2.0 * a * ds_i, 0.0))
                v_fwd[i] = min(v_next, v_ceiling[i])
            v_fwd[0] = min(v_fwd[0], v_fwd[-1])  # close the loop

        # Pass 3: backward (braking-limited)
        v = v_fwd.copy()
        for _ in range(2):
            for i in range(N - 2, -1, -1):
                ds_i = max(s_pts[i + 1] - s_pts[i], 1e-3)
                v1 = v[i + 1]
                lat_used = (v1 * v1 / radii[i + 1] / 9.81) if math.isfinite(radii[i + 1]) else 0.0
                d = _decel_long(veh, v1, pt, max_lat_g, lat_used)
                v_prev = math.sqrt(max(v1 * v1 + 2.0 * d * ds_i, 0.0))
                v[i] = min(v[i], v_prev, v_ceiling[i])
            v[-1] = min(v[-1], v[0])

        v = np.clip(v, 0.05, 1e6)   # avoid div-by-zero in dt

        # Integrate time: dt = ds / v_mean over each interval
        t = 0.0
        for i in range(1, N):
            ds_i = max(s_pts[i] - s_pts[i - 1], 0.0)
            v_mean = 0.5 * (v[i] + v[i - 1])
            t += ds_i / max(v_mean, 0.05)

        dist = s_pts[-1]
        warning = ""
        if max_lat_g == 1.4:
            warning = "grip fell back to default 1.4 g — verify tire/geometry"
        return LapResult(
            lap_time_s=float(t),
            avg_speed_ms=float(dist / t) if t > 0 else 0.0,
            top_speed_ms=float(np.max(v)),
            min_speed_ms=float(np.min(v)),
            distance_m=float(dist),
            s=[float(x) for x in s_pts],
            v=[float(x) for x in v],
            ok=True, warning=warning)
    except Exception as e:
        return _safe_lap(warning=f"lap sim failed safely: {e}")


# --------------------------------------------------------------------------- #
#  A representative autocross track (parameterisable)
# --------------------------------------------------------------------------- #
def default_autocross(scale: float = 1.0) -> Track:
    """
    A representative FSAE-style autocross lap: ~800 m mixing slow hairpins, a
    slalom (modelled as a chain of short alternating-radius corners), sweepers and
    short straights. It is NOT a specific competition map — it's a fixed, sensible
    yardstick so that comparing two setups on it is an apples-to-apples ranking.
    `scale` stretches every length if you want a longer/shorter lap.
    """
    R = lambda r: max(r, 0.1)
    segs = [
        Segment(40 * scale),                 # start straight
        Segment(18 * scale, R(9.0)),         # right sweeper
        Segment(25 * scale),
        Segment(12 * scale, R(5.0)),         # hairpin
        Segment(20 * scale),
        # slalom: alternating tight corners
        *[Segment(6 * scale, R(7.0)) for _ in range(6)],
        Segment(30 * scale),                 # back straight
        Segment(15 * scale, R(12.0)),        # fast sweeper
        Segment(22 * scale),
        Segment(10 * scale, R(6.0)),         # medium corner
        Segment(35 * scale),                 # straight
        Segment(11 * scale, R(4.5)),         # tight hairpin
        Segment(18 * scale),
        Segment(14 * scale, R(8.0)),         # sweeper to finish
        Segment(28 * scale),                 # finish straight
    ]
    return Track(name=f"Representative autocross (×{scale:g})", segments=segs, ds=1.0)


# --------------------------------------------------------------------------- #
#  Build a track from real geometry: GPS centreline or cone coordinates
# --------------------------------------------------------------------------- #
def _menger_curvature(x, y):
    """
    Local curvature (1/radius) at each point of a 2-D path, from the Menger
    curvature of consecutive triples (the reciprocal of the circumradius of the
    triangle formed by points i-1, i, i+1). Endpoints copy their neighbour. Sign
    is dropped — the lap sim only uses |radius|. Returns curvature array (1/m).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    k = np.zeros(n)
    for i in range(1, n - 1):
        ax, ay = x[i - 1], y[i - 1]
        bx, by = x[i],     y[i]
        cx, cy = x[i + 1], y[i + 1]
        # triangle side lengths
        a = math.hypot(bx - cx, by - cy)
        b = math.hypot(ax - cx, ay - cy)
        c = math.hypot(ax - bx, ay - by)
        # twice the signed area
        area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        if a * b * c < 1e-9 or area2 < 1e-12:
            k[i] = 0.0                       # collinear => straight
        else:
            k[i] = 2.0 * area2 / (a * b * c)  # = 1 / circumradius
    if n >= 2:
        k[0] = k[1]
        k[-1] = k[-2]
    return k


def track_from_path(x, y, name: str = "Imported path", ds: float = 1.0,
                    smooth_window: int = 5, min_radius_m: float = 3.0,
                    straight_radius_m: float = 200.0,
                    closed: bool = True) -> Track:
    """
    Build a lap-sim Track from an ordered 2-D centreline — GPS breadcrumbs, or the
    midpoints between left/right cone rows. This replaces "manual only" track
    entry: drive (or walk) the course with a phone GPS, or drop the cone
    coordinates from the event map, and the sim runs your ACTUAL layout.

    Method: resample the path to a uniform arc-length step, compute local
    curvature (Menger), smooth it (cones/GPS are noisy and raw curvature is very
    spiky), then emit one short segment per station with radius = 1/curvature.
    Anything flatter than `straight_radius_m` becomes a straight; anything tighter
    than `min_radius_m` is clamped (real FSAE corners rarely beat ~3 m and noise
    can imply absurd radii). Never raises — returns an empty Track with a sane name
    on bad input, which the sim then reports as "track is empty".

    Units: metres. If you have lat/long, project to a local metric frame first
    (see `latlon_to_xy`).
    """
    try:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        if x.size < 3 or x.size != y.size:
            return Track(name=name, segments=[], ds=ds)

        if closed and (abs(x[0] - x[-1]) > 1e-6 or abs(y[0] - y[-1]) > 1e-6):
            x = np.append(x, x[0]); y = np.append(y, y[0])

        # cumulative arc length, then resample to uniform ds
        seg = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 0:
            return Track(name=name, segments=[], ds=ds)
        n_samp = max(int(round(total / max(ds, 0.25))), 3)
        s_u = np.linspace(0.0, total, n_samp)
        xu = np.interp(s_u, s, x)
        yu = np.interp(s_u, s, y)

        k = _menger_curvature(xu, yu)
        # smooth curvature with a centred moving average
        w = max(int(smooth_window), 1)
        if w > 1:
            kernel = np.ones(w) / w
            k = np.convolve(k, kernel, mode="same")

        ds_u = total / (n_samp - 1)
        segs: List[Segment] = []
        for ki in k[:-1]:
            if ki <= 1.0 / straight_radius_m:
                segs.append(Segment(ds_u, None))           # straight
            else:
                r = max(1.0 / ki, min_radius_m)            # clamp absurd tight noise
                segs.append(Segment(ds_u, r))
        return Track(name=name, segments=segs, ds=ds)
    except Exception:
        return Track(name=name, segments=[], ds=ds)


def cones_to_centerline(left_x, left_y, right_x, right_y):
    """
    Midpoint centreline from two ordered cone rows (left and right track edges).
    Rows may differ in length; the shorter is interpolated onto the longer's
    arc-length parameterisation so midpoints pair up sensibly. Returns (x, y).
    """
    lx = np.asarray(left_x, float); ly = np.asarray(left_y, float)
    rx = np.asarray(right_x, float); ry = np.asarray(right_y, float)
    if lx.size < 2 or rx.size < 2:
        return np.array([]), np.array([])

    def _arc(px, py):
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        return d / d[-1] if d[-1] > 0 else d

    # resample both rows to a common normalised parameter
    n = max(lx.size, rx.size)
    u = np.linspace(0, 1, n)
    lu = _arc(lx, ly); ru = _arc(rx, ry)
    lxi = np.interp(u, lu, lx); lyi = np.interp(u, lu, ly)
    rxi = np.interp(u, ru, rx); ryi = np.interp(u, ru, ry)
    return 0.5 * (lxi + rxi), 0.5 * (lyi + ryi)


def latlon_to_xy(lat, lon):
    """
    Project GPS latitude/longitude (degrees) to a local metric x/y frame using an
    equirectangular approximation about the path's mean latitude — accurate to a
    fraction of a percent over an FSAE-sized site (a few hundred metres), which is
    far below the curvature smoothing noise. Returns (x_east_m, y_north_m).
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat.size == 0:
        return np.array([]), np.array([])
    R = 6_371_000.0
    lat0 = math.radians(float(np.mean(lat)))
    x = np.radians(lon - np.mean(lon)) * math.cos(lat0) * R
    y = np.radians(lat - np.mean(lat)) * R
    return x, y


def _path_normals(x, y):
    """Unit normal (pointing left of travel) at each point of a 2-D path."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = np.gradient(x); dy = np.gradient(y)
    t = np.hypot(dx, dy); t[t < 1e-9] = 1e-9
    tx, ty = dx / t, dy / t
    return -ty, tx               # left normal


def optimise_racing_line(x, y, track_width_m: float = 3.0, closed: bool = True,
                         iterations: int = 60, n_offsets: int = 2000):
    """
    Find a faster line than the centreline by letting the car use track width.

    Real lap time is dominated by minimum corner radius, and a car is allowed to
    use the full width — entering wide, apexing late, exiting wide straightens a
    corner and raises its radius, hence its speed. This is the single biggest
    "free" lap-time lever a point-mass sim can expose, and it needs no extra data:
    just how wide the track is (FSAE autocross is ~3-4.5 m).

    Method: minimise total path curvature subject to staying within ±width/2 of
    the centreline. Curvature minimisation is the standard, well-behaved proxy for
    the minimum-time line on a point-mass model (the true min-time line differs
    only slightly and needs the full speed solver in the loop; this is the version
    that's fast, stable, and honest about being a curvature-optimal — not a
    fully-coupled minimum-time — line). Returns (x_line, y_line, offset) where
    offset is the signed lateral displacement (m, + = left) actually used.

    Iterative relaxation: each point moves toward the average of its neighbours
    (which reduces curvature), projected back inside the track edges. Converges to
    the smoothest admissible path.
    """
    try:
        x = np.asarray(x, float).ravel().copy()
        y = np.asarray(y, float).ravel().copy()
        n = x.size
        if n < 5 or x.size != y.size:
            return x, y, np.zeros(n)
        half = max(track_width_m, 0.2) / 2.0
        nx, ny = _path_normals(x, y)
        cx, cy = x.copy(), y.copy()            # fixed centreline reference
        offset = np.zeros(n)

        for _ in range(max(int(iterations), 1)):
            # desired smoothed position = average of neighbours
            if closed:
                xm = 0.5 * (np.roll(x, 1) + np.roll(x, -1))
                ym = 0.5 * (np.roll(y, 1) + np.roll(y, -1))
            else:
                xm = x.copy(); ym = y.copy()
                xm[1:-1] = 0.5 * (x[:-2] + x[2:])
                ym[1:-1] = 0.5 * (y[:-2] + y[2:])
            # move a fraction toward the smoothed target
            x_new = x + 0.5 * (xm - x)
            y_new = y + 0.5 * (ym - y)
            # project the displacement from centreline onto the normal and clamp
            off = (x_new - cx) * nx + (y_new - cy) * ny
            off = np.clip(off, -half, half)
            x = cx + off * nx
            y = cy + off * ny
            offset = off
        return x, y, offset
    except Exception:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        return x, y, np.zeros(x.size)


def compare_line_vs_centerline(veh, x, y, track_width_m: float = 3.0,
                               pt: Optional[Powertrain] = None,
                               ds: float = 1.0) -> dict:
    """
    Convenience: build the centreline track and the curvature-optimal racing-line
    track from the same path and report both lap times, so the value of using
    track width reads out directly in seconds. Returns a dict with both tracks,
    both LapResults, and the time gained. Never raises.
    """
    pt = pt or Powertrain()
    base_track = track_from_path(x, y, name="Centreline", ds=ds)
    lx, ly, off = optimise_racing_line(x, y, track_width_m=track_width_m)
    line_track = track_from_path(lx, ly, name="Racing line", ds=ds)
    base = simulate_lap(veh, base_track, pt)
    line = simulate_lap(veh, line_track, pt)
    gained = (base.lap_time_s - line.lap_time_s) if (base.ok and line.ok) else float("nan")
    return dict(centerline_track=base_track, line_track=line_track,
                centerline_result=base, line_result=line,
                line_x=lx, line_y=ly, offset=off, time_gained_s=gained)


def event_points_estimate(your_time: float, best_time: float,
                          event: str = "autocross") -> float:
    """
    FSAE-style points estimate so a time delta reads as what it's worth on the
    scoresheet. Uses the standard dynamic-event shape: points scale between a
    floor at the max allowed time (~145% of best) and the max at the best time.
    Returns a points estimate (0..~max) — indicative, for prioritisation only.
    """
    try:
        maxpts = {"skidpad": 75.0, "autocross": 125.0,
                  "endurance": 275.0}.get(event, 100.0)
        minpts = 0.05 * maxpts
        if your_time <= 0 or best_time <= 0 or not math.isfinite(your_time):
            return 0.0
        t_max = 1.45 * best_time
        if your_time >= t_max:
            return minpts
        frac = (t_max / your_time - 1.0) / (t_max / best_time - 1.0)
        return float(minpts + (maxpts - minpts) * max(0.0, min(frac, 1.0)))
    except Exception:
        return 0.0
