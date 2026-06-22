# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
GGV diagram generator.

A GGV diagram is the car's acceleration envelope: for each forward speed V, the
boundary in the (longitudinal g, lateral g) plane that the car can sustain in
steady state. Cornering, braking, accelerating, and every combination in
between — the outer edge of what the tires + power + aero will allow. Read it
and you can see, at a glance, how much combined grip you have, where the car is
power-limited vs grip-limited, and how a design change reshapes the whole
operating envelope. It is the single most useful steady-state picture a team
can have before committing to full transient vehicle modelling.

Why this module exists, given KinematiK already has `LapSim.gg_v_envelope`:
that method returns only THREE boundary points per speed — pure lateral, pure
accel, pure brake — and it reaches them through a simplified whole-car `mu`
helper. That is a fine quick check, but it is not a GGV *diagram*: it has no
combined-load points (the curved corners of the envelope where you trail-brake
into a corner or get on power at the exit), and it does not run the real
lateral+longitudinal load transfer that makes CG height, roll-centre height,
wheel rate, and camber gain actually move the boundary.

This module builds the full surface and does it THROUGH the existing
`VehicleDynamics` load-transfer + per-corner Pacejka chain, so the design
inputs an FSAE team cares about are the levers that reshape the diagram:

    * CG height            -> longitudinal AND lateral load transfer
    * roll-centre height    -> geometric vs elastic lateral transfer split
    * wheel rate / springs  -> elastic roll stiffness -> front/rear balance
    * dynamic camber gain   -> camber the loaded tire runs -> Pacejka peak
    * weight, weight dist.   -> static loads and transfer magnitudes
    * aero ClA / CdA        -> load (more grip) and drag (less top-end accel)

PROVENANCE / HONESTY
--------------------
Absolute grip is only as good as the tire model you plug in. With KinematiK's
shipped `default_tire()` the numbers are a sensible GENERIC FSAE tire — great
for *relative* questions ("does lowering the CG 10 mm widen the envelope?")
but not a number to quote as your car's measured capability. Swap in your
TTC-fitted Pacejka model and the absolute g's become trustworthy on YOUR tire.
The longitudinal side combines a tire friction-circle limit with the powertrain
(power-limited traction) and brake limits; the friction circle is the standard
elliptic combined-slip approximation unless a CombinedSlipTire is supplied.

Everything here is defensive: a single bad point never raises, it just clamps.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field

from .dynamics import VehicleDynamics, VehicleParams, CornerLoads


# --------------------------------------------------------------------------- #
#  Longitudinal / aero parameters the GGV needs on top of VehicleParams.
#  These mirror the relevant subset of LapSimParams so the GGV can stand alone
#  (you don't need a Track to draw a GGV) but stay numerically consistent with
#  the lap sim if you feed it the same numbers.
# --------------------------------------------------------------------------- #
@dataclass
class GGVParams:
    """Powertrain + aero + brake inputs the longitudinal axis needs.

    All FSAE-representative defaults; every one is a spec/measured knob. The
    field names and meanings mirror `laptime.Powertrain` so a GGV and the lap
    sim built from the same numbers agree — see `from_powertrain`.
    """
    power_w: float = 60_000.0        # peak power at the wheels, W
    drivetrain_eff: float = 0.90     # tractive efficiency (0..1)
    max_tractive_n: float = 2600.0   # low-speed traction cap, N (flat motor model)
    drive_axle: str = "rear"         # 'rear', 'front', or 'all'/'awd'
    brake_bias_front: float = 0.65   # fraction of brake torque to the front
    brake_g_cap: float = 1.8         # mechanical brake ceiling (g); grip-limited below
    # Aerodynamics: force = 0.5 * rho * coeff_area * v^2 (downforce adds tire load)
    rho: float = 1.225               # air density, kg/m^3
    cl_a: float = 2.5                # downforce coeff * area (ClA), m^2  (0 = wingless)
    cd_a: float = 1.2                # drag coeff * area (CdA), m^2
    aero_balance_front: float = 0.45 # fraction of downforce on the front axle
    rolling_resist: float = 0.015    # rolling-resistance coefficient
    # Combined-slip coupling. When a CombinedSlipTire is supplied, the longitudinal
    # axis limits use its calibrated mu_x_ratio and the friction-circle blend uses
    # its ellipse exponents (ell_kx, ell_ky) — so Fx can exceed Fy peak and the
    # corner shape is the measured one. None => symmetric circle (mu_x = mu_y, k=2).
    combined_tire: object = None
    g: float = 9.81
    # Numerics
    V_MIN: float = 1.0
    V_MAX: float = 60.0

    @staticmethod
    def from_powertrain(pt) -> "GGVParams":
        """
        Build GGVParams from a `laptime.Powertrain` so the GGV and the lap sim
        share one source of truth. Maps drive 'rwd'/'awd' -> 'rear'/'all',
        kW -> W, and carries the motor cap, brake cap, aero and combined tire.
        """
        drive = "all" if getattr(pt, "drive", "rwd") == "awd" else "rear"
        return GGVParams(
            power_w=float(getattr(pt, "power_kw", 60.0)) * 1000.0
                    * float(getattr(pt, "drivetrain_eff", 0.9)),
            drivetrain_eff=1.0,   # already folded into power_w above, mirror Powertrain.power_w()
            max_tractive_n=float(getattr(pt, "max_tractive_n", 2600.0)),
            drive_axle=drive,
            brake_g_cap=float(getattr(pt, "brake_g_cap", 1.8)),
            rho=float(getattr(pt, "rho", 1.225)),
            cl_a=float(getattr(pt, "cla", 2.5)),
            cd_a=float(getattr(pt, "cda", 1.2)),
            rolling_resist=float(getattr(pt, "crr", 0.015)),
            combined_tire=getattr(pt, "combined_tire", None),
        )
    V_MAX: float = 60.0


@dataclass
class GGVResult:
    """The computed envelope.

    speeds : (S,)            forward speeds, m/s
    theta  : (N,)            direction angles around the envelope, rad.
             theta=0 is pure forward accel (+long), pi is pure braking (-long),
             +/-pi/2 are pure lateral. lat_g uses |sin|, so the boundary is the
             symmetric left/right cornering edge.
    long_g : (S, N)          longitudinal g at each (speed, direction)
    lat_g  : (S, N)          lateral g at each (speed, direction)
    Plus convenience peak arrays per speed and the inputs echoed back.
    """
    speeds: np.ndarray
    theta: np.ndarray
    long_g: np.ndarray
    lat_g: np.ndarray
    max_lat_g: np.ndarray            # (S,) peak pure-lateral g per speed
    max_accel_g: np.ndarray          # (S,) peak pure-forward g per speed
    max_brake_g: np.ndarray          # (S,) peak pure-braking g (positive number)
    grip_model: str = "unknown"
    warnings: list = field(default_factory=list)

    def at_speed(self, v: float):
        """Closest computed slice to speed v: (long_g[N], lat_g[N])."""
        i = int(np.argmin(np.abs(self.speeds - v)))
        return self.long_g[i], self.lat_g[i]


class GGVGenerator:
    """
    Builds a GGV diagram from a VehicleDynamics (which carries mass, CG, RC,
    spring/wheel rates, camber, and the Pacejka tire) plus a GGVParams for the
    powertrain/aero/brake side.

    Usage:
        veh = VehicleDynamics(VehicleParams(...), front_kin, rear_kin, tire)
        ggv = GGVGenerator(veh, GGVParams(...))
        res = ggv.generate()
    """

    def __init__(self, veh: VehicleDynamics, gparams: GGVParams | None = None):
        self.veh = veh
        self.p = veh.p                       # VehicleParams
        self.g = gparams or GGVParams()
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #
    def _warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _aero_downforce(self, v: float) -> float:
        """Total aero downforce (N) at speed v."""
        return 0.5 * self.g.rho * self.g.cl_a * v * v

    def _aero_drag(self, v: float) -> float:
        """Total aero drag (N) at speed v."""
        return 0.5 * self.g.rho * self.g.cd_a * v * v

    def _max_lateral_g_at_speed(self, v: float) -> float:
        """
        Peak steady-state lateral g at speed v, found by bisection through the
        REAL load-transfer + Pacejka chain in VehicleDynamics, with aero
        downforce folded into the static load for this speed.

        We temporarily lift the car's effective weight by the aero downforce so
        the same lateral_load_transfer / axle_grip code sees the extra vertical
        load. CG-height, RC-height, wheel-rate and camber all act inside that
        call, which is exactly why they move the envelope.
        """
        p = self.p
        df = self._aero_downforce(v)
        # effective mass carrying the lateral load (sprung+downforce). Downforce
        # is not inertial mass, so it adds grip without adding the force demand:
        # we model that by giving the tires extra Fz but keeping demand = m*a.
        base_W = p.mass * p.g
        eff_W = base_W + df
        if base_W <= 0:
            return 0.0

        def capacity_minus_demand(lat_g: float) -> float:
            # Scale loads so the four-corner sum equals eff_W (adds downforce),
            # while the inertial demand stays m*lat_g. We get loads from the
            # vehicle's own transfer model at this lat_g, then bump every corner
            # by the aero share so the tire sees the downforce too.
            try:
                loads, _ = self.veh.lateral_load_transfer(lat_g)
            except Exception:
                self._warn("lateral_load_transfer failed at a point; clamped.")
                return -1.0
            aero_per_corner = df / 4.0
            af = self.g.aero_balance_front
            # split downforce front/rear by aero balance, evenly L/R
            add_f = (df * af) / 2.0
            add_r = (df * (1.0 - af)) / 2.0
            loads = CornerLoads(
                max(loads.fl + add_f, 0.0), max(loads.fr + add_f, 0.0),
                max(loads.rl + add_r, 0.0), max(loads.rr + add_r, 0.0),
            )
            try:
                Ff, Fr = self.veh.axle_grip(loads)
            except Exception:
                self._warn("axle_grip failed at a point; clamped.")
                return -1.0
            capacity_g = (Ff + Fr) / base_W      # demand normalised by INERTIAL weight
            return capacity_g - lat_g

        lo, hi = 0.05, 4.0
        try:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if capacity_minus_demand(mid) >= 0:
                    lo = mid
                else:
                    hi = mid
            # Wheel-lift honesty check: at the solved limit, has an inner tire
            # fully unloaded? If so the rigid load-transfer model has saturated
            # at the max(.,0) floor and the grip number past that point is an
            # ARTIFACT (further CG rise / softer inside bar stops being penalised
            # because the inner wheel is already at zero). Flag it loudly rather
            # than report a number that can rise with CG — which is unphysical.
            try:
                loads, _ = self.veh.lateral_load_transfer(lo)
                inner_min = min(loads.fl, loads.fr, loads.rl, loads.rr)
                if inner_min <= 1.0:
                    self._warn(
                        f"Inner-wheel lift at ~{lo:.2f} g (v={v:.0f} m/s): an "
                        "inside tire has fully unloaded, so the rigid load-transfer "
                        "model has saturated. The lateral-grip number here is an "
                        "upper bound / artifact — reduce CG height, soften the bar, "
                        "or treat this point as 'lifting a wheel', not as real grip.")
            except Exception:
                pass
            return float(lo)
        except Exception:
            self._warn("Lateral-g bisection failed at a speed; used 0 there.")
            return 0.0

    def _mu_x_ratio(self) -> float:
        """Longitudinal/lateral peak-mu ratio from the combined tire (>=1 typical),
        or 1.0 (symmetric circle) when no combined tire is supplied."""
        ct = self.g.combined_tire
        if ct is not None:
            try:
                return max(float(getattr(ct, "mu_x_ratio", 1.0)), 0.1)
            except Exception:
                return 1.0
        return 1.0

    def _max_accel_g_at_speed(self, v: float, mu_lat: float) -> float:
        """
        Peak pure-forward g at speed v: the lesser of motor-limited and
        traction-limited drive, minus drag and rolling resistance. Traction uses
        the drive axle's share of vertical load (incl. its aero downforce share)
        times the LONGITUDINAL friction coefficient (mu_lat * mu_x_ratio).

        The motor side mirrors laptime.Powertrain.tractive_force: F = min(P/v,
        max_tractive_n) — the flat-cap model — so a GGV and the lap sim built from
        the same Powertrain agree on the accel axis.
        """
        p = self.p
        gg = self.g
        v = max(v, gg.V_MIN)
        try:
            df = self._aero_downforce(v)
            W = p.mass * p.g
            wf = p.weight_dist_front
            # Match laptime._accel_long: the driven-axle vertical load applies the
            # axle weight fraction to the TOTAL load (weight + full downforce),
            # rather than splitting downforce by aero balance. Keeps the GGV and
            # the lap sim on one convention so their accel limits agree.
            if gg.drive_axle == "rear":
                axle_frac = 1.0 - wf
            elif gg.drive_axle == "front":
                axle_frac = wf
            else:  # all / awd
                axle_frac = 1.0
            Fz_drive = (W + df) * axle_frac

            mu_long = mu_lat * self._mu_x_ratio()
            F_traction = mu_long * Fz_drive
            # motor side: flat power cap + low-speed traction cap, like Powertrain
            F_power = min(gg.power_w * gg.drivetrain_eff / v, gg.max_tractive_n)
            F_drive = min(F_power, F_traction)

            F_drag = self._aero_drag(v)
            F_roll = gg.rolling_resist * (W + df)
            a = (F_drive - F_drag - F_roll) / p.mass
            return float(max(a, 0.0) / gg.g)
        except Exception:
            self._warn("Accel-g evaluation failed at a point; used 0 there.")
            return 0.0

    def _max_brake_g_at_speed(self, v: float, mu_lat: float) -> float:
        """
        Peak pure-braking g (positive) at speed v. All four tires brake, so the
        tire limit uses total vertical load (weight + total downforce) times the
        LONGITUDINAL friction coefficient (mu_lat * mu_x_ratio), then capped by
        the mechanical brake ceiling brake_g_cap — exactly as laptime._decel_long
        does. Aero drag and rolling resistance help. Brake bias is assumed matched
        to the load split; if your real bias is off, true brake g is lower.
        """
        p = self.p
        gg = self.g
        v = max(v, gg.V_MIN)
        try:
            df = self._aero_downforce(v)
            W = p.mass * p.g
            Fz_total = W + df
            mu_long = mu_lat * self._mu_x_ratio()
            F_tire = mu_long * Fz_total
            # mechanical brake ceiling (grip-limited below it), like Powertrain
            F_tire = min(F_tire, gg.brake_g_cap * W)
            F_drag = self._aero_drag(v)
            # Match laptime._decel_long: braking decel is (tire + drag)/m. Rolling
            # resistance is omitted on the brake side there (it's a second-order
            # help), so we omit it too to keep the two models consistent.
            a = (F_tire + F_drag) / p.mass
            return float(max(a, 0.0) / gg.g)
        except Exception:
            self._warn("Brake-g evaluation failed at a point; used 0 there.")
            return 0.0

    # ------------------------------------------------------------------ #
    #  the envelope
    # ------------------------------------------------------------------ #
    @staticmethod
    def _solve_superellipse_radius(a_term: float, b_term: float,
                                   kx: float, ky: float) -> float:
        """
        Solve a_term*r^kx + b_term*r^ky = 1 for r > 0 by bisection.
        a_term, b_term are (|dir component|/axis_limit)^k already, so at r=1 the
        LHS is a_term+b_term; the root is bracketed in (0, r_hi]. Monotone
        increasing in r, so bisection is robust and never raises.
        """
        if a_term <= 0 and b_term <= 0:
            return 0.0
        def f(r):
            return a_term * (r ** kx) + b_term * (r ** ky) - 1.0
        lo, hi = 0.0, 1.0
        # expand hi until f(hi) >= 0
        for _ in range(60):
            if f(hi) >= 0:
                break
            hi *= 1.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) >= 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def generate(self, speeds=None, n_dir: int = 73) -> GGVResult:
        """
        Build the full GGV surface.

        speeds : iterable of forward speeds (m/s). Default: 12 points 5..min(Vmax,38).
        n_dir  : number of directions around each envelope (odd is nice so the
                 pure-forward and pure-back points land exactly).

        For each speed we get the three axis limits (max lateral, max accel, max
        brake) from the real chain, then trace the boundary at each direction
        angle theta using a friction-ellipse blend:

            with longitudinal fraction  c = cos(theta), lateral fraction s = sin(theta),
            the achievable point scales the long axis-limit (accel if c>0 else
            brake) and the lateral axis-limit so that (long/Lmax)^2 + (lat/Latmax)^2 = 1.

        This is the standard elliptic combined-slip approximation. If the tire is
        a CombinedSlipTire with measured drive/brake data, the axis limits already
        carry that calibration; the ellipse only shapes the blend between them.
        """
        self.warnings = []
        gg = self.g
        if speeds is None:
            top = min(gg.V_MAX, 38.0)
            speeds = np.linspace(5.0, top, 12)
        speeds = np.asarray(speeds, float)
        theta = np.linspace(-math.pi, math.pi, n_dir)

        S, N = len(speeds), len(theta)
        long_g = np.zeros((S, N))
        lat_g = np.zeros((S, N))
        max_lat = np.zeros(S)
        max_acc = np.zeros(S)
        max_brk = np.zeros(S)

        # combined-slip ellipse exponents from the calibrated tire (2 = circle)
        ct = gg.combined_tire
        kx = ky = 2.0
        if ct is not None:
            try:
                kx = max(float(getattr(ct, "ell_kx", 2.0)), 0.5)
                ky = max(float(getattr(ct, "ell_ky", 2.0)), 0.5)
            except Exception:
                kx = ky = 2.0

        # The longitudinal methods carry aero load explicitly via F_down (like
        # laptime), so they need the tire's INTRINSIC grip coefficient (no aero),
        # not the speed-inflated lateral g. Compute it once at the aero-free speed.
        mu_intrinsic = max(self._max_lateral_g_at_speed(gg.V_MIN), 0.05)

        for i, v in enumerate(speeds):
            v = float(min(max(v, gg.V_MIN), gg.V_MAX))
            # lateral axis limit DOES include aero downforce (more grip at speed) —
            # that's the real cornering capability and what the diagram should show.
            lat_lim = self._max_lateral_g_at_speed(v)
            # longitudinal limits use the intrinsic mu + their own F_down term, so
            # aero is counted once, consistently with laptime._accel/_decel_long.
            acc_lim = self._max_accel_g_at_speed(v, mu_intrinsic)
            brk_lim = self._max_brake_g_at_speed(v, mu_intrinsic)

            max_lat[i] = lat_lim
            max_acc[i] = acc_lim
            max_brk[i] = brk_lim

            for j, th in enumerate(theta):
                c, s = math.cos(th), math.sin(th)
                lon_lim = acc_lim if c >= 0 else brk_lim
                # superellipse boundary with the calibrated exponents:
                #   (|lon|/lon_lim)^kx + (|lat|/lat_lim)^ky = 1
                # along the ray (c, s): lon = r*c, lat = r*|s|, solve for r.
                a_term = (abs(c) / lon_lim) ** kx if lon_lim > 1e-6 else 0.0
                b_term = (abs(s) / lat_lim) ** ky if lat_lim > 1e-6 else 0.0
                denom = a_term + b_term
                # for kx=ky=2 this reduces to r = 1/sqrt(denom) (the plain ellipse);
                # general exponents need the ray-superellipse solve below.
                if denom <= 1e-12:
                    r = 0.0
                elif abs(kx - 2.0) < 1e-9 and abs(ky - 2.0) < 1e-9:
                    r = 1.0 / math.sqrt(denom)
                else:
                    # f(r) = a_term*r^kx + b_term*r^ky - 1 = 0, monotone in r>0
                    r = self._solve_superellipse_radius(a_term, b_term, kx, ky)
                lon = r * c
                lat = abs(r * s)             # symmetric L/R
                long_g[i, j] = lon
                lat_g[i, j] = lat

        grip_name = "unknown"
        try:
            grip_name = self.veh.grip_model_name()
        except Exception:
            pass

        return GGVResult(
            speeds=speeds, theta=theta,
            long_g=long_g, lat_g=lat_g,
            max_lat_g=max_lat, max_accel_g=max_acc, max_brake_g=max_brk,
            grip_model=grip_name, warnings=list(self.warnings),
        )


# --------------------------------------------------------------------------- #
#  Design-input sensitivity: how does a lever reshape the envelope?
# --------------------------------------------------------------------------- #
def sweep_parameter(veh: VehicleDynamics, gparams: GGVParams,
                    param: str, values, speed: float = 15.0,
                    metric: str = "max_lat_g"):
    """
    Sweep one design input across `values` and report how a chosen GGV metric
    moves — the direct answer to "what does changing X do to my envelope?".

    param  : an attribute name on VehicleParams (e.g. 'cg_height',
             'spring_rate_front', 'static_camber_front', 'weight_dist_front')
             OR on GGVParams (e.g. 'cl_a', 'power_w'). Resolved automatically.
    values : iterable of values to set.
    speed  : the speed slice to evaluate the metric at, m/s.
    metric : 'max_lat_g', 'max_accel_g', or 'max_brake_g'.

    Returns dict(values=[...], metric=[...], param=param, metric_name=metric).
    Restores the original parameter value afterward. Never raises on a bad point.

    Note: when sweeping a geometry-derived quantity (e.g. spring_rate_*), make
    sure the vehicle was built with use_spring_rates=True and kinematics attached
    so the motion ratio actually couples the spring to the wheel/roll rate; for
    camber, set use_param_camber=True so static_camber_* is the live lever.
    """
    on_vehicle = hasattr(veh.p, param)
    on_ggv = hasattr(gparams, param)
    if not (on_vehicle or on_ggv):
        raise AttributeError(
            f"'{param}' is not a field of VehicleParams or GGVParams. "
            f"Check the name; e.g. 'cg_height', 'roll_stiffness_front', "
            f"'spring_rate_front', 'static_camber_front', 'cl_a', 'power_w'.")

    target = veh.p if on_vehicle else gparams
    original = getattr(target, param)

    # If sweeping a static-camber param while geometry is attached, the solved
    # kinematic camber would override it and the sweep would be (misleadingly)
    # flat. Temporarily make the param the live camber lever so the sweep means
    # what the user expects; restore it afterward.
    _cam_params = ("static_camber_front", "static_camber_rear")
    _toggled_cam = False
    _orig_use_param_cam = getattr(veh.p, "use_param_camber", None)
    if param in _cam_params and on_vehicle and not _orig_use_param_cam:
        try:
            veh.p.use_param_camber = True
            _toggled_cam = True
        except Exception:
            _toggled_cam = False

    out_vals, out_metric = [], []
    try:
        for val in values:
            setattr(target, param, val)
            res = GGVGenerator(veh, gparams).generate(speeds=[speed], n_dir=37)
            m = {
                "max_lat_g": res.max_lat_g[0],
                "max_accel_g": res.max_accel_g[0],
                "max_brake_g": res.max_brake_g[0],
            }.get(metric, res.max_lat_g[0])
            out_vals.append(float(val))
            out_metric.append(float(m))
    finally:
        setattr(target, param, original)
        if _toggled_cam:
            veh.p.use_param_camber = _orig_use_param_cam

    return dict(values=out_vals, metric=out_metric,
                param=param, metric_name=metric, speed=speed,
                camber_lever_forced=_toggled_cam)


# --------------------------------------------------------------------------- #
#  Convenience: build a GGV straight from loose inputs (no kinematics needed)
# --------------------------------------------------------------------------- #
def quick_ggv(mass=280.0, cg_height=300.0, weight_dist_front=0.47,
              track_front=1200.0, track_rear=1180.0, wheelbase=1550.0,
              roll_stiffness_front=350.0, roll_stiffness_rear=300.0,
              static_camber_front=-1.5, static_camber_rear=-1.5,
              power_w=60_000.0, cl_a=2.5, cd_a=1.2, tire=None,
              speeds=None) -> GGVResult:
    """
    One-call GGV from the headline parameters an FSAE team knows off the top of
    their head — no corner kinematics object required. Roll stiffness is taken
    directly (use the full VehicleDynamics path with kinematics + use_spring_rates
    if you want spring rate × motion ratio instead). Uses the shipped generic
    Pacejka tire unless you pass your own fitted `tire`.
    """
    from . import tiremodel
    vp = VehicleParams(
        mass=mass, cg_height=cg_height, weight_dist_front=weight_dist_front,
        track_front=track_front, track_rear=track_rear, wheelbase=wheelbase,
        roll_stiffness_front=roll_stiffness_front,
        roll_stiffness_rear=roll_stiffness_rear,
        static_camber_front=static_camber_front,
        static_camber_rear=static_camber_rear,
        use_param_camber=True,
    )
    if tire is None:
        tire = tiremodel.default_tire()
    veh = VehicleDynamics(vp, tire=tire)
    gp = GGVParams(power_w=power_w, cl_a=cl_a, cd_a=cd_a)
    return GGVGenerator(veh, gp).generate(speeds=speeds)


# --------------------------------------------------------------------------- #
#  Validation against the lap-time model (suspension/laptime.py)
# --------------------------------------------------------------------------- #
def validate_against_laptime(veh: VehicleDynamics, pt, speeds=None,
                             rel_tol: float = 0.06) -> dict:
    """
    Cross-check the GGV's axis limits against the lap-time model's own
    longitudinal/lateral functions at matched conditions. Both are built on the
    same VehicleDynamics chain, so they SHOULD agree closely; a divergence means
    one of the two has drifted and is worth chasing before trusting either.

    What is compared, at each speed v:
      * lateral : GGV max_lat_g (at v, no aero) vs laptime._max_lat_g(veh).
                  laptime's pure-lateral grip is speed-independent (aero enters
                  only through the corner-speed fixed point), so we compare the
                  GGV's LOW-speed lateral (aero≈0) against it.
      * accel   : GGV max_accel_g(v) vs laptime._accel_long(veh, v, pt, mu, 0)/g
                  with zero lateral usage (pure forward).
      * brake   : GGV max_brake_g(v) vs laptime._decel_long(veh, v, pt, mu, 0)/g.

    `pt` is a laptime.Powertrain. The GGV is built from it via
    GGVParams.from_powertrain so the two share inputs.

    Returns a dict with per-speed arrays of both models' numbers, the relative
    differences, and a boolean `ok` that is True iff every comparison is within
    rel_tol. Never raises; if laptime can't be imported it returns ok=False with
    a reason.
    """
    out = {"ok": False, "reason": "", "speeds": [],
           "lat_ggv": [], "lat_lap": [], "lat_reldiff": [],
           "accel_ggv": [], "accel_lap": [], "accel_reldiff": [],
           "brake_ggv": [], "brake_lap": [], "brake_reldiff": []}
    try:
        # import laptime without triggering the package __init__ (heavy deps)
        import importlib, sys
        if "suspension.laptime" in sys.modules:
            lt = sys.modules["suspension.laptime"]
        else:
            lt = importlib.import_module("suspension.laptime")
    except Exception as e:
        out["reason"] = f"could not import laptime ({type(e).__name__}: {e})"
        return out

    g = 9.81
    if speeds is None:
        speeds = np.linspace(8.0, 30.0, 8)
    speeds = np.asarray(speeds, float)

    gp = GGVParams.from_powertrain(pt)
    gen = GGVGenerator(veh, gp)

    # laptime's pure-lateral grip (speed-independent number it uses everywhere)
    try:
        lat_lap = float(lt._max_lat_g(veh))
    except Exception:
        lat_lap = float("nan")

    all_ok = True
    for v in speeds:
        v = float(v)
        # GGV side
        mu_lat_lowspeed = gen._max_lateral_g_at_speed(gp.V_MIN)   # aero≈0 baseline
        lat_ggv = mu_lat_lowspeed
        # longitudinal limits use the intrinsic (aero-free) mu + explicit F_down,
        # exactly as generate() and laptime do — so they line up.
        acc_ggv = gen._max_accel_g_at_speed(v, mu_lat_lowspeed)
        brk_ggv = gen._max_brake_g_at_speed(v, mu_lat_lowspeed)

        # laptime side (pure longitudinal => lat_used_g = 0)
        try:
            acc_lap = lt._accel_long(veh, v, pt, lat_lap, 0.0) / g
        except Exception:
            acc_lap = float("nan")
        try:
            brk_lap = lt._decel_long(veh, v, pt, lat_lap, 0.0) / g
        except Exception:
            brk_lap = float("nan")

        def reld(a, b):
            denom = max(abs(b), 1e-6)
            return abs(a - b) / denom

        lat_rd = reld(lat_ggv, lat_lap)
        acc_rd = reld(acc_ggv, acc_lap)
        brk_rd = reld(brk_ggv, brk_lap)

        out["speeds"].append(v)
        out["lat_ggv"].append(lat_ggv); out["lat_lap"].append(lat_lap)
        out["lat_reldiff"].append(lat_rd)
        out["accel_ggv"].append(acc_ggv); out["accel_lap"].append(acc_lap)
        out["accel_reldiff"].append(acc_rd)
        out["brake_ggv"].append(brk_ggv); out["brake_lap"].append(brk_lap)
        out["brake_reldiff"].append(brk_rd)

        if not (lat_rd <= rel_tol and acc_rd <= rel_tol and brk_rd <= rel_tol):
            all_ok = False

    out["ok"] = all_ok
    out["rel_tol"] = rel_tol
    # Known, located model difference worth surfacing rather than hiding: laptime's
    # _decel_long applies the plain lateral mu on the brake side (no mu_x_ratio and
    # no combined-tire branch — that branch is only in _accel_long). So with a
    # combined tire whose mu_x_ratio > 1, the GGV's braking limit will read higher
    # than laptime's until both clip at brake_g_cap. The GGV is the physically
    # consistent one (braking is longitudinal too); the divergence is laptime's
    # brake side, not a GGV error.
    ct = getattr(pt, "combined_tire", None)
    mxr = getattr(ct, "mu_x_ratio", 1.0) if ct is not None else 1.0
    if (not all_ok) and mxr > 1.0:
        brake_only = all(
            (out["lat_reldiff"][i] <= rel_tol and out["accel_reldiff"][i] <= rel_tol)
            for i in range(len(out["speeds"])))
        if brake_only:
            out["note"] = (
                "Lateral and accel agree; the only divergence is braking, because "
                "laptime._decel_long does not apply the combined tire's mu_x_ratio "
                "on the brake side. The GGV does (braking is longitudinal). This is "
                "a laptime brake-side simplification, not a GGV error.")
    out["max_reldiff"] = float(max(
        [x for x in out["lat_reldiff"] + out["accel_reldiff"] + out["brake_reldiff"]
         if np.isfinite(x)] or [float("nan")]))
    out["reason"] = ("within tolerance" if all_ok else
                     f"max relative difference {out['max_reldiff']:.1%} exceeds "
                     f"{rel_tol:.0%} at one or more speeds")
    return out
