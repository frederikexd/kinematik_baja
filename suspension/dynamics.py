# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Vehicle-level dynamics built on top of the corner kinematics.

This module turns single-corner geometry into the numbers that decide how the
car actually behaves: lateral load transfer split front/rear, individual tire
vertical loads in a steady-state corner, roll-centre heights and migration, and
a simple load-sensitive grip model so you can see understeer/oversteer balance
shift as you change geometry. This is the part spreadsheets do badly and the
reason a coupled kinematics+dynamics tool is worth open-sourcing.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .kinematics import SuspensionKinematics, Hardpoints
from .tiremodel import PacejkaLateral


@dataclass
class VehicleParams:
    # NOTE on provenance: the tyre constants below (tire_load_sens, mu_peak) are
    # representative FSAE PLACEHOLDERS, not measured values. They make the balance
    # model useful for COMPARING setups (which way does the balance move?), not for
    # predicting absolute grip. Replace with values from your own tyre data (e.g. the
    # FSAE Tire Test Consortium) before trusting absolute numbers.
    mass: float = 280.0            # total mass incl. driver, kg
    cg_height: float = 300.0       # mm above ground
    wheelbase: float = 1550.0      # mm
    track_front: float = 1200.0    # mm
    track_rear: float = 1180.0     # mm
    weight_dist_front: float = 0.47  # fraction of mass on front axle
    roll_stiffness_front: float = 350.0  # N·m/deg  (used directly UNLESS use_spring_rates)
    roll_stiffness_rear: float = 300.0   # N·m/deg
    # ----------------------------------------------------------------------- #
    #  Spring rates + anti-roll bars. When use_spring_rates=True AND a corner
    #  kinematics object is attached, axle roll stiffness is DERIVED from the
    #  spring rate through the real motion ratio (wheel rate = k_spring * MR**2)
    #  plus the ARB contribution — rather than taken from roll_stiffness_*. This
    #  is the whole point of the rocker geometry: a quoted spring rate only maps
    #  to a wheel/roll rate THROUGH the motion ratio, so the optimiser sweeps the
    #  thing you actually buy (springs/bars), not an abstract roll stiffness.
    spring_rate_front: float = 35.0   # N/mm, coil rate at the spring/damper
    spring_rate_rear: float = 35.0    # N/mm
    arb_rate_front: float = 0.0       # N·m/deg of body roll, added at the axle
    arb_rate_rear: float = 0.0        # N·m/deg
    use_spring_rates: bool = False
    tire_load_sens: float = 0.00018  # 1/N: mu drop per N of vertical load (LINEAR fallback only)
    mu_peak: float = 1.55          # peak friction at light load (LINEAR fallback only)
    # Static camber per axle (deg, negative = top inboard). Fed to the Pacejka model
    # so the grip number reflects the camber the loaded outside tire actually runs.
    # When a front/rear kinematics object is supplied, the SOLVED camber is used
    # instead of these — these are the fallback when geometry isn't attached.
    static_camber_front: float = -1.5
    static_camber_rear: float = -1.5
    # When True, the grip model uses static_camber_* directly instead of the solved
    # kinematic camber. The setup-sensitivity/optimiser sets this so it can explore
    # camber as a free design lever; the live geometry tabs leave it False so the
    # camber the tires see comes from the actual linkage.
    use_param_camber: bool = False
    g: float = 9.81


@dataclass
class CornerLoads:
    fl: float
    fr: float
    rl: float
    rr: float

    def as_tuple(self):
        return (self.fl, self.fr, self.rl, self.rr)


class VehicleDynamics:
    def __init__(self, params: VehicleParams,
                 front_kin: SuspensionKinematics | None = None,
                 rear_kin: SuspensionKinematics | None = None,
                 tire: PacejkaLateral | None = None):
        """
        params : vehicle mass/geometry/stiffness.
        front_kin / rear_kin : corner kinematics. When present, the SOLVED camber at
            each axle is fed to the tire model, so geometry changes show up in grip.
        tire : a Pacejka lateral model. When supplied, grip and balance run on the
            real Magic Formula (load-sensitive, camber-aware). When None, the model
            falls back to the legacy linear placeholder (mu = mu_peak - load_sens*Fz)
            so nothing breaks if no tire is loaded — but absolute grip is then only
            indicative. Pass tiremodel.default_tire() for sensible generic behaviour,
            or your TTC-fitted model for numbers you can trust on YOUR tire.
        """
        self.p = params
        self.front_kin = front_kin
        self.rear_kin = rear_kin
        self.tire = tire
        # Per-instance caches for geometry-derived quantities that do NOT depend
        # on lateral_g. max_lateral_g() bisects with ~40 calls to
        # lateral_load_transfer(), each of which previously re-solved the rocker
        # linkage (motion_ratio) and roll-centre kinematics from scratch. Those
        # are functions of the (immutable) hardpoints + params only, so we solve
        # them once per VehicleDynamics and reuse the result. A fresh
        # VehicleDynamics is built per setup evaluation, so the cache lifetime is
        # exactly one setup — no risk of going stale across param changes.
        self._roll_stiffness_cache: dict[str, float] = {}

    # ---------------- camber actually seen by each axle ------------------ #
    def _axle_camber_rad(self, axle: str) -> float:
        """
        Inclination angle (rad, magnitude) the loaded tire on `axle` runs at static
        ride height. Priority:
          1. an explicit camber override on params (use_param_camber=True) — this is
             what the setup-sweep/optimiser uses to vary camber as a free design
             lever even when geometry is attached;
          2. otherwise the SOLVED kinematics camber when geometry is attached, so
             real geometry changes flow through to grip;
          3. otherwise the static_camber_* param.
        The Pacejka model treats camber magnitude as grip-relevant, so we pass |cam|.
        """
        kin = self.front_kin if axle == "front" else self.rear_kin
        param_cam = (self.p.static_camber_front if axle == "front"
                     else self.p.static_camber_rear)
        if getattr(self.p, "use_param_camber", False):
            cam_deg = param_cam
        elif kin is not None:
            cam_deg = kin.static.camber
        else:
            cam_deg = param_cam
        return abs(np.radians(cam_deg))

    # ---------------- roll stiffness from springs + geometry ------------- #
    def _axle_roll_stiffness(self, axle: str) -> float:
        """
        Axle roll stiffness in N·m/deg actually used by the load-transfer model.

        Two modes:
          * use_spring_rates=True and the corresponding corner kinematics is
            attached → DERIVE it from the spring rate through the real motion
            ratio plus the ARB:
                k_wheel [N/mm] = k_spring [N/mm] * MR**2          (MR=spring/wheel)
                K_spring_roll [N·m/deg] = k_wheel*1000 * (track_m**2 / 2) * (pi/180)
                K_axle = K_spring_roll + arb_rate
            The track**2/2 factor is the standard ride-rate → roll-rate map for
            two wheels at half-track each; *1000 converts N/mm·m → N/m·m? — no:
            we keep wheel rate in N/mm and track in metres and fold the unit
            factor explicitly below so the arithmetic is auditable.
          * otherwise → fall back to the directly-specified roll_stiffness_*.

        When MR can't be solved (no rocker, degenerate linkage) we fall back to
        the direct roll_stiffness_* value and the model stays usable.
        """
        cached = self._roll_stiffness_cache.get(axle)
        if cached is not None:
            return cached
        result = self._compute_axle_roll_stiffness(axle)
        self._roll_stiffness_cache[axle] = result
        return result

    def _compute_axle_roll_stiffness(self, axle: str) -> float:
        p = self.p
        if axle == "front":
            kin = self.front_kin
            k_spring = p.spring_rate_front
            arb = p.arb_rate_front
            track_mm = p.track_front
            direct = p.roll_stiffness_front
        else:
            kin = self.rear_kin
            k_spring = p.spring_rate_rear
            arb = p.arb_rate_rear
            track_mm = p.track_rear
            direct = p.roll_stiffness_rear

        if not getattr(p, "use_spring_rates", False) or kin is None:
            return direct

        mr = kin.motion_ratio()
        if not np.isfinite(mr) or mr <= 0:
            return direct

        k_wheel = k_spring * mr * mr            # N/mm at the contact patch (vertical)
        # Ride rate (N/mm) → roll stiffness (N·m/deg):
        #   a body roll of 1 deg lifts/drops each wheel by (track/2)*tan(1deg) mm,
        #   producing a wheel-force couple. Linearised:
        #   K_roll [N·m/rad] = k_wheel[N/mm] * 1000[mm/m] ... handled below.
        track_m = track_mm / 1000.0
        # force per wheel per rad of roll = k_wheel[N/mm] * (track/2 in mm) per rad
        # moment per rad = 2 * (force/2 arm) → k_wheel * (track_mm**2 / 2) per rad, in N·mm/rad
        k_roll_Nmm_per_rad = k_wheel * (track_mm ** 2) / 2.0
        k_roll_Nm_per_deg = k_roll_Nmm_per_rad / 1000.0 * (np.pi / 180.0)
        return float(k_roll_Nm_per_deg + arb)

    # ---------------- roll centres from kinematics ----------------------- #
    def roll_center_height(self, kin: SuspensionKinematics, track: float,
                           state=None) -> float:
        """
        Front-view roll-centre height: intersection of the line from contact patch
        through the instant centre with the car centreline (y = 0).

        By default uses the static pose, but pass `state` (any CornerState) to
        evaluate at a given travel — roll centres migrate substantially through
        travel/roll, which is a primary reason teams care about RC, so the model
        can now report that migration instead of assuming RC is fixed.
        """
        st = state if state is not None else kin.static
        ic = st.instant_center          # (y, z)
        cp_y = st.contact_patch[1]
        cp_z = st.contact_patch[2]
        if not np.all(np.isfinite(ic)):
            return np.nan
        dy = ic[0] - cp_y
        dz = ic[1] - cp_z
        if abs(dy) < 1e-9:
            return cp_z
        slope = dz / dy
        rc_z = cp_z + slope * (0.0 - cp_y)
        return float(rc_z)

    def roll_center_migration(self, kin: SuspensionKinematics, track: float,
                              travel_min=-30.0, travel_max=30.0, n=21):
        """RC height across a travel sweep → (travels, rc_heights), for an honest
        picture of how the roll centre moves rather than a single static number."""
        travels = np.linspace(travel_min, travel_max, n)
        rc = [self.roll_center_height(kin, track, state=kin.solve_at_travel(t))
              for t in travels]
        return list(travels), rc

    # ---------------- longitudinal load transfer & anti-features --------- #
    def longitudinal_load_transfer(self, long_g: float):
        """
        Fore/aft load transfer (N) at a given longitudinal g (>0 braking,
        <0 accel), point-mass:  dW = m * a * h / wheelbase, split evenly across
        the two wheels of each axle. Returns (per-axle dW, info).
        """
        p = self.p
        dW = p.mass * (long_g * p.g) * (p.cg_height / 1000.0) / (p.wheelbase / 1000.0)
        return dW, dict(dW_axle=dW)

    def anti_dive_pct(self, brake_bias_front: float = 0.65) -> float:
        """Front anti-dive (%) from the attached FRONT corner geometry. Needs
        front_kin; returns NaN otherwise."""
        if self.front_kin is None:
            return np.nan
        return self.front_kin.anti_dive_pct(self.p.cg_height, self.p.wheelbase,
                                            brake_bias_front=brake_bias_front)

    def anti_squat_pct(self, drive_bias_rear: float = 1.0) -> float:
        """Rear anti-squat (%) from the attached REAR corner geometry. Needs
        rear_kin; returns NaN otherwise."""
        if self.rear_kin is None:
            return np.nan
        return self.rear_kin.anti_squat_pct(self.p.cg_height, self.p.wheelbase,
                                            drive_bias_rear=drive_bias_rear)

    def motion_ratios(self):
        """(MR_front, MR_rear) from the attached geometry; NaN where unavailable."""
        mf = self.front_kin.motion_ratio() if self.front_kin else np.nan
        mr = self.rear_kin.motion_ratio() if self.rear_kin else np.nan
        return mf, mr

    # ---------------- steady-state lateral load transfer ----------------- #
    def lateral_load_transfer(self, lateral_g: float):
        p = self.p
        W = p.mass * p.g
        Wf = W * p.weight_dist_front
        Wr = W * (1 - p.weight_dist_front)

        rc_f = (self.roll_center_height(self.front_kin, p.track_front)
                if self.front_kin else 50.0)
        rc_r = (self.roll_center_height(self.rear_kin, p.track_rear)
                if self.rear_kin else 60.0)
        if not np.isfinite(rc_f):
            rc_f = 50.0
        if not np.isfinite(rc_r):
            rc_r = 60.0

        a_lat = lateral_g * p.g
        # Sprung CG roll moment about the roll axis
        roll_axis_at_cg = rc_f + (rc_r - rc_f) * p.weight_dist_front
        h_roll = p.cg_height - roll_axis_at_cg          # roll moment arm, mm
        M_roll = p.mass * a_lat * (h_roll / 1000.0)     # N·m

        # Axle roll stiffness: derived from spring rate × motion ratio (+ARB) when
        # use_spring_rates is set and geometry is attached, else the direct value.
        ks_f = self._axle_roll_stiffness("front")
        ks_r = self._axle_roll_stiffness("rear")
        k_tot = ks_f + ks_r
        # roll stiffness is specified in N·m/deg, so M/k is already in degrees
        roll_angle = (M_roll / k_tot) if k_tot > 0 else 0.0

        # Elastic (sprung) transfer split by roll stiffness
        dWf_elastic = (ks_f / k_tot) * M_roll / (p.track_front / 1000.0) if k_tot > 0 else 0.0
        dWr_elastic = (ks_r / k_tot) * M_roll / (p.track_rear / 1000.0) if k_tot > 0 else 0.0

        # Geometric (unsprung/RC) transfer reacts instantly through the linkage
        dWf_geo = (p.mass * p.weight_dist_front) * a_lat * (rc_f / 1000.0) / (p.track_front / 1000.0)
        dWr_geo = (p.mass * (1 - p.weight_dist_front)) * a_lat * (rc_r / 1000.0) / (p.track_rear / 1000.0)

        dWf = dWf_elastic + dWf_geo
        dWr = dWr_elastic + dWr_geo

        # In a left turn the right (outer) wheels gain load
        fl = Wf / 2 - dWf
        fr = Wf / 2 + dWf
        rl = Wr / 2 - dWr
        rr = Wr / 2 + dWr
        loads = CornerLoads(max(fl, 0), max(fr, 0), max(rl, 0), max(rr, 0))
        return loads, dict(roll_angle=roll_angle, rc_front=rc_f, rc_rear=rc_r,
                           ltd_front=dWf, ltd_rear=dWr,
                           roll_stiffness_front=ks_f, roll_stiffness_rear=ks_r)

    # ---------------- load-sensitive grip & balance ---------------------- #
    def _corner_force(self, Fz: float, axle: str) -> float:
        """
        Max lateral force a single tire makes at vertical load Fz on the given axle.
        Uses the Pacejka model (load-sensitive + camber-aware) when a tire is
        attached; otherwise the legacy linear placeholder. This single switch is
        what upgrades the whole grip/balance/max-g stack from a straight line to a
        real measured-tire curve.
        """
        if Fz <= 0:
            return 0.0
        if self.tire is not None:
            gamma = self._axle_camber_rad(axle)
            return self.tire.peak_force(Fz, gamma)
        mu = self.p.mu_peak - self.p.tire_load_sens * Fz
        mu = max(mu, 0.3)
        return mu * Fz

    def axle_grip(self, loads: CornerLoads):
        """Max lateral force each axle can make (sum of its two tires)."""
        front = self._corner_force(loads.fl, "front") + self._corner_force(loads.fr, "front")
        rear = self._corner_force(loads.rl, "rear") + self._corner_force(loads.rr, "rear")
        return front, rear

    def grip_model_name(self) -> str:
        """Which grip model is live — for honest UI labelling."""
        return "Pacejka MF5.2" if self.tire is not None else "linear placeholder"

    def balance_index(self, lateral_g: float):
        """
        > 0 means front-limited (understeer), < 0 means rear-limited (oversteer).
        Computed as the normalised difference in axle grip utilisation.
        """
        loads, _ = self.lateral_load_transfer(lateral_g)
        Ff, Fr = self.axle_grip(loads)
        p = self.p
        Wf = p.mass * p.g * p.weight_dist_front
        Wr = p.mass * p.g * (1 - p.weight_dist_front)
        demand_f = Wf * lateral_g
        demand_r = Wr * lateral_g
        util_f = demand_f / Ff if Ff > 0 else 99
        util_r = demand_r / Fr if Fr > 0 else 99
        return util_f - util_r, util_f, util_r

    def corner_compliance(self, lateral_g: float = 1.5,
                          axle: str = "front", outer: bool = True,
                          stiffness: dict | None = None,
                          corner=None, long_g: float = 0.0,
                          **solve_kw):
        """
        Drive the flexible-body (compliance) solve for one corner at a steady
        lateral g — by default the loaded FRONT-OUTER wheel at 1.5 g, the case
        that sets compliance steer.

        The vertical load comes straight from this object's real load-transfer
        model (lateral_load_transfer); the contact-patch lateral force is the
        equal-utilisation Fy = lateral_g·Fz. That patch wrench is fed to a
        CompliantCorner, which resolves member axial loads, stretches each link
        by its series stiffness (tube + optional chassis-tab + optional imported
        FEA body), and re-solves the kinematics to read off compliance toe/camber.

        Parameters
        ----------
        lateral_g : sustained lateral acceleration (g). 1.5 g is the headline case.
        axle      : "front" or "rear" — which corner's kinematics to flex.
        outer     : True for the loaded outer wheel (worst compliance case).
        stiffness : per-member MemberStiffness dict. If omitted, a uniform-tube
                    corner is built from default 4130 tubes (honest baseline:
                    bare-tube axial compliance only, no tab/rod-end softness).
        corner    : a pre-built CompliantCorner (e.g. carrying imported FEA flex
                    bodies). Takes precedence over `stiffness` when supplied.
        long_g    : optional simultaneous longitudinal g.
        **solve_kw: forwarded to CompliantCorner.solve (max_iter, tol_deg).

        Returns a CompliantResult (None if the requested axle has no kinematics
        attached). The result carries rigid-vs-compliant CornerStates, per-member
        forces and deflections, and the compliance toe/camber/caster deltas.
        """
        from . import compliance as _cmp

        kin = self.front_kin if axle == "front" else self.rear_kin
        if kin is None:
            return None

        if corner is None:
            if stiffness is None:
                corner = _cmp.CompliantCorner.uniform_tube(kin.hp)
            else:
                corner = _cmp.CompliantCorner(kin.hp, stiffness)

        load = _cmp.corner_wheel_load(self, axle, lateral_g,
                                      outer=outer, long_g=long_g)
        return corner.solve(load, **solve_kw)

    def max_lateral_g(self):
        """Bisection for the steady-state lateral g the car can sustain."""
        lo, hi = 0.1, 3.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            loads, _ = self.lateral_load_transfer(mid)
            Ff, Fr = self.axle_grip(loads)
            capacity = (Ff + Fr) / (self.p.mass * self.p.g)
            if capacity >= mid:
                lo = mid
            else:
                hi = mid
        return lo
