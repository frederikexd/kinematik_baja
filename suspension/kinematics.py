# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Double-wishbone kinematics solver for Formula SAE suspension.

This is the engineering core of the tool. Given the 3D hardpoint locations of an
unequal-length double-wishbone corner, it solves the upright position as a function
of vertical wheel travel by enforcing the rigid-link constraints:

    - upper wishbone: upper outer ball joint stays at fixed length from BOTH
      upper-front and upper-rear chassis pickups (a circle/arc constraint)
    - lower wishbone: same for the lower ball joint
    - upright: the distance between upper and lower ball joints is rigid

We parameterise travel by the lower ball joint's vertical position and solve the
resulting nonlinear constraint system with a Levenberg-Marquardt least-squares step
(scipy.optimize.least_squares, method="lm") at each position. From the solved upright
pose we extract the kinematic outputs FSAE teams actually tune around:

    camber gain, toe (bump steer), caster, kingpin inclination (KPI), scrub radius,
    and the front-view instant-centre location.

Roll-centre height is derived from the instant centre at the vehicle level (see
dynamics.py), not here. Motion ratio is now computed from the ACTUAL pushrod/rocker
linkage when those hardpoints are supplied (solving the rocker rotation that the
pushrod imposes and differentiating the spring length against wheel travel); it
falls back to a clearly-labelled direct-acting ball-joint proxy only when no rocker
is defined. Anti-dive and anti-squat are computed from the side-view geometry of the
wishbones (the side-view swing-arm / n-line construction).

All coordinates are SAE-style vehicle axes, in millimetres:
    x : rearward positive (vehicle longitudinal)
    y : right positive (lateral, toward the driver's right)
    z : upward positive (vertical)

A single "corner" is modelled. Left/right symmetry is handled by mirroring y.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
#  Hardpoint container
# --------------------------------------------------------------------------- #
@dataclass
class Hardpoints:
    """3D pickup-point coordinates for one corner, mm, SAE axes (x rear, y right, z up)."""

    # Upper wishbone chassis pickups
    upper_front_inner: np.ndarray
    upper_rear_inner: np.ndarray
    # Lower wishbone chassis pickups
    lower_front_inner: np.ndarray
    lower_rear_inner: np.ndarray
    # Outboard ball joints (on the upright) at static ride height
    upper_outer: np.ndarray
    lower_outer: np.ndarray
    # Steering / tie rod
    tie_rod_inner: np.ndarray
    tie_rod_outer: np.ndarray
    # Wheel
    wheel_center: np.ndarray
    contact_patch: np.ndarray
    # ----------------------------------------------------------------------- #
    #  Pushrod / rocker (bell-crank) actuation. OPTIONAL — when present these
    #  give the REAL motion ratio (wheel travel / spring travel) instead of the
    #  ball-joint-axis proxy. A pushrod runs from `pushrod_outer` (a point on a
    #  control arm or the upright) up to `rocker_pushrod` on a rigid rocker that
    #  pivots about the axis through `rocker_pivot` with direction
    #  `rocker_axis`. The spring/damper acts between `rocker_spring` (on the
    #  rocker) and `spring_inner` (on the chassis). All four rocker/spring points
    #  plus the pushrod outer are needed to define the linkage; if any is None we
    #  fall back to the direct-acting proxy and SAY SO.
    #
    #  `pushrod_attach` selects which moving body the pushrod outer rides on:
    #  "lower" (lower wishbone / lower ball joint, the common FSAE layout),
    #  "upper" (upper wishbone), or "upright". This matters because the pushrod
    #  outer must move rigidly with the body it is bolted to as the wheel travels.
    pushrod_outer: np.ndarray | None = None
    rocker_pivot: np.ndarray | None = None
    rocker_axis: np.ndarray | None = None       # rotation axis direction (need not be unit)
    rocker_pushrod: np.ndarray | None = None    # pushrod attachment on the rocker
    rocker_spring: np.ndarray | None = None     # spring/damper attachment on the rocker
    spring_inner: np.ndarray | None = None      # spring/damper chassis mount
    pushrod_attach: str = "lower"               # "lower" | "upper" | "upright"
    # Design intent at static ride height (deg). These set the wheel spin-axis
    # orientation, which the linkage then carries rigidly through travel & steer.
    static_camber: float = -1.5
    static_toe: float = 0.0

    def has_rocker(self) -> bool:
        """True iff a complete pushrod/rocker linkage is defined."""
        return all(getattr(self, k) is not None for k in (
            "pushrod_outer", "rocker_pivot", "rocker_axis",
            "rocker_pushrod", "rocker_spring", "spring_inner"))

    @staticmethod
    def default() -> "Hardpoints":
        """A sane FSAE front-corner geometry (right side). Roughly a 1.55 m track car.

        Includes a representative pushrod-actuated rocker (static motion ratio
        ~0.52, spring/wheel) and a deliberate front-pickup stagger giving a mild
        ~26% anti-dive, so the side-view tools return sensible numbers out of the
        box rather than a degenerate (flat-pickup) zero.
        """
        return Hardpoints(
            upper_front_inner=np.array([-100.0, 240.0, 280.8]),
            upper_rear_inner=np.array([130.0, 240.0, 299.2]),
            lower_front_inner=np.array([-110.0, 200.0, 122.5]),
            lower_rear_inner=np.array([140.0, 200.0, 117.5]),
            upper_outer=np.array([12.0, 540.0, 300.0]),
            lower_outer=np.array([-5.0, 575.0, 110.0]),
            tie_rod_inner=np.array([100.0, 230.0, 160.0]),
            tie_rod_outer=np.array([90.0, 560.0, 150.0]),
            wheel_center=np.array([0.0, 600.0, 228.0]),
            contact_patch=np.array([0.0, 605.0, 0.0]),
            # Pushrod picks up on the lower wishbone and runs up to a bell-crank
            # pivoting on a fore-aft axis high on the chassis; the spring lies
            # across the rocker to an inboard chassis mount, set roughly
            # perpendicular to the spring arm at static for a near-linear ratio.
            # Static motion ratio ~0.52 (spring/wheel), mild falling rate into
            # bump — representative of a well-set-up FSAE pushrod corner.
            pushrod_outer=np.array([-5.0, 408.0, 120.0]),
            rocker_pivot=np.array([-5.0, 360.0, 330.0]),
            rocker_axis=np.array([1.0, 0.0, 0.0]),       # fore-aft pivot axis
            rocker_pushrod=np.array([-5.0, 402.0, 308.0]),
            rocker_spring=np.array([-5.0, 335.0, 360.0]),
            spring_inner=np.array([-5.0, 196.7, 244.8]),
            pushrod_attach="lower",
        )

    def as_dict(self):
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    @staticmethod
    def from_dict(d) -> "Hardpoints":
        # Required outboard/inboard linkage points, always 3-vectors.
        vec_keys = {"upper_front_inner", "upper_rear_inner", "lower_front_inner",
                    "lower_rear_inner", "upper_outer", "lower_outer",
                    "tie_rod_inner", "tie_rod_outer", "wheel_center", "contact_patch"}
        # Optional rocker/pushrod points: 3-vectors when present, else None.
        opt_vec_keys = {"pushrod_outer", "rocker_pivot", "rocker_axis",
                        "rocker_pushrod", "rocker_spring", "spring_inner"}
        # Only accept keys that are actually fields of Hardpoints. A saved project
        # from a different version (or a hand-edited file) may carry extra or renamed
        # keys; silently ignoring unknowns here is what stops a cryptic
        # "Hardpoints.__init__() got an unexpected keyword argument ..." crash and
        # lets an old/new project still load with whatever fields it does share.
        valid = set(Hardpoints.__dataclass_fields__.keys())
        kwargs = {}
        for k, v in (d or {}).items():
            if k not in valid:
                continue
            if k in vec_keys:
                kwargs[k] = np.array(v, float)
            elif k in opt_vec_keys:
                kwargs[k] = np.array(v, float) if v is not None else None
            else:
                kwargs[k] = v
        return Hardpoints(**kwargs)

    def copy(self) -> "Hardpoints":
        return Hardpoints.from_dict(self.as_dict())


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix rotating unit vector a onto unit vector b (Rodrigues)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# --------------------------------------------------------------------------- #
#  Solved corner state
# --------------------------------------------------------------------------- #
@dataclass
class CornerState:
    travel: float                 # vertical wheel-centre travel from static, mm (+ = bump)
    upper_outer: np.ndarray
    lower_outer: np.ndarray
    tie_rod_outer: np.ndarray
    wheel_center: np.ndarray
    contact_patch: np.ndarray
    camber: float                 # deg, negative = top leans inboard
    toe: float                    # deg, positive = toe-out
    caster: float                 # deg
    kpi: float                    # deg
    scrub_radius: float           # mm
    instant_center: np.ndarray    # front-view IC (y,z) of the linkage, mm
    roll_center_height: float     # mm, computed at vehicle level
    pushrod_outer: np.ndarray | None = None  # solved pushrod outer position (rides with its body)
    converged: bool = True


# --------------------------------------------------------------------------- #
#  The kinematics solver
# --------------------------------------------------------------------------- #
class SuspensionKinematics:
    """Solves a double-wishbone corner over a range of wheel travel."""

    def __init__(self, hp: Hardpoints, length_deltas: dict | None = None,
                 pickup_deltas: dict | None = None):
        """
        hp : the rigid hardpoint geometry.

        length_deltas / pickup_deltas : OPTIONAL compliance inputs. They let a
        flexible-body layer (see compliance.py) inject the small deflections a real
        car develops under load WITHOUT touching the rigid solver's logic — when both
        are None (the default) the behaviour is identical to the rigid tool.

          length_deltas : dict of link-length changes (mm) added to the cached rigid
            lengths the solver enforces. Keys (all optional, default 0):
              'upper_f','upper_r','lower_f','lower_r'  wishbone legs
              'tie'                                    tie rod
              'upright'                                upper↔lower ball-joint spacing
              'tro_lo','tro_uo'                        tie-rod-outer rigidity to the
                                                       ball joints
            A positive delta means the link has stretched in tension, so its ball
            joint sits that much further from the chassis pickup — exactly the
            geometric effect of a compliant link under load.

          pickup_deltas : dict mapping an inner-pickup field name (e.g.
            'lower_front_inner') to a 3-vector shift (mm), for modelling a chassis
            tab that deflects under its mount load. Applied to a COPY of hp, so the
            caller's hardpoints are never mutated.
        """
        if pickup_deltas:
            hp_nominal = hp.copy()      # rest lengths come from the UNSHIFTED geometry
            hp = hp.copy()
            for name, shift in pickup_deltas.items():
                if hasattr(hp, name) and getattr(hp, name) is not None:
                    setattr(hp, name, np.asarray(getattr(hp, name), float)
                            + np.asarray(shift, float))
        else:
            hp_nominal = hp
        self.hp = hp
        # The geometry the *rest lengths* are measured from. With a chassis-tab
        # (pickup) deflection the pickup moves but the link keeps its unloaded
        # length, so lengths are referenced to the nominal (unshifted) pickups while
        # the residuals use the shifted pickups for position — the net effect is the
        # ball joint being pushed by the tab deflection, which is the real physics.
        self._hp_ref = hp_nominal
        self._dL = {k: 0.0 for k in ("upper_f", "upper_r", "lower_f", "lower_r",
                                     "upright", "tie", "tro_lo", "tro_uo")}
        if length_deltas:
            for k, v in length_deltas.items():
                if k in self._dL:
                    self._dL[k] = float(v)
        self._validate(hp)
        self._cache_static()

    @staticmethod
    def _validate(hp: "Hardpoints"):
        """Fail fast with a clear message rather than a cryptic solver error."""
        point_fields = [
            "upper_front_inner", "upper_rear_inner", "lower_front_inner",
            "lower_rear_inner", "upper_outer", "lower_outer",
            "tie_rod_inner", "tie_rod_outer", "wheel_center", "contact_patch",
        ]
        for name in point_fields:
            v = np.asarray(getattr(hp, name), float)
            if v.shape != (3,):
                raise ValueError(
                    f"Hardpoint '{name}' must be a 3D point [x, y, z]; "
                    f"got shape {v.shape}.")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"Hardpoint '{name}' contains non-finite values: {v}.")
        # Degenerate geometry: ball joints must be distinct, wishbone arms non-zero.
        if np.linalg.norm(hp.upper_outer - hp.lower_outer) < 1e-6:
            raise ValueError("Upper and lower ball joints are coincident — "
                             "the upright would have zero length.")
        for inner, outer, label in [
            (hp.upper_front_inner, hp.upper_outer, "upper front"),
            (hp.upper_rear_inner, hp.upper_outer, "upper rear"),
            (hp.lower_front_inner, hp.lower_outer, "lower front"),
            (hp.lower_rear_inner, hp.lower_outer, "lower rear"),
        ]:
            if np.linalg.norm(np.asarray(inner, float) - np.asarray(outer, float)) < 1e-6:
                raise ValueError(f"The {label} wishbone has zero length "
                                 "(inner and outer points coincide).")

    def _cache_static(self):
        hp = self.hp
        ref = getattr(self, "_hp_ref", hp)   # nominal geometry for rest lengths
        dL = getattr(self, "_dL", {})
        # Rigid link lengths captured at static ride height. Any compliance length
        # deltas (default 0) are added here, so the constraint the solver enforces
        # becomes the LOADED length of each link — the only change a flexible link
        # makes to the geometry is that it is a little longer (in tension) or shorter
        # (in compression) than its unloaded length.
        self.L_upper_f = np.linalg.norm(ref.upper_outer - ref.upper_front_inner) + dL.get("upper_f", 0.0)
        self.L_upper_r = np.linalg.norm(ref.upper_outer - ref.upper_rear_inner) + dL.get("upper_r", 0.0)
        self.L_lower_f = np.linalg.norm(ref.lower_outer - ref.lower_front_inner) + dL.get("lower_f", 0.0)
        self.L_lower_r = np.linalg.norm(ref.lower_outer - ref.lower_rear_inner) + dL.get("lower_r", 0.0)
        self.L_upright = np.linalg.norm(ref.upper_outer - ref.lower_outer) + dL.get("upright", 0.0)
        self.L_tie = np.linalg.norm(ref.tie_rod_outer - ref.tie_rod_inner) + dL.get("tie", 0.0)
        # Tie-rod outer is rigid to the upright: fixed distances to both ball joints.
        self.L_tro_lo = np.linalg.norm(ref.tie_rod_outer - ref.lower_outer) + dL.get("tro_lo", 0.0)
        self.L_tro_uo = np.linalg.norm(ref.tie_rod_outer - ref.upper_outer) + dL.get("tro_uo", 0.0)
        # Signed side of tro relative to the kingpin plane, to disambiguate the
        # two mirror solutions for the tie-rod outer position.
        kp0 = _unit(hp.upper_outer - hp.lower_outer)
        n0 = _unit(np.cross(kp0, np.array([1.0, 0.0, 0.0])))
        self._tro_side = np.sign(np.dot(hp.tie_rod_outer - hp.lower_outer, n0)) or 1.0

        # Rigid offsets of the wheel-carrier points relative to the upright frame.
        # The upright frame is defined by the lower ball joint (origin) and the
        # kingpin axis (lower->upper). We store body points in that local frame so
        # they ride rigidly with the upright as it moves.
        self.kp_axis_static = _unit(hp.upper_outer - hp.lower_outer)
        # Define a rigid upright frame from TWO physical points (lower & upper ball
        # joints) plus the tie-rod outer, which fixes rotation about the kingpin.
        self._R0, self._o0 = self._upright_pose(hp.lower_outer, hp.upper_outer, hp.tie_rod_outer)
        # All carrier points expressed in this static frame -> ride rigidly.
        self._wc_local = self._R0.T @ (hp.wheel_center - self._o0)
        self._cp_local = self._R0.T @ (hp.contact_patch - self._o0)
        # Static wheel spin axis (outboard +y) tilted by design camber & toe, then
        # frozen into the upright-local frame so it rides rigidly through travel.
        cam = np.radians(hp.static_camber)
        toe = np.radians(hp.static_toe)
        spin0 = _unit(np.array([np.sin(toe), np.cos(toe) * np.cos(cam), -np.sin(cam)]))
        self._spin_local = self._R0.T @ spin0

        # ------------------------------------------------------------------ #
        #  Pushrod / rocker static caching (only when the linkage is defined).
        #  The pushrod outer is rigid to a MOVING body (lower or upper wishbone,
        #  or the upright). We store it in that body's local frame so it rides
        #  rigidly as the wheel travels, then recover its position each solve.
        # ------------------------------------------------------------------ #
        self._has_rocker = hp.has_rocker()
        if self._has_rocker:
            attach = getattr(hp, "pushrod_attach", "lower")
            if attach == "upright":
                self._pushrod_body = "upright"
                self._pushrod_local = self._R0.T @ (hp.pushrod_outer - self._o0)
            else:
                self._pushrod_body = attach  # "lower" | "upper"
                Rw0, ow0 = self._wishbone_pose_static(attach)
                self._pushrod_wb_R0 = Rw0
                self._pushrod_wb_o0 = ow0
                self._pushrod_local = Rw0.T @ (hp.pushrod_outer - ow0)
            # Rigid link lengths for the pushrod/rocker.
            self.L_pushrod = np.linalg.norm(hp.rocker_pushrod - hp.pushrod_outer)
            self.L_rocker_push = np.linalg.norm(hp.rocker_pushrod - hp.rocker_pivot)
            self.L_rocker_spring = np.linalg.norm(hp.rocker_spring - hp.rocker_pivot)
            self.L_spring_static = np.linalg.norm(hp.rocker_spring - hp.spring_inner)
            self._rocker_axis = _unit(np.asarray(hp.rocker_axis, float))

        self.static = self.solve_at_travel(0.0)

    def _wishbone_pose_static(self, which: str):
        """Static rigid frame for the lower or upper wishbone.

        Origin = front inner pivot. Local z runs along the (fixed) inner pivot
        axis; local x is the component of (outer - front_inner) perpendicular to
        that axis. This frame is reconstructed at each travel from the same two
        fixed inner pivots plus the solved outer ball joint, so any point bolted
        to the wishbone (e.g. the pushrod outer) rides with it rigidly.
        """
        hp = self.hp
        if which == "lower":
            fi, ri, outer = hp.lower_front_inner, hp.lower_rear_inner, hp.lower_outer
        else:
            fi, ri, outer = hp.upper_front_inner, hp.upper_rear_inner, hp.upper_outer
        return self._wishbone_pose(fi, ri, outer)

    @staticmethod
    def _wishbone_pose(front_inner, ri, outer):
        origin = np.asarray(front_inner, float).copy()
        z = _unit(np.asarray(ri, float) - origin)
        ref = np.asarray(outer, float) - origin
        x = _unit(ref - np.dot(ref, z) * z)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        return R, origin

    def _pushrod_outer_at(self, lo, uo, tro):
        """Position of the pushrod outer in the solved configuration."""
        if not self._has_rocker:
            return None
        if self._pushrod_body == "upright":
            R, o = self._upright_pose(lo, uo, tro)
            return o + R @ self._pushrod_local
        # wishbone-mounted: rebuild that wishbone's frame from its fixed inner
        # pivots and the solved outer ball joint.
        hp = self.hp
        if self._pushrod_body == "lower":
            R, o = self._wishbone_pose(hp.lower_front_inner, hp.lower_rear_inner, lo)
        else:
            R, o = self._wishbone_pose(hp.upper_front_inner, hp.upper_rear_inner, uo)
        return o + R @ self._pushrod_local

    def _upright_pose(self, lo, uo, tro):
        """
        Build a rigid orthonormal frame + origin for the upright from three of its
        physical points: lower ball joint (origin), kingpin axis (lo->uo) as local z,
        and the tie-rod outer to fix rotation about the kingpin (local x in-plane).
        Returns (R, origin) such that any local point p maps to origin + R @ p.
        """
        origin = lo.copy()
        z = _unit(uo - lo)
        ref = tro - lo
        x = _unit(ref - np.dot(ref, z) * z)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        return R, origin

    def _tro_local_static(self):
        return self._R0.T @ (self.hp.tie_rod_outer - self._o0)

    # --------------------------------------------------------------------- #
    def _residuals(self, q, target_lower_z):
        """
        Unknowns q = [lo(3), uo(3), tro(3)]
        Constraints (10 eqns, 9 unknowns -> least squares):
          lower outer on both lower-arm spheres                (2)
          upper outer on both upper-arm spheres                (2)
          upright rigid |uo-lo|                                (1)
          tie-rod outer rigid to upright (dist to lo and uo)   (2)
          tie-rod length to inner pickup                       (1)
          lower-outer z drives travel                          (1)
        """
        hp = self.hp
        lo, uo, tro = q[0:3], q[3:6], q[6:9]
        r = [
            np.linalg.norm(lo - hp.lower_front_inner) - self.L_lower_f,
            np.linalg.norm(lo - hp.lower_rear_inner) - self.L_lower_r,
            np.linalg.norm(uo - hp.upper_front_inner) - self.L_upper_f,
            np.linalg.norm(uo - hp.upper_rear_inner) - self.L_upper_r,
            np.linalg.norm(uo - lo) - self.L_upright,
            np.linalg.norm(tro - lo) - self.L_tro_lo,
            np.linalg.norm(tro - uo) - self.L_tro_uo,
            np.linalg.norm(tro - hp.tie_rod_inner) - self.L_tie,
            lo[2] - target_lower_z,
        ]
        return np.array(r)

    def solve_at_travel(self, travel_mm: float, seed=None) -> CornerState:
        """
        Solve the linkage at a given wheel travel. `seed` optionally provides a
        warm-start vector [lo, uo, tro] from a nearby solved position — passing the
        previous step's solution keeps the solver on the correct configuration branch
        and prevents it from jumping to the mirror (flipped-linkage) solution at large
        travel. When seed is None it starts from the static pose.
        """
        hp = self.hp
        target_lower_z = hp.lower_outer[2] + travel_mm
        q0 = seed if seed is not None else np.concatenate(
            [hp.lower_outer, hp.upper_outer, hp.tie_rod_outer])
        sol = least_squares(
            self._residuals, q0, args=(target_lower_z,),
            method="lm", max_nfev=400, xtol=1e-12, ftol=1e-12,
        )
        lo, uo, tro = sol.x[0:3], sol.x[3:6], sol.x[6:9]
        max_resid = float(np.max(np.abs(sol.fun)))
        converged = max_resid < 0.1

        # Rigid pose of the upright in the solved configuration.
        R, o = self._upright_pose(lo, uo, tro)
        wc = o + R @ self._wc_local
        cp = o + R @ self._cp_local
        spin = R @ self._spin_local       # current wheel spin axis (outboard)

        camber = self._camber(spin)
        toe = self._toe(spin)
        caster, kpi = self._caster_kpi(uo, lo)
        scrub = self._scrub_radius(uo, lo, cp)
        ic = self._instant_center(uo, lo)
        pro = self._pushrod_outer_at(lo, uo, tro) if getattr(self, "_has_rocker", False) else None

        return CornerState(
            travel=travel_mm, upper_outer=uo, lower_outer=lo, tie_rod_outer=tro,
            wheel_center=wc, contact_patch=cp, camber=camber, toe=toe,
            caster=caster, kpi=kpi, scrub_radius=scrub,
            instant_center=ic, roll_center_height=np.nan,
            pushrod_outer=pro, converged=converged,
        )

    # ------------------------ kinematic outputs -------------------------- #
    def _camber(self, spin):
        # Camber = lean of the wheel plane. Wheel plane normal = spin axis. The wheel
        # plane's tilt from vertical equals the spin axis tilt from horizontal in the
        # front (y-z) view. Top-inboard => negative camber (racing convention).
        s = _unit(spin)
        ang = np.degrees(np.arctan2(s[2], abs(s[1])))
        return -ang

    def _toe(self, spin):
        # Toe = steer of the wheel plane in the top (x-y) view. Spin axis points
        # outboard (+y); its fore/aft component gives toe. Positive = toe-out.
        s = _unit(spin)
        return np.degrees(np.arctan2(s[0], abs(s[1])))

    def _caster_kpi(self, uo, lo):
        kp = uo - lo
        # Caster: side-view kingpin lean. Positive when the top of the kingpin is
        # rearward of the bottom (x rear-positive in SAE), giving self-centering.
        caster = np.degrees(np.arctan2(kp[0], kp[2]))
        # KPI: front-view kingpin lean. Positive when top leans inboard (toward
        # centreline). On a right corner inboard is -y, so top has smaller y.
        kpi = np.degrees(np.arctan2(-kp[1], kp[2]))
        return caster, kpi

    def _scrub_radius(self, uo, lo, cp):
        # distance in ground plane between kingpin-axis ground intersection and contact patch
        kp = _unit(uo - lo)
        if abs(kp[2]) < 1e-9:
            return np.nan
        t = -lo[2] / kp[2]
        ground = lo + t * kp
        return float(cp[1] - ground[1])

    def _instant_center(self, uo, lo):
        """Front-view instant centre (y,z) from the two wishbone projections."""
        hp = self.hp
        # upper arm line in y-z (use mean of front/rear inner pickups)
        u_in = 0.5 * (hp.upper_front_inner + hp.upper_rear_inner)
        l_in = 0.5 * (hp.lower_front_inner + hp.lower_rear_inner)
        # upper line: through uo and u_in (in y-z)
        p1, d1 = np.array([uo[1], uo[2]]), np.array([u_in[1] - uo[1], u_in[2] - uo[2]])
        p2, d2 = np.array([lo[1], lo[2]]), np.array([l_in[1] - lo[1], l_in[2] - lo[2]])
        # solve p1 + t d1 = p2 + s d2
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-9:
            return np.array([np.nan, np.nan])
        ts = np.linalg.solve(A, p2 - p1)
        ic = p1 + ts[0] * d1
        return ic

    # ------------------------- sweep & metrics --------------------------- #
    def sweep(self, travel_min=-30.0, travel_max=30.0, n=41):
        """
        Solve across a travel range. Marches outward from the static position in both
        directions, warm-starting each step from the previous solved pose so the
        solver stays on the physically correct branch instead of risking a jump to the
        mirror configuration at the extremes. Results are returned in ascending travel.
        """
        travels = np.linspace(travel_min, travel_max, n)
        # split into droop side (descending from 0) and bump side (ascending from 0)
        below = sorted([t for t in travels if t < 0], reverse=True)
        above = sorted([t for t in travels if t > 0])
        zero = [t for t in travels if t == 0]

        results = {}
        # solve static first as the anchor seed
        static = self.solve_at_travel(0.0)
        seed0 = np.concatenate([static.lower_outer, static.upper_outer,
                                static.tie_rod_outer])
        for t in zero:
            results[t] = static

        seed = seed0
        for t in above:
            st = self.solve_at_travel(t, seed=seed)
            seed = np.concatenate([st.lower_outer, st.upper_outer, st.tie_rod_outer])
            results[t] = st

        seed = seed0
        for t in below:
            st = self.solve_at_travel(t, seed=seed)
            seed = np.concatenate([st.lower_outer, st.upper_outer, st.tie_rod_outer])
            results[t] = st

        return [results[t] for t in travels]

    # ===================================================================== #
    #  Pushrod / rocker motion ratio  (REAL geometry when available)
    # ===================================================================== #
    def _rotate_about_axis(self, point, angle):
        """Rotate `point` about the rocker pivot/axis by `angle` (rad), Rodrigues."""
        hp = self.hp
        piv = np.asarray(hp.rocker_pivot, float)
        k = self._rocker_axis
        v = np.asarray(point, float) - piv
        c, s = np.cos(angle), np.sin(angle)
        rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)
        return piv + rot

    def _solve_rocker_angle(self, pushrod_outer, seed=0.0):
        """
        Find the rocker rotation (rad about its axis) such that the rocker's
        pushrod point sits exactly one pushrod-length from the (moved) pushrod
        outer. Returns the signed angle on the branch closest to `seed`.
        """
        hp = self.hp
        rp0 = np.asarray(hp.rocker_pushrod, float)

        def resid(theta):
            rp = self._rotate_about_axis(rp0, theta)
            return np.linalg.norm(rp - np.asarray(pushrod_outer, float)) - self.L_pushrod

        sol = least_squares(lambda t: [resid(t[0])], [seed],
                            method="lm", max_nfev=200, xtol=1e-12, ftol=1e-12)
        return float(sol.x[0]), float(abs(sol.fun[0]))

    def spring_length_at(self, state: "CornerState", seed=0.0):
        """
        Installed spring/damper length (mm) at a solved corner state, by driving
        the real pushrod→rocker→spring chain. Returns (length, rocker_angle, ok).
        Requires a defined rocker; raises otherwise.
        """
        if not getattr(self, "_has_rocker", False):
            raise ValueError("No pushrod/rocker geometry defined on these hardpoints.")
        hp = self.hp
        pro = state.pushrod_outer
        if pro is None:
            pro = self._pushrod_outer_at(state.lower_outer, state.upper_outer,
                                         state.tie_rod_outer)
        theta, err = self._solve_rocker_angle(pro, seed=seed)
        rs = self._rotate_about_axis(np.asarray(hp.rocker_spring, float), theta)
        length = float(np.linalg.norm(rs - np.asarray(hp.spring_inner, float)))
        return length, theta, (err < 0.05)

    def motion_ratio(self, spring_inner=None, rocker_pivot=None,
                     push_outer_local=None, d=5.0):
        """
        Installation / motion ratio of the corner.

        Convention: MR = spring (damper) travel / wheel travel, evaluated about
        static by central difference over ±`d` mm of wheel travel. With this
        convention the WHEEL rate is  k_wheel = k_spring * MR**2  (see
        `wheel_rate`). A typical FSAE pushrod car has MR in roughly 0.4–0.8.

        Behaviour:
          * If a full pushrod/rocker linkage is defined on the hardpoints, the
            REAL linkage is solved (pushrod drives the bell-crank, spring length
            read across the rocker) and the legacy positional arguments are
            ignored. This is the number you should trust.
          * Otherwise it falls back to the legacy DIRECT-ACTING PROXY: the
            spring is assumed to act along (spring_inner - rocker_pivot) at the
            lower ball joint. This is only a rough placeholder and is flagged as
            such by `motion_ratio_is_real()`.
        """
        # Fast path: the no-argument call (what the dynamics/roll-stiffness model
        # uses) depends only on the immutable hardpoints, so solve the linkage once
        # and cache it. The optimiser rebuilds VehicleDynamics on every trial setup
        # but only spring/ARB/camber knobs change between most trials — the geometry
        # (and hence MR) is identical, so without this each rebuild re-ran two
        # nonlinear linkage solves for nothing.
        if (spring_inner is None and rocker_pivot is None
                and push_outer_local is None and d == 5.0):
            cached = getattr(self, "_motion_ratio_cache", None)
            if cached is not None:
                return cached
            mr = self._compute_motion_ratio(spring_inner, rocker_pivot,
                                            push_outer_local, d)
            self._motion_ratio_cache = mr
            return mr
        return self._compute_motion_ratio(spring_inner, rocker_pivot,
                                          push_outer_local, d)

    def _compute_motion_ratio(self, spring_inner=None, rocker_pivot=None,
                              push_outer_local=None, d=5.0):
        s_up = self.solve_at_travel(+d)
        s_dn = self.solve_at_travel(-d)
        wheel_disp = s_up.wheel_center[2] - s_dn.wheel_center[2]
        if abs(wheel_disp) < 1e-9:
            return np.nan

        if getattr(self, "_has_rocker", False):
            # Real rocker: change in installed spring length per wheel travel.
            l_up, th_up, ok_up = self.spring_length_at(s_up, seed=0.0)
            l_dn, th_dn, ok_dn = self.spring_length_at(s_dn, seed=th_up)
            spring_disp = l_up - l_dn
            mr = abs(spring_disp / wheel_disp)
            return mr if (ok_up and ok_dn and np.isfinite(mr)) else np.nan

        # ---- legacy direct-acting proxy (no rocker defined) ----
        si = self.hp.spring_inner if spring_inner is None else spring_inner
        rp = self.hp.rocker_pivot if rocker_pivot is None else rocker_pivot
        if si is None or rp is None:
            # No actuation info at all: assume direct 1:1 along vertical.
            return 1.0
        if push_outer_local is None:
            p_up, p_dn = s_up.lower_outer, s_dn.lower_outer
        else:
            p_up = s_up.lower_outer + push_outer_local
            p_dn = s_dn.lower_outer + push_outer_local
        axis = _unit(np.asarray(si, float) - np.asarray(rp, float))
        spring_disp = np.dot(p_up - p_dn, axis)
        if abs(spring_disp) < 1e-9:
            return np.nan
        return abs(spring_disp / wheel_disp)

    def motion_ratio_is_real(self) -> bool:
        """True when motion_ratio() uses the actual rocker linkage (not the proxy)."""
        return bool(getattr(self, "_has_rocker", False))

    def wheel_rate(self, spring_rate_N_per_mm: float) -> float:
        """
        Vertical wheel rate (N/mm) from the spring rate, through the real motion
        ratio:  k_wheel = k_spring * MR**2, with MR = spring travel / wheel
        travel. This is the conversion that makes a quoted spring rate physically
        meaningful — without the rocker geometry the same spring gives a wildly
        different wheel rate.
        """
        mr = self.motion_ratio()
        if not np.isfinite(mr):
            return np.nan
        return float(spring_rate_N_per_mm * mr * mr)

    def motion_ratio_curve(self, travel_min=-25.0, travel_max=25.0, n=21):
        """
        Motion ratio as a function of wheel travel (rising/falling rate behaviour).
        Returns (travels, mr) where mr[i] = local d(spring)/d(wheel). NaN where the
        rocker can't be solved or no rocker is defined. Differentiated with a
        warm-started central difference so the rocker stays on one branch.
        """
        travels = np.linspace(travel_min, travel_max, n)
        if not getattr(self, "_has_rocker", False):
            return list(travels), [self.motion_ratio()] * n  # proxy: ~constant
        h = 1.0
        out = []
        seed = 0.0
        for t in travels:
            su = self.solve_at_travel(t + h)
            sd = self.solve_at_travel(t - h)
            lu, seed, ok_u = self.spring_length_at(su, seed=seed)
            ld, _, ok_d = self.spring_length_at(sd, seed=seed)
            wd = su.wheel_center[2] - sd.wheel_center[2]
            mr = abs((lu - ld) / wd) if abs(wd) > 1e-9 and ok_u and ok_d else np.nan
            out.append(float(mr) if np.isfinite(mr) else np.nan)
        return list(travels), out

    # ===================================================================== #
    #  Anti-dive / anti-squat  (side-view geometry)
    # ===================================================================== #
    def _side_view_swing_arm(self, state=None):
        """
        Side-view (x–z plane) instant centre of the wishbone linkage — the point
        about which the wheel swings fore/aft, which sets anti-dive/anti-squat.

        Construction (textbook, e.g. Milliken): in side view each wishbone is the
        line through its outboard ball joint with the slope of that wishbone's
        CHASSIS PIVOT AXIS (the line through its front and rear inner pickups)
        projected into x–z. The pivot-axis slope is what the wheel actually
        articulates about longitudinally; using it — rather than the line to the
        averaged inner pickup — is what makes anti-dive depend on the front/rear
        pickup-height stagger, as it physically does. The two pivot-axis lines are
        intersected to give the side-view instant centre (x, z). Returns NaNs when
        the lines are parallel in side view (infinite swing arm => zero anti-dive).
        """
        hp = self.hp
        st = state if state is not None else getattr(self, "static", None)
        uo = st.upper_outer if st is not None else hp.upper_outer
        lo = st.lower_outer if st is not None else hp.lower_outer

        def axis_slope_xz(front_inner, rear_inner):
            d = np.asarray(rear_inner, float) - np.asarray(front_inner, float)
            # side-view direction = (dx, dz); if the pivot axis is purely lateral
            # (dx==0) the wishbone is flat in side view => horizontal line.
            if abs(d[0]) < 1e-9:
                return np.array([1.0, 0.0])
            return np.array([d[0], d[2]])

        d1 = axis_slope_xz(hp.upper_front_inner, hp.upper_rear_inner)
        d2 = axis_slope_xz(hp.lower_front_inner, hp.lower_rear_inner)
        p1 = np.array([uo[0], uo[2]])
        p2 = np.array([lo[0], lo[2]])
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-9:
            return np.array([np.nan, np.nan])
        ts = np.linalg.solve(A, p2 - p1)
        return p1 + ts[0] * d1

    def anti_dive_pct(self, cg_height: float, wheelbase: float,
                      brake_bias_front: float = 0.65, state=None) -> float:
        """
        Front anti-dive, percent. Anti-dive resists nose-dive under braking by
        reacting part of the longitudinal load transfer through the suspension
        links instead of the springs.

        Geometry: with the side-view instant centre (SVIC) at horizontal distance
        `Lsva` ahead of the contact patch and height `hsva` above ground, the
        side-view swing-arm angle is  tan(phi) = hsva / Lsva.  For an OUTBOARD
        (upright-mounted) front brake the standard result (Gillespie / Milliken)
        is:

            anti-dive%  =  tan(phi) * (wheelbase / cg_height) * brake_bias_front * 100
                        =  (hsva / Lsva) * (L / h) * bias_f * 100

        `brake_bias_front` is the fraction of total braking force at the front
        axle. 100% anti-dive fully cancels front-end pitch from braking; FSAE
        cars typically run 0–30% to keep braking feel and tyre-load sensitivity.
        Returns NaN if the SVIC is at infinity (zero anti-dive geometry).
        """
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return 0.0  # parallel wishbones in side view => zero anti-dive
        st = state if state is not None else self.static
        cp = st.contact_patch
        Lsva = svic[0] - cp[0]        # +x is rearward; SVIC ahead of patch => negative
        hsva = svic[1] - cp[2]
        if abs(Lsva) < 1e-9:
            return np.nan
        tan_phi = hsva / abs(Lsva)
        return float(tan_phi * (wheelbase / cg_height) * brake_bias_front * 100.0)

    def anti_squat_pct(self, cg_height: float, wheelbase: float,
                       drive_bias_rear: float = 1.0, state=None) -> float:
        """
        Rear anti-squat, percent. Anti-squat resists rear-end squat under
        acceleration. For an independent rear with INBOARD drive (chassis-mounted
        diff, half-shafts to the hubs) the reaction geometry uses the same
        side-view swing-arm construction as anti-dive:

            anti-squat%  =  tan(phi) * (wheelbase / cg_height) * drive_bias_rear * 100

        with tan(phi) = hsva / Lsva from the rear corner's side-view instant
        centre relative to the rear contact patch. `drive_bias_rear` is the
        fraction of tractive force at this axle (1.0 for a RWD FSAE car). 100%
        anti-squat fully cancels acceleration squat. Returns NaN if undefined.

        NOTE: this is the geometry for an inboard-brake/inboard-drive layout
        (the usual FSAE rear). A solid axle or outboard drive would use a
        different reaction path; that is out of scope here and would mislead, so
        it is not approximated.
        """
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return 0.0
        st = state if state is not None else self.static
        cp = st.contact_patch
        Lsva = svic[0] - cp[0]
        hsva = svic[1] - cp[2]
        if abs(Lsva) < 1e-9:
            return np.nan
        tan_phi = hsva / abs(Lsva)
        return float(tan_phi * (wheelbase / cg_height) * drive_bias_rear * 100.0)

    def side_view_swing_arm_length(self, state=None) -> float:
        """Horizontal distance (mm) from contact patch to the side-view instant
        centre — the side-view swing-arm length. Long arm => little pitch
        coupling; short arm => lots. Sign: positive when the SVIC is ahead of the
        contact patch (forward), the usual anti-dive sense for a front corner."""
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return np.inf
        st = state if state is not None else self.static
        return float(-(svic[0] - st.contact_patch[0]))
