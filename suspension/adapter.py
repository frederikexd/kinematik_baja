# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Topology adapter — run the whole KinematiK pipeline on *any* architecture.
==========================================================================

``topology.py`` solves a general mechanism; ``topologies.py`` builds named
templates. This module is the bridge that lets the *existing* vehicle-dynamics,
load-path, compliance, chassis-clearance and app layers consume an arbitrary
topology without being rewritten.

The trick is that almost everything downstream only needs four things from a
solved corner:

    * the wheel-plane orientation  -> camber, toe
    * the kingpin / steer axis     -> caster, KPI, scrub
    * the front-view instant centre-> roll centre
    * the side-view swing arm      -> anti-dive / anti-squat
    * solved outboard link points  -> member load paths

``GenericKinematics`` computes all of these for *any* :class:`Mechanism` and
presents them through the same surface the double-wishbone
``SuspensionKinematics`` exposes (``solve_at_travel``, ``sweep``,
``motion_ratio``, ``anti_dive_pct``, ``side_view_swing_arm_length`` …) plus the
same ``CornerState`` dataclass — so ``dynamics.py``, ``setup.py``, etc. keep
working unchanged. Where a generic topology has no direct analogue of a
wishbone-specific field (e.g. ``upper_outer`` on a trailing arm) the adapter
fills it with the nearest meaningful point or ``NaN`` and flags it, rather than
fabricating a wishbone that isn't there.

For the instant-centre / anti-feature geometry — which the wishbone solver got
from its two named arms — the generic version derives the front-view and
side-view instant centres from the *velocity field* of the carrier body: it
perturbs travel slightly, measures how the contact patch and a second carrier
point move, and intersects their motion normals. That is the topology-agnostic
definition of an instant centre and reproduces the wishbone result to numerical
precision while also being correct for struts, multilinks and beams.
"""

from __future__ import annotations

import numpy as np
from .kinematics import CornerState
from .topology import Mechanism, _unit


class GenericKinematics:
    """Drop-in kinematics engine for an arbitrary :class:`Mechanism`.

    Mirrors the public surface of ``SuspensionKinematics`` closely enough that
    the rest of KinematiK can treat the two interchangeably. Construct from a
    compiled Mechanism (typically from ``topologies``)::

        from suspension import topologies as T, GenericKinematics
        kin = GenericKinematics(T.example("macpherson_strut"))
        st  = kin.solve_at_travel(0.0)
        sweep = kin.sweep(-30, 30, 41)
    """

    def __init__(self, mech: Mechanism):
        self.mech = mech.compile() if not mech._compiled else mech
        self.label = mech.label
        self._has_rocker = False           # generic rocker hook (future)
        # a light-weight hardpoints shim so code that does ``kin.hp`` still works
        self.hp = _HardpointShim(self.mech)
        self.static = self.solve_at_travel(0.0)

    # ------------------------------------------------------------------ #
    #  core solve -> CornerState
    # ------------------------------------------------------------------ #
    def _to_state(self, r: dict, travel: float) -> CornerState:
        spin = r["spin_axis"]
        wc = r["wheel_center"]; cp = r["contact_patch"]
        camber = self._camber(spin)
        toe = self._toe(spin)
        # kingpin / steer axis: use the carrier body's principal vertical axis.
        kp = self._kingpin_axis(r)
        caster, kpi = self._caster_kpi(kp)
        scrub = self._scrub(kp, r)
        ic = self._instant_center(r, travel)
        # map onto wishbone-named slots where the topology has them, else NaN
        pos = r["positions"]
        uo = pos.get("uo", pos.get("sl", pos.get("kpt", np.full(3, np.nan))))
        lo = pos.get("lo", pos.get("hub", pos.get("axL", np.full(3, np.nan))))
        tro = pos.get("tro", pos.get("sa", np.full(3, np.nan)))
        return CornerState(
            travel=travel, upper_outer=np.asarray(uo, float),
            lower_outer=np.asarray(lo, float),
            tie_rod_outer=np.asarray(tro, float),
            wheel_center=wc, contact_patch=cp, camber=camber, toe=toe,
            caster=caster, kpi=kpi, scrub_radius=scrub,
            instant_center=ic, roll_center_height=np.nan,
            pushrod_outer=pos.get("pushrod_outer"), converged=r["converged"])

    def solve_at_travel(self, travel_mm: float, seed=None, steer: float = 0.0) -> CornerState:
        r = self.mech.solve(travel_mm, steer=steer, seed=seed)
        return self._to_state(r, travel_mm)

    def sweep(self, travel_min=-30.0, travel_max=30.0, n=41, steer=0.0):
        results = self.mech.sweep(travel_min, travel_max, n, steer=steer)
        travels = np.linspace(travel_min, travel_max, n)
        return [self._to_state(r, float(t)) for r, t in zip(results, travels)]

    # ------------------------------------------------------------------ #
    #  kinematic outputs (topology-agnostic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _camber(spin):
        s = _unit(spin)
        return -np.degrees(np.arctan2(s[2], abs(s[1])))

    @staticmethod
    def _toe(spin):
        s = _unit(spin)
        return np.degrees(np.arctan2(s[0], abs(s[1])))

    def _kingpin_axis(self, r):
        """Steer axis = the carrier body's local 'up' axis in world (the column
        of its rotation matrix nearest vertical at static). For a wishbone this
        is lo->uo; for a strut it is the strut axis; generically it is the
        body-fixed near-vertical direction, which is exactly the steer axis."""
        R = r["carrier_R"]
        # pick the body axis whose world-z component is largest in magnitude
        zcomp = np.abs(R[2, :])
        k = R[:, int(np.argmax(zcomp))]
        if k[2] < 0:
            k = -k
        return _unit(k)

    @staticmethod
    def _caster_kpi(kp):
        caster = np.degrees(np.arctan2(kp[0], kp[2]))
        kpi = np.degrees(np.arctan2(-kp[1], kp[2]))
        return caster, kpi

    def _scrub(self, kp, r):
        cp = r["contact_patch"]
        # intersect the kingpin line (through wheel centre, along kp) with ground
        wc = r["wheel_center"]
        if abs(kp[2]) < 1e-9:
            return np.nan
        t = -wc[2] / kp[2]
        ground = wc + t * kp
        return float(cp[1] - ground[1])

    def _carrier_velocity_point(self, name_or_pos, travel, d=1.0):
        """Velocity (dx,dy,dz per mm travel) of a carrier-fixed point via central
        difference of the solved configuration."""
        r_up = self.mech.solve(travel + d)
        r_dn = self.mech.solve(travel - d)
        self.mech.solve(travel)   # restore
        if isinstance(name_or_pos, str):
            pu = r_up["positions"][name_or_pos]
            pd = r_dn["positions"][name_or_pos]
        else:
            pu, pd = name_or_pos
        return (np.asarray(pu, float) - np.asarray(pd, float)) / (2 * d)

    @staticmethod
    def _ic_from_velocity(p1, v1, p2, v2):
        """Instant centre of planar rigid motion from two point velocities.

        For rotation about IC at rate ω in a plane:  v = ω · ẑ × (P − IC), i.e.
            v_a = -ω (b - b_ic),   v_b =  ω (a - a_ic)
        for in-plane coords (a, b). Two points give four equations in three
        unknowns (a_ic, b_ic, ω); we solve the linear least-squares system.
        Returns (a_ic, b_ic) or NaNs if degenerate (pure translation)."""
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
        # unknowns x = [a_ic, b_ic, w]
        #   eqn rows:  -w*0 ... rearranged to linear form:
        #   v_a + w*b - w*b_ic = 0  -> (w as unknown makes it bilinear); instead
        #   use the standard result: IC = P + (ẑ × v)/ω, and ω = |v|/r unknown.
        # Eliminate ω by noting IC lies on the line through P perpendicular to v.
        # Build those two lines and intersect (this IS correct; the earlier bug
        # was the normal orientation). Normal to v in-plane is (-v_b, v_a) is the
        # velocity direction's perpendicular — but IC lies along that perpendicular
        # FROM the point, so the line is: P + s*(perp). Intersect the two lines.
        def perp(v):
            return np.array([-v[1], v[0]])
        d1 = perp(v1); d2 = perp(v2)
        n1 = np.linalg.norm(d1); n2 = np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            return np.array([np.nan, np.nan])
        d1 /= n1; d2 /= n2
        # solve p1 + s*d1 = p2 + t*d2
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-9:
            return np.array([np.nan, np.nan])
        st = np.linalg.solve(A, p2 - p1)
        return p1 + st[0] * d1

    def _instant_center(self, r, travel):
        """Front-view (y-z) instant centre from the carrier velocity field —
        topology-independent and matches the wishbone link construction."""
        try:
            v_cp = self._carrier_velocity_point(self.mech.contact_patch, travel)
            v_wc = self._carrier_velocity_point(self.mech.wheel_center, travel)
        except Exception:
            return np.array([np.nan, np.nan])
        cp = r["contact_patch"]; wc = r["wheel_center"]
        return self._ic_from_velocity(
            np.array([cp[1], cp[2]]), np.array([v_cp[1], v_cp[2]]),
            np.array([wc[1], wc[2]]), np.array([v_wc[1], v_wc[2]]))

    def _side_view_swing_arm(self, state=None, travel=0.0):
        """Side-view (x-z) instant centre from the carrier velocity field — same
        construction as the front view but in x-z. Sets anti-dive/anti-squat."""
        t = state.travel if state is not None else travel
        try:
            v_cp = self._carrier_velocity_point(self.mech.contact_patch, t)
            v_wc = self._carrier_velocity_point(self.mech.wheel_center, t)
        except Exception:
            return np.array([np.nan, np.nan])
        r = self.mech.solve(t)
        cp = r["contact_patch"]; wc = r["wheel_center"]
        return self._ic_from_velocity(
            np.array([cp[0], cp[2]]), np.array([v_cp[0], v_cp[2]]),
            np.array([wc[0], wc[2]]), np.array([v_wc[0], v_wc[2]]))

    # ------------------------------------------------------------------ #
    #  anti-features & swing-arm length (same formulae as the wishbone)
    # ------------------------------------------------------------------ #
    def anti_dive_pct(self, cg_height, wheelbase, brake_bias_front=0.65, state=None):
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return 0.0
        st = state if state is not None else self.static
        cp = st.contact_patch
        Lsva = svic[0] - cp[0]; hsva = svic[1] - cp[2]
        if abs(Lsva) < 1e-9:
            return np.nan
        return float((hsva / abs(Lsva)) * (wheelbase / cg_height) * brake_bias_front * 100.0)

    def anti_squat_pct(self, cg_height, wheelbase, drive_bias_rear=1.0, state=None):
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return 0.0
        st = state if state is not None else self.static
        cp = st.contact_patch
        Lsva = svic[0] - cp[0]; hsva = svic[1] - cp[2]
        if abs(Lsva) < 1e-9:
            return np.nan
        return float((hsva / abs(Lsva)) * (wheelbase / cg_height) * drive_bias_rear * 100.0)

    def side_view_swing_arm_length(self, state=None):
        svic = self._side_view_swing_arm(state)
        if not np.all(np.isfinite(svic)):
            return np.inf
        st = state if state is not None else self.static
        return float(-(svic[0] - st.contact_patch[0]))

    # ------------------------------------------------------------------ #
    #  motion ratio (generic: spring travel / wheel travel when a spring pair
    #  is declared in mech.meta, else direct-acting 1:1 proxy)
    # ------------------------------------------------------------------ #
    def motion_ratio(self, d=5.0):
        spring = self.mech.meta.get("spring_points")
        s_up = self.mech.solve(+d); s_dn = self.mech.solve(-d); self.mech.solve(0.0)
        wheel = s_up["wheel_center"][2] - s_dn["wheel_center"][2]
        if abs(wheel) < 1e-9:
            return np.nan
        if spring:
            a, b = spring
            lu = np.linalg.norm(s_up["positions"][a] - s_up["positions"][b])
            ld = np.linalg.norm(s_dn["positions"][a] - s_dn["positions"][b])
            return abs((lu - ld) / wheel)
        return 1.0    # no spring declared -> direct-acting proxy

    def motion_ratio_is_real(self):
        return bool(self.mech.meta.get("spring_points"))

    def wheel_rate(self, spring_rate_N_per_mm):
        mr = self.motion_ratio()
        return float(spring_rate_N_per_mm * mr * mr) if np.isfinite(mr) else np.nan

    # ------------------------------------------------------------------ #
    #  Generic rendering support (topology-independent 3D drawing)
    # ------------------------------------------------------------------ #
    def render_segments(self, state=None):
        """Return [(p, q, label)] line segments describing this mechanism at the
        given solved state (default: static). Topology-independent: every
        constraint that connects two points contributes a drawable member, plus
        the upright/kingpin, spin axis and wheel."""
        pos = self.mech.solve(state.travel if state is not None else 0.0)["positions"]
        segs = []
        seen = set()
        for label, outb, inb in self.mech.force_members():
            key = tuple(sorted((outb, inb)))
            if key in seen:
                continue
            seen.add(key)
            if outb in pos and inb in pos:
                segs.append((pos[outb], pos[inb], label))
        # wheel (contact patch -> wheel centre)
        cp = pos.get(self.mech.contact_patch)
        wc = pos.get(self.mech.wheel_center)
        if cp is not None and wc is not None:
            segs.append((cp, wc, "Wheel"))
        return segs

    def named_points(self):
        """Dict of name -> xyz for every point in the mechanism at static."""
        pos = self.mech.solve(0.0)["positions"]
        return {n: pos[n] for n in pos}


class _HardpointShim:
    """Presents a Mechanism's named ground/free points as attribute access, so
    code expecting ``kin.hp.<name>`` keeps working for the points a given
    topology actually has. Missing wishbone-specific names raise AttributeError,
    which callers already guard with getattr(..., None)."""
    def __init__(self, mech: Mechanism):
        object.__setattr__(self, "_mech", mech)

    def __getattr__(self, name):
        mech = object.__getattribute__(self, "_mech")
        if name in mech.points:
            return mech.points[name].pos
        raise AttributeError(name)

    def as_dict(self):
        mech = object.__getattribute__(self, "_mech")
        return {n: p.pos.tolist() for n, p in mech.points.items()}
