# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Architecture-agnostic suspension topology engine.
====================================================

KinematiK began life as a *double-wishbone* studio: the solver in
``kinematics.py`` hard-codes ten constraint equations that only describe an
unequal-length A-arm corner. That is excellent for FSAE but useless the moment
you point it at a MacPherson strut, a five-link rear, a trailing arm, a beam
axle, a twist-beam, or a heavy-truck steering linkage.

This module replaces the hard-coded constraint set with a small **general
multibody kinematics kernel**. A suspension — *any* suspension — is expressed as

    * a set of named 3-D **points** (some fixed to the chassis, some free,
      some rigidly carried by a moving body),
    * a set of rigid **bodies** (the upright/knuckle, each control arm if you
      want it as a body, a rocker, a beam axle, a steering drag link, …), and
    * a set of parameterised **constraints** between those points/bodies
      (links, struts, revolutes, planar joints, in-line guides, body-rigidity,
      ground/axle-roll couplings, …).

The kernel parameterises the configuration by **one driving coordinate** (wheel
vertical travel, by default) plus an optional **steer/rack coordinate**, and
solves the remaining unknowns with a damped least-squares (Levenberg–Marquardt)
step — exactly the numerical approach the original double-wishbone solver used,
just generalised so the constraint list is *data*, not *code*.

Why this matters: every topology template in ``topologies.py`` is now just a
function that emits a ``Mechanism`` (points + bodies + constraints + which
point is the wheel carrier, where the contact patch and spin axis live). New or
experimental architectures that do not fit any textbook definition are built by
listing whatever constraints actually hold them together — the kernel does not
care whether the result has a name.

Coordinates are the same SAE vehicle axes used everywhere else in KinematiK,
millimetres:  x rear+, y right+, z up+.

The kernel is deliberately self-contained (numpy + scipy only) and knows nothing
about tyres, lap time, or FSAE. ``kinematics.py`` adapts its rich
``CornerState`` output on top of this; downstream physics is untouched.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _vec(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {a.shape}")
    return a


# --------------------------------------------------------------------------- #
#  Points
# --------------------------------------------------------------------------- #
@dataclass
class Point:
    """A named 3-D location in the mechanism.

    role:
      - ``"ground"``  : fixed to the chassis. Never an unknown. (A chassis pickup.)
      - ``"free"``    : a solved unknown (an outboard ball joint, a free node).
      - ``"carried"`` : rigidly carried by a *body*; its position is reconstructed
                        from that body's pose, not solved directly. Set
                        ``carrier`` to the body name.

    ``pos`` is always the *current* position (mm). For ground points it stays
    put; for free points it is the live unknown; for carried points it is
    refreshed from the carrier each solve.
    """
    name: str
    pos: np.ndarray
    role: str = "free"            # "ground" | "free" | "carried"
    carrier: str | None = None    # body name when role == "carried"

    def __post_init__(self):
        self.pos = _vec(self.pos)
        if self.role not in ("ground", "free", "carried"):
            raise ValueError(f"point {self.name}: bad role {self.role!r}")


# --------------------------------------------------------------------------- #
#  Rigid bodies
# --------------------------------------------------------------------------- #
@dataclass
class Body:
    """A rigid body whose pose (R, o) is recovered each solve from a set of its
    *defining* points, then used to carry other ('carried') points rigidly.

    A body is defined by >= 3 non-collinear points it owns. Its local frame is
    built once at static from those points (``_frame_from_points``) and the local
    coordinates of every carried point are frozen. At each travel the same
    defining points (now moved) rebuild the frame, and carried points follow.

    This is the mechanism by which a wheel carrier (upright/knuckle), a rocker,
    a beam axle, or any other rigid member transports the wheel-centre, contact
    patch, spin axis, pushrod pickup, etc., through travel — for *every*
    topology, not just the wishbone.
    """
    name: str
    defining_points: list[str]          # >=3 point names that move with the body
    _R0: np.ndarray | None = field(default=None, repr=False)
    _o0: np.ndarray | None = field(default=None, repr=False)
    _local: dict | None = field(default=None, repr=False)   # carried name -> local xyz

    @staticmethod
    def _frame_from_points(pts: list[np.ndarray]):
        """Orthonormal frame (R, origin) from >=3 points.

        origin = pts[0]; x toward pts[1]; the plane is completed by pts[2]. With
        exactly 3 points this is the unique rigid frame; with more, the extra
        points are still carried correctly because the body is rigid (they were
        frozen relative to this same frame at static).
        """
        o = np.asarray(pts[0], float).copy()
        x = _unit(np.asarray(pts[1], float) - o)
        ref = np.asarray(pts[2], float) - o
        z = _unit(np.cross(x, ref))
        if np.linalg.norm(z) < 1e-9:
            # collinear fallback: perturb with a world axis
            z = _unit(np.cross(x, np.array([0.0, 0.0, 1.0])))
            if np.linalg.norm(z) < 1e-9:
                z = _unit(np.cross(x, np.array([0.0, 1.0, 0.0])))
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        return R, o

    def freeze(self, point_lookup: Callable[[str], np.ndarray],
               carried: dict[str, np.ndarray]):
        """Capture the static frame and freeze local coords of carried points."""
        defs = [point_lookup(n) for n in self.defining_points]
        self._R0, self._o0 = self._frame_from_points(defs)
        self._local = {}
        for nm, world in carried.items():
            self._local[nm] = self._R0.T @ (np.asarray(world, float) - self._o0)

    def pose(self, point_lookup: Callable[[str], np.ndarray]):
        """Current (R, origin) of the body from its (now moved) defining points."""
        defs = [point_lookup(n) for n in self.defining_points]
        return self._frame_from_points(defs)

    def carry(self, local_name: str, R: np.ndarray, o: np.ndarray) -> np.ndarray:
        return o + R @ self._local[local_name]


# --------------------------------------------------------------------------- #
#  Constraints
# --------------------------------------------------------------------------- #
#  Every constraint exposes:
#     n_eq                 : number of scalar residual equations it contributes
#     residual(ctx)        : -> length-n_eq array (0 when satisfied)
#  `ctx` gives the constraint live access to point positions and body poses.
# --------------------------------------------------------------------------- #
class _Ctx:
    """Solve-time context handed to each constraint's residual()."""
    def __init__(self, get_point: Callable[[str], np.ndarray],
                 get_body_pose: Callable[[str], tuple]):
        self.P = get_point
        self.body_pose = get_body_pose


@dataclass
class Constraint:
    """Base class. Subclasses set ``n_eq`` and implement ``residual``."""
    n_eq: int = 1

    def residual(self, ctx: _Ctx) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError

    # introspection used by templates / reports
    def members(self) -> list[tuple[str, str, str]]:
        """Optional: list of (label, outboard_pt, inboard_pt) two-force members
        this constraint represents, for load-path analysis. Empty if N/A."""
        return []


@dataclass
class Link(Constraint):
    """Rigid two-force link: |a - b| held at its static length.

    The workhorse — a control-arm leg, a tie rod, a track rod, a radius rod, a
    pushrod, a toe link, a camber link, a Panhard bar … anything that is a
    distance constraint between two points. One scalar equation.
    """
    a: str = ""
    b: str = ""
    length: float | None = None      # filled at freeze() from static geometry
    label: str | None = None

    def __post_init__(self):
        self.n_eq = 1

    def residual(self, ctx):
        return np.array([np.linalg.norm(ctx.P(self.a) - ctx.P(self.b)) - self.length])

    def members(self):
        return [(self.label or f"{self.a}->{self.b}", self.a, self.b)]


@dataclass
class Coincident(Constraint):
    """Two points are the same point (a spherical/ball joint expressed as a
    shared node, or a pin tying two bodies). Three scalar equations."""
    a: str = ""
    b: str = ""

    def __post_init__(self):
        self.n_eq = 3

    def residual(self, ctx):
        return ctx.P(self.a) - ctx.P(self.b)


@dataclass
class OnLine(Constraint):
    """Point ``p`` lies on the line through fixed ``base`` with fixed direction
    ``axis`` (a *prismatic/sliding guide*). Used for the MacPherson strut axis,
    a steering-rack slider, or any in-line joint. Two scalar equations (the two
    components of the perpendicular offset)."""
    p: str = ""
    base: str = ""
    axis: np.ndarray | None = None     # world direction; need not be unit
    _e1: np.ndarray | None = None
    _e2: np.ndarray | None = None

    def __post_init__(self):
        self.n_eq = 2

    def _frame(self):
        d = _unit(np.asarray(self.axis, float))
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, d)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(d, ref))
        e2 = np.cross(d, e1)
        return e1, e2

    def residual(self, ctx):
        if self._e1 is None:
            self._e1, self._e2 = self._frame()
        v = ctx.P(self.p) - ctx.P(self.base)
        return np.array([np.dot(v, self._e1), np.dot(v, self._e2)])


@dataclass
class InPlane(Constraint):
    """Point ``p`` stays in a plane through ``base`` with unit normal ``normal``
    (a planar joint half, a leaf-spring shackle approximation, a body kept in a
    symmetry plane). One scalar equation."""
    p: str = ""
    base: str = ""
    normal: np.ndarray | None = None
    offset: float | None = None        # frozen so the plane passes through static p

    def __post_init__(self):
        self.n_eq = 1

    def residual(self, ctx):
        n = _unit(np.asarray(self.normal, float))
        return np.array([np.dot(ctx.P(self.p) - ctx.P(self.base), n) - (self.offset or 0.0)])


@dataclass
class Revolute(Constraint):
    """``p`` is constrained to a circle: fixed distance from pivot ``base`` AND
    in the plane normal to ``axis`` through ``base`` (a hinge / trailing-arm
    pivot / twist-beam bush expressed kinematically). Distance + plane = two
    equations; combined with the body rigidity this realises a revolute joint."""
    p: str = ""
    base: str = ""
    axis: np.ndarray | None = None
    radius: float | None = None
    offset: float | None = None

    def __post_init__(self):
        self.n_eq = 2

    def residual(self, ctx):
        n = _unit(np.asarray(self.axis, float))
        v = ctx.P(self.p) - ctx.P(self.base)
        return np.array([np.linalg.norm(v) - self.radius,
                         np.dot(v, n) - (self.offset or 0.0)])


@dataclass
class DriveZ(Constraint):
    """Driving constraint: pin a point's z to (static_z + travel). This is the
    single DOF the sweep marches. Replaceable by DriveAlong for non-vertical
    actuation (e.g. a strut that travels along its own axis)."""
    p: str = ""
    z0: float | None = None
    travel: float = 0.0

    def __post_init__(self):
        self.n_eq = 1

    def residual(self, ctx):
        return np.array([ctx.P(self.p)[2] - (self.z0 + self.travel)])


@dataclass
class RackTranslation(Constraint):
    """Steering DOF: hold the inner tie/track-rod point at its static position
    plus a lateral rack displacement ``delta`` along ``axis`` (default +y). Lets
    any steered topology be swept for bump-steer *and* steered to any rack
    position. Three equations (it fully places the inner point)."""
    p: str = ""
    base_pos: np.ndarray | None = None
    axis: np.ndarray | None = None
    delta: float = 0.0

    def __post_init__(self):
        self.n_eq = 3

    def residual(self, ctx):
        d = _unit(np.asarray(self.axis if self.axis is not None else [0, 1, 0], float))
        target = np.asarray(self.base_pos, float) + self.delta * d
        return ctx.P(self.p) - target


@dataclass
class AxleRoll(Constraint):
    """Couples the two wheel ends of a *rigid beam/solid axle*: the axle is one
    rigid body, so the two carried wheel points keep a fixed distance and the
    axle stays rigid. This is expressed simply by making both wheel carriers
    defining points of one Body; the AxleRoll constraint additionally fixes the
    axle's *length* between the two ends so a single-corner drive coordinate
    rolls the whole axle. One equation."""
    end_a: str = ""
    end_b: str = ""
    length: float | None = None

    def __post_init__(self):
        self.n_eq = 1

    def residual(self, ctx):
        return np.array([np.linalg.norm(ctx.P(self.end_a) - ctx.P(self.end_b)) - self.length])


# --------------------------------------------------------------------------- #
#  Mechanism — the assembled topology
# --------------------------------------------------------------------------- #
@dataclass
class Mechanism:
    """A complete, solvable suspension topology.

    Fields:
      points        : name -> Point
      bodies        : name -> Body
      constraints   : list[Constraint]
      carrier_body  : name of the rigid body that is the wheel carrier (upright /
                      knuckle / hub / axle end). Its pose defines wheel motion.
      wheel_center  : name of the carried point at the wheel centre
      contact_patch : name of the carried point at the contact patch
      spin_axis_pts : (a, b) carried point names whose difference is the wheel
                      spin axis (outboard +). Optional; if None a default spin
                      axis is frozen from static camber/toe of the carrier.
      drive_point   : name of the point whose vertical position is the DOF
      label         : human topology name (e.g. "MacPherson strut")
    """
    points: dict
    bodies: dict
    constraints: list
    carrier_body: str
    wheel_center: str
    contact_patch: str
    drive_point: str
    spin_axis_pts: tuple | None = None
    static_camber: float = 0.0
    static_toe: float = 0.0
    label: str = "custom"
    steer_point: str | None = None        # inner rod point that the rack moves
    steer_axis: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ---- internal solve bookkeeping (filled by compile()) -------------- #
    _free_names: list = field(default=None, repr=False)
    _drive: DriveZ | None = field(default=None, repr=False)
    _rack: RackTranslation | None = field(default=None, repr=False)
    _spin_local: np.ndarray | None = field(default=None, repr=False)
    _compiled: bool = field(default=False, repr=False)

    # =================================================================== #
    def compile(self):
        """Freeze static lengths, body frames, carried locals, and assemble the
        list of free unknowns. Idempotent-ish: call once after construction."""
        # 1) classify points
        self._free_names = [n for n, p in self.points.items() if p.role == "free"]

        # current-position lookup that resolves carried points via their body
        def get_point(name):
            return self.points[name].pos

        # 2) freeze body frames + locals. A body's carried points are every
        #    point whose carrier is this body.
        for bname, body in self.bodies.items():
            carried = {n: p.pos for n, p in self.points.items()
                       if p.role == "carried" and p.carrier == bname}
            # defining points must be resolvable now (free/ground/other-carried)
            body.freeze(get_point, carried)

        # 3) freeze constraint static parameters
        for c in self.constraints:
            if isinstance(c, Link) and c.length is None:
                c.length = float(np.linalg.norm(self.points[c.a].pos - self.points[c.b].pos))
            if isinstance(c, OnLine):
                c._e1, c._e2 = c._frame()
            if isinstance(c, InPlane) and c.offset is None:
                n = _unit(np.asarray(c.normal, float))
                c.offset = float(np.dot(self.points[c.p].pos - self.points[c.base].pos, n))
            if isinstance(c, Revolute):
                v = self.points[c.p].pos - self.points[c.base].pos
                if c.radius is None:
                    c.radius = float(np.linalg.norm(v))
                if c.offset is None:
                    n = _unit(np.asarray(c.axis, float))
                    c.offset = float(np.dot(v, n))
            if isinstance(c, AxleRoll) and c.length is None:
                c.length = float(np.linalg.norm(self.points[c.end_a].pos - self.points[c.end_b].pos))

        # 4) create / locate the driving + rack constraints
        self._drive = DriveZ(p=self.drive_point, z0=float(self.points[self.drive_point].pos[2]))
        self.constraints = [c for c in self.constraints if not isinstance(c, DriveZ)]
        self.constraints.append(self._drive)

        if self.steer_point is not None:
            self._rack = RackTranslation(
                p=self.steer_point,
                base_pos=self.points[self.steer_point].pos.copy(),
                axis=self.steer_axis if self.steer_axis is not None else np.array([0, 1, 0.]))
            self.constraints = [c for c in self.constraints if not isinstance(c, RackTranslation)]
            self.constraints.append(self._rack)

        # 5) freeze the wheel spin axis in the carrier-local frame
        body = self.bodies[self.carrier_body]
        R0, o0 = body._R0, body._o0
        if self.spin_axis_pts is not None:
            a, b = self.spin_axis_pts
            spin0 = _unit(self.points[b].pos - self.points[a].pos)
        else:
            cam = np.radians(self.static_camber)
            toe = np.radians(self.static_toe)
            spin0 = _unit(np.array([np.sin(toe), np.cos(toe) * np.cos(cam), -np.sin(cam)]))
        self._spin_local = R0.T @ spin0
        self._compiled = True
        return self

    # ------------------------------------------------------------------- #
    def _refresh_carried(self):
        """Rebuild every carried point from its body's current pose."""
        def get_point(name):
            return self.points[name].pos
        for bname, body in self.bodies.items():
            R, o = body.pose(get_point)
            for n, p in self.points.items():
                if p.role == "carried" and p.carrier == bname:
                    p.pos = body.carry(n, R, o)

    def _set_free(self, q):
        for nm, i in zip(self._free_names, range(0, len(q), 3)):
            self.points[nm].pos = q[i:i + 3]

    def _get_free(self):
        return np.concatenate([self.points[n].pos for n in self._free_names])

    def _residuals(self, q):
        self._set_free(q)
        self._refresh_carried()

        def get_point(name):
            return self.points[name].pos

        def get_body_pose(name):
            return self.bodies[name].pose(get_point)

        ctx = _Ctx(get_point, get_body_pose)
        out = []
        for c in self.constraints:
            out.append(np.asarray(c.residual(ctx), float).reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)

    # ------------------------------------------------------------------- #
    def solve(self, travel: float = 0.0, steer: float = 0.0, seed=None):
        """Solve the mechanism at a wheel travel (mm) and steer/rack (mm).

        Returns a dict of solved point positions plus diagnostics. ``seed`` is a
        prior free-vector for branch-stable warm starting (passed by sweep())."""
        if not self._compiled:
            self.compile()
        self._drive.travel = float(travel)
        if self._rack is not None:
            self._rack.delta = float(steer)

        q0 = seed if seed is not None else self._get_free()
        # LM requires n_residuals >= n_unknowns. A well-posed 1-DOF mechanism is
        # square or over-determined, but a template may legitimately be built
        # under-determined (extra internal freedom the user hasn't pinned, e.g. a
        # body that can still spin about an unconstrained axis). Fall back to the
        # trust-region 'trf' method, which handles the rectangular case and still
        # finds the minimum-norm-ish solution, so the tool degrades gracefully
        # instead of crashing.
        n_res = self._residuals(q0).size
        n_unk = q0.size
        method = "lm" if n_res >= n_unk else "trf"
        sol = least_squares(self._residuals, q0, method=method,
                            max_nfev=600, xtol=1e-12, ftol=1e-12)
        self._set_free(sol.x)
        self._refresh_carried()
        max_resid = float(np.max(np.abs(sol.fun))) if sol.fun.size else 0.0

        def get_point(name):
            return self.points[name].pos
        body = self.bodies[self.carrier_body]
        R, o = body.pose(get_point)
        spin = _unit(R @ self._spin_local)

        positions = {n: self.points[n].pos.copy() for n in self.points}
        return {
            "positions": positions,
            "free": sol.x.copy(),
            "carrier_R": R, "carrier_o": o,
            "spin_axis": spin,
            "wheel_center": positions[self.wheel_center],
            "contact_patch": positions[self.contact_patch],
            "max_residual": max_resid,
            "converged": max_resid < 0.1,
            "travel": float(travel), "steer": float(steer),
        }

    def sweep(self, travel_min=-30.0, travel_max=30.0, n=41, steer=0.0):
        """Branch-stable travel sweep (warm-started outward from static)."""
        if not self._compiled:
            self.compile()
        travels = np.linspace(travel_min, travel_max, n)
        below = sorted([t for t in travels if t < 0], reverse=True)
        above = sorted([t for t in travels if t > 0])
        zero = [t for t in travels if t == 0]

        results = {}
        static = self.solve(0.0, steer=steer)
        seed0 = static["free"]
        for t in zero:
            results[t] = static
        seed = seed0
        for t in above:
            r = self.solve(t, steer=steer, seed=seed)
            seed = r["free"]; results[t] = r
        seed = seed0
        for t in below:
            r = self.solve(t, steer=steer, seed=seed)
            seed = r["free"]; results[t] = r
        # always restore static so the mechanism's live state is the rest pose
        self.solve(0.0, steer=steer)
        return [results[t] for t in travels]

    # ------------------------------------------------------------------- #
    #  two-force member list (for load-path / member-force analysis)
    # ------------------------------------------------------------------- #
    def force_members(self):
        """List of (label, outboard_pt, inboard_pt) two-force members across all
        Link/Revolute-style constraints — the generic analogue of loadpath.MEMBERS.
        Outboard = the point nearer the wheel carrier (heuristic: a carried or
        free point); inboard = the ground/chassis point."""
        out = []
        for c in self.constraints:
            for (label, a, b) in c.members():
                ra = self.points[a].role
                rb = self.points[b].role
                # inboard = ground if exactly one end is ground
                if ra == "ground" and rb != "ground":
                    out.append((label, b, a))
                elif rb == "ground" and ra != "ground":
                    out.append((label, a, b))
                else:
                    out.append((label, a, b))
        return out


# --------------------------------------------------------------------------- #
#  builder convenience
# --------------------------------------------------------------------------- #
class MechanismBuilder:
    """Fluent helper for assembling a Mechanism — used by the topology templates
    and available to users for entirely free-form / experimental layouts.

    Example (a contrived 'experimental' corner that fits no textbook name)::

        b = MechanismBuilder("my_weird_corner")
        b.ground("c1", [ -50, 200, 120]); b.ground("c2", [120, 200, 118])
        b.free("hub", [0, 575, 115])
        b.body("knuckle", ["hub", "wc", "cp"])
        b.carried("wc", "knuckle", [0, 600, 228])
        b.carried("cp", "knuckle", [0, 605, 0])
        b.link("hub", "c1"); b.link("hub", "c2")
        b.in_plane("hub", "c1", normal=[1,0,0])   # weird planar guide
        m = b.finish(carrier="knuckle", wheel_center="wc",
                     contact_patch="cp", drive_point="hub")
    """
    def __init__(self, label="custom"):
        self.label = label
        self.points: dict = {}
        self.bodies: dict = {}
        self.constraints: list = []

    # points
    def ground(self, name, xyz):
        self.points[name] = Point(name, _vec(xyz), role="ground"); return self

    def free(self, name, xyz):
        self.points[name] = Point(name, _vec(xyz), role="free"); return self

    def carried(self, name, body, xyz):
        self.points[name] = Point(name, _vec(xyz), role="carried", carrier=body); return self

    # bodies
    def body(self, name, defining_points):
        self.bodies[name] = Body(name, list(defining_points)); return self

    # constraints
    def link(self, a, b, label=None):
        self.constraints.append(Link(a=a, b=b, label=label or f"{a}->{b}")); return self

    def coincident(self, a, b):
        self.constraints.append(Coincident(a=a, b=b)); return self

    def on_line(self, p, base, axis):
        self.constraints.append(OnLine(p=p, base=base, axis=_vec(axis))); return self

    def in_plane(self, p, base, normal):
        self.constraints.append(InPlane(p=p, base=base, normal=_vec(normal))); return self

    def revolute(self, p, base, axis):
        self.constraints.append(Revolute(p=p, base=base, axis=_vec(axis))); return self

    def axle_roll(self, end_a, end_b):
        self.constraints.append(AxleRoll(end_a=end_a, end_b=end_b)); return self

    def constraint(self, c: Constraint):
        self.constraints.append(c); return self

    def finish(self, *, carrier, wheel_center, contact_patch, drive_point,
               spin_axis_pts=None, static_camber=0.0, static_toe=0.0,
               steer_point=None, steer_axis=None, label=None, meta=None) -> Mechanism:
        m = Mechanism(
            points=self.points, bodies=self.bodies, constraints=self.constraints,
            carrier_body=carrier, wheel_center=wheel_center,
            contact_patch=contact_patch, drive_point=drive_point,
            spin_axis_pts=spin_axis_pts, static_camber=static_camber,
            static_toe=static_toe, steer_point=steer_point, steer_axis=steer_axis,
            label=label or self.label, meta=meta or {})
        return m.compile()
