# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Parameterised suspension-topology templates.
=============================================

Each function here returns a fully-compiled :class:`~suspension.topology.Mechanism`
for a named architecture, built from a handful of physically meaningful
parameters. They are the "drop-in" library the brief asks for:

    * ``double_wishbone``   — unequal-length A-arms (the classic FSAE corner)
    * ``macpherson_strut``  — strut + single lower arm + tie/track rod
    * ``multilink``         — N independent links to one upright (3/4/5-link)
    * ``trailing_arm``      — pure trailing arm on a lateral pivot axis
    * ``semi_trailing_arm`` — trailing arm with a skewed pivot axis
    * ``solid_axle``        — live/beam axle on links + Panhard or Watt's
    * ``twist_beam``        — torsion-beam (compound-crank) rear
    * ``truck_steer_linkage`` — heavy-truck beam-axle steering (kingpin +
                               drag link + tie rod / steering arm)
    * ``custom`` / ``from_links`` — free-form builder for experimental layouts
                               that fit no standard geometric definition.

All templates accept SAE-axis hardpoints in millimetres (x rear+, y right+,
z up+) and return a Mechanism whose ``carrier_body`` pose drives the wheel.
Where a template needs the wheel carrier to be rigid it supplies the carrier as
a Body defined by enough of its own points; where a link's far end is a chassis
pickup it is a ground point; where a node is shared between members it is a free
point. The driving DOF is wheel-vertical unless noted.

These templates are intentionally thin: the physics all lives in the kernel.
Adding a new architecture is a matter of listing its points, its carrier body,
and its constraints — not writing a new solver.
"""

from __future__ import annotations

import numpy as np
from .topology import (
    MechanismBuilder, Mechanism, Link, OnLine, InPlane, Revolute,
    Coincident, AxleRoll, Constraint,
)


def _v(x):
    return np.asarray(x, float).reshape(3)


# --------------------------------------------------------------------------- #
#  DOUBLE WISHBONE  (reference / FSAE)
# --------------------------------------------------------------------------- #
def double_wishbone(
    upper_front_inner, upper_rear_inner, lower_front_inner, lower_rear_inner,
    upper_outer, lower_outer, tie_rod_inner, tie_rod_outer,
    wheel_center, contact_patch, *,
    static_camber=-1.5, static_toe=0.0, label="double wishbone",
) -> Mechanism:
    """Unequal-length A-arm corner. Identical kinematics to the legacy solver,
    expressed in the generic kernel so it sits alongside every other topology."""
    b = MechanismBuilder(label)
    b.ground("ufi", upper_front_inner); b.ground("uri", upper_rear_inner)
    b.ground("lfi", lower_front_inner); b.ground("lri", lower_rear_inner)
    b.ground("tri", tie_rod_inner)
    b.free("uo", upper_outer); b.free("lo", lower_outer); b.free("tro", tie_rod_outer)
    b.body("upright", ["lo", "uo", "tro"])
    b.carried("wc", "upright", wheel_center)
    b.carried("cp", "upright", contact_patch)
    b.link("uo", "ufi", "UF"); b.link("uo", "uri", "UR")
    b.link("lo", "lfi", "LF"); b.link("lo", "lri", "LR")
    b.link("uo", "lo", "upright")
    b.link("tro", "tri", "TR")
    b.link("tro", "lo"); b.link("tro", "uo")
    return b.finish(carrier="upright", wheel_center="wc", contact_patch="cp",
                    drive_point="lo", static_camber=static_camber,
                    static_toe=static_toe, steer_point="tri", label=label,
                    meta={"family": "independent", "steered": True})


# --------------------------------------------------------------------------- #
#  MACPHERSON STRUT
# --------------------------------------------------------------------------- #
def macpherson_strut(
    strut_top, lower_front_inner, lower_rear_inner, lower_outer,
    strut_lower, tie_rod_inner, tie_rod_outer,
    wheel_center, contact_patch, *,
    static_camber=-1.0, static_toe=0.0, label="MacPherson strut",
) -> Mechanism:
    """MacPherson strut: a single lower control arm locates the lower ball joint,
    and the strut axis (top mount -> lower strut/knuckle point) is the upper
    guide. The knuckle slides/rotates about the strut axis; the tie rod sets toe.

    Modelled as: lower ball joint on the two lower-arm spheres; a strut node
    ``sl`` carried by the knuckle and held on the line from ``strut_top`` along
    the strut axis (OnLine = the prismatic strut). The knuckle is the carrier
    body defined by the lower ball joint, the strut node, and the tie-rod outer.
    """
    b = MechanismBuilder(label)
    b.ground("lfi", lower_front_inner); b.ground("lri", lower_rear_inner)
    b.ground("st", strut_top); b.ground("tri", tie_rod_inner)
    b.free("lo", lower_outer); b.free("sl", strut_lower); b.free("tro", tie_rod_outer)
    b.body("knuckle", ["lo", "sl", "tro"])
    b.carried("wc", "knuckle", wheel_center)
    b.carried("cp", "knuckle", contact_patch)
    # lower control arm (two legs to the same ball joint)
    b.link("lo", "lfi", "LF"); b.link("lo", "lri", "LR")
    # strut: knuckle strut-point slides along the strut axis through the top mount
    axis = _v(strut_lower) - _v(strut_top)
    b.on_line("sl", "st", axis)            # prismatic strut guide
    b.link("sl", "lo", "strut_body")       # strut is rigid to the knuckle (length)
    # tie rod sets toe; tro rigid to the knuckle
    b.link("tro", "tri", "TR")
    b.link("tro", "lo"); b.link("tro", "sl")
    return b.finish(carrier="knuckle", wheel_center="wc", contact_patch="cp",
                    drive_point="lo", static_camber=static_camber,
                    static_toe=static_toe, steer_point="tri", label=label,
                    meta={"family": "independent", "steered": True})


# --------------------------------------------------------------------------- #
#  MULTI-LINK  (N independent links)
# --------------------------------------------------------------------------- #
def multilink(
    links, upright_points, wheel_center, contact_patch, *,
    steer_link_index=None, static_camber=-1.0, static_toe=0.0,
    label="multi-link",
) -> Mechanism:
    """General N-link independent suspension (3-, 4-, or 5-link).

    Parameters
    ----------
    links : list of (inner_xyz, outer_xyz) pairs
        Each is one rigid link from a chassis pickup (inner) to a point on the
        upright (outer). Typically 5 for a fully-located rear knuckle, 4 with a
        toe link, etc. The outer points are the upright's hardpoints.
    upright_points : list of 3 outer-point indices
        Which three of the link outers define the rigid upright frame. Must be
        non-collinear. (For a 5-link, e.g. the two camber-link outers and a toe
        outer.)
    steer_link_index : int | None
        If given, that link's inner point is the steered (rack) node, so the
        suspension can be swept for bump-steer and steered.

    Each outer point is a *free* node tied to its inner by a Link; three of them
    additionally serve as the upright's defining points, and every other outer
    is *carried* by the upright so it rides rigidly (this is what makes the link
    set over-determine the knuckle into 1 DOF, exactly like a real 5-link).
    """
    b = MechanismBuilder(label)
    n = len(links)
    if n < 3:
        raise ValueError("multilink needs at least 3 links to locate an upright")
    # name inner/outer points
    for i, (inner, outer) in enumerate(links):
        b.ground(f"in{i}", inner)
    # the three defining outers are free; the rest are carried by the upright
    def_idx = list(upright_points)
    if len(set(def_idx)) != 3:
        raise ValueError("upright_points must be three distinct link indices")
    for i, (inner, outer) in enumerate(links):
        if i in def_idx:
            b.free(f"out{i}", outer)
    names_def = [f"out{i}" for i in def_idx]
    b.body("upright", names_def)
    # carried outers + wheel pts
    for i, (inner, outer) in enumerate(links):
        if i not in def_idx:
            b.carried(f"out{i}", "upright", outer)
    b.carried("wc", "upright", wheel_center)
    b.carried("cp", "upright", contact_patch)
    # link constraints (every link holds its length)
    for i, (inner, outer) in enumerate(links):
        b.link(f"out{i}", f"in{i}", f"L{i}")
    # drive on one of the defining outers (lowest one is a good DOF anchor)
    drive_i = min(def_idx, key=lambda i: links[i][1][2])
    steer_pt = f"in{steer_link_index}" if steer_link_index is not None else None
    return b.finish(carrier="upright", wheel_center="wc", contact_patch="cp",
                    drive_point=f"out{drive_i}", static_camber=static_camber,
                    static_toe=static_toe, steer_point=steer_pt, label=label,
                    meta={"family": "independent", "n_links": n,
                          "steered": steer_link_index is not None})


# --------------------------------------------------------------------------- #
#  TRAILING ARM  /  SEMI-TRAILING ARM
# --------------------------------------------------------------------------- #
def trailing_arm(
    pivot_inboard, pivot_outboard, hub, wheel_center, contact_patch, *,
    static_camber=0.0, static_toe=0.0, label="trailing arm",
) -> Mechanism:
    """Pure trailing arm: the wheel carrier is rigidly part of an arm that
    pivots about the lateral axis through ``pivot_inboard``→``pivot_outboard``
    (a transverse hinge). The hub rises and falls on a circle about that axis,
    holding camber/toe fixed in the arm frame (the defining trait of a trailing
    arm: no camber gain, pure vertical-ish motion).

    The arm is one rigid body defined by the two pivot points and the hub; the
    two pivots are revolute-constrained to ground (they are the hinge), and the
    hub is constrained to the circle about the pivot axis.
    """
    b = MechanismBuilder(label)
    b.ground("pi", pivot_inboard); b.ground("po", pivot_outboard)
    b.free("hub", hub)
    b.body("arm", ["pi", "po", "hub"])
    b.carried("wc", "arm", wheel_center)
    b.carried("cp", "arm", contact_patch)
    axis = _v(pivot_outboard) - _v(pivot_inboard)
    # hub swings on a circle about the pivot axis (revolute = dist + plane)
    b.revolute("hub", "pi", axis)
    # second distance to the far pivot makes the arm length rigid both ends
    b.link("hub", "po", "arm_brace")
    return b.finish(carrier="arm", wheel_center="wc", contact_patch="cp",
                    drive_point="hub", static_camber=static_camber,
                    static_toe=static_toe, label=label,
                    meta={"family": "dependent-arm", "steered": False})


def semi_trailing_arm(
    pivot_inboard, pivot_outboard, hub, wheel_center, contact_patch, *,
    static_camber=-1.0, static_toe=0.0, label="semi-trailing arm",
) -> Mechanism:
    """Semi-trailing arm: identical construction to ``trailing_arm`` but the
    pivot axis is deliberately *skewed* (not purely lateral), so the arm gains
    camber and toe with travel — the classic semi-trailing characteristic. The
    skew is entirely encoded by where the two pivot points sit; the kernel needs
    no special case."""
    return trailing_arm(pivot_inboard, pivot_outboard, hub, wheel_center,
                        contact_patch, static_camber=static_camber,
                        static_toe=static_toe, label=label)


# --------------------------------------------------------------------------- #
#  SOLID / LIVE AXLE
# --------------------------------------------------------------------------- #
def solid_axle(
    axle_left, axle_right, links, lateral_device, wheel_center_left,
    contact_patch_left, *, label="solid axle", static_camber=0.0,
    static_toe=0.0,
) -> Mechanism:
    """Beam / live axle located by longitudinal links + a lateral device.

    Parameters
    ----------
    axle_left, axle_right : the two ends of the rigid axle tube (the hubs).
    links : list of (inner_xyz, outer_xyz) — trailing/radius links and an
        upper link or two; their outers sit on the axle tube and are carried by
        the axle body.
    lateral_device : ("panhard", inner_xyz, outer_xyz)  OR
                     ("watts", pivot_xyz, top_chassis_xyz, bot_chassis_xyz)
        Locates the axle laterally. Panhard = one link chassis->axle. Watt's =
        a central pivot on the axle with two links to the chassis.
    wheel_center_left, contact_patch_left : left-hub carried points.

    The whole axle is ONE rigid body. Its pose is defined by the two tube ends
    plus the lateral-device attachment (the third, off-axis defining point), so
    every link outer, hub, wheel-centre and contact patch is *carried* and rides
    rigidly. The driving DOF is the left hub's z; ``AxleRoll`` keeps the tube
    length rigid so driving one end rolls the whole axle.
    """
    b = MechanismBuilder(label)
    b.free("axL", axle_left); b.free("axR", axle_right)

    # lateral device attachment becomes the 3rd (off-axis) defining point of the
    # axle body, so the axle frame is well-posed.
    kind = lateral_device[0].lower()
    if kind == "panhard":
        _, pin, pout = lateral_device
        b.free("lat", pout)            # axle-side panhard point (defining)
        b.ground("lat_in", pin)
    elif kind == "watts":
        _, pivot, top_ch, bot_ch = lateral_device
        b.free("lat", pivot)           # central pivot on axle (defining)
        b.ground("watt_top", top_ch); b.ground("watt_bot", bot_ch)
    else:
        raise ValueError("lateral_device must be ('panhard',...) or ('watts',...)")

    b.body("axle", ["axL", "axR", "lat"])
    # everything else rides on the axle
    for i, (inner, outer) in enumerate(links):
        b.ground(f"lin{i}", inner)
        b.carried(f"lout{i}", "axle", outer)
    b.carried("wc", "axle", wheel_center_left)
    b.carried("cp", "axle", contact_patch_left)

    # longitudinal links (each holds the axle-side outer to its chassis pickup)
    for i, (inner, outer) in enumerate(links):
        b.link(f"lout{i}", f"lin{i}", f"L{i}")
    # lateral device link(s)
    if kind == "panhard":
        b.link("lat", "lat_in", "panhard")
    else:
        b.link("lat", "watt_top", "watt_top_link")
        b.link("lat", "watt_bot", "watt_bot_link")
    # rigid axle tube length between the two ends
    b.axle_roll("axL", "axR")
    return b.finish(carrier="axle", wheel_center="wc", contact_patch="cp",
                    drive_point="axL", static_camber=static_camber,
                    static_toe=static_toe, label=label,
                    meta={"family": "beam-axle", "steered": False})


# --------------------------------------------------------------------------- #
#  TWIST-BEAM  (compound crank)
# --------------------------------------------------------------------------- #
def twist_beam(
    pivot_left, pivot_right, hub_left, hub_right, beam_left, beam_right,
    wheel_center_left, contact_patch_left, *, label="twist beam",
    static_camber=-0.5, static_toe=0.0,
) -> Mechanism:
    """Torsion-beam (compound-crank) rear axle.

    Two trailing arms pivot on the chassis at ``pivot_left/right`` and are joined
    by a transverse beam (``beam_left``→``beam_right``). In pure single-wheel
    bump the beam twists; in ride/roll the arms move together. Here we model ONE
    side as the swept corner: the left arm is a rigid body pivoting about the
    left bush, carrying its hub. The beam compliance (torsion) is out of the
    rigid-kinematics scope — geometrically the arm behaves like a trailing arm
    whose pivot axis runs from the bush toward the beam, which is exactly the
    twist-beam roll-camber behaviour. The right side is provided so the beam axis
    and inter-arm geometry are captured for reporting.
    """
    b = MechanismBuilder(label)
    b.ground("piL", pivot_left)
    b.ground("beamR_ground", beam_right)  # opposite beam end as a reference anchor
    b.free("hubL", hub_left)
    b.free("beamL", beam_left)
    b.body("armL", ["piL", "hubL", "beamL"])
    b.carried("wc", "armL", wheel_center_left)
    b.carried("cp", "armL", contact_patch_left)
    # arm pivots about the left bush axis (toward the beam = compound-crank axis)
    axis = _v(beam_left) - _v(pivot_left)
    b.revolute("hubL", "piL", axis)
    b.link("hubL", "beamL", "arm_to_beam")
    # The transverse beam joins the two arms: the left beam end is held a fixed
    # distance from the (anchored) right beam end AND from the left pivot, which
    # pins the remaining freedom so the single-corner sweep is determinate. In a
    # real twist beam this distance changes only by beam torsion (a compliance
    # effect handled elsewhere); kinematically it is the rigid beam length.
    b.link("beamL", "beamR_ground", "torsion_beam")
    b.link("beamL", "piL", "beam_brace")
    return b.finish(carrier="armL", wheel_center="wc", contact_patch="cp",
                    drive_point="hubL", static_camber=static_camber,
                    static_toe=static_toe, label=label,
                    meta={"family": "semi-dependent", "steered": False})


# --------------------------------------------------------------------------- #
#  HEAVY-TRUCK STEERING LINKAGE (beam axle + kingpin + drag/tie rod)
# --------------------------------------------------------------------------- #
def truck_steer_linkage(
    kingpin_top, kingpin_bottom, spindle, steering_arm, tie_rod_inner,
    drag_link_chassis, wheel_center, contact_patch, *,
    spring_perches=None, static_camber=0.0, static_toe=0.0,
    label="heavy-truck steer axle",
) -> Mechanism:
    """Heavy-truck front steer axle: a beam axle carries a kingpin about which
    the knuckle rotates; a steering arm on the knuckle is driven by the drag
    link (from the pitman arm / chassis) and tied to the opposite wheel by the
    tie rod.

    Modelled at the single-knuckle level (the kingpin is fixed to the beam axle,
    treated here as ground because the axle's own vertical motion is the leaf/
    air-spring travel handled separately):

      * knuckle rotates about the kingpin axis (top->bottom) — a revolute,
        realised by holding the spindle on the circle about the kingpin and the
        steering-arm end likewise, with the knuckle rigid.
      * the drag link / tie rod inner point is the steer (rack-equivalent) DOF,
        so the linkage can be swept for Ackermann and bump-steer.

    The vertical DOF here is the spindle height (axle travel); steering is the
    drag-link motion.
    """
    b = MechanismBuilder(label)
    b.ground("kpt", kingpin_top); b.ground("kpb", kingpin_bottom)
    b.ground("dragc", drag_link_chassis); b.ground("tri", tie_rod_inner)
    # The knuckle has exactly ONE DOF: rotation about the kingpin axis (steer).
    # Build a minimal, exactly-determined hinge: the spindle and steering-arm
    # points are solved (6 unknowns); the kingpin axis is the hinge. Two revolute
    # constraints (spindle and arm each on their kingpin circle = 4 eqns) plus
    # one rigid span between them (1 eqn) leaves a single shared rotation. The
    # remaining DOF is the steer, driven by holding the steering-arm end at a
    # commanded position along the drag-link / rack direction (the DriveZ on its
    # height selects the rotation; the tie rod to the opposite wheel and the drag
    # link to the pitman are the same kinematic input, reported via Ackermann).
    kp_axis = _v(kingpin_top) - _v(kingpin_bottom)
    b.free("sp", spindle); b.free("sa", steering_arm)
    b.body("knuckle", ["kpb", "sp", "sa"])     # kpb (ground) = hinge base
    b.carried("wc", "knuckle", wheel_center)
    b.carried("cp", "knuckle", contact_patch)
    b.revolute("sp", "kpb", kp_axis)           # 2 eqns
    b.revolute("sa", "kpb", kp_axis)           # 2 eqns
    b.link("sp", "sa", "knuckle_rigid")        # 1 eqn  -> 5 of 6 fixed, 1 DOF
    return b.finish(carrier="knuckle", wheel_center="wc", contact_patch="cp",
                    drive_point="sa", static_camber=static_camber,
                    static_toe=static_toe, steer_point="tri", label=label,
                    meta={"family": "beam-axle-steer", "steered": True,
                          "drive_is_steer": True,
                          "note": "1-DOF kingpin steer; 'travel' selects steer angle"})


# --------------------------------------------------------------------------- #
#  FREE-FORM / EXPERIMENTAL builder
# --------------------------------------------------------------------------- #
def from_links(
    chassis_points, free_points, carried_points, links,
    carrier_body, carrier_defining, wheel_center, contact_patch,
    drive_point, *, extra_constraints=None, steer_point=None, steer_axis=None,
    static_camber=0.0, static_toe=0.0, label="experimental",
) -> Mechanism:
    """Build *any* topology from raw ingredients — for configurations that fit no
    standard geometric definition.

    Parameters
    ----------
    chassis_points : dict name -> xyz   (ground / fixed pickups)
    free_points    : dict name -> xyz   (solved nodes, e.g. ball joints)
    carried_points : dict name -> (body, xyz)  (ride rigidly with a body)
    links          : list of (a, b[, label])   (rigid distance members)
    carrier_body   : name of the wheel-carrier body
    carrier_defining : list of >=3 point names defining the carrier
    extra_constraints : list of Constraint instances (OnLine, InPlane,
        Revolute, Coincident, AxleRoll, or custom subclasses) for guides, planar
        joints, sliders — whatever your experimental mechanism needs.

    This is the escape hatch: if you can describe how the parts are connected as
    a set of distance/guide/plane constraints, the kernel will solve it, with no
    requirement that the result resemble any catalogued suspension.
    """
    b = MechanismBuilder(label)
    for nm, xyz in (chassis_points or {}).items():
        b.ground(nm, xyz)
    for nm, xyz in (free_points or {}).items():
        b.free(nm, xyz)
    # the carrier body must be declared before its carried points
    bodies_needed = {carrier_body: carrier_defining}
    for nm, (body, xyz) in (carried_points or {}).items():
        bodies_needed.setdefault(body, None)
    for body, defn in bodies_needed.items():
        if defn := bodies_needed[body]:
            b.body(body, defn)
    for nm, (body, xyz) in (carried_points or {}).items():
        b.carried(nm, body, xyz)
    for ln in links:
        if len(ln) == 3:
            b.link(ln[0], ln[1], ln[2])
        else:
            b.link(ln[0], ln[1])
    for c in (extra_constraints or []):
        b.constraint(c)
    return b.finish(carrier=carrier_body, wheel_center=wheel_center,
                    contact_patch=contact_patch, drive_point=drive_point,
                    spin_axis_pts=None, static_camber=static_camber,
                    static_toe=static_toe, steer_point=steer_point,
                    steer_axis=steer_axis, label=label,
                    meta={"family": "experimental"})


# --------------------------------------------------------------------------- #
#  registry + sensible defaults for one-call template instantiation
# --------------------------------------------------------------------------- #
TEMPLATES = {
    "double_wishbone": double_wishbone,
    "macpherson_strut": macpherson_strut,
    "multilink": multilink,
    "trailing_arm": trailing_arm,
    "semi_trailing_arm": semi_trailing_arm,
    "solid_axle": solid_axle,
    "twist_beam": twist_beam,
    "truck_steer_linkage": truck_steer_linkage,
    "from_links": from_links,
}


def list_templates():
    """Names of all shipped topology templates."""
    return sorted(TEMPLATES.keys())


def example(name: str) -> Mechanism:
    """Return a ready-to-solve example Mechanism for a template name, with
    representative (right-side, ~1.55 m track) hardpoints. Handy for the UI's
    'load a starting geometry' menu and for tests.
    """
    if name == "double_wishbone":
        return double_wishbone(
            [-100, 240, 280.8], [130, 240, 299.2], [-110, 200, 122.5],
            [140, 200, 117.5], [12, 540, 300], [-5, 575, 110],
            [100, 230, 160], [90, 560, 150], [0, 600, 228], [0, 605, 0])
    if name == "macpherson_strut":
        return macpherson_strut(
            strut_top=[10, 520, 620], lower_front_inner=[-110, 200, 122.5],
            lower_rear_inner=[140, 200, 117.5], lower_outer=[-5, 575, 110],
            strut_lower=[0, 560, 360], tie_rod_inner=[120, 230, 160],
            tie_rod_outer=[110, 560, 175], wheel_center=[0, 600, 228],
            contact_patch=[0, 605, 0])
    if name == "multilink":
        links = [
            ([-60, 210, 130], [-10, 560, 120]),   # lower fore
            ([150, 215, 128], [60, 565, 118]),    # lower aft
            ([-40, 250, 330], [0, 545, 320]),     # upper camber
            ([160, 255, 332], [70, 548, 322]),    # upper aft
            ([130, 240, 200], [120, 560, 190]),   # toe link
        ]
        return multilink(links, upright_points=[0, 2, 4],
                         wheel_center=[0, 600, 228], contact_patch=[0, 605, 0],
                         steer_link_index=4)
    if name == "trailing_arm":
        return trailing_arm(
            pivot_inboard=[300, 150, 150], pivot_outboard=[300, 450, 150],
            hub=[-100, 575, 150], wheel_center=[0, 600, 228],
            contact_patch=[0, 605, 0])
    if name == "semi_trailing_arm":
        return semi_trailing_arm(
            pivot_inboard=[300, 150, 130], pivot_outboard=[260, 460, 175],
            hub=[-100, 575, 150], wheel_center=[0, 600, 228],
            contact_patch=[0, 605, 0])
    if name == "solid_axle":
        links = [
            ([400, 250, 120], [120, 250, 120]),   # lower trailing left
            ([400, -250, 120], [120, -250, 120]), # lower trailing right
            ([350, 120, 320], [120, 120, 300]),   # upper link
        ]
        return solid_axle(
            axle_left=[0, 600, 230], axle_right=[0, -600, 230], links=links,
            lateral_device=("panhard", [-50, -550, 280], [-50, 580, 285]),
            wheel_center_left=[0, 620, 230], contact_patch_left=[0, 625, 0])
    if name == "twist_beam":
        return twist_beam(
            pivot_left=[450, 350, 180], pivot_right=[450, -350, 180],
            hub_left=[-150, 575, 160], hub_right=[-150, -575, 160],
            beam_left=[150, 300, 170], beam_right=[150, -300, 170],
            wheel_center_left=[0, 600, 228], contact_patch_left=[0, 605, 0])
    if name == "truck_steer_linkage":
        return truck_steer_linkage(
            kingpin_top=[0, 900, 560], kingpin_bottom=[0, 920, 360],
            spindle=[0, 980, 470], steering_arm=[160, 870, 380],
            tie_rod_inner=[160, 400, 380], drag_link_chassis=[160, 300, 600],
            wheel_center=[0, 1010, 470], contact_patch=[0, 1015, 0])
    raise ValueError(f"no example for template {name!r}; "
                     f"choose from {list_templates()}")
