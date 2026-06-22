# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the architecture-agnostic topology engine (topology / topologies /
adapter). Verifies that:

  * the generic kernel reproduces the legacy double-wishbone kinematics,
  * every shipped template compiles, solves, and sweeps without diverging,
  * the adapter exposes the CornerState surface the rest of the tool needs, and
  * a non-wishbone topology drives the existing VehicleDynamics layer (roll
    centre, anti-dive) — the geometry->balance coupling, for any architecture.
  * the free-form builder accepts an experimental layout with no standard name.
"""

import numpy as np
import pytest

from suspension import topologies as T
from suspension.adapter import GenericKinematics
from suspension.topology import MechanismBuilder
from suspension.kinematics import SuspensionKinematics, Hardpoints


# --------------------------------------------------------------------------- #
def test_generic_matches_native_double_wishbone():
    """The generic kernel must reproduce the hard-coded wishbone solver."""
    gen = GenericKinematics(T.example("double_wishbone"))
    nat = SuspensionKinematics(Hardpoints.default())
    for t in (-30, -15, 0, 15, 30):
        gs = gen.solve_at_travel(t)
        ns = nat.solve_at_travel(t)
        assert abs(gs.camber - ns.camber) < 0.05, f"camber mismatch at {t}"
        assert abs(gs.wheel_center[2] - ns.wheel_center[2]) < 0.5
        assert abs(gs.caster - ns.caster) < 0.2
        assert abs(gs.kpi - ns.kpi) < 0.2


def test_native_and_generic_roll_centre_agree():
    """Roll-centre height (the z of the front-view IC) is what feeds dynamics;
    the velocity-based generic IC must agree with the link-based native one."""
    from suspension.dynamics import VehicleDynamics, VehicleParams
    gen = GenericKinematics(T.example("double_wishbone"))
    nat = SuspensionKinematics(Hardpoints.default())
    vd = VehicleDynamics(VehicleParams())
    rc_gen = vd.roll_center_height(gen, track=1550)
    rc_nat = vd.roll_center_height(nat, track=1550)
    assert abs(rc_gen - rc_nat) < 5.0, (rc_gen, rc_nat)


@pytest.mark.parametrize("name", [
    "double_wishbone", "macpherson_strut", "multilink",
    "trailing_arm", "semi_trailing_arm", "solid_axle", "twist_beam",
])
def test_every_template_sweeps_and_converges(name):
    kin = GenericKinematics(T.example(name))
    states = kin.sweep(-25, 25, 11)
    assert len(states) == 11
    # at least the central, well-conditioned portion must converge cleanly
    central = states[3:8]
    assert all(s.converged for s in central), f"{name} failed to converge near static"
    # camber must be finite and monotone-ish (no solver branch jumps)
    cambers = [s.camber for s in states if s.converged]
    assert all(np.isfinite(c) for c in cambers)
    assert max(cambers) - min(cambers) < 15.0, f"{name} camber jumped (branch flip?)"


def test_trailing_arm_has_no_camber_gain():
    """A pure trailing arm holds camber through travel — its defining trait."""
    kin = GenericKinematics(T.example("trailing_arm"))
    s = kin.sweep(-25, 25, 11)
    cambers = [x.camber for x in s]
    assert max(cambers) - min(cambers) < 0.05, cambers


def test_macpherson_gains_negative_camber_in_bump():
    kin = GenericKinematics(T.example("macpherson_strut"))
    droop = kin.solve_at_travel(-20).camber
    bump = kin.solve_at_travel(+20).camber
    assert bump < droop, "MacPherson should gain negative camber into bump"


def test_adapter_drives_vehicle_dynamics_for_non_wishbone():
    """A MacPherson front + multilink rear must run the existing dynamics layer."""
    from suspension.dynamics import VehicleDynamics, VehicleParams
    front = GenericKinematics(T.example("macpherson_strut"))
    rear = GenericKinematics(T.example("multilink"))
    vd = VehicleDynamics(VehicleParams(), front_kin=front, rear_kin=rear)
    rc_f = vd.roll_center_height(front, track=1550)
    rc_r = vd.roll_center_height(rear, track=1500)
    assert np.isfinite(rc_f) and np.isfinite(rc_r)
    assert -200 < rc_f < 400 and -200 < rc_r < 400
    ad = vd.anti_dive_pct(0.65)
    assert np.isfinite(ad)


def test_freeform_experimental_topology():
    """The free-form builder must solve a layout that fits no textbook name:
    a single lower arm + two arbitrary skew links + an in-plane guide."""
    b = MechanismBuilder("experimental")
    b.ground("c1", [-60, 200, 120]); b.ground("c2", [140, 205, 118])
    b.ground("c3", [40, 260, 330]); b.ground("c4", [120, 250, 200])
    b.free("lo", [-5, 575, 110]); b.free("uo", [10, 545, 320]); b.free("tro", [110, 560, 175])
    b.body("knuckle", ["lo", "uo", "tro"])
    b.carried("wc", "knuckle", [0, 600, 228])
    b.carried("cp", "knuckle", [0, 605, 0])
    b.link("lo", "c1"); b.link("lo", "c2")
    b.link("uo", "c3"); b.link("uo", "c1")
    b.link("tro", "c4"); b.link("tro", "lo"); b.link("tro", "uo")
    b.link("uo", "lo", "upright")
    m = b.finish(carrier="knuckle", wheel_center="wc", contact_patch="cp",
                 drive_point="lo", static_camber=-1.2)
    kin = GenericKinematics(m)
    s0 = kin.solve_at_travel(0.0)
    assert s0.converged
    assert abs(s0.camber - (-1.2)) < 0.5
    s_bump = kin.solve_at_travel(15.0)
    assert s_bump.converged


def test_hardpoint_shim_exposes_named_points():
    kin = GenericKinematics(T.example("macpherson_strut"))
    # the strut top is a real ground point on this topology
    assert hasattr(kin.hp, "st")
    assert kin.hp.st.shape == (3,)
    # a wishbone-only name is absent and raises (callers guard with getattr)
    with pytest.raises(AttributeError):
        _ = kin.hp.upper_front_inner


def test_list_templates_and_examples():
    names = T.list_templates()
    assert "macpherson_strut" in names and "solid_axle" in names
    for n in names:
        if n == "from_links":
            continue
        m = T.example(n)
        assert m._compiled
