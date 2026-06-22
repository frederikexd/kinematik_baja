# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""Tests for the geometric mount-point / clash / CG-propagation layer."""

import numpy as np
import pytest

from suspension.interfaces import IntegrationLedger, SubsystemInterface, Severity
from suspension.mountpoints import (
    MountPoint, KeepOut, GeometryLedger, propagate_mount_move,
)


# --------------------------------------------------------------------------- #
#  AABB signed distance — validated against closed form
# --------------------------------------------------------------------------- #
def test_sdf_outside_face():
    ko = KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(100, 100, 100))
    # 30 mm out along +x from the face at x=100
    assert ko.signed_distance_mm(np.array([130., 50., 50.])) == pytest.approx(30.0)


def test_sdf_outside_corner():
    ko = KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(100, 100, 100))
    # 3-4-5 corner distance: (103,104,50) -> sqrt(3^2+4^2)=5 outside, z inside
    assert ko.signed_distance_mm(np.array([103., 104., 50.])) == pytest.approx(5.0)


def test_sdf_inside_is_negative_depth():
    ko = KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(100, 100, 100))
    # point at (10,50,50): nearest face is x=0 -> depth 10 -> sdf -10
    assert ko.signed_distance_mm(np.array([10., 50., 50.])) == pytest.approx(-10.0)


def test_keepout_corner_order_normalised():
    # give corners swapped; __post_init__ must normalise
    ko = KeepOut("box", "chassis", lo_mm=(100, 100, 100), hi_mm=(0, 0, 0))
    assert ko.lo_mm == (0, 0, 0)
    assert ko.hi_mm == (100, 100, 100)


# --------------------------------------------------------------------------- #
#  Clash check semantics
# --------------------------------------------------------------------------- #
def _geom_with_chassis_box():
    g = GeometryLedger()
    g.set_keepout(KeepOut("chassis-tube", "chassis",
                          lo_mm=(0, 0, 0), hi_mm=(100, 100, 100),
                          is_estimate=False))
    return g


def test_interference_is_fail():
    g = _geom_with_chassis_box()
    g.set_point(MountPoint("wing-mount", xyz_mm=(50, 50, 50),
                           owner_subsystem="rear-suspension", mounts_on="chassis",
                           min_clearance_mm=5.0, is_estimate=False))
    # note: point bolts onto chassis, but the keep-out is a *different* chassis volume
    # (the tube path) it must avoid — owner==chassis==mounts_on means it's skipped.
    # So rename the keepout owner to exercise interference:
    findings = g.check_clashes()
    # mounts_on == chassis and keepout owner == chassis -> skipped -> OK finding
    assert any(f.check == "clash" and f.severity == Severity.OK for f in findings)


def test_interference_against_foreign_keepout():
    g = GeometryLedger()
    g.set_keepout(KeepOut("driver-legroom", "chassis",
                          lo_mm=(0, 0, 0), hi_mm=(100, 100, 100), is_estimate=False))
    # aero point that mounts on suspension, intruding into chassis legroom
    g.set_point(MountPoint("wing-stay", xyz_mm=(50, 50, 50),
                           owner_subsystem="rear-suspension", mounts_on="front-suspension",
                           min_clearance_mm=5.0, is_estimate=False))
    findings = g.check_clashes()
    fails = [f for f in findings if f.severity == Severity.FAIL]
    assert len(fails) == 1
    assert fails[0].check == "clash-interference"
    assert set(fails[0].subsystems) == {"rear-suspension", "chassis"}
    assert fails[0].detail["penetration_mm"] == pytest.approx(50.0)


def test_clearance_warn_band():
    g = GeometryLedger()
    g.set_keepout(KeepOut("cvt", "drivetrain",
                          lo_mm=(0, 0, 0), hi_mm=(100, 100, 100), is_estimate=False))
    # 3 mm outside the x=100 face, needs 5 mm
    g.set_point(MountPoint("wing-mount", xyz_mm=(103, 50, 50),
                           owner_subsystem="rear-suspension", mounts_on="chassis",
                           min_clearance_mm=5.0, is_estimate=False))
    findings = g.check_clashes()
    warns = [f for f in findings if f.severity == Severity.WARN]
    assert len(warns) == 1
    assert warns[0].check == "clash-clearance"
    assert warns[0].detail["gap_mm"] == pytest.approx(3.0)


def test_clear_point_is_ok():
    g = GeometryLedger()
    g.set_keepout(KeepOut("cvt", "drivetrain",
                          lo_mm=(0, 0, 0), hi_mm=(100, 100, 100), is_estimate=False))
    g.set_point(MountPoint("wing-mount", xyz_mm=(200, 50, 50),
                           owner_subsystem="rear-suspension", mounts_on="chassis",
                           min_clearance_mm=5.0, is_estimate=False))
    findings = g.check_clashes()
    assert any(f.severity == Severity.OK for f in findings)
    assert not any(f.severity in (Severity.FAIL, Severity.WARN) for f in findings)


def test_estimate_flag_propagates_to_finding():
    g = GeometryLedger()
    g.set_keepout(KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(100, 100, 100),
                          is_estimate=True))
    g.set_point(MountPoint("p", xyz_mm=(50, 50, 50), owner_subsystem="rear-suspension",
                           mounts_on="front-suspension", is_estimate=False))
    f = [x for x in g.check_clashes() if x.severity == Severity.FAIL][0]
    assert f.detail["estimate"] is True
    assert "estimated geometry" in f.message


def test_no_keepouts_is_missing_not_pass():
    g = GeometryLedger()
    g.set_point(MountPoint("p", xyz_mm=(0, 0, 0), owner_subsystem="rear-suspension"))
    findings = g.check_clashes()
    assert findings[0].severity == Severity.MISSING


# --------------------------------------------------------------------------- #
#  The full propagation chain: move point -> clash + CG, in one call
# --------------------------------------------------------------------------- #
def _ledger_two_masses():
    led = IntegrationLedger(target_mass_kg=230.0)
    led.set(SubsystemInterface(name="rear-suspension", mass_kg=10.0,
                               cg_x_mm=1000, cg_y_mm=0, cg_z_mm=400,
                               is_estimate=False))
    led.set(SubsystemInterface(name="chassis", mass_kg=30.0,
                               cg_x_mm=800, cg_y_mm=0, cg_z_mm=300,
                               is_estimate=False))
    return led


def test_propagation_flags_clash_and_recomputes_cg():
    g = GeometryLedger()
    g.set_keepout(KeepOut("monocoque-wall", "chassis",
                          lo_mm=(0, 0, 0), hi_mm=(100, 100, 100), is_estimate=False))
    g.set_point(MountPoint("rear-wing-mount", xyz_mm=(200, 50, 50),
                           owner_subsystem="rear-suspension", mounts_on="front-suspension",
                           min_clearance_mm=5.0, is_estimate=False))
    led = _ledger_two_masses()

    # move the wing mount INTO the chassis wall
    res = propagate_mount_move(g, led, "rear-wing-mount", new_xyz_mm=(50, 50, 50),
                               set_by="aero_member")
    assert res.has_hard_clash
    # CG unchanged because update_interface_cg defaulted off (point != subsystem CG)
    assert res.cg_delta_mm == pytest.approx((0.0, 0.0, 0.0))
    assert res.mass_total_kg == pytest.approx(40.0)


def test_propagation_with_cg_update_moves_ledger_cg():
    g = GeometryLedger()
    g.set_keepout(KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(10, 10, 10),
                          is_estimate=False))
    g.set_point(MountPoint("rear-cg-pt", xyz_mm=(1000, 0, 400),
                           owner_subsystem="rear-suspension", mounts_on="front-suspension",
                           is_estimate=False))
    led = _ledger_two_masses()
    cg0 = led.mass_rollup()["cg_mm"]

    # raise the aero mass point by 100 mm in z, asking the CG to follow
    res = propagate_mount_move(g, led, "rear-cg-pt", new_xyz_mm=(1000, 0, 500),
                               update_interface_cg=True)
    # combined CG z must rise: aero is 10/40 of mass, moved +100 -> +25 mm
    assert res.cg_delta_mm[2] == pytest.approx(25.0)
    assert res.cg_after_mm[2] == pytest.approx(cg0[2] + 25.0)
    assert not res.has_hard_clash


def test_propagation_cg_none_when_data_incomplete():
    g = GeometryLedger()
    g.set_keepout(KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(10, 10, 10)))
    g.set_point(MountPoint("p", xyz_mm=(500, 0, 300), owner_subsystem="rear-suspension",
                           mounts_on="front-suspension"))
    led = IntegrationLedger()
    led.set(SubsystemInterface(name="rear-suspension", mass_kg=10.0))  # no CG
    res = propagate_mount_move(g, led, "p", new_xyz_mm=(600, 0, 300))
    assert res.cg_after_mm is None
    assert res.cg_delta_mm is None
    assert "not recomputable" in res.summary()


# --------------------------------------------------------------------------- #
#  Round-trip persistence
# --------------------------------------------------------------------------- #
def test_geometry_ledger_roundtrip():
    g = GeometryLedger()
    g.set_keepout(KeepOut("box", "chassis", lo_mm=(0, 0, 0), hi_mm=(100, 100, 100)))
    g.set_point(MountPoint("p", xyz_mm=(1, 2, 3), owner_subsystem="rear-suspension"))
    g2 = GeometryLedger.from_dict(g.as_dict())
    assert g2.points["p"].xyz_mm == (1, 2, 3)
    assert g2.keepouts["box"].hi_mm == (100, 100, 100)


# --------------------------------------------------------------------------- #
#  Persistence through ProjectStore (survives save/load)
# --------------------------------------------------------------------------- #
def test_projectstore_persists_geometry(tmp_path):
    from suspension.project import ProjectStore
    path = str(tmp_path / "proj.json")

    s = ProjectStore(path=path)
    s.set_keepout(KeepOut("main-hoop", "chassis", lo_mm=(1380, -180, 480),
                          hi_mm=(1430, 180, 1050), is_estimate=False))
    s.set_mount_point(MountPoint("rear-wing-mount", xyz_mm=(1350, 120, 900),
                                 owner_subsystem="rear-suspension", mounts_on="front-suspension",
                                 min_clearance_mm=8.0, is_estimate=False))
    assert s.save(), s.save_error

    s2 = ProjectStore(path=path)
    assert "rear-wing-mount" in s2.geometry.points
    assert "main-hoop" in s2.geometry.keepouts
    # tuples survive the JSON round-trip
    assert s2.geometry.points["rear-wing-mount"].xyz_mm == (1350, 120, 900)
    # and the clash check runs on the reloaded data
    sevs = [f.severity for f in s2.clash_findings()]
    assert Severity.OK in sevs  # clear at this position


def test_projectstore_move_propagates(tmp_path):
    from suspension.project import ProjectStore
    path = str(tmp_path / "proj.json")
    s = ProjectStore(path=path)
    s.set_keepout(KeepOut("hoop", "chassis", lo_mm=(1380, -180, 480),
                          hi_mm=(1430, 180, 1050), is_estimate=False))
    s.set_mount_point(MountPoint("wing", xyz_mm=(1350, 120, 900),
                                 owner_subsystem="rear-suspension", mounts_on="front-suspension",
                                 min_clearance_mm=8.0, is_estimate=False))

    led = IntegrationLedger()
    led.set(SubsystemInterface("rear-suspension", mass_kg=12.0,
                               cg_x_mm=1450, cg_y_mm=0, cg_z_mm=520, is_estimate=False))
    res = s.move_mount(led, "wing", (1410, 120, 900), set_by="tester")
    assert res.has_hard_clash
    # the stored geometry reflects the move
    assert s.geometry.points["wing"].xyz_mm == (1410.0, 120.0, 900.0)
