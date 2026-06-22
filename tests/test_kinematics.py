# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Physics sanity tests for the kinematics engine.

These aren't exhaustive validation against a commercial solver — that's a great
PR to add — but they pin the conventions and catch regressions in the signs and
gains that matter most when tuning a real car.

Run:  python -m pytest tests/  (or just: python tests/test_kinematics.py)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import SuspensionKinematics, Hardpoints, VehicleDynamics, VehicleParams


def test_static_matches_design_intent():
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    assert abs(kin.static.camber - hp.static_camber) < 0.05
    assert abs(kin.static.toe - hp.static_toe) < 0.05


def test_linkage_closes_over_travel():
    kin = SuspensionKinematics(Hardpoints.default())
    for s in kin.sweep(-30, 30, 21):
        assert s.converged, f"linkage failed to close at travel {s.travel}"


def test_negative_camber_gain_in_bump():
    # Good FSAE geometry gains negative camber as the wheel moves into bump.
    kin = SuspensionKinematics(Hardpoints.default())
    c_bump = kin.solve_at_travel(20).camber
    c_droop = kin.solve_at_travel(-20).camber
    assert c_bump < c_droop, "expected more negative camber in bump"


def test_caster_positive_for_rearward_kingpin():
    kin = SuspensionKinematics(Hardpoints.default())
    assert kin.static.caster > 0


def test_roll_angle_is_physical():
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin)
    _, info = veh.lateral_load_transfer(1.2)
    assert 0 < info["roll_angle"] < 6, "roll angle out of physical range"


def test_outer_wheels_gain_load():
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin)
    loads, _ = veh.lateral_load_transfer(1.0)
    assert loads.fr > loads.fl and loads.rr > loads.rl


def test_load_conservation():
    kin = SuspensionKinematics(Hardpoints.default())
    p = VehicleParams()
    veh = VehicleDynamics(p, front_kin=kin, rear_kin=kin)
    loads, _ = veh.lateral_load_transfer(0.0)
    total = sum(loads.as_tuple())
    assert abs(total - p.mass * p.g) < 1.0


def test_max_g_in_reasonable_range():
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin)
    assert 0.9 < veh.max_lateral_g() < 2.2


def test_parallel_equal_arms_zero_camber_gain():
    """
    Ground-truth check against known physics (not internal consistency):
    a parallel, equal-length double wishbone translates the upright vertically
    with no rotation, so camber gain must be ~zero through travel. This is the
    textbook result and catches sign/geometry regressions in the solver.
    """
    import numpy as np
    hp = Hardpoints.default()
    hp.upper_front_inner = np.array([-110.0, 200.0, 290.0])
    hp.upper_rear_inner  = np.array([140.0, 200.0, 290.0])
    hp.lower_front_inner = np.array([-110.0, 200.0, 120.0])
    hp.lower_rear_inner  = np.array([140.0, 200.0, 120.0])
    hp.upper_outer = np.array([0.0, 575.0, 290.0])
    hp.lower_outer = np.array([0.0, 575.0, 120.0])
    hp.static_camber = 0.0
    hp.static_toe = 0.0
    kin = SuspensionKinematics(hp)
    cambers = [kin.solve_at_travel(t).camber for t in (-20, -10, 0, 10, 20)]
    assert max(abs(c) for c in cambers) < 0.05, cambers


def test_sweep_stays_on_branch():
    """Incremental seeding: a full sweep must converge at every point and vary
    smoothly (no jump to the mirror configuration at the extremes)."""
    kin = SuspensionKinematics(Hardpoints.default())
    states = kin.sweep(-30, 30, 25)
    assert all(s.converged for s in states)
    cambers = [s.camber for s in states]
    # monotonic-ish: no sudden sign flip / discontinuity between adjacent steps
    jumps = [abs(cambers[i+1] - cambers[i]) for i in range(len(cambers) - 1)]
    assert max(jumps) < 1.0, f"discontinuity in camber sweep: {max(jumps)}"


def test_validation_rejects_2d_point():
    import numpy as _np
    hp = Hardpoints.default()
    hp.upper_outer = _np.array([1.0, 2.0])
    try:
        SuspensionKinematics(hp)
        assert False, "should reject 2D point"
    except ValueError:
        pass


def test_validation_rejects_coincident_balljoints():
    import numpy as _np
    hp = Hardpoints.default()
    hp.upper_outer = hp.lower_outer.copy()
    try:
        SuspensionKinematics(hp)
        assert False, "should reject coincident ball joints"
    except ValueError:
        pass


def test_validation_rejects_nonfinite():
    import numpy as _np
    hp = Hardpoints.default()
    hp.wheel_center = _np.array([_np.nan, 0.0, 0.0])
    try:
        SuspensionKinematics(hp)
        assert False, "should reject non-finite"
    except ValueError:
        pass


def test_from_dict_ignores_unknown_keys():
    # Regression: a saved project / foreign or future version may carry keys the
    # current Hardpoints doesn't define. from_dict must ignore them rather than
    # crash with "Hardpoints.__init__() got an unexpected keyword argument ...".
    d = Hardpoints.default().as_dict()
    d["pushrod_outer"] = d["pushrod_outer"]   # the field from the screenshot bug
    d["some_future_field"] = 123              # genuinely unknown
    d["another_unknown"] = [1, 2, 3]
    hp = Hardpoints.from_dict(d)              # must not raise
    SuspensionKinematics(hp)                  # must still solve
    assert not hasattr(hp, "some_future_field")


def test_from_dict_legacy_without_rocker():
    # An old project saved before pushrod/rocker existed must still load and solve,
    # falling back to the direct-acting proxy motion ratio.
    d = {k: v for k, v in Hardpoints.default().as_dict().items()
         if not (k.startswith("rocker") or k in
                 ("pushrod_outer", "spring_inner", "pushrod_attach"))}
    hp = Hardpoints.from_dict(d)
    kin = SuspensionKinematics(hp)
    assert not hp.has_rocker()
    assert not kin.motion_ratio_is_real()


def test_from_dict_handles_none_and_empty():
    Hardpoints.from_dict(Hardpoints.default().as_dict())   # round trip
    # tolerate a None/empty dict without crashing the constructor path
    try:
        Hardpoints.from_dict({})
    except TypeError:
        pass   # missing required fields is an acceptable, clear failure mode


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
