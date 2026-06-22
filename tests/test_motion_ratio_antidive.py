# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the pushrod/rocker motion ratio and anti-dive / anti-squat features.

These pin the physics conventions that the setup optimiser now depends on:
  * a defined rocker yields a REAL motion ratio (not the ball-joint proxy),
  * wheel rate scales as k_spring * MR**2,
  * anti-dive/anti-squat come from the side-view swing-arm geometry, and
  * the optimiser drives roll stiffness through the motion ratio so spring
    rates are physically meaningful levers.

Run:  python -m pytest tests/test_motion_ratio_antidive.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import SuspensionKinematics, Hardpoints, VehicleDynamics, VehicleParams
from suspension import setup as setup_mod


# --------------------------------------------------------------------------- #
#  Motion ratio
# --------------------------------------------------------------------------- #
def test_default_has_real_rocker():
    kin = SuspensionKinematics(Hardpoints.default())
    assert kin.motion_ratio_is_real(), "default geometry should define a rocker"


def test_motion_ratio_in_physical_range():
    kin = SuspensionKinematics(Hardpoints.default())
    mr = kin.motion_ratio()
    assert np.isfinite(mr)
    # FSAE pushrod cars run roughly 0.4-0.8 spring/wheel.
    assert 0.3 < mr < 1.0, f"motion ratio {mr} out of typical FSAE range"


def test_wheel_rate_scales_with_mr_squared():
    kin = SuspensionKinematics(Hardpoints.default())
    mr = kin.motion_ratio()
    k_spring = 40.0
    wr = kin.wheel_rate(k_spring)
    assert abs(wr - k_spring * mr * mr) < 1e-6


def test_proxy_fallback_when_no_rocker():
    hp = Hardpoints.default()
    # Strip the rocker so the linkage is direct-acting.
    for k in ("pushrod_outer", "rocker_pivot", "rocker_axis",
              "rocker_pushrod", "rocker_spring", "spring_inner"):
        setattr(hp, k, None)
    kin = SuspensionKinematics(hp)
    assert not kin.motion_ratio_is_real()
    assert np.isfinite(kin.motion_ratio())  # proxy still returns a number


def test_motion_ratio_curve_returns_per_travel():
    kin = SuspensionKinematics(Hardpoints.default())
    travels, mr = kin.motion_ratio_curve(-20, 20, 9)
    assert len(travels) == len(mr) == 9
    assert all(np.isfinite(m) for m in mr)


def test_serialization_round_trip_preserves_rocker():
    import json
    hp = Hardpoints.default()
    hp2 = Hardpoints.from_dict(json.loads(json.dumps(hp.as_dict())))
    assert hp2.has_rocker()
    k1, k2 = SuspensionKinematics(hp), SuspensionKinematics(hp2)
    assert abs(k1.motion_ratio() - k2.motion_ratio()) < 1e-9


# --------------------------------------------------------------------------- #
#  Anti-dive / anti-squat
# --------------------------------------------------------------------------- #
def test_default_anti_dive_is_modest_and_positive():
    kin = SuspensionKinematics(Hardpoints.default())
    ad = kin.anti_dive_pct(cg_height=300.0, wheelbase=1550.0, brake_bias_front=0.65)
    assert np.isfinite(ad)
    assert 0 < ad < 60, f"default anti-dive {ad}% should be a modest positive value"


def test_flat_pickups_give_zero_anti_dive():
    # No front/rear pickup stagger => side-view arms parallel => zero anti-dive.
    hp = Hardpoints.default()
    for k in ("upper_front_inner", "upper_rear_inner"):
        getattr(hp, k)[2] = 290.0
    for k in ("lower_front_inner", "lower_rear_inner"):
        getattr(hp, k)[2] = 120.0
    kin = SuspensionKinematics(hp)
    assert abs(kin.anti_dive_pct(300.0, 1550.0)) < 1e-6


def test_anti_dive_scales_with_brake_bias():
    kin = SuspensionKinematics(Hardpoints.default())
    a_half = kin.anti_dive_pct(300.0, 1550.0, brake_bias_front=0.5)
    a_full = kin.anti_dive_pct(300.0, 1550.0, brake_bias_front=1.0)
    assert abs(a_full - 2 * a_half) < 1e-6


def test_anti_squat_uses_rear_geometry_via_vehicle():
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin)
    assert np.isfinite(veh.anti_squat_pct())
    assert np.isfinite(veh.anti_dive_pct())


# --------------------------------------------------------------------------- #
#  Spring rate -> roll stiffness through the motion ratio
# --------------------------------------------------------------------------- #
def test_roll_stiffness_from_spring_rate():
    kin = SuspensionKinematics(Hardpoints.default())
    p = VehicleParams(use_spring_rates=True, spring_rate_front=40.0,
                      spring_rate_rear=40.0, arb_rate_front=0.0, arb_rate_rear=0.0)
    veh = VehicleDynamics(p, front_kin=kin, rear_kin=kin)
    _, info = veh.lateral_load_transfer(1.0)
    mr = kin.motion_ratio()
    k_wheel = 40.0 * mr * mr
    expected = k_wheel * (p.track_front ** 2) / 2.0 / 1000.0 * (np.pi / 180.0)
    assert abs(info["roll_stiffness_front"] - expected) < 1e-3


def test_arb_adds_to_roll_stiffness():
    kin = SuspensionKinematics(Hardpoints.default())
    base = VehicleParams(use_spring_rates=True, spring_rate_front=40.0)
    withbar = VehicleParams(use_spring_rates=True, spring_rate_front=40.0,
                            arb_rate_front=100.0)
    vb = VehicleDynamics(base, front_kin=kin, rear_kin=kin)
    vw = VehicleDynamics(withbar, front_kin=kin, rear_kin=kin)
    _, ib = vb.lateral_load_transfer(1.0)
    _, iw = vw.lateral_load_transfer(1.0)
    assert abs((iw["roll_stiffness_front"] - ib["roll_stiffness_front"]) - 100.0) < 1e-6


def test_higher_spring_stiffens_that_end_balance():
    # Stiffening the front spring should move the balance toward the front limit
    # (more understeer), i.e. balance index increases.
    kin = SuspensionKinematics(Hardpoints.default())
    from suspension import tiremodel
    tire = tiremodel.default_tire()
    soft = VehicleParams(use_spring_rates=True, spring_rate_front=25.0, spring_rate_rear=40.0)
    stiff = VehicleParams(use_spring_rates=True, spring_rate_front=80.0, spring_rate_rear=40.0)
    vs = VehicleDynamics(soft, front_kin=kin, rear_kin=kin, tire=tire)
    vt = VehicleDynamics(stiff, front_kin=kin, rear_kin=kin, tire=tire)
    bal_soft, _, _ = vs.balance_index(1.0)
    bal_stiff, _, _ = vt.balance_index(1.0)
    assert bal_stiff > bal_soft, "stiffer front should push balance toward understeer"


def test_optimiser_uses_spring_rate_knobs():
    kin = SuspensionKinematics(Hardpoints.default())
    from suspension import tiremodel
    tire = tiremodel.default_tire()
    sens = setup_mod.sensitivity(VehicleParams(), front_kin=kin, rear_kin=kin, tire=tire)
    knobs = {r["knob"] for r in sens["rankings"]}
    assert "spring_rate_front" in knobs and "spring_rate_rear" in knobs
    assert "roll_stiffness_front" not in knobs  # replaced by the physical lever


def test_optimiser_runs_and_holds_balance():
    kin = SuspensionKinematics(Hardpoints.default())
    from suspension import tiremodel
    tire = tiremodel.default_tire()
    opt = setup_mod.optimise(VehicleParams(), front_kin=kin, rear_kin=kin,
                             tire=tire, n_grid=4, passes=1)
    assert opt["best_eval"]["max_g"] > 0
    assert opt["delta_maxg"] >= -1e-9  # never worse than the start


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
