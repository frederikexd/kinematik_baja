# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the v0.9 capability additions:
  * MotorMap — real torque/speed curve replacing the flat power cap
  * GPS/cone track import + racing-line optimisation
  * Combined-slip tyre (friction ellipse) and relaxation length
  * Damper force-velocity model and damping ratio

Each new model is honest about calibration: the physics is pinned here; the
calibration flags are checked so an uncalibrated model can never masquerade as
measured.

Run:  python -m pytest tests/test_capabilities_v09.py
"""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import (SuspensionKinematics, Hardpoints, VehicleDynamics,
                        VehicleParams, default_tire)
from suspension import laptime as L
from suspension import tiremodel as TM
from suspension import damper as D


def _veh():
    kin = SuspensionKinematics(Hardpoints.default())
    return VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin,
                           tire=default_tire())


# --------------------------------------------------------------------------- #
#  Motor map
# --------------------------------------------------------------------------- #
def test_motor_map_from_peak_shape():
    mm = L.MotorMap.from_peak(peak_torque_nm=230, peak_power_kw=100,
                              redline_rpm=6000, final_drive=3.5, wheel_radius_m=0.20)
    assert mm.source == "representative"
    # torque plateau at low speed, falling at high speed (constant power tail)
    f_lo = mm.wheel_force(3.0)
    f_hi = mm.wheel_force(28.0)
    assert f_lo > 0 and f_hi > 0
    assert f_lo >= f_hi            # force should not rise with speed past base
    # past the implied redline, no drive
    assert mm.wheel_force(200.0) == 0.0


def test_powertrain_uses_map_when_present():
    flat = L.Powertrain()
    mm = L.MotorMap.from_peak(230, 100, 6000)
    mapped = L.Powertrain(motor_map=mm)
    assert not flat.uses_real_motor_map()
    assert mapped.uses_real_motor_map()
    # the two give different tractive ceilings at a mid speed
    assert abs(flat.tractive_force(15.0) - mapped.tractive_force(15.0)) > 1.0


def test_motor_map_changes_accel_time():
    veh = _veh()
    t_flat = L.acceleration_time(veh, L.Powertrain()).lap_time_s
    mm = L.MotorMap.from_peak(230, 100, 6000)
    t_map = L.acceleration_time(veh, L.Powertrain(motor_map=mm)).lap_time_s
    assert math.isfinite(t_flat) and math.isfinite(t_map)
    assert t_map != t_flat


# --------------------------------------------------------------------------- #
#  GPS / cone track import
# --------------------------------------------------------------------------- #
def test_track_from_path_recovers_known_radius():
    # A circle of radius 20 m: every station should report ~20 m radius.
    th = np.linspace(0, 2 * np.pi, 400)
    x, y = 20 * np.cos(th), 20 * np.sin(th)
    trk = L.track_from_path(x, y, ds=1.0)
    radii = [s.radius_m for s in trk.segments if s.is_corner]
    assert len(radii) > 10
    assert abs(np.median(radii) - 20.0) < 2.0


def test_track_from_path_detects_straight():
    x = np.linspace(0, 100, 200)
    y = np.zeros_like(x)
    trk = L.track_from_path(x, y, ds=1.0, closed=False)
    assert all(not s.is_corner for s in trk.segments)


def test_track_from_path_safe_on_garbage():
    trk = L.track_from_path([0], [0])
    assert trk.segments == []


def test_cones_to_centerline_midpoint():
    x = np.linspace(0, 50, 60)
    y = np.zeros_like(x)
    cx, cy = L.cones_to_centerline(x, y - 2, x, y + 2)
    assert np.allclose(cy, 0.0, atol=1e-6)       # midline between ±2 is 0


def test_latlon_projection_scale():
    lat = np.array([52.0, 52.0 + 1.0 / 111.32e3 * 100])  # ~100 m north
    lon = np.array([4.0, 4.0])
    x, y = L.latlon_to_xy(lat, lon)
    assert abs((y[1] - y[0]) - 100.0) < 2.0


def test_imported_track_runs_in_sim():
    veh = _veh()
    th = np.linspace(0, 2 * np.pi, 300)
    x, y = 25 * np.cos(th), 25 * np.sin(th)
    trk = L.track_from_path(x, y, ds=1.0)
    res = L.simulate_lap(veh, trk, L.Powertrain())
    assert res.ok and res.lap_time_s > 0


# --------------------------------------------------------------------------- #
#  Racing-line optimisation
# --------------------------------------------------------------------------- #
def test_racing_line_stays_within_track():
    th = np.linspace(0, 2 * np.pi, 200)
    x, y = 30 * np.cos(th), 30 * np.sin(th)
    lx, ly, off = L.optimise_racing_line(x, y, track_width_m=3.0)
    assert np.max(np.abs(off)) <= 3.0 / 2 + 1e-6   # within ±half-width


def test_racing_line_not_slower_than_centerline():
    veh = _veh()
    # wavy path with room to straighten
    s = np.linspace(0, 4 * np.pi, 240)
    x = 40 * np.cos(s)
    y = 20 * np.sin(2 * s)
    cmp = L.compare_line_vs_centerline(veh, x, y, track_width_m=4.0)
    assert cmp["centerline_result"].ok and cmp["line_result"].ok
    # the curvature-optimal line should not be slower than the centreline
    assert cmp["time_gained_s"] >= -0.05


# --------------------------------------------------------------------------- #
#  Combined slip + relaxation length
# --------------------------------------------------------------------------- #
def test_combined_tire_uncalibrated_by_default():
    ct = TM.default_combined_tire()
    assert not ct.is_calibrated
    assert "UNCALIBRATED" in ct.status()


def test_friction_ellipse_tradeoff():
    ct = TM.default_combined_tire()
    Fz = 1100.0
    fy_no_fx = ct.available_fy(0.0, Fz)
    fy_some_fx = ct.available_fy(800.0, Fz)
    fy_max_fx = ct.available_fy(1e6, Fz)
    assert fy_no_fx > fy_some_fx > fy_max_fx       # using Fx eats into Fy
    assert abs(fy_max_fx) < 1e-6                    # all budget spent on Fx


def test_friction_ellipse_symmetric_default():
    ct = TM.default_combined_tire()
    assert ct.ell_kx == 2.0 and ct.ell_ky == 2.0   # circle/ellipse until calibrated


def test_relaxation_length_grows_with_load():
    s_lo = TM.relaxation_length(500, 1100)
    s_hi = TM.relaxation_length(2200, 1100)
    assert s_hi > s_lo > 0


def test_relaxation_lag_reaches_63pct_at_one_sigma():
    sigma = 0.45
    a = TM.apply_relaxation_lag(1.0, ds=sigma, sigma_m=sigma, alpha_prev=0.0)
    assert abs(a - (1 - math.exp(-1.0))) < 1e-9    # ~0.632


def test_relaxation_lag_converges():
    a = TM.apply_relaxation_lag(1.0, ds=100.0, sigma_m=0.45, alpha_prev=0.0)
    assert abs(a - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
#  Damper
# --------------------------------------------------------------------------- #
def test_damper_uncalibrated_by_default():
    dc = D.default_damper()
    assert not dc.is_calibrated and "UNCALIBRATED" in dc.status()


def test_damper_force_signs_and_monotonic():
    dc = D.default_damper()
    assert dc.force(0.1) > 0       # bump resists with +force
    assert dc.force(-0.1) < 0      # rebound with -force
    assert dc.force(0.3) > dc.force(0.1)   # more velocity, more force


def test_damper_digressive_knee():
    dc = D.default_damper()
    # slope below the knee (low-speed) steeper than above (high-speed)
    f_a = dc.force(0.02); f_b = dc.force(0.04)     # both below knee=0.05
    slope_low = (f_b - f_a) / 0.02
    f_c = dc.force(0.2); f_d = dc.force(0.4)       # both above knee
    slope_high = (f_d - f_c) / 0.2
    assert slope_low > slope_high


def test_damper_dyno_fit_sets_calibrated():
    truth = D.DamperCurve(c_bump_low=5000, c_bump_high=1500,
                          c_reb_low=7000, c_reb_high=2200, v_knee=0.05)
    v = np.concatenate([np.linspace(-0.4, -0.001, 40), np.linspace(0.001, 0.4, 40)])
    f = truth.force(v)
    fit = D.DamperCurve.from_dyno_points(v, f, v_knee=0.05)
    assert fit.is_calibrated and fit.source == "dyno"
    assert abs(fit.c_bump_low - 5000) < 200
    assert abs(fit.c_reb_low - 7000) < 200


def test_damping_ratio_ballpark():
    dc = D.default_damper()
    z = D.damping_ratio(dc, corner_mass_kg=70, wheel_rate_N_per_mm=25, motion_ratio=0.52)
    assert 0.4 < z < 1.0           # representative FSAE bump damping is sub-critical


def test_combined_tire_in_lapsim_runs():
    veh = _veh()
    ct = TM.default_combined_tire()
    res = L.acceleration_time(veh, L.Powertrain(combined_tire=ct))
    assert res.ok and res.lap_time_s > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
