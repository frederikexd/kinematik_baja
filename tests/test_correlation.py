# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the correlation / validation module.

These pin the behaviour that makes the feature trustworthy:
  * a perfect prediction reports ~0 error and "within tolerance",
  * a biased prediction is flagged OFF with the correct fast/slow diagnostic,
  * the skidpad g<->time conversions are mutual inverses,
  * R² is suppressed (NaN) on a constant-speed trace rather than going negative,
  * bad/mismatched data degrades to a flagged "could not correlate", never raises.

Run:  python -m pytest tests/test_correlation.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import (SuspensionKinematics, Hardpoints, VehicleDynamics,
                        VehicleParams, default_tire)
from suspension import correlation as C


def _veh():
    kin = SuspensionKinematics(Hardpoints.default())
    return VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin,
                           tire=default_tire())


# --------------------------------------------------------------------------- #
#  Skidpad g <-> time conversions
# --------------------------------------------------------------------------- #
def test_skidpad_g_time_roundtrip():
    g = 1.4
    t = C.skidpad_time_from_lateral_g(g)
    g2 = C.lateral_g_from_skidpad_time(t)
    assert abs(g - g2) < 1e-9


def test_skidpad_conversions_reject_nonphysical():
    assert not np.isfinite(C.skidpad_time_from_lateral_g(-1.0))
    assert not np.isfinite(C.lateral_g_from_skidpad_time(0.0))


# --------------------------------------------------------------------------- #
#  Skidpad correlation
# --------------------------------------------------------------------------- #
def test_skidpad_perfect_match_is_within_tol():
    veh = _veh()
    pred_g = veh.max_lateral_g()
    rep = C.correlate_skidpad(veh, measured_g=pred_g)
    assert rep.ok and rep.overall_within_tol
    g_ch = rep.channels[0]
    assert abs(g_ch.error_pct) < 1e-6


def test_skidpad_large_mismatch_is_flagged():
    veh = _veh()
    pred_g = veh.max_lateral_g()
    rep = C.correlate_skidpad(veh, measured_g=pred_g * 0.7)  # 30% off
    assert rep.ok and not rep.overall_within_tol


def test_skidpad_accepts_time_or_g():
    veh = _veh()
    pred_g = veh.max_lateral_g()
    t = C.skidpad_time_from_lateral_g(pred_g)
    rep = C.correlate_skidpad(veh, measured_time_s=t)
    assert rep.overall_within_tol  # same underlying number, just supplied as time


def test_skidpad_requires_some_measurement():
    rep = C.correlate_skidpad(_veh())
    assert not rep.ok


# --------------------------------------------------------------------------- #
#  Acceleration correlation
# --------------------------------------------------------------------------- #
def test_acceleration_match_and_miss():
    good = C.correlate_acceleration(measured_time_s=4.00, predicted_time_s=4.05)
    bad = C.correlate_acceleration(measured_time_s=4.00, predicted_time_s=4.60)
    assert good.overall_within_tol
    assert not bad.overall_within_tol
    assert bad.channels[0].error > 0  # predicted slower than measured


# --------------------------------------------------------------------------- #
#  Speed-trace correlation
# --------------------------------------------------------------------------- #
def test_trace_perfect_match():
    d = np.linspace(0, 75, 76)
    v = np.sqrt(2 * 8 * d) + 1.0
    rep = C.correlate_speed_trace(d, v, predicted_distance=d, predicted_speed=v.copy())
    assert rep.trace.ok and rep.trace.within_tol
    assert rep.trace.rmse < 1e-6
    assert rep.trace.r2 > 0.999


def test_trace_systematically_fast_is_diagnosed():
    d = np.linspace(0, 75, 76)
    true_v = np.sqrt(2 * 8 * d) + 1.0
    sim_v = true_v * 1.12
    rep = C.correlate_speed_trace(d, true_v, predicted_distance=d, predicted_speed=sim_v)
    assert not rep.trace.within_tol
    assert rep.trace.bias > 0
    assert "FAST" in rep.trace.note


def test_trace_systematically_slow_is_diagnosed():
    d = np.linspace(0, 75, 76)
    true_v = np.sqrt(2 * 8 * d) + 1.0
    sim_v = true_v * 0.88
    rep = C.correlate_speed_trace(d, true_v, predicted_distance=d, predicted_speed=sim_v)
    assert rep.trace.bias < 0
    assert "SLOW" in rep.trace.note


def test_trace_r2_suppressed_on_constant_speed():
    # Constant speed + tiny noise: R² must be NaN, not a big negative number.
    d = np.linspace(0, 100, 101)
    v = np.full_like(d, 12.5)
    rng = np.random.default_rng(0)
    meas = v + rng.normal(0, 0.1, d.shape)
    rep = C.correlate_speed_trace(d, meas, predicted_distance=d, predicted_speed=v)
    assert rep.trace.ok
    assert not np.isfinite(rep.trace.r2)


def test_trace_kmh_conversion():
    d = np.linspace(0, 75, 76)
    v_ms = np.sqrt(2 * 8 * d) + 1.0
    rep = C.correlate_speed_trace(d, v_ms * 3.6, predicted_distance=d,
                                  predicted_speed=v_ms, measured_speed_kmh=True)
    assert rep.trace.within_tol and rep.trace.rmse < 1e-6


def test_trace_resamples_mismatched_axes():
    # Prediction on a coarser grid than the measurement; interp should align them.
    dm = np.linspace(0, 75, 151)
    vm = np.sqrt(2 * 8 * dm) + 1.0
    dp = np.linspace(0, 75, 40)
    vp = np.sqrt(2 * 8 * dp) + 1.0
    rep = C.correlate_speed_trace(dm, vm, predicted_distance=dp, predicted_speed=vp)
    assert rep.trace.ok and rep.trace.within_tol


def test_trace_bad_data_is_safe_data_error():
    rep = C.correlate_speed_trace([1.0], [2.0], predicted_distance=[1, 2],
                                  predicted_speed=[3, 4])
    assert not rep.ok
    assert "could not correlate" in rep.summary


def test_trace_non_overlapping_ranges():
    rep = C.correlate_speed_trace([0, 1, 2], [10, 11, 12],
                                  predicted_distance=[100, 101, 102],
                                  predicted_speed=[10, 11, 12])
    assert not rep.trace.ok


def test_extract_trace_from_lap_result_like_objects():
    class A:  # lapsim.py convention
        distance = np.linspace(0, 10, 11)
        speed = np.linspace(5, 15, 11)
    class B:  # laptime.py convention
        s = np.linspace(0, 10, 11)
        v = np.linspace(5, 15, 11)
    for obj in (A(), B()):
        d, v = C._extract_trace(obj)
        assert d is not None and v is not None and len(d) == 11


def test_report_as_dict_is_serializable():
    import json
    veh = _veh()
    rep = C.correlate_skidpad(veh, measured_g=1.45)
    js = json.dumps(rep.as_dict())  # must not raise
    assert "summary" in json.loads(js)


def test_nothing_raises_on_garbage():
    # The robustness contract: bad input returns a flagged report, never raises.
    C.correlate_speed_trace(None, None, predicted_distance=None, predicted_speed=None)
    C.correlate_skidpad(_veh(), measured_g=float("nan"))
    C.correlate_acceleration(float("nan"), float("nan"))


# --------------------------------------------------------------------------- #
#  Standing-start acceleration helper (laptime.acceleration_time)
# --------------------------------------------------------------------------- #
def test_acceleration_time_is_physical():
    from suspension import laptime as lap
    veh = _veh()
    res = lap.acceleration_time(veh, lap.Powertrain(), distance_m=75.0)
    assert res.ok
    # A standing-start 75 m FSAE run is roughly 3.5-5.5 s; never zero (the bug
    # that closed-loop simulate_lap had on a bare straight).
    assert 3.0 < res.lap_time_s < 6.0
    assert res.top_speed_ms > 10.0
    assert len(res.s) == len(res.v) >= 2


def test_acceleration_time_safe_on_bad_distance():
    from suspension import laptime as lap
    res = lap.acceleration_time(_veh(), lap.Powertrain(), distance_m=-5.0)
    assert not res.ok and "invalid" in res.warning


def test_acceleration_time_feeds_correlation():
    from suspension import laptime as lap
    veh = _veh()
    res = lap.acceleration_time(veh, lap.Powertrain(), distance_m=75.0)
    rep = C.correlate_acceleration(measured_time_s=res.lap_time_s,
                                   predicted_time_s=res.lap_time_s)
    assert rep.overall_within_tol  # predicting itself is a perfect match


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
