# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the lap-time simulator.

Two things matter here:
  1. the physics is sane (skidpad time in the real FSAE band; faster grip => faster
     lap; aero/power move time the right direction), and
  2. the never-crash contract holds — bad inputs and an exploding dynamics object
     must return a flagged safe default, never raise.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import (SuspensionKinematics, Hardpoints,
                        VehicleDynamics, VehicleParams)
from suspension import tiremodel as tm
from suspension import laptime as lt


def _veh(tire=True, **vp):
    kin = SuspensionKinematics(Hardpoints.default())
    t = tm.default_tire() if tire else None
    return VehicleDynamics(VehicleParams(**vp), front_kin=kin, rear_kin=kin, tire=t)


# --------------------------- physics sanity ------------------------------- #
def test_skidpad_in_realistic_band():
    r = lt.skidpad_time(_veh())
    assert r.ok
    # FSAE skidpad timed circles are roughly 4.5–5.5 s
    assert 4.0 < r.lap_time_s < 6.0


def test_skidpad_closed_form_consistency():
    veh = _veh()
    r = lt.skidpad_time(veh)
    # v should satisfy v^2/R ~ a_lat (within aero adjustment); recompute roughly
    v = r.avg_speed_ms
    assert v > 0
    # t = circumference / v
    assert math.isclose(r.lap_time_s, (2 * math.pi * lt.SKIDPAD_RADIUS_M) / v, rel_tol=1e-3)


def test_more_grip_is_faster_skidpad():
    # a higher-mu tire should lower skidpad time
    base = lt.skidpad_time(_veh())
    grippier = tm.default_tire()
    grippier.coeffs["PDY1"] = 1.85   # raise peak friction
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin, tire=grippier)
    better = lt.skidpad_time(veh)
    assert better.lap_time_s < base.lap_time_s


def test_autocross_runs_and_is_ordered():
    veh = _veh()
    tr = lt.default_autocross()
    lap = lt.simulate_lap(veh, tr)
    assert lap.ok
    assert lap.lap_time_s > 0 and math.isfinite(lap.lap_time_s)
    assert lap.min_speed_ms <= lap.avg_speed_ms <= lap.top_speed_ms
    assert math.isclose(lap.distance_m, tr.total_length(), rel_tol=0.05)


def test_more_power_lowers_lap_time():
    veh = _veh()
    tr = lt.default_autocross()
    slow = lt.simulate_lap(veh, tr, lt.Powertrain(power_kw=40))
    fast = lt.simulate_lap(veh, tr, lt.Powertrain(power_kw=120))
    assert fast.lap_time_s < slow.lap_time_s


def test_corner_limit_speed_scales_with_radius():
    veh = _veh()
    pt = lt.Powertrain(cla=0.0)  # no aero so it's a clean sqrt(g*R)
    g = lt._max_lat_g(veh)
    v_small = lt._corner_limit_speed(veh, 5.0, pt, g)
    v_big = lt._corner_limit_speed(veh, 20.0, pt, g)
    assert v_big > v_small
    # closed form check with no aero: v = sqrt(a_lat * R)
    assert math.isclose(v_small, math.sqrt(g * 9.81 * 5.0), rel_tol=1e-6)


def test_points_estimate_monotonic():
    best = 25.0
    assert lt.event_points_estimate(best, best) > lt.event_points_estimate(best * 1.1, best)
    assert lt.event_points_estimate(best * 2.0, best) >= 0.0


# --------------------------- never-crash contract -------------------------- #
def test_empty_track_returns_safe_default():
    r = lt.simulate_lap(_veh(), lt.Track("empty", []))
    assert r.ok is False
    assert r.warning
    assert math.isnan(r.lap_time_s)  # flagged, not a bogus number


def test_bad_radius_skidpad_safe():
    r = lt.skidpad_time(_veh(), radius_m=-3.0)
    assert r.ok is False and r.warning


def test_exploding_dynamics_does_not_raise():
    class Boom:
        p = VehicleParams()
        def max_lateral_g(self):
            raise RuntimeError("solver blew up")
    # both entry points must swallow it and flag a fallback
    sp = lt.skidpad_time(Boom())
    assert sp.ok and "default" in sp.warning.lower()
    lap = lt.simulate_lap(Boom(), lt.default_autocross())
    assert lap.ok and lap.lap_time_s > 0


def test_nan_grip_falls_back():
    class Nanny:
        p = VehicleParams()
        def max_lateral_g(self):
            return float("nan")
    r = lt.skidpad_time(Nanny())
    assert r.ok and math.isfinite(r.lap_time_s)


def test_degenerate_segment_lengths_safe():
    tr = lt.Track("weird", [lt.Segment(0.0), lt.Segment(-5.0, 3.0),
                            lt.Segment(10.0, 4.0)])
    r = lt.simulate_lap(_veh(), tr)
    # should still produce a finite time, not crash
    assert r.ok and math.isfinite(r.lap_time_s)


def test_traces_are_finite():
    lap = lt.simulate_lap(_veh(), lt.default_autocross())
    assert all(math.isfinite(x) for x in lap.v)
    assert all(math.isfinite(x) for x in lap.s)
