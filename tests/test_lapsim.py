# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the quasi-steady-state lap-time simulator.

These pin the behaviour the lap-time upgrade depends on — the thing that turns a
geometry/setup change into seconds, which is the number that wins FSAE events:

  - the sim produces physically sane event times for a representative car,
  - more grip (better tire) => faster lap (the comparator is monotone the right way),
  - more downforce raises corner speed but adds drag (the aero trade is real),
  - the robustness contract HOLDS: a NaN/raising tire, absurd params, and
    degenerate tracks never raise — they return a safe result with a warning,
  - simulate_events never raises and always returns all three events,
  - the FSAE points model rewards a faster time.

It imports the engine modules directly (not via the package __init__) so it can
run without the heavy CAD dependencies (trimesh/rtree) that the chassis tools pull
in — the lap sim itself needs only numpy.

Run:  python tests/test_lapsim.py   (or: python -m pytest tests/)
"""

import os
import sys
import math
import importlib.util
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_engine():
    """Load kinematics/tiremodel/dynamics/lapsim without triggering the package
    __init__ (which imports trimesh). Returns a namespace dict."""
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(_ROOT, "suspension")]
    sys.modules.setdefault("suspension", pkg)
    mods = {}
    for m in ["kinematics", "tiremodel", "dynamics", "lapsim"]:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"suspension.{m}"] = mod
        mods[m] = mod
    for m in ["kinematics", "tiremodel", "dynamics", "lapsim"]:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        spec.loader.exec_module(mods[m])
    return mods


_M = _load_engine()
K, T, D, L = _M["kinematics"], _M["tiremodel"], _M["dynamics"], _M["lapsim"]


_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _veh(tire=None):
    kin = K.SuspensionKinematics(K.Hardpoints.default())
    return D.VehicleDynamics(D.VehicleParams(), front_kin=kin, rear_kin=kin,
                             tire=tire if tire is not None else T.default_tire())


# ----------------------- sane event times ------------------------------- #
def test_event_times_physical():
    res = L.simulate_events(_veh(), L.LapSimParams())
    sk, ac, ax = res["skidpad"], res["acceleration"], res["autocross"]
    check("all three events ran ok", sk.ok and ac.ok and ax.ok)
    # FSAE skidpad timed circle is typically ~4.5-6 s
    check("skidpad time in 3.5-7 s", 3.5 < sk.lap_time < 7.0)
    # 75 m acceleration is typically ~3.5-5 s for an FSAE car
    check("acceleration time in 3.0-6 s", 3.0 < ac.lap_time < 6.0)
    # ~800 m autocross lap order-of-magnitude
    check("autocross lap in 20-60 s", 20.0 < ax.lap_time < 60.0)
    # top speeds plausible (< ~130 km/h)
    check("accel top speed < 130 km/h", ac.top_speed * 3.6 < 130.0)


# ----------------------- monotone in grip ------------------------------- #
def test_more_grip_is_faster():
    # scale the tire's peak factor up => more grip => should not be slower
    base = T.default_tire()
    strong_coeffs = dict(base.coeffs)
    strong_coeffs["PDY1"] = base.coeffs["PDY1"] * 1.15   # +15% peak friction
    strong = T.PacejkaLateral(coeffs=strong_coeffs, FNOMIN=base.FNOMIN)

    t_base = L.LapSimulator(_veh(base)).simulate(L.autocross_track()).lap_time
    t_strong = L.LapSimulator(_veh(strong)).simulate(L.autocross_track()).lap_time
    check("more grip => not slower autocross lap", t_strong <= t_base + 1e-6)
    check("more grip => meaningfully faster", t_strong < t_base)


# ----------------------- aero trade is real ----------------------------- #
def test_downforce_helps_corners():
    veh = _veh()
    no_aero = L.LapSimParams(cl_a=0.0, cd_a=0.0)
    with_aero = L.LapSimParams(cl_a=3.0, cd_a=1.2)
    t_none = L.LapSimulator(veh, no_aero).simulate(L.autocross_track()).lap_time
    t_aero = L.LapSimulator(veh, with_aero).simulate(L.autocross_track()).lap_time
    # on a corner-dominated autocross lap, downforce should help overall
    check("downforce improves autocross lap", t_aero < t_none)
    # ... but on the 75 m drag-dominated accel run it should not help (drag cost)
    a_none = L.LapSimulator(veh, no_aero).simulate(L.acceleration_track()).lap_time
    a_aero = L.LapSimulator(veh, with_aero).simulate(L.acceleration_track()).lap_time
    check("downforce does not speed up 75 m accel", a_aero >= a_none - 1e-6)


# ----------------------- corner speed physics --------------------------- #
def test_corner_speed_scales_with_radius():
    sim = L.LapSimulator(_veh(), L.LapSimParams(cl_a=0.0, cd_a=0.0))
    v_tight = sim._corner_speed(5.0)
    v_open = sim._corner_speed(30.0)
    check("bigger radius => higher corner speed", v_open > v_tight)
    # v ~ sqrt(mu g R): ratio should be roughly sqrt(30/5) ~ 2.45 (no aero)
    ratio = v_open / max(v_tight, 1e-6)
    check("corner-speed radius scaling ~sqrt(R)", 2.0 < ratio < 3.0)


# ----------------------- ROBUSTNESS: never raises ----------------------- #
def test_nan_tire_safe():
    class NanTire:
        FNOMIN = 1100.0
        def peak_force(self, Fz, g=0.0):
            return float("nan")
    r = L.LapSimulator(_veh(NanTire())).simulate(L.autocross_track())
    check("NaN tire still returns a finite lap", r.ok and np.isfinite(r.lap_time))
    check("NaN tire surfaces a warning", len(r.warnings) >= 1)


def test_raising_tire_safe():
    class BoomTire:
        FNOMIN = 1100.0
        def peak_force(self, Fz, g=0.0):
            raise RuntimeError("boom")
    r = L.LapSimulator(_veh(BoomTire())).simulate(L.autocross_track())
    check("raising tire still returns a finite lap", r.ok and np.isfinite(r.lap_time))
    check("raising tire surfaces a warning", len(r.warnings) >= 1)


def test_degenerate_track_safe():
    deg = L.Track("degenerate",
                  [L.Segment(0.0, None), L.Segment(-5.0, 3.0)], closed=True)
    r = L.LapSimulator(_veh()).simulate(deg)
    check("degenerate track returns failed-but-safe result", (not r.ok))
    check("degenerate track is non-raising with warning", len(r.warnings) >= 1)
    # the failed result must still be a usable object with arrays
    check("failed result has array fields", hasattr(r, "speed") and len(r.speed) >= 1)


def test_empty_track_safe():
    r = L.LapSimulator(_veh()).simulate(L.Track("empty", [], closed=True))
    check("empty track safe", (not r.ok) and len(r.warnings) >= 1)


def test_absurd_params_flagged():
    bad = L.LapSimParams(power_w=0.0, brake_g=-1.0, cl_a=-5.0, cd_a=-2.0)
    r = L.LapSimulator(_veh(), bad).simulate(L.autocross_track())
    # must not raise; should either flag implausible pace or still be finite
    check("absurd params do not raise", isinstance(r, L.LapResult))
    check("absurd params flagged or finite",
          (len(r.warnings) >= 1) or np.isfinite(r.lap_time))


def test_simulate_events_never_raises():
    class BoomTire:
        FNOMIN = 1100.0
        def peak_force(self, Fz, g=0.0):
            raise RuntimeError("boom")
    res = L.simulate_events(_veh(BoomTire()))
    check("simulate_events returns all three", set(res) == {"skidpad", "acceleration", "autocross"})
    check("simulate_events results are LapResult",
          all(isinstance(v, L.LapResult) for v in res.values()))


def test_gg_v_envelope_safe():
    gg = L.LapSimulator(_veh()).gg_v_envelope()
    check("gg-V has all curves", all(k in gg for k in ("speed", "lat_g", "accel_g", "brake_g")))
    check("gg-V lateral g positive", float(np.max(gg["lat_g"])) > 0.5)


# ----------------------- points model ----------------------------------- #
def test_points_reward_speed():
    # faster time should score >= slower time for the same event references
    fast = L.event_points("autocross", 28.0, best_time=28.0, max_time=40.0)
    slow = L.event_points("autocross", 38.0, best_time=28.0, max_time=40.0)
    check("faster time scores higher", fast > slow)
    check("best time gets near-max points", fast > 120.0)
    check("bad input scores 0 safely", L.event_points("autocross", float("nan")) == 0.0)


def main():
    print("\n=== lap-time simulator ===")
    for fn in [
        test_event_times_physical,
        test_more_grip_is_faster,
        test_downforce_helps_corners,
        test_corner_speed_scales_with_radius,
        test_nan_tire_safe,
        test_raising_tire_safe,
        test_degenerate_track_safe,
        test_empty_track_safe,
        test_absurd_params_flagged,
        test_simulate_events_never_raises,
        test_gg_v_envelope_safe,
        test_points_reward_speed,
    ]:
        fn()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILURES:", _FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
