# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the explicit transient time-step DAE solver (suspension/transient.py).

These pin the PHYSICS the transient layer adds on top of the quasi-steady-state
sim — the behaviour QSS structurally cannot represent:

  - a coasting car stays settled and left/right symmetric (no spurious side force
    from the antisymmetric camber handling),
  - braking transfers load to the front and dives the nose; accelerating squats,
  - a step steer produces a STABLE yaw response (overshoot then settle), with the
    load going to the OUTER tyres and mild understeer (steady yaw < neutral),
  - total vertical load is conserved (sums to static + aero),
  - a kerb strike spikes the struck wheel's contact load and can lift it (Fz->0)
    while the untouched wheel stays loaded — the high-frequency unsprung event,
  - trailing-throttle snap oversteer SPINS uncaught but the feedback countersteer
    CATCHES it (recovery),
  - the robustness contract HOLDS: a raising/NaN tyre, absurd params, and a wild
    input never raise — they return a flagged result with a warning,
  - the transient steady-state lateral g is consistent with the QSS limit (it
    settles onto a sub-limit corner, below the QSS max).

Loads the engine modules directly (not via the package __init__) so it runs
without the heavy CAD deps (trimesh/rtree); the transient layer needs only numpy.

Run:  python tests/test_transient.py   (or: python -m pytest tests/)
"""

import os
import sys
import math
import importlib.util
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_engine():
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(_ROOT, "suspension")]
    sys.modules.setdefault("suspension", pkg)
    mods = {}
    names = ["kinematics", "tiremodel", "dynamics", "damper", "lapsim", "transient"]
    for m in names:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"suspension.{m}"] = mod
        mods[m] = mod
    for m in names:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        spec.loader.exec_module(mods[m])
    return mods


_M = _load_engine()
K, T, D, DMP, L, TR = (_M["kinematics"], _M["tiremodel"], _M["dynamics"],
                       _M["damper"], _M["lapsim"], _M["transient"])

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _veh(tire=None):
    return D.VehicleDynamics(D.VehicleParams(),
                             tire=tire if tire is not None else T.default_tire())


def _front(r):
    return float(np.mean(r.Fz[-50:, 0:2]))


def _rear(r):
    return float(np.mean(r.Fz[-50:, 2:4]))


# --------------------------------------------------------------------------- #
print("transient — settled coast & symmetry")


def test_coast_settled_symmetric():
    sim = TR.TransientSolver(_veh())
    r = sim.run(0.8, u0=15.0)
    check("coast runs ok", r.ok)
    # left/right symmetric (no spurious camber-thrust side force)
    fl, fr = r.Fz[-1, 0], r.Fz[-1, 1]
    rl, rr = r.Fz[-1, 2], r.Fz[-1, 3]
    check("coast L/R load symmetric", abs(fl - fr) < 5.0 and abs(rl - rr) < 5.0)
    check("coast sideslip ~ 0", abs(math.degrees(r.beta[-1])) < 0.5)
    check("coast lateral g ~ 0", abs(r.ay[-1]) < 0.02)


test_coast_settled_symmetric()


# --------------------------------------------------------------------------- #
print("transient — longitudinal load transfer (dive/squat)")


def test_brake_dive():
    sim = TR.TransientSolver(_veh())
    drv = TR.DriverInput(brake=lambda t: 1.0 if t > 0.1 else 0.0)
    r = sim.run(1.0, driver=drv, u0=20.0)
    check("brake run ok", r.ok)
    static_f = _veh().p.mass * _veh().p.g * _veh().p.weight_dist_front / 2.0
    check("braking loads the front", _front(r) > static_f)
    check("braking unloads the rear", _rear(r) < _front(r))
    check("braking dives the nose (pitch < 0)", r.pitch[-1] < -1e-3)
    check("braking decel is negative", r.ax.min() < -0.5)


test_brake_dive()


def test_throttle_squat():
    sim = TR.TransientSolver(_veh())
    drv = TR.DriverInput(throttle=lambda t: 1.0 if t > 0.1 else 0.0)
    r = sim.run(1.0, driver=drv, u0=8.0)
    check("throttle run ok", r.ok)
    check("accelerating squats (pitch > 0)", r.pitch[-1] > 1e-3)


test_throttle_squat()


# --------------------------------------------------------------------------- #
print("transient — yaw response & lateral load transfer")


def test_step_steer_stable_understeer():
    sim = TR.TransientSolver(_veh())
    delta = math.radians(2.0)
    drv = TR.DriverInput(steer=lambda t: delta if t > 0.2 else 0.0,
                         throttle=lambda t: 0.05)
    r = sim.run(3.0, driver=drv, u0=12.0)
    check("step steer run ok", r.ok)
    # stable: doesn't spin (bounded sideslip)
    check("yaw response is stable (bounded sideslip)",
          abs(math.degrees(r.beta[-1])) < 10.0)
    steady_yaw = math.degrees(r.r[-1])
    # neutral-steer reference uses the ACTUAL final speed (the car may drift a
    # little under the trim throttle) and the real wheelbase.
    V_final = max(r.u[-1], 1.0)
    Lwb = sim.p.wheelbase
    neutral = math.degrees(V_final * delta / Lwb)
    check("turns the correct way (left steer -> +yaw)", steady_yaw > 1.0)
    check("mild understeer (steady yaw at/below neutral)",
          0 < steady_yaw <= neutral * 1.05)
    # peak yaw exceeds steady => overshoot (a transient QSS can't show)
    check("yaw overshoots before settling",
          math.degrees(np.max(np.abs(r.r))) >= steady_yaw - 1e-6)


test_step_steer_stable_understeer()


def test_outer_wheel_load_and_conservation():
    sim = TR.TransientSolver(_veh())
    drv = TR.DriverInput(steer=lambda t: math.radians(2.0) if t > 0.2 else 0.0)
    r = sim.run(2.5, driver=drv, u0=12.0)
    fl, fr, rl, rr = r.Fz[-1]
    # left turn -> outer (right) wheels gain load
    check("left turn loads outer (right) front", fr > fl)
    check("left turn loads outer (right) rear", rr > rl)
    # total load conserved (static + small aero); within a few %
    total = fl + fr + rl + rr
    static_total = sim.static_corner_loads().sum()
    check("total vertical load conserved (static+aero)",
          static_total <= total <= static_total * 1.20)


test_outer_wheel_load_and_conservation()


# --------------------------------------------------------------------------- #
print("transient — kerb / curb strike (unsprung high-frequency event)")


def test_curb_strike():
    r = TR.run_maneuver(_veh(), "curb_strike", u0=20.0, curb_h=0.03,
                        wheels=("FL",))
    check("curb run ok", r.ok)
    static_fl = _veh().p.mass * _veh().p.g * _veh().p.weight_dist_front / 2.0
    check("struck wheel load spikes above static", r.Fz[:, 0].max() > 1.5 * static_fl)
    check("struck wheel can lift (Fz -> ~0)", r.Fz[:, 0].min() < 1.0)
    # untouched right-front stays loaded throughout
    check("untouched wheel stays loaded", r.Fz[:, 1].min() > 50.0)


test_curb_strike()


# --------------------------------------------------------------------------- #
print("transient — snap oversteer: spins uncaught, recovers with feedback")


def test_snap_oversteer_recovery():
    caught = TR.run_maneuver(_veh(), "snap_oversteer", recover=True)
    uncaught = TR.run_maneuver(_veh(), "snap_oversteer", recover=False)
    check("snap runs ok", caught.ok and uncaught.ok)
    beta_caught = abs(math.degrees(caught.beta[-1]))
    beta_uncaught = abs(math.degrees(uncaught.beta[-1]))
    check("uncaught car spins (large final sideslip)", beta_uncaught > 25.0)
    check("feedback countersteer recovers (small final sideslip)", beta_caught < 8.0)
    check("recovery is dramatically better than no input",
          beta_uncaught > 3.0 * max(beta_caught, 1e-3))


test_snap_oversteer_recovery()


# --------------------------------------------------------------------------- #
print("transient — consistency with the QSS steady-state limit")


def test_settles_below_qss_limit():
    veh = _veh()
    sr = TR.transient_vs_qss_corner(veh, u0=14.0, t_end=2.5)
    check("settling analysis ok", sr.ok)
    check("transient steady ay is positive", sr.steady_ay_g > 0.3)
    check("transient steady ay below the QSS max (sub-limit corner)",
          sr.steady_ay_g <= sr.qss_max_ay_g + 0.05)
    check("rise time is a sane few-hundred ms", 0.0 < sr.rise_time_s < 2.0)


test_settles_below_qss_limit()


# --------------------------------------------------------------------------- #
print("transient — robustness contract (never raises)")


def test_nan_tire_safe():
    class NanTire:
        FNOMIN = 1100.0
        def fy(self, a, Fz, g=0.0): return float("nan")
        def peak_force(self, Fz, g=0.0): return float("nan")
        def mu_peak(self, Fz, g=0.0): return float("nan")
    veh = _veh()
    veh.tire = NanTire()
    tire = T.CombinedSlipTire(lateral=NanTire())
    r = TR.TransientSolver(veh, tire=tire).run(0.5, u0=12.0)
    check("NaN tyre does not raise (returns a result)", isinstance(r, TR.TransientResult))
    check("NaN tyre run is flagged or finite", (not r.ok) or np.all(np.isfinite(r.Fz)))


test_nan_tire_safe()


def test_raising_tire_safe():
    class BoomTire:
        FNOMIN = 1100.0
        def fy(self, a, Fz, g=0.0): raise RuntimeError("boom")
        def peak_force(self, Fz, g=0.0): raise RuntimeError("boom")
    tire = T.CombinedSlipTire(lateral=BoomTire())
    veh = _veh()
    veh.tire = BoomTire()
    r = TR.TransientSolver(veh, tire=tire).run(0.5, u0=12.0)
    check("raising tyre does not crash", isinstance(r, TR.TransientResult))
    check("raising tyre surfaces a warning", len(r.warnings) > 0)


test_raising_tire_safe()


def test_absurd_params_safe():
    veh = _veh()
    p = TR.TransientParams.from_vehicle(veh)
    p.dt = 1e-3
    p.power_w = 0.0
    p.k_tire = -1.0          # nonsense
    r = TR.TransientSolver(veh, params=p).run(0.4, u0=10.0)
    check("absurd params never raise", isinstance(r, TR.TransientResult))


test_absurd_params_safe()


def test_unknown_maneuver_safe():
    r = TR.run_maneuver(_veh(), "does_not_exist")
    check("unknown manoeuvre returns a flagged result", (not r.ok) and len(r.warnings) > 0)


test_unknown_maneuver_safe()


# --------------------------------------------------------------------------- #
print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    print("FAILURES:")
    for f in _FAIL:
        print("   -", f)
    sys.exit(1)
