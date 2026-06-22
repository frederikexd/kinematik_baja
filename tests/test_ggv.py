# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the GGV diagram generator (suspension/ggv.py).

These pin the behaviour a team relies on when reading a GGV to make decisions:

  - the envelope has the right SHAPE: pure-forward points carry ~zero lateral,
    pure-lateral points carry ~zero longitudinal, and the combined corners sit
    inside the axis limits (the friction ellipse is respected);
  - the design inputs the user asked for actually move the envelope the right
    way: more aero ClA -> more lateral grip; more power -> more accel g (until
    traction limited); CG height and weight distribution shift things;
  - aero downforce raises both lateral and braking g with speed, while accel g
    eventually falls as the car goes power-limited (the classic GGV signature);
  - the robustness contract HOLDS: absurd params never raise, they clamp and
    warn; a single bad direction never crashes the surface;
  - inner-wheel lift is FLAGGED rather than silently reported as grip;
  - sweep_parameter restores the parameter it touched and rejects bad names.

Imports the engine modules directly (not via the package __init__) so it runs
without the heavy CAD dependencies (trimesh/rtree). Needs only numpy.

Run:  python tests/test_ggv.py   (or: python -m pytest tests/)
"""

import os
import sys
import importlib.util
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_engine():
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(_ROOT, "suspension")]
    sys.modules.setdefault("suspension", pkg)
    mods = {}
    names = ["kinematics", "tiremodel", "dynamics", "ggv"]
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
K, T, D, G = _M["kinematics"], _M["tiremodel"], _M["dynamics"], _M["ggv"]

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _veh(**kw):
    vp = D.VehicleParams(use_param_camber=True, **kw)
    return D.VehicleDynamics(vp, tire=T.default_tire())


# --------------------------------------------------------------------------- #
def test_shape():
    print("\n[shape of the envelope]")
    res = G.quick_ggv()
    n = res.long_g.shape[1]
    mid = n // 2                       # theta = 0  -> pure forward
    check("runs on the real Pacejka chain", res.grip_model == "Pacejka MF5.2")
    check("surface shape is (speeds, directions)",
          res.long_g.shape == res.lat_g.shape == (len(res.speeds), n))
    # pure forward: lateral ~ 0, longitudinal positive (accel)
    check("pure-forward point has ~zero lateral", abs(res.lat_g[0, mid]) < 1e-6)
    check("pure-forward point is positive long g", res.long_g[0, mid] > 0)
    # pure lateral (theta = +pi/2): longitudinal ~ 0, lateral positive
    q = mid + n // 4
    check("pure-lateral point has ~zero long g", abs(res.long_g[0, q]) < 1e-3)
    check("pure-lateral point is positive lat g", res.lat_g[0, q] > 0.5)
    # a 45-degree combined point sits INSIDE both axis limits (ellipse)
    accel = res.max_accel_g[0]
    lat = res.max_lat_g[0]
    th = res.theta
    j = int(np.argmin(np.abs(th - np.pi / 4)))
    check("combined point is inside the accel axis limit",
          res.long_g[0, j] <= accel + 1e-6)
    check("combined point is inside the lateral axis limit",
          res.lat_g[0, j] <= lat + 1e-6)


def test_aero_and_power_levers():
    print("\n[design-input levers move the envelope correctly]")
    veh = _veh()
    gp = G.GGVParams()

    s = G.sweep_parameter(veh, gp, "cl_a", [0.0, 2.5, 4.0], speed=20.0,
                          metric="max_lat_g")
    check("more aero ClA -> more lateral grip",
          s["metric"][0] < s["metric"][1] < s["metric"][2])

    s = G.sweep_parameter(veh, gp, "power_w", [30000, 60000, 90000], speed=15.0,
                          metric="max_accel_g")
    check("more power -> more accel g (when traction allows)",
          s["metric"][0] < s["metric"][1] <= s["metric"][2] + 1e-9
          and s["metric"][0] < s["metric"][2])

    # weight distribution shifts balance -> changes peak lateral somewhere
    s = G.sweep_parameter(veh, gp, "weight_dist_front", [0.42, 0.50, 0.58],
                          speed=20.0, metric="max_lat_g")
    check("weight distribution changes lateral grip",
          len(set(round(x, 3) for x in s["metric"])) > 1)


def test_speed_trends():
    print("\n[speed trends: the classic GGV signature]")
    res = G.quick_ggv(cl_a=3.0, cd_a=1.4, power_w=60000)
    # downforce builds with speed -> lateral and braking g should rise
    check("lateral g rises with speed (downforce builds)",
          res.max_lat_g[-1] > res.max_lat_g[0])
    check("braking g rises with speed (downforce builds)",
          res.max_brake_g[-1] > res.max_brake_g[0])
    # accel g should eventually fall as power/v drops and drag grows
    check("accel g falls off at high speed (power limited)",
          res.max_accel_g[-1] < res.max_accel_g[np.argmax(res.max_accel_g)])


def test_wheel_lift_flagged():
    print("\n[honesty: inner-wheel lift is flagged]")
    # very high CG + narrow track forces inner-wheel lift at the limit
    res = G.quick_ggv(cg_height=520.0, track_front=900.0, track_rear=900.0,
                      cl_a=3.5)
    lifted = any("Inner-wheel lift" in w for w in res.warnings)
    check("inner-wheel lift produces a warning", lifted)
    # and it never raised getting there
    check("wheel-lift case still returns a full surface",
          res.long_g.shape[0] == len(res.speeds))


def test_robustness():
    print("\n[robustness contract]")
    # absurd params: zero mass-ish, negative-ish, huge — must not raise
    raised = False
    try:
        veh = _veh(mass=1.0, cg_height=5000.0)
        res = G.GGVGenerator(veh, G.GGVParams(power_w=0.0, cl_a=0.0)).generate(
            speeds=[5, 20, 40])
        finite = np.all(np.isfinite(res.long_g)) and np.all(np.isfinite(res.lat_g))
    except Exception:
        raised = True
        finite = False
    check("absurd params never raise", not raised)
    check("output stays finite under absurd params", finite)


def test_sweep_contract():
    print("\n[sweep_parameter contract]")
    veh = _veh()
    gp = G.GGVParams()
    before = veh.p.cg_height
    G.sweep_parameter(veh, gp, "cg_height", [250, 350, 450], speed=15.0)
    check("sweep restores the parameter it changed", veh.p.cg_height == before)

    bad = False
    try:
        G.sweep_parameter(veh, gp, "not_a_real_param", [1, 2])
    except AttributeError:
        bad = True
    check("sweep rejects an unknown parameter name", bad)

    # a GGVParams-side parameter is resolved too
    s = G.sweep_parameter(veh, gp, "cd_a", [0.5, 2.0], speed=30.0,
                          metric="max_accel_g")
    check("sweep resolves GGVParams-side parameters (cd_a)",
          s["metric"][0] >= s["metric"][1])   # more drag -> less accel


def test_combined_slip():
    print("\n[combined-slip calibration path]")
    veh = _veh()
    # symmetric circle baseline (no combined tire), no aero to isolate the effect
    base = G.GGVGenerator(veh, G.GGVParams(cl_a=0.0)).generate(speeds=[15], n_dir=73)
    # combined tire: 15% more longitudinal grip, fatter superellipse corners
    ct = T.CombinedSlipTire(lateral=T.default_tire(), mu_x_ratio=1.15,
                            ell_kx=2.2, ell_ky=2.2, is_calibrated=True)
    comb = G.GGVGenerator(veh, G.GGVParams(cl_a=0.0, combined_tire=ct)).generate(
        speeds=[15], n_dir=73)
    check("combined tire raises accel g via mu_x_ratio",
          comb.max_accel_g[0] > base.max_accel_g[0] + 1e-3)
    check("combined tire raises braking g via mu_x_ratio",
          comb.max_brake_g[0] > base.max_brake_g[0] + 1e-3)
    check("combined tire leaves pure-lateral g unchanged",
          abs(comb.max_lat_g[0] - base.max_lat_g[0]) < 1e-3)
    # the 45-degree combined point should sit farther out for the fatter ellipse
    th = base.theta
    j = int(np.argmin(np.abs(th - np.pi / 4)))
    mag_base = np.hypot(base.long_g[0, j], base.lat_g[0, j])
    mag_comb = np.hypot(comb.long_g[0, j], comb.lat_g[0, j])
    check("fatter superellipse gives more combined-g headroom",
          mag_comb > mag_base + 1e-3)


def test_laptime_agreement():
    print("\n[cross-validation against laptime.py]")
    # load laptime alongside the engine (same direct-import trick)
    path = os.path.join(_ROOT, "suspension", "laptime.py")
    spec = importlib.util.spec_from_file_location("suspension.laptime", path)
    lt = importlib.util.module_from_spec(spec)
    sys.modules["suspension.laptime"] = lt
    spec.loader.exec_module(lt)

    veh = _veh()
    # agreement should hold across downforce levels, including wingless
    all_ok = True
    worst = 0.0
    for cla in (0.0, 2.6, 4.0):
        pt = lt.Powertrain(power_kw=80, max_tractive_n=2600, cda=1.10, cla=cla,
                           crr=0.018, drive="rwd", brake_g_cap=1.8)
        res = G.validate_against_laptime(veh, pt, rel_tol=0.06)
        all_ok = all_ok and res["ok"]
        worst = max(worst, res["max_reldiff"])
    check("GGV axis limits agree with laptime within 6%% (worst %.2f%%)"
          % (worst * 100), all_ok)
    check("agreement is actually tight (<1%)", worst < 0.01)

    # and the combined tire flows through to laptime too. NOTE: laptime's brake
    # side ignores mu_x_ratio, so with mu_x_ratio>1 the brake limits diverge by
    # design; lateral and accel must still agree tightly, and the function should
    # tell us the divergence is brake-only.
    ct = T.CombinedSlipTire(lateral=T.default_tire(), mu_x_ratio=1.1,
                            ell_kx=2.0, ell_ky=2.0, is_calibrated=True)
    pt = lt.Powertrain(power_kw=80, cla=2.6, drive="rwd", combined_tire=ct)
    res = G.validate_against_laptime(veh, pt, rel_tol=0.06)
    lat_acc_ok = all(
        res["lat_reldiff"][i] <= 0.06 and res["accel_reldiff"][i] <= 0.06
        for i in range(len(res["speeds"])))
    check("with a combined tire, lateral & accel still agree tightly", lat_acc_ok)
    check("validator identifies the brake-only divergence as laptime's",
          (not res["ok"]) and ("note" in res) and "brake" in res["note"].lower())


def main():
    test_shape()
    test_aero_and_power_levers()
    test_speed_trends()
    test_wheel_lift_flagged()
    test_robustness()
    test_sweep_contract()
    test_combined_slip()
    test_laptime_agreement()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILED:", ", ".join(_FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
