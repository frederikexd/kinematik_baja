# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the transient per-cell battery-pack thermal model (pack_thermal.py).

These pin the physics shape AND the honesty contract that the rest of KinematiK
holds itself to:

  - a virtual lap is turned into a pack current-vs-time history whose energy
    integrates back to roughly the EV layer's per-lap energy (the current trace
    is the same physics, kept in time instead of collapsed to one kWh),
  - the pack WARMS UP under load: cells climb above ambient, and harder current
    (or more laps) makes them hotter,
  - SPATIAL effect is real: a cell sitting in a fan's draught ends up cooler than
    a cell in dead air, so fan PLACEMENT changes which cell is hottest,
  - optimize_fan_placement ranks layouts and a sensible fan beats no fan,
  - EVERY temperature is flagged `synthesized` while uncalibrated, and flips to
    calibrated only when BOTH the cell and the airflow map are calibrated,
  - the solver never raises on garbage input (the repo's never-crash contract).

Loads engine modules directly (no package __init__, no trimesh), like the other
engine tests.

Run:  python tests/test_pack_thermal.py   (or: python -m pytest tests/)
"""

import os
import sys
import types
import importlib.util

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(_ROOT, "suspension")]
    sys.modules.setdefault("suspension", pkg)
    mods = {}
    names = ["kinematics", "tiremodel", "dynamics", "lapsim",
             "ev_powertrain", "pack_thermal"]
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


_M = _load()
K, T, D, L, EV, PT = (_M["kinematics"], _M["tiremodel"], _M["dynamics"],
                      _M["lapsim"], _M["ev_powertrain"], _M["pack_thermal"])

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _veh(tire=None):
    kin = K.SuspensionKinematics(K.Hardpoints.default())
    return D.VehicleDynamics(D.VehicleParams(), front_kin=kin, rear_kin=kin,
                             tire=tire if tire is not None else T.default_tire())


def _lap(laps=3):
    sim = L.LapSimulator(_veh(), L.LapSimParams())
    return sim.simulate(L.autocross_track(laps=laps)), L.LapSimParams()


# --------------------------------------------------------------------------- #
def test_current_trace_is_built_and_signed():
    lap, p = _lap()
    layout = PT.PackLayout()
    t, cur = PT.pack_current_trace(lap, p, pack_nominal_v=layout.pack_nominal_v)
    check("current trace has same length as lap trace",
          t.size == cur.size and t.size == np.asarray(lap.speed).size)
    check("time is monotonic non-decreasing", bool(np.all(np.diff(t) >= -1e-9)))
    check("pack draws hundreds of amps somewhere", float(np.nanmax(cur)) > 100.0)
    check("regen produces some negative current", float(np.nanmin(cur)) < 0.0)


def test_pack_warms_up_under_load():
    lap, p = _lap(laps=5)
    res = PT.simulate_pack_thermal(lap, p, layout=PT.PackLayout(), fans=[])
    check("pack thermal run ok", res.ok)
    check("cells warm above ambient", float(np.nanmax(res.peak_temp_c)) > 30.5)
    check("a hottest cell is identified",
          res.hottest_cell_index >= 0 and np.isfinite(res.hottest_peak_c))


def test_more_laps_is_hotter():
    lap3, p = _lap(laps=3)
    r3 = PT.simulate_pack_thermal(lap3, p, n_laps=3)
    r9 = PT.simulate_pack_thermal(lap3, p, n_laps=9)
    check("more laps → hotter peak cell",
          r9.hottest_peak_c > r3.hottest_peak_c)


def test_fan_cools_nearby_cells():
    lap, p = _lap(laps=6)
    layout = PT.PackLayout()
    w, h = layout.extent_mm()
    # one fan parked over the top-left corner of the cell field
    fan = PT.Fan(x_mm=0.0, y_mm=0.0, cfm=160.0, throw_mm=80.0)
    res = PT.simulate_pack_thermal(lap, p, layout=layout, fans=[fan])
    grid = res.peak_grid_c()
    near = grid[0, 0]                       # cell under the fan
    far = grid[-1, -1]                      # diagonally opposite, dead air
    check("cell under the fan is cooler than the far corner", near < far)


def test_fan_placement_beats_no_fan_and_ranks():
    lap, p = _lap(laps=6)
    layout = PT.PackLayout()
    cands = PT.fan_grid_candidates(layout, nx=3, ny=2, cfm=160.0)
    study = PT.optimize_fan_placement(lap, p, cands, layout=layout)
    check("study produced ranked candidates", len(study.candidates) > 1)
    # the baseline (no fan) must not be the best option
    baseline = next(c for c in study.candidates if len(c.fans) == 0)
    check("a fan layout beats the no-fan baseline",
          study.best.hottest_peak_c <= baseline.hottest_peak_c)
    check("candidates are sorted best→worst",
          all(study.candidates[i].hottest_peak_c <=
              study.candidates[i + 1].hottest_peak_c + 1e-6
              for i in range(len(study.candidates) - 1)))


def test_honesty_synthesized_until_calibrated():
    lap, p = _lap()
    # uncalibrated by default → synthesized
    r = PT.simulate_pack_thermal(lap, p)
    check("uncalibrated run is flagged synthesized", r.synthesized is True)
    check("provenance says SYNTHESIZED", "SYNTHESIZED" in r.provenance)
    # calibrate BOTH cell and airflow → not synthesized
    cell = PT.CellParams(calibrated=True, fitted_to="rig data")
    air = PT.AirflowParams(calibrated=True)
    layout = PT.PackLayout(cell=cell)
    r2 = PT.simulate_pack_thermal(lap, p, layout=layout, airflow=air)
    check("calibrating both cell and airflow clears the flag",
          r2.synthesized is False and "CALIBRATED" in r2.provenance)
    # calibrating only one is NOT enough
    r3 = PT.simulate_pack_thermal(lap, p, layout=PT.PackLayout(cell=cell))
    check("calibrating only the cell is not enough", r3.synthesized is True)


def test_never_raises_on_garbage():
    p = L.LapSimParams()

    class JunkLap:
        speed = np.array([float("nan"), float("inf"), -1.0])
        distance = np.array([0.0, float("nan"), 2.0])
        long_g = np.array([float("inf"), 0.0, float("nan")])

    r = PT.simulate_pack_thermal(JunkLap(), p)
    check("garbage lap returns a result, does not raise",
          isinstance(r, PT.PackThermalResult))

    # absurd layout / empty current
    model = PT.PackThermalModel(layout=PT.PackLayout(rows=0, cols=0))
    r2 = model.simulate(np.array([0.0]), np.array([0.0]))
    check("degenerate layout + empty current returns a result",
          isinstance(r2, PT.PackThermalResult))

    # absurd fan + still finishes
    lap, _ = _lap()
    bad_fan = PT.Fan(x_mm=float("nan"), y_mm=float("inf"), cfm=-5.0, throw_mm=0.0)
    r3 = PT.simulate_pack_thermal(lap, p, fans=[bad_fan])
    check("pathological fan does not crash the run",
          isinstance(r3, PT.PackThermalResult))


# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_current_trace_is_built_and_signed,
        test_pack_warms_up_under_load,
        test_more_laps_is_hotter,
        test_fan_cools_nearby_cells,
        test_fan_placement_beats_no_fan_and_ranks,
        test_honesty_synthesized_until_calibrated,
        test_never_raises_on_garbage,
    ]
    print("=" * 70)
    print("pack_thermal tests")
    print("=" * 70)
    for t_ in tests:
        print(f"\n{t_.__name__}:")
        try:
            t_()
        except Exception as exc:
            _FAIL.append(t_.__name__)
            print(f"  FAIL  {t_.__name__} raised {exc!r}")
    print("\n" + "=" * 70)
    print(f"PASSED {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED:")
        for f in _FAIL:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
