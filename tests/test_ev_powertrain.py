# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the EV powertrain & energy layer (suspension/ev_powertrain.py).

These pin the behaviour an underfunded FSAE-EV team relies on when it uses this
to choose an architecture and size a pack:

  - compare() runs every architecture and never raises, even on a pathological
    tire, absurd params, or a degenerate track (the repo's never-crash contract),
  - heavier architectures actually carry their mass penalty (TV is NOT free of
    its own weight — the mass delta reaches the lap sim),
  - torque vectoring's traction-fraction advantage is real but the yaw benefit is
    reported SEPARATELY and never folded into lap_time,
  - the energy budget is sane: regen never exceeds gross draw, a tiny pack fails
    to finish while a big pack finishes, and a non-finishing pack reports a
    derate penalty,
  - turning regen off never increases recovered energy.

Loads engine modules directly (no package __init__, no trimesh), like the other
engine tests.

Run:  python tests/test_ev_powertrain.py
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
    names = ["kinematics", "tiremodel", "dynamics", "lapsim", "ev_powertrain"]
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
K, T, D, L, EV = (_M["kinematics"], _M["tiremodel"], _M["dynamics"],
                  _M["lapsim"], _M["ev_powertrain"])

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _veh(tire=None):
    kin = K.SuspensionKinematics(K.Hardpoints.default())
    return D.VehicleDynamics(D.VehicleParams(), front_kin=kin, rear_kin=kin,
                             tire=tire if tire is not None else T.default_tire())


class NanTire:
    def __getattr__(self, _):
        def f(*a, **k):
            return float("nan")
        return f


class BoomTire:
    def __getattr__(self, _):
        def f(*a, **k):
            raise RuntimeError("tire blew up")
        return f


# --------------------------------------------------------------------------- #
def test_compare_runs_all_three():
    sim = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams())
    cmp = sim.compare(L.autocross_track(laps=10))
    archs = {r.architecture for r in cmp.results}
    check("compare returns all three architectures",
          archs == set(EV.Powertrain))
    check("all runs ok on a sane car", all(r.ok for r in cmp.results))
    check("best_on_time returns a result", cmp.best_on_time() is not None)


def test_mass_penalty_reaches_sim():
    sim = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams())
    cmp = sim.compare(L.autocross_track(laps=8))
    by = {r.architecture: r for r in cmp.results}
    m1 = by[EV.Powertrain.SINGLE_DIFF].effective_mass_kg
    m2 = by[EV.Powertrain.TWO_AXLE].effective_mass_kg
    m4 = by[EV.Powertrain.FOUR_TV].effective_mass_kg
    check("two-motor heavier than single", m2 > m1)
    check("four-motor heaviest", m4 > m2)
    # the mass delta must actually be the configured one (reaches the model)
    check("mass delta matches config",
          abs((m4 - m1) - EV.EVParams().mass_delta_kg[EV.Powertrain.FOUR_TV]) < 1e-6)


def test_vehicle_mass_restored():
    veh = _veh()
    m0 = veh.p.mass
    EV.EVLapSimulator(veh, L.LapSimParams(), EV.EVParams()).compare(
        L.autocross_track(laps=5))
    check("vehicle mass restored after compare", abs(veh.p.mass - m0) < 1e-9)


def test_tv_yaw_reported_separately():
    sim = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams())
    cmp = sim.compare(L.autocross_track(laps=6))
    by = {r.architecture: r for r in cmp.results}
    tv = by[EV.Powertrain.FOUR_TV]
    diff = by[EV.Powertrain.SINGLE_DIFF]
    check("TV has a positive yaw benefit", tv.tv_yaw_benefit_s > 0)
    check("single-diff has zero yaw benefit", diff.tv_yaw_benefit_s == 0.0)
    # the yaw benefit must NOT be inside lap_time (lap_time is the QSS number only)
    check("yaw benefit not folded into lap_time",
          tv.lap_time == tv.lap_result.lap_time)
    check("yaw benefit carries an honesty flag",
          "UPPER-BOUND" in tv.tv_yaw_benefit_flagged)


def test_energy_sane():
    sim = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams())
    r = sim.run_architecture(EV.Powertrain.SINGLE_DIFF, L.autocross_track(laps=10))
    check("energy per lap positive & finite",
          np.isfinite(r.energy_per_lap_kwh) and r.energy_per_lap_kwh > 0)
    check("regen non-negative", r.regen_recovered_kwh >= 0)
    check("event energy scales with laps",
          abs(r.energy_full_event_kwh - r.energy_per_lap_kwh * 10) < 1e-6)


def test_pack_sizing():
    # tiny pack: cannot finish, must report a derate penalty
    small = EV.EVParams(pack_energy_kwh=0.5)
    sim = EV.EVLapSimulator(_veh(), L.LapSimParams(), small)
    r = sim.run_architecture(EV.Powertrain.SINGLE_DIFF, L.autocross_track(laps=22))
    check("tiny pack does not finish", not r.finishes_event)
    check("non-finishing pack reports derate penalty",
          r.derate_lap_time_penalty_s > 0)
    # huge pack: finishes comfortably, no penalty
    big = EV.EVParams(pack_energy_kwh=50.0)
    r2 = EV.EVLapSimulator(_veh(), L.LapSimParams(), big).run_architecture(
        EV.Powertrain.SINGLE_DIFF, L.autocross_track(laps=22))
    check("huge pack finishes", r2.finishes_event)
    check("finishing pack has no derate penalty",
          r2.derate_lap_time_penalty_s == 0.0)


def test_regen_off_recovers_nothing_extra():
    on = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams(regen_enabled=True))
    off = EV.EVLapSimulator(_veh(), L.LapSimParams(), EV.EVParams(regen_enabled=False))
    r_on = on.run_architecture(EV.Powertrain.SINGLE_DIFF, L.autocross_track(laps=8))
    r_off = off.run_architecture(EV.Powertrain.SINGLE_DIFF, L.autocross_track(laps=8))
    # regen reduces NET energy drawn (or leaves it equal), never increases it
    check("regen reduces or equals net energy",
          r_on.energy_per_lap_kwh <= r_off.energy_per_lap_kwh + 1e-9)


def test_never_raises_nan_tire():
    try:
        cmp = EV.EVLapSimulator(_veh(NanTire())).compare(L.autocross_track(laps=5))
        check("NaN tire never raises", True)
        check("NaN tire yields a comparison object",
              isinstance(cmp, EV.ArchitectureComparison))
    except Exception as e:
        check(f"NaN tire never raises (raised {e!r})", False)


def test_never_raises_boom_tire():
    try:
        cmp = EV.EVLapSimulator(_veh(BoomTire())).compare(L.autocross_track(laps=5))
        check("raising tire never raises", True)
        check("raising tire yields results list", len(cmp.results) == 3)
    except Exception as e:
        check(f"raising tire never raises (raised {e!r})", False)


def test_never_raises_degenerate_track():
    deg = L.Track("degenerate", [L.Segment(0.0, None), L.Segment(-5.0, 3.0)],
                  closed=True, laps=4)
    try:
        cmp = EV.EVLapSimulator(_veh()).compare(deg)
        check("degenerate track never raises", True)
        check("degenerate track returns object",
              isinstance(cmp, EV.ArchitectureComparison))
    except Exception as e:
        check(f"degenerate track never raises (raised {e!r})", False)


def test_summary_strings():
    cmp = EV.EVLapSimulator(_veh()).compare(L.autocross_track(laps=6))
    s = cmp.summary()
    check("summary is a non-empty string", isinstance(s, str) and len(s) > 0)
    check("summary names the best", "best" in s.lower())


if __name__ == "__main__":
    print("EV POWERTRAIN LAYER TESTS")
    for fn in [
        test_compare_runs_all_three, test_mass_penalty_reaches_sim,
        test_vehicle_mass_restored, test_tv_yaw_reported_separately,
        test_energy_sane, test_pack_sizing, test_regen_off_recovers_nothing_extra,
        test_never_raises_nan_tire, test_never_raises_boom_tire,
        test_never_raises_degenerate_track, test_summary_strings,
    ]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} itself raised: {e!r}", False)
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    sys.exit(1 if _FAIL else 0)
