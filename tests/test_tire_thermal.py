# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the lumped-parameter tyre thermal channel (tire_thermal.py).

These pin the physics shape AND the honesty contract:
  - ThermalTireModel satisfies the StructuralTireModel protocol (drops into the
    same co-sim driver as the reference backend),
  - it actually fills tread/carcass/gas temperature + hot pressure (the channels
    the reference backend leaves None),
  - the tyre WARMS UP under sliding load, the carcass/gas lag the tread, and the
    inflation pressure rises with gas temperature (real energy balance),
  - across-width band spread appears under camber/slip (the pyrometer signature),
  - EVERY thermal channel is flagged `synthesized` while uncalibrated, and the
    provenance reports THERMAL fidelity + UNCALIBRATED — the honesty contract,
  - flipping `calibrated=True` stops the flagging (the only thing that should),
  - the optional mu(T) grip feedback is off by default, flagged when on, and makes
    a cold tyre produce less grip than a warm one,
  - the backend never raises on garbage input,
  - the factory builds it via make_tire_backend("thermal").

Run:  python tests/test_tire_thermal.py   (or: python -m pytest tests/)
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prefer the real package (so the full-cosim integration test can run); if its
# heavy optional deps (trimesh, etc.) are missing, fall back to a lightweight stub
# that exposes only the tyre submodules this test needs.
_FULL_PACKAGE = True
try:
    import suspension  # noqa: F401
    importlib = __import__("importlib")
    tiremodel = importlib.import_module("suspension.tiremodel")
    tire_cosim = importlib.import_module("suspension.tire_cosim")
    tire_thermal = importlib.import_module("suspension.tire_thermal")
except Exception:
    _FULL_PACKAGE = False
    import types
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "suspension")]
    sys.modules["suspension"] = pkg
    import importlib
    tiremodel = importlib.import_module("suspension.tiremodel")
    tire_cosim = importlib.import_module("suspension.tire_cosim")
    tire_thermal = importlib.import_module("suspension.tire_thermal")

from suspension.tire_cosim import (StructuralTireModel, WheelState, TireOutput,
                                   TireProvenance, TireFidelity, make_tire_backend)
from suspension.tire_thermal import (ThermalTireModel, ThermalParams, ThermalRun,
                                     default_thermal_params, simulate_warmup, psi)

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# ----------------------------- protocol --------------------------------- #
def test_thermal_satisfies_protocol():
    m = ThermalTireModel()
    check("thermal is a StructuralTireModel", isinstance(m, StructuralTireModel))
    check("thermal exposes provenance", isinstance(m.provenance(), TireProvenance))
    check("thermal fidelity is THERMAL",
          m.provenance().fidelity == TireFidelity.THERMAL)


def test_factory_builds_thermal():
    m = make_tire_backend("thermal")
    check("factory builds ThermalTireModel", isinstance(m, ThermalTireModel))
    m2 = make_tire_backend("lumped-thermal")
    check("factory alias lumped-thermal works", isinstance(m2, ThermalTireModel))


# ----------------------------- physics ---------------------------------- #
def test_fills_thermal_channels():
    m = ThermalTireModel()
    m.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    out = m.step(WheelState(alpha=math.radians(4.0), Fz=1200.0, v_x=18.0, dt=5e-3))
    check("tread_temp_c populated", out.tread_temp_c is not None)
    check("carcass_temp_c populated", out.carcass_temp_c is not None)
    check("gas_temp_c populated", out.gas_temp_c is not None)
    check("inflation_pressure_pa populated", out.inflation_pressure_pa is not None)
    check("is_thermal() true", out.is_thermal())


def test_tire_warms_up():
    run = simulate_warmup(alpha_deg=4.0, Fz=1300.0, v_x=20.0, duration_s=120.0,
                          dt=5e-3)
    mean = run.tread_mean_c()
    check("tread heats above its start", mean[-1] > mean[0] + 5.0)
    # early warm-up is monotone (energy in dominates losses)
    early = np.diff(mean[:200])
    check("warm-up is monotone early", bool(np.all(early >= -1e-6)))
    # carcass and gas LAG the tread (slower thermal masses)
    check("carcass lags tread", run.carcass_c[-1] < mean[-1])
    check("gas lags carcass-ish", run.gas_c[-1] <= run.carcass_c[-1] + 1.0)


def test_pressure_rises_with_gas_temp():
    run = simulate_warmup(alpha_deg=5.0, Fz=1400.0, v_x=22.0, duration_s=150.0,
                          dt=5e-3)
    check("hot pressure exceeds cold set", run.pressure_pa[-1] > run.pressure_pa[0])
    # sanity: a modest few-psi rise, not absurd
    rise_psi = psi(run.pressure_pa[-1]) - psi(run.pressure_pa[0])
    check("pressure rise is physically modest (<6 psi)", 0.0 < rise_psi < 6.0)


def test_across_width_spread_under_camber():
    p = default_thermal_params()
    if p.n_bands < 2:
        check("multi-band model for spread test", False)
        return
    run = simulate_warmup(alpha_deg=5.0, gamma_deg=3.0, Fz=1400.0, v_x=22.0,
                          duration_s=120.0, dt=5e-3)
    spread = run.tread_c[-1].max() - run.tread_c[-1].min()
    check("camber/slip produces across-width spread", spread > 1.0)


def test_no_load_no_heating():
    m = ThermalTireModel()
    m.reset(WheelState(ambient_temp_c=25.0, track_temp_c=25.0))
    start = m.mean_tread_c()
    for _ in range(500):
        m.step(WheelState(alpha=0.0, Fz=0.0, v_x=0.0, dt=5e-3))
    check("no load / no speed -> no significant heating",
          abs(m.mean_tread_c() - start) < 1.0)


# --------------------------- honesty contract --------------------------- #
def test_uncalibrated_is_flagged_synthesized():
    m = ThermalTireModel()  # default params: calibrated=False
    m.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    out = m.step(WheelState(alpha=math.radians(4.0), Fz=1200.0, v_x=18.0, dt=5e-3))
    for ch in ("tread_temp_c", "carcass_temp_c", "gas_temp_c",
               "inflation_pressure_pa"):
        check(f"{ch} flagged synthesized while uncalibrated",
              ch in out.synthesized)
    check("provenance reports UNCALIBRATED",
          m.provenance().is_calibrated is False)
    check("provenance status says UNCALIBRATED",
          "UNCALIBRATED" in m.provenance().status())
    check("warnings name the uncalibrated thermal state",
          any("UNCALIBRATED" in w for w in m.warnings()))


def test_calibrated_stops_flagging():
    p = ThermalParams(calibrated=True, fitted_to="swept TTC + IR rig")
    m = ThermalTireModel(params=p)
    m.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    out = m.step(WheelState(alpha=math.radians(4.0), Fz=1200.0, v_x=18.0, dt=5e-3))
    check("calibrated: tread not flagged synthesized",
          "tread_temp_c" not in out.synthesized)
    check("calibrated: gas not flagged synthesized",
          "gas_temp_c" not in out.synthesized)
    check("calibrated provenance is_calibrated True",
          m.provenance().is_calibrated is True)
    # structural channels this backend genuinely lacks are STILL honestly absent
    check("structural channels still flagged (no carcass mesh here)",
          "pressure_distribution" in out.synthesized)
    check("carcass_deflection still None", out.carcass_deflection_m is None)


# ----------------------------- mu(T) feedback --------------------------- #
def test_mu_feedback_off_by_default():
    m = ThermalTireModel()
    check("mu feedback off by default", m.params.enable_mu_feedback is False)
    m.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    out = m.step(WheelState(alpha=math.radians(4.0), Fz=1200.0, v_x=18.0, dt=5e-3))
    check("mu_thermal NOT flagged when feedback off",
          "mu_thermal" not in out.synthesized)


def test_mu_feedback_flagged_and_changes_grip():
    p = ThermalParams(enable_mu_feedback=True)
    m = ThermalTireModel(params=p)
    m.reset(WheelState(ambient_temp_c=15.0, track_temp_c=20.0))
    out0 = m.step(WheelState(alpha=math.radians(4.0), Fz=1300.0, v_x=20.0, dt=5e-3))
    check("mu_thermal flagged synthesized when feedback on",
          "mu_thermal" in out0.synthesized)
    cold_scale = m._mu_scale_last
    # warm it up and compare the grip multiplier
    for _ in range(8000):
        m.step(WheelState(alpha=math.radians(4.0), Fz=1300.0, v_x=20.0, dt=5e-3))
    warm_scale = m._mu_scale_last
    check("cold tyre scales grip below 1.0", cold_scale < 1.0)
    check("warming toward optimum increases grip scale",
          warm_scale > cold_scale)


# ------------------------------ robustness ------------------------------ #
def test_never_raises_on_garbage():
    m = ThermalTireModel()
    m.reset()
    bad = WheelState(alpha=float("nan"), Fz=-1e9, v_x=float("inf"), kappa=1e6,
                     dt=-1.0)
    try:
        out = m.step(bad)
        ok = isinstance(out, TireOutput)
    except Exception:
        ok = False
    check("thermal backend never raises on garbage input", ok)


def test_temperatures_stay_bounded():
    m = ThermalTireModel()
    m.reset(WheelState(ambient_temp_c=25.0, track_temp_c=30.0))
    for _ in range(20000):
        m.step(WheelState(alpha=math.radians(8.0), Fz=2000.0, v_x=30.0,
                          kappa=0.1, dt=5e-3))
    check("tread temperature stays physically bounded",
          -40.0 <= m.mean_tread_c() <= 350.0)


def test_drives_full_cosim():
    """The thermal backend must drop into the existing four-corner co-sim driver."""
    if not _FULL_PACKAGE:
        check("thermal backend full co-sim (skipped: optional deps absent)", True)
        return
    from suspension import VehicleDynamics, VehicleParams
    from suspension.tiremodel import default_tire
    from suspension.tire_cosim_driver import run_cosim_maneuver
    veh = VehicleDynamics(VehicleParams(), tire=default_tire())
    out = run_cosim_maneuver(veh, kind="step_steer", steer_deg=4.0, t_end=1.0,
                             backend_factory=lambda: ThermalTireModel())
    check("thermal backend runs a full manoeuvre", out["result"] is not None)
    check("thermal backend develops grip",
          out["result"] is not None and np.max(np.abs(out["result"].ay)) > 0.05)
    check("thermal backend status names it",
          "lumped-thermal" in out["backend_status"])


def run_all():
    print("lumped tyre thermal channel")
    for fn in (test_thermal_satisfies_protocol, test_factory_builds_thermal,
               test_fills_thermal_channels, test_tire_warms_up,
               test_pressure_rises_with_gas_temp,
               test_across_width_spread_under_camber, test_no_load_no_heating,
               test_uncalibrated_is_flagged_synthesized,
               test_calibrated_stops_flagging, test_mu_feedback_off_by_default,
               test_mu_feedback_flagged_and_changes_grip,
               test_never_raises_on_garbage, test_temperatures_stay_bounded,
               test_drives_full_cosim):
        fn()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILED:", ", ".join(_FAIL))
    return not _FAIL


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
