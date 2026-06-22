# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the structural tire co-simulation boundary (tire_cosim*.py).

These pin the contract the FTire/CDTire integration depends on:
  - the reference backend satisfies the StructuralTireModel protocol and is stateful
    in the one channel it honestly owns (slip relaxation),
  - it returns real forces but refuses to invent structural/thermal channels (None,
    and named in `synthesized`) — the honesty contract,
  - provenance correctly reports UNCALIBRATED for the reference backend,
  - the external (FTire/CDTire) stubs raise a clear, actionable error rather than
    faking physics, and report the binding they need,
  - the staggered co-sim driver advances the tyre once per macro-step and runs a
    full manoeuvre end-to-end without raising.

Run:  python tests/test_tire_cosim.py   (or: python -m pytest tests/)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension.tire_cosim import (
    StructuralTireModel, ReferenceTireModel, FTireModel, CDTireModel,
    WheelState, TireOutput, TireProvenance, TireFidelity,
    make_tire_backend, default_structural_tire)
from suspension.tire_cosim_driver import (CosimCornerSet, run_cosim_maneuver,
                                          CORNER_NAMES)

_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# ----------------------- protocol & basic forces ------------------------ #
def test_reference_satisfies_protocol():
    m = default_structural_tire()
    check("reference is a StructuralTireModel", isinstance(m, StructuralTireModel))
    check("reference exposes provenance", isinstance(m.provenance(), TireProvenance))


def test_reference_makes_force():
    m = ReferenceTireModel()
    m.reset()
    # steady slip at a representative load
    out = None
    for _ in range(50):
        out = m.step(WheelState(alpha=np.radians(4.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    check("lateral force is nonzero under slip", out is not None and abs(out.Fy) > 1.0)
    # sign: positive alpha gives negative Fy in this tyre's convention
    check("Fy sign matches tiremodel convention", out.Fy < 0.0)
    check("Fz passes through", abs(out.Fz - 1100.0) < 1e-6)


def test_reference_is_stateful_relaxation():
    """The one honest internal state: force should LAG a step change in slip."""
    m = ReferenceTireModel()
    m.reset(WheelState(alpha=0.0, Fz=1100.0, v_x=18.0))
    # step the slip target; first sample should be smaller than the settled value
    first = m.step(WheelState(alpha=np.radians(5.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    settled = None
    for _ in range(400):
        settled = m.step(WheelState(alpha=np.radians(5.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    check("force builds over distance (relaxation lag is real state)",
          abs(first.Fy) < abs(settled.Fy))


# ----------------------- the honesty contract --------------------------- #
def test_reference_refuses_to_fake_structural():
    m = ReferenceTireModel()
    m.reset()
    out = m.step(WheelState(alpha=np.radians(3.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    check("carcass deflection is None (not faked)", out.carcass_deflection_m is None)
    check("pressure distribution is None (not faked)", out.pressure_distribution is None)
    check("tread temp is None (not faked)", out.tread_temp_c is None)
    check("gas temp is None (not faked)", out.gas_temp_c is None)
    check("absent channels are named in `synthesized`",
          "carcass_deflection_m" in out.synthesized and "gas_temp_c" in out.synthesized)
    check("output reports not-structural", not out.is_structural())
    check("output reports not-thermal", not out.is_thermal())


def test_reference_provenance_is_uncalibrated():
    p = ReferenceTireModel().provenance()
    check("reference fidelity is handling-only", p.fidelity == TireFidelity.HANDLING)
    check("reference is UNCALIBRATED", p.is_calibrated is False)
    check("status string says uncalibrated", "UNCALIBRATED" in p.status())


def test_reference_never_raises_on_garbage():
    m = ReferenceTireModel()
    m.reset()
    for ws in (WheelState(alpha=float("nan"), Fz=1100.0, v_x=18.0),
               WheelState(alpha=0.05, Fz=-500.0, v_x=0.0),
               WheelState(alpha=1e9, Fz=1e9, v_x=-1e9)):
        try:
            out = m.step(ws)
            ok = isinstance(out, TireOutput)
        except Exception:
            ok = False
        check(f"no raise on pathological input (Fz={ws.Fz:g})", ok)


# ----------------------- external backend stubs ------------------------- #
def test_ftire_stub_raises_actionably():
    raised = False
    msg = ""
    try:
        FTireModel(parameter_file="my_tire.tir")  # no binding -> must refuse
    except NotImplementedError as e:
        raised = True
        msg = str(e)
    check("FTire stub refuses without a binding", raised)
    check("FTire error names what's needed (binding)", "binding" in msg.lower())
    check("FTire error points to ReferenceTireModel fallback",
          "ReferenceTireModel" in msg)


def test_cdtire_stub_raises_actionably():
    raised = False
    try:
        CDTireModel(parameter_file="my_tire.cdt")
    except NotImplementedError:
        raised = True
    check("CDTire stub refuses without a binding", raised)


def test_external_provenance_fidelity():
    # construct via factory without binding -> raises; check the declared fidelity
    # on the class instead, since we can't instantiate unbound.
    check("FTire declares structural+thermal fidelity",
          FTireModel._fidelity == TireFidelity.STRUCTURAL_THERMAL)
    check("CDTire declares structural+thermal fidelity",
          CDTireModel._fidelity == TireFidelity.STRUCTURAL_THERMAL)


def test_factory():
    m = make_tire_backend("reference")
    check("factory builds reference", isinstance(m, ReferenceTireModel))
    raised = False
    try:
        make_tire_backend("ftire", parameter_file="x.tir")
    except NotImplementedError:
        raised = True
    check("factory ftire without binding raises", raised)
    raised = False
    try:
        make_tire_backend("nonsense")
    except ValueError:
        raised = True
    check("factory rejects unknown backend", raised)


# ----------------------- the staggered co-sim driver -------------------- #
def test_corner_set_advances_and_holds_force():
    cs = CosimCornerSet()
    cs.reset()
    Fx0, Fy0, Fz0 = cs.forces()
    check("forces start at zero before any step", float(np.sum(np.abs(Fy0))) == 0.0)
    ws = [WheelState(alpha=np.radians(4.0), Fz=1100.0, v_x=18.0, dt=1e-3)
          for _ in range(4)]
    for _ in range(50):
        cs.advance(ws)
    Fx1, Fy1, Fz1 = cs.forces()
    check("four corners develop force after stepping",
          float(np.sum(np.abs(Fy1))) > 1.0)
    check("absent channels reported at corner-set level",
          "gas_temp_c" in cs.absent_channels())


def test_cosim_maneuver_end_to_end():
    # build a minimal vehicle for the driver to integrate
    from suspension import VehicleDynamics, VehicleParams
    from suspension.tiremodel import default_tire
    veh = VehicleDynamics(VehicleParams(), tire=default_tire())

    out = run_cosim_maneuver(veh, kind="step_steer", steer_deg=4.0, t_end=1.0)
    res = out["result"]
    check("cosim step-steer returns a result", res is not None)
    check("cosim run did not blow up", res is not None and res.ok)
    check("cosim developed lateral g", res is not None and np.max(np.abs(res.ay)) > 0.05)
    check("backend status is reported", "reference-pacejka" in out["backend_status"])
    check("structural channels honestly absent in history",
          out["tire_history"] is not None and
          "carcass_deflection_m" in out["tire_history"].absent_channels)


def test_cosim_curb_strike_runs():
    from suspension import VehicleDynamics, VehicleParams
    from suspension.tiremodel import default_tire
    veh = VehicleDynamics(VehicleParams(), tire=default_tire())
    out = run_cosim_maneuver(veh, kind="curb_strike", t_end=0.6)
    check("cosim curb strike returns a result", out["result"] is not None)
    # NOTE: the reference backend has no carcass enveloping, so the curb's real
    # high-frequency content is only what the solver's own unsprung model provides;
    # this is exactly the limitation a real FTire backend would remove. The test
    # just confirms the path runs and stays honest about the missing channel.
    check("curb-strike still flags carcass channel absent",
          out["tire_history"] is not None and
          "carcass_deflection_m" in out["tire_history"].absent_channels)


def run_all():
    print("structural tire co-sim boundary")
    for fn in (test_reference_satisfies_protocol, test_reference_makes_force,
               test_reference_is_stateful_relaxation,
               test_reference_refuses_to_fake_structural,
               test_reference_provenance_is_uncalibrated,
               test_reference_never_raises_on_garbage,
               test_ftire_stub_raises_actionably, test_cdtire_stub_raises_actionably,
               test_external_provenance_fidelity, test_factory,
               test_corner_set_advances_and_holds_force,
               test_cosim_maneuver_end_to_end, test_cosim_curb_strike_runs,
               test_example_backend_conforms,
               test_example_backend_is_honest_about_fake_binding,
               test_example_backend_drives_full_cosim):
        fn()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILED:", ", ".join(_FAIL))
    return not _FAIL


# ----------------------- the worked-example wrapper --------------------- #
def test_example_backend_conforms():
    """The example wrapper is the template a real FTire/CDTire wrapper copies;
    pin that it satisfies the protocol and the conformance checklist."""
    from suspension.tire_cosim_ftire_example import ExampleStructuralBackend
    m = ExampleStructuralBackend()
    check("example backend is a StructuralTireModel",
          isinstance(m, StructuralTireModel))
    m.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    out = None
    for _ in range(200):
        out = m.step(WheelState(alpha=np.radians(4.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    # it DOES fill structural + thermal channels (unlike the reference backend)
    check("example fills carcass deflection", out.carcass_deflection_m is not None)
    check("example fills pressure distribution", out.pressure_distribution is not None)
    check("example fills tread temp", out.tread_temp_c is not None)
    check("example fills gas temp", out.gas_temp_c is not None)
    check("example reports structural", out.is_structural())
    check("example reports thermal", out.is_thermal())


def test_example_backend_is_honest_about_fake_binding():
    """Critical: fabricated channels must be flagged and the backend uncalibrated."""
    from suspension.tire_cosim_ftire_example import ExampleStructuralBackend
    m = ExampleStructuralBackend(parameter_file="pretend_real.tir")
    m.reset()
    out = m.step(WheelState(alpha=np.radians(3.0), Fz=1100.0, v_x=18.0, dt=1e-3))
    # even with a real-looking file string, the fake binding is never calibrated
    check("example stays UNCALIBRATED despite file string",
          m.provenance().is_calibrated is False)
    check("fabricated structural channel flagged in synthesized",
          "pressure_distribution" in out.synthesized)
    check("fabricated thermal channel flagged in synthesized",
          "gas_temp_c" in out.synthesized)


def test_example_backend_drives_full_cosim():
    from suspension import VehicleDynamics, VehicleParams
    from suspension.tiremodel import default_tire
    from suspension.tire_cosim_ftire_example import example_backend_factory
    from suspension.tire_cosim_driver import run_cosim_maneuver
    veh = VehicleDynamics(VehicleParams(), tire=default_tire())
    out = run_cosim_maneuver(veh, kind="step_steer", steer_deg=4.0, t_end=1.0,
                             backend_factory=example_backend_factory())
    check("example backend runs a full manoeuvre", out["result"] is not None)
    check("example backend develops grip",
          out["result"] is not None and np.max(np.abs(out["result"].ay)) > 0.05)
    check("example backend status names it",
          "example-structural" in out["backend_status"])


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
