# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for suspension.electronics — copper survival + signal integrity.

These exercise the exact scenario in the brief:
  * brake light + both cooling fans firing simultaneously, on traces sized so the
    worst case both melts (fusing) and browns out the ECU, and a fixed version
    that survives;
  * a CAN differential pair routed too close to the HV motor-controller net,
    and a fixed routing that clears it.

Every physical model used (IPC-2221, Onderdonk, IPC-2141 microstrip) is analytic,
so the numbers are reproducible and the asserts are exact-ish.
"""

import math
import numpy as np

from suspension.interfaces import (
    Severity, IntegrationLedger, SubsystemInterface, blank_ledger,
)
from suspension.electronics import (
    Trace, DiffPair, Aggressor, BoardLedger, check_board, worst_case_currents,
)


# --------------------------------------------------------------------------- #
#  Trace physics sanity
# --------------------------------------------------------------------------- #
def test_trace_geometry_and_resistance():
    # 0.25 mm wide, 1 oz (34.8 um), 100 mm long
    t = Trace(name="t", net="x", owner_subsystem="electrics",
              width_mm=0.25, copper_oz=1.0, length_mm=100.0)
    assert math.isclose(t.thickness_mm, 0.0348, rel_tol=1e-3)
    # R = rho*L/A ; A = 0.25*0.0348 mm^2 = 8.7e-3 mm^2
    R = t.resistance_ohm(20.0)
    assert 0.15 < R < 0.25          # ~0.198 ohm
    # hotter copper -> higher resistance
    assert t.resistance_ohm(100.0) > R


def test_fusing_current_monotonic_in_area():
    thin = Trace(name="thin", net="x", owner_subsystem="e", width_mm=0.2)
    wide = Trace(name="wide", net="x", owner_subsystem="e", width_mm=2.0)
    assert wide.fusing_current_a() > thin.fusing_current_a()
    # fusing current is finite and positive
    assert math.isfinite(thin.fusing_current_a()) and thin.fusing_current_a() > 0


def test_ipc_temp_rise_increases_with_current():
    t = Trace(name="t", net="x", owner_subsystem="e", width_mm=0.5)
    assert t.temp_rise_c(10.0) > t.temp_rise_c(2.0) > 0.0
    assert t.temp_rise_c(0.0) == 0.0


# --------------------------------------------------------------------------- #
#  The request scenario: brake light + both fans at once
# --------------------------------------------------------------------------- #
def _ledger_with_loads():
    led = blank_ledger() if callable(blank_ledger) else IntegrationLedger()
    # declare the loads as interfaces — the single source of truth
    led.set(SubsystemInterface(name="cooling", peak_current_a=8.0,
                               voltage_v=12.0, is_estimate=False,
                               owner="thermal"))
    led.set(SubsystemInterface(name="brakes", peak_current_a=2.0,
                               voltage_v=12.0, is_estimate=False,
                               owner="electrics"))
    return led


def test_simultaneous_load_rollup_reads_interfaces():
    led = _ledger_with_loads()
    board = BoardLedger()
    # one shared 12V feed trace carries BOTH fans (2x cooling) + brake light
    scenario = {"main_feed": ["cooling", "cooling", "brakes"]}
    currents = worst_case_currents(board, led, scenario)
    # 8 + 8 + 2 = 18 A, summed from the declared interfaces, not re-typed
    assert math.isclose(currents["main_feed"], 18.0, rel_tol=1e-9)


def test_undersized_trace_melts_and_browns_out_ecu():
    led = _ledger_with_loads()
    board = BoardLedger(rail_nominal_v=5.0, ecu_brownout_v=4.5, ambient_c=40.0)
    # a deliberately thin/long trace feeding the ECU rail
    board.set_trace(Trace(name="main_feed", net="lv_rail", owner_subsystem="electrics",
                          feeds="ecu", width_mm=0.15, copper_oz=1.0, length_mm=150.0,
                          is_estimate=False))
    scenario = {"main_feed": ["cooling", "cooling", "brakes"]}   # 18 A
    res = check_board(board, led, scenario)
    checks = {f.check for f in res.findings}
    sevs = {f.check: f.severity for f in res.findings}
    # under 18 A the thin trace should both overheat/fuse AND brown out the ECU
    assert res.has_hard_fail()
    assert "ecu-brownout" in checks
    assert sevs["ecu-brownout"] == Severity.FAIL
    # at least one of the copper-survival checks must be a FAIL
    assert any(sevs.get(c) == Severity.FAIL
               for c in ("trace-fusing", "trace-heating"))


def test_properly_sized_trace_survives():
    led = _ledger_with_loads()
    board = BoardLedger(rail_nominal_v=5.0, ecu_brownout_v=4.5, ambient_c=40.0)
    # a wide, short, heavy-copper feed: 6 mm, 2 oz, 50 mm
    board.set_trace(Trace(name="main_feed", net="lv_rail", owner_subsystem="electrics",
                          feeds="ecu", width_mm=6.0, copper_oz=2.0, length_mm=50.0,
                          is_estimate=False))
    scenario = {"main_feed": ["cooling", "cooling", "brakes"]}   # 18 A
    res = check_board(board, led, scenario)
    assert not res.has_hard_fail()
    brownout = [f for f in res.findings if f.check == "ecu-brownout"]
    assert brownout and brownout[0].severity == Severity.OK


def test_missing_current_is_missing_not_fail():
    board = BoardLedger()
    board.set_trace(Trace(name="orphan", net="x", owner_subsystem="electrics",
                          feeds="something"))
    res = check_board(board, ledger=None, scenario={})
    miss = [f for f in res.findings if f.check == "trace-current"]
    assert miss and miss[0].severity == Severity.MISSING


# --------------------------------------------------------------------------- #
#  Signal integrity: CAN pair vs HV motor-controller net
# --------------------------------------------------------------------------- #
def test_diffpair_impedance_estimate_reasonable():
    dp = DiffPair(name="can", owner_subsystem="electrics",
                  trace_w_mm=0.20, spacing_mm=0.20, height_mm=0.20, eps_r=4.3,
                  path_mm=[(0, 0), (50, 0)])
    z = dp.differential_z0_ohm()
    # edge-coupled differential on thin FR-4 lands in a sane band
    assert 60.0 < z < 180.0


def test_can_pair_too_close_to_hv_inverter_fails():
    board = BoardLedger(coupling_warn_mm=2.0, coupling_fail_mm=0.5)
    # CAN pair runs straight along y=0
    board.set_pair(DiffPair(name="can", owner_subsystem="electrics",
                            net_p="can_h", net_n="can_l",
                            path_mm=[(0, 0), (60, 0)], target_z0_ohm=120.0))
    # HV inverter trace runs parallel only 0.3 mm away for the whole length
    board.set_aggressor(Aggressor(name="inv", owner_subsystem="powertrain",
                                  net="hv_inverter", sw_voltage_v=400.0,
                                  edge_v_per_ns=8.0,
                                  path_mm=[(0, 0.3), (60, 0.3)]))
    res = check_board(board, ledger=None, scenario={})
    xtalk = [f for f in res.findings if f.check == "si-crosstalk"]
    assert xtalk and xtalk[0].severity == Severity.FAIL
    # both owners named so it has an owner on the board
    assert set(xtalk[0].subsystems) == {"electrics", "powertrain"}
    # and the finding declares it's a screening estimate, not a field solve
    assert "screening" in xtalk[0].detail.get("method", "")


def test_can_pair_rerouted_clear_is_ok():
    board = BoardLedger(coupling_warn_mm=2.0, coupling_fail_mm=0.5)
    board.set_pair(DiffPair(name="can", owner_subsystem="electrics",
                            path_mm=[(0, 0), (60, 0)]))
    # same inverter net but now 10 mm away
    board.set_aggressor(Aggressor(name="inv", owner_subsystem="powertrain",
                                  net="hv_inverter",
                                  path_mm=[(0, 10.0), (60, 10.0)]))
    res = check_board(board, ledger=None, scenario={})
    xtalk = [f for f in res.findings if f.check == "si-crosstalk"]
    assert xtalk and all(f.severity == Severity.OK for f in xtalk)


def test_si_detail_returns_none_for_field_solver_channels():
    board = BoardLedger()
    board.set_pair(DiffPair(name="can", owner_subsystem="electrics",
                            path_mm=[(0, 0), (60, 0)]))
    d = board.si_detail("can")
    # analytic channels present, field-solver channels honestly None
    assert d["differential_z0_ohm"] is not None
    assert d["eye_height_v"] is None
    assert d["reflection_coeff"] is None
    assert d["coupled_noise_v"] is None


# --------------------------------------------------------------------------- #
#  Persistence round-trip
# --------------------------------------------------------------------------- #
def test_board_ledger_roundtrip():
    board = BoardLedger(ambient_c=55.0)
    board.set_trace(Trace(name="t", net="lv_rail", owner_subsystem="electrics",
                          feeds="ecu", width_mm=1.0))
    board.set_pair(DiffPair(name="can", owner_subsystem="electrics",
                            path_mm=[(0, 0), (10, 0)]))
    board.set_aggressor(Aggressor(name="inv", owner_subsystem="powertrain",
                                  path_mm=[(0, 5), (10, 5)]))
    d = board.as_dict()
    back = BoardLedger.from_dict(d)
    assert back.ambient_c == 55.0
    assert "t" in back.traces and "can" in back.pairs and "inv" in back.aggressors
    assert back.pairs["can"].path_mm[1] == (10.0, 0.0)
