# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the surface pressure-tap pipeline — raw transducer volts -> C_p mapped
onto the wing -> RMSE vs CFD.

These pin the behaviour that makes the feature a faithful reduction rather than a
pretty plot of made-up numbers:

  1. the volts -> C_p reduction is EXACT: a known gauge pressure round-trips to the
     C_p = (p - p_inf)/q it must, against the run's real dynamic pressure,
  2. the wing mapping reads as a wing — the chordwise C_p(x/c) comes back sorted LE
     -> TE, the suction peak and loading integral are where the geometry says,
  3. STALL is detectable: a flat aft plateau (separated boundary layer) is flagged
     where a healthy pressure recovery is not — the thing the coefficient correlation
     structurally cannot see,
  4. the honesty contract holds — an uncalibrated channel, a railed transducer, and a
     tap with no CFD counterpart are HOLES, NaN / unpaired, never zero-filled or
     snapped to a neighbour; the RMSE is quoted only over taps that genuinely paired,
  5. the RMSE correlation is right: a perfect twin gives RMSE 0; a known offset gives
     exactly that offset; coverage below the floor refuses to certify a match,
  6. provenance surfaces the q source, the short-averaging and floor/blockage
     warnings that decide whether a C_p is comparable to free-air CFD at all.

Run:  python -m pytest tests/test_pressure_tap.py
"""
import math

import numpy as np
import pytest

from suspension.aero import (
    WingSurface, TapLocation, TapCalibration, ScanProvenance, RawPressureScan,
    CpField, CFDSurfaceCp, correlate_cp, DEFAULT_CP_TOL,
)


# --------------------------------------------------------------------------- #
#  Helpers — synthesise a raw scan from a target C_p distribution
# --------------------------------------------------------------------------- #
def _scan_from_cp(cp_by_tap, taps, *, rho=1.225, V=25.0, p_inf=0.0,
                  sens=1000.0, zero=0.0, n_samples=64, noise=0.0, seed=0):
    """Build a RawPressureScan whose reduction yields the given C_p per tap."""
    q = 0.5 * rho * V * V
    rng = np.random.default_rng(seed)
    cols, cals = [], {}
    for t in taps:
        cp = cp_by_tap[t.tap_id]
        p = p_inf + cp * q                       # gauge pressure giving this C_p
        v_mean = p / sens + zero
        col = np.full(n_samples, v_mean)
        if noise:
            col = col + rng.normal(0.0, noise, n_samples)
        cols.append(col)
        cals[t.tap_id] = TapCalibration(
            sensitivity_pa_per_v=sens, zero_offset_v=zero,
            is_calibrated=True, saturation_v=10.0)
    volts = np.column_stack(cols)
    prov = ScanProvenance(facility="A2", rho=rho, speed_ms=V,
                          p_static_inf_pa=p_inf, sample_seconds=8.0)
    return RawPressureScan(volts, taps, cals), prov


def _healthy_main_wing():
    """A main element with a sharp suction peak and smooth recovery, both surfaces."""
    xc = [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]
    cp_s = [-2.6, -2.2, -1.6, -1.0, -0.5, -0.1]      # recovers toward 0
    cp_p = [0.6, 0.4, 0.25, 0.1, 0.0, -0.05]
    taps, cp_by = [], {}
    for i, x in enumerate(xc):
        ts = TapLocation(f"s{i}", "main", x, surface=WingSurface.SUCTION)
        tp = TapLocation(f"p{i}", "main", x, surface=WingSurface.PRESSURE)
        taps += [ts, tp]
        cp_by[ts.tap_id] = cp_s[i]
        cp_by[tp.tap_id] = cp_p[i]
    return taps, cp_by


# --------------------------------------------------------------------------- #
#  1. The volts -> C_p reduction is exact
# --------------------------------------------------------------------------- #
def test_volts_to_cp_is_exact():
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    cp_by = {"t0": -1.5}
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    assert abs(cp.cp["t0"] - (-1.5)) < 1e-9
    assert cp.coverage() == 1.0


def test_dynamic_pressure_half_rho_v_squared():
    prov = ScanProvenance("A2", rho=1.2, speed_ms=30.0)
    assert abs(prov.dynamic_pressure() - 0.5 * 1.2 * 900.0) < 1e-9


def test_pitot_q_preferred_over_nominal_speed():
    # logged total-static gives the measured q; it should win over ½ρV²
    prov = ScanProvenance("A2", rho=1.225, speed_ms=25.0,
                          p_static_inf_pa=0.0, p_total_inf_pa=400.0)
    assert abs(prov.dynamic_pressure() - 400.0) < 1e-9


def test_static_reference_subtracted():
    # C_p must be measured from p_inf, not absolute pressure
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    q = 0.5 * 1.225 * 25.0 ** 2
    # build a scan at p_inf = 50 Pa with a tap at p_inf (so C_p must be 0)
    scan, prov = _scan_from_cp({"t0": 0.0}, taps, p_inf=50.0)
    cp = scan.to_cp(prov)
    assert abs(cp.cp["t0"]) < 1e-9


# --------------------------------------------------------------------------- #
#  2. The wing mapping reads as a wing
# --------------------------------------------------------------------------- #
def test_chordwise_sorted_leading_to_trailing():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    xc, cps, ids = cp.chordwise("main", WingSurface.SUCTION)
    assert list(xc) == sorted(xc)                 # LE -> TE
    assert xc[0] == pytest.approx(0.05)
    assert cps[0] == pytest.approx(-2.6, abs=1e-6)  # peak suction near LE


def test_suction_peak_location_and_depth():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    cp_min, x_at = cp.suction_peak("main")
    assert cp_min == pytest.approx(-2.6, abs=1e-6)
    assert x_at == pytest.approx(0.05)


def test_normal_load_coefficient_positive_for_downforce_wing():
    # pressure side above suction side everywhere -> positive C_n (loaded)
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    cn = cp.normal_load_coefficient("main")
    assert math.isfinite(cn) and cn > 0.5


def test_normal_load_nan_without_both_surfaces():
    # only a suction surface instrumented -> cannot integrate the pressure difference
    taps = [TapLocation(f"s{i}", "main", x, surface=WingSurface.SUCTION)
            for i, x in enumerate([0.1, 0.5, 0.9])]
    cp_by = {t.tap_id: -1.0 for t in taps}
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    assert math.isnan(cp.normal_load_coefficient("main"))


# --------------------------------------------------------------------------- #
#  3. Stall is detectable — the thing coefficients can't see
# --------------------------------------------------------------------------- #
def test_healthy_recovery_not_flagged_stalled():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    v = cp.stall_indicator("main")
    assert v.stalled is False
    assert v.recovery_slope > 0.5                 # clearly recovering


def test_flat_plateau_flagged_stalled():
    # peak at LE then a near-constant C_p aft = separated boundary layer
    xc = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
    cp_flat = [-2.4, -1.9, -1.85, -1.84, -1.85, -1.83]
    taps = [TapLocation(f"s{i}", "main", x, surface=WingSurface.SUCTION)
            for i, x in enumerate(xc)]
    cp_by = {t.tap_id: cp_flat[i] for i, t in enumerate(taps)}
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    v = cp.stall_indicator("main")
    assert v.stalled is True
    assert v.recovery_slope < 0.5
    assert "stall" in v.note.lower() or "separat" in v.note.lower()


def test_stall_verdict_refuses_with_too_few_taps():
    taps = [TapLocation("s0", "main", 0.1, surface=WingSurface.SUCTION),
            TapLocation("s1", "main", 0.5, surface=WingSurface.SUCTION)]
    cp_by = {"s0": -2.0, "s1": -1.0}
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    v = cp.stall_indicator("main")
    assert v.stalled is False
    assert "too few" in v.note.lower()


# --------------------------------------------------------------------------- #
#  4. The honesty contract — holes are holes
# --------------------------------------------------------------------------- #
def test_uncalibrated_channel_is_a_hole():
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    cal = {"t0": TapCalibration(sensitivity_pa_per_v=1000.0, is_calibrated=False)}
    scan = RawPressureScan(np.full((10, 1), 1.0), taps, cal)
    cp = scan.to_cp(ScanProvenance("A2", speed_ms=25.0, sample_seconds=8.0))
    assert math.isnan(cp.cp["t0"])
    assert cp.valid_taps() == []


def test_railed_transducer_is_a_hole():
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    cal = {"t0": TapCalibration(sensitivity_pa_per_v=1000.0, is_calibrated=True,
                                saturation_v=10.0)}
    # every sample at the rail -> no usable pressure
    scan = RawPressureScan(np.full((10, 1), 10.0), taps, cal)
    cp = scan.to_cp(ScanProvenance("A2", speed_ms=25.0, sample_seconds=8.0))
    assert math.isnan(cp.cp["t0"])


def test_railed_samples_excluded_from_mean_not_corrupting():
    # one railed sample among good ones must not drag the mean (NaN-safe average)
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    cal = {"t0": TapCalibration(sensitivity_pa_per_v=1000.0, zero_offset_v=0.0,
                                is_calibrated=True, saturation_v=10.0)}
    col = np.array([1.0, 1.0, 1.0, 10.0])        # last sample railed
    scan = RawPressureScan(col[:, None], taps, cal)
    pres = scan.tap_pressures_pa()
    assert pres["t0"] == pytest.approx(1000.0)   # mean of the 3 good (1V*1000), rail dropped


def test_volts_taps_mismatch_rejected():
    taps = [TapLocation("a"), TapLocation("b")]
    with pytest.raises(ValueError):
        RawPressureScan(np.zeros((5, 3)), taps, {})


def test_duplicate_tap_ids_rejected():
    taps = [TapLocation("dup"), TapLocation("dup")]
    with pytest.raises(ValueError):
        RawPressureScan(np.zeros((5, 2)), taps, {})


# --------------------------------------------------------------------------- #
#  5. The RMSE correlation
# --------------------------------------------------------------------------- #
def test_perfect_twin_rmse_zero():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    cfd = CFDSurfaceCp.from_pairs(
        {t.tap_id: cp.cp[t.tap_id] for t in taps},
        backend="OpenFOAM", turbulence_model="kOmegaSST")
    rep = correlate_cp(cp, cfd)
    assert rep.n_paired == len(taps)
    assert rep.rmse == pytest.approx(0.0, abs=1e-9)
    assert rep.within_tol is True
    assert "MATCHED" in rep.summary


def test_known_offset_gives_that_rmse():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    # add a uniform +0.07 C_p to every CFD tap -> RMSE and bias are exactly 0.07
    cfd = CFDSurfaceCp.from_pairs(
        {t.tap_id: cp.cp[t.tap_id] + 0.07 for t in taps}, backend="Star-CCM+")
    rep = correlate_cp(cp, cfd)
    assert rep.rmse == pytest.approx(0.07, abs=1e-9)
    assert rep.bias == pytest.approx(0.07, abs=1e-9)
    assert rep.max_abs_residual == pytest.approx(0.07, abs=1e-9)


def test_residual_sign_is_cfd_minus_phys():
    taps = [TapLocation("t0", "main", 0.2, surface=WingSurface.SUCTION)]
    scan, prov = _scan_from_cp({"t0": -2.0}, taps)
    cp = scan.to_cp(prov)
    cfd = CFDSurfaceCp.from_pairs({"t0": -1.5})   # CFD less suction than phys
    rep = correlate_cp(cp, cfd)
    assert rep.residuals[0].residual == pytest.approx(0.5)  # -1.5 - (-2.0)


def test_unpaired_cfd_tap_is_a_hole_not_snapped():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    pairs = {t.tap_id: cp.cp[t.tap_id] for t in taps}
    pairs["ghost"] = -0.9                         # CFD tap with no physical tap
    rep = correlate_cp(cp, CFDSurfaceCp.from_pairs(pairs))
    assert rep.n_paired == len(taps)              # ghost excluded
    assert rep.n_unpaired == 1
    assert any(not r.paired and r.tap.tap_id == "ghost" for r in rep.residuals)


def test_measured_hole_excluded_from_rmse():
    taps, cp_by = _healthy_main_wing()
    # knock one channel uncalibrated -> measured hole
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    bad = taps[0].tap_id
    cp.cp[bad] = float("nan")
    cfd = CFDSurfaceCp.from_pairs({t.tap_id: 0.0 for t in taps})
    rep = correlate_cp(cp, cfd)
    assert rep.n_paired == len(taps) - 1
    assert any(not r.paired and r.tap.tap_id == bad for r in rep.residuals)


def test_low_coverage_refuses_to_certify_match():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    # CFD only covers 2 of the 12 taps -> coverage 1/6, below the 0.6 floor
    cfd = CFDSurfaceCp.from_pairs(
        {taps[0].tap_id: cp.cp[taps[0].tap_id],
         taps[1].tap_id: cp.cp[taps[1].tap_id]})
    rep = correlate_cp(cp, cfd)
    assert rep.rmse == pytest.approx(0.0, abs=1e-9)   # the two it has are perfect
    assert rep.within_tol is False                    # but coverage too thin to certify
    assert rep.coverage < DEFAULT_CP_TOL["min_coverage"]


def test_nothing_paired_is_honest():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    rep = correlate_cp(cp, CFDSurfaceCp.from_pairs({"unrelated": 0.0}))
    assert rep.ok is False
    assert rep.n_paired == 0
    assert "Nothing could be compared" in rep.summary


# --------------------------------------------------------------------------- #
#  6. Provenance surfaces the honesty warnings
# --------------------------------------------------------------------------- #
def test_short_averaging_window_warns():
    prov = ScanProvenance("A2", speed_ms=25.0, sample_seconds=1.0)
    assert prov.averaging_ok() is False
    assert "averaging" in prov.status().lower()


def test_fixed_floor_warns():
    from suspension.aero import GroundState
    prov = ScanProvenance("A2", speed_ms=25.0, sample_seconds=8.0,
                          ground_state=GroundState.FIXED_FLOOR)
    assert "floor" in prov.status().lower()
    assert "WARNING" in prov.status()


def test_uncorrected_blockage_warns():
    prov = ScanProvenance("A2", speed_ms=25.0, sample_seconds=8.0,
                          blockage_corrected=False)
    assert "blockage" in prov.status().lower()


def test_report_as_dict_roundtrips():
    taps, cp_by = _healthy_main_wing()
    scan, prov = _scan_from_cp(cp_by, taps)
    cp = scan.to_cp(prov)
    cfd = CFDSurfaceCp.from_pairs({t.tap_id: cp.cp[t.tap_id] for t in taps})
    d = correlate_cp(cp, cfd).as_dict()
    assert d["n_paired"] == len(taps)
    assert "residuals" in d and len(d["residuals"]) == len(taps)
    assert d["rmse"] == pytest.approx(0.0, abs=1e-9)
