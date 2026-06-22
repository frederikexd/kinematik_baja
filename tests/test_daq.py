# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for suspension.aero.daq — the live acquisition front end.

These exercise the contract the module promises: the balance interaction matrix
actually decouples cross-talk (and refuses when it can't), the vibration filter
removes the fan tone and reports what it removed, time-averaging beats noise down,
holes stay holes, and the Virtual Instrument streams clean raw forces + a
RawPressureScan that the existing to_cp reduction reads.
"""

import math

import numpy as np
import pytest

from suspension.aero import (
    BalanceAxis, BalanceCalibration, ScannerVendor, PressureScannerSpec,
    ForceBalanceSpec, DAQChassis, VibrationFilter, AcquisitionSpec,
    BalanceReading, DAQUnavailable, OfflineDAQ, SyntheticDAQ, VirtualInstrument,
    TapLocation, TapCalibration, WingSurface, RawPressureScan,
    ChannelFilter, StreamingVibrationFilter,
)


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #
def _cross_coupled_cal(serial="B6"):
    """A 6x6 cal with real off-diagonal cross-talk (drag bleeds into lift/pitch)."""
    m = np.eye(6) * np.array([200., 200., 400., 50., 50., 50.])  # N or N·m per V
    m[2, 0] = 30.0     # drag bridge leaks into lift load
    m[4, 0] = 8.0      # drag bridge leaks into pitch moment
    m[2, 1] = 12.0     # side bridge leaks into lift load
    return BalanceCalibration(matrix=m, is_calibrated=True, serial=serial,
                              saturation_v=10.0)


def _two_tap_scanner():
    taps = [
        TapLocation("t0", element="main", x_over_c=0.10, surface=WingSurface.SUCTION),
        TapLocation("t1", element="main", x_over_c=0.50, surface=WingSurface.SUCTION),
    ]
    cals = {
        "t0": TapCalibration(sensitivity_pa_per_v=1000.0, is_calibrated=True,
                             saturation_v=5.0, serial="s0"),
        "t1": TapCalibration(sensitivity_pa_per_v=1000.0, is_calibrated=True,
                             saturation_v=5.0, serial="s1"),
    }
    return PressureScannerSpec(vendor=ScannerVendor.SCANIVALVE, port_taps=taps,
                               calibrations=cals, serial="ZOC1")


def _vi(true_loads, true_pressures, *, cal=None, fan_hz=137.0, fan_amp=0.05,
        turb=0.005, rate=2000.0, seed=1):
    cal = cal or _cross_coupled_cal()
    chassis = DAQChassis(sample_rate_hz=rate)
    balance = ForceBalanceSpec(calibration=cal, serial="B6")
    scanner = _two_tap_scanner()
    backend = SyntheticDAQ(true_loads, true_pressures, fan_hz=fan_hz,
                           fan_amp_v=fan_amp, turb_v=turb, balance_cal=cal, seed=seed)
    return VirtualInstrument(facility="A2", chassis=chassis, balance=balance,
                             scanners=[scanner], backend=backend)


# --------------------------------------------------------------------------- #
#  balance decoupling
# --------------------------------------------------------------------------- #
def test_interaction_matrix_recovers_true_forces_through_crosstalk():
    # known true loads; synthetic backend mixes them through C, VI must un-mix them.
    true = {"Fx": 150.0, "Fy": -40.0, "Fz": -800.0, "Mx": 0.0, "My": 5.0, "Mz": 0.0}
    vi = _vi(true, {"t0": -1500.0, "t1": -1000.0})
    spec = AcquisitionSpec(seconds=12.0, speed_ms=25.0,
                           vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137.0))
    r = vi.acquire_forces(spec)
    assert r.is_usable()
    assert r.Fx == pytest.approx(150.0, abs=3.0)
    assert r.Fy == pytest.approx(-40.0, abs=3.0)
    assert r.Fz == pytest.approx(-800.0, abs=5.0)
    assert r.downforce_N() == pytest.approx(800.0, abs=5.0)


def test_diagonal_only_cal_gets_lift_wrong_when_crosstalk_present():
    # Same physics, but decouple with a diagonal-only cal: the drag->lift cross-talk
    # is NOT removed, so Fz is biased. Proves the off-diagonal terms matter.
    true = {"Fx": 150.0, "Fy": 0.0, "Fz": -800.0, "Mx": 0.0, "My": 0.0, "Mz": 0.0}
    real = _cross_coupled_cal()
    diag = BalanceCalibration.identity([200., 200., 400., 50., 50., 50.],
                                       saturation_v=10.0)
    chassis = DAQChassis(2000.0)
    backend = SyntheticDAQ(true, {}, fan_amp_v=0.0, turb_v=0.0, balance_cal=real, seed=0)
    vi = VirtualInstrument(facility="A2", chassis=chassis,
                           balance=ForceBalanceSpec(calibration=diag), backend=backend)
    spec = AcquisitionSpec(seconds=5.0)
    r = vi.acquire_forces(spec)
    # diagonal decouple leaves a clear Fz error; true cal would nail it
    assert abs(r.Fz - (-800.0)) > 5.0


def test_uncalibrated_balance_is_a_hole():
    true = {"Fx": 100.0, "Fz": -500.0}
    cal = BalanceCalibration(is_calibrated=False)
    chassis = DAQChassis(2000.0)
    backend = SyntheticDAQ(true, {}, balance_cal=_cross_coupled_cal())
    vi = VirtualInstrument(facility="A2", chassis=chassis,
                           balance=ForceBalanceSpec(calibration=cal), backend=backend)
    r = vi.acquire_forces(AcquisitionSpec(seconds=5.0))
    assert math.isnan(r.Fx) and math.isnan(r.Fz)
    assert not r.is_usable()


def test_singular_interaction_matrix_refuses():
    m = np.ones((6, 6))            # rank 1, singular
    cal = BalanceCalibration(matrix=m, is_calibrated=True)
    assert not cal.usable()
    out = cal.decouple(np.ones((10, 6)))
    assert np.all(np.isnan(out))


def test_railed_bridge_sample_nans_whole_decouple_row():
    cal = _cross_coupled_cal()         # rail at 10 V
    volts = np.zeros((4, 6))
    volts[2, 0] = 10.0                 # one railed sample on the drag bridge
    out = cal.decouple(volts)
    assert np.all(np.isnan(out[2]))    # the whole row is a hole (matrix mixes axes)
    assert np.all(np.isfinite(out[0]))


def test_downforce_is_negative_fz():
    r = BalanceReading(forces_N={"Fx": 100., "Fy": 0., "Fz": -750.},
                       stderr_N={}, n_used=10)
    assert r.downforce_N() == pytest.approx(750.0)


# --------------------------------------------------------------------------- #
#  vibration filter
# --------------------------------------------------------------------------- #
def test_fan_tone_removed_and_reported():
    fs = 2000.0
    n = int(fs * 8)
    t = np.arange(n) / fs
    signal = 3.0 + 0.8 * np.sin(2 * np.pi * 137.0 * t)        # DC + fan tone
    filt = VibrationFilter(fs, fan_blade_pass_hz=137.0, aero_cutoff_hz=30.0)
    clean, rep = filt.apply(signal)
    # the tone is gone: cleaned variance is a tiny fraction of the raw tone variance
    assert np.nanvar(clean) < 0.05 * np.nanvar(signal)
    assert np.nanmean(clean) == pytest.approx(3.0, abs=0.05)   # DC preserved
    assert 137.0 in rep.harmonics_notched
    assert rep.variance_removed_frac > 0.9
    assert rep.resolved


def test_filter_preserves_aero_signal_below_cutoff():
    fs = 2000.0
    n = int(fs * 8)
    t = np.arange(n) / fs
    aero = 2.0 * np.sin(2 * np.pi * 3.0 * t)                  # 3 Hz aero unsteadiness
    contaminated = aero + 0.5 * np.sin(2 * np.pi * 137.0 * t)
    filt = VibrationFilter(fs, fan_blade_pass_hz=137.0, aero_cutoff_hz=30.0)
    clean, _ = filt.apply(contaminated)
    # the 3 Hz content survives (below cutoff), the 137 Hz does not
    assert np.nanvar(clean) == pytest.approx(np.nanvar(aero), rel=0.1)


def test_filter_nan_samples_preserved():
    fs = 2000.0
    sig = np.sin(2 * np.pi * 137.0 * np.arange(2000) / fs)
    sig[10] = np.nan
    clean, _ = VibrationFilter(fs, fan_blade_pass_hz=137.0).apply(sig)
    assert math.isnan(clean[10])
    assert np.isfinite(clean[0])


def test_filtering_reduces_averaging_bias_from_fan_tone():
    # A window that is NOT an integer number of fan periods: the raw mean carries a
    # tone residual; filtering removes it so the mean lands on the true DC.
    fs = 2000.0
    n = 1234                                  # deliberately not aligned to 137 Hz
    t = np.arange(n) / fs
    true_dc = 5.0
    sig = true_dc + 1.0 * np.sin(2 * np.pi * 137.0 * t + 0.6)
    raw_bias = abs(np.mean(sig) - true_dc)
    clean, _ = VibrationFilter(fs, fan_blade_pass_hz=137.0).apply(sig)
    filt_bias = abs(np.nanmean(clean) - true_dc)
    assert filt_bias < raw_bias
    assert filt_bias < 0.05


# --------------------------------------------------------------------------- #
#  pressures through the VI -> existing RawPressureScan / to_cp
# --------------------------------------------------------------------------- #
def test_vi_pressures_round_trip_through_to_cp():
    # true tap pressures -> volts (synthetic) -> VI scan -> to_cp recovers C_p.
    rho, V = 1.225, 25.0
    q = 0.5 * rho * V * V
    # choose pressures that give clean round numbers of C_p
    p0, p1 = -2.0 * q, -1.0 * q
    vi = _vi({"Fz": -500.0}, {"t0": p0, "t1": p1}, turb=0.002)
    spec = AcquisitionSpec(seconds=10.0, speed_ms=V, rho=rho,
                           vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137.0))
    scan = vi.acquire_pressures(spec)
    assert isinstance(scan, RawPressureScan)
    cp = scan.to_cp(vi.scan_provenance(spec))
    assert cp.cp["t0"] == pytest.approx(-2.0, abs=0.05)
    assert cp.cp["t1"] == pytest.approx(-1.0, abs=0.05)
    assert cp.coverage() == 1.0


def test_vi_acquire_returns_both_force_and_scan():
    vi = _vi({"Fx": 120.0, "Fz": -600.0}, {"t0": -1000.0, "t1": -500.0})
    spec = AcquisitionSpec(seconds=8.0, speed_ms=25.0,
                           vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137.0))
    forces, scan = vi.acquire(spec)
    assert isinstance(forces, BalanceReading)
    assert isinstance(scan, RawPressureScan)
    assert forces.Fx == pytest.approx(120.0, abs=4.0)


def test_stderr_shrinks_with_longer_window():
    true = {"Fz": -700.0}
    short = _vi(true, {}, turb=0.02, seed=3)
    long = _vi(true, {}, turb=0.02, seed=3)
    rs = short.acquire_forces(AcquisitionSpec(seconds=2.0))
    rl = long.acquire_forces(AcquisitionSpec(seconds=20.0))
    assert rl.stderr_N["Fz"] < rs.stderr_N["Fz"]


# --------------------------------------------------------------------------- #
#  provenance & honesty
# --------------------------------------------------------------------------- #
def test_synthetic_provenance_flagged():
    vi = _vi({"Fz": -500.0}, {})
    r = vi.acquire_forces(AcquisitionSpec(seconds=10.0,
                          vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137.0)))
    assert r.provenance.synthetic is True
    assert "SYNTHETIC" in r.provenance.status()


def test_unfiltered_acquisition_warns():
    vi = _vi({"Fz": -500.0}, {})
    r = vi.acquire_forces(AcquisitionSpec(seconds=10.0))   # no vibration filter
    assert r.provenance.filtered is False
    assert "UNFILTERED" in r.provenance.status()


def test_short_window_warns():
    vi = _vi({"Fz": -500.0}, {})
    r = vi.acquire_forces(AcquisitionSpec(seconds=1.0,
                          vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137.0)))
    assert r.provenance.averaging_ok() is False
    assert "short" in r.provenance.status()


# --------------------------------------------------------------------------- #
#  offline backend refuses to fabricate
# --------------------------------------------------------------------------- #
def test_offline_backend_raises_not_fabricates():
    vi = VirtualInstrument(facility="A2", chassis=DAQChassis(2000.0),
                           balance=ForceBalanceSpec(calibration=_cross_coupled_cal()),
                           scanners=[_two_tap_scanner()], backend=OfflineDAQ())
    with pytest.raises(DAQUnavailable):
        vi.acquire_forces(AcquisitionSpec(seconds=5.0))
    with pytest.raises(DAQUnavailable):
        vi.acquire_pressures(AcquisitionSpec(seconds=5.0))


def test_vi_without_balance_refuses_forces():
    vi = VirtualInstrument(facility="A2", chassis=DAQChassis(2000.0),
                           scanners=[_two_tap_scanner()],
                           backend=SyntheticDAQ({}, {"t0": -1000.0, "t1": -500.0}))
    with pytest.raises(DAQUnavailable):
        vi.acquire_forces(AcquisitionSpec(seconds=5.0))


def test_chassis_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        DAQChassis(sample_rate_hz=0.0)


def test_balance_stream_must_be_six_channels():
    cal = _cross_coupled_cal()
    with pytest.raises(ValueError):
        cal.decouple(np.zeros((10, 4)))


# --------------------------------------------------------------------------- #
#  streaming biquad cascade — the real-time twin of VibrationFilter
# --------------------------------------------------------------------------- #
def test_streaming_filter_satisfies_channel_filter_protocol():
    f = StreamingVibrationFilter(2000.0, fan_blade_pass_hz=137.0)
    assert isinstance(f, ChannelFilter)
    assert isinstance(VibrationFilter(2000.0, fan_blade_pass_hz=137.0), ChannelFilter)


def test_streaming_filter_removes_fan_tone():
    fs = 2000.0
    n = int(fs * 8)
    t = np.arange(n) / fs
    signal = 3.0 + 0.8 * np.sin(2 * np.pi * 137.0 * t)
    clean, rep = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0,
                                          aero_cutoff_hz=30.0).apply(signal)
    w = int(0.25 * fs)
    assert np.nanvar(clean[w:]) < 0.05 * np.nanvar(signal)   # tone gone in steady state
    assert np.nanmean(clean[w:]) == pytest.approx(3.0, abs=0.05)   # DC preserved
    assert 137.0 in rep.harmonics_notched
    assert "phase lag" in rep.note


def test_streaming_filter_preserves_aero_band():
    fs = 2000.0
    n = int(fs * 8)
    t = np.arange(n) / fs
    aero = 2.0 * np.sin(2 * np.pi * 3.0 * t)
    contaminated = aero + 0.5 * np.sin(2 * np.pi * 137.0 * t)
    clean, _ = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0,
                                        aero_cutoff_hz=30.0).apply(contaminated)
    w = int(0.5 * fs)
    # 3 Hz content survives (amplitude, not just variance, within a few %)
    assert (np.nanmax(clean[w:]) - np.nanmin(clean[w:])) == pytest.approx(4.0, rel=0.1)


def test_streaming_filter_reduces_averaging_bias():
    fs = 2000.0
    n = 4321
    t = np.arange(n) / fs
    true_dc = 5.0
    sig = true_dc + 1.0 * np.sin(2 * np.pi * 137.0 * t + 0.6)
    raw_bias = abs(np.mean(sig) - true_dc)
    clean, _ = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0).apply(sig)
    w = int(0.25 * fs)
    filt_bias = abs(np.nanmean(clean[w:]) - true_dc)
    assert filt_bias < raw_bias
    assert filt_bias < 0.05


def test_streaming_primitives_match_apply():
    # process_sample/process_block must reproduce apply() exactly (same state machine)
    fs = 2000.0
    t = np.arange(int(fs * 2)) / fs
    sig = 4.0 + 0.5 * np.sin(2 * np.pi * 137.0 * t)
    f1 = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0)
    clean_apply, _ = f1.apply(sig)
    # replicate apply's reset (DC-preload to window mean) then stream the block
    f2 = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0)
    f2.reset(float(np.nanmean(sig)))
    clean_stream = f2.process_block(sig)
    assert np.allclose(clean_apply, clean_stream, atol=1e-9)


def test_streaming_block_preserves_state_across_calls():
    # feeding two halves separately must equal feeding the whole (continuous state)
    fs = 2000.0
    t = np.arange(2000) / fs
    sig = 4.0 + 0.5 * np.sin(2 * np.pi * 137.0 * t)
    whole = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0)
    whole.reset(4.0)
    y_whole = whole.process_block(sig)
    split = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0)
    split.reset(4.0)
    y_split = np.concatenate([split.process_block(sig[:1000]),
                              split.process_block(sig[1000:])])
    assert np.allclose(y_whole, y_split, atol=1e-12)


def test_streaming_filter_nan_gap_resettled_not_invented():
    fs = 2000.0
    sig = 3.0 + 0.5 * np.sin(2 * np.pi * 137.0 * np.arange(2000) / fs)
    sig[500:510] = np.nan          # a railed gap
    clean, _ = StreamingVibrationFilter(fs, fan_blade_pass_hz=137.0).apply(sig)
    assert np.all(np.isnan(clean[500:510]))    # holes preserved, never filled
    assert np.isfinite(clean[520])             # recovers after the gap


def test_streaming_filter_drops_into_vi_unchanged():
    # the whole point: swap the filter type, the VI code path is identical.
    true = {"Fx": 150.0, "Fy": -40.0, "Fz": -800.0, "Mx": 0.0, "My": 5.0, "Mz": 0.0}
    vi = _vi(true, {"t0": -1500.0, "t1": -1000.0})
    spec = AcquisitionSpec(seconds=12.0, speed_ms=25.0,
                           vibration=StreamingVibrationFilter(2000.0,
                                                              fan_blade_pass_hz=137.0))
    r = vi.acquire_forces(spec)
    assert r.is_usable()
    assert r.Fx == pytest.approx(150.0, abs=4.0)
    assert r.Fz == pytest.approx(-800.0, abs=6.0)
    assert r.provenance.filtered is True


def test_streaming_filter_rejects_bad_rate():
    with pytest.raises(ValueError):
        StreamingVibrationFilter(0.0, fan_blade_pass_hz=137.0)
