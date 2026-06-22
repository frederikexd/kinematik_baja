# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Data acquisition — the live SEAM between a wind-tunnel run and the raw numbers
KinematiK already knows how to read.

WHY THIS MODULE EXISTS (read this before using it)
---------------------------------------------------
`pressure_tap.py` begins one step too late on purpose. Its docstring says it
"does not talk to a DAQ, does not own a transducer" and starts from a finished
`RawPressureScan` — a matrix of volts that has somehow already appeared in memory.
`windtunnel.py` begins later still, from a finished C_l/C_d. Neither answers the
question a real test engineer faces first: the car is bolted to a six-component
under-floor force balance and skinned with hundreds of static pressure taps plumbed
into electronic pressure scanners (Scanivalve ZOC/MPS, Chell nanoDAQ), all of it
streamed through a high-speed DAQ chassis at several kHz. What comes off that
hardware is NOT a clean force and NOT a clean C_p. It is:

    * six bridge-voltage channels from the balance, mechanically CROSS-COUPLED — a
      pure drag load leaks into the lift and side channels through the flexure, so a
      raw F_z channel is not F_z until the balance interaction matrix is applied;
    * hundreds of transducer-voltage channels from the scanners, each on its own
      calibration; and
    * riding on ALL of it, the wind-tunnel fan's blade-pass tone and the model/
      sting structural resonances — a periodic vibration that a naive average does
      not remove, because the run is never quite an integer number of fan periods
      long, so the tone leaks into the mean as a bias.

This module owns that front end. It is the analogue of PIV's `PIVRig`/`PIVProcessor`
seam, applied to forces and pressures: a typed description of the chassis, the
balance, and the scanners (the SEAM a real `nidaqmx`/Scanivalve/Chell driver plugs
into) and the signal MATH (interaction-matrix decoupling, fan-tone notch + anti-
alias / vibration low-pass, and honest time-averaging) that turns the raw streams
into the clean, time-averaged `BalanceReading` (F_x, F_y, F_z) and the
`RawPressureScan` (P_static per tap) that the rest of the aero package consumes.

A "virtual instrument" (VI) in the LabVIEW/test-floor sense is exactly the small
object at the end of this file: `VirtualInstrument` binds one balance + one or more
scanners to a `DAQChassis`, runs an `AcquisitionSpec`, and streams reduced readings.
Swap the synthetic backend for a real driver and the SAME VI runs the real tunnel.

THE HONESTY CONTRACT (identical discipline to cfd.py / windtunnel.py / pressure_tap.py)
--------------------------------------------------------------------------------------
A reduced force looks authoritative the instant it has a unit on it, and a quietly
mis-decoupled or under-filtered force lies with the full authority of a load cell.
So, by construction:

  * A balance with no interaction matrix, or a singular one, does not produce a
    force — it produces a HOLE (NaN), never a raw channel scaled by a guess. The
    interaction matrix is recorded with the reading; a force decoupled by an
    identity matrix is a different measurement from one decoupled by a calibrated
    matrix, and the provenance says so.
  * The vibration filter REPORTS what it removed — which tones, how much power, how
    much of the band — it never silently reshapes the signal. A notch placed where
    no tone exists is logged as removing ~nothing; a run too short to resolve the
    fan tone is flagged, because you cannot notch a frequency you cannot see.
  * A channel that railed, a chassis that dropped samples, or a window too short to
    average down turbulence/vibration is a hole or a warning, never filled.
  * No backend invents a sample. `OfflineDAQ` raises `DAQUnavailable` rather than
    fabricate a stream; the synthetic backend is clearly labelled synthetic in its
    provenance so a number it produced can never be mistaken for a measured one.

DELIBERATE NON-GOALS: this module does not own a vendor driver, does not solve
Navier-Stokes, and does not non-dimensionalise (that is `pressure_tap.to_cp` and the
coefficient reductions downstream). It owns the seam and the signal processing, so
the whole front end is writable and testable now, on synthetic streams, against no
hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Sequence, Protocol, runtime_checkable

import numpy as np

from .cfd import Attitude
from .windtunnel import GroundState
from .pressure_tap import (
    TapLocation, TapCalibration, RawPressureScan, ScanProvenance,
)


# --------------------------------------------------------------------------- #
#  The six axes of an under-floor multi-axis balance
# --------------------------------------------------------------------------- #
class BalanceAxis(str, Enum):
    """
    The six components a full external balance resolves. The three forces are what
    this module is asked for (raw F_x/F_y/F_z); the three moments come for free from
    the same flexure and are carried so a downstream consumer can place the centre
    of pressure / compute aero balance without a second instrument.

    Sign convention (SAE-ish, model axes, wind from +x toward -x over the car):
      FX  drag,  +x = downstream (drag positive)
      FY  side,  +y = to the car's left
      FZ  lift,  +z = up;  DOWNFORCE is therefore NEGATIVE FZ
    """
    FX = "Fx"   # drag
    FY = "Fy"   # side force
    FZ = "Fz"   # lift  (downforce = -Fz)
    MX = "Mx"   # roll moment
    MY = "My"   # pitch moment
    MZ = "Mz"   # yaw moment

    @staticmethod
    def forces() -> tuple["BalanceAxis", "BalanceAxis", "BalanceAxis"]:
        return (BalanceAxis.FX, BalanceAxis.FY, BalanceAxis.FZ)

    @staticmethod
    def all_six() -> list["BalanceAxis"]:
        return [BalanceAxis.FX, BalanceAxis.FY, BalanceAxis.FZ,
                BalanceAxis.MX, BalanceAxis.MY, BalanceAxis.MZ]


# --------------------------------------------------------------------------- #
#  Balance calibration — the interaction matrix that turns bridge volts into force
# --------------------------------------------------------------------------- #
#
# A multi-component balance is a single elastic element with six strain-gauge
# bridges on it. Load it with a PURE drag force and you do not see voltage on the
# drag bridge alone — the flexure deflects in a way that bleeds a little signal into
# the lift and pitch bridges too. So the map from the six bridge voltages to the six
# loads is a full 6x6 matrix C (the "interaction" or "calibration" matrix), found by
# loading the balance one component at a time on a calibration rig:
#
#       load_vector (6,)  =  C (6x6)  @  (bridge_volts - zero_volts) (6,)
#
# The off-diagonal terms of C are the cross-talk. Using only the diagonal (treating
# each bridge as if it read its own axis) is the single most common way a student rig
# reports a confidently wrong side force. This dataclass holds C and the wind-off
# zero, applies it honestly, and refuses (NaN) when C is missing or singular.
@dataclass
class BalanceCalibration:
    """
    The 6x6 interaction matrix and wind-off zero that turn raw balance bridge
    voltages into decoupled loads, in the axis order `BalanceAxis.all_six()`
    (Fx, Fy, Fz, Mx, My, Mz).

    `matrix` maps (volts - zero_volts) -> loads. Forces come out in Newtons, moments
    in Newton-metres, given a `matrix` in those units per volt. `is_calibrated` is
    False until a real matrix is supplied: an uncalibrated balance does not produce
    forces, it produces holes (mirrors TapCalibration.is_calibrated exactly). A
    singular matrix is treated as uncalibrated — you cannot decouple through it.
    """
    matrix: Optional[np.ndarray] = None          # (6, 6), loads per (volt - zero)
    zero_volts: Optional[np.ndarray] = None      # (6,), wind-off tare per bridge
    is_calibrated: bool = False
    saturation_v: Optional[float] = None         # bridge rail; |V| >= this is garbage
    serial: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.matrix is not None:
            self.matrix = np.asarray(self.matrix, dtype=float)
            if self.matrix.shape != (6, 6):
                raise ValueError("balance interaction matrix must be 6x6 "
                                 "(Fx,Fy,Fz,Mx,My,Mz)")
        if self.zero_volts is None:
            self.zero_volts = np.zeros(6, dtype=float)
        else:
            self.zero_volts = np.asarray(self.zero_volts, dtype=float).reshape(-1)
            if self.zero_volts.shape != (6,):
                raise ValueError("zero_volts must have 6 entries (one per bridge)")

    @staticmethod
    def identity(sensitivities_per_v: Sequence[float],
                 **kw) -> "BalanceCalibration":
        """
        A DIAGONAL-only calibration: each bridge scaled by its own sensitivity, zero
        cross-talk assumed. This is the deliberately naive balance — useful as a
        baseline to show how much the off-diagonal terms matter, and honest about
        what it is (`notes` records that cross-talk was assumed zero). Real balances
        are never truly diagonal.
        """
        s = np.asarray(sensitivities_per_v, dtype=float).reshape(-1)
        if s.shape != (6,):
            raise ValueError("need 6 sensitivities (Fx,Fy,Fz,Mx,My,Mz)")
        m = np.diag(s)
        notes = (kw.pop("notes", "") +
                 " [diagonal-only: cross-talk assumed zero — baseline, not a real "
                 "balance cal]").strip()
        return BalanceCalibration(matrix=m, is_calibrated=True, notes=notes, **kw)

    def _well_conditioned(self) -> bool:
        if self.matrix is None:
            return False
        try:
            cond = np.linalg.cond(self.matrix)
        except np.linalg.LinAlgError:
            return False
        return math.isfinite(cond) and cond < 1e12

    def usable(self) -> bool:
        return bool(self.is_calibrated and self.matrix is not None
                    and self._well_conditioned())

    def decouple(self, bridge_volts: np.ndarray) -> np.ndarray:
        """
        Apply the interaction matrix to a block of samples. `bridge_volts` is
        (n_samples, 6); returns (n_samples, 6) of decoupled loads in the axis order
        Fx,Fy,Fz,Mx,My,Mz. Samples at/over the rail on ANY bridge are NaN across all
        six components for that sample (a railed bridge corrupts the whole decouple,
        not just its own axis — the matrix mixes them). An unusable calibration
        returns all-NaN: an honest hole, never a raw channel passed through.
        """
        v = np.asarray(bridge_volts, dtype=float)
        if v.ndim == 1:
            v = v[None, :]
        if v.shape[1] != 6:
            raise ValueError("balance stream must have 6 bridge channels "
                             "(Fx,Fy,Fz,Mx,My,Mz order)")
        if not self.usable():
            return np.full(v.shape, np.nan)
        loads = (v - self.zero_volts) @ self.matrix.T
        if self.saturation_v is not None:
            railed = np.any(np.abs(v) >= abs(self.saturation_v) - 1e-12, axis=1)
            loads[railed, :] = np.nan
        return loads

    def status(self) -> str:
        if not self.is_calibrated or self.matrix is None:
            return (f"balance {self.serial or '?'}: UNCALIBRATED — no interaction "
                    "matrix; bridge volts are not forces and reduce to holes")
        if not self._well_conditioned():
            return (f"balance {self.serial or '?'}: SINGULAR interaction matrix "
                    f"(cond={np.linalg.cond(self.matrix):.2e}) — cannot decouple; holes")
        offdiag = self.matrix.copy()
        np.fill_diagonal(offdiag, 0.0)
        diag = np.abs(np.diag(self.matrix))
        worst = float(np.max(np.abs(offdiag)) / (np.max(diag) + 1e-30)) * 100.0
        rail = (f", rail ±{abs(self.saturation_v):g} V"
                if self.saturation_v is not None else "")
        return (f"balance {self.serial or '?'}: 6x6 cal, cond="
                f"{np.linalg.cond(self.matrix):.1f}, max cross-talk {worst:.0f}% "
                f"of full-scale{rail}")


# --------------------------------------------------------------------------- #
#  Hardware description — the SEAM a real driver plugs into
# --------------------------------------------------------------------------- #
class ScannerVendor(str, Enum):
    """Electronic pressure scanner families a tap harness plumbs into."""
    SCANIVALVE = "scanivalve"      # e.g. ZOC33, MPS4264
    CHELL = "chell"                # e.g. nanoDAQ
    GENERIC = "generic"


@dataclass
class PressureScannerSpec:
    """
    One electronic pressure scanner: a box of N transducer channels, each plumbed to
    one surface tap, sampled together. This is description, not a driver — it records
    what the hardware IS (vendor, channel count, full-scale range, the tap each port
    maps to, the per-port calibration) so a `nidaqmx`/Scanivalve TCP/Chell backend can
    fill the stream, and so the produced `RawPressureScan` knows which column is which
    tap. `port_taps[i]` is the tap on channel i; `calibrations[tap_id]` its cal.
    """
    vendor: ScannerVendor
    port_taps: Sequence[TapLocation]
    calibrations: dict                          # tap_id -> TapCalibration
    full_scale_pa: float = 6895.0               # ±1 psi default; sets transducer rail
    serial: str = ""

    @property
    def n_channels(self) -> int:
        return len(self.port_taps)

    def status(self) -> str:
        cal = sum(1 for t in self.port_taps
                  if self.calibrations.get(t.tap_id)
                  and self.calibrations[t.tap_id].is_calibrated)
        return (f"{self.vendor.value} scanner {self.serial or '?'}: "
                f"{self.n_channels} ports, {cal} calibrated, "
                f"±{self.full_scale_pa:g} Pa FS")


@dataclass
class ForceBalanceSpec:
    """
    The under-floor multi-axis balance the car is mounted on: six bridge channels and
    the calibration that decouples them. `mount` records how the model sits on it (the
    sting/strut, the centre of the balance relative to the car) for the human; the
    physics that matters is in `calibration`.
    """
    calibration: BalanceCalibration
    mount: str = "under-floor sting"
    serial: str = ""

    def status(self) -> str:
        return f"force balance {self.serial or '?'} [{self.mount}]: {self.calibration.status()}"


@dataclass
class DAQChassis:
    """
    The high-speed DAQ chassis that clocks every channel together. The number that
    matters for everything downstream is `sample_rate_hz` — it sets the Nyquist limit
    (the highest frequency, e.g. a fan blade-pass tone, you can even SEE, let alone
    notch out) and, with the run length, how many samples the average beats the
    turbulence/vibration down over. `n_slots`/`model` are bookkeeping for the human.

    A real chassis (NI cDAQ/PXIe, etc.) is driven by a vendor library; here the
    chassis just declares the clock the backend must honour, so the synthetic and the
    real backend produce streams at the SAME rate the reduction assumes.
    """
    sample_rate_hz: float = 2000.0
    model: str = "synthetic-cDAQ"
    n_slots: int = 4

    def __post_init__(self):
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * self.sample_rate_hz

    def status(self) -> str:
        return (f"{self.model}: {self.sample_rate_hz:g} Hz "
                f"(Nyquist {self.nyquist_hz:g} Hz), {self.n_slots} slots")


# --------------------------------------------------------------------------- #
#  Vibration filtering — remove the fan tone and structural resonance, honestly
# --------------------------------------------------------------------------- #
#
# The wind-tunnel fan turns at a fixed rate; with B blades it stamps a blade-pass
# tone at f_bp = rpm/60 * B and its harmonics onto everything the model is bolted to.
# The model + sting also have structural resonances the flow excites. Both are
# PERIODIC contamination: a naive time-average over a window that is not an integer
# number of tone periods leaves a residual of the tone in the mean — a bias that
# looks like signal. The fix is to NOTCH the known tones (and their first few
# harmonics) and LOW-PASS above the aero band before averaging.
#
# The honesty rule: the filter says what it did. It returns, alongside the cleaned
# signal, how much power sat in each notched band and how much of the total variance
# it removed, so "I filtered it" is auditable rather than magic. A notch at a
# frequency the run is too short to resolve is reported as such — you cannot remove a
# tone you cannot see.
@dataclass
class VibrationFilterReport:
    """What the vibration filter actually did to one channel — auditable, not magic."""
    fan_tone_hz: Optional[float]
    harmonics_notched: list                      # [hz, ...] actually notched
    variance_removed_frac: float                 # fraction of raw variance removed
    tone_power_frac: float                       # fraction of raw variance that was in the notched bands
    resolved: bool                               # was the run long enough to see the lowest tone?
    note: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass
class VibrationFilter:
    """
    A real-time-able digital filter that strips wind-tunnel fan blade-pass tones and
    structural-vibration energy out of a sampled channel before it is averaged.

    Parameters describe the contamination, not the cleaning trick:
      * `fan_blade_pass_hz` — the known fan tone (rpm/60 * blades). If given, it and
        its first `n_harmonics` are notched.
      * `structural_hz` — extra known model/sting resonances to notch.
      * `aero_cutoff_hz` — anything above this is not aerodynamic load on an FSAE
        model at tunnel speed; a low-pass removes it (this is also the anti-alias
        intent if applied before decimation).
      * `notch_q` — notch sharpness; higher = narrower (less signal collateral).

    The implementation is an FFT-domain notch + brick-wall low-pass — exact and
    frame-based. It is the right tool for OFFLINE reduction of a captured window: it
    is zero-phase, has no settling transient, and removes a tone cleanly even when
    the window is a non-integer number of tone periods. Its sibling
    `StreamingVibrationFilter` is the same intent as a causal biquad cascade for a
    REAL-TIME VI that must emit a filtered sample the instant one arrives; both share
    the `apply()` interface so either drops into an `AcquisitionSpec` unchanged. It
    operates per channel and reports what it removed.
    """
    sample_rate_hz: float
    fan_blade_pass_hz: Optional[float] = None
    structural_hz: Sequence[float] = ()
    n_harmonics: int = 3
    aero_cutoff_hz: float = 30.0
    notch_q: float = 30.0

    def _notch_freqs(self) -> list:
        out = []
        if self.fan_blade_pass_hz and self.fan_blade_pass_hz > 0:
            for k in range(1, max(1, self.n_harmonics) + 1):
                out.append(self.fan_blade_pass_hz * k)
        out.extend(float(f) for f in self.structural_hz if f and f > 0)
        # only those we can actually represent below Nyquist
        return [f for f in out if f < 0.5 * self.sample_rate_hz]

    def apply(self, signal: np.ndarray) -> tuple[np.ndarray, VibrationFilterReport]:
        """
        Filter one channel (1D). Returns (clean_signal, report). NaN samples (railed)
        are preserved as NaN in the output and excluded from the variance bookkeeping
        — the filter does not invent values where the transducer gave none. If every
        sample is NaN, the input is returned unchanged with a hole report.
        """
        x = np.asarray(signal, dtype=float).reshape(-1)
        n = x.size
        finite = np.isfinite(x)
        if not finite.any() or n < 4:
            return x.copy(), VibrationFilterReport(
                fan_tone_hz=self.fan_blade_pass_hz, harmonics_notched=[],
                variance_removed_frac=0.0, tone_power_frac=0.0, resolved=False,
                note="too few finite samples to filter — passed through")

        # Work on a mean-filled copy so the FFT is not poisoned by NaN, then restore.
        mean_val = float(np.nanmean(x))
        xf = np.where(finite, x, mean_val)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)
        spec = np.fft.rfft(xf)
        raw_var = float(np.nanvar(x))

        notch_list = self._notch_freqs()
        df = self.sample_rate_hz / n            # frequency resolution of this window
        # can we even resolve the lowest tone? need the window to span >~1 period
        lowest = min([f for f in notch_list], default=None)
        resolved = (lowest is None) or (df <= lowest)

        removed_power = 0.0
        total_power = float(np.sum(np.abs(spec) ** 2)) + 1e-30
        notched = []
        for f0 in notch_list:
            # notch bandwidth from Q; at least one bin wide so a tone between bins still dies
            bw = max(f0 / max(self.notch_q, 1e-6), df)
            band = np.abs(freqs - f0) <= bw
            if band.any():
                removed_power += float(np.sum(np.abs(spec[band]) ** 2))
                spec[band] = 0.0
                notched.append(float(f0))

        # low-pass: kill everything above the aero band (and the anti-alias region)
        lp = freqs > self.aero_cutoff_hz
        removed_power += float(np.sum(np.abs(spec[lp]) ** 2))
        spec[lp] = 0.0

        clean = np.fft.irfft(spec, n=n)
        clean = np.where(finite, clean, np.nan)   # restore holes

        clean_var = float(np.nanvar(clean))
        var_removed = 0.0 if raw_var <= 0 else max(0.0, 1.0 - clean_var / raw_var)
        tone_frac = removed_power / total_power

        note = ""
        if not resolved and lowest is not None:
            note = (f"run too short to resolve {lowest:g} Hz "
                    f"(resolution {df:.2g} Hz) — notch is approximate")
        return clean, VibrationFilterReport(
            fan_tone_hz=self.fan_blade_pass_hz, harmonics_notched=notched,
            variance_removed_frac=var_removed, tone_power_frac=tone_frac,
            resolved=resolved, note=note)


# --------------------------------------------------------------------------- #
#  The filter seam — both filters satisfy this, so either drops into the VI
# --------------------------------------------------------------------------- #
@runtime_checkable
class ChannelFilter(Protocol):
    """
    The one method the VI needs from a vibration filter: reduce one channel and say
    what it did. `VibrationFilter` (offline FFT) and `StreamingVibrationFilter`
    (real-time biquad cascade) both satisfy it, so swapping one for the other is a
    backend choice in the `AcquisitionSpec`, never an API change in the VI.
    """
    def apply(self, signal: np.ndarray) -> tuple[np.ndarray, VibrationFilterReport]: ...


# --------------------------------------------------------------------------- #
#  Streaming biquad cascade — what a REAL-TIME VI actually runs
# --------------------------------------------------------------------------- #
#
# The FFT filter above needs the whole window in hand before it can produce a single
# clean sample — fine for post-run reduction, useless for a live VI that must put a
# filtered value on a strip chart the instant the chassis clocks one in. The real-
# time form is a cascade of second-order sections (biquads): each known tone gets an
# RBJ notch biquad, and the aero-band low-pass is a Butterworth pair, all run in
# Direct-Form-II-transposed so each section carries two state words and updates in a
# handful of multiply-adds per sample. That is exactly what runs on the DAQ's FPGA /
# in the LabVIEW point-by-point loop.
#
# Two honest differences from the FFT form, both inherent to causal real-time
# filtering and both surfaced rather than hidden:
#   * It has a SETTLING TRANSIENT: the state words start at zero, so the first ~few
#     time-constants of output are wrong. `warmup_samples` is reported, and `apply`
#     excludes that transient from the variance bookkeeping so the report describes
#     the steady-state filter, not the turn-on.
#   * It has PHASE LAG (it is causal — it cannot use future samples). This does not
#     affect the time-AVERAGE the balance reading needs, which is the quantity that
#     matters here, but it is noted so nobody reads the streamed waveform as zero-
#     phase. (Run the cascade forward then backward for zero phase if a waveform,
#     not a mean, is the deliverable — at the cost of being offline again.)
def _rbj_notch(f0: float, fs: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """RBJ cookbook notch biquad at f0. Returns (b, a) with a[0] normalised to 1."""
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cosw = math.cos(w0)
    b = np.array([1.0, -2.0 * cosw, 1.0])
    a0 = 1.0 + alpha
    a = np.array([1.0, -2.0 * cosw, 1.0 - alpha])
    return b / a0, a / a0


def _rbj_lowpass(f0: float, fs: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """RBJ cookbook low-pass biquad at f0 with quality q (sets the pair's damping)."""
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cosw = math.cos(w0)
    b1 = 1.0 - cosw
    b = np.array([b1 / 2.0, b1, b1 / 2.0])
    a0 = 1.0 + alpha
    a = np.array([1.0, -2.0 * cosw, 1.0 - alpha])
    return b / a0, a / a0


class _Biquad:
    """One second-order section, Direct-Form-II-transposed, streamed sample by sample."""
    __slots__ = ("b", "a", "z1", "z2")

    def __init__(self, b: np.ndarray, a: np.ndarray):
        self.b = b
        self.a = a
        self.z1 = 0.0
        self.z2 = 0.0

    def reset(self, steady_x: float = 0.0):
        """
        Preload the state to the steady-state for a constant input `steady_x`, so a DC
        load (the bulk of a balance reading) passes through with NO settling transient
        — only the AC contamination has to settle. This is the standard trick that
        makes a causal cascade usable on a short tunnel point.
        """
        # For constant x, steady output y = x * sum(b)/sum(a). Solve the DF2T states.
        sb = float(self.b[0] + self.b[1] + self.b[2])
        sa = float(self.a[0] + self.a[1] + self.a[2])
        y = steady_x * (sb / sa) if sa != 0 else 0.0
        self.z1 = self.b[1] * steady_x - self.a[1] * y + (self.b[2] * steady_x - self.a[2] * y)
        self.z2 = self.b[2] * steady_x - self.a[2] * y

    def step(self, x: float) -> float:
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y


@dataclass
class StreamingVibrationFilter:
    """
    The real-time twin of `VibrationFilter`: a causal biquad CASCADE that emits a
    filtered sample as each raw sample arrives, suitable for an FPGA / point-by-point
    VI loop. Same constructor parameters and same `apply()` contract as the FFT
    filter, so it is a drop-in replacement in an `AcquisitionSpec` — the VI does not
    change. It adds the genuine streaming primitives a live rig calls:
    `process_sample(x)` for one value and `process_block(arr)` for a chassis frame.

    The cascade is: one RBJ notch per fan harmonic and per structural resonance, then
    a 4th-order Butterworth low-pass (two biquads) at `aero_cutoff_hz`. `warmup_s`
    sets how long the transient is given to settle before the output is trusted; with
    DC-preloaded state (see `_Biquad.reset`) only the AC has to settle, so this is
    short.
    """
    sample_rate_hz: float
    fan_blade_pass_hz: Optional[float] = None
    structural_hz: Sequence[float] = ()
    n_harmonics: int = 3
    aero_cutoff_hz: float = 30.0
    notch_q: float = 30.0
    warmup_s: float = 0.25

    def __post_init__(self):
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self._sections: list[_Biquad] = []
        self._build()

    def _notch_freqs(self) -> list:
        out = []
        if self.fan_blade_pass_hz and self.fan_blade_pass_hz > 0:
            for k in range(1, max(1, self.n_harmonics) + 1):
                out.append(self.fan_blade_pass_hz * k)
        out.extend(float(f) for f in self.structural_hz if f and f > 0)
        return [f for f in out if f < 0.5 * self.sample_rate_hz]

    def _build(self):
        self._sections = []
        self._notched = []
        for f0 in self._notch_freqs():
            b, a = _rbj_notch(f0, self.sample_rate_hz, self.notch_q)
            self._sections.append(_Biquad(b, a))
            self._notched.append(float(f0))
        # 4th-order Butterworth low-pass = two biquads with these quality factors
        if self.aero_cutoff_hz < 0.5 * self.sample_rate_hz:
            for q in (0.54119610, 1.30656296):
                b, a = _rbj_lowpass(self.aero_cutoff_hz, self.sample_rate_hz, q)
                self._sections.append(_Biquad(b, a))

    @property
    def warmup_samples(self) -> int:
        return int(round(self.warmup_s * self.sample_rate_hz))

    # -- the real-time primitives a live VI calls ------------------------- #
    def reset(self, steady_x: float = 0.0):
        """Reset every section, DC-preloaded to `steady_x` so DC needs no settling."""
        for s in self._sections:
            s.reset(steady_x)

    def process_sample(self, x: float) -> float:
        """Filter one sample through the whole cascade. The live VI calls this."""
        y = x
        for s in self._sections:
            y = s.step(y)
        return y

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Filter a chassis frame in arrival order, preserving filter state across calls."""
        out = np.empty(block.shape[0], dtype=float)
        for i, x in enumerate(block):
            out[i] = self.process_sample(float(x))
        return out

    # -- the same offline-looking contract as VibrationFilter ------------- #
    def apply(self, signal: np.ndarray) -> tuple[np.ndarray, VibrationFilterReport]:
        """
        Run the streaming cascade over a captured window and return (clean, report),
        matching `VibrationFilter.apply` so the VI is agnostic to which filter it
        holds. State is reset (DC-preloaded to the window's mean) at the start so the
        result is reproducible; the `warmup_samples` transient is preserved in the
        returned signal but EXCLUDED from the variance bookkeeping, and the report
        notes the causal phase lag. NaN (railed) samples break the recursion, so the
        cascade is re-settled across each gap and the holes are restored as NaN —
        never invented through.
        """
        x = np.asarray(signal, dtype=float).reshape(-1)
        n = x.size
        finite = np.isfinite(x)
        if not finite.any() or n < 4:
            return x.copy(), VibrationFilterReport(
                fan_tone_hz=self.fan_blade_pass_hz, harmonics_notched=[],
                variance_removed_frac=0.0, tone_power_frac=0.0, resolved=False,
                note="too few finite samples to filter — passed through")

        mean_val = float(np.nanmean(x))
        self.reset(mean_val)
        out = np.full(n, np.nan)
        in_gap = False
        for i in range(n):
            if not finite[i]:
                in_gap = True
                continue
            if in_gap:
                # a hole broke the recursion; re-settle on the current sample so the
                # post-gap output is not a transient that the gap injected
                self.reset(x[i])
                in_gap = False
            out[i] = self.process_sample(float(x[i]))

        w = self.warmup_samples
        # variance bookkeeping over the trusted (post-warmup, finite) region only
        trust = finite.copy()
        if w < n:
            trust[:w] = False
        raw_var = float(np.nanvar(x[trust])) if trust.any() else float(np.nanvar(x))
        clean_var = float(np.nanvar(out[trust])) if trust.any() else float(np.nanvar(out[finite]))
        var_removed = 0.0 if raw_var <= 0 else max(0.0, 1.0 - clean_var / raw_var)

        notch_list = self._notch_freqs()
        lowest = min(notch_list, default=None)
        # a streaming filter "resolves" a tone if it ran long enough to settle past it
        resolved = (lowest is None) or (n - w) * (1.0 / self.sample_rate_hz) > (1.0 / lowest)

        note = "causal cascade (has phase lag; time-average is unaffected)"
        if w > 0:
            note += f"; first {w} samples are settling transient, excluded from stats"
        if not resolved and lowest is not None:
            note = (f"window too short for the cascade to settle past {lowest:g} Hz — "
                    "mean may still carry tone residual; " + note)
        return out, VibrationFilterReport(
            fan_tone_hz=self.fan_blade_pass_hz, harmonics_notched=list(self._notched),
            variance_removed_frac=var_removed,
            tone_power_frac=var_removed,   # streaming form measures removal by variance, not band power
            resolved=resolved, note=note)


# --------------------------------------------------------------------------- #
#  Acquisition spec + provenance — what to sample and what the sample is worth
# --------------------------------------------------------------------------- #
@dataclass
class AcquisitionSpec:
    """
    One acquisition: how long to sample, at what tunnel condition, with which filter.
    `seconds` x chassis rate = the sample count the average beats noise down over;
    `attitude`/`speed_ms`/`rho` fix the operating point (and, downstream, q). The
    `vibration` filter is optional but, on a real tunnel, never actually optional —
    leaving it None is recorded so the reduction can warn that the fan tone is still
    in the mean.
    """
    seconds: float = 10.0
    attitude: Optional[Attitude] = None
    speed_ms: float = 25.0
    rho: float = 1.225
    p_static_inf_pa: float = 0.0
    p_total_inf_pa: Optional[float] = None
    vibration: Optional[ChannelFilter] = None    # FFT or streaming-biquad; VI is agnostic

    def n_samples(self, chassis: DAQChassis) -> int:
        return max(1, int(round(self.seconds * chassis.sample_rate_hz)))


@dataclass
class DAQProvenance:
    """
    Where a reduced reading came from and what it's worth — the acquisition twin of
    ScanProvenance/TunnelProvenance. Carries the chassis rate, the run length, whether
    a real or synthetic backend produced the stream, and the vibration filter's own
    report so a force/pressure can never imply more than the acquisition behind it.
    """
    facility: str
    backend: str                                 # "synthetic" | driver name
    sample_rate_hz: float
    seconds: float
    n_samples: int
    speed_ms: float
    rho: float
    ground_state: GroundState = GroundState.MOVING_BELT
    synthetic: bool = False
    filtered: bool = False
    dropped_samples: int = 0
    notes: str = ""

    def averaging_ok(self, min_seconds: float = 5.0) -> bool:
        return self.seconds >= min_seconds

    def status(self) -> str:
        src = "SYNTHETIC" if self.synthetic else self.backend
        filt = "fan/vibration-filtered" if self.filtered else "UNFILTERED"
        warn = ""
        if not self.averaging_ok():
            warn += (f" — WARNING: {self.seconds:g}s window is short; turbulence/"
                     "vibration left in the mean as bias")
        if not self.filtered:
            warn += (" — WARNING: no vibration filter; fan blade-pass tone biases "
                     "the time-average")
        if self.dropped_samples:
            warn += f" — WARNING: chassis dropped {self.dropped_samples} samples"
        if self.synthetic:
            warn += " — NOTE: synthetic stream, not a measured run"
        return (f"{self.facility} via {src}: {self.n_samples} samp @ "
                f"{self.sample_rate_hz:g} Hz ({self.seconds:g}s), {filt}{warn}")


# --------------------------------------------------------------------------- #
#  The clean deliverable — time-averaged raw forces
# --------------------------------------------------------------------------- #
@dataclass
class BalanceReading:
    """
    The clean, time-averaged decoupled balance output for one acquisition — the
    answer to "what are the raw forces". Forces in Newtons (Fx drag, Fy side, Fz lift;
    downforce = -Fz), moments in N·m, each with its standard error of the mean so the
    reading carries its own uncertainty. A component that reduced to a hole (no/ bad
    cal, all-railed) is NaN, never zero. `downforce_N` is the convenience the test
    engineer actually wants.
    """
    forces_N: dict                               # axis.value -> mean load (Fx,Fy,Fz,...)
    stderr_N: dict                               # axis.value -> standard error of mean
    n_used: int                                  # finite samples that fed the mean
    interaction_matrix: Optional[np.ndarray] = None   # the C that decoupled it
    provenance: Optional[DAQProvenance] = None
    notes: str = ""

    def value(self, axis: BalanceAxis) -> float:
        return self.forces_N.get(axis.value, float("nan"))

    @property
    def Fx(self) -> float: return self.value(BalanceAxis.FX)
    @property
    def Fy(self) -> float: return self.value(BalanceAxis.FY)
    @property
    def Fz(self) -> float: return self.value(BalanceAxis.FZ)

    def downforce_N(self) -> float:
        """Positive downforce = -Fz (lift is +z up)."""
        fz = self.Fz
        return float("nan") if not math.isfinite(fz) else -fz

    def is_usable(self) -> bool:
        return all(math.isfinite(self.value(a)) for a in BalanceAxis.forces())

    def status(self) -> str:
        def f(a):
            v = self.value(a); s = self.stderr_N.get(a.value, float("nan"))
            return ("hole" if not math.isfinite(v)
                    else f"{v:+.1f}±{s:.1f} N")
        prov = self.provenance.status() if self.provenance else "provenance unknown"
        return (f"Balance: Fx(drag)={f(BalanceAxis.FX)}, Fy(side)={f(BalanceAxis.FY)}, "
                f"Fz(lift)={f(BalanceAxis.FZ)}, downforce={self.downforce_N():+.1f} N "
                f"[{self.n_used} samp]; {prov}")


# --------------------------------------------------------------------------- #
#  Backends — the seam a real driver implements; synthetic + offline provided
# --------------------------------------------------------------------------- #
class DAQUnavailable(RuntimeError):
    """
    Raised by a backend that has no hardware/driver here, instead of fabricating a
    stream. Exact analogue of RigUnavailable / SolverUnavailable.
    """


@runtime_checkable
class DAQBackend(Protocol):
    """
    The seam. A real backend wraps nidaqmx / a Scanivalve TCP socket / Chell's API and
    returns RAW voltage blocks clocked by the chassis. It returns volts, never forces
    or pressures — the decoupling and reduction are this module's job, not the
    driver's, so they are done one way, tested, in one place.

    `read_balance` -> (n_samples, 6) bridge volts in Fx..Mz order.
    `read_scanner` -> (n_samples, n_channels) transducer volts, columns matching the
                      scanner's port order.
    """
    def read_balance(self, spec: "AcquisitionSpec",
                     chassis: DAQChassis) -> np.ndarray: ...
    def read_scanner(self, scanner: PressureScannerSpec, spec: "AcquisitionSpec",
                     chassis: DAQChassis) -> np.ndarray: ...
    @property
    def name(self) -> str: ...
    @property
    def synthetic(self) -> bool: ...


class OfflineDAQ:
    """
    The no-hardware backend. Honest about having nothing to read: it raises
    DAQUnavailable rather than invent a stream, so wiring a real VI against it fails
    loudly instead of returning fiction.
    """
    name = "offline"
    synthetic = False

    def read_balance(self, spec, chassis):
        raise DAQUnavailable(
            "no balance DAQ backend bound — plug in a nidaqmx/vendor backend; "
            "this offline stub will not fabricate bridge voltages")

    def read_scanner(self, scanner, spec, chassis):
        raise DAQUnavailable(
            "no scanner backend bound — plug in a Scanivalve/Chell backend; "
            "this offline stub will not fabricate transducer voltages")


class SyntheticDAQ:
    """
    A test/teaching backend that generates physically-shaped RAW streams from a known
    truth, so the whole front end is exercisable with no tunnel. It is LABELLED
    synthetic (its provenance flag is True) so a number it produced can never be
    mistaken for a measurement.

    It builds each stream as: the true (decoupled) load mapped BACK through the
    interaction matrix into bridge volts (so the decoupler has real cross-talk to
    undo), plus a fan blade-pass tone, plus broadband turbulence — exactly the
    contamination the VibrationFilter exists to remove. Pressures are built the same
    way: true tap pressure -> volts via the inverse calibration, plus tone + noise.
    """
    name = "synthetic"
    synthetic = True

    def __init__(self, true_loads_N: dict, true_pressures_pa: dict, *,
                 fan_hz: float = 137.0, fan_amp_v: float = 0.05,
                 turb_v: float = 0.01, balance_cal: Optional[BalanceCalibration] = None,
                 seed: int = 0):
        self.true_loads = true_loads_N               # axis.value -> N
        self.true_pressures = true_pressures_pa       # tap_id -> Pa
        self.fan_hz = fan_hz
        self.fan_amp_v = fan_amp_v
        self.turb_v = turb_v
        self.balance_cal = balance_cal
        self._rng = np.random.default_rng(seed)

    def _tvec(self, n, chassis):
        return np.arange(n) / chassis.sample_rate_hz

    def read_balance(self, spec, chassis):
        n = spec.n_samples(chassis)
        t = self._tvec(n, chassis)
        # true load vector in Fx..Mz order
        load = np.array([self.true_loads.get(a.value, 0.0)
                         for a in BalanceAxis.all_six()], dtype=float)
        cal = self.balance_cal
        if cal is not None and cal.usable():
            # invert decouple: bridge volts = C^-1 @ load + zero
            Cinv = np.linalg.inv(cal.matrix)
            base_v = (Cinv @ load) + cal.zero_volts        # (6,)
        else:
            base_v = load / 1000.0                          # fallback scaling
        stream = np.tile(base_v, (n, 1))
        # fan tone + turbulence on every bridge (correlated tone, independent noise)
        tone = self.fan_amp_v * np.sin(2 * np.pi * self.fan_hz * t)
        stream += tone[:, None]
        stream += self.turb_v * self._rng.standard_normal(stream.shape)
        return stream

    def read_scanner(self, scanner, spec, chassis):
        n = spec.n_samples(chassis)
        t = self._tvec(n, chassis)
        cols = []
        tone = self.fan_amp_v * np.sin(2 * np.pi * self.fan_hz * t)
        for tap in scanner.port_taps:
            cal = scanner.calibrations.get(tap.tap_id)
            p_true = self.true_pressures.get(tap.tap_id, 0.0)
            if cal is not None and cal.is_calibrated and cal.sensitivity_pa_per_v != 0:
                base_v = p_true / cal.sensitivity_pa_per_v + cal.zero_offset_v
            else:
                base_v = 0.0
            col = base_v + tone + self.turb_v * self._rng.standard_normal(n)
            cols.append(col)
        return np.column_stack(cols) if cols else np.empty((n, 0))


# --------------------------------------------------------------------------- #
#  The Virtual Instrument — bind hardware, acquire, reduce, stream clean output
# --------------------------------------------------------------------------- #
class VirtualInstrument:
    """
    The VI: one object that binds an under-floor force balance and one-or-more
    electronic pressure scanners to a high-speed DAQ chassis through a backend, then
    on demand acquires a window, applies the vibration filter, decouples the balance,
    time-averages everything, and hands back clean RAW forces (`BalanceReading`) and a
    `RawPressureScan` (P_static per tap) ready for `pressure_tap.to_cp`.

    This is the piece the test engineer "connects": construct it once with the rig,
    then call `acquire_forces(spec)` / `acquire_pressures(spec)` (or `acquire()` for
    both) as many times as the run plan needs. Swap `backend=SyntheticDAQ(...)` for a
    real `nidaqmx`/Scanivalve/Chell backend and the SAME calls drive the real tunnel.
    """

    def __init__(self, *, facility: str, chassis: DAQChassis,
                 balance: Optional[ForceBalanceSpec] = None,
                 scanners: Sequence[PressureScannerSpec] = (),
                 backend: Optional[DAQBackend] = None,
                 ground_state: GroundState = GroundState.MOVING_BELT):
        self.facility = facility
        self.chassis = chassis
        self.balance = balance
        self.scanners = list(scanners)
        self.backend: DAQBackend = backend or OfflineDAQ()
        self.ground_state = ground_state

    # -- provenance helper ------------------------------------------------- #
    def _provenance(self, spec: AcquisitionSpec, *, filtered: bool,
                    dropped: int = 0) -> DAQProvenance:
        return DAQProvenance(
            facility=self.facility, backend=self.backend.name,
            sample_rate_hz=self.chassis.sample_rate_hz, seconds=spec.seconds,
            n_samples=spec.n_samples(self.chassis), speed_ms=spec.speed_ms,
            rho=spec.rho, ground_state=self.ground_state,
            synthetic=bool(getattr(self.backend, "synthetic", False)),
            filtered=filtered, dropped_samples=dropped)

    # -- forces ------------------------------------------------------------ #
    def acquire_forces(self, spec: AcquisitionSpec) -> BalanceReading:
        """
        Acquire the balance, filter each bridge, decouple through the interaction
        matrix, and time-average to clean raw forces. The filter runs on the RAW
        bridge channels (where the fan tone lives) BEFORE decoupling, so the tone is
        gone before the matrix mixes the channels. A missing/uncalibrated balance, or
        an uncalibrated one, yields a reading of holes (NaN) — never a fabricated
        force.
        """
        if self.balance is None:
            raise DAQUnavailable("no force balance bound to this VI")
        raw = np.asarray(self.backend.read_balance(spec, self.chassis), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 6:
            raise ValueError("balance backend must return (n_samples, 6) bridge volts")

        filtered = spec.vibration is not None
        if filtered:
            cleaned = np.empty_like(raw)
            for j in range(6):
                cleaned[:, j], _ = spec.vibration.apply(raw[:, j])
        else:
            cleaned = raw

        cal = self.balance.calibration
        loads = cal.decouple(cleaned)             # (n, 6); all-NaN if uncalibrated
        forces, stderr = {}, {}
        n_used = 0
        for k, axis in enumerate(BalanceAxis.all_six()):
            col = loads[:, k]
            good = col[np.isfinite(col)]
            if good.size == 0:
                forces[axis.value] = float("nan")
                stderr[axis.value] = float("nan")
            else:
                forces[axis.value] = float(np.mean(good))
                stderr[axis.value] = float(np.std(good, ddof=1) / math.sqrt(good.size)
                                           if good.size > 1 else 0.0)
                n_used = max(n_used, good.size)
        return BalanceReading(
            forces_N=forces, stderr_N=stderr, n_used=n_used,
            interaction_matrix=(None if cal.matrix is None else cal.matrix.copy()),
            provenance=self._provenance(spec, filtered=filtered))

    # -- pressures --------------------------------------------------------- #
    def acquire_pressures(self, spec: AcquisitionSpec) -> RawPressureScan:
        """
        Acquire every bound scanner, filter each channel, and assemble ONE
        `RawPressureScan` across all taps — the exact object `pressure_tap.to_cp`
        consumes. The filtered voltage stream (fan tone removed) is what is handed on,
        so the time-average inside `RawPressureScan.tap_pressures_pa()` is over clean
        samples. Returns raw P_static as volts+calibration (reduction to Pa/C_p stays
        in pressure_tap, one place, tested).
        """
        if not self.scanners:
            raise DAQUnavailable("no pressure scanners bound to this VI")
        all_volts, all_taps, all_cals = [], [], {}
        for sc in self.scanners:
            block = np.asarray(self.backend.read_scanner(sc, spec, self.chassis),
                               dtype=float)
            if block.ndim != 2 or block.shape[1] != sc.n_channels:
                raise ValueError("scanner backend must return (n_samples, n_channels)")
            if spec.vibration is not None:
                for j in range(block.shape[1]):
                    block[:, j], _ = spec.vibration.apply(block[:, j])
            all_volts.append(block)
            all_taps.extend(sc.port_taps)
            all_cals.update(sc.calibrations)
        volts = np.column_stack(all_volts) if all_volts else np.empty((0, 0))
        return RawPressureScan(volts=volts, taps=all_taps, calibrations=all_cals,
                               attitude=spec.attitude)

    # -- both, for a run point -------------------------------------------- #
    def acquire(self, spec: AcquisitionSpec) -> tuple[BalanceReading, RawPressureScan]:
        """One operating point: clean raw forces AND the raw pressure scan together."""
        return self.acquire_forces(spec), self.acquire_pressures(spec)

    def scan_provenance(self, spec: AcquisitionSpec) -> ScanProvenance:
        """
        The ScanProvenance to hand `RawPressureScan.to_cp`, built from THIS VI's
        acquisition so q and the reference static come from the run that was actually
        sampled, not a guess.
        """
        return ScanProvenance(
            facility=self.facility, rho=spec.rho, speed_ms=spec.speed_ms,
            p_static_inf_pa=spec.p_static_inf_pa, p_total_inf_pa=spec.p_total_inf_pa,
            sample_rate_hz=self.chassis.sample_rate_hz,
            sample_seconds=spec.seconds, ground_state=self.ground_state,
            blockage_corrected=False,
            notes="reduced from a live VI acquisition (volts -> Pa -> C_p)")

    def status(self) -> str:
        parts = [f"VirtualInstrument @ {self.facility} [backend={self.backend.name}]",
                 "  " + self.chassis.status()]
        if self.balance:
            parts.append("  " + self.balance.status())
        for sc in self.scanners:
            parts.append("  " + sc.status())
        if not self.balance and not self.scanners:
            parts.append("  (no instruments bound)")
        return "\n".join(parts)
