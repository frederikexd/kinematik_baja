# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Surface pressure taps — turn a wind-tunnel run's RAW channels into a mapped C_p
field on the wing, and correlate that field against CFD by RMSE.

WHY THIS MODULE EXISTS (read this before using it)
---------------------------------------------------
`windtunnel.py` begins where the engineering insight has already been thrown away.
It correlates FINISHED coefficients — one C_l, one C_d, one balance number per ride
height. Those integrals answer "how much" but never "where", and "where" is the
question a tunnel run is uniquely able to answer. A wing makes its total downforce
number with the flow attached over the whole suction surface, or it makes the SAME
number with the leading edge over-loaded and the trailing third already stalled and
leaking — and the integral cannot tell those two wings apart. The team only finds
out which one they built when the car does something the lap sim never predicted.

A tunnel run does not actually produce a C_l. It produces a wall of RAW numbers:
a matrix of pressure-transducer voltages (one column per surface tap, one row per
sample), a load-cell force trace, and a logged wind speed. Left in that form it is
unreadable — nobody can look at forty thousand millivolt readings and see a stall.
This module is the reduction that makes the run legible, in three honest stages:

    raw transducer volts (+ calibration, + wind speed)   ->   non-dimensional C_p
      ->   C_p mapped onto the wing's surface coordinates (chordwise x/c, span)
      ->   an RMSE C_p correlation against the CFD surface at the SAME tap points,
           with the loading integral and a suction-collapse (stall) flag the
           coefficient correlation structurally cannot produce.

WHAT C_p IS, AND WHY IT IS THE RIGHT NON-DIMENSIONAL QUANTITY
-------------------------------------------------------------
The pressure coefficient

        C_p = (p - p_inf) / q ,        q = 1/2 rho V^2     (dynamic pressure)

collapses a measured surface pressure onto a scale that is independent of how fast
the tunnel was run. C_p = 1 is the stagnation point (flow brought to rest); C_p = 0
is freestream static; strongly NEGATIVE C_p is suction — and suction on the right
surface is what makes downforce. Plotting C_p along the chord is how every aero
engineer reads a wing: a healthy suction surface shows a sharp negative peak just
aft of the leading edge and a smooth pressure recovery to the trailing edge; a
stalled one shows that recovery go flat — the pressure stops recovering because the
boundary layer has detached and the surface pressure has gone constant under the
separation bubble. That flat tail is a thing you can SEE in C_p(x/c) and cannot see
in C_l. This module computes it, maps it to the wing, and measures it against CFD.

THE HONESTY CONTRACT (identical discipline to cfd.py / windtunnel.py / piv.py)
------------------------------------------------------------------------------
A reduced C_p curve looks authoritative — it is "the measurement" — but it is only
as good as the calibration and the reference pressures behind it, and a quietly
mis-scaled transducer or a wrong q lies with the full authority of real data. So,
by construction:

  * Every `CpField` carries `TapCalibration` + the dynamic pressure it was reduced
    against. A channel whose transducer span was guessed, or whose zero was not
    taken, is a different measurement from a calibrated one, and the provenance says
    so. The freestream static and total reference pressures are recorded, not
    assumed, because C_p is meaningless without the p_inf and q it is measured from.
  * A tap with no calibration, a saturated/railed transducer, or a sample window too
    short to average down the turbulence is a HOLE — it reduces to NaN and is never
    silently filled with a neighbour or a zero.
  * The RMSE correlation pairs a measured tap to a CFD value at the SAME surface
    location (matched by tap id, never snapped to the nearest node). A tap the CFD
    surface does not cover is an unpaired hole; an RMSE it reports is over the taps
    that genuinely paired, and it says how many that was.

DELIBERATE NON-GOALS, same as the rest of the aero package: this module does not
talk to a DAQ, does not own a transducer, and does not solve Navier-Stokes. It owns
the SEAM (a typed description of the raw channels a real Scanivalve/DAQ produces)
and the MATH (volts -> C_p reduction, surface mapping, loading integral, stall
detection, and the C_p RMSE), so the whole pipeline is writable and testable now,
with synthetic raw data, against no hardware and no solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .cfd import Attitude
from .windtunnel import GroundState


# --------------------------------------------------------------------------- #
#  Where a tap sits on the wing — the lock that makes "same surface point" literal
# --------------------------------------------------------------------------- #
#
# A surface pressure tap is a small hole drilled normal to the wing skin, plumbed to
# one transducer channel. To read the run as a wing you must know WHERE each tap is,
# in the wing's own coordinates: which element (a multi-element FSAE wing has a main
# plane plus one or more flaps), the chordwise fraction x/c (0 = leading edge,
# 1 = trailing edge), the spanwise station, and which surface — suction or pressure.
# This is exactly the role LaserSheetPlane plays for PIV: the shared geometry that
# makes a measured point and a CFD point the SAME point. The CFD post-processor
# samples its surface solution at these SAME (element, x/c, span) locations, so the
# overlay is like-for-like and the RMSE is meaningful rather than an artefact of two
# different sampling grids.
class WingSurface(str, Enum):
    """Which side of the skin the tap reads. Suction side makes the downforce."""
    SUCTION = "suction"        # upper side of an upside-down (downforce) wing
    PRESSURE = "pressure"      # lower side; the higher-pressure face
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TapLocation:
    """
    One pressure tap's position on the wing, in wing-relative coordinates. `element`
    names the aerofoil element (e.g. "main", "flap1") so a multi-element wing's taps
    don't collide; `x_over_c` is the chordwise fraction along THAT element's chord;
    `span_m` is the spanwise station; `surface` is suction/pressure. `tap_id` is the
    DAQ channel label and is the key the CFD surface is matched against — it must be
    unique across the wing, and it is what stops a measured tap being paired to the
    wrong CFD node.
    """
    tap_id: str
    element: str = "main"
    x_over_c: float = 0.0
    span_m: float = 0.0
    surface: WingSurface = WingSurface.SUCTION

    def key(self) -> str:
        """The pairing key. Tap id is authoritative; location is for the human."""
        return self.tap_id

    def label(self) -> str:
        return (f"{self.tap_id} [{self.element} {self.surface.value} "
                f"x/c={self.x_over_c:.2f} y={self.span_m:.3f}m]")


# --------------------------------------------------------------------------- #
#  Transducer calibration — what turns a voltage into a pressure honestly
# --------------------------------------------------------------------------- #
@dataclass
class TapCalibration:
    """
    The per-channel calibration that maps a raw transducer voltage to a gauge
    pressure (Pa). A pressure transducer is, to first order, linear:

        p_gauge = sensitivity_pa_per_v * (V - zero_offset_v)

    `sensitivity_pa_per_v` is the transducer span (from its calibration sheet, or a
    deadweight/manometer calibration); `zero_offset_v` is the wind-OFF reading taken
    at the start of the run (the "tare" — it drifts with temperature, so it is taken,
    not assumed zero). `is_calibrated` is False until BOTH are real: a channel with a
    guessed span or no zero is not calibrated, and a reduction must treat it as a hole
    rather than scale a voltage by a number nobody measured. This mirrors
    DamperCurve.is_calibrated / the CFD and tyre provenance exactly.
    """
    sensitivity_pa_per_v: float
    zero_offset_v: float = 0.0
    is_calibrated: bool = False
    saturation_v: Optional[float] = None       # transducer rail; readings at it are clipped/garbage
    serial: str = ""
    notes: str = ""

    def volts_to_pressure(self, volts: np.ndarray) -> np.ndarray:
        """
        Apply the linear calibration to a voltage trace -> gauge pressure (Pa). A
        sample at or beyond the saturation rail is railed (the transducer maxed out)
        and is returned as NaN — a railed reading is not a pressure, and pretending it
        is would corrupt the tap's mean. Uncalibrated channels reduce to all-NaN.
        """
        v = np.asarray(volts, dtype=float)
        if not self.is_calibrated:
            return np.full(v.shape, np.nan)
        p = self.sensitivity_pa_per_v * (v - self.zero_offset_v)
        if self.saturation_v is not None:
            railed = np.abs(v) >= abs(self.saturation_v) - 1e-12
            p = np.where(railed, np.nan, p)
        return p

    def status(self) -> str:
        if not self.is_calibrated:
            return (f"channel {self.serial or '?'}: UNCALIBRATED — span/zero not "
                    "established; readings are not pressures and reduce to holes")
        rail = (f", rail ±{abs(self.saturation_v):g} V"
                if self.saturation_v is not None else "")
        return (f"channel {self.serial or '?'}: {self.sensitivity_pa_per_v:g} Pa/V, "
                f"zero {self.zero_offset_v:+.4f} V{rail}")


# --------------------------------------------------------------------------- #
#  Provenance — the wind-tunnel-run honesty record for a pressure scan
# --------------------------------------------------------------------------- #
@dataclass
class ScanProvenance:
    """
    Where a surface-pressure scan came from and what it's worth — the pressure-tap
    twin of TunnelProvenance / PIVProvenance. The fields here are the ones that decide
    whether a C_p is a measurement of the wing or a measurement of the rig's own
    error: the reference pressures C_p is defined against, the air density and wind
    speed that set the dynamic pressure q, the sample window that the turbulence is
    averaged over, the floor state, and the blockage correction. A C_p reduced against
    a guessed q, or a one-sample "average", is a different number from a properly
    referenced, time-averaged one, and this records the difference.
    """
    facility: str
    rho: float = 1.225                          # air density, kg/m^3
    speed_ms: float = 20.0                       # tunnel wind speed (sets q with rho)
    p_static_inf_pa: float = 0.0                 # freestream static reference (gauge)
    p_total_inf_pa: Optional[float] = None       # freestream total/pitot (gauge); q if given
    sample_rate_hz: Optional[float] = None
    sample_seconds: Optional[float] = None       # averaging window per tap
    ground_state: GroundState = GroundState.MOVING_BELT
    blockage_corrected: bool = True
    blockage_ratio: Optional[float] = None
    reynolds: Optional[float] = None
    notes: str = ""

    def dynamic_pressure(self) -> float:
        """
        q = 1/2 rho V^2, the scale C_p is normalised by. If a pitot total pressure was
        logged, prefer the measured q = p_total - p_static over the computed one — the
        pitot is what the freestream actually was, computing from a nominal V trusts a
        speed set-point that the tunnel may not have held.
        """
        if self.p_total_inf_pa is not None:
            q_meas = self.p_total_inf_pa - self.p_static_inf_pa
            if q_meas > 0:
                return float(q_meas)
        return 0.5 * self.rho * self.speed_ms * self.speed_ms

    def averaging_ok(self, min_seconds: float = 5.0) -> bool:
        """
        Enough averaging to beat down freestream turbulence? A surface pressure on an
        FSAE wing fluctuates several percent; a window under a few seconds leaves that
        turbulence in the mean as a bias masquerading as signal.
        """
        return (self.sample_seconds is not None
                and self.sample_seconds >= min_seconds)

    def status(self) -> str:
        q = self.dynamic_pressure()
        qsrc = "pitot" if (self.p_total_inf_pa is not None
                           and (self.p_total_inf_pa - self.p_static_inf_pa) > 0) else "½ρV²"
        blk = "blockage-corrected" if self.blockage_corrected else "NOT blockage-corrected"
        re = f", Re={self.reynolds:.2e}" if self.reynolds else ""
        warn = ""
        if not self.averaging_ok():
            sec = "unknown" if self.sample_seconds is None else f"{self.sample_seconds:g}s"
            warn += (f" — WARNING: averaging window {sec} is short; freestream "
                     "turbulence is left in the tap means as bias")
        if self.ground_state is not GroundState.MOVING_BELT:
            warn += (" — WARNING: fixed/suction floor distorts the underbody/wing-floor "
                     "pressure field; C_p near the ground is not ground-effect-true")
        if not self.blockage_corrected:
            warn += (" — WARNING: uncorrected static reference inflates suction C_p; "
                     "correct blockage before comparing to free-air CFD")
        return (f"{self.facility}: q={q:.1f} Pa ({qsrc}, V={self.speed_ms:g} m/s, "
                f"rho={self.rho:g}), p_inf={self.p_static_inf_pa:+.1f} Pa gauge, "
                f"{self.ground_state.value} floor, {blk}{re}{warn}")


# --------------------------------------------------------------------------- #
#  The raw tunnel deliverable — what actually comes off the DAQ
# --------------------------------------------------------------------------- #
@dataclass
class RawPressureScan:
    """
    The unreadable thing a tunnel run hands you: a matrix of transducer voltages,
    `volts` shaped (n_samples, n_taps) — one column per surface tap, one row per
    time sample — plus the tap each column belongs to (`taps`, same order as the
    columns) and the per-tap calibration (`calibrations`, keyed by tap_id). This is
    the object that replaces "a massive CSV matrix of pressure readings": it knows
    which column is which tap and how to turn volts into Pa, so it can be reduced to
    a C_p field instead of stared at.

    It owns no physics beyond bookkeeping. The reduction to C_p lives in `to_cp`,
    which needs the run's ScanProvenance (for q and the reference static), so the
    dynamic pressure a C_p is divided by is always the one the run was actually at.
    """
    volts: np.ndarray                            # (n_samples, n_taps)
    taps: Sequence[TapLocation]
    calibrations: dict                           # tap_id -> TapCalibration
    attitude: Optional[Attitude] = None

    def __post_init__(self):
        self.volts = np.asarray(self.volts, dtype=float)
        if self.volts.ndim == 1:
            # a single-sample scan is allowed; promote to one row
            self.volts = self.volts[None, :]
        if self.volts.ndim != 2:
            raise ValueError("volts must be a 2D (n_samples, n_taps) matrix")
        if self.volts.shape[1] != len(self.taps):
            raise ValueError(
                f"volts has {self.volts.shape[1]} columns but {len(self.taps)} taps "
                "were given — one column per tap is required")
        ids = [t.tap_id for t in self.taps]
        if len(set(ids)) != len(ids):
            raise ValueError("tap_id values must be unique across the wing")

    @property
    def n_samples(self) -> int:
        return self.volts.shape[0]

    @property
    def n_taps(self) -> int:
        return self.volts.shape[1]

    def tap_pressures_pa(self) -> dict:
        """
        Reduce each column to a time-averaged gauge pressure (Pa), tap_id -> p̄.
        Averaging is over the sample axis with NaN-safe mean: railed/garbage samples
        (NaN out of the calibration) are excluded so one clipped sample does not drag
        the mean. A tap that is uncalibrated, or all-railed, averages to NaN — an
        honest hole.
        """
        out: dict[str, float] = {}
        for j, tap in enumerate(self.taps):
            cal = self.calibrations.get(tap.tap_id)
            col = self.volts[:, j]
            if cal is None:
                out[tap.tap_id] = float("nan")
                continue
            p = cal.volts_to_pressure(col)
            if np.all(np.isnan(p)):
                out[tap.tap_id] = float("nan")
            else:
                out[tap.tap_id] = float(np.nanmean(p))
        return out

    def to_cp(self, prov: ScanProvenance) -> "CpField":
        """
        The headline reduction: raw volts -> non-dimensional C_p, mapped onto the
        wing. For each tap, time-average the calibrated gauge pressure, subtract the
        freestream static reference, and divide by the run's dynamic pressure q:

            C_p = (p_tap - p_static_inf) / q

        The result is a CpField — C_p per tap, carrying the tap geometry so it reads
        as a wing, and the provenance so it can never imply more than the run behind
        it. Uncalibrated/railed taps are NaN holes, never zero.
        """
        q = prov.dynamic_pressure()
        pres = self.tap_pressures_pa()
        cps: dict[str, float] = {}
        for tap in self.taps:
            p = pres.get(tap.tap_id, float("nan"))
            if not math.isfinite(p) or q <= 0:
                cps[tap.tap_id] = float("nan")
            else:
                cps[tap.tap_id] = (p - prov.p_static_inf_pa) / q
        return CpField(taps=list(self.taps), cp=cps,
                       attitude=self.attitude, provenance=prov)


# --------------------------------------------------------------------------- #
#  The mapped C_p field — the run made legible as a wing
# --------------------------------------------------------------------------- #
@dataclass
class CpField:
    """
    C_p at every tap, mapped onto the wing's surface coordinates — the deliverable
    that replaces the raw matrix. It is the pressure-tap analogue of PIV's
    VelocityField: a measured field on a known geometry, ready to overlay on CFD.

    `cp` is tap_id -> C_p (NaN where the tap was a hole). The taps carry element /
    x_over_c / span / surface, so this field knows the wing it sits on and can be
    sliced into a chordwise C_p(x/c) curve per element/surface — the curve an aero
    engineer actually reads to see where the wing is loaded and where it has stalled.
    """
    taps: Sequence[TapLocation]
    cp: dict                                    # tap_id -> C_p (may be NaN)
    attitude: Optional[Attitude] = None
    provenance: Optional[ScanProvenance] = None

    def _tap_by_id(self) -> dict:
        return {t.tap_id: t for t in self.taps}

    def valid_taps(self) -> list:
        """Tap ids whose C_p is a real number (calibrated, not railed)."""
        return [tid for tid, v in self.cp.items() if math.isfinite(v)]

    def coverage(self) -> float:
        """Fraction of taps that reduced to a usable C_p."""
        return (len(self.valid_taps()) / len(self.cp)) if self.cp else 0.0

    # -- the chordwise curve the engineer reads --------------------------- #
    def chordwise(self, element: str, surface: WingSurface,
                  span_tol_m: float = 1e9) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Pull the C_p(x/c) distribution for one element's one surface, sorted leading
        edge -> trailing edge. Returns (x_over_c, cp, tap_ids) over the VALID taps
        only — holes are dropped, never interpolated, so a gap in the curve is a real
        gap in the instrumentation. `span_tol_m` restricts to taps near one spanwise
        station for a 3D wing; the default takes the whole span.
        """
        rows = []
        for t in self.taps:
            if t.element != element or t.surface != surface:
                continue
            v = self.cp.get(t.tap_id, float("nan"))
            if not math.isfinite(v):
                continue
            rows.append((t.x_over_c, v, t.tap_id, t.span_m))
        if span_tol_m < 1e9 and rows:
            # keep taps near the most-populated span station
            spans = sorted({r[3] for r in rows})
            s0 = spans[len(spans) // 2]
            rows = [r for r in rows if abs(r[3] - s0) <= span_tol_m]
        rows.sort(key=lambda r: r[0])
        if not rows:
            return np.array([]), np.array([]), []
        xc = np.array([r[0] for r in rows], dtype=float)
        cp = np.array([r[1] for r in rows], dtype=float)
        ids = [r[2] for r in rows]
        return xc, cp, ids

    # -- the loading integral: where the downforce comes from ------------- #
    def normal_load_coefficient(self, element: str,
                                span_tol_m: float = 1e9) -> float:
        """
        Integrate (C_p_pressure - C_p_suction) along the chord to get the element's
        sectional normal-force coefficient C_n — the chordwise integral of the
        pressure difference between the two surfaces. This is HOW the section makes
        its load, computed from the surface field rather than read off a load cell, so
        it can be attributed chordwise: the trapezoidal integrand at each x/c is the
        local loading, and that is what reveals an over-loaded leading edge or an
        unloaded (stalled) tail. Requires both surfaces instrumented; returns NaN if
        either surface has fewer than two valid taps to integrate.
        """
        xs, cps_s, _ = self.chordwise(element, WingSurface.SUCTION, span_tol_m)
        xp, cpp, _ = self.chordwise(element, WingSurface.PRESSURE, span_tol_m)
        if len(xs) < 2 or len(xp) < 2:
            return float("nan")
        # common chord grid = the union of both surfaces' x/c, clipped to overlap
        lo = max(xs.min(), xp.min())
        hi = min(xs.max(), xp.max())
        if hi <= lo:
            return float("nan")
        grid = np.linspace(lo, hi, 50)
        cp_s = np.interp(grid, xs, cps_s)
        cp_p = np.interp(grid, xp, cpp)
        # C_n = ∮ C_p d(x/c) with pressure side minus suction side (downforce +ve here)
        integrand = cp_p - cp_s
        return float(np.trapezoid(integrand, grid))

    # -- the thing the integral cannot see: stall ------------------------- #
    def suction_peak(self, element: str,
                     span_tol_m: float = 1e9) -> tuple[float, float]:
        """
        The minimum (most negative) C_p on the suction surface and the x/c it sits at
        — the suction peak. Its depth is how hard the wing is working; its position is
        how far forward the load is. Returns (cp_min, x_over_c_at_min), or (nan, nan)
        if the suction surface has no valid taps.
        """
        xc, cp, _ = self.chordwise(element, WingSurface.SUCTION, span_tol_m)
        if len(cp) == 0:
            return float("nan"), float("nan")
        i = int(np.argmin(cp))
        return float(cp[i]), float(xc[i])

    def stall_indicator(self, element: str, span_tol_m: float = 1e9,
                        recovery_slope_tol: float = 0.5) -> "StallVerdict":
        """
        Read the suction surface for the signature of separation. A healthy suction
        surface recovers pressure aft of the peak — C_p climbs back toward 0 as the
        flow decelerates to the trailing edge. A STALLED one stops recovering: the
        boundary layer has detached, and under the separation bubble the surface
        pressure goes flat, leaving a constant-C_p plateau on the back of the wing.

        This measures the recovery: it fits the slope dC_p/d(x/c) from the suction
        peak to the trailing edge. A healthy wing has a clearly positive recovery
        slope (C_p rising toward 0). A slope near zero over the aft chord — below
        `recovery_slope_tol` — is the flat plateau of a stalled surface. The verdict
        carries the peak, the recovery slope, and the aft-chord C_p spread so the call
        is auditable, not a bare boolean.
        """
        xc, cp, ids = self.chordwise(element, WingSurface.SUCTION, span_tol_m)
        if len(cp) < 3:
            return StallVerdict(
                element=element, stalled=False, peak_cp=float("nan"),
                peak_xc=float("nan"), recovery_slope=float("nan"),
                aft_cp_spread=float("nan"),
                note="too few valid suction taps to judge recovery — not a verdict")
        i_peak = int(np.argmin(cp))
        peak_cp = float(cp[i_peak]); peak_xc = float(xc[i_peak])
        # aft of the peak — where recovery should happen
        xc_aft = xc[i_peak:]; cp_aft = cp[i_peak:]
        if len(cp_aft) < 2:
            return StallVerdict(
                element=element, stalled=False, peak_cp=peak_cp, peak_xc=peak_xc,
                recovery_slope=float("nan"), aft_cp_spread=float("nan"),
                note="suction peak is at the trailing edge — no aft chord to recover")
        # least-squares recovery slope dCp/d(x/c) over the aft chord
        A = np.vstack([xc_aft, np.ones_like(xc_aft)]).T
        slope, _ = np.linalg.lstsq(A, cp_aft, rcond=None)[0]
        aft_spread = float(np.nanmax(cp_aft) - np.nanmin(cp_aft))
        stalled = bool(slope < recovery_slope_tol)
        if stalled:
            note = (f"FLAT recovery (dCp/d(x/c)={slope:+.2f} < {recovery_slope_tol:g}) "
                    "aft of the suction peak — pressure stopped recovering; the "
                    "suction surface has separated / stalled over the aft chord")
        else:
            note = (f"healthy recovery (dCp/d(x/c)={slope:+.2f}) from the suction peak "
                    "to the trailing edge — flow attached")
        return StallVerdict(
            element=element, stalled=stalled, peak_cp=peak_cp, peak_xc=peak_xc,
            recovery_slope=float(slope), aft_cp_spread=aft_spread, note=note)

    def status(self) -> str:
        n = len(self.cp); good = len(self.valid_taps())
        prov = self.provenance.status() if self.provenance else "provenance unknown"
        return (f"CpField: {good}/{n} taps reduced ({100*self.coverage():.0f}% "
                f"coverage); {prov}")


@dataclass
class StallVerdict:
    """Auditable separation call for one element's suction surface."""
    element: str
    stalled: bool
    peak_cp: float
    peak_xc: float
    recovery_slope: float
    aft_cp_spread: float
    note: str = ""

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
#  The CFD surface C_p — the digital twin sampled at the SAME taps
# --------------------------------------------------------------------------- #
@dataclass
class CFDSurfaceCp:
    """
    The CFD surface pressure sampled at the EXACT tap locations — the digital twin of
    a CpField. The team's post-processor probes the solved surface at each tap_id's
    (element, x/c, span) and exports tap_id -> C_p. This object just holds that, so
    the RMSE correlation compares the two C_p fields tap-for-tap. Like every other
    seam in the package: KinematiK does not sample the CFD itself; it owns the
    contract and refuses to pair a tap the CFD did not cover.
    """
    cp: dict                                    # tap_id -> C_p from CFD
    attitude: Optional[Attitude] = None
    backend: str = ""
    turbulence_model: str = ""
    notes: str = ""

    @classmethod
    def from_pairs(cls, pairs, **kw) -> "CFDSurfaceCp":
        return cls(cp={str(k): float(v) for k, v in dict(pairs).items()}, **kw)


# --------------------------------------------------------------------------- #
#  The RMSE correlation — the deliverable the user asked for
# --------------------------------------------------------------------------- #
@dataclass
class TapResidual:
    """The C_p delta at ONE tap — measured vs CFD — for a per-tap error map."""
    tap: TapLocation
    cp_phys: Optional[float]
    cp_cfd: Optional[float]
    paired: bool = False
    note: str = ""

    @property
    def residual(self) -> float:
        """CFD minus physical, in C_p units (the RMSE is over these)."""
        if self.cp_phys is None or self.cp_cfd is None:
            return float("nan")
        if not (math.isfinite(self.cp_phys) and math.isfinite(self.cp_cfd)):
            return float("nan")
        return self.cp_cfd - self.cp_phys

    def as_dict(self):
        d = asdict(self)
        d["tap"] = self.tap.label()
        d["residual"] = self.residual
        return d


DEFAULT_CP_TOL = {
    "rmse": 0.10,          # RMSE in C_p units across the wing (a tenth of a C_p)
    "max_abs": 0.20,       # worst single-tap |C_p error| allowed
    "min_coverage": 0.6,   # at least this fraction of taps must pair to trust the RMSE
}


@dataclass
class CpCorrelationReport:
    """
    Did the CFD surface pressure reproduce the measured one — by RMSE, tap for tap?
    This is the spatial complement to the windtunnel coefficient correlation: it can
    fail (CFD got the pressure distribution wrong) even when the integrated C_l lands,
    because two different C_p curves can integrate to the same force. Nothing here
    tunes CFD; it quantifies the C_p gap in auditable numbers and points at the taps
    that drive it.
    """
    ok: bool
    rmse: float                                 # root-mean-square C_p error over paired taps
    bias: float                                 # signed mean C_p error (cfd - phys)
    max_abs_residual: float
    n_paired: int
    n_unpaired: int
    coverage: float                             # paired / total measured taps
    worst: Optional[TapResidual]
    residuals: list                             # list[TapResidual]
    within_tol: bool
    tolerances: dict
    summary: str = ""
    physical_provenance: str = ""
    cfd_provenance: str = ""

    def as_dict(self):
        return dict(
            ok=self.ok, rmse=self.rmse, bias=self.bias,
            max_abs_residual=self.max_abs_residual,
            n_paired=self.n_paired, n_unpaired=self.n_unpaired,
            coverage=self.coverage, within_tol=self.within_tol,
            tolerances=dict(self.tolerances), summary=self.summary,
            physical_provenance=self.physical_provenance,
            cfd_provenance=self.cfd_provenance,
            worst=self.worst.as_dict() if self.worst else None,
            residuals=[r.as_dict() for r in self.residuals],
        )


def correlate_cp(phys: CpField, cfd: CFDSurfaceCp,
                 tol: Optional[dict] = None) -> CpCorrelationReport:
    """
    Compute the RMSE between the measured surface C_p and the CFD surface C_p, tap
    for tap. Each measured tap is paired to the CFD C_p with the SAME tap_id — never
    snapped to a nearest node — so the residual at a tap is a real disagreement at a
    real surface point. The RMSE is taken over the taps that genuinely paired (both
    sides finite); taps the CFD did not cover, or that reduced to holes, are reported
    as unpaired and excluded from the error, with the count surfaced so the RMSE is
    never quoted over a coverage too thin to mean anything.

        RMSE = sqrt( mean_over_paired( (C_p_cfd - C_p_phys)^2 ) )

    This is the number that says, in C_p units, how far the simulated pressure
    distribution sits from the measured one across the whole wing.
    """
    tol = {**DEFAULT_CP_TOL, **(tol or {})}
    by_id = {t.tap_id: t for t in phys.taps}

    residuals: list[TapResidual] = []
    sq_errs: list[float] = []
    errs: list[float] = []

    # measured taps, paired against CFD where possible
    for tid, tap in by_id.items():
        cp_p = phys.cp.get(tid, float("nan"))
        cp_c = cfd.cp.get(tid, None)
        if cp_c is None:
            residuals.append(TapResidual(
                tap=tap, cp_phys=(cp_p if math.isfinite(cp_p) else None),
                cp_cfd=None, paired=False,
                note="no CFD C_p at this tap — not paired"))
            continue
        if not math.isfinite(cp_p):
            residuals.append(TapResidual(
                tap=tap, cp_phys=None, cp_cfd=float(cp_c), paired=False,
                note="measured tap is a hole (uncalibrated/railed) — not paired"))
            continue
        tr = TapResidual(tap=tap, cp_phys=float(cp_p), cp_cfd=float(cp_c),
                         paired=True)
        residuals.append(tr)
        r = tr.residual
        if math.isfinite(r):
            sq_errs.append(r * r)
            errs.append(r)

    # CFD taps the physical scan never had (instrumentation hole the other way)
    for tid, cp_c in cfd.cp.items():
        if tid not in by_id:
            residuals.append(TapResidual(
                tap=TapLocation(tap_id=tid), cp_phys=None, cp_cfd=float(cp_c),
                paired=False, note="CFD tap not present in the physical scan — not paired"))

    n_paired = len(sq_errs)
    n_unpaired = sum(1 for r in residuals if not r.paired)
    n_measured = len(by_id)
    coverage = (n_paired / n_measured) if n_measured else 0.0

    rmse = math.sqrt(sum(sq_errs) / n_paired) if n_paired else float("nan")
    bias = (sum(errs) / n_paired) if n_paired else float("nan")

    paired_res = [r for r in residuals if r.paired and math.isfinite(r.residual)]
    worst = max(paired_res, key=lambda r: abs(r.residual), default=None)
    max_abs = abs(worst.residual) if worst else float("nan")

    within = (
        bool(n_paired)
        and math.isfinite(rmse) and rmse <= tol["rmse"]
        and math.isfinite(max_abs) and max_abs <= tol["max_abs"]
        and coverage >= tol["min_coverage"]
    )

    summary = _summarise_cp(rmse, bias, max_abs, n_paired, n_unpaired, coverage,
                            worst, within, tol, cfd)

    return CpCorrelationReport(
        ok=bool(n_paired), rmse=rmse, bias=bias, max_abs_residual=max_abs,
        n_paired=n_paired, n_unpaired=n_unpaired, coverage=coverage,
        worst=worst, residuals=residuals, within_tol=within, tolerances=tol,
        summary=summary,
        physical_provenance=(phys.provenance.status() if phys.provenance else ""),
        cfd_provenance=(f"{cfd.backend or 'CFD'}"
                        + (f" ({cfd.turbulence_model})" if cfd.turbulence_model else "")),
    )


def _summarise_cp(rmse, bias, max_abs, n_paired, n_unpaired, coverage,
                  worst, within, tol, cfd):
    head = (f"[Surface C_p] {cfd.backend or 'CFD'}"
            + (f" ({cfd.turbulence_model})" if cfd.turbulence_model else "")
            + f" vs measured taps: {n_paired} tap(s) paired")
    if n_unpaired:
        head += f", {n_unpaired} unpaired (holes — excluded from RMSE)"
    if not n_paired:
        return (head + ". Nothing could be compared tap-for-tap — sample the CFD "
                "surface at the same tap_ids as the physical scan, then re-correlate.")
    if coverage < tol["min_coverage"]:
        cov = (f"  Coverage {100*coverage:.0f}% is below the "
               f"{100*tol['min_coverage']:.0f}% floor — the RMSE is over too few taps "
               "to trust; instrument or sample more of the wing.")
    else:
        cov = ""
    verdict = ("MATCHED — the CFD surface pressure reproduces the measured C_p "
               "distribution inside tolerance; the simulated wing is loaded where the "
               "real one is"
               if within else
               "MISMATCHED — the CFD pressure distribution drifts from the measured "
               "C_p beyond tolerance; the integrated force may still agree while the "
               "wing is loaded differently, so do NOT trust CFD surface detail here")
    worst_txt = ""
    if worst is not None and math.isfinite(worst.residual):
        worst_txt = (f"  Worst tap: {worst.tap.label()} "
                     f"ΔC_p={worst.residual:+.3f} (CFD−phys).")
    return (f"{head}. {verdict}. "
            f"RMSE {rmse:.3f} C_p (bias {bias:+.3f}, max |ΔC_p| {max_abs:.3f})."
            + cov + worst_txt)
