# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Correlation / validation — make the sim earn its trust against real data.

A sim only changes a decision if people believe it, and the honest way to earn
belief is to show that it predicted something you measured. This module takes data
a cash-strapped FSAE team can actually collect — a skidpad time, a 75 m
acceleration time, or a GPS/datalogger speed-vs-distance trace — and reports, in
plain auditable numbers, how well KinematiK's prediction matches it.

It is deliberately NOT a tuning knob. Nothing here changes the model to fit the
data. It quantifies the gap and tells you which way the model is biased, so you can
either trust the prediction for the decision in front of you, or go find the
modelling assumption that's wrong. The point is a number a stubborn lead can check
by hand, not an oracle that says "trust me".

What it reports for each comparison:
    measured, predicted, error (signed + percent), and a per-channel verdict
    against tolerances you can see and change. For a full trace it adds RMSE,
    mean bias (does the sim run systematically fast or slow?), peak-speed error,
    and a coefficient of determination (R²) of predicted vs measured speed.

Conventions match the rest of KinematiK: SI internally, lateral g as a multiple of
g, speeds in m/s, distances in m. Helper constructors accept the friendlier units a
team logs in (lap times in seconds, speeds in km/h) and convert explicitly.

Design rule, same as the lap sim: never raise on bad data. Every public function
returns a result object with an `ok` flag and a human-readable note; a malformed
trace degrades to a clearly-flagged "could not correlate" rather than crashing an
interactive session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np


G = 9.81

# Default tolerances. These are the bands inside which a QSS point-mass model on a
# single tyre set is doing as well as it credibly can on FSAE-scale data. They are
# defaults, not truths — every report carries the tolerance it used so you can
# tighten or loosen it and see the verdict move. Sources for the rough magnitudes:
# QSS lap models typically land within ~2-4% of measured lap/segment times; skidpad
# (closed-form steady state) should be tighter; a noisy GPS trace looser.
DEFAULT_TOL = {
    "skidpad_g_pct": 5.0,        # % error in peak lateral g
    "skidpad_time_pct": 4.0,     # % error in timed-circle time
    "accel_time_pct": 4.0,       # % error in 75 m time
    "trace_rmse_frac": 0.06,     # RMSE / mean measured speed
    "trace_bias_frac": 0.04,     # |mean bias| / mean measured speed
}


# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #
@dataclass
class ChannelResult:
    """One measured-vs-predicted comparison on a single scalar channel."""
    channel: str
    measured: float
    predicted: float
    unit: str
    error: float            # predicted - measured, signed, in `unit`
    error_pct: float        # 100 * error / measured (NaN if measured ~ 0)
    tolerance_pct: float    # the band this verdict used
    within_tol: bool
    note: str = ""

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.error_pct):
            return "n/a"
        return "match" if self.within_tol else "off"

    def as_dict(self):
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


@dataclass
class TraceResult:
    """Aggregate correlation of a predicted speed trace against a measured one."""
    ok: bool
    n_points: int
    mean_measured_speed: float      # m/s
    rmse: float                     # m/s
    rmse_frac: float                # rmse / mean_measured_speed
    bias: float                     # mean(pred - meas), m/s  (+ => sim too fast)
    bias_frac: float
    peak_speed_error: float         # pred peak - meas peak, m/s
    r2: float                       # coefficient of determination, pred vs meas
    within_tol: bool
    distance: np.ndarray = field(default=None, repr=False)
    measured: np.ndarray = field(default=None, repr=False)
    predicted: np.ndarray = field(default=None, repr=False)
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        # keep arrays out of the lightweight dict view
        for k in ("distance", "measured", "predicted"):
            d.pop(k, None)
        return d


@dataclass
class CorrelationReport:
    """Everything from one validation run, ready to render or log to handover."""
    ok: bool
    channels: list                         # list[ChannelResult]
    trace: TraceResult | None = None
    overall_within_tol: bool = False
    summary: str = ""
    tolerances: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(
            ok=self.ok,
            overall_within_tol=self.overall_within_tol,
            summary=self.summary,
            tolerances=dict(self.tolerances),
            channels=[c.as_dict() for c in self.channels],
            trace=self.trace.as_dict() if self.trace else None,
        )


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _pct_err(predicted: float, measured: float) -> float:
    if not (np.isfinite(predicted) and np.isfinite(measured)) or abs(measured) < 1e-9:
        return float("nan")
    return 100.0 * (predicted - measured) / measured


def _channel(channel, measured, predicted, unit, tol_pct, note=""):
    err = (predicted - measured) if (np.isfinite(predicted) and np.isfinite(measured)) else float("nan")
    epct = _pct_err(predicted, measured)
    within = bool(np.isfinite(epct) and abs(epct) <= tol_pct)
    return ChannelResult(channel=channel, measured=float(measured),
                         predicted=float(predicted), unit=unit,
                         error=float(err), error_pct=float(epct),
                         tolerance_pct=float(tol_pct), within_tol=within, note=note)


def skidpad_time_from_lateral_g(lat_g: float, radius_m: float = 9.125) -> float:
    """Closed-form timed-circle time from a measured/known peak lateral g.
    v = sqrt(a_lat * R); t = 2*pi*R / v. Lets a team that only logged a skidpad
    *time* convert to the *g* the sim reports (and vice-versa) so the comparison
    is on the same channel. Returns NaN on non-physical input."""
    if not (np.isfinite(lat_g) and lat_g > 0 and np.isfinite(radius_m) and radius_m > 0):
        return float("nan")
    v = math.sqrt(lat_g * G * radius_m)
    return (2.0 * math.pi * radius_m) / v


def lateral_g_from_skidpad_time(time_s: float, radius_m: float = 9.125) -> float:
    """Inverse of the above: infer the steady-state lateral g a measured skidpad
    time implies. a_lat = v^2 / R, v = 2*pi*R / t."""
    if not (np.isfinite(time_s) and time_s > 0 and np.isfinite(radius_m) and radius_m > 0):
        return float("nan")
    v = (2.0 * math.pi * radius_m) / time_s
    return (v * v / radius_m) / G


# --------------------------------------------------------------------------- #
#  Scalar correlations
# --------------------------------------------------------------------------- #
def correlate_skidpad(veh, measured_g: float = None, measured_time_s: float = None,
                      radius_m: float = 9.125, tol=None) -> CorrelationReport:
    """
    Correlate the steady-state grip model against a real skidpad run — the
    cleanest validation case in FSAE because it is (a) a closed-form steady state
    and (b) something every team runs.

    Provide EITHER the measured peak lateral g OR the measured timed-circle time
    (seconds); the other is derived so both channels are reported. `veh` is a
    VehicleDynamics whose `max_lateral_g()` gives the prediction.
    """
    tol = {**DEFAULT_TOL, **(tol or {})}
    channels = []
    try:
        pred_g = float(veh.max_lateral_g())
    except Exception as e:
        return CorrelationReport(ok=False, channels=[],
                                 summary=f"could not evaluate model grip: {e}",
                                 tolerances=tol)

    if measured_g is None and measured_time_s is None:
        return CorrelationReport(ok=False, channels=[],
                                 summary="provide measured_g or measured_time_s",
                                 tolerances=tol)

    if measured_g is None:
        measured_g = lateral_g_from_skidpad_time(measured_time_s, radius_m)
    if measured_time_s is None:
        measured_time_s = skidpad_time_from_lateral_g(measured_g, radius_m)

    pred_time = skidpad_time_from_lateral_g(pred_g, radius_m)

    channels.append(_channel("Skidpad peak lateral g", measured_g, pred_g, "g",
                             tol["skidpad_g_pct"]))
    channels.append(_channel("Skidpad timed-circle", measured_time_s, pred_time, "s",
                             tol["skidpad_time_pct"],
                             note="derived from g via v=sqrt(a·R)"))
    return _finish(channels, None, tol,
                   context="Skidpad (steady-state grip)")


def correlate_acceleration(measured_time_s: float, predicted_time_s: float,
                           tol=None) -> CorrelationReport:
    """
    Correlate the 75 m acceleration time. The prediction comes from whichever lap
    sim you run (pass its event time in); this keeps the correlation module
    decoupled from the two lap-sim implementations and from the powertrain model.
    """
    tol = {**DEFAULT_TOL, **(tol or {})}
    ch = _channel("Acceleration 75 m", measured_time_s, predicted_time_s, "s",
                  tol["accel_time_pct"])
    return _finish([ch], None, tol, context="Acceleration (75 m)")


# --------------------------------------------------------------------------- #
#  Speed-trace correlation
# --------------------------------------------------------------------------- #
def _extract_trace(lap_result):
    """Pull (distance, speed) from either lap-sim implementation's result object,
    tolerating both field-naming conventions. Returns (dist, speed) float arrays
    or (None, None) if the object doesn't carry a usable trace."""
    if lap_result is None:
        return None, None
    dist = getattr(lap_result, "distance", None)
    spd = getattr(lap_result, "speed", None)
    if dist is None:
        dist = getattr(lap_result, "s", None)
    if spd is None:
        spd = getattr(lap_result, "v", None)
    if dist is None or spd is None:
        return None, None
    try:
        dist = np.asarray(dist, float).ravel()
        spd = np.asarray(spd, float).ravel()
    except Exception:
        return None, None
    if dist.size < 2 or spd.size < 2 or dist.size != spd.size:
        return None, None
    return dist, spd


def correlate_speed_trace(measured_distance, measured_speed,
                          predicted_distance=None, predicted_speed=None,
                          lap_result=None, measured_speed_kmh=False,
                          tol=None) -> CorrelationReport:
    """
    Correlate a measured speed-vs-distance trace (GPS / wheel-speed datalogger)
    against the simulated trace, resampling the prediction onto the measured
    distance axis so the comparison is point-for-point.

    Pass the prediction either as explicit (predicted_distance, predicted_speed)
    arrays, or as a `lap_result` from either lap sim (the trace is extracted
    automatically). Set measured_speed_kmh=True if your log is in km/h.

    Reports RMSE, mean bias (sign tells you if the sim runs fast or slow), peak-
    speed error, and R². No fitting — pure comparison.
    """
    tol = {**DEFAULT_TOL, **(tol or {})}

    if predicted_distance is None or predicted_speed is None:
        predicted_distance, predicted_speed = _extract_trace(lap_result)

    try:
        md = np.asarray(measured_distance, float).ravel()
        ms = np.asarray(measured_speed, float).ravel()
        if measured_speed_kmh:
            ms = ms / 3.6
        pd_ = np.asarray(predicted_distance, float).ravel()
        ps = np.asarray(predicted_speed, float).ravel()
    except Exception as e:
        tr = TraceResult(ok=False, n_points=0, mean_measured_speed=float("nan"),
                         rmse=float("nan"), rmse_frac=float("nan"), bias=float("nan"),
                         bias_frac=float("nan"), peak_speed_error=float("nan"),
                         r2=float("nan"), within_tol=False,
                         note=f"could not parse traces: {e}")
        return _finish([], tr, tol, context="Speed trace")

    bad = (md.size < 2 or ms.size != md.size or pd_.size < 2 or ps.size != pd_.size)
    if bad:
        tr = TraceResult(ok=False, n_points=0, mean_measured_speed=float("nan"),
                         rmse=float("nan"), rmse_frac=float("nan"), bias=float("nan"),
                         bias_frac=float("nan"), peak_speed_error=float("nan"),
                         r2=float("nan"), within_tol=False,
                         note="trace arrays missing, too short, or mismatched lengths")
        return _finish([], tr, tol, context="Speed trace")

    # Resample prediction onto the measured distance axis. Both sims produce a
    # monotonically increasing distance; sort defensively and clip to overlap so
    # extrapolation noise at the ends can't dominate the error.
    order = np.argsort(pd_)
    pd_s, ps_s = pd_[order], ps[order]
    lo = max(md.min(), pd_s.min())
    hi = min(md.max(), pd_s.max())
    mask = (md >= lo) & (md <= hi)
    if mask.sum() < 2:
        tr = TraceResult(ok=False, n_points=int(mask.sum()),
                         mean_measured_speed=float("nan"), rmse=float("nan"),
                         rmse_frac=float("nan"), bias=float("nan"),
                         bias_frac=float("nan"), peak_speed_error=float("nan"),
                         r2=float("nan"), within_tol=False,
                         note="measured and predicted distance ranges do not overlap")
        return _finish([], tr, tol, context="Speed trace")

    md_o, ms_o = md[mask], ms[mask]
    ps_on_m = np.interp(md_o, pd_s, ps_s)

    resid = ps_on_m - ms_o
    mean_meas = float(np.mean(ms_o))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    peak_err = float(np.max(ps_on_m) - np.max(ms_o))
    # R² is only meaningful when the measured speed actually VARIES. On a near-
    # constant trace (e.g. a steady skidpad circle) the total variance is just
    # logging noise and R² becomes large-negative and misleading, so report it as
    # NaN there and let RMSE/bias carry the verdict.
    denom = float(np.sum((ms_o - mean_meas) ** 2))
    meas_std = float(np.std(ms_o))
    if denom > 1e-9 and meas_std > 0.05 * max(mean_meas, 1e-9):
        r2 = float(1.0 - np.sum(resid ** 2) / denom)
    else:
        r2 = float("nan")

    rmse_frac = rmse / mean_meas if mean_meas > 1e-9 else float("nan")
    bias_frac = abs(bias) / mean_meas if mean_meas > 1e-9 else float("nan")
    within = bool(np.isfinite(rmse_frac) and rmse_frac <= tol["trace_rmse_frac"]
                  and np.isfinite(bias_frac) and bias_frac <= tol["trace_bias_frac"])

    note = ""
    if np.isfinite(bias) and abs(bias_frac) > tol["trace_bias_frac"]:
        note = ("sim runs systematically FAST — check grip/aero too optimistic"
                if bias > 0 else
                "sim runs systematically SLOW — check drag/rolling/grip too pessimistic")

    tr = TraceResult(ok=True, n_points=int(md_o.size), mean_measured_speed=mean_meas,
                     rmse=rmse, rmse_frac=rmse_frac, bias=bias, bias_frac=bias_frac,
                     peak_speed_error=peak_err, r2=r2, within_tol=within,
                     distance=md_o, measured=ms_o, predicted=ps_on_m, note=note)
    return _finish([], tr, tol, context="Speed trace")


# --------------------------------------------------------------------------- #
#  Report assembly + verdict
# --------------------------------------------------------------------------- #
def _finish(channels, trace, tol, context="") -> CorrelationReport:
    parts_ok = [c.within_tol for c in channels]
    if trace is not None and trace.ok:
        parts_ok.append(trace.within_tol)
    overall = bool(parts_ok) and all(parts_ok)
    ok = (len(channels) > 0) or (trace is not None and trace.ok)

    # A trace that couldn't be computed is a DATA problem, not a model verdict —
    # don't tell the user the model is wrong when we never actually compared it.
    data_error = (trace is not None and not trace.ok and len(channels) == 0)

    bits = []
    for c in channels:
        if np.isfinite(c.error_pct):
            bits.append(f"{c.channel}: {c.predicted:.3g}{_u(c.unit)} vs "
                        f"{c.measured:.3g}{_u(c.unit)} measured "
                        f"({c.error_pct:+.1f}%, {c.verdict})")
        else:
            bits.append(f"{c.channel}: comparison unavailable")
    if trace is not None:
        if trace.ok:
            r2_txt = f"R²={trace.r2:.3f}" if np.isfinite(trace.r2) else "R²=n/a (constant-speed trace)"
            bits.append(f"Speed trace: RMSE {trace.rmse:.2f} m/s "
                        f"({trace.rmse_frac*100:.1f}% of mean), bias {trace.bias:+.2f} m/s, "
                        f"{r2_txt} over {trace.n_points} pts "
                        f"({'match' if trace.within_tol else 'off'})")
            if trace.note:
                bits.append("→ " + trace.note)
        else:
            bits.append(f"Speed trace: {trace.note}")

    if data_error:
        verdict = ("could not correlate — the measured data couldn't be compared "
                   "(see note); this says nothing about the model")
    elif overall:
        verdict = ("within tolerance — the model predicts this data; trust it for "
                   "the decision in front of you")
    else:
        verdict = ("outside tolerance — do NOT trust the absolute number here until "
                   "you find the modelling assumption that's off (the per-channel "
                   "signs above point at which one)")
    summary = (f"[{context}] " if context else "") + verdict + ". " + "  ".join(bits)

    return CorrelationReport(ok=ok, channels=channels, trace=trace,
                             overall_within_tol=overall, summary=summary,
                             tolerances=dict(tol))


def _u(unit: str) -> str:
    return "" if unit in ("", None) else f" {unit}"
