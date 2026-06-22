# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
AeroMap — the deliverable. A function from car attitude to aero coefficients,
assembled from a sweep of CFD results, queryable by the lap sim.

The map is intentionally dumb and honest:
  * it stores only the CoeffResults that are USABLE (converged, with lift+drag),
  * it interpolates multilinearly on whatever axes actually varied,
  * it REFUSES to invent a channel: if no result carried `c_side`, the map returns
    None for side force rather than zero, and the lap sim degrades accordingly,
  * it clamps queries to the swept envelope (no silent extrapolation into attitudes
    you never solved) and flags when it does.

A CSV round-trip (`to_csv`/`from_csv`) is first-class so a team can bring a map
from ANY source — the in-house solver, a sponsor's run, last year's data — and
feed the lap sim without KinematiK ever running CFD. That is the "B" path.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Optional

from .cfd import Attitude, CoeffResult, CFDProvenance, SolverFidelity


_AXES = ("roll_deg", "pitch_deg", "yaw_deg", "ride_height_mm", "speed_ms")


@dataclass
class AeroQuery:
    """Result of querying the map: coefficients + whether they were extrapolated."""
    c_lift: Optional[float]
    c_drag: Optional[float]
    c_side: Optional[float]
    aero_balance_front: Optional[float]
    extrapolated: bool = False
    note: str = ""


class AeroMap:
    """
    Gridded aero map with multilinear interpolation. Built from CoeffResults; only
    keeps usable ones. Query with an Attitude (or the lap sim's scalar speed) and
    get back coefficients, clamped to the solved envelope.
    """

    def __init__(self, reference_area_m2: float = 1.0,
                 reference_length_m: float = 1.55,
                 provenance: Optional[CFDProvenance] = None):
        self.reference_area_m2 = reference_area_m2
        self.reference_length_m = reference_length_m
        self.provenance = provenance
        self._points: dict[tuple, CoeffResult] = {}

    # -- construction ------------------------------------------------------ #
    def add(self, result: CoeffResult) -> bool:
        """Add a result; returns True if it was usable and stored."""
        if not result.is_usable():
            return False
        self._points[result.attitude.key()] = result
        if self.provenance is None and result.provenance is not None:
            self.provenance = result.provenance
        return True

    @classmethod
    def from_results(cls, results, reference_area_m2=1.0,
                     reference_length_m=1.55) -> "AeroMap":
        m = cls(reference_area_m2, reference_length_m)
        for r in results:
            m.add(r)
        return m

    def __len__(self) -> int:
        return len(self._points)

    # -- axis introspection ------------------------------------------------ #
    def _axis_values(self, axis: str) -> list[float]:
        vals = sorted({getattr(r.attitude, axis) for r in self._points.values()})
        return vals

    def swept_axes(self) -> list[str]:
        return [a for a in _AXES if len(self._axis_values(a)) > 1]

    def envelope(self) -> dict[str, tuple[float, float]]:
        env = {}
        for a in _AXES:
            vals = self._axis_values(a)
            if vals:
                env[a] = (vals[0], vals[-1])
        return env

    # -- query ------------------------------------------------------------- #
    def query(self, attitude: Attitude) -> AeroQuery:
        """Multilinear interpolation on the swept axes; clamp to the envelope."""
        if not self._points:
            return AeroQuery(None, None, None, None, note="empty map")

        # Clamp the query to the solved envelope per swept axis, flag if we did.
        env = self.envelope()
        clamped = False
        q = {}
        for a in _AXES:
            v = getattr(attitude, a)
            lo, hi = env.get(a, (v, v))
            if v < lo:
                v, clamped = lo, True
            elif v > hi:
                v, clamped = hi, True
            q[a] = v

        # Channels: interpolate each independently; a channel missing anywhere in
        # the surrounding cell yields None for that channel (no fabrication).
        def interp(channel: str) -> Optional[float]:
            return self._interp_channel(channel, q)

        return AeroQuery(
            c_lift=interp("c_lift"),
            c_drag=interp("c_drag"),
            c_side=interp("c_side"),
            aero_balance_front=interp("aero_balance_front"),
            extrapolated=clamped,
            note="clamped to solved envelope" if clamped else "",
        )

    def _interp_channel(self, channel: str, q: dict) -> Optional[float]:
        # Nearest-neighbour weighted multilinear over swept axes only. For a single
        # point this returns that point's value; for a full grid it does true
        # multilinear. Robust to non-perfect grids by inverse-distance fallback.
        axes = self.swept_axes()
        pts = list(self._points.values())
        # exact grid path: find the 2^k bounding box on swept axes
        grids = {a: self._axis_values(a) for a in axes}

        def bracket(a):
            vals = grids[a]
            x = q[a]
            lo = max([v for v in vals if v <= x], default=vals[0])
            hi = min([v for v in vals if v >= x], default=vals[-1])
            return lo, hi

        if axes:
            corners = [{}]
            for a in axes:
                lo, hi = bracket(a)
                nxt = []
                for c in corners:
                    for val in ({lo, hi}):
                        d = dict(c); d[a] = val; nxt.append(d)
                corners = nxt
            # gather corner values; bail to IDW if any corner is absent/missing chan
            total_w = 0.0
            acc = 0.0
            have_all = True
            for c in corners:
                key = self._corner_key(c)
                r = self._points.get(key)
                if r is None:
                    have_all = False
                    break
                cv = getattr(r, channel)
                if cv is None:
                    return None                 # channel genuinely absent -> honest None
                w = 1.0
                for a in axes:
                    lo, hi = bracket(a)
                    if hi > lo:
                        frac = (q[a] - lo) / (hi - lo)
                        w *= (1 - frac) if c[a] == lo else frac
                acc += w * cv
                total_w += w
            if have_all and total_w > 0:
                return acc / total_w

        # Fallback: inverse-distance over all points (also the single-point answer).
        num = 0.0; den = 0.0
        for r in pts:
            cv = getattr(r, channel)
            if cv is None:
                continue
            d2 = 0.0
            for a in (axes or _AXES):
                rng = 1.0
                vals = self._axis_values(a)
                if len(vals) > 1:
                    rng = vals[-1] - vals[0]
                d2 += ((getattr(r.attitude, a) - q[a]) / (rng or 1.0)) ** 2
            if d2 < 1e-12:
                return cv
            w = 1.0 / d2
            num += w * cv; den += w
        return (num / den) if den > 0 else None

    def _corner_key(self, partial: dict) -> tuple:
        # Build a full attitude key for a corner, filling unswept axes from any point.
        any_att = next(iter(self._points.values())).attitude
        vals = {a: getattr(any_att, a) for a in _AXES}
        vals.update(partial)
        att = Attitude(vals["roll_deg"], vals["pitch_deg"], vals["yaw_deg"],
                       vals["ride_height_mm"], vals["speed_ms"])
        return att.key()

    # -- CSV round-trip (the "B" path: bring any map in) ------------------- #
    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["roll_deg", "pitch_deg", "yaw_deg", "ride_height_mm",
                    "speed_ms", "c_lift", "c_drag", "c_side",
                    "aero_balance_front", "converged"])
        for r in self._points.values():
            a = r.attitude
            w.writerow([a.roll_deg, a.pitch_deg, a.yaw_deg, a.ride_height_mm,
                        a.speed_ms, r.c_lift, r.c_drag,
                        "" if r.c_side is None else r.c_side,
                        "" if r.aero_balance_front is None else r.aero_balance_front,
                        int(r.converged)])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str, reference_area_m2=1.0, reference_length_m=1.55,
                 provenance: Optional[CFDProvenance] = None) -> "AeroMap":
        m = cls(reference_area_m2, reference_length_m, provenance)
        rdr = csv.DictReader(io.StringIO(text))
        for row in rdr:
            def fnum(k):
                v = (row.get(k) or "").strip()
                return float(v) if v not in ("", None) else None
            att = Attitude(
                float(row.get("roll_deg", 0) or 0),
                float(row.get("pitch_deg", 0) or 0),
                float(row.get("yaw_deg", 0) or 0),
                float(row.get("ride_height_mm", 30) or 30),
                float(row.get("speed_ms", 20) or 20),
            )
            conv = (row.get("converged", "1") or "1").strip().lower() in ("1", "true", "yes")
            res = CoeffResult(
                attitude=att, c_lift=fnum("c_lift"), c_drag=fnum("c_drag"),
                c_side=fnum("c_side"), aero_balance_front=fnum("aero_balance_front"),
                converged=conv, provenance=provenance,
            )
            m.add(res)
        if provenance is None:
            m.provenance = CFDProvenance(
                backend="imported-csv", fidelity=SolverFidelity.RANS,
                is_correlated=False,
                notes="Imported aero map; provenance is whatever the source was. "
                      "KinematiK did not run or validate this CFD.")
        return m

    def status(self) -> str:
        if not self._points:
            return "empty aero map"
        prov = self.provenance.status() if self.provenance else "provenance unknown"
        return (f"AeroMap: {len(self)} points over {self.swept_axes() or ['single point']}; "
                f"{prov}")
