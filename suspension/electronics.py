# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Electronics / PCB layer — the copper-survival and signal-integrity chain the
integration ledger names but never closes.

`interfaces.py` owns the *interface* between subsystems: each declares a peak
current, a supply voltage, a power draw, and the LV/HV check sums those against
the bus. What it does NOT own is the thing an electrical member actually does the
afternoon before they send a board to fab: pick a trace width, route a CAN pair
past the motor-controller, and need to know, *immediately*,

    1. under the worst simultaneous load (brake-light + both cooling fans firing
       at once), does this copper trace heat past its rating — or in the limit,
       fuse — and how much voltage does it drop on the way to the ECU? A drop
       that pushes the 5 V/3.3 V rail below the microcontroller's brown-out
       threshold resets the car mid-event, and
    2. does a CAN-bus differential pair run close enough to a switching
       high-voltage net (the inverter/motor-controller trace) that injected
       noise threatens the pair — i.e. is the coupling budget blown?

This module adds exactly that, and nothing more. It is deliberately NOT a PCB CAD
kernel and NOT a full-wave field solver — the same non-goal the rest of KinematiK
keeps (it is "not a CAD kernel and not an FEA tool"). It works on explicit traces
and explicit nets the electrical team declares, runs *analytic, standards-backed*
models with zero dependencies beyond numpy:

  * trace DC resistance and voltage drop from copper geometry + temperature,
  * IPC-2221 / IPC-2152 steady-state temperature rise and the fusing
    (Onderdonk / Preece) current at which the trace physically melts,
  * a microstrip / edge-coupled differential-pair impedance estimate and a
    geometric crosstalk / noise-coupling budget between an aggressor HV net and a
    victim signal pair,

and it emits the same typed `Finding` objects the rest of the integration board
already renders, so a melted trace or a noisy CAN pair shows up in the existing UI
with both owners named — exactly the way a mount-point clash does.

Honesty rules the rest of the codebase keeps, kept here too:
  * Anything that genuinely needs a 2-D/3-D field solver — the *true* coupled-line
    SI waveform, eye diagram, reflection from a real stackup — is returned as
    `None` with a stated reason, never as an invented number. The analytic
    coupling budget is labelled an ESTIMATE / screening result, not a measured
    margin, in the same spirit as the tire co-sim backend returning `None` for
    carcass deformation it cannot compute.
  * Provenance is preserved: a `Trace`/`Net` carries who set it and whether it is
    an estimate, and a finding on placeholder geometry is flagged as such.
  * The current the trace must survive is not re-invented here when the subsystem
    already declared it — `peak_current_a` on the `SubsystemInterface` is the
    single source of truth, and a helper rolls up simultaneous loads from the
    ledger so "what fires at once" and "what the interface says it draws" can't
    drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .interfaces import Finding, Severity, IntegrationLedger, SubsystemInterface


# --------------------------------------------------------------------------- #
#  Physical constants (copper on FR-4) — all SI unless a name says otherwise
# --------------------------------------------------------------------------- #
# Copper resistivity at 20 °C and its temperature coefficient.
RHO_CU_20C = 1.724e-8          # ohm·m
ALPHA_CU = 3.93e-3             # 1/°C  (temperature coefficient of resistance)
# 1 oz/ft^2 finished copper ≈ 34.8 µm thick (the FSAE-standard outer-layer weight).
OZ_TO_UM = 34.8                # micrometres of copper per oz/ft^2
# Onderdonk fusing constant for copper (trace melts), amps from area[mil^2], time[s].
# I = A * sqrt( log10( (Tm - Ta)/(234 + Ta) + 1 ) / (33 * t) ) in mil^2 form.
# Vacuum permittivity for the impedance estimates.
EPS0 = 8.854e-12               # F/m


# --------------------------------------------------------------------------- #
#  Geometry / declaration primitives
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    """
    One copper trace (a single net segment) on the board, owned by one subsystem
    and feeding another. This is the power trace the electrical member sizes, or
    one conductor of a differential pair.

    Geometry is the cross-section the IPC charts use plus the run length:
        width_mm   : finished copper width
        copper_oz  : plating weight (1 oz ≈ 34.8 µm thick); thickness derived
        length_mm  : routed length carrying the current
    `is_external` selects the IPC-2221 external (cooler-running, more conservative)
    vs internal (buried, hotter) constant.

    Provenance mirrors MountPoint: who placed it and whether it's still a guess.
    """
    name: str
    net: str                       # logical net, e.g. "fan_pwr", "can_h", "hv_inv"
    owner_subsystem: str           # who routes/owns it, e.g. "electrics"
    feeds: str = "ecu"             # what it powers / connects to
    width_mm: float = 0.25
    copper_oz: float = 1.0         # outer-layer default
    length_mm: float = 100.0
    is_external: bool = True       # outer layer (vs buried inner plane)
    is_estimate: bool = True
    set_by: str = ""
    notes: str = ""

    # ---- derived geometry --------------------------------------------------- #
    @property
    def thickness_mm(self) -> float:
        return self.copper_oz * OZ_TO_UM / 1000.0

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.thickness_mm

    @property
    def area_mil2(self) -> float:
        # 1 mm = 39.3701 mil -> mm^2 to mil^2
        return self.area_mm2 * (39.3701 ** 2)

    # ---- DC electrical ------------------------------------------------------ #
    def resistance_ohm(self, temp_c: float = 20.0) -> float:
        """DC resistance at a given copper temperature (rho varies with T)."""
        rho = RHO_CU_20C * (1.0 + ALPHA_CU * (temp_c - 20.0))
        L = self.length_mm * 1e-3
        A = self.area_mm2 * 1e-6
        return rho * L / A

    def voltage_drop_v(self, current_a: float, temp_c: float = 20.0) -> float:
        """IR drop along the trace at a current and operating temperature."""
        return current_a * self.resistance_ohm(temp_c)

    def power_dissipated_w(self, current_a: float, temp_c: float = 20.0) -> float:
        return current_a ** 2 * self.resistance_ohm(temp_c)

    # ---- IPC-2221 steady-state temperature rise ----------------------------- #
    def temp_rise_c(self, current_a: float) -> float:
        """
        IPC-2221 trace-heating chart in closed form:
            I = k * dT^0.44 * A_mil2^0.725   ->  dT = (I / (k * A^0.725))^(1/0.44)
        with k = 0.048 (external) / 0.024 (internal). Returns steady-state rise
        above ambient in °C for the given DC current. This is the same curve every
        FSAE team reads off the IPC nomograph, evaluated analytically.

        The IPC chart is only valid up to a few hundred °C of rise; the closed form
        extrapolates to non-physical values well past that. We clamp the *reported*
        rise so it can never exceed copper's melting point (~1063 °C above a 20 °C
        reference) — past that the trace has fused and a finite temperature is
        meaningless. `fusing_current_a` is the authoritative "it melts" limit; this
        is the steady-state running temperature below that.
        """
        if current_a <= 0.0 or self.area_mil2 <= 0.0:
            return 0.0
        k = 0.048 if self.is_external else 0.024
        dT = float((current_a / (k * self.area_mil2 ** 0.725)) ** (1.0 / 0.44))
        # copper melts ~1083 °C; cap reported rise there (chart is invalid above)
        return min(dT, 1063.0)

    def fusing_current_a(self, t_s: float = 10.0, ambient_c: float = 25.0) -> float:
        """
        Onderdonk fusing current — the current that melts this copper trace in
        `t_s` seconds from `ambient_c`. This is the "physically melt" limit the
        request asks for. Melting point of copper Tm = 1083 °C.

            I = A_mil2 * sqrt( log10((Tm-Ta)/(234+Ta) + 1) / (33 * t) )
        """
        Tm = 1083.0
        Ta = ambient_c
        if t_s <= 0.0 or self.area_mil2 <= 0.0:
            return float("inf")
        inner = np.log10((Tm - Ta) / (234.0 + Ta) + 1.0) / (33.0 * t_s)
        return float(self.area_mil2 * np.sqrt(inner))

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "Trace":
        d = dict(d)
        valid = Trace.__dataclass_fields__.keys()
        return Trace(**{k: v for k, v in d.items() if k in valid})


@dataclass
class DiffPair:
    """
    A high-speed differential pair (the CAN-bus H/L conductors). Held as a route:
    a centreline polyline in board coordinates (mm) plus the pair geometry. The
    edge-coupled-microstrip impedance estimate and the aggressor-coupling budget
    both read off this.

        trace_w_mm : each conductor width
        spacing_mm : edge-to-edge gap between the two conductors of the pair
        height_mm  : dielectric height to the nearest reference plane
        eps_r      : substrate relative permittivity (FR-4 ≈ 4.3)
        path_mm    : list of (x, y) points the pair routes through
    """
    name: str
    owner_subsystem: str
    net_p: str = "can_h"
    net_n: str = "can_l"
    trace_w_mm: float = 0.20
    spacing_mm: float = 0.20
    height_mm: float = 0.20
    copper_oz: float = 1.0
    eps_r: float = 4.3
    path_mm: list = field(default_factory=list)   # [(x,y), ...] centreline
    target_z0_ohm: float = 120.0                   # CAN differential target
    is_estimate: bool = True
    set_by: str = ""
    notes: str = ""

    @property
    def thickness_mm(self) -> float:
        return self.copper_oz * OZ_TO_UM / 1000.0

    def single_ended_z0_ohm(self) -> float:
        """
        IPC-2141 microstrip single-ended Z0 (analytic, the standard closed form):
            Z0 = 87/sqrt(eps_r+1.41) * ln( 5.98 h / (0.8 w + t) )
        Valid for 0.1 < w/h < 2.0 and 1 < eps_r < 15 — the normal PCB regime.
        """
        h, w, t = self.height_mm, self.trace_w_mm, self.thickness_mm
        if h <= 0 or (0.8 * w + t) <= 0:
            return float("nan")
        return float(87.0 / np.sqrt(self.eps_r + 1.41)
                     * np.log(5.98 * h / (0.8 * w + t)))

    def differential_z0_ohm(self) -> float:
        """
        Edge-coupled differential impedance from the single-ended Z0 with the
        standard coupling correction:
            Zdiff ≈ 2 * Z0 * (1 - 0.48 * exp(-0.96 * s/h))
        An analytic ESTIMATE — good to a few ohms for typical FR-4, NOT a 2-D
        field-solver result. Labelled as such wherever it surfaces.
        """
        z0 = self.single_ended_z0_ohm()
        s, h = self.spacing_mm, self.height_mm
        if h <= 0 or not np.isfinite(z0):
            return float("nan")
        return float(2.0 * z0 * (1.0 - 0.48 * np.exp(-0.96 * s / h)))

    def as_polyline(self) -> np.ndarray:
        if not self.path_mm:
            return np.zeros((0, 2))
        return np.asarray(self.path_mm, dtype=float)

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "DiffPair":
        d = dict(d)
        if isinstance(d.get("path_mm"), list):
            d["path_mm"] = [tuple(p) for p in d["path_mm"]]
        valid = DiffPair.__dataclass_fields__.keys()
        return DiffPair(**{k: v for k, v in d.items() if k in valid})


@dataclass
class Aggressor:
    """
    A noisy net the SI check must keep signal pairs away from — the high-voltage
    motor-controller / inverter trace whose fast switching edges inject noise.
    Held as its own routed polyline plus the electrical quantities that set how
    hard it couples: switched voltage and edge rate.

        sw_voltage_v : peak switched voltage (e.g. 400 V HV bus node)
        edge_v_per_ns: slew rate of the switching edge (faster = worse coupling)
        path_mm      : centreline polyline (mm)
    """
    name: str
    owner_subsystem: str            # e.g. "powertrain" / "electrics"
    net: str = "hv_inverter"
    sw_voltage_v: float = 400.0
    edge_v_per_ns: float = 5.0
    path_mm: list = field(default_factory=list)
    is_estimate: bool = True
    set_by: str = ""
    notes: str = ""

    def as_polyline(self) -> np.ndarray:
        if not self.path_mm:
            return np.zeros((0, 2))
        return np.asarray(self.path_mm, dtype=float)

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "Aggressor":
        d = dict(d)
        if isinstance(d.get("path_mm"), list):
            d["path_mm"] = [tuple(p) for p in d["path_mm"]]
        valid = Aggressor.__dataclass_fields__.keys()
        return Aggressor(**{k: v for k, v in d.items() if k in valid})


# --------------------------------------------------------------------------- #
#  Geometry helper: min distance between two 2-D polylines (segment-to-segment)
# --------------------------------------------------------------------------- #
def _seg_seg_dist(p1, p2, q1, q2) -> float:
    """Exact minimum distance between two 2-D line segments (no sampling)."""
    p1, p2, q1, q2 = map(lambda v: np.asarray(v, float), (p1, p2, q1, q2))
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - q1))
    if a <= 1e-12:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    cp1 = p1 + d1 * s
    cp2 = q1 + d2 * t
    return float(np.linalg.norm(cp1 - cp2))


def min_parallel_distance_mm(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Closest approach (mm) between two routed polylines, or None if either empty."""
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    best = float("inf")
    for i in range(a.shape[0] - 1):
        for j in range(b.shape[0] - 1):
            best = min(best, _seg_seg_dist(a[i], a[i + 1], b[j], b[j + 1]))
    return best


def parallel_run_length_mm(a: np.ndarray, b: np.ndarray, within_mm: float) -> float:
    """
    Approximate length of `a` that runs within `within_mm` of `b` — the coupled
    length that drives crosstalk. Walks `a` in fine steps and measures each step's
    distance to the nearest segment of `b`. Geometric screening, not a field solve.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    coupled = 0.0
    for i in range(a.shape[0] - 1):
        p0, p1 = a[i], a[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len <= 1e-9:
            continue
        n = max(2, int(seg_len / 0.5))   # 0.5 mm sub-steps
        for k in range(n):
            mid = p0 + (p1 - p0) * (k + 0.5) / n
            d = min(_seg_seg_dist(mid, mid + 1e-6, b[j], b[j + 1])
                    for j in range(b.shape[0] - 1))
            if d <= within_mm:
                coupled += seg_len / n
    return coupled


# --------------------------------------------------------------------------- #
#  The board ledger: traces + pairs + aggressors, with the survival + SI checks
# --------------------------------------------------------------------------- #
@dataclass
class BoardLedger:
    """
    Holds the traces the electrical team sizes, the differential pairs they route,
    and the aggressor nets they must avoid, and runs the two checks that close the
    electrical half of the integration chain:

       * copper survival  — IPC-2221 heating + Onderdonk fusing + IR-drop/brown-out,
       * signal integrity  — differential-pair impedance + HV-aggressor coupling.

    Findings carry both owners (e.g. the trace owner and the load it feeds, or the
    pair owner and the aggressor owner), so a melted fan trace or a CAN pair routed
    too close to the inverter shows up on the same board as a suspension clash.

    Brown-out modelling: a rail that feeds the ECU has a nominal voltage and a
    brown-out threshold; if simultaneous-load IR drop pulls the delivered voltage
    below threshold, that's the "resets the ECU" FAIL the request calls out.
    """
    traces: dict = field(default_factory=dict)       # name -> Trace
    pairs: dict = field(default_factory=dict)        # name -> DiffPair
    aggressors: dict = field(default_factory=dict)   # name -> Aggressor

    # rail context for brown-out checks
    rail_nominal_v: float = 5.0
    ecu_brownout_v: float = 4.5
    ambient_c: float = 40.0          # underhood ambient for FSAE EV (hot)
    max_trace_temp_c: float = 105.0  # typical FR-4 / connector derate ceiling
    fuse_safety_factor: float = 2.0  # require fusing current >= SF * peak current
    coupling_warn_mm: float = 2.0    # pair-to-aggressor gap that triggers WARN
    coupling_fail_mm: float = 0.5    # gap that triggers FAIL

    def set_trace(self, t: Trace):
        self.traces[t.name] = t

    def set_pair(self, p: DiffPair):
        self.pairs[p.name] = p

    def set_aggressor(self, a: Aggressor):
        self.aggressors[a.name] = a

    # ---- copper survival check --------------------------------------------- #
    def check_traces(self, currents: Optional[dict] = None,
                     undeclared: Optional[dict] = None) -> list:
        """
        For every trace, against the current it must carry, emit a Finding for:
          * fusing margin (does it physically melt under the worst load?),
          * steady-state temperature (does it exceed the board's derate ceiling?),
          * IR drop / ECU brown-out (does the delivered voltage reset the micro?).

        `currents` maps trace-name -> amps for the worst simultaneous load. When a
        trace has no entry, no current finding is emitted for it (MISSING-style),
        rather than inventing a load.

        `undeclared` maps trace-name -> list of subsystems the user *did* select for
        that trace but which carry no declared peak_current_a. This lets the MISSING
        finding name the specific picked subsystem(s) to fix, so the check responds
        to the chosen subsystem combination instead of reporting a generic blank.
        """
        out: list = []
        currents = currents or {}
        undeclared = undeclared or {}
        if not self.traces:
            return out
        for tr in self.traces.values():
            I = currents.get(tr.name)
            est = tr.is_estimate
            tag = " (estimated geometry)" if est else ""
            pair = sorted({tr.owner_subsystem, tr.feeds})
            if I is None:
                picked_missing = undeclared.get(tr.name) or []
                if picked_missing:
                    # The user DID pick load(s) for this trace, but the selected
                    # subsystem(s) have no declared peak current — so name them.
                    who = ", ".join(sorted(set(picked_missing)))
                    msg = (f"Trace '{tr.name}' ({tr.owner_subsystem}->{tr.feeds}) has "
                           f"selected load(s) [{who}], but {'that subsystem has' if len(set(picked_missing))==1 else 'those subsystems have'} "
                           f"no declared peak current — cannot check heating or drop. "
                           f"Declare each one's peak current in the Integration tab.")
                else:
                    msg = (f"Trace '{tr.name}' ({tr.owner_subsystem}->{tr.feeds}) has no "
                           f"declared worst-case current — cannot check heating or drop. "
                           f"Pick its load(s) in the scenario above, and declare each "
                           f"subsystem's peak current in the Integration tab.")
                out.append(Finding(
                    "trace-current", Severity.MISSING, msg,
                    subsystems=pair,
                    detail=dict(trace=tr.name, width_mm=tr.width_mm,
                                copper_oz=tr.copper_oz,
                                undeclared_loads=sorted(set(picked_missing)))))
                continue

            # --- fusing (physical melt) --- #
            i_fuse = tr.fusing_current_a(ambient_c=self.ambient_c)
            if I >= i_fuse:
                out.append(Finding(
                    "trace-fusing", Severity.FAIL,
                    f"Trace '{tr.name}' carries {I:.1f} A but fuses (melts) at "
                    f"{i_fuse:.1f} A — the copper physically fails under this "
                    f"load{tag}.",
                    subsystems=pair,
                    detail=dict(trace=tr.name, current_a=I, fusing_a=i_fuse,
                                estimate=est)))
            elif I * self.fuse_safety_factor >= i_fuse:
                out.append(Finding(
                    "trace-fusing", Severity.WARN,
                    f"Trace '{tr.name}' at {I:.1f} A is within {self.fuse_safety_factor:.0f}x "
                    f"of its {i_fuse:.1f} A fusing current — thin margin to melt{tag}.",
                    subsystems=pair,
                    detail=dict(trace=tr.name, current_a=I, fusing_a=i_fuse,
                                safety_factor=self.fuse_safety_factor, estimate=est)))

            # --- steady-state heating --- #
            dT = tr.temp_rise_c(I)
            T = self.ambient_c + dT
            if T > self.max_trace_temp_c:
                out.append(Finding(
                    "trace-heating", Severity.FAIL,
                    f"Trace '{tr.name}' runs to {T:.0f} °C at {I:.1f} A "
                    f"(rise {dT:.0f} °C over {self.ambient_c:.0f} °C ambient), past "
                    f"the {self.max_trace_temp_c:.0f} °C ceiling — widen it{tag}.",
                    subsystems=pair,
                    detail=dict(trace=tr.name, current_a=I, temp_c=T,
                                rise_c=dT, ceiling_c=self.max_trace_temp_c,
                                estimate=est)))
            elif T > self.max_trace_temp_c - 20.0:
                out.append(Finding(
                    "trace-heating", Severity.WARN,
                    f"Trace '{tr.name}' reaches {T:.0f} °C at {I:.1f} A — within "
                    f"20 °C of the {self.max_trace_temp_c:.0f} °C ceiling{tag}.",
                    subsystems=pair,
                    detail=dict(trace=tr.name, current_a=I, temp_c=T,
                                ceiling_c=self.max_trace_temp_c, estimate=est)))

            # --- IR drop / ECU brown-out --- #
            # use the hot resistance, not the 20 °C value — drop is worse when warm
            vdrop = tr.voltage_drop_v(I, temp_c=T)
            already_fused = I >= i_fuse
            if tr.feeds == "ecu" or tr.net.lower() in ("ecu_pwr", "mcu_pwr", "lv_rail"):
                delivered = self.rail_nominal_v - vdrop
                if already_fused:
                    # the trace melts before steady state — IR drop is academic.
                    out.append(Finding(
                        "ecu-brownout", Severity.FAIL,
                        f"Trace '{tr.name}' fuses before steady state at {I:.1f} A, "
                        f"so the ECU rail is lost entirely (open circuit), not merely "
                        f"browned out{tag}.",
                        subsystems=sorted({tr.owner_subsystem, "ecu"}),
                        detail=dict(trace=tr.name, current_a=I, fused=True,
                                    brownout_v=self.ecu_brownout_v, estimate=est)))
                elif delivered < self.ecu_brownout_v:
                    out.append(Finding(
                        "ecu-brownout", Severity.FAIL,
                        f"Trace '{tr.name}' drops {vdrop*1000:.0f} mV at {I:.1f} A, "
                        f"pulling the {self.rail_nominal_v:.1f} V rail to "
                        f"{delivered:.2f} V — below the {self.ecu_brownout_v:.2f} V "
                        f"ECU brown-out threshold. The main MCU would reset{tag}.",
                        subsystems=sorted({tr.owner_subsystem, "ecu"}),
                        detail=dict(trace=tr.name, current_a=I, vdrop_v=vdrop,
                                    delivered_v=delivered,
                                    brownout_v=self.ecu_brownout_v, estimate=est)))
                else:
                    out.append(Finding(
                        "ecu-brownout", Severity.OK,
                        f"Trace '{tr.name}' holds the rail at {delivered:.2f} V "
                        f"under {I:.1f} A (>{self.ecu_brownout_v:.2f} V brown-out).",
                        subsystems=sorted({tr.owner_subsystem, "ecu"}),
                        detail=dict(trace=tr.name, delivered_v=delivered)))
            elif vdrop > 0.05 * self.rail_nominal_v:
                out.append(Finding(
                    "trace-vdrop", Severity.WARN,
                    f"Trace '{tr.name}' drops {vdrop*1000:.0f} mV at {I:.1f} A "
                    f"(> 5 % of a {self.rail_nominal_v:.1f} V rail){tag}.",
                    subsystems=pair,
                    detail=dict(trace=tr.name, vdrop_v=vdrop, estimate=est)))

        if not any(f.severity in (Severity.FAIL, Severity.WARN) for f in out) and out:
            out.append(Finding(
                "trace-survival", Severity.OK,
                f"All checked traces survive their worst-case current with margin "
                f"to fusing, the {self.max_trace_temp_c:.0f} °C ceiling, and the "
                f"ECU brown-out threshold.",
                subsystems=["electrics", "ecu"]))
        return out

    # ---- signal-integrity check -------------------------------------------- #
    def check_signal_integrity(self) -> list:
        """
        For every differential pair: check the analytic differential impedance
        against its target, then for every aggressor net check the closest
        approach and coupled run length. A pair routed inside the FAIL gap of an
        HV switching net is a hard finding; inside the WARN gap is a warning.

        The impedance and coupling numbers are analytic ESTIMATES (screening), not
        field-solver results — every finding says so, and the true coupled-line
        waveform / eye is returned as None by `si_detail()` rather than faked.
        """
        out: list = []
        if not self.pairs:
            return out
        for dp in self.pairs.values():
            zdiff = dp.differential_z0_ohm()
            est = dp.is_estimate
            tag = " (analytic estimate, not field-solved)"
            # impedance vs target
            if np.isfinite(zdiff):
                err = abs(zdiff - dp.target_z0_ohm)
                if err > 0.15 * dp.target_z0_ohm:
                    out.append(Finding(
                        "diffpair-impedance", Severity.WARN,
                        f"Pair '{dp.name}' estimates {zdiff:.0f} Ω differential vs a "
                        f"{dp.target_z0_ohm:.0f} Ω target ({err/dp.target_z0_ohm*100:.0f} % off)"
                        f"{tag} — retune width/spacing or confirm on a field solver.",
                        subsystems=[dp.owner_subsystem],
                        detail=dict(pair=dp.name, z_diff_ohm=zdiff,
                                    target_ohm=dp.target_z0_ohm, estimate=True)))
                else:
                    out.append(Finding(
                        "diffpair-impedance", Severity.OK,
                        f"Pair '{dp.name}' estimates {zdiff:.0f} Ω differential, "
                        f"within 15 % of the {dp.target_z0_ohm:.0f} Ω target{tag}.",
                        subsystems=[dp.owner_subsystem],
                        detail=dict(pair=dp.name, z_diff_ohm=zdiff,
                                    target_ohm=dp.target_z0_ohm, estimate=True)))

            # aggressor coupling
            a_poly = dp.as_polyline()
            if a_poly.shape[0] < 2:
                out.append(Finding(
                    "diffpair-route", Severity.MISSING,
                    f"Pair '{dp.name}' has no routed centreline — cannot check its "
                    f"distance to noisy nets yet.",
                    subsystems=[dp.owner_subsystem]))
                continue
            for ag in self.aggressors.values():
                b_poly = ag.as_polyline()
                gap = min_parallel_distance_mm(a_poly, b_poly)
                if gap is None:
                    continue
                run = parallel_run_length_mm(a_poly, b_poly,
                                             within_mm=max(self.coupling_warn_mm, gap + 0.01))
                est2 = est or ag.is_estimate
                etag = " (estimated routing)" if est2 else ""
                pair = sorted({dp.owner_subsystem, ag.owner_subsystem})
                # qualitative coupling severity from edge rate * (1/gap) * run
                if gap <= self.coupling_fail_mm:
                    out.append(Finding(
                        "si-crosstalk", Severity.FAIL,
                        f"CAN pair '{dp.name}' runs {gap:.2f} mm from "
                        f"{ag.owner_subsystem}'s switching net '{ag.net}' "
                        f"({ag.sw_voltage_v:.0f} V, {ag.edge_v_per_ns:.0f} V/ns edges) "
                        f"for ~{run:.0f} mm — the differential pair is exposed to "
                        f"motor-controller noise. Move the route or add a guard "
                        f"ground{etag}.",
                        subsystems=pair,
                        detail=dict(pair=dp.name, aggressor=ag.name, gap_mm=gap,
                                    coupled_len_mm=run, sw_voltage_v=ag.sw_voltage_v,
                                    edge_v_per_ns=ag.edge_v_per_ns, estimate=est2,
                                    method="geometric screening, not field-solved")))
                elif gap <= self.coupling_warn_mm:
                    out.append(Finding(
                        "si-crosstalk", Severity.WARN,
                        f"CAN pair '{dp.name}' passes within {gap:.2f} mm of "
                        f"{ag.owner_subsystem}'s '{ag.net}' for ~{run:.0f} mm — "
                        f"tighten spacing-to-HV or add isolation{etag}.",
                        subsystems=pair,
                        detail=dict(pair=dp.name, aggressor=ag.name, gap_mm=gap,
                                    coupled_len_mm=run, estimate=est2,
                                    method="geometric screening, not field-solved")))
                else:
                    out.append(Finding(
                        "si-crosstalk", Severity.OK,
                        f"CAN pair '{dp.name}' clears '{ag.net}' by {gap:.1f} mm "
                        f"(> {self.coupling_warn_mm:.0f} mm){etag}.",
                        subsystems=pair,
                        detail=dict(pair=dp.name, aggressor=ag.name, gap_mm=gap,
                                    estimate=est2)))
        return out

    def si_detail(self, pair_name: str) -> dict:
        """
        Return the analytic SI quantities for one pair, and explicitly None for the
        channels that need a real coupled-line field solver / SPICE — never a faked
        eye height or reflection. Same contract the tire co-sim backend uses when it
        returns None for what it cannot compute.
        """
        dp = self.pairs.get(pair_name)
        if dp is None:
            return {}
        return dict(
            pair=pair_name,
            single_ended_z0_ohm=dp.single_ended_z0_ohm(),
            differential_z0_ohm=dp.differential_z0_ohm(),
            method="IPC-2141 microstrip + edge-coupling correction (analytic)",
            # things that genuinely need a field solver / transient sim:
            eye_height_v=None,
            insertion_loss_db=None,
            reflection_coeff=None,
            coupled_noise_v=None,
            note="impedance is an analytic screening estimate; eye/loss/reflection "
                 "require a 2-D field solver and transient sim and are intentionally "
                 "not invented here.",
        )

    # ---- persistence -------------------------------------------------------- #
    def as_dict(self):
        return dict(
            traces={k: v.as_dict() for k, v in self.traces.items()},
            pairs={k: v.as_dict() for k, v in self.pairs.items()},
            aggressors={k: v.as_dict() for k, v in self.aggressors.items()},
            rail_nominal_v=self.rail_nominal_v,
            ecu_brownout_v=self.ecu_brownout_v,
            ambient_c=self.ambient_c,
            max_trace_temp_c=self.max_trace_temp_c,
            fuse_safety_factor=self.fuse_safety_factor,
            coupling_warn_mm=self.coupling_warn_mm,
            coupling_fail_mm=self.coupling_fail_mm,
        )

    @staticmethod
    def from_dict(d) -> "BoardLedger":
        d = d or {}
        bl = BoardLedger()
        for k, v in (d.get("traces") or {}).items():
            bl.set_trace(Trace.from_dict(v))
        for k, v in (d.get("pairs") or {}).items():
            bl.set_pair(DiffPair.from_dict(v))
        for k, v in (d.get("aggressors") or {}).items():
            bl.set_aggressor(Aggressor.from_dict(v))
        for sk in ("rail_nominal_v", "ecu_brownout_v", "ambient_c",
                   "max_trace_temp_c", "fuse_safety_factor",
                   "coupling_warn_mm", "coupling_fail_mm"):
            if d.get(sk) is not None:
                setattr(bl, sk, d[sk])
        return bl


# --------------------------------------------------------------------------- #
#  Simultaneous-load roll-up — "what fires at once" from the integration ledger
# --------------------------------------------------------------------------- #
def worst_case_currents(board: BoardLedger,
                        ledger: Optional[IntegrationLedger],
                        scenario: dict) -> dict:
    """
    Build the worst-case current per trace for a named simultaneous-load scenario —
    e.g. {"fan_pwr": ["cooling"], "brake_pwr": ["brakes"]} meaning the fan trace
    carries the cooling subsystem's peak current and the brake trace the brakes'.
    The per-subsystem current is read from the SubsystemInterface.peak_current_a on
    the integration ledger (the single source of truth), NOT re-typed here, so the
    "brake light + both fans at once" load and the declared interface can't drift.

    `scenario` maps trace-name -> list of subsystem names whose loads sum onto that
    trace simultaneously. A subsystem with no declared peak_current_a contributes
    nothing and is silently skipped (the trace check will still flag MISSING if a
    trace ends up with no load). To find out *which* picked subsystems were
    undeclared (so the UI can name them), use `undeclared_loads()`.
    """
    currents: dict = {}
    for trace_name, subsystems in scenario.items():
        total = 0.0
        any_declared = False
        for s in subsystems:
            it = ledger.interfaces.get(s) if ledger else None
            if it is not None and it.peak_current_a is not None:
                total += float(it.peak_current_a)
                any_declared = True
        if any_declared:
            currents[trace_name] = total
    return currents


def undeclared_loads(ledger: Optional[IntegrationLedger],
                     scenario: dict) -> dict:
    """
    For each trace in the scenario, return the subsystems the user selected that
    have NO declared peak_current_a on the integration ledger. This is what lets
    the board check tell the user *which* picked subsystem is missing its amps —
    so the finding reflects the chosen subsystem combination rather than a generic
    "no declared current" that never changes with the selection.

    Returns {trace_name: [subsystem, ...]} containing only traces that have at
    least one undeclared picked subsystem.
    """
    out: dict = {}
    for trace_name, subsystems in scenario.items():
        missing = []
        for s in (subsystems or []):
            it = ledger.interfaces.get(s) if ledger else None
            if it is None or it.peak_current_a is None:
                missing.append(s)
        if missing:
            out[trace_name] = missing
    return out


# --------------------------------------------------------------------------- #
#  One-call board check — the "before you send it to fab" gate
# --------------------------------------------------------------------------- #
@dataclass
class BoardCheckResult:
    """Bundles the survival + SI findings with a one-line headline."""
    findings: list = field(default_factory=list)

    def has_hard_fail(self) -> bool:
        return any(f.severity == Severity.FAIL for f in self.findings)

    def summary(self) -> str:
        from .interfaces import summarize
        s = summarize(self.findings)
        return (f"{s['worst'].upper()}: "
                f"{s['counts'].get('fail',0)} fail / "
                f"{s['counts'].get('warning',0)} warn / "
                f"{s['counts'].get('ok',0)} ok across "
                f"{len(self.findings)} board checks")


def check_board(board: BoardLedger,
                ledger: Optional[IntegrationLedger] = None,
                scenario: Optional[dict] = None) -> BoardCheckResult:
    """
    Run the full pre-fab board gate: roll up simultaneous loads from the integration
    ledger for the given scenario, run copper-survival on the traces and
    signal-integrity on the pairs, and return all findings on the same typed
    surface the integration board renders. This is the one call a student makes
    before committing a board to manufacture.
    """
    scenario = scenario or {}
    currents = worst_case_currents(board, ledger, scenario)
    missing = undeclared_loads(ledger, scenario)
    findings = []
    # Pass `undeclared` only if this BoardLedger's check_traces actually accepts it.
    # Guards against a partial deploy where a newer check_board meets an older
    # check_traces (or vice-versa) — better to lose the richer MISSING wording than
    # to crash the entire pre-fab gate with a TypeError.
    try:
        import inspect
        _accepts_undeclared = "undeclared" in inspect.signature(
            board.check_traces).parameters
    except (TypeError, ValueError):
        _accepts_undeclared = False
    if _accepts_undeclared:
        findings += board.check_traces(currents, undeclared=missing)
    else:
        findings += board.check_traces(currents)
    findings += board.check_signal_integrity()
    return BoardCheckResult(findings=findings)
