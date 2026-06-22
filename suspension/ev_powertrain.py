# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
EV powertrain & energy layer — the part an underfunded FSAE-EV team wins on.

WHY THIS MODULE EXISTS
----------------------
KinematiK was born combustion. Its lap sim (`lapsim.py`) carries a single
`power_w` cap and a `drive_grip_frac` friction-circle proxy — fine for a car
with one engine and a diff, but it cannot answer the three questions that
actually decide an FSAE-EV team's season:

  1. ARCHITECTURE  — one motor + diff, two motors (axle split), or four
     hub/upright motors (full torque vectoring)? This is an expensive,
     hard-to-reverse decision (motor count, inverter count, cost, mass,
     control complexity, rules/cost-event impact). It must be made ONCE, and
     it should be made on seconds, not vibes.

  2. ENERGY        — can the car finish the 22 km endurance event on its pack
     without derating, and if it must shed power to finish, what does that
     cost in lap time? A funded team brute-forces this with a big pack; you
     predict it and size the pack you can afford.

  3. REGEN         — how much energy does regenerative braking actually put
     back, and how does the regen torque (applied at the driven axle only)
     interact with brake balance and the anti-dive the rest of the tool models?

This module answers all three by WRAPPING the existing `LapSimulator`, not
replacing it. It reuses the verified QSS speed/grip/aero solution and adds an
architecture-aware tractive-limit model and an energy integral on top. It is
the EV analogue of how `aero/` and `tire_cosim` extend the tool at a seam.

HONEST SCOPE (same contract as the rest of KinematiK)
-----------------------------------------------------
- The *traction-limit* difference between architectures is modelled from first
  principles: an open diff is limited by the LESS-loaded driven wheel in a
  corner-exit (load transfer steals its grip), a torque-vectored car can use
  each wheel up to its own load-limited grip. That delta is real, defensible,
  and is the core reason TV buys lap time. It is computed here.
- The *yaw-moment* benefit of torque vectoring (using a left/right torque
  difference to rotate the car and reduce understeer) is REAL but is a
  closed-loop control behaviour the QSS point-mass cannot resolve. It is
  reported as a SEPARATE, clearly-flagged upper-bound estimate — never folded
  silently into the lap time. We will not fake a number QSS can't earn.
- Energy is integrated from the QSS power demand (tractive power + drag +
  rolling, minus regen capture). Thermal derate of the pack/motors over a long
  run is modelled as a simple energy-fraction trigger, flagged as a planning
  estimate, not a validated battery model.

Never raises. Every entry point returns a result object with `.warnings`,
mirroring `lapsim.LapResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .lapsim import LapSimulator, LapSimParams, Track, LapResult
from .dynamics import VehicleDynamics


# --------------------------------------------------------------------------- #
#  Architecture enum
# --------------------------------------------------------------------------- #
class Powertrain(str, Enum):
    """The three EV architectures an FSAE team realistically chooses between."""
    SINGLE_DIFF = "single_diff"      # 1 motor + open/limited-slip diff
    TWO_AXLE = "two_axle"            # 2 motors, axle (front/rear) torque split
    FOUR_TV = "four_tv"             # 4 hub/upright motors, full torque vectoring

    def label(self) -> str:
        return {
            "single_diff": "Single motor + diff",
            "two_axle": "Two motors (axle split)",
            "four_tv": "Four motors (torque vectoring)",
        }[self.value]

    def n_motors(self) -> int:
        return {"single_diff": 1, "two_axle": 2, "four_tv": 4}[self.value]


# --------------------------------------------------------------------------- #
#  EV parameters
# --------------------------------------------------------------------------- #
@dataclass
class EVParams:
    """
    EV-specific parameters layered ON TOP of LapSimParams. FSAE-EV-representative
    defaults so it runs out of the box; every one is a knob from spec/measured
    data. Power is kept consistent with LapSimParams.power_w (the FS rules cap is
    80 kW; many EV cars run lower for energy reasons).

    Mass deltas per architecture let the comparison be honest: more motors and
    inverters weigh more, and that mass costs lap time too — so TV must pay for
    its own weight before it shows a net gain.
    """
    # Pack
    pack_energy_kwh: float = 6.5          # usable pack energy, kWh (typical FSAE-EV)
    pack_usable_frac: float = 0.92        # fraction of nameplate energy actually usable
    # Regen
    regen_enabled: bool = True
    regen_eff: float = 0.55               # round-trip capture efficiency of braking energy
    regen_max_g: float = 0.35             # max decel the driven axle can regen (rest is friction)
    # Drivetrain efficiency chain (battery -> wheels), traction direction
    inverter_motor_eff: float = 0.90      # combined inverter+motor efficiency
    # Per-architecture curb-mass delta vs the single-motor baseline, kg.
    # (extra motors, inverters, wiring, cooling). Defensible planning numbers.
    mass_delta_kg: dict = field(default_factory=lambda: {
        Powertrain.SINGLE_DIFF: 0.0,
        Powertrain.TWO_AXLE: 7.0,
        Powertrain.FOUR_TV: 16.0,
    })
    # Drive-grip fraction each architecture can USE on corner exit.
    # Physics: at a power-down corner exit, lateral load transfer unloads the
    # inside driven wheel. An OPEN DIFF is capped by that inside wheel, so it
    # can deploy a smaller fraction of axle grip longitudinally before it spins
    # the inside tyre. Independent per-wheel control (TV) deploys each wheel to
    # its own limit and recovers most of that loss. These fractions multiply the
    # tyre mu in the tractive limit.
    drive_grip_frac: dict = field(default_factory=lambda: {
        Powertrain.SINGLE_DIFF: 0.78,     # inside-wheel limited (open diff)
        Powertrain.TWO_AXLE: 0.88,         # 2 axles share, still no L/R control
        Powertrain.FOUR_TV: 0.98,          # each wheel to its own load-limit
    })
    # Upper-bound yaw-moment lap-time benefit of L/R torque vectoring, as a
    # FRACTION of lap time, reported separately (NOT auto-applied). Conservative
    # literature-scale band for FSAE autocross; flagged as control-dependent.
    tv_yaw_benefit_frac: dict = field(default_factory=lambda: {
        Powertrain.SINGLE_DIFF: 0.0,
        Powertrain.TWO_AXLE: 0.0,          # axle split gives no L/R yaw moment
        Powertrain.FOUR_TV: 0.015,         # ~1.5% upper bound, control-dependent
    })
    g: float = 9.81


# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #
@dataclass
class EVRunResult:
    """One architecture on one track: lap time + energy story."""
    architecture: Powertrain
    ok: bool
    lap_result: LapResult                 # the underlying QSS result (traces, limits)
    lap_time: float                       # s, single lap (mirrors lap_result for ease)
    event_time: float                     # s, lap_time * laps
    energy_per_lap_kwh: float             # net tractive energy drawn per lap (after regen)
    regen_recovered_kwh: float            # energy returned by regen per lap
    energy_full_event_kwh: float          # projected energy for the full event distance
    laps_until_empty: float               # how many laps the pack lasts (no derate)
    finishes_event: bool                  # does pack outlast the event distance?
    derate_lap_time_penalty_s: float      # extra s/lap if power must be shed to finish
    # Torque-vectoring yaw benefit, reported separately, NEVER folded into lap_time
    tv_yaw_benefit_s: float               # upper-bound s/lap from L/R yaw moment
    tv_yaw_benefit_flagged: str           # the honesty flag describing what this is
    effective_mass_kg: float              # mass used (baseline + architecture delta)
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def failed(arch: Powertrain, warnings: list[str]) -> "EVRunResult":
        return EVRunResult(
            architecture=arch, ok=False,
            lap_result=LapResult.failed("ev", warnings),
            lap_time=float("nan"), event_time=float("nan"),
            energy_per_lap_kwh=float("nan"), regen_recovered_kwh=float("nan"),
            energy_full_event_kwh=float("nan"), laps_until_empty=float("nan"),
            finishes_event=False, derate_lap_time_penalty_s=float("nan"),
            tv_yaw_benefit_s=0.0,
            tv_yaw_benefit_flagged="not evaluated (run failed)",
            effective_mass_kg=float("nan"), warnings=list(warnings),
        )


@dataclass
class ArchitectureComparison:
    """The deliverable: all chosen architectures ranked on the same car/track."""
    track_name: str
    results: list[EVRunResult]
    warnings: list[str] = field(default_factory=list)

    def best_on_time(self) -> Optional[EVRunResult]:
        ok = [r for r in self.results if r.ok and np.isfinite(r.event_time)]
        return min(ok, key=lambda r: r.event_time) if ok else None

    def summary(self) -> str:
        lines = [f"Architecture comparison on {self.track_name}:"]
        ok = [r for r in self.results if r.ok and np.isfinite(r.event_time)]
        if not ok:
            return "\n".join(lines + ["  (no architecture produced a usable run)"])
        base = min(r.event_time for r in ok)
        for r in sorted(self.results, key=lambda r: (not r.ok, r.event_time)):
            if not r.ok or not np.isfinite(r.event_time):
                lines.append(f"  ✗ {r.architecture.label():32s} — run failed")
                continue
            d = r.event_time - base
            tag = "  ← best" if abs(d) < 1e-6 else f"(+{d:5.2f} s)"
            fin = "finishes" if r.finishes_event else f"EMPTY at {r.laps_until_empty:.1f} laps"
            lines.append(
                f"  {r.architecture.label():32s} {r.event_time:7.2f} s {tag:>12s}  "
                f"| {r.energy_full_event_kwh:5.2f} kWh, {fin}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  The EV simulator
# --------------------------------------------------------------------------- #
class EVLapSimulator:
    """
    Wraps a VehicleDynamics + LapSimParams and runs the QSS lap sim for any EV
    architecture, then layers architecture-aware traction limits and a full
    energy budget on top. Never raises.
    """

    def __init__(self, veh: VehicleDynamics, base_params: LapSimParams | None = None,
                 ev: EVParams | None = None, ds: float = 1.0):
        self.veh = veh
        self.base = base_params or LapSimParams()
        self.ev = ev or EVParams()
        self.ds = ds
        self.warnings: list[str] = []

    def _warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _params_for(self, arch: Powertrain) -> LapSimParams:
        """Clone base LapSimParams and apply architecture-specific mass + grip."""
        import copy
        p = copy.deepcopy(self.base)
        try:
            p.mass = float(self.base.mass) + float(self.ev.mass_delta_kg.get(arch, 0.0))
            p.drive_grip_frac = float(self.ev.drive_grip_frac.get(arch, p.drive_grip_frac))
            # keep power consistent: EV pack feeds the same wheel-power cap
        except Exception:
            self._warn("Architecture parameter application failed; used base params.")
        return p

    # ---- energy integral from a QSS trace ------------------------------- #
    def _energy_from_trace(self, lap: LapResult, p: LapSimParams) -> tuple[float, float]:
        """
        Integrate net tractive energy (kWh) and regen recovered (kWh) over ONE lap
        from the QSS speed/long_g trace. Tractive energy is drawn through the
        inverter+motor efficiency; regen returns braking energy at regen_eff,
        capped by regen_max_g (only the driven-axle share within that decel is
        recoverable; the rest is friction brakes).

        Defensive: any non-finite point is skipped, never poisons the integral.
        """
        ev = self.ev
        try:
            v = np.asarray(lap.speed, float)
            d = np.asarray(lap.distance, float)
            lg = np.asarray(lap.long_g, float)
            if v.size < 2:
                return float("nan"), float("nan")
            traction_J = 0.0
            regen_J = 0.0
            for i in range(1, v.size):
                ds = d[i] - d[i - 1]
                if not (np.isfinite(ds) and ds > 0):
                    continue
                vi = max(v[i], p.V_MIN)
                a = lg[i] * p.g                       # m/s^2 (signed)
                # inertial + resistive force the wheels must supply (accel side)
                F_drag = 0.5 * p.rho * p.cd_a * vi * vi
                F_roll = p.rolling_g * p.mass * p.g
                if a > 0:                             # accelerating: draw energy
                    F_trac = p.mass * a + F_drag + F_roll
                    if F_trac > 0:
                        # energy at the battery = wheel work / efficiency
                        traction_J += (F_trac * ds) / max(ev.inverter_motor_eff, 1e-3)
                elif a < 0:                           # braking: regen what we can
                    if ev.regen_enabled:
                        a_regen = min(-a, ev.regen_max_g * p.g)  # decel captured
                        F_regen = p.mass * a_regen
                        regen_J += F_regen * ds * ev.regen_eff
            kwh = 1.0 / 3.6e6
            net = max(traction_J - (regen_J if ev.regen_enabled else 0.0), 0.0)
            return net * kwh, regen_J * kwh
        except Exception:
            self._warn("Energy integration failed; energy values flagged NaN.")
            return float("nan"), float("nan")

    # ---- one architecture ----------------------------------------------- #
    def run_architecture(self, arch: Powertrain, track: Track) -> EVRunResult:
        # LapSimulator overwrites p.mass from veh.p.mass on construction, so the
        # per-architecture mass penalty must be applied to the VEHICLE mass for
        # the duration of this run, then restored — otherwise heavier
        # architectures would look free of their own weight (a lie this tool
        # refuses to tell). We snapshot and always restore, even on failure.
        _saved_mass = None
        try:
            p = self._params_for(arch)
            try:
                if getattr(self.veh, "p", None) is not None:
                    _saved_mass = float(self.veh.p.mass)
                    self.veh.p.mass = p.mass
            except Exception:
                self._warn("Could not apply architecture mass to vehicle; "
                           "mass penalty may be understated.")
            sim = LapSimulator(self.veh, p, ds=self.ds)
            lap = sim.simulate(track)
            for w in lap.warnings:
                self._warn(w)
            if not lap.ok:
                return EVRunResult.failed(arch, lap.warnings or ["QSS lap failed"])

            e_lap, regen = self._energy_from_trace(lap, p)
            laps = max(getattr(track, "laps", 1), 1)
            e_event = e_lap * laps if np.isfinite(e_lap) else float("nan")
            usable = self.ev.pack_energy_kwh * self.ev.pack_usable_frac

            if np.isfinite(e_lap) and e_lap > 1e-9:
                laps_until_empty = usable / e_lap
            else:
                laps_until_empty = float("inf")
            finishes = np.isfinite(laps_until_empty) and laps_until_empty >= laps

            # If it can't finish, estimate the lap-time penalty of derating power
            # to the energy budget. Approx: if you must cut energy by factor f,
            # power scales ~f, and on a power-limited car lap time scales ~ f^-k.
            derate_pen = 0.0
            if not finishes and np.isfinite(e_event) and e_event > usable:
                f = usable / e_event                  # fraction of needed energy available
                # crude but honest: ~30% of a lap is power-limited; shedding power
                # by (1-f) costs roughly that fraction of that portion in time.
                derate_pen = lap.lap_time * 0.30 * (1.0 - f)
                self._warn(
                    f"{arch.label()}: pack does not cover the event; a ~"
                    f"{(1-f)*100:.0f}% energy shortfall implies derating "
                    f"(+{derate_pen:.2f} s/lap est., planning-grade)."
                )

            # Torque-vectoring yaw benefit — reported SEPARATELY, never folded in.
            tv_frac = self.ev.tv_yaw_benefit_frac.get(arch, 0.0)
            tv_s = lap.lap_time * tv_frac
            tv_flag = (
                "QSS cannot resolve closed-loop L/R yaw control; this is an "
                "UPPER-BOUND estimate from a literature-scale fraction, shown "
                "separately and NOT added to the lap time."
                if tv_s > 0 else
                "no L/R yaw moment available for this architecture"
            )

            return EVRunResult(
                architecture=arch, ok=True, lap_result=lap,
                lap_time=lap.lap_time, event_time=lap.event_time,
                energy_per_lap_kwh=e_lap, regen_recovered_kwh=regen,
                energy_full_event_kwh=e_event,
                laps_until_empty=laps_until_empty, finishes_event=bool(finishes),
                derate_lap_time_penalty_s=derate_pen,
                tv_yaw_benefit_s=tv_s, tv_yaw_benefit_flagged=tv_flag,
                effective_mass_kg=p.mass,
                warnings=list(lap.warnings),
            )
        except Exception as exc:
            return EVRunResult.failed(arch, [f"architecture run crashed: {exc!r}"])
        finally:
            # always restore the vehicle's mass, success or failure
            if _saved_mass is not None:
                try:
                    self.veh.p.mass = _saved_mass
                except Exception:
                    pass

    # ---- the headline: compare architectures ---------------------------- #
    def compare(self, track: Track,
                architectures: Optional[list[Powertrain]] = None
                ) -> ArchitectureComparison:
        """Run every architecture on the same car/track and rank them."""
        archs = architectures or list(Powertrain)
        results = [self.run_architecture(a, track) for a in archs]
        return ArchitectureComparison(
            track_name=getattr(track, "name", "track"),
            results=results, warnings=list(self.warnings),
        )
