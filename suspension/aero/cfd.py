# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
CFD co-simulation boundary — the seam where STAR-CCM+ / Fluent / OpenFOAM plug in.

WHY THIS MODULE EXISTS (read this before adding a solver backend)
-----------------------------------------------------------------
The rest of KinematiK is analytic and instant: a constraint solve, a Pacejka
curve, a point-mass integration all return in milliseconds, the same answer every
time. A full-car CFD solve is the opposite animal — a meshed Navier–Stokes case of
tens of millions of cells, run on an MPI cluster under a scheduler, taking hours
per attitude, behind a four/five-figure solver license (STAR-CCM+, Fluent) or a
free open stack (OpenFOAM). KinematiK cannot, and must not pretend to, *be* that
solver. What it can own well is the part it's already good at: parameterising a
sweep, orchestrating the runs, and folding the result back into the lap sim.

So integration is NOT "add a CFD function that returns downforce". It is defining a
co-simulation boundary KinematiK can drive and being honest that the meshing and
the Navier–Stokes solve live OUTSIDE KinematiK, inside a licensed binary or an
OpenFOAM install on a cluster the team owns. This module is that boundary:

    * `Attitude`        — one operating point: roll, pitch, yaw, ride height, speed,
    * `RunMatrix`       — the swept grid of attitudes that becomes an aero map,
    * `CaseSpec`        — the solver-neutral description of one case to run,
    * `CoeffResult`     — the force/moment coefficients that come back (or don't),
    * `CFDProvenance`   — where the numbers came from and what's trustworthy,
    * `CFDSolver`       — the Protocol every backend implements (the contract),
    * `SolverFidelity`  — honest labelling of what a backend actually resolves.

THE HONESTY CONTRACT (why it is strict here, specifically)
----------------------------------------------------------
This mirrors `tire_cosim.py` deliberately, and for the same reason. A CFD result
*looks* like measurement — it's a number with four significant figures and a
pretty wake behind it — so an uncalibrated, unconverged, or under-resolved run is
MORE dangerous than an obviously-rough hand estimate, because nobody questions it.
The trust in real aero work lives in PROVENANCE: mesh count and y+, turbulence
model, convergence of the force monitors, and correlation against a straight-line
or a wind-tunnel/coastdown point. Therefore, by construction:

  - Every `CoeffResult` carries `converged`, the residual/force-monitor history
    summary, the cell count and the y+ band — not just the coefficients.
  - A backend returns `None` for a channel it did not actually compute. It never
    fabricates a coefficient to fill a hole in the map.
  - `is_correlated` is False until a result is tied to physical reference data,
    exactly like `DamperCurve.is_calibrated` and the tyre provenance.

DELIBERATE NON-GOAL: this module does not mesh anything and does not solve
Navier–Stokes. Writing a fake RANS solve would be the failure above. It owns the
SEAM — a clean, typed, tested protocol — so a real solver drops in at one place,
and the run-matrix / aero-map / lap-sim machinery can be written and tested now
against a runnable reference backend with no external binary or license.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence, runtime_checkable


# --------------------------------------------------------------------------- #
#  Provenance — the thing that makes this trustworthy instead of dangerous
# --------------------------------------------------------------------------- #
class SolverFidelity(str, Enum):
    """How physically complete a CFD backend's result is — for honest labelling."""
    POTENTIAL = "potential"            # inviscid / panel: trends only, no real drag
    RANS = "rans"                      # steady RANS: standard FSAE working point
    URANS = "urans"                    # unsteady RANS: gross separation, some transient
    DES = "des"                        # DES/DDES/LES: resolved wake, the expensive truth


@dataclass
class CFDProvenance:
    """
    Where a CFD result's numbers came from and what they're worth — the single most
    important object in this module. A clean-looking coefficient from an
    unconverged, coarse, uncorrelated run is NOT engineering data, and this records
    the difference so a map can never imply more than the solve behind it.
    """
    backend: str                       # "reference-analytic", "openfoam", "starccm", "fluent"
    fidelity: SolverFidelity
    is_correlated: bool = False        # True only when tied to physical reference data
    turbulence_model: str = ""         # "kOmegaSST", "Spalart-Allmaras", ...
    cell_count: Optional[int] = None   # mesh size actually run
    yplus_mean: Optional[float] = None # wall-treatment sanity
    correlated_against: str = ""       # "straightline coastdown", "MIRA tunnel", ...
    notes: str = ""

    def status(self) -> str:
        """One-line human summary, in the style of TireProvenance.status()."""
        mesh = f", {self.cell_count/1e6:.1f}M cells" if self.cell_count else ""
        if self.is_correlated:
            ref = self.correlated_against or "physical reference data"
            return (f"{self.backend} ({self.fidelity.value}{mesh}), "
                    f"correlated against {ref}")
        return (f"{self.backend} ({self.fidelity.value}{mesh}), UNCORRELATED — "
                f"raw solver output not tied to measured data; treat absolute "
                f"coefficients as provisional, trust deltas more than levels")


# --------------------------------------------------------------------------- #
#  One operating point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Attitude:
    """
    A single car attitude — one point in the aero map. Angles in degrees, lengths
    in metres, speed in m/s. Sign convention (SAE-ish, car frame):
      roll  > 0 : right side down (positive about +x, rearward)
      pitch > 0 : nose up         (positive about +y, to the right)
      yaw   > 0 : nose to the right of the relative wind (sideslip beta)
    ride_height is the front reference clearance; rake is captured by pitch.
    """
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    ride_height_mm: float = 30.0
    speed_ms: float = 20.0

    def key(self) -> tuple:
        """Hashable rounded key for de-duplication / map indexing."""
        return (round(self.roll_deg, 3), round(self.pitch_deg, 3),
                round(self.yaw_deg, 3), round(self.ride_height_mm, 2),
                round(self.speed_ms, 2))

    def label(self) -> str:
        return (f"roll={self.roll_deg:+.1f} pitch={self.pitch_deg:+.1f} "
                f"yaw={self.yaw_deg:+.1f} h={self.ride_height_mm:.0f}mm "
                f"V={self.speed_ms:.0f}")


# --------------------------------------------------------------------------- #
#  The swept grid that becomes a map
# --------------------------------------------------------------------------- #
@dataclass
class RunMatrix:
    """
    A factorial sweep of attitudes. Give the axes you care about; the matrix is the
    Cartesian product. A wing-angle study is just a 1-axis matrix; a full
    roll/pitch/yaw map is a 3-axis one. Ride height and speed are axes too.

    Keep it honest about cost: `len()` is how many CFD cases this is. At hours per
    case, a 5x5x5 yaw/pitch/roll map is 125 overnight-class solves — the tool
    should *show* that number before anyone submits it, not hide it.
    """
    roll_deg: Sequence[float] = (0.0,)
    pitch_deg: Sequence[float] = (0.0,)
    yaw_deg: Sequence[float] = (0.0,)
    ride_height_mm: Sequence[float] = (30.0,)
    speed_ms: Sequence[float] = (20.0,)

    def attitudes(self) -> list[Attitude]:
        out = []
        for r, p, y, h, v in itertools.product(
            self.roll_deg, self.pitch_deg, self.yaw_deg,
            self.ride_height_mm, self.speed_ms
        ):
            out.append(Attitude(float(r), float(p), float(y), float(h), float(v)))
        return out

    def __len__(self) -> int:
        return (len(self.roll_deg) * len(self.pitch_deg) * len(self.yaw_deg)
                * len(self.ride_height_mm) * len(self.speed_ms))

    def axes_swept(self) -> list[str]:
        """Which dimensions actually vary — what the resulting map is a function of."""
        axes = []
        if len(self.roll_deg) > 1: axes.append("roll")
        if len(self.pitch_deg) > 1: axes.append("pitch")
        if len(self.yaw_deg) > 1: axes.append("yaw")
        if len(self.ride_height_mm) > 1: axes.append("ride_height")
        if len(self.speed_ms) > 1: axes.append("speed")
        return axes

    def cost_summary(self, minutes_per_case: float = 180.0,
                     concurrent: int = 1) -> str:
        """Plain-language wall-clock estimate so nobody submits 125 cases blind."""
        n = len(self)
        wall_h = n * minutes_per_case / 60.0 / max(concurrent, 1)
        return (f"{n} case(s) over axes {self.axes_swept() or ['(single point)']}; "
                f"~{minutes_per_case:.0f} min/case at {concurrent}x concurrency "
                f"=> ~{wall_h:.1f} h wall-clock")


# --------------------------------------------------------------------------- #
#  Solver-neutral description of one case
# --------------------------------------------------------------------------- #
@dataclass
class CaseSpec:
    """
    Everything a backend needs to write ONE case, with nothing solver-specific in
    it. The geometry is referenced by path (an STL/CAD the team drops in); KinematiK
    does not own the mesh. `reference_area` and `reference_length` are what the
    coefficients get normalised by — they MUST match what the post-processor uses or
    the map is silently wrong, so they live here explicitly.
    """
    attitude: Attitude
    geometry_path: str                      # STL / surface mesh the team supplies
    reference_area_m2: float = 1.0          # frontal area A for C_L, C_D
    reference_length_m: float = 1.55        # wheelbase for moment coeff C_my
    rho: float = 1.225                      # air density
    target_yplus: float = 1.0               # wall-resolved vs wall-function intent
    target_cells: Optional[int] = None      # mesh budget, if the mesher is driven
    fidelity: SolverFidelity = SolverFidelity.RANS
    extra: dict = field(default_factory=dict)   # backend-specific escape hatch

    def case_name(self) -> str:
        a = self.attitude
        return (f"case_r{a.roll_deg:+05.1f}_p{a.pitch_deg:+05.1f}"
                f"_y{a.yaw_deg:+05.1f}_h{a.ride_height_mm:04.0f}"
                f"_v{a.speed_ms:04.1f}").replace("+", "p").replace("-", "m").replace(".", "")


# --------------------------------------------------------------------------- #
#  What comes back across the boundary
# --------------------------------------------------------------------------- #
@dataclass
class CoeffResult:
    """
    The force/moment coefficients for one attitude — or honest holes where the
    solve did not produce them. Coefficients are non-dimensional (already divided by
    1/2 rho V^2 A and, for the moment, the reference length).

    A value of `None` means "this backend did not compute this channel" — NOT zero,
    NOT a guess. The map assembler treats None as missing and refuses to interpolate
    across it. `converged` and `force_monitor_range` are first-class because an
    unconverged force is not a force.
    """
    attitude: Attitude
    c_lift: Optional[float] = None          # negative = downforce in this convention
    c_drag: Optional[float] = None
    c_side: Optional[float] = None          # side force, relevant under yaw
    c_pitch: Optional[float] = None         # pitching moment coeff (aero balance)
    aero_balance_front: Optional[float] = None  # fraction of downforce on the front axle
    converged: bool = False
    force_monitor_range: Optional[float] = None  # last-N% spread of the C_L monitor
    wall_clock_s: Optional[float] = None
    provenance: Optional[CFDProvenance] = None
    notes: str = ""

    def is_usable(self) -> bool:
        """A result you can put in a map: it converged and has lift+drag."""
        return (self.converged and self.c_lift is not None
                and self.c_drag is not None)

    def downforce_N(self, rho: float, area_m2: float, speed_ms: float) -> Optional[float]:
        """Dimensional downforce (positive up-force-removed) at a given speed."""
        if self.c_lift is None:
            return None
        q = 0.5 * rho * speed_ms * speed_ms
        return -self.c_lift * q * area_m2     # convention: negative C_L => +downforce


# --------------------------------------------------------------------------- #
#  The contract every solver backend implements
# --------------------------------------------------------------------------- #
@runtime_checkable
class CFDSolver(Protocol):
    """
    The seam. A backend turns a `CaseSpec` into solver input files (`write_case`),
    optionally runs them (`run_case`), and parses results back into a `CoeffResult`
    (`read_result`). Splitting write/run/read is deliberate: it lets an
    unlicensed/offline backend WRITE correct STAR-CCM+ or Fluent input that a team
    runs on their own cluster, then READ the result back — without KinematiK ever
    holding the license or the mesh.

    A backend that cannot run in this environment (no license / no binary) must
    still implement `write_case` faithfully and must raise a clear, actionable error
    from `run_case` rather than fabricating a `CoeffResult`.
    """
    name: str

    def provenance(self) -> CFDProvenance: ...

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        """Write solver input for one case into workdir; return the entry path."""
        ...

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """Run (or submit) one case and return its coefficients."""
        ...

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """Parse an already-run case's output into a CoeffResult."""
        ...


class SolverUnavailable(RuntimeError):
    """
    Raised by a backend that can write input but cannot solve here (no license, no
    binary, no cluster). Carries an actionable message — exactly which binary /
    license / parameter is missing — never a silent fallback to a fake number.
    """
