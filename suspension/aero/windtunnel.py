# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Virtual Wind Tunnel — correlate a PHYSICAL aero-map run against a matching CFD run.

WHY THIS MODULE EXISTS (read this before using it)
---------------------------------------------------
The headline reason a team runs a wind tunnel is NOT "is the car fast" — the lap
sim already answers that, faster and cheaper. The reason is CALIBRATION. A modern
aero programme lives or dies on CFD: hundreds of digital configurations get
screened before one part is ever made. But CFD only earns that trust if its
turbulence closure (for FSAE, almost always k-omega SST RANS) reproduces numbers
you actually measured. So the tunnel's job is to be the ruler the CFD is checked
against. If the digital C_d / C_l don't land on the physical C_d / C_l at the SAME
operating points, the whole digital pipeline is decoration.

That correlation has one non-negotiable rule, and it is the rule this module
enforces: **you must compare like-for-like operating points.** The physical aero
map is swept over front and rear ride height (the platform pitches and heaves under
aero load, and downforce is exquisitely ride-height sensitive on a ground-effect
car). The "Virtual Wind Tunnel" is the CFD sweep run at those EXACT same front/rear
ride heights and the same wind speed — not the team's convenient CFD grid, the
tunnel's grid. Compare a CFD point at 25/45 mm against a tunnel point at 20/40 mm
and the delta you "calibrate" away is just ride-height sensitivity masquerading as
a turbulence-model error. So this module's spine is:

    physical ride-height/speed grid  ->  the IDENTICAL CFD run matrix  ->  a
    channel-by-channel C_d/C_l/balance correlation that says, per point and
    overall, whether k-omega SST reproduced the tunnel.

WHAT THIS MODULE OWNS (and what it deliberately does not)
---------------------------------------------------------
It owns three things, in the established KinematiK idiom:

  * `RideHeights` + `ride_heights_to_attitude` / `attitude_to_ride_heights` — the
    geometry that turns the motorsport aero-map convention (FRH, RRH measured at
    the reference planes a wheelbase apart) into the `Attitude` (ride_height + rake
    pitch) the CFD seam in `cfd.py` already speaks, and back, with NO lossy
    rounding. This is the lock that makes "same operating point" literally true.

  * `PhysicalAeroMap` — the measured tunnel deliverable. It is an `AeroMap` (so the
    lap sim can consume it unchanged via `AeroProvider`) but carries TUNNEL
    provenance instead of CFD provenance: facility, blockage correction, moving-
    ground state, model scale and the Reynolds number the run was at. A tunnel
    number that wasn't blockage-corrected, or was taken on a fixed floor, is not
    the same measurement as one that was, and the provenance records the
    difference so a correlation can never imply more than the run behind it.

  * `VirtualWindTunnel` — given a `PhysicalAeroMap`, it emits the matching
    `RunMatrix` (the exact swept points) and, through any `CFDSolver` backend
    (`StarCCMSolver`, `OpenFOAMSolver`, or the vendor `TSAutoSolver` added here),
    the `CaseSpec` list a team submits. Then `correlate()` takes the CFD results
    back and produces a `TunnelCorrelationReport`: per-point and aggregate C_d /
    C_l / balance error, with an explicit verdict on whether the turbulence model
    is calibrated to the tunnel inside tolerance.

DELIBERATE NON-GOALS, same discipline as cfd.py: this module does not mesh, does
not solve Navier-Stokes, and does not invent a tunnel reading or a CFD coefficient
to fill a hole. It refuses to correlate points it cannot pair like-for-like. A
mismatch it reports is a real mismatch; a hole it reports is a real hole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Sequence

from .cfd import (
    Attitude, RunMatrix, CaseSpec, CoeffResult, CFDProvenance, SolverFidelity,
)
from .aeromap import AeroMap


# --------------------------------------------------------------------------- #
#  Ride-height geometry — the lock that makes "same operating point" literal
# --------------------------------------------------------------------------- #
#
# Aero people sweep the map in FRONT ride height (FRH) and REAR ride height (RRH),
# measured at two reference planes a known longitudinal distance apart (here, the
# wheelbase, which is also `reference_length_m` in the CFD seam). The CFD `Attitude`
# instead carries a single `ride_height_mm` (the FRONT reference clearance) and a
# `pitch_deg` (nose-up positive) that captures rake. Those describe the SAME plane,
# so the map between them is exact and invertible:
#
#     rake (rear minus front, +ve = nose up) = RRH - FRH        [mm over wheelbase]
#     pitch_deg = atan2(RRH - FRH, wheelbase_mm)  in degrees
#     ride_height_mm = FRH
#
# Doing this in one place, tested, is what stops a 0.5 deg pitch ambiguity from
# quietly decorrelating a tunnel-vs-CFD comparison.
@dataclass(frozen=True)
class RideHeights:
    """
    One platform position in the convention the tunnel rig actually sets: front and
    rear ride height at the reference planes, plus the wind speed. `wheelbase_mm` is
    the longitudinal distance between the two reference planes (the CFD reference
    length); it MUST match `reference_length_m` used to non-dimensionalise, or the
    pitch implied here won't equal the pitch the solver sees.
    """
    front_mm: float
    rear_mm: float
    speed_ms: float = 20.0
    wheelbase_mm: float = 1550.0

    @property
    def rake_mm(self) -> float:
        """Rear minus front; positive = nose-up rake (the usual fast direction)."""
        return self.rear_mm - self.front_mm

    @property
    def pitch_deg(self) -> float:
        return math.degrees(math.atan2(self.rear_mm - self.front_mm, self.wheelbase_mm))

    def label(self) -> str:
        return (f"FRH={self.front_mm:.0f} RRH={self.rear_mm:.0f} "
                f"(rake {self.rake_mm:+.0f}mm) V={self.speed_ms:.0f}")


def ride_heights_to_attitude(rh: RideHeights, roll_deg: float = 0.0,
                             yaw_deg: float = 0.0) -> Attitude:
    """
    Convert a physical front/rear ride-height point to the CFD `Attitude`. This is
    the function the Virtual Wind Tunnel uses so the digital case runs at the EXACT
    physical platform position. Roll/yaw default to zero (a straight-ahead ride-
    height map) but are accepted for yaw-aware maps.
    """
    return Attitude(
        roll_deg=float(roll_deg),
        pitch_deg=rh.pitch_deg,
        yaw_deg=float(yaw_deg),
        ride_height_mm=float(rh.front_mm),
        speed_ms=float(rh.speed_ms),
    )


def attitude_to_ride_heights(att: Attitude, wheelbase_mm: float = 1550.0) -> RideHeights:
    """Inverse of ride_heights_to_attitude — recover FRH/RRH from an Attitude."""
    front = att.ride_height_mm
    rear = front + wheelbase_mm * math.tan(math.radians(att.pitch_deg))
    return RideHeights(front_mm=float(front), rear_mm=float(rear),
                       speed_ms=float(att.speed_ms), wheelbase_mm=float(wheelbase_mm))


@dataclass
class AeroMapGrid:
    """
    The swept ride-height/speed grid that defines BOTH the physical run plan and the
    matching virtual one. Front and rear ride height are swept independently (the
    motorsport aero-map standard); speed is usually a single tunnel speed but can be
    a Reynolds sweep. `len()` is how many physical configurations this is — show it
    before anyone books tunnel time or submits CFD.
    """
    front_mm: Sequence[float]
    rear_mm: Sequence[float]
    speed_ms: Sequence[float] = (20.0,)
    wheelbase_mm: float = 1550.0

    def ride_height_points(self) -> list[RideHeights]:
        out = []
        for v in self.speed_ms:
            for f in self.front_mm:
                for r in self.rear_mm:
                    out.append(RideHeights(float(f), float(r), float(v),
                                           self.wheelbase_mm))
        return out

    def __len__(self) -> int:
        return len(self.front_mm) * len(self.rear_mm) * len(self.speed_ms)

    def to_run_matrix(self) -> RunMatrix:
        """
        The IDENTICAL CFD run matrix. Because pitch is a function of (FRH, RRH), a
        rectangular FRH x RRH grid is NOT a rectangular ride_height x pitch grid, so
        we cannot express it as a single Cartesian `RunMatrix`. Instead we hand back
        the explicit attitude list via `attitudes()`; callers that need a RunMatrix-
        shaped object for cost previews get the degenerate single-axis form here and
        the true point list from `attitudes()`.
        """
        atts = self.attitudes()
        return RunMatrix(
            ride_height_mm=sorted({a.ride_height_mm for a in atts}),
            pitch_deg=sorted({round(a.pitch_deg, 4) for a in atts}),
            speed_ms=sorted({a.speed_ms for a in atts}),
        )

    def attitudes(self) -> list[Attitude]:
        return [ride_heights_to_attitude(rh) for rh in self.ride_height_points()]


# --------------------------------------------------------------------------- #
#  Tunnel provenance — the wind-tunnel analogue of CFDProvenance
# --------------------------------------------------------------------------- #
class GroundState(str, Enum):
    """How the floor was simulated — the single biggest tunnel honesty flag."""
    MOVING_BELT = "moving-belt"        # rolling road: the only ground-effect-true state
    FIXED_FLOOR = "fixed-floor"        # boundary layer on the floor; underbody is wrong
    SUCTION_FIXED = "suction-fixed"    # fixed floor with BL suction: a partial fix


@dataclass
class TunnelProvenance:
    """
    Where a PHYSICAL aero-map number came from and what it's worth — the tunnel twin
    of CFDProvenance. The fields are the ones that decide whether a tunnel reading is
    comparable to free-air CFD at all: blockage correction, the moving-ground state,
    the model scale, and the Reynolds number the run was actually at. An uncorrected,
    fixed-floor, quarter-scale reading is a different measurement from a corrected,
    moving-belt, full-scale one, and pretending otherwise is how a correlation lies.
    """
    facility: str                          # "A2 Wind Shear", "in-house 0.5x", ...
    ground_state: GroundState = GroundState.MOVING_BELT
    model_scale: float = 1.0               # 1.0 = full scale; 0.5 = half-scale model
    blockage_corrected: bool = True        # solid+wake blockage applied to coeffs?
    blockage_ratio: Optional[float] = None # model frontal area / test-section area
    reynolds: Optional[float] = None       # chord/length Reynolds number of the run
    reference_area_m2: Optional[float] = None
    reference_length_m: Optional[float] = None
    notes: str = ""

    def to_cfd_provenance(self) -> CFDProvenance:
        """
        Express the tunnel run as a CFDProvenance so a PhysicalAeroMap is a drop-in
        AeroMap the lap sim already understands. Crucially it is marked
        `is_correlated=True` against this facility: a measured tunnel map IS physical
        reference data — it is the thing CFD gets correlated against, not the other
        way round.
        """
        return CFDProvenance(
            backend=f"windtunnel:{self.facility}",
            fidelity=SolverFidelity.DES,   # measurement resolves everything CFD models
            is_correlated=True,
            turbulence_model="n/a (physical measurement)",
            correlated_against=self.facility,
            notes=self.status(),
        )

    def status(self) -> str:
        scale = "full-scale" if abs(self.model_scale - 1.0) < 1e-6 else f"{self.model_scale:g}x scale"
        gs = self.ground_state.value
        blk = "blockage-corrected" if self.blockage_corrected else "NOT blockage-corrected"
        re = f", Re={self.reynolds:.2e}" if self.reynolds else ""
        warn = ""
        if self.ground_state is not GroundState.MOVING_BELT:
            warn = (" — WARNING: fixed/suction floor underpredicts ground-effect "
                    "downforce; pair against free-air CFD with the same floor model")
        if not self.blockage_corrected:
            warn += (" — WARNING: uncorrected coefficients are inflated by tunnel "
                     "blockage; correct before comparing to CFD")
        return f"{self.facility} ({scale}, {gs}, {blk}{re}){warn}"


# --------------------------------------------------------------------------- #
#  The physical aero map — a measured AeroMap with tunnel provenance
# --------------------------------------------------------------------------- #
class PhysicalAeroMap(AeroMap):
    """
    The wind tunnel deliverable. It is an AeroMap (so `AeroProvider` feeds it to the
    lap sim with zero new plumbing), built from MEASURED coefficients at swept
    front/rear ride heights, and it carries TunnelProvenance.

    Build it the way a team logs the run: per ride-height point, drop in the measured
    C_l / C_d (and optional front aero balance). The map stores the underlying
    `Attitude` so the Virtual Wind Tunnel can reproduce the exact same points.
    """

    def __init__(self, tunnel: TunnelProvenance,
                 reference_area_m2: float = 1.0,
                 reference_length_m: float = 1.55,
                 wheelbase_mm: Optional[float] = None):
        super().__init__(reference_area_m2, reference_length_m,
                         provenance=tunnel.to_cfd_provenance())
        self.tunnel = tunnel
        self.wheelbase_mm = (wheelbase_mm if wheelbase_mm is not None
                             else reference_length_m * 1000.0)

    def add_measurement(self, rh: RideHeights, c_lift: float, c_drag: float,
                        aero_balance_front: Optional[float] = None,
                        c_side: Optional[float] = None) -> bool:
        """
        Record one measured operating point. Sign convention matches the CFD seam:
        c_lift NEGATIVE = downforce. A tunnel that logs positive-down downforce
        should be converted by the caller (see `from_downforce` helper).
        """
        att = ride_heights_to_attitude(rh)
        res = CoeffResult(
            attitude=att, c_lift=float(c_lift), c_drag=float(c_drag),
            c_side=c_side, aero_balance_front=aero_balance_front,
            converged=True,                       # a logged measurement is "final"
            force_monitor_range=0.0,
            provenance=self.tunnel.to_cfd_provenance(),
            notes=f"measured @ {rh.label()}",
        )
        return self.add(res)

    def measured_points(self) -> list[CoeffResult]:
        """The raw measured results, for pairing against CFD point-for-point."""
        return list(self._points.values())

    def ride_height_grid(self) -> AeroMapGrid:
        """Recover the swept grid this map was measured on (for the virtual run)."""
        atts = [r.attitude for r in self._points.values()]
        rhs = [attitude_to_ride_heights(a, self.wheelbase_mm) for a in atts]
        return AeroMapGrid(
            front_mm=sorted({round(rh.front_mm, 3) for rh in rhs}),
            rear_mm=sorted({round(rh.rear_mm, 3) for rh in rhs}),
            speed_ms=sorted({round(rh.speed_ms, 3) for rh in rhs}),
            wheelbase_mm=self.wheelbase_mm,
        )

    def status(self) -> str:
        base = super().status()
        return base + f"  [PHYSICAL — {self.tunnel.status()}]"


def downforce_to_clift(downforce_N: float, rho: float, area_m2: float,
                       speed_ms: float) -> float:
    """
    Helper: convert a tunnel-logged downforce (Newtons, positive DOWN) to the
    c_lift the map expects (negative = downforce). q = 1/2 rho V^2.
    """
    q = 0.5 * rho * speed_ms * speed_ms
    if q <= 0 or area_m2 <= 0:
        return float("nan")
    return -(downforce_N) / (q * area_m2)


def drag_to_cdrag(drag_N: float, rho: float, area_m2: float,
                  speed_ms: float) -> float:
    q = 0.5 * rho * speed_ms * speed_ms
    if q <= 0 or area_m2 <= 0:
        return float("nan")
    return drag_N / (q * area_m2)


# --------------------------------------------------------------------------- #
#  Per-point and aggregate correlation result containers
# --------------------------------------------------------------------------- #
@dataclass
class PointCorrelation:
    """C_d/C_l/balance comparison at ONE ride-height point — tunnel vs CFD."""
    ride_heights: RideHeights
    cl_phys: Optional[float]
    cl_cfd: Optional[float]
    cd_phys: Optional[float]
    cd_cfd: Optional[float]
    bal_phys: Optional[float] = None
    bal_cfd: Optional[float] = None
    paired: bool = False
    note: str = ""

    @staticmethod
    def _pct(cfd, phys):
        if cfd is None or phys is None or abs(phys) < 1e-9:
            return float("nan")
        return 100.0 * (cfd - phys) / phys

    @property
    def cl_err_pct(self) -> float:
        return self._pct(self.cl_cfd, self.cl_phys)

    @property
    def cd_err_pct(self) -> float:
        return self._pct(self.cd_cfd, self.cd_phys)

    @property
    def bal_err_pts(self) -> float:
        """Aero-balance error in percentage POINTS (e.g. 0.45 vs 0.42 -> +3.0 pts)."""
        if self.bal_cfd is None or self.bal_phys is None:
            return float("nan")
        return 100.0 * (self.bal_cfd - self.bal_phys)

    def as_dict(self):
        d = asdict(self)
        d["ride_heights"] = self.ride_heights.label()
        d["cl_err_pct"] = self.cl_err_pct
        d["cd_err_pct"] = self.cd_err_pct
        d["bal_err_pts"] = self.bal_err_pts
        return d


# Default correlation tolerances. These are the bands inside which a well-run
# k-omega SST RANS solve credibly reproduces a corrected, moving-ground tunnel map
# for an FSAE-scale car. Defaults, not truths — every report carries the tolerance
# it used. Rough magnitudes: RANS typically lands within a few % on C_d and C_l for
# attached-flow attitudes, looser where the floor/diffuser is near separation; aero
# balance (the moment split) is the hardest channel and gets a points band.
DEFAULT_TUNNEL_TOL = {
    "cl_pct": 5.0,        # % error in C_l (downforce) per point
    "cd_pct": 5.0,        # % error in C_d (drag) per point
    "balance_pts": 2.0,   # aero-balance error in percentage points per point
    "map_rms_pct": 4.0,   # RMS of |C_l error| across the whole map
}


@dataclass
class TunnelCorrelationReport:
    """
    The deliverable: did the CFD (k-omega SST) reproduce the tunnel, point by point
    and overall? Nothing here tunes the model — it quantifies the gap and says, in
    auditable numbers, whether the turbulence closure is calibrated to the facility.
    """
    ok: bool
    backend: str
    turbulence_model: str
    points: list                            # list[PointCorrelation]
    n_paired: int
    cl_rms_pct: float
    cd_rms_pct: float
    cl_bias_pct: float                      # signed mean of (cfd-phys)/phys; +ve = CFD over-predicts downforce magnitude
    cd_bias_pct: float
    worst_cl: Optional[PointCorrelation]
    overall_within_tol: bool
    tolerances: dict
    summary: str = ""
    physical_provenance: str = ""
    cfd_provenance: str = ""

    def as_dict(self):
        return dict(
            ok=self.ok, backend=self.backend,
            turbulence_model=self.turbulence_model,
            n_paired=self.n_paired,
            cl_rms_pct=self.cl_rms_pct, cd_rms_pct=self.cd_rms_pct,
            cl_bias_pct=self.cl_bias_pct, cd_bias_pct=self.cd_bias_pct,
            overall_within_tol=self.overall_within_tol,
            summary=self.summary,
            tolerances=dict(self.tolerances),
            physical_provenance=self.physical_provenance,
            cfd_provenance=self.cfd_provenance,
            points=[p.as_dict() for p in self.points],
        )


# --------------------------------------------------------------------------- #
#  The Virtual Wind Tunnel — generate the matching CFD run, then correlate
# --------------------------------------------------------------------------- #
class VirtualWindTunnel:
    """
    Drives the digital side of the calibration loop. Given a measured
    `PhysicalAeroMap`, it produces:

      * `run_matrix()` / `case_specs()` — the EXACT same ride-height/speed points,
        as CFD cases for any backend (Star-CCM+, OpenFOAM, the vendor TS-Auto stub).
        This is the guarantee of like-for-like: the digital sweep is generated FROM
        the physical map, never hand-typed alongside it.
      * `correlate()` — takes the CFD `CoeffResult`s back and reports, per point and
        overall, whether C_d/C_l/balance match the tunnel inside tolerance, i.e.
        whether the k-omega SST model is calibrated to the facility.

    It never solves anything itself; the backend owns write/run/read and raises
    SolverUnavailable rather than faking a number, exactly as in `cfd.py`.
    """

    def __init__(self, physical: PhysicalAeroMap, geometry_path: str,
                 rho: float = 1.225):
        self.physical = physical
        self.geometry_path = geometry_path
        self.rho = rho
        self.reference_area_m2 = physical.reference_area_m2
        self.reference_length_m = physical.reference_length_m

    # -- generate the matching digital run -------------------------------- #
    def grid(self) -> AeroMapGrid:
        return self.physical.ride_height_grid()

    def run_matrix(self) -> RunMatrix:
        return self.grid().to_run_matrix()

    def attitudes(self) -> list[Attitude]:
        """The exact physical operating points, as CFD attitudes."""
        return [r.attitude for r in self.physical.measured_points()]

    def case_specs(self, fidelity: SolverFidelity = SolverFidelity.RANS,
                   target_yplus: float = 1.0) -> list[CaseSpec]:
        """
        One CaseSpec per measured tunnel point, carrying the SAME reference area and
        length the physical coefficients were normalised by (so the two are
        comparable at all) and the same air density.
        """
        specs = []
        for att in self.attitudes():
            specs.append(CaseSpec(
                attitude=att, geometry_path=self.geometry_path,
                reference_area_m2=self.reference_area_m2,
                reference_length_m=self.reference_length_m,
                rho=self.rho, target_yplus=target_yplus, fidelity=fidelity,
            ))
        return specs

    def plan(self, minutes_per_case: float = 180.0, concurrent: int = 1) -> str:
        """Cost preview of the digital sweep, before any solve."""
        g = self.grid()
        n = len(g)
        wall_h = n * minutes_per_case / 60.0 / max(concurrent, 1)
        return (f"Virtual Wind Tunnel: {n} CFD case(s) matching the physical map "
                f"(front {list(g.front_mm)} mm x rear {list(g.rear_mm)} mm x "
                f"speed {list(g.speed_ms)} m/s); ~{minutes_per_case:.0f} min/case "
                f"at {concurrent}x => ~{wall_h:.1f} h wall-clock.")

    # -- correlate physical vs digital ------------------------------------ #
    def correlate(self, cfd_results: Sequence[CoeffResult],
                  tol: Optional[dict] = None) -> TunnelCorrelationReport:
        """
        Pair each CFD result to the physical measurement at the SAME attitude key and
        report C_d/C_l/balance error per point and overall. CFD results whose
        attitude doesn't match a physical point are reported as unpaired (a hole),
        never silently snapped to the nearest tunnel point.
        """
        tol = {**DEFAULT_TUNNEL_TOL, **(tol or {})}
        phys_by_key = {r.attitude.key(): r for r in self.physical.measured_points()}

        backend_name = ""
        turb = ""
        for r in cfd_results:
            if r.provenance is not None:
                backend_name = r.provenance.backend or backend_name
                turb = r.provenance.turbulence_model or turb

        points: list[PointCorrelation] = []
        cl_errs, cd_errs = [], []
        for cr in cfd_results:
            key = cr.attitude.key()
            phys = phys_by_key.get(key)
            rh = attitude_to_ride_heights(cr.attitude, self.physical.wheelbase_mm)
            if phys is None:
                points.append(PointCorrelation(
                    ride_heights=rh, cl_phys=None, cl_cfd=cr.c_lift,
                    cd_phys=None, cd_cfd=cr.c_drag, paired=False,
                    note="no physical point at this attitude — not paired"))
                continue
            pc = PointCorrelation(
                ride_heights=rh,
                cl_phys=phys.c_lift, cl_cfd=cr.c_lift,
                cd_phys=phys.c_drag, cd_cfd=cr.c_drag,
                bal_phys=phys.aero_balance_front, bal_cfd=cr.aero_balance_front,
                paired=(cr.is_usable() and phys.is_usable()),
                note="" if cr.converged else "CFD point not converged — error meaningless",
            )
            points.append(pc)
            if pc.paired and math.isfinite(pc.cl_err_pct):
                cl_errs.append(pc.cl_err_pct)
            if pc.paired and math.isfinite(pc.cd_err_pct):
                cd_errs.append(pc.cd_err_pct)

        # also flag physical points the CFD never covered
        cfd_keys = {cr.attitude.key() for cr in cfd_results}
        for key, phys in phys_by_key.items():
            if key not in cfd_keys:
                rh = attitude_to_ride_heights(phys.attitude, self.physical.wheelbase_mm)
                points.append(PointCorrelation(
                    ride_heights=rh, cl_phys=phys.c_lift, cl_cfd=None,
                    cd_phys=phys.c_drag, cd_cfd=None, paired=False,
                    note="physical point not covered by the CFD sweep — not paired"))

        n_paired = sum(1 for p in points if p.paired)

        def _rms(xs):
            return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")

        def _bias(xs):
            return (sum(xs) / len(xs)) if xs else float("nan")

        cl_rms = _rms(cl_errs)
        cd_rms = _rms(cd_errs)
        cl_bias = _bias(cl_errs)
        cd_bias = _bias(cd_errs)

        paired_pts = [p for p in points if p.paired and math.isfinite(p.cl_err_pct)]
        worst = max(paired_pts, key=lambda p: abs(p.cl_err_pct), default=None)

        per_point_ok = all(
            (abs(p.cl_err_pct) <= tol["cl_pct"]) and (abs(p.cd_err_pct) <= tol["cd_pct"])
            and (not math.isfinite(p.bal_err_pts) or abs(p.bal_err_pts) <= tol["balance_pts"])
            for p in paired_pts
        ) if paired_pts else False
        map_ok = math.isfinite(cl_rms) and cl_rms <= tol["map_rms_pct"]
        overall = bool(paired_pts) and per_point_ok and map_ok

        summary = self._summarise(backend_name, turb, n_paired, len(paired_pts),
                                  cl_rms, cd_rms, cl_bias, cd_bias, worst,
                                  overall, tol, points)

        return TunnelCorrelationReport(
            ok=bool(paired_pts), backend=backend_name or "unknown",
            turbulence_model=turb or "unknown",
            points=points, n_paired=n_paired,
            cl_rms_pct=cl_rms, cd_rms_pct=cd_rms,
            cl_bias_pct=cl_bias, cd_bias_pct=cd_bias,
            worst_cl=worst, overall_within_tol=overall, tolerances=tol,
            summary=summary,
            physical_provenance=self.physical.tunnel.status(),
            cfd_provenance=(cfd_results[0].provenance.status()
                            if cfd_results and cfd_results[0].provenance else ""),
        )

    @staticmethod
    def _summarise(backend, turb, n_paired, n_usable, cl_rms, cd_rms,
                   cl_bias, cd_bias, worst, overall, tol, points):
        n_unpaired = sum(1 for p in points if not p.paired)
        head = (f"[Virtual Wind Tunnel] {backend or 'CFD'} "
                f"({turb or 'turbulence model unknown'}) vs physical map: "
                f"{n_usable} point(s) paired")
        if n_unpaired:
            head += f", {n_unpaired} unpaired (holes — not compared)"
        if not n_usable:
            return (head + ". Nothing could be compared like-for-like — generate the "
                    "CFD run from this map's run_matrix() so the attitudes match "
                    "exactly, then re-correlate.")

        if overall:
            verdict = ("CALIBRATED — k-omega SST reproduced the tunnel inside "
                       "tolerance; the CFD pipeline can be trusted to screen "
                       "configurations the tunnel never saw")
        else:
            verdict = ("NOT CALIBRATED — the digital coefficients drift from the "
                       "tunnel beyond tolerance; do NOT trust absolute CFD levels for "
                       "decisions until the model is reconciled")

        # diagnostic direction — what the bias is telling the aero lead.
        # cl_err_pct = 100*(cfd - phys)/phys. C_l is down-NEGATIVE, so a POSITIVE
        # bias means CFD's C_l is more negative than the tunnel's => CFD predicts
        # MORE downforce magnitude (over-predicts); negative bias => under-predicts.
        diag = ""
        if math.isfinite(cl_bias) and abs(cl_bias) > tol["cl_pct"]:
            diag = ("  CFD systematically " +
                    ("OVER-predicts downforce magnitude vs the tunnel — check mesh "
                     "y+ in the floor/diffuser, transition, and that the tunnel was "
                     "blockage-corrected"
                     if cl_bias > 0 else
                     "UNDER-predicts downforce magnitude vs the tunnel — check "
                     "moving-ground modelling and underbody mesh resolution") + ".")
        worst_txt = ""
        if worst is not None and math.isfinite(worst.cl_err_pct):
            worst_txt = (f"  Worst point: {worst.ride_heights.label()} "
                         f"C_l off {worst.cl_err_pct:+.1f}%.")

        return (f"{head}. {verdict}. "
                f"C_l RMS {cl_rms:.1f}% (bias {cl_bias:+.1f}%), "
                f"C_d RMS {cd_rms:.1f}% (bias {cd_bias:+.1f}%)."
                + diag + worst_txt)
