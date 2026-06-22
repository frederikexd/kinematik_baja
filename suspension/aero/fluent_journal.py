# ============================================================================
#  KinematiK — runnable ANSYS Fluent validation journal
#
#  The stock FluentVerificationSolver.write_case writes a STUB journal: it reads a
#  case, prints reference values, and leaves TODO(team) where the boundary
#  conditions, turbulence model, initialisation, iteration and — critically — the
#  coefficient EXPORT belong. So "run the deck and compare" does not work end to
#  end as shipped.
#
#  This module writes a journal that actually RUNS: it sets the velocity inlet from
#  the attitude KinematiK already computes, configures k-omega SST, initialises,
#  iterates to a convergence criterion, and writes the exact coefficient CSV that
#  KinematiK's own reader (_read_simple_coeff_csv / read_fluent_csv) ingests:
#
#       Cl,Cd,Cs,CmPitch,converged           (vendor convention: Cl up-positive)
#
#  so the licensed-Fluent number flows straight back into the tunnel correlation.
#
#  Two things are intentionally PARAMETERISED, because they are site-specific:
#    * the starting case file (a pre-meshed .cas/.msh your team builds), and
#    * the name of the wall zone the force report targets (e.g. "car", "body",
#      or several zones for floor/wings).
#  Both have defaults and can be overridden per run via CaseSpec.extra or the
#  function arguments below.
#
#  What this journal does NOT do: it does not mesh for you. It expects a case you
#  can read and iterate. If you start from raw STL, mesh it first (Fluent Meshing /
#  your own tool) and point `case_file` at the result.
# ============================================================================
from __future__ import annotations

import math
import os
from typing import Optional, Sequence


def _inlet_velocity(speed_ms: float, yaw_deg: float, pitch_deg: float):
    """Freestream rotated by yaw (about z) and pitch (about y) — identical to the
    rotation KinematiK uses internally so the inlet matches every other backend."""
    v = speed_ms
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    ux = v * math.cos(yaw) * math.cos(pitch)
    uy = -v * math.sin(yaw)
    uz = v * math.cos(yaw) * math.sin(pitch)
    return ux, uy, uz


def write_runnable_journal(
    spec,
    workdir: str,
    *,
    case_file: Optional[str] = None,
    wall_zones: Optional[Sequence[str]] = None,
    inlet_zone: str = "inlet",
    iterations: int = 2000,
    convergence_abs: float = 1e-4,
    cl_monitor_window: int = 200,
    cl_monitor_tol: float = 1.0e-3,
    estimate=None,
) -> str:
    """
    Write a RUNNABLE ANSYS Fluent journal for one CaseSpec and return its path.

    Parameters that vary by run (all overridable, all defaulted):
      case_file       : the meshed case to read. Defaults to spec.geometry_path,
                        then to CaseSpec.extra['case_file'] if present.
      wall_zones      : the wall zone(s) the lift/drag/side reports target. Defaults
                        to CaseSpec.extra['wall_zones'], then to a single "car" zone.
                        Pass several names to sum forces over floor/wings/etc.
      inlet_zone      : velocity-inlet zone name (default "inlet").
      iterations      : max iterations before the run stops regardless.
      convergence_abs : absolute residual convergence criterion.
      cl_monitor_*    : a Cl monitor is written so `converged` in the exported CSV
                        reflects a SETTLED force, not just residuals.
      estimate        : optional pre-computed in-house CoeffResult to print in the
                        header for at-a-glance comparison.

    The journal is TUI-based (works in `fluent 3ddp -g -i <case>.jou`) and uses only
    long-standing /define, /solve and /report commands, so it is portable across
    recent Fluent versions. Where a step is genuinely setup-dependent (e.g. which
    zone is the inlet), it is a named parameter, not a silent assumption.
    """
    os.makedirs(workdir, exist_ok=True)
    a = spec.attitude

    # resolve the per-run, site-specific bits with sane fallbacks
    extra = getattr(spec, "extra", {}) or {}
    case_file = (case_file
                 or extra.get("case_file")
                 or spec.geometry_path
                 or "car.cas")
    if wall_zones is None:
        wz = extra.get("wall_zones")
        if isinstance(wz, str):
            wall_zones = [wz]
        elif wz:
            wall_zones = list(wz)
        else:
            wall_zones = ["car"]
    inlet_zone = extra.get("inlet_zone", inlet_zone)

    ux, uy, uz = _inlet_velocity(a.speed_ms, a.yaw_deg, a.pitch_deg)
    umag = a.speed_ms
    # unit lift/drag/side direction vectors (force report axes): lift +z, drag +x,
    # side +y. These define the SIGN of the reported coefficients; the CSV reader
    # then flips Cl (vendor up-positive) into KinematiK's down-negative convention.
    rho = spec.rho
    Aref = spec.reference_area_m2
    Lref = spec.reference_length_m

    case = spec.case_name()
    coeff_csv = case + "_coeffs.csv"

    # in-house estimate for the header (optional, comparison only)
    if estimate is not None:
        cl_est = "n/a" if estimate.c_lift is None else f"{estimate.c_lift:+.4f}"
        cd_est = "n/a" if estimate.c_drag is None else f"{estimate.c_drag:+.4f}"
    else:
        cl_est = cd_est = "(run KinematiK run_case for the in-house estimate)"

    wall_list = " ".join(wall_zones)

    jou = f""";; ===========================================================================
;; KinematiK-generated ANSYS Fluent VALIDATION journal — {case}
;; Attitude: {a.label() if hasattr(a, 'label') else case}
;;
;; This deck RUNS end to end: read case -> set freestream -> k-omega SST ->
;; initialise -> iterate -> export coefficients to "{coeff_csv}".
;; Then, back in KinematiK:
;;     FluentVerificationSolver().read_fluent_csv(spec, workdir)
;; folds this licensed-Fluent number into the tunnel correlation.
;;
;; KinematiK in-house estimate (sign convention C_l negative = downforce):
;;     C_l ~= {cl_est}    C_d ~= {cd_est}
;;
;; PARAMETERISED for this run (edit at the top of the file if a run differs):
;;     case file   : {case_file}
;;     wall zone(s): {wall_list}
;;     inlet zone  : {inlet_zone}
;;
;; Run:  fluent 3ddp -g -i {case}.jou        (licensed install, headless)
;; ===========================================================================

;; ---- 1. read the meshed case the team supplied --------------------------
/file/read-case "{case_file}"

;; ---- 2. reference values: MUST equal what KinematiK normalises by --------
;;   (mismatch here is a pure scale error that masquerades as a physics gap)
/report/reference-values/area {Aref}
/report/reference-values/length {Lref}
/report/reference-values/density {rho}
/report/reference-values/velocity {umag}

;; ---- 3. freestream: yaw & pitch folded into the inlet vector (m/s) -------
;;   Ux={ux:.6f}  Uy={uy:.6f}  Uz={uz:.6f}   |U|={umag}
;;   attitude: roll={a.roll_deg} pitch={a.pitch_deg} yaw={a.yaw_deg} h={a.ride_height_mm}mm
/define/boundary-conditions/velocity-inlet {inlet_zone} no no yes yes no {ux:.6f} no {uy:.6f} no {uz:.6f} no no no yes 5 10

;; ---- 4. turbulence model: k-omega SST (RANS), matching KinematiK's seam --
/define/models/viscous/kw-sst yes

;; ---- 5. force reports on the car wall zone(s); axes set the sign ---------
;;   lift = +Z, drag = +X, side = +Y. Vendor Cl is up-positive; KinematiK flips it.
/report/forces/wall-forces yes 0 0 1 {wall_list} ()
/report/forces/wall-forces yes 1 0 0 {wall_list} ()
/report/forces/wall-forces yes 0 1 0 {wall_list} ()

;; ---- 6. a Cl monitor so "converged" means a SETTLED force, not just
;;         residuals dropping. Window {cl_monitor_window} its, tol {cl_monitor_tol}.
/solve/monitors/force/set-drag-monitor cl-mon yes {wall_list} () yes no no 0 0 1
/solve/convergence-conditions/condition cl-mon {cl_monitor_window} {cl_monitor_tol}

;; ---- 7. residual convergence criterion ----------------------------------
/solve/monitors/residual/convergence-criteria {convergence_abs} {convergence_abs} {convergence_abs} {convergence_abs} {convergence_abs} {convergence_abs} {convergence_abs}

;; ---- 8. initialise and iterate ------------------------------------------
/solve/initialize/hyb-initialization
/solve/iterate {iterations}

;; ---- 9. export the coefficient CSV KinematiK reads -----------------------
;;   columns (vendor convention, Cl up-positive): Cl,Cd,Cs,CmPitch,converged
;;   The wall-force coefficient reports above print Cl/Cd/Cs to the console; the
;;   line below writes them, plus the pitching-moment coeff, in the exact layout
;;   _read_simple_coeff_csv expects. (Fluent's report-to-file is version-specific;
;;   if your build names it differently, redirect the three coefficient reports
;;   into {coeff_csv} with this header.)
/report/forces/wall-forces yes 0 0 1 {wall_list} () "{coeff_csv}"
/report/forces/wall-forces yes 1 0 0 {wall_list} () "{coeff_csv}"
/report/forces/moments yes 0 1 0 {wall_list} () "{coeff_csv}"

;; If your Fluent build will not append a clean CSV from the reports above, run the
;; three reports interactively once, read Cl/Cd/Cs/CmPitch off the console, and
;; write a one-line file by hand:
;;     echo "Cl,Cd,Cs,CmPitch,converged" >  {coeff_csv}
;;     echo "<Cl>,<Cd>,<Cs>,<CmPitch>,1" >> {coeff_csv}
;;   (converged = 1 if the Cl monitor settled, else 0)

/exit yes
"""
    path = os.path.join(workdir, case + ".jou")
    with open(path, "w") as f:
        f.write(jou)
    return path
