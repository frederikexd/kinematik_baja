# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
CFD backends for the aero co-sim boundary defined in `cfd.py`.

  * `ReferenceAeroModel` — a runnable backend with NO external solver. It wraps a
    transparent analytic attitude model (a smooth, physically-shaped response of
    C_L/C_D/C_side to roll/pitch/yaw/ride-height) so the ENTIRE orchestration, map
    assembly and lap-sim coupling can be exercised and unit-tested today. It is
    labelled `POTENTIAL` fidelity and `is_correlated=False`, and it says loudly that
    it is a stand-in, not a Navier–Stokes solve. This is the aero analogue of
    `ReferenceTireModel`.

  * `FluentVerificationSolver` — the DEFAULT path of the Virtual (Wind) Tunnel. It
    is self-contained: KinematiK computes the aero coefficients INTERNALLY (it wraps
    the same analytic attitude model as `ReferenceAeroModel`), so the user needs NO
    ANSYS Fluent license, no mesh, and no external solver to get a usable answer. At
    the same time, for every case it ALWAYS writes a complete, ready-to-run ANSYS
    Fluent journal (.jou) so the user can independently VERIFY the internal number
    on their own licensed Fluent install whenever they want. The journal is a
    verification artefact, not a prerequisite. Provenance shouts that the headline
    number is an in-house analytic estimate, not a Navier–Stokes solve, and that the
    Fluent deck is provided for confirmation.

  * `OpenFOAMSolver` — a REAL adapter, kept for teams who want to drive a true RANS
    solve themselves. It writes a valid OpenFOAM case (controlDict with a
    forceCoeffs function object, the freestream/attitude as a rotated inlet
    velocity), can invoke `simpleFoam` when OpenFOAM is on PATH, and parses
    postProcessing/forceCoeffs output back into a CoeffResult. With no OpenFOAM
    install it still writes the case and raises SolverUnavailable from run_case.

  * `FluentSolver` / `StarCCMSolver` / `TSAutoSolver` — honest external-solver stubs,
    kept for teams that prefer to run a real licensed solve rather than the in-house
    estimate. They emit a correct driver file (a Fluent journal / STAR-CCM+ Java
    macro / TS-Auto config) that a team runs on their own licensed install, and raise
    SolverUnavailable from run_case until that license and a results path are wired
    in. They NEVER return a fabricated CoeffResult. These are no longer part of the
    default Virtual Tunnel path — that path is now the self-contained
    `FluentVerificationSolver`.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from typing import Optional

from .cfd import (
    Attitude, CaseSpec, CoeffResult, CFDProvenance, SolverFidelity,
    SolverUnavailable,
)


# --------------------------------------------------------------------------- #
#  Reference analytic backend — runnable today, honestly labelled as a stand-in
# --------------------------------------------------------------------------- #
class ReferenceAeroModel:
    """
    A transparent analytic aero response. NOT a CFD solve — a smooth surrogate whose
    only job is to make the orchestration + map + lap-sim machinery runnable and
    testable with no license and no mesh. Every coefficient it returns is derived
    from a handful of FSAE-plausible sensitivities, and its provenance shouts that.

    Physical shape it encodes (all deliberately simple, all sign-correct):
      * baseline downforce and drag at zero attitude,
      * downforce LOST and drag GAINED with yaw (the car stalls in sideslip),
      * a ride-height / ground-effect term (lower => more downforce, to a floor),
      * a pitch term shifting aero balance fore/aft (rake sensitivity),
      * a side-force that grows roughly linearly with yaw.
    """
    name = "reference-analytic"

    def __init__(self,
                 cl0: float = -2.6,        # baseline C_L (negative = downforce)
                 cd0: float = 1.05,        # baseline C_D
                 yaw_cl_loss_per_deg: float = 0.012,   # downforce lost per deg yaw
                 yaw_cd_gain_per_deg: float = 0.010,
                 yaw_cside_per_deg: float = 0.018,
                 ride_ref_mm: float = 30.0,
                 ride_ground_gain: float = 0.010,      # ΔC_L per mm closer to ground
                 pitch_balance_per_deg: float = 0.015, # front-balance shift per deg nose-up
                 baseline_front_balance: float = 0.45):
        self.cl0 = cl0
        self.cd0 = cd0
        self.yaw_cl_loss_per_deg = yaw_cl_loss_per_deg
        self.yaw_cd_gain_per_deg = yaw_cd_gain_per_deg
        self.yaw_cside_per_deg = yaw_cside_per_deg
        self.ride_ref_mm = ride_ref_mm
        self.ride_ground_gain = ride_ground_gain
        self.pitch_balance_per_deg = pitch_balance_per_deg
        self.baseline_front_balance = baseline_front_balance

    def provenance(self) -> CFDProvenance:
        return CFDProvenance(
            backend=self.name,
            fidelity=SolverFidelity.POTENTIAL,
            is_correlated=False,
            turbulence_model="none (analytic surrogate)",
            notes=("Analytic stand-in, NOT a Navier–Stokes solve. Exists to make the "
                   "orchestration/map/lap-sim pipeline runnable and testable without a "
                   "solver. Use for plumbing and trends only; never report as CFD."),
        )

    def _coeffs(self, a: Attitude) -> tuple[float, float, float, float]:
        ay = abs(a.yaw_deg)
        # ground effect: closer than ref gains downforce, with a saturating floor
        dh = self.ride_ref_mm - a.ride_height_mm
        ground = -self.ride_ground_gain * dh
        ground = max(min(ground, 0.6), -0.6)          # saturate
        c_lift = self.cl0 + self.yaw_cl_loss_per_deg * ay + ground
        c_drag = self.cd0 + self.yaw_cd_gain_per_deg * ay + 0.15 * abs(ground)
        c_side = math.copysign(self.yaw_cside_per_deg * ay, a.yaw_deg)
        # pitch shifts aero balance fore/aft; clamp to a sane fraction
        front = self.baseline_front_balance + self.pitch_balance_per_deg * a.pitch_deg
        front = max(min(front, 0.75), 0.25)
        return c_lift, c_drag, c_side, front

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        os.makedirs(workdir, exist_ok=True)
        path = os.path.join(workdir, "reference_case.txt")
        with open(path, "w") as f:
            f.write(f"# reference analytic case — NOT a CFD input deck\n"
                    f"{spec.attitude.label()}\n")
        return path

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        t0 = time.time()
        c_lift, c_drag, c_side, front = self._coeffs(spec.attitude)
        return CoeffResult(
            attitude=spec.attitude,
            c_lift=c_lift, c_drag=c_drag, c_side=c_side,
            c_pitch=None,                       # not modelled — honest hole
            aero_balance_front=front,
            converged=True,                     # the surrogate is "exact" by construction
            force_monitor_range=0.0,
            wall_clock_s=time.time() - t0,
            provenance=self.provenance(),
            notes="analytic surrogate value",
        )

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        return self.run_case(spec, workdir)


# --------------------------------------------------------------------------- #
#  Fluent verification backend — the self-contained default of the Virtual Tunnel
#  Computes coefficients IN-HOUSE; writes an ANSYS Fluent journal to verify against
# --------------------------------------------------------------------------- #
class FluentVerificationSolver:
    """
    The default Virtual (Wind) Tunnel backend, and the one that makes the feature
    usable with NOTHING installed.

    Two jobs, kept deliberately separate so the second never blocks the first:

      1. ANSWER IN-HOUSE.  KinematiK computes the aero coefficients itself, with no
         ANSYS Fluent, license, mesh or external solver needed. Two fidelity levels
         are available via `method`:

           * "panel"    — a 3D source-panel (boundary-element) potential-flow solve
                          on the team's ACTUAL STL, with a ground-image plane. The
                          pressure field, ground effect and downforce trend come from
                          the geometry, not a curve fit, so geometry deltas are
                          meaningful. Inviscid: no separation/stall/wake (that is what
                          the Fluent deck checks). This is the higher-fidelity path.
           * "analytic" — the transparent analytic attitude surrogate
                          (`ReferenceAeroModel`): instant, geometry-insensitive, for
                          plumbing and trends.
           * "auto"     — (default) try the panel solve on the supplied geometry; if
                          there is no usable STL (or trimesh is missing), fall back to
                          the analytic surrogate, recording WHY in the result notes.

         Either way the user gets a number back from `run_case` with nothing installed.

      2. WRITE A FLUENT DECK TO VERIFY.  For every case it ALSO writes a complete,
         ready-to-run ANSYS Fluent journal (`<case>.jou`) with the freestream vector
         (yaw/pitch folded in), reference values and a coefficient-report/export
         block. That deck is the user's independent check: run it on a licensed
         Fluent install when you want to confirm the in-house number. It is a
         verification artefact, never a prerequisite.

    HONESTY CONTRACT.  The headline coefficient is labelled exactly for what it is —
    a potential-flow / analytic in-house estimate at `POTENTIAL` fidelity,
    `is_correlated=False` — not a Navier–Stokes solve, and the provenance/notes say
    so on every result, including which method actually produced it. The Fluent
    journal is generated for confirmation; KinematiK does not launch Fluent and never
    reads back a Fluent number unless you point it at one via `read_fluent_csv`.
    Nothing is fabricated: the estimate is openly an estimate.

    Why Fluent specifically: it is the single solver the verification deck targets,
    replacing the old Star-CCM+ / TS-Auto / OpenFOAM multi-code generation. One code,
    one deck, and a number you already have before you ever open it.
    """
    name = "fluent"

    def __init__(self,
                 fidelity: SolverFidelity = SolverFidelity.POTENTIAL,
                 model: "Optional[ReferenceAeroModel]" = None,
                 method: str = "auto",
                 panel_params: "Optional[object]" = None):
        if method not in ("auto", "panel", "analytic"):
            raise ValueError("method must be 'auto', 'panel' or 'analytic'")
        # The analytic surrogate, used directly ("analytic") or as the "auto"/panel
        # fallback when no usable geometry is present.
        self.model = model if model is not None else ReferenceAeroModel()
        self.fidelity = fidelity
        self.method = method
        # Lazily built panel engine (imported here to avoid a hard numpy/trimesh
        # dependency for callers that only ever use the analytic path).
        self._panel = None
        if method in ("auto", "panel"):
            from .panel_method import PanelMethodModel, PanelParams
            self._panel = PanelMethodModel(panel_params or PanelParams())

    def provenance(self) -> CFDProvenance:
        if self.method == "analytic":
            tm = "none (in-house analytic surrogate)"
            how = ("Computed internally from the analytic attitude surrogate so no "
                   "ANSYS Fluent license, mesh or external solver is required. ")
        elif self.method == "panel":
            tm = "none (in-house 3D panel / potential flow)"
            how = ("Computed internally by a 3D source-panel (boundary-element) "
                   "potential-flow solve on your STL, with a ground-image plane, so "
                   "no ANSYS Fluent license or external solver is required. ")
        else:  # auto
            tm = "none (in-house panel solve, analytic fallback)"
            how = ("Computed internally — a 3D source-panel potential-flow solve on "
                   "your STL when geometry is available, otherwise the analytic "
                   "surrogate — so no ANSYS Fluent license or external solver is "
                   "required. ")
        return CFDProvenance(
            backend=self.name,
            fidelity=self.fidelity,
            is_correlated=False,
            turbulence_model=tm,
            notes=("In-house KinematiK aero estimate — NOT a Navier–Stokes solve. "
                   + how +
                   "A ready-to-run ANSYS Fluent journal is written alongside each "
                   "case so you can independently VERIFY this number on your own "
                   "licensed Fluent. Correlate against a tunnel/coastdown point "
                   "before trusting absolute levels; trust deltas more than levels."),
        )

    # -- the Fluent verification deck ------------------------------------- #
    def write_case(self, spec: CaseSpec, workdir: str,
                   estimate: "Optional[CoeffResult]" = None) -> str:
        """
        Write a RUNNABLE ANSYS Fluent validation journal for one case and return its
        path. The deck reads the team's meshed case, sets the freestream from the
        attitude, configures k-omega SST, initialises, iterates, and exports the
        coefficient CSV that `read_fluent_csv` ingests — so a licensed-Fluent run
        flows straight back into the tunnel correlation.

        Per-run, site-specific bits (the meshed case file and the wall zone name(s)
        the force report targets) are read from `spec.extra` with sane defaults:
            spec.extra['case_file']   -> the .cas/.msh to read   (default geometry_path)
            spec.extra['wall_zones']  -> wall zone name(s)        (default "car")
            spec.extra['inlet_zone']  -> velocity-inlet zone      (default "inlet")
        Pass `estimate` to print the in-house number in the header for comparison.
        """
        from .fluent_journal import write_runnable_journal
        est = estimate if estimate is not None else self._estimate(spec)
        return write_runnable_journal(spec, workdir, estimate=est)

    def _write_stub_case(self, spec: CaseSpec, workdir: str,
                         estimate: "Optional[CoeffResult]" = None) -> str:
        """
        The original header-only stub deck (kept for reference / fallback). Prints
        reference values and the in-house estimate but leaves the solve as TODOs;
        it does NOT run end to end. Prefer `write_case` (runnable).
        """
        os.makedirs(workdir, exist_ok=True)
        ux, uy, uz = OpenFOAMSolver._inlet_velocity(spec.attitude)
        a = spec.attitude
        # The in-house estimate, embedded in the deck header so the user can compare
        # at a glance once Fluent finishes.
        est = estimate if estimate is not None else self._estimate(spec)
        cl_est = "n/a" if est.c_lift is None else f"{est.c_lift:+.4f}"
        cd_est = "n/a" if est.c_drag is None else f"{est.c_drag:+.4f}"
        path = os.path.join(workdir, spec.case_name() + ".jou")
        jou = f""";; ===========================================================================
;; KinematiK-generated ANSYS Fluent verification journal — {spec.case_name()}
;; Attitude: {spec.attitude.label()}
;;
;; PURPOSE: this deck is OPTIONAL. KinematiK has already computed an in-house
;; estimate for this case (see below); run this journal only if you want to verify
;; that estimate on your own licensed ANSYS Fluent install.
;;
;; KinematiK in-house estimate (our sign convention, C_l negative = downforce):
;;     C_l ~= {cl_est}    C_d ~= {cd_est}
;;
;; Run:  fluent 3ddp -g -i {spec.case_name()}.jou      (on a licensed install)
;; ===========================================================================
/file/read-case "{spec.geometry_path}"
;; ---- freestream (yaw/pitch folded into the inlet vector), m/s ----
;;   Ux={ux:.5f}  Uy={uy:.5f}  Uz={uz:.5f}   rho={spec.rho}
;;   magUInf={a.speed_ms}  Aref={spec.reference_area_m2}  Lref={spec.reference_length_m}
;;   attitude: roll={a.roll_deg} pitch={a.pitch_deg} yaw={a.yaw_deg} ride_height_mm={a.ride_height_mm}
;; ---- TODO(team): set the velocity-inlet components to (Ux,Uy,Uz), set the
;;   reference values (area, length, density, velocity), initialise, and iterate
;;   to convergence. Then export the force/moment coefficients: ----
/report/reference-values/area {spec.reference_area_m2}
/report/reference-values/length {spec.reference_length_m}
/report/reference-values/density {spec.rho}
/report/reference-values/velocity {a.speed_ms}
/report/forces/wall-forces yes 0 0 1     ; lift  (Fz)
/report/forces/wall-forces yes 1 0 0     ; drag  (Fx)
;; export a coefficient CSV "{spec.case_name()}_coeffs.csv" with columns:
;;     Cl,Cd,Cs,CmPitch,converged
;; then, back in KinematiK:
;;     FluentVerificationSolver().read_fluent_csv(spec, workdir)
;; to compare the licensed-solver number against the in-house estimate above.
/exit yes
"""
        with open(path, "w") as f:
            f.write(jou)
        return path

    # -- the in-house answer (no external solver needed) ------------------ #
    def _analytic_estimate(self, spec: CaseSpec) -> CoeffResult:
        """The transparent analytic attitude surrogate (geometry-insensitive)."""
        c_lift, c_drag, c_side, front = self.model._coeffs(spec.attitude)
        return CoeffResult(
            attitude=spec.attitude,
            c_lift=c_lift, c_drag=c_drag, c_side=c_side,
            c_pitch=None,
            aero_balance_front=front,
            converged=True,                 # the analytic surrogate is exact by construction
            force_monitor_range=0.0,
            provenance=self.provenance(),
            notes=("in-house analytic surrogate; an ANSYS Fluent verification journal "
                   "was written alongside this case for optional confirmation"),
        )

    def _estimate(self, spec: CaseSpec) -> CoeffResult:
        """
        Compute the coefficients internally. Dispatch on `method`:
          * "analytic" -> the surrogate,
          * "panel"    -> the 3D panel solve (propagates its error if geometry is bad),
          * "auto"     -> try the panel solve, fall back to the surrogate on any
                          PanelMethodUnavailable, recording WHY in the notes.
        A Fluent verification deck is written for the case regardless.
        """
        if self.method == "analytic" or self._panel is None:
            return self._analytic_estimate(spec)

        from .panel_method import PanelMethodUnavailable
        if self.method == "panel":
            res = self._panel.solve(spec)            # may raise PanelMethodUnavailable
            res.notes = (res.notes + "; ANSYS Fluent verification journal written "
                         "alongside for optional confirmation")
            return res

        # auto: panel first, analytic fallback with an honest reason
        try:
            res = self._panel.solve(spec)
            res.notes = (res.notes + "; ANSYS Fluent verification journal written "
                         "alongside for optional confirmation")
            return res
        except PanelMethodUnavailable as e:
            res = self._analytic_estimate(spec)
            res.notes = (f"panel solve unavailable ({e}); fell back to the analytic "
                         "surrogate. " + res.notes)
            return res

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """
        Return the in-house coefficient estimate, and write the Fluent verification
        journal as a side effect. This NEVER raises SolverUnavailable — the whole
        design goal is that the user always gets an answer from KinematiK alone, with
        the Fluent deck left on disk to confirm it if they choose.
        """
        t0 = time.time()
        res = self._estimate(spec)                       # solve once
        self.write_case(spec, workdir, estimate=res)     # reuse it in the deck header
        res.wall_clock_s = time.time() - t0
        return res

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """
        Read back a result for this case. If the user has run the Fluent deck and
        staged `<case>_coeffs.csv`, return THAT (the licensed-solver verification);
        otherwise fall back to the in-house estimate so a read never fails. Either
        way the deck is (re)written so it is always available.
        """
        csv = os.path.join(workdir, spec.case_name() + "_coeffs.csv")
        if os.path.isfile(csv):
            self.write_case(spec, workdir)
            return self.read_fluent_csv(spec, workdir)
        res = self._estimate(spec)
        self.write_case(spec, workdir, estimate=res)
        return res

    def read_fluent_csv(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """
        Parse a Fluent-exported `<case>_coeffs.csv` (Cl,Cd,Cs,CmPitch,converged) the
        user produced by running the verification deck. Use this to fold the licensed
        solver's confirmation back into KinematiK. Raises SolverUnavailable (not a
        fabricated number) if the CSV is not there.
        """
        csv = os.path.join(workdir, spec.case_name() + "_coeffs.csv")
        if not os.path.isfile(csv):
            raise SolverUnavailable(
                f"No Fluent verification CSV at {csv}. This is OPTIONAL — KinematiK's "
                "in-house estimate is available from run_case() without it. To verify, "
                f"run {workdir}/{spec.case_name()}.jou in licensed Fluent and export "
                "the coefficient report there first.")
        prov = CFDProvenance(
            backend=self.name + ":fluent-verified", fidelity=SolverFidelity.RANS,
            is_correlated=False, turbulence_model="(from licensed Fluent run)",
            notes="parsed from a user-run ANSYS Fluent verification export.")
        return _read_simple_coeff_csv(spec, csv, prov)


# --------------------------------------------------------------------------- #
#  OpenFOAM — a real adapter (writes a valid case; runs simpleFoam if present)
# --------------------------------------------------------------------------- #
class OpenFOAMSolver:
    """
    Working OpenFOAM adapter. The free, open backend that fits KinematiK's ethos and
    is the only one we can actually run end-to-end without a license.

    write_case  -> a minimal but valid case skeleton: system/controlDict with a
                   forceCoeffs function object, system/{fvSchemes,fvSolution},
                   and the attitude folded into the inlet velocity vector (yaw and
                   pitch rotate the freestream; roll/ride-height are geometry-side
                   and recorded for the mesher).
    run_case    -> invokes `simpleFoam` if it is on PATH, else raises
                   SolverUnavailable with the exact missing binary.
    read_result -> parses postProcessing/forceCoeffs*/0/coefficient.dat for the
                   converged Cl/Cd/Cm tail.
    """
    name = "openfoam"

    def __init__(self, turbulence_model: str = "kOmegaSST",
                 fidelity: SolverFidelity = SolverFidelity.RANS,
                 application: str = "simpleFoam",
                 mesh_params: "Optional[object]" = None):
        self.turbulence_model = turbulence_model
        self.fidelity = fidelity
        self.application = application
        # When mesh_params is set, write_case also emits the snappyHexMesh tool-chain
        # and run_case meshes before solving. Left None => solver files only (the
        # team supplies their own mesh), preserving the original behaviour.
        self.mesh_params = mesh_params

    def provenance(self, cell_count: Optional[int] = None,
                   yplus: Optional[float] = None) -> CFDProvenance:
        return CFDProvenance(
            backend=self.name,
            fidelity=self.fidelity,
            is_correlated=False,
            turbulence_model=self.turbulence_model,
            cell_count=cell_count,
            yplus_mean=yplus,
            notes="OpenFOAM result. Correlate against a coastdown/tunnel point "
                  "before trusting absolute levels.",
        )

    @staticmethod
    def _inlet_velocity(a: Attitude) -> tuple[float, float, float]:
        """Rotate the freestream by yaw (about z) and pitch (about y)."""
        v = a.speed_ms
        yaw = math.radians(a.yaw_deg)
        pitch = math.radians(a.pitch_deg)
        ux = v * math.cos(yaw) * math.cos(pitch)
        uy = -v * math.sin(yaw)
        uz = v * math.cos(yaw) * math.sin(pitch)
        return ux, uy, uz

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        case = os.path.join(workdir, spec.case_name())
        for d in ("system", "constant", "0"):
            os.makedirs(os.path.join(case, d), exist_ok=True)
        ux, uy, uz = self._inlet_velocity(spec.attitude)
        liftdir = "(0 0 1)"
        dragdir = "(1 0 0)"
        controldict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| KinematiK-generated OpenFOAM controlDict for {spec.case_name()}            |
| Attitude: {spec.attitude.label()}
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}

application     {self.application};
startFrom       startTime;  startTime 0;
stopAt          endTime;    endTime  2000;
deltaT          1;          writeControl timeStep;  writeInterval 200;
purgeWrite      2;          runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;  writeInterval 1;
        patches         (car);
        rho             rhoInf;    rhoInf {spec.rho};
        liftDir         {liftdir}; dragDir {dragdir};
        CofR            (0 0 0);   pitchAxis (0 1 0);
        magUInf         {spec.attitude.speed_ms};
        lRef            {spec.reference_length_m};
        Aref            {spec.reference_area_m2};
    }}
}}
"""
        self._w(case, "system/controlDict", controldict)
        self._w(case, "0/U",
                f"FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\n"
                f"dimensions [0 1 -1 0 0 0 0];\n"
                f"internalField uniform ({ux:.5f} {uy:.5f} {uz:.5f});\n"
                f"// inlet rotated for yaw={spec.attitude.yaw_deg} pitch={spec.attitude.pitch_deg}\n")
        self._w(case, "system/fvSchemes",
                "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
                "ddtSchemes { default steadyState; }\n"
                "gradSchemes { default Gauss linear; }\n"
                "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n")
        self._w(case, "system/fvSolution",
                "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
                "solvers { p { solver GAMG; tolerance 1e-6; relTol 0.01; } "
                "\"(U|k|omega)\" { solver smoothSolver; smoother symGaussSeidel; "
                "tolerance 1e-7; relTol 0.05; } }\n"
                "SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; }\n")
        # turbulence + attitude manifest the mesher reads for roll / ride-height
        self._w(case, "constant/momentumTransport",
                "FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }\n"
                f"simulationType RAS;\nRAS {{ model {self.turbulence_model}; turbulence on; }}\n")
        self._w(case, "kinematik_attitude.json",
                f'{{"roll_deg": {spec.attitude.roll_deg}, '
                f'"pitch_deg": {spec.attitude.pitch_deg}, '
                f'"yaw_deg": {spec.attitude.yaw_deg}, '
                f'"ride_height_mm": {spec.attitude.ride_height_mm}, '
                f'"speed_ms": {spec.attitude.speed_ms}, '
                f'"geometry": "{spec.geometry_path}"}}\n')
        # Optional: emit the full snappyHexMesh tool-chain so a team can mesh the STL
        # at this attitude. Roll + ride-height are applied geometry-side by the mesher;
        # pitch + yaw stay on the inlet velocity above. No mesh is run here.
        if self.mesh_params is not None:
            from .meshing import SnappyMesher
            SnappyMesher(self.mesh_params).write(spec, case)
        return case

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        case = os.path.join(workdir, spec.case_name())
        if not os.path.isdir(case):
            self.write_case(spec, workdir)
        # If meshing was requested, mesh first (Allmesh), then solve.
        if self.mesh_params is not None:
            self._run_mesh(spec, case)
        if shutil.which(self.application) is None:
            raise SolverUnavailable(
                f"OpenFOAM application '{self.application}' is not on PATH. The case "
                f"was written to {case}; run it on a machine with OpenFOAM (source "
                f"the OpenFOAM environment, then `{self.application}` in that case "
                f"directory), or point KinematiK at an OpenFOAM-enabled cluster.")
        t0 = time.time()
        subprocess.run([self.application], cwd=case, check=True,
                       capture_output=True, text=True)
        res = self.read_result(spec, workdir)
        res.wall_clock_s = time.time() - t0
        return res

    def _run_mesh(self, spec: CaseSpec, case: str) -> None:
        """Run the generated Allmesh script if the geometry + OpenFOAM are present;
        otherwise raise an actionable SolverUnavailable rather than solving on no mesh."""
        allmesh = os.path.join(case, "Allmesh")
        stl = os.path.join(case, "constant", "triSurface",
                           os.path.basename(spec.geometry_path) or "car.stl")
        if not os.path.isfile(stl):
            raise SolverUnavailable(
                f"Meshing requested but the STL is not staged at {stl}. Copy your "
                f"surface there (the Allmesh script checks for it), then re-run. The "
                f"snappyHexMesh dictionaries are already written in {case}/system.")
        if shutil.which("snappyHexMesh") is None:
            raise SolverUnavailable(
                f"snappyHexMesh is not on PATH. The full meshing tool-chain was "
                f"written to {case} (run ./Allmesh on an OpenFOAM machine/cluster). "
                f"KinematiK will not solve on an absent mesh.")
        subprocess.run(["sh", allmesh], cwd=case, check=True,
                       capture_output=True, text=True)

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        case = os.path.join(workdir, spec.case_name())
        dat = self._find_coeff_file(case)
        if dat is None:
            raise SolverUnavailable(
                f"No forceCoeffs output found under {case}/postProcessing. The case "
                f"has not been run, or the forceCoeffs function object did not write. "
                f"Run {self.application} in that case directory first.")
        cl, cd, cm, conv, spread = self._parse_coeff_file(dat)
        # Real cell count comes ONLY from a mesh log, never from the recipe budget.
        cell_count = None
        try:
            from .meshing import parse_checkmesh
            cell_count = parse_checkmesh(case)
        except Exception:                       # noqa: BLE001
            cell_count = None
        prov = self.provenance(cell_count=cell_count)
        return CoeffResult(
            attitude=spec.attitude,
            c_lift=-cl if cl is not None else None,   # OF Cl up-positive -> our down-negative
            c_drag=cd, c_side=None, c_pitch=cm,
            aero_balance_front=None,
            converged=conv, force_monitor_range=spread,
            provenance=prov,
            notes="parsed from OpenFOAM forceCoeffs",
        )

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _w(case: str, rel: str, text: str) -> None:
        path = os.path.join(case, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    @staticmethod
    def _find_coeff_file(case: str) -> Optional[str]:
        base = os.path.join(case, "postProcessing")
        if not os.path.isdir(base):
            return None
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if fn in ("coefficient.dat", "forceCoeffs.dat"):
                    return os.path.join(root, fn)
        return None

    @staticmethod
    def _parse_coeff_file(path: str):
        """Return (Cl, Cd, Cm, converged, last-tail spread) from a forceCoeffs file."""
        cd_vals, cl_vals, cm_vals = [], [], []
        header_cols = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "Cd" in line and "Cl" in line:
                        header_cols = line.lstrip("#").split()
                    continue
                parts = line.split()
                try:
                    nums = [float(x) for x in parts]
                except ValueError:
                    continue
                # Default OF column order: time Cd Cs Cl CmRoll CmPitch CmYaw Cd(f) ...
                if header_cols:
                    idx = {c: i for i, c in enumerate(header_cols)}
                    def get(name):
                        i = idx.get(name)
                        return nums[i] if i is not None and i < len(nums) else None
                    cd_vals.append(get("Cd")); cl_vals.append(get("Cl"))
                    cm_vals.append(get("CmPitch"))
                else:
                    if len(nums) >= 4:
                        cd_vals.append(nums[1]); cl_vals.append(nums[3])
                        cm_vals.append(nums[5] if len(nums) > 5 else None)
        cd_vals = [v for v in cd_vals if v is not None]
        cl_vals = [v for v in cl_vals if v is not None]
        cm_vals = [v for v in cm_vals if v is not None]
        if not cl_vals or not cd_vals:
            return None, None, None, False, None
        tail = max(1, len(cl_vals) // 10)
        cl_tail = cl_vals[-tail:]
        spread = (max(cl_tail) - min(cl_tail)) / (abs(sum(cl_tail) / len(cl_tail)) + 1e-9)
        converged = spread < 0.01            # <1% spread over last 10% of iters
        return (sum(cl_tail) / len(cl_tail),
                sum(cd_vals[-tail:]) / tail,
                (sum(cm_vals[-tail:]) / tail) if cm_vals else None,
                converged, spread)


# --------------------------------------------------------------------------- #
#  STAR-CCM+ — honest stub: writes a Java macro, refuses to fake a run
# --------------------------------------------------------------------------- #
class StarCCMSolver:
    """
    STAR-CCM+ adapter STUB. STAR-CCM+ is a commercial license KinematiK does not
    hold and cannot run here, so this writes a correct driver Java macro the team
    runs on their own licensed install, and raises SolverUnavailable from run_case
    rather than inventing a coefficient. read_result parses a CSV the macro exports.
    """
    name = "starccm"

    def __init__(self, fidelity: SolverFidelity = SolverFidelity.RANS):
        self.fidelity = fidelity

    def provenance(self, cell_count: Optional[int] = None) -> CFDProvenance:
        return CFDProvenance(
            backend=self.name, fidelity=self.fidelity, is_correlated=False,
            cell_count=cell_count,
            notes="STAR-CCM+ adapter stub — runs on the team's licensed install.",
        )

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        os.makedirs(workdir, exist_ok=True)
        ux, uy, uz = OpenFOAMSolver._inlet_velocity(spec.attitude)
        path = os.path.join(workdir, spec.case_name() + ".java")
        macro = f"""// KinematiK-generated STAR-CCM+ macro — {spec.case_name()}
// Attitude: {spec.attitude.label()}
// Run inside STAR-CCM+: File > Macro > Play, on the team's licensed install.
import star.common.*; import star.base.neo.*; import star.flow.*;
public class {spec.case_name()} extends StarMacro {{
  public void execute() {{
    Simulation sim = getActiveSimulation();
    // import surface: {spec.geometry_path}
    // freestream (yaw/pitch folded into inlet vector), m/s:
    double[] U = {{ {ux:.5f}, {uy:.5f}, {uz:.5f} }};
    double rho = {spec.rho}, Aref = {spec.reference_area_m2}, Lref = {spec.reference_length_m};
    // TODO(team): set inlet velocity to U, density rho, run, and export a
    // force-coefficient report to "{spec.case_name()}_coeffs.csv" with columns:
    //   Cl,Cd,Cs,CmPitch,converged
    sim.println("KinematiK STAR-CCM+ macro ready: {spec.attitude.label()}");
  }}
}}
"""
        with open(path, "w") as f:
            f.write(macro)
        return path

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        self.write_case(spec, workdir)
        raise SolverUnavailable(
            "STAR-CCM+ cannot be launched by KinematiK (commercial license required). "
            f"A driver macro was written to {workdir}/{spec.case_name()}.java — play it "
            "in your licensed STAR-CCM+ install (locally or via its batch mode on your "
            "cluster), have it export <case>_coeffs.csv, then call read_result().")

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        csv = os.path.join(workdir, spec.case_name() + "_coeffs.csv")
        if not os.path.isfile(csv):
            raise SolverUnavailable(
                f"No STAR-CCM+ result CSV at {csv}. Run the macro in STAR-CCM+ and "
                "export the coefficient report there first.")
        return _read_simple_coeff_csv(spec, csv, self.provenance())


# --------------------------------------------------------------------------- #
#  Fluent — honest stub: writes a journal, refuses to fake a run
# --------------------------------------------------------------------------- #
class FluentSolver:
    """
    ANSYS Fluent external-solver STUB. Same discipline as the STAR-CCM+ stub: writes
    a TUI journal the team runs on their licensed Fluent, refuses to fabricate a run.

    NOTE: this is the pure pass-through stub (it computes NOTHING in-house and raises
    SolverUnavailable from run_case). The default Virtual Tunnel path is instead
    `FluentVerificationSolver`, which computes the coefficients in-house AND writes a
    Fluent deck to verify. Use this stub directly only if you specifically want a
    Fluent deck with no in-house estimate attached.
    """
    name = "fluent-stub"

    def __init__(self, fidelity: SolverFidelity = SolverFidelity.RANS):
        self.fidelity = fidelity

    def provenance(self, cell_count: Optional[int] = None) -> CFDProvenance:
        return CFDProvenance(
            backend=self.name, fidelity=self.fidelity, is_correlated=False,
            cell_count=cell_count,
            notes="Fluent adapter stub — runs on the team's licensed install.",
        )

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        os.makedirs(workdir, exist_ok=True)
        ux, uy, uz = OpenFOAMSolver._inlet_velocity(spec.attitude)
        path = os.path.join(workdir, spec.case_name() + ".jou")
        jou = f""";; KinematiK-generated Fluent journal — {spec.case_name()}
;; Attitude: {spec.attitude.label()}
;; Run:  fluent 3ddp -g -i {spec.case_name()}.jou   (on a licensed install)
/file/read-case "{spec.geometry_path}"
;; freestream components (yaw/pitch folded in), m/s:
;;   Ux={ux:.5f} Uy={uy:.5f} Uz={uz:.5f}   rho={spec.rho}
;;   Aref={spec.reference_area_m2}  Lref={spec.reference_length_m}
;; TODO(team): set velocity-inlet components to (Ux,Uy,Uz), reference values,
;; init, iterate to convergence, then:
/report/forces/wall-forces yes 0 0 1     ; lift
/report/forces/wall-forces yes 1 0 0     ; drag
;; export a coeff CSV "{spec.case_name()}_coeffs.csv": Cl,Cd,Cs,CmPitch,converged
/exit yes
"""
        with open(path, "w") as f:
            f.write(jou)
        return path

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        self.write_case(spec, workdir)
        raise SolverUnavailable(
            "Fluent cannot be launched by KinematiK (commercial license required). "
            f"A journal was written to {workdir}/{spec.case_name()}.jou — run it in your "
            "licensed Fluent (`fluent 3ddp -g -i <case>.jou`), export <case>_coeffs.csv, "
            "then call read_result().")

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        csv = os.path.join(workdir, spec.case_name() + "_coeffs.csv")
        if not os.path.isfile(csv):
            raise SolverUnavailable(
                f"No Fluent result CSV at {csv}. Run the journal in Fluent and export "
                "the coefficient report there first.")
        return _read_simple_coeff_csv(spec, csv, self.provenance())


# --------------------------------------------------------------------------- #
#  TS-Auto (TotalSim) — honest stub: writes a run config, refuses to fake a run
# --------------------------------------------------------------------------- #
class TSAutoSolver:
    """
    TotalSim TS-Auto adapter STUB. TS-Auto is a productised automotive aero workflow
    built on an OpenFOAM core, run on the vendor's / team's licensed install. Same
    discipline as the STAR-CCM+ and Fluent stubs: KinematiK writes a faithful run
    descriptor (a JSON/YAML-style config the TS-Auto launcher consumes, with the
    attitude folded into the inlet vector and the reference values made explicit),
    and raises SolverUnavailable from run_case rather than inventing a coefficient.
    read_result parses the coefficient CSV the workflow exports.

    This is the second solver the wind-tunnel correlation path targets (alongside
    STAR-CCM+): a team maps the physical aero map, then runs the IDENTICAL ride-
    height/speed points through TS-Auto's Virtual Wind Tunnel and feeds the results
    back into VirtualWindTunnel.correlate() to calibrate k-omega SST.
    """
    name = "tsauto"

    def __init__(self, turbulence_model: str = "kOmegaSST",
                 fidelity: SolverFidelity = SolverFidelity.RANS):
        self.turbulence_model = turbulence_model
        self.fidelity = fidelity

    def provenance(self, cell_count: Optional[int] = None) -> CFDProvenance:
        return CFDProvenance(
            backend=self.name, fidelity=self.fidelity, is_correlated=False,
            turbulence_model=self.turbulence_model, cell_count=cell_count,
            notes="TS-Auto (TotalSim) adapter stub — runs on the team's licensed "
                  "install; correlate against the physical tunnel map before trusting "
                  "absolute levels.",
        )

    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        os.makedirs(workdir, exist_ok=True)
        ux, uy, uz = OpenFOAMSolver._inlet_velocity(spec.attitude)
        path = os.path.join(workdir, spec.case_name() + "_tsauto.json")
        a = spec.attitude
        config = f"""{{
  "_comment": "KinematiK-generated TS-Auto run config — {spec.case_name()}",
  "_attitude": "{a.label()}",
  "_run": "Play in TS-Auto on the team's licensed install; export coeffs CSV.",
  "geometry": "{spec.geometry_path}",
  "turbulenceModel": "{self.turbulence_model}",
  "freestream_ms": {{"ux": {ux:.5f}, "uy": {uy:.5f}, "uz": {uz:.5f}}},
  "magUInf_ms": {a.speed_ms},
  "rho": {spec.rho},
  "referenceArea_m2": {spec.reference_area_m2},
  "referenceLength_m": {spec.reference_length_m},
  "attitude": {{"roll_deg": {a.roll_deg}, "pitch_deg": {a.pitch_deg},
               "yaw_deg": {a.yaw_deg}, "ride_height_mm": {a.ride_height_mm}}},
  "exports": {{"coeffsCsv": "{spec.case_name()}_coeffs.csv",
              "columns": ["Cl", "Cd", "Cs", "CmPitch", "converged"]}}
}}
"""
        with open(path, "w") as f:
            f.write(config)
        return path

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        self.write_case(spec, workdir)
        raise SolverUnavailable(
            "TS-Auto cannot be launched by KinematiK (licensed vendor workflow). "
            f"A run config was written to {workdir}/{spec.case_name()}_tsauto.json — "
            "run it in your TS-Auto install, export <case>_coeffs.csv "
            "(Cl,Cd,Cs,CmPitch,converged), then call read_result().")

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        csv = os.path.join(workdir, spec.case_name() + "_coeffs.csv")
        if not os.path.isfile(csv):
            raise SolverUnavailable(
                f"No TS-Auto result CSV at {csv}. Run the config in TS-Auto and "
                "export the coefficient report there first.")
        return _read_simple_coeff_csv(spec, csv, self.provenance())


# --------------------------------------------------------------------------- #
#  Shared CSV reader for the commercial stubs (Cl,Cd,Cs,CmPitch,converged)
# --------------------------------------------------------------------------- #
def _read_simple_coeff_csv(spec: CaseSpec, csv_path: str,
                           prov: CFDProvenance) -> CoeffResult:
    with open(csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise SolverUnavailable(f"{csv_path} has no data row.")
    # header is the first non-data line, with or without a leading '#'
    header_line = lines[0].lstrip("#").strip()
    header = [h.strip().lower() for h in header_line.split(",")]
    data_line = lines[-1]
    vals = data_line.split(",")
    row = {}
    for h, v in zip(header, vals):
        v = v.strip()
        try:
            row[h] = float(v)
        except ValueError:
            row[h] = v
    def g(name):
        return row.get(name) if isinstance(row.get(name), float) else None
    conv = row.get("converged")
    if isinstance(conv, float):
        conv = conv != 0.0
    elif conv is not None:
        conv = str(conv).strip().lower() in ("1", "true", "yes")
    else:
        conv = False
    cl = g("cl")
    return CoeffResult(
        attitude=spec.attitude,
        c_lift=(-cl if cl is not None else None),   # vendor Cl up-positive -> down-negative
        c_drag=g("cd"), c_side=g("cs"), c_pitch=g("cmpitch"),
        converged=conv, provenance=prov,
        notes=f"parsed from {os.path.basename(csv_path)}",
    )


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
def _make_ensemble(**kwargs):
    """Factory for the Virtual Tunnel Solver (imported lazily to avoid a cycle:
    ensemble.py imports the member backends from this module)."""
    from .ensemble import EnsembleTunnelSolver
    return EnsembleTunnelSolver(**kwargs)


BACKENDS = {
    "reference": ReferenceAeroModel,
    "fluent": FluentVerificationSolver,      # in-house estimate + Fluent deck (default)
    "fluent-stub": FluentSolver,             # pure pass-through Fluent deck, no estimate
    "openfoam": OpenFOAMSolver,
    "starccm": StarCCMSolver,
    "tsauto": TSAutoSolver,
    # The Virtual Tunnel Solver — now self-contained on the in-house Fluent backend.
    "virtual-tunnel": _make_ensemble,
}


def get_backend(name: str, **kwargs):
    key = name.lower().replace("-", "").replace("+", "").replace("ccm", "ccm")
    aliases = {"reference": "reference", "referenceanalytic": "reference",
               # Fluent / ANSYS resolve to the self-contained in-house verification
               # solver (computes coefficients internally, writes a Fluent deck).
               "fluent": "fluent", "ansys": "fluent", "ansysfluent": "fluent",
               "fluentverify": "fluent", "fluentverification": "fluent",
               # the pure external stub, for teams that want only the deck
               "fluentstub": "fluent-stub",
               "openfoam": "openfoam", "of": "openfoam",
               "starccm": "starccm", "star": "starccm", "starccmplus": "starccm",
               "tsauto": "tsauto", "ts": "tsauto", "totalsim": "tsauto",
               "virtualtunnel": "virtual-tunnel", "ensemble": "virtual-tunnel",
               "vts": "virtual-tunnel", "consensus": "virtual-tunnel"}
    cls = BACKENDS.get(aliases.get(key, key))
    if cls is None:
        raise KeyError(f"unknown CFD backend '{name}'; have {sorted(BACKENDS)}")
    return cls(**kwargs)
