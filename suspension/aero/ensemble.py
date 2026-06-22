# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Virtual Tunnel Solver — self-contained on KinematiK's in-house ANSYS-Fluent backend.

WHY THIS MODULE EXISTS (read this before using it)
---------------------------------------------------
The Virtual (Wind) Tunnel used to generate driver files for THREE external codes at
once — Star-CCM+, TS-Auto and OpenFOAM — and fuse their outputs to flag inter-code
divergence. That made the feature unusable for anyone who did not own those licenses
and a cluster to run them on: you got driver files and nothing else until you ran
them elsewhere.

This module now does the opposite. Everything happens INSIDE KinematiK:

  * the headline coefficient is computed in-house by `FluentVerificationSolver`
    (the analytic attitude model), so the user needs NO ANSYS Fluent, no other
    solver, no license and no mesh to get a usable number, and
  * for every case a complete ANSYS Fluent journal is still written, purely so the
    user can independently VERIFY the in-house number on their own Fluent install if
    and when they want to. The deck is a confirmation artefact, never a prerequisite.

The class is still called `EnsembleTunnelSolver` and still implements the full
`CFDSolver` seam (`write_case` / `run_case` / `read_result` + `provenance`), so it
remains a drop-in everywhere a backend plugs in — including
`VirtualWindTunnel.case_specs()` / `.correlate()`. What changed is its default
roster: instead of three external codes it now wraps a SINGLE member, the in-house
Fluent backend. The fusion machinery (mean/median reduction, inter-code spread,
honest holes) is fully retained for the advanced case where a caller deliberately
supplies several backends to cross-check — e.g. adding a real OpenFOAM solve next to
the in-house estimate — but the out-of-the-box behaviour is one self-contained code.

    write_case   -> writes the ANSYS Fluent verification journal for one attitude
                    (under a `fluent/` sub-directory), so the deck is on disk to run
                    whenever the user wants to confirm.
    run_case     -> returns the in-house coefficient for that attitude immediately,
                    with no external solver; the Fluent deck is left alongside it.
    read_result  -> returns the in-house number, or — if the user has run the deck
                    and staged a Fluent coeff CSV — the licensed-solver confirmation.

THE HONESTY CONTRACT (same discipline as cfd.py / backends.py)
--------------------------------------------------------------
  * The in-house coefficient is labelled exactly for what it is: an analytic estimate
    at `POTENTIAL` fidelity, `is_correlated=False`. It is never dressed up as a
    Navier–Stokes solve, and its provenance/notes say so on every result.
  * When more than one member is supplied, the fused coefficient is a transparent
    reduction (mean / median) of ONLY the members that produced a converged number;
    a member that raised SolverUnavailable or did not converge contributes NOTHING.
  * `c_lift` sign convention is preserved end to end (negative = downforce).

DELIBERATE NON-GOALS, identical to the seam it sits on: this module meshes nothing
and solves no Navier-Stokes itself. The in-house number is an openly-labelled
analytic estimate; the Fluent deck it writes is for the user to run, not KinematiK.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .cfd import (
    Attitude, CaseSpec, CoeffResult, CFDProvenance, SolverFidelity,
    SolverUnavailable,
)
from .backends import FluentVerificationSolver


# --------------------------------------------------------------------------- #
#  Default roster: the single in-house Fluent backend the Virtual Tunnel runs on
# --------------------------------------------------------------------------- #
# The Virtual Tunnel Solver is now self-contained: by default it wraps ONE member,
# KinematiK's in-house `FluentVerificationSolver`, which computes the coefficient
# internally and writes an ANSYS Fluent journal for optional verification. The
# member-based machinery below still supports several backends, so a caller can pass
# `members=[...]` to deliberately cross-check the in-house number against a real
# OpenFOAM solve — but no external solver is needed for the default path.
DEFAULT_MEMBER_NAMES = ("fluent",)


def _default_members(turbulence_model: str, fidelity: SolverFidelity,
                     mesh_params=None) -> "list":
    """
    Construct the default roster — a single in-house Fluent backend. `mesh_params`
    and `turbulence_model` are accepted for signature compatibility with the old
    multi-code roster but are not needed by the in-house estimate (there is no mesh
    and no turbulence closure to drive); they are ignored here.
    """
    return [FluentVerificationSolver()]


# --------------------------------------------------------------------------- #
#  Per-member outcome — one code's contribution to one fused point
# --------------------------------------------------------------------------- #
@dataclass
class MemberOutcome:
    """
    What ONE code did at ONE attitude. `result` is the member's CoeffResult if it
    produced one; `error` is the actionable reason it didn't (e.g. the
    SolverUnavailable message a licensed stub raises here). Exactly one of the two
    is set. This is the audit trail behind a fused number: you can always see which
    codes voted and which were holes, and why.
    """
    backend: str
    result: Optional[CoeffResult] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        """A usable vote: the member ran, converged, and has lift+drag."""
        return self.result is not None and self.result.is_usable()

    @property
    def ran(self) -> bool:
        """The member produced SOME result (maybe unconverged), not an exception."""
        return self.result is not None


# --------------------------------------------------------------------------- #
#  Fused result — the consensus coefficient plus the inter-code spread
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleResult:
    """
    The Virtual Tunnel Solver's answer at one attitude: the fused `CoeffResult`
    (cross-code consensus, the object the rest of KinematiK consumes) plus the raw
    per-member outcomes and the per-channel spread that earned that consensus its
    `converged` verdict. `spread_pct` is the peak-to-peak disagreement between the
    converged members as a percentage of their mean — the single number that says
    whether the codes actually agree.
    """
    fused: CoeffResult
    members: list                          # list[MemberOutcome]
    n_voted: int
    cl_spread_pct: float = float("nan")
    cd_spread_pct: float = float("nan")

    def as_dict(self):
        return dict(
            attitude=self.fused.attitude.label(),
            c_lift=self.fused.c_lift, c_drag=self.fused.c_drag,
            converged=self.fused.converged,
            n_voted=self.n_voted,
            cl_spread_pct=self.cl_spread_pct,
            cd_spread_pct=self.cd_spread_pct,
            members=[{"backend": m.backend,
                      "ok": m.ok,
                      "c_lift": (m.result.c_lift if m.ran else None),
                      "c_drag": (m.result.c_drag if m.ran else None),
                      "error": m.error}
                     for m in self.members],
        )


def _spread_pct(values: Sequence[float]) -> float:
    """Peak-to-peak as a percentage of |mean|; nan if <2 values or mean ~0."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return float("nan")
    return 100.0 * (max(vals) - min(vals)) / abs(mean)


def _reduce(values: Sequence[float], how: str) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if how == "median":
        return float(statistics.median(vals))
    return float(sum(vals) / len(vals))      # default: mean


# --------------------------------------------------------------------------- #
#  The Virtual Tunnel Solver — a CFDSolver built on the three codes
# --------------------------------------------------------------------------- #
class EnsembleTunnelSolver:
    """
    The Virtual Tunnel Solver. It implements the `CFDSolver` protocol (so it drops
    into `VirtualWindTunnel`, `AeroOrchestrator`, and anywhere a backend is taken).

    By default it is SELF-CONTAINED: it wraps a single member, KinematiK's in-house
    `FluentVerificationSolver`, which computes the aero coefficient internally (no
    ANSYS Fluent, no other solver, no license, no mesh) and writes an ANSYS Fluent
    journal alongside each case so the user can independently verify that number on
    their own Fluent install. With one member there is nothing to "fuse" — the result
    is simply the in-house estimate, openly labelled as such.

    The fusion machinery is retained for the advanced case: pass `members=[...]` with
    several backends (e.g. the in-house estimate plus a real OpenFOAM solve) and the
    solver will reduce the converged members to a consensus and report the inter-code
    spread — agreement between independent solvers is the strongest cheap evidence a
    number is physical. That spread is recorded on every multi-member result.

    Parameters
    ----------
    reduction       : "mean" (default) or "median" — how converged members are
                      combined into the consensus coefficient (only meaningful with
                      more than one member).
    agreement_tol   : maximum inter-code spread (% of mean, peak-to-peak) for a
                      multi-member fused result to be called `converged`.
    min_members     : minimum number of converged members required to report a
                      number. Default 1 — the in-house solver alone IS the answer.
                      Raise it when you supply several backends and want to require
                      cross-code agreement.
    turbulence_model: accepted for compatibility; the in-house estimate has no
                      turbulence closure, but any solver members you add can use it.
    fidelity        : labelled fidelity of the ensemble.
    mesh_params     : accepted for compatibility; ignored by the in-house estimate,
                      passed through if you supply solver members that mesh.
    members         : advanced — supply your own list of CFDSolver backends to
                      cross-check instead of the default single in-house code.
    """
    name = "virtual-tunnel"

    def __init__(self,
                 reduction: str = "mean",
                 agreement_tol: float = 5.0,
                 min_members: int = 1,
                 turbulence_model: str = "kOmegaSST",
                 fidelity: SolverFidelity = SolverFidelity.RANS,
                 mesh_params=None,
                 members: "Optional[list]" = None):
        if reduction not in ("mean", "median"):
            raise ValueError("reduction must be 'mean' or 'median'")
        self.reduction = reduction
        self.agreement_tol = float(agreement_tol)
        self.min_members = max(1, int(min_members))
        self.turbulence_model = turbulence_model
        self.fidelity = fidelity
        self.members = (members if members is not None
                        else _default_members(turbulence_model, fidelity,
                                              mesh_params))
        if not self.members:
            raise ValueError("EnsembleTunnelSolver needs at least one member backend")
        self._member_names = [getattr(m, "name", f"member{i}")
                              for i, m in enumerate(self.members)]

    # -- provenance ------------------------------------------------------- #
    def provenance(self, n_voted: Optional[int] = None,
                   spread_pct: Optional[float] = None,
                   member_names: "Optional[list]" = None) -> CFDProvenance:
        members = member_names if member_names is not None else self._member_names
        roster = "+".join(members)
        vote = "" if n_voted is None else f", {n_voted}/{len(members)} codes voted"
        spr = "" if spread_pct is None or spread_pct != spread_pct \
            else f", inter-code spread {spread_pct:.1f}%"
        return CFDProvenance(
            backend=f"{self.name}[{roster}]",
            fidelity=self.fidelity,
            is_correlated=False,
            turbulence_model=self.turbulence_model,
            notes=("Virtual Tunnel Solver — self-contained on KinematiK's in-house "
                   "ANSYS-Fluent backend. The coefficient is computed internally (no "
                   "external solver, license or mesh required) and a Fluent journal "
                   "is written for optional verification. When several backends are "
                   "supplied the result is the "
                   f"{self.reduction} of the converged members only; members that "
                   "could not run or did not converge contribute nothing and are "
                   "recorded as holes, and inter-code agreement is the confidence "
                   "signal. Correlate against the physical tunnel map before "
                   f"trusting absolute levels{vote}{spr}."),
        )

    # -- the CFDSolver seam: write / run / read --------------------------- #
    def write_case(self, spec: CaseSpec, workdir: str) -> str:
        """
        Write each member's input for this attitude, each into its own sub-directory
        under <workdir>/<case_name>/<member>. By default this is a single `fluent/`
        folder containing the ANSYS Fluent verification journal. Returns the parent
        case directory. If you supplied extra solver members, each gets its own
        sub-folder too.
        """
        case_dir = os.path.join(workdir, spec.case_name())
        os.makedirs(case_dir, exist_ok=True)
        for member, mname in zip(self.members, self._member_names):
            sub = os.path.join(case_dir, mname)
            os.makedirs(sub, exist_ok=True)
            try:
                member.write_case(spec, sub)
            except Exception as e:                          # noqa: BLE001
                # A member that cannot even write its input is recorded, not fatal —
                # the other codes still get written. We leave a breadcrumb file.
                with open(os.path.join(sub, "WRITE_FAILED.txt"), "w") as f:
                    f.write(f"{mname} write_case failed: {e}\n")
        return case_dir

    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """
        Drive every member at this attitude, then reduce. The default in-house Fluent
        backend returns its internally-computed coefficient immediately (no external
        solver) and leaves a Fluent verification journal on disk. If you supplied
        extra solver members, each that cannot run here is captured as an honest hole,
        never faked. The returned CoeffResult is the result (the consensus when there
        is more than one member); the full per-member breakdown is available via
        `solve_detailed`.
        """
        return self.solve_detailed(spec, workdir).fused

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        """
        Parse whatever each member already produced under its sub-directory and fuse,
        without launching anything. Use this after a team has run the written cases
        on their cluster and staged each code's result back.
        """
        return self._fuse_from(spec, workdir, run=False).fused

    # -- the ensemble engine ---------------------------------------------- #
    def solve_detailed(self, spec: CaseSpec, workdir: str) -> EnsembleResult:
        """Run (where possible) + fuse, returning the full EnsembleResult."""
        return self._fuse_from(spec, workdir, run=True)

    def _fuse_from(self, spec: CaseSpec, workdir: str, run: bool) -> EnsembleResult:
        case_dir = os.path.join(workdir, spec.case_name())
        outcomes: list[MemberOutcome] = []
        for member, mname in zip(self.members, self._member_names):
            sub = os.path.join(case_dir, mname)
            os.makedirs(sub, exist_ok=True)
            outcomes.append(self._drive_member(member, mname, spec, sub, run))
        return self._fuse(spec, outcomes)

    def _drive_member(self, member, mname: str, spec: CaseSpec, sub: str,
                      run: bool) -> MemberOutcome:
        """Run or read ONE member, capturing an unavailable/failed code as a hole."""
        try:
            if run:
                res = member.run_case(spec, sub)
            else:
                res = member.read_result(spec, sub)
            return MemberOutcome(backend=mname, result=res)
        except SolverUnavailable as e:
            return MemberOutcome(backend=mname, error=str(e))
        except Exception as e:                              # noqa: BLE001
            return MemberOutcome(backend=mname, error=f"{type(e).__name__}: {e}")

    def _fuse(self, spec: CaseSpec, outcomes: "list") -> EnsembleResult:
        """
        Reduce the converged members into one consensus CoeffResult. ONLY usable
        members vote; the spread between them sets the converged verdict. Nothing is
        invented to fill a hole.
        """
        voting = [m for m in outcomes if m.ok]
        n_voted = len(voting)

        # No usable member: an honest, fully-unconverged hole carrying the reasons.
        if n_voted == 0:
            why = "; ".join(f"{m.backend}: {m.error or 'no usable result'}"
                            for m in outcomes)
            fused = CoeffResult(
                attitude=spec.attitude,
                converged=False,
                provenance=self.provenance(n_voted=0,
                                           member_names=[m.backend for m in outcomes]),
                notes=f"Virtual Tunnel Solver: no code produced a usable result — {why}",
            )
            return EnsembleResult(fused=fused, members=outcomes, n_voted=0)

        cls = [m.result.c_lift for m in voting]
        cds = [m.result.c_drag for m in voting]
        csides = [m.result.c_side for m in voting if m.result.c_side is not None]
        cpitch = [m.result.c_pitch for m in voting if m.result.c_pitch is not None]
        bals = [m.result.aero_balance_front for m in voting
                if m.result.aero_balance_front is not None]

        cl_spread = _spread_pct(cls)
        cd_spread = _spread_pct(cds)

        # Converged consensus requires enough codes AND that they agree. With a
        # single voting member spread is nan (no disagreement to measure) — then the
        # min_members gate alone decides, so a lone code can only pass if you set
        # min_members=1 on purpose.
        enough = n_voted >= self.min_members
        agree = True
        for s in (cl_spread, cd_spread):
            if s == s and s > self.agreement_tol:        # s==s filters nan
                agree = False
        converged = bool(enough and agree)

        # Worst-channel spread, for the human-facing number on the result.
        worst_spread = max((s for s in (cl_spread, cd_spread) if s == s),
                           default=float("nan"))

        note = self._fuse_note(outcomes, n_voted, cl_spread, cd_spread,
                               enough, agree)
        fused = CoeffResult(
            attitude=spec.attitude,
            c_lift=_reduce(cls, self.reduction),
            c_drag=_reduce(cds, self.reduction),
            c_side=_reduce(csides, self.reduction) if csides else None,
            c_pitch=_reduce(cpitch, self.reduction) if cpitch else None,
            aero_balance_front=_reduce(bals, self.reduction) if bals else None,
            converged=converged,
            force_monitor_range=worst_spread / 100.0 if worst_spread == worst_spread
            else None,
            provenance=self.provenance(n_voted=n_voted, spread_pct=worst_spread,
                                       member_names=[m.backend for m in voting]),
            notes=note,
        )
        return EnsembleResult(fused=fused, members=outcomes, n_voted=n_voted,
                              cl_spread_pct=cl_spread, cd_spread_pct=cd_spread)

    def _fuse_note(self, outcomes, n_voted, cl_spread, cd_spread,
                   enough, agree) -> str:
        voted = ", ".join(m.backend for m in outcomes if m.ok)
        holes = [m for m in outcomes if not m.ok]
        head = (f"Virtual Tunnel Solver: {self.reduction} consensus of {n_voted} "
                f"code(s) [{voted}]")
        spr = ""
        if cl_spread == cl_spread or cd_spread == cd_spread:
            spr = (f"; inter-code spread C_l {cl_spread:.1f}% / C_d {cd_spread:.1f}%")
        verdict = ""
        if not enough:
            verdict = (f"; NOT converged — only {n_voted} code(s) voted, "
                       f"need {self.min_members}")
        elif not agree:
            verdict = ("; NOT converged — codes disagree beyond "
                       f"{self.agreement_tol:.0f}% (treat as a flag, not a number)")
        else:
            verdict = "; converged consensus (codes agree within tolerance)"
        hole_txt = ""
        if holes:
            hole_txt = "; holes: " + ", ".join(
                f"{h.backend} ({(h.error or 'unconverged')[:48]}"
                + ("…" if len(h.error) > 48 else "") + ")" for h in holes)
        return head + spr + verdict + hole_txt

    # -- batch convenience over a whole matched run ----------------------- #
    def solve_matrix(self, specs: "Sequence[CaseSpec]", workdir: str,
                     run: bool = True) -> "list":
        """
        Drive + fuse a whole list of matched CaseSpecs (e.g. the output of
        `VirtualWindTunnel.case_specs()`), returning one EnsembleResult per point.
        The fused CoeffResults inside are exactly what `VirtualWindTunnel.correlate`
        consumes — so the consensus, not any single code, is what gets compared to
        the physical tunnel map.
        """
        out = []
        for s in specs:
            out.append(self.solve_detailed(s, workdir) if run
                       else self._fuse_from(s, workdir, run=False))
        return out


def fused_results(ensemble_results: "Sequence[EnsembleResult]") -> "list":
    """Pull the fused CoeffResults out of a list of EnsembleResults (for correlate)."""
    return [er.fused for er in ensemble_results]
