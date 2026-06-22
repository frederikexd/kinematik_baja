# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the Virtual Tunnel Solver — the meta-solver built ON STAR-CCM+, TS-Auto
and OpenFOAM at once (suspension/aero/ensemble.py).

These pin the behaviour that makes it a faithful cross-code consensus rather than a
solver chooser dressed up:

  1. it IS a CFDSolver (write/run/read + provenance) and drops into the registry,
  2. write_case lays down EVERY member code's input, one sub-folder per code,
  3. converged members fuse to the mean/median consensus, sign convention preserved,
  4. the inter-code spread sets the converged verdict — agreeing codes pass, codes
     that disagree beyond tolerance are flagged NOT converged (a flag, not a number),
  5. the honesty contract holds — a code that cannot run is an honest HOLE that
     contributes NOTHING; with no usable code the fused coefficient is None, never
     fabricated and never an exception,
  6. min_members gates a lone code (a "consensus" of one is just that one code),
  7. it plugs into VirtualWindTunnel so the CONSENSUS, not any single code, is what
     gets correlated against the physical tunnel map.

Run:  python -m pytest tests/test_ensemble.py
"""
import math
import os
import tempfile

import pytest

from suspension.aero import (
    Attitude, CaseSpec, CoeffResult, CFDProvenance, SolverFidelity, CFDSolver,
    SolverUnavailable, get_backend,
    EnsembleTunnelSolver, EnsembleResult, MemberOutcome, fused_results,
    DEFAULT_MEMBER_NAMES,
)
from suspension.aero import windtunnel as wt


# --------------------------------------------------------------------------- #
#  Helpers — fakes so we can drive the fusion engine deterministically without
#  any external solver. Each fake is a faithful CFDSolver.
# --------------------------------------------------------------------------- #
class _FakeCode:
    """A member that returns a fixed (Cl, Cd) — our-convention c_lift negative."""
    def __init__(self, name, c_lift, c_drag, converged=True, bal=None):
        self.name = name
        self._cl = c_lift
        self._cd = c_drag
        self._conv = converged
        self._bal = bal

    def provenance(self):
        return CFDProvenance(backend=self.name, fidelity=SolverFidelity.RANS)

    def write_case(self, spec, workdir):
        os.makedirs(workdir, exist_ok=True)
        p = os.path.join(workdir, f"{self.name}.txt")
        with open(p, "w") as f:
            f.write("fake\n")
        return p

    def run_case(self, spec, workdir):
        return CoeffResult(attitude=spec.attitude, c_lift=self._cl, c_drag=self._cd,
                           aero_balance_front=self._bal, converged=self._conv,
                           provenance=self.provenance())

    def read_result(self, spec, workdir):
        return self.run_case(spec, workdir)


class _DeadCode:
    """A member that cannot run here — raises SolverUnavailable like the real stubs."""
    def __init__(self, name="deadcode"):
        self.name = name

    def provenance(self):
        return CFDProvenance(backend=self.name, fidelity=SolverFidelity.RANS)

    def write_case(self, spec, workdir):
        os.makedirs(workdir, exist_ok=True)
        return workdir

    def run_case(self, spec, workdir):
        raise SolverUnavailable(f"{self.name} has no license here")

    def read_result(self, spec, workdir):
        raise SolverUnavailable(f"{self.name} produced no result")


def _spec(pitch=0.0, h=20.0, v=27.0):
    return CaseSpec(attitude=Attitude(pitch_deg=pitch, ride_height_mm=h, speed_ms=v),
                    geometry_path="car.stl", reference_area_m2=1.0,
                    reference_length_m=1.55)


# --------------------------------------------------------------------------- #
#  1. It IS a solver, and the registry builds it
# --------------------------------------------------------------------------- #
def test_is_a_cfdsolver_and_in_registry():
    vts = get_backend("virtual-tunnel")
    assert isinstance(vts, EnsembleTunnelSolver)
    assert isinstance(vts, CFDSolver)            # runtime_checkable Protocol
    assert vts.name == "virtual-tunnel"
    # self-contained on the single in-house Fluent backend
    assert vts._member_names == list(DEFAULT_MEMBER_NAMES)
    assert DEFAULT_MEMBER_NAMES == ("fluent",)


def test_registry_aliases():
    for alias in ("virtual-tunnel", "ensemble", "vts", "consensus"):
        assert isinstance(get_backend(alias), EnsembleTunnelSolver)


def test_provenance_names_the_in_house_fluent_backend():
    prov = get_backend("virtual-tunnel").provenance()
    # the default roster is the single in-house Fluent code
    assert "fluent" in prov.backend
    # and it no longer drags in the old external three-code roster
    assert "starccm" not in prov.backend
    assert "tsauto" not in prov.backend
    assert "openfoam" not in prov.backend


# --------------------------------------------------------------------------- #
#  2. write_case writes every member's input (default: the Fluent verification deck)
# --------------------------------------------------------------------------- #
def test_write_case_writes_each_member_input():
    vts = get_backend("virtual-tunnel")
    wd = tempfile.mkdtemp(prefix="vts_w_")
    case_dir = vts.write_case(_spec(), wd)
    for code in DEFAULT_MEMBER_NAMES:
        sub = os.path.join(case_dir, code)
        assert os.path.isdir(sub), f"{code} sub-folder missing"
        assert os.listdir(sub), f"{code} wrote no input"
    # the default member is Fluent and it writes a .jou verification deck
    fluent_sub = os.path.join(case_dir, "fluent")
    assert any(f.endswith(".jou") for f in os.listdir(fluent_sub))


# --------------------------------------------------------------------------- #
#  3. Fusion of converged members + sign convention
# --------------------------------------------------------------------------- #
def test_mean_consensus_of_agreeing_codes():
    vts = EnsembleTunnelSolver(
        members=[_FakeCode("a", -2.90, 1.04), _FakeCode("b", -2.94, 1.06)],
        reduction="mean", agreement_tol=5.0, min_members=2)
    wd = tempfile.mkdtemp(prefix="vts_mean_")
    det = vts.solve_detailed(_spec(), wd)
    assert det.n_voted == 2
    assert det.fused.c_lift == pytest.approx(-2.92)      # mean
    assert det.fused.c_drag == pytest.approx(1.05)
    assert det.fused.c_lift < 0                          # downforce sign preserved
    assert det.fused.converged is True


def test_median_is_robust_to_one_outlier():
    vts = EnsembleTunnelSolver(
        members=[_FakeCode("a", -2.90, 1.04),
                 _FakeCode("b", -2.92, 1.05),
                 _FakeCode("c", -3.60, 1.30)],          # outlier
        reduction="median", agreement_tol=100.0, min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.fused.c_lift == pytest.approx(-2.92)      # median ignores the outlier


# --------------------------------------------------------------------------- #
#  4. Inter-code spread sets the converged verdict
# --------------------------------------------------------------------------- #
def test_agreeing_codes_converge():
    vts = EnsembleTunnelSolver(
        members=[_FakeCode("a", -2.90, 1.04), _FakeCode("b", -2.95, 1.05)],
        agreement_tol=5.0, min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.fused.converged is True
    assert det.cl_spread_pct < 5.0


def test_disagreeing_codes_do_not_converge():
    # -2.6 vs -3.4 => ~27% peak-to-peak about the mean: way over tolerance
    vts = EnsembleTunnelSolver(
        members=[_FakeCode("a", -2.60, 1.00), _FakeCode("b", -3.40, 1.30)],
        agreement_tol=5.0, min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.fused.converged is False
    assert det.cl_spread_pct > 5.0
    # the fused number still exists (it's the consensus) but is flagged
    assert det.fused.c_lift is not None
    assert "disagree" in det.fused.notes.lower()


# --------------------------------------------------------------------------- #
#  5. Honesty contract — holes contribute nothing, nothing is fabricated
# --------------------------------------------------------------------------- #
def test_dead_member_is_a_hole_not_a_vote():
    vts = EnsembleTunnelSolver(
        members=[_FakeCode("a", -2.90, 1.04), _DeadCode("b"),
                 _FakeCode("c", -2.92, 1.05)],
        agreement_tol=5.0, min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.n_voted == 2                              # the dead code did not vote
    # consensus is the mean of the two LIVE codes only
    assert det.fused.c_lift == pytest.approx(-2.91)
    holes = [m for m in det.members if not m.ok]
    assert len(holes) == 1 and holes[0].backend == "b"
    assert "license" in holes[0].error.lower()


def test_no_usable_code_yields_an_honest_hole_not_an_exception():
    vts = EnsembleTunnelSolver(members=[_DeadCode("a"), _DeadCode("b")],
                               min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())   # must NOT raise
    assert det.n_voted == 0
    assert det.fused.c_lift is None                      # NOT zero, NOT a guess
    assert det.fused.c_drag is None
    assert det.fused.converged is False
    assert det.fused.is_usable() is False


def test_real_default_member_answers_in_house_no_external_solver():
    # The whole point of the new design: with NOTHING installed (no license, no mesh,
    # no external solver), the default Virtual Tunnel Solver still returns a usable,
    # honestly-labelled in-house number and writes a Fluent deck to verify it.
    vts = get_backend("virtual-tunnel")
    wd = tempfile.mkdtemp()
    det = vts.solve_detailed(_spec(), wd)
    assert det.n_voted == 1                       # the in-house code voted
    assert det.fused.c_lift is not None           # a real number, computed internally
    assert det.fused.c_lift < 0                   # downforce sign convention
    assert det.fused.converged is True
    assert det.fused.is_usable() is True
    # the provenance is honest about what this is
    assert det.fused.provenance.is_correlated is False
    assert "in-house" in det.fused.provenance.notes.lower()
    # and the ANSYS Fluent verification journal is on disk for confirmation
    case_dir = os.path.join(wd, _spec().case_name(), "fluent")
    assert any(f.endswith(".jou") for f in os.listdir(case_dir))


# --------------------------------------------------------------------------- #
#  6. min_members gates a lone code
# --------------------------------------------------------------------------- #
def test_lone_code_not_converged_by_default():
    vts = EnsembleTunnelSolver(members=[_FakeCode("a", -2.9, 1.04)],
                               min_members=2)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.n_voted == 1
    assert det.fused.c_lift == pytest.approx(-2.9)       # the number exists
    assert det.fused.converged is False                  # but a lone vote isn't consensus


def test_lone_code_allowed_when_min_members_one():
    vts = EnsembleTunnelSolver(members=[_FakeCode("a", -2.9, 1.04)],
                               min_members=1)
    det = vts.solve_detailed(_spec(), tempfile.mkdtemp())
    assert det.fused.converged is True


# --------------------------------------------------------------------------- #
#  7. Plugs into VirtualWindTunnel — the consensus is what gets correlated
# --------------------------------------------------------------------------- #
def _physical_map():
    prov = wt.TunnelProvenance(
        facility="A2", ground_state=wt.GroundState.MOVING_BELT,
        blockage_corrected=True, reference_area_m2=1.0, reference_length_m=1.55)
    pm = wt.PhysicalAeroMap(prov, reference_area_m2=1.0, reference_length_m=1.55,
                            wheelbase_mm=1550.0)
    for front in (18.0, 25.0):
        for rear in (40.0, 55.0):
            rh = wt.RideHeights(front, rear, speed_ms=27.0, wheelbase_mm=1550.0)
            pm.add_measurement(rh, c_lift=-2.95 + 0.012 * (front - 18.0),
                               c_drag=1.04, aero_balance_front=0.43)
    return pm


def test_consensus_correlates_against_tunnel():
    pm = _physical_map()
    vwt = wt.VirtualWindTunnel(pm, geometry_path="car.stl", rho=1.225)
    specs = vwt.case_specs()

    # Build an ensemble of two fakes that, per point, sit ~1% off the tunnel.
    class _Near:
        def __init__(self, name, scale):
            self.name = name
            self._scale = scale
        def provenance(self):
            return CFDProvenance(backend=self.name, fidelity=SolverFidelity.RANS)
        def write_case(self, spec, workdir):
            os.makedirs(workdir, exist_ok=True); return workdir
        def run_case(self, spec, workdir):
            phys = next(p for p in pm.measured_points()
                        if p.attitude.key() == spec.attitude.key())
            return CoeffResult(attitude=spec.attitude,
                               c_lift=phys.c_lift * self._scale,
                               c_drag=phys.c_drag * self._scale,
                               converged=True, provenance=self.provenance())
        def read_result(self, spec, workdir):
            return self.run_case(spec, workdir)

    vts = EnsembleTunnelSolver(members=[_Near("a", 1.00), _Near("b", 1.02)],
                               agreement_tol=5.0, min_members=2)
    wd = tempfile.mkdtemp(prefix="vts_corr_")
    ens = vts.solve_matrix(specs, wd, run=True)
    rep = vwt.correlate(fused_results(ens))

    assert rep.n_paired == 4
    assert rep.overall_within_tol is True
    assert rep.cl_rms_pct < 4.0
    assert "virtual-tunnel" in rep.backend


def test_fused_results_helper_extracts_coeffresults():
    er = EnsembleResult(
        fused=CoeffResult(attitude=Attitude(), c_lift=-2.9, c_drag=1.0,
                          converged=True),
        members=[], n_voted=2)
    out = fused_results([er, er])
    assert len(out) == 2
    assert all(isinstance(r, CoeffResult) for r in out)
    assert out[0].c_lift == pytest.approx(-2.9)


def test_bad_reduction_rejected():
    with pytest.raises(ValueError):
        EnsembleTunnelSolver(reduction="geometric-mean")
