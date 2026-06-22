# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the Virtual Wind Tunnel — physical aero map vs CFD correlation.

These pin the behaviour that makes the feature a faithful calibration tool rather
than cosplay:

  1. ride-height <-> attitude conversion is EXACT and invertible (the lock that
     makes "same operating point" literally true, not approximately),
  2. the Virtual Wind Tunnel generates CFD cases at the EXACT physical points, so
     a self-consistent backend correlates to ~0 error,
  3. the honesty contract holds — a CFD point with no matching physical point is an
     unpaired HOLE, never snapped to a neighbour; a real coefficient offset is
     flagged NOT CALIBRATED with the correct over/under-prediction diagnostic,
  4. tunnel provenance surfaces the floor/blockage warnings that decide whether a
     reading is comparable to CFD at all,
  5. the TS-Auto and Star-CCM+ stubs write a faithful case and refuse to fabricate.

Run:  python -m pytest tests/test_windtunnel.py
"""
import math
import os
import tempfile

import pytest

from suspension.aero import (
    Attitude, CoeffResult, CFDProvenance, SolverFidelity, SolverUnavailable,
    ReferenceAeroModel, StarCCMSolver, TSAutoSolver, LocalSubmitter,
    RideHeights, AeroMapGrid, GroundState, TunnelProvenance, PhysicalAeroMap,
    VirtualWindTunnel, ride_heights_to_attitude, attitude_to_ride_heights,
    downforce_to_clift, drag_to_cdrag, AeroProvider,
)


# --------------------------------------------------------------------------- #
#  Ride-height geometry — the exactness lock
# --------------------------------------------------------------------------- #
def test_ride_heights_to_attitude_roundtrip_is_exact():
    rh = RideHeights(front_mm=18.0, rear_mm=46.0, speed_ms=27.0, wheelbase_mm=1550.0)
    att = ride_heights_to_attitude(rh)
    back = attitude_to_ride_heights(att, wheelbase_mm=1550.0)
    assert abs(back.front_mm - rh.front_mm) < 1e-9
    assert abs(back.rear_mm - rh.rear_mm) < 1e-9
    assert abs(back.speed_ms - rh.speed_ms) < 1e-9


def test_rake_sign_and_pitch():
    # rear higher than front => positive rake => nose-up pitch (positive)
    rh = RideHeights(front_mm=20.0, rear_mm=50.0, wheelbase_mm=1500.0)
    assert rh.rake_mm == 30.0
    assert rh.pitch_deg > 0
    att = ride_heights_to_attitude(rh)
    assert att.pitch_deg > 0
    assert att.ride_height_mm == 20.0
    # level platform => zero pitch
    level = ride_heights_to_attitude(RideHeights(30.0, 30.0))
    assert abs(level.pitch_deg) < 1e-9


def test_grid_point_count_and_attitudes():
    grid = AeroMapGrid(front_mm=[15, 20, 25], rear_mm=[40, 50], speed_ms=[25.0])
    assert len(grid) == 6
    assert len(grid.attitudes()) == 6
    # every attitude carries the right front clearance
    fronts = {a.ride_height_mm for a in grid.attitudes()}
    assert fronts == {15, 20, 25}


# --------------------------------------------------------------------------- #
#  Physical aero map — measured, with tunnel provenance, lap-sim-ready
# --------------------------------------------------------------------------- #
def _phys_map():
    prov = TunnelProvenance(facility="A2 Wind Shear", ground_state=GroundState.MOVING_BELT,
                            model_scale=1.0, blockage_corrected=True, reynolds=4.0e5)
    m = PhysicalAeroMap(prov, reference_area_m2=1.0, reference_length_m=1.55)
    # a small 2x2 ride-height map at one speed
    for f in (20.0, 30.0):
        for r in (40.0, 50.0):
            rh = RideHeights(f, r, speed_ms=25.0, wheelbase_mm=1550.0)
            # downforce grows as the car gets lower; pick sign-correct numbers
            cl = -2.9 + 0.01 * (f - 20.0) + 0.005 * (r - 40.0)
            cd = 1.02 + 0.001 * (f - 20.0)
            m.add_measurement(rh, c_lift=cl, c_drag=cd, aero_balance_front=0.44)
    return m


def test_physical_map_is_usable_by_lap_sim_provider():
    m = _phys_map()
    assert len(m) == 4
    # PhysicalAeroMap IS an AeroMap: AeroProvider consumes it unchanged
    prov = AeroProvider(reference_area_m2=1.0, aero_map=m)
    assert prov.is_mapped()
    att = ride_heights_to_attitude(RideHeights(20.0, 40.0, 25.0))
    cl_a, cd_a = prov.cla_cda(att)
    assert cl_a > 0 and cd_a > 0          # down-negative C_L -> positive cl_a


def test_tunnel_provenance_is_marked_correlated():
    m = _phys_map()
    # a measured tunnel map is physical reference data: correlated, not raw CFD
    assert m.provenance.is_correlated is True


def test_tunnel_provenance_warns_on_fixed_floor_and_blockage():
    bad = TunnelProvenance(facility="garage rig", ground_state=GroundState.FIXED_FLOOR,
                           blockage_corrected=False)
    s = bad.status()
    assert "fixed" in s.lower()
    assert "WARNING" in s
    assert "blockage" in s.lower()


def test_recover_ride_height_grid_from_map():
    m = _phys_map()
    grid = m.ride_height_grid()
    assert set(grid.front_mm) == {20.0, 30.0}
    assert set(grid.rear_mm) == {40.0, 50.0}
    assert set(grid.speed_ms) == {25.0}


# --------------------------------------------------------------------------- #
#  Virtual Wind Tunnel — same points, self-consistent backend correlates ~0
# --------------------------------------------------------------------------- #
def test_virtual_tunnel_generates_matching_points():
    m = _phys_map()
    vwt = VirtualWindTunnel(m, geometry_path="car.stl")
    specs = vwt.case_specs()
    assert len(specs) == 4
    # every CFD case attitude key must exist in the physical map (like-for-like)
    phys_keys = {r.attitude.key() for r in m.measured_points()}
    for s in specs:
        assert s.attitude.key() in phys_keys
        # reference normalisation matches the physical map, or coeffs aren't comparable
        assert s.reference_area_m2 == m.reference_area_m2
        assert s.reference_length_m == m.reference_length_m


def test_self_consistent_cfd_correlates_to_zero():
    # Build "CFD" results that are exactly the physical numbers, tagged k-omega SST.
    m = _phys_map()
    vwt = VirtualWindTunnel(m, "car.stl")
    cfd = []
    for phys in m.measured_points():
        cfd.append(CoeffResult(
            attitude=phys.attitude, c_lift=phys.c_lift, c_drag=phys.c_drag,
            aero_balance_front=phys.aero_balance_front, converged=True,
            provenance=CFDProvenance(backend="starccm", fidelity=SolverFidelity.RANS,
                                     turbulence_model="kOmegaSST"),
        ))
    rep = vwt.correlate(cfd)
    assert rep.ok
    assert rep.n_paired == 4
    assert rep.overall_within_tol
    assert abs(rep.cl_rms_pct) < 1e-6
    assert abs(rep.cd_rms_pct) < 1e-6
    assert "CALIBRATED" in rep.summary
    assert rep.turbulence_model == "kOmegaSST"


def test_offset_cfd_is_flagged_not_calibrated_with_direction():
    m = _phys_map()
    vwt = VirtualWindTunnel(m, "car.stl")
    cfd = []
    for phys in m.measured_points():
        # CFD over-predicts downforce magnitude by 12% (c_lift more negative)
        cfd.append(CoeffResult(
            attitude=phys.attitude, c_lift=phys.c_lift * 1.12, c_drag=phys.c_drag,
            aero_balance_front=phys.aero_balance_front, converged=True,
            provenance=CFDProvenance(backend="starccm", fidelity=SolverFidelity.RANS,
                                     turbulence_model="kOmegaSST"),
        ))
    rep = vwt.correlate(cfd)
    assert rep.n_paired == 4
    assert not rep.overall_within_tol
    assert "NOT CALIBRATED" in rep.summary
    # c_lift is negative; *1.12 makes it more negative. cl_err = (cfd-phys)/phys,
    # and phys<0, so a more-negative cfd gives a POSITIVE percent error => CFD
    # over-predicts downforce magnitude.
    assert rep.cl_bias_pct > 0
    assert "OVER-predicts" in rep.summary


def test_unmatched_cfd_point_is_an_honest_hole_not_snapped():
    m = _phys_map()
    vwt = VirtualWindTunnel(m, "car.stl")
    # one matching point + one point at an attitude the tunnel never measured
    good = m.measured_points()[0]
    stray_att = ride_heights_to_attitude(RideHeights(99.0, 99.0, 25.0))
    cfd = [
        CoeffResult(attitude=good.attitude, c_lift=good.c_lift, c_drag=good.c_drag,
                    converged=True,
                    provenance=CFDProvenance(backend="tsauto",
                                             fidelity=SolverFidelity.RANS,
                                             turbulence_model="kOmegaSST")),
        CoeffResult(attitude=stray_att, c_lift=-3.0, c_drag=1.1, converged=True,
                    provenance=CFDProvenance(backend="tsauto",
                                             fidelity=SolverFidelity.RANS,
                                             turbulence_model="kOmegaSST")),
    ]
    rep = vwt.correlate(cfd)
    # exactly one pairs; the stray is reported unpaired, not matched to a neighbour
    assert rep.n_paired == 1
    unpaired = [p for p in rep.points if not p.paired]
    assert any("not paired" in p.note for p in unpaired)
    # the 3 physical points the CFD didn't cover are also reported as holes
    assert any("not covered" in p.note for p in unpaired)


def test_empty_pairing_refuses_to_claim_calibration():
    m = _phys_map()
    vwt = VirtualWindTunnel(m, "car.stl")
    # CFD entirely at non-matching attitudes
    cfd = [CoeffResult(attitude=ride_heights_to_attitude(RideHeights(99, 99, 25)),
                       c_lift=-3.0, c_drag=1.1, converged=True,
                       provenance=CFDProvenance(backend="starccm",
                                                fidelity=SolverFidelity.RANS,
                                                turbulence_model="kOmegaSST"))]
    rep = vwt.correlate(cfd)
    assert rep.n_paired == 0
    assert not rep.overall_within_tol
    assert "like-for-like" in rep.summary or "Nothing could be compared" in rep.summary


# --------------------------------------------------------------------------- #
#  Coefficient helpers
# --------------------------------------------------------------------------- #
def test_downforce_and_drag_to_coeff_signs():
    # 800 N downforce, 1 m^2, 25 m/s, rho 1.225 => negative c_lift
    cl = downforce_to_clift(800.0, 1.225, 1.0, 25.0)
    assert cl < 0
    cd = drag_to_cdrag(300.0, 1.225, 1.0, 25.0)
    assert cd > 0


# --------------------------------------------------------------------------- #
#  Backend stubs — faithful write, honest refusal
# --------------------------------------------------------------------------- #
def test_tsauto_writes_config_and_refuses_to_run():
    from suspension.aero import CaseSpec
    b = TSAutoSolver(turbulence_model="kOmegaSST")
    assert b.provenance().turbulence_model == "kOmegaSST"
    assert b.provenance().is_correlated is False
    spec = CaseSpec(Attitude(ride_height_mm=20.0, pitch_deg=1.0, speed_ms=25.0),
                    "car.stl", reference_area_m2=1.0, reference_length_m=1.55)
    with tempfile.TemporaryDirectory() as d:
        path = b.write_case(spec, d)
        assert os.path.isfile(path)
        txt = open(path).read()
        assert "kOmegaSST" in txt and "referenceArea_m2" in txt
        with pytest.raises(SolverUnavailable):
            b.run_case(spec, d)
        # read_result before any run also refuses, never fabricates
        with pytest.raises(SolverUnavailable):
            b.read_result(spec, d)


def test_get_backend_resolves_tsauto_aliases():
    from suspension.aero import get_backend
    for name in ("tsauto", "ts", "totalsim", "TS-Auto"):
        b = get_backend(name)
        assert isinstance(b, TSAutoSolver)


def test_end_to_end_reference_backend_through_virtual_tunnel():
    # The reference analytic backend can stand in for "CFD" to exercise the whole
    # pipeline with no license: generate matching cases, run them, correlate.
    m = _phys_map()
    vwt = VirtualWindTunnel(m, "car.stl")
    backend = ReferenceAeroModel()
    specs = vwt.case_specs()
    with tempfile.TemporaryDirectory() as d:
        results = [backend.run_case(s, d) for s in specs]
    rep = vwt.correlate(results)
    # the surrogate isn't the tunnel, so this need not be "calibrated" — but it must
    # PAIR all four points like-for-like and produce finite errors, never crash.
    assert rep.n_paired == 4
    assert math.isfinite(rep.cl_rms_pct)
    assert math.isfinite(rep.cd_rms_pct)
