# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the aero CFD co-simulation boundary.

Covers the three things that make this a faithful seam rather than cosplay:
  1. the orchestration/map/lap-sim pipeline runs end-to-end on the reference backend,
  2. the honesty contract holds — unavailable solvers raise, never fabricate; the
     map refuses to invent a channel it never received,
  3. the real OpenFOAM adapter writes a valid case and parses a coeff file.
"""

import math
import os
import tempfile

import pytest

from suspension.aero import (
    Attitude, RunMatrix, CaseSpec, CoeffResult, SolverFidelity, SolverUnavailable,
    ReferenceAeroModel, FluentVerificationSolver, OpenFOAMSolver, StarCCMSolver,
    FluentSolver, LocalSubmitter, SlurmSSHSubmitter,
    AeroMap, AeroOrchestrator, AeroProvider, estimate_attitude, get_backend,
)


# --------------------------------------------------------------------------- #
#  RunMatrix
# --------------------------------------------------------------------------- #
def test_runmatrix_cartesian_product_and_len():
    m = RunMatrix(roll_deg=[0, 1], yaw_deg=[0, 2, 4])
    assert len(m) == 6
    assert len(m.attitudes()) == 6
    assert set(m.axes_swept()) == {"roll", "yaw"}


def test_runmatrix_cost_summary_mentions_case_count():
    m = RunMatrix(yaw_deg=[0, 2, 4, 6, 8])
    s = m.cost_summary(minutes_per_case=180, concurrent=1)
    assert "5 case" in s and "h wall-clock" in s


# --------------------------------------------------------------------------- #
#  Reference backend + orchestrator end-to-end
# --------------------------------------------------------------------------- #
def test_reference_backend_runs_and_is_honest_about_fidelity():
    b = ReferenceAeroModel()
    prov = b.provenance()
    assert prov.fidelity is SolverFidelity.POTENTIAL
    assert prov.is_correlated is False
    assert "NOT" in prov.notes or "not" in prov.notes


def test_orchestrator_builds_map_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        orch = AeroOrchestrator(ReferenceAeroModel(), "car.stl",
                                reference_area_m2=1.0)
        matrix = RunMatrix(yaw_deg=[0, 2, 4, 6])
        # cost preview available before any work
        assert "case" in orch.plan(matrix)
        report = orch.run(matrix, workdir=d)
        assert report.n_total == 4
        assert report.n_usable == 4
        assert len(report.aero_map) == 4


def test_downforce_decreases_with_yaw_in_reference_model():
    b = ReferenceAeroModel()
    spec0 = CaseSpec(Attitude(yaw_deg=0), "car.stl")
    spec8 = CaseSpec(Attitude(yaw_deg=8), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        r0 = b.run_case(spec0, d)
        r8 = b.run_case(spec8, d)
    # convention: c_lift negative = downforce; yaw should make it LESS negative
    assert r8.c_lift > r0.c_lift
    # and drag should rise
    assert r8.c_drag > r0.c_drag


# --------------------------------------------------------------------------- #
#  AeroMap interpolation + honesty
# --------------------------------------------------------------------------- #
def test_map_interpolates_between_solved_points():
    b = ReferenceAeroModel()
    with tempfile.TemporaryDirectory() as d:
        results = [b.run_case(CaseSpec(Attitude(yaw_deg=y), "c.stl"), d)
                   for y in (0, 4)]
    amap = AeroMap.from_results(results)
    q0 = amap.query(Attitude(yaw_deg=0))
    q4 = amap.query(Attitude(yaw_deg=4))
    q2 = amap.query(Attitude(yaw_deg=2))
    # midpoint lift between the two endpoints
    assert min(q0.c_lift, q4.c_lift) <= q2.c_lift <= max(q0.c_lift, q4.c_lift)


def test_map_clamps_outside_envelope_and_flags_it():
    b = ReferenceAeroModel()
    with tempfile.TemporaryDirectory() as d:
        results = [b.run_case(CaseSpec(Attitude(yaw_deg=y), "c.stl"), d)
                   for y in (0, 4)]
    amap = AeroMap.from_results(results)
    q = amap.query(Attitude(yaw_deg=20))     # well outside [0,4]
    assert q.extrapolated is True
    # clamped to the yaw=4 value, not extrapolated linearly past it
    q4 = amap.query(Attitude(yaw_deg=4))
    assert q.c_lift == pytest.approx(q4.c_lift, rel=1e-6)


def test_map_refuses_to_invent_missing_channel():
    # build results with c_side = None everywhere
    res = [CoeffResult(Attitude(yaw_deg=y), c_lift=-2.5, c_drag=1.1,
                       c_side=None, converged=True) for y in (0, 4)]
    amap = AeroMap.from_results(res)
    q = amap.query(Attitude(yaw_deg=2))
    assert q.c_lift is not None
    assert q.c_side is None             # never fabricated to zero


def test_map_drops_unconverged_results():
    good = CoeffResult(Attitude(yaw_deg=0), c_lift=-2.5, c_drag=1.1, converged=True)
    bad = CoeffResult(Attitude(yaw_deg=4), c_lift=-9.9, c_drag=9.9, converged=False)
    amap = AeroMap.from_results([good, bad])
    assert len(amap) == 1


def test_map_csv_roundtrip():
    res = [CoeffResult(Attitude(yaw_deg=y), c_lift=-2.5 + 0.01 * y, c_drag=1.1,
                       c_side=0.02 * y, aero_balance_front=0.45, converged=True)
           for y in (0, 2, 4)]
    amap = AeroMap.from_results(res)
    text = amap.to_csv()
    amap2 = AeroMap.from_csv(text)
    assert len(amap2) == 3
    q = amap2.query(Attitude(yaw_deg=2))
    assert q.c_lift == pytest.approx(-2.5 + 0.01 * 2, rel=1e-6)


# --------------------------------------------------------------------------- #
#  Honesty contract for unavailable commercial solvers
# --------------------------------------------------------------------------- #
def test_starccm_writes_macro_but_refuses_to_run():
    b = StarCCMSolver()
    with tempfile.TemporaryDirectory() as d:
        path = b.write_case(CaseSpec(Attitude(yaw_deg=3), "car.stl"), d)
        assert os.path.isfile(path) and path.endswith(".java")
        with pytest.raises(SolverUnavailable):
            b.run_case(CaseSpec(Attitude(yaw_deg=3), "car.stl"), d)


def test_fluent_writes_journal_but_refuses_to_run():
    b = FluentSolver()
    with tempfile.TemporaryDirectory() as d:
        path = b.write_case(CaseSpec(Attitude(yaw_deg=3), "car.stl"), d)
        assert os.path.isfile(path) and path.endswith(".jou")
        with pytest.raises(SolverUnavailable):
            b.run_case(CaseSpec(Attitude(yaw_deg=3), "car.stl"), d)


# --------------------------------------------------------------------------- #
#  In-house Fluent verification backend — answers without any external solver
# --------------------------------------------------------------------------- #
def test_fluent_verification_answers_in_house_and_writes_deck():
    b = FluentVerificationSolver()
    spec = CaseSpec(Attitude(yaw_deg=3, ride_height_mm=20.0, speed_ms=27.0), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        # run_case must NOT raise — KinematiK answers on its own
        res = b.run_case(spec, d)
        assert res.c_lift is not None and res.c_lift < 0     # downforce, computed in-house
        assert res.c_drag is not None and res.c_drag > 0
        assert res.converged is True
        assert res.provenance.is_correlated is False         # honest: an estimate
        # the ANSYS Fluent verification journal is written alongside, for the user
        jou = os.path.join(d, spec.case_name() + ".jou")
        assert os.path.isfile(jou)
        text = open(jou).read()
        assert "ANSYS Fluent" in text
        assert "in-house estimate" in text.lower()


def test_fluent_verification_matches_reference_model_physics():
    # The in-house number is exactly the analytic attitude model's number.
    b = FluentVerificationSolver()
    ref = ReferenceAeroModel()
    spec = CaseSpec(Attitude(yaw_deg=4, ride_height_mm=22.0), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        got = b.run_case(spec, d)
    want = ref.run_case(spec, tempfile.mkdtemp())
    assert got.c_lift == pytest.approx(want.c_lift)
    assert got.c_drag == pytest.approx(want.c_drag)


def test_fluent_verification_reads_back_user_run_csv():
    # If the user runs the deck and stages a coeff CSV, read_result returns THAT.
    b = FluentVerificationSolver()
    spec = CaseSpec(Attitude(yaw_deg=2), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        b.write_case(spec, d)
        csv_path = os.path.join(d, spec.case_name() + "_coeffs.csv")
        with open(csv_path, "w") as f:
            f.write("Cl,Cd,Cs,CmPitch,converged\n2.7,1.06,0.05,0.1,1\n")
        res = b.read_result(spec, d)
    assert res.c_lift == pytest.approx(-2.7)     # vendor up-positive -> down-negative
    assert res.c_drag == pytest.approx(1.06)
    assert "fluent-verified" in res.provenance.backend


def test_fluent_verification_read_fluent_csv_requires_the_csv():
    b = FluentVerificationSolver()
    spec = CaseSpec(Attitude(), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SolverUnavailable):
            b.read_fluent_csv(spec, d)           # optional check, but must be explicit


def test_commercial_stub_reads_exported_csv():
    b = StarCCMSolver()
    spec = CaseSpec(Attitude(yaw_deg=3), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        b.write_case(spec, d)
        csv_path = os.path.join(d, spec.case_name() + "_coeffs.csv")
        with open(csv_path, "w") as f:
            f.write("Cl,Cd,Cs,CmPitch,converged\n2.6,1.05,0.04,0.1,1\n")
        res = b.read_result(spec, d)
    assert res.converged is True
    assert res.c_lift == pytest.approx(-2.6)     # vendor up-positive -> down-negative
    assert res.c_drag == pytest.approx(1.05)


# --------------------------------------------------------------------------- #
#  OpenFOAM real adapter
# --------------------------------------------------------------------------- #
def test_openfoam_writes_valid_case_skeleton():
    b = OpenFOAMSolver()
    spec = CaseSpec(Attitude(yaw_deg=5, pitch_deg=1), "car.stl",
                    reference_area_m2=1.0, reference_length_m=1.55)
    with tempfile.TemporaryDirectory() as d:
        case = b.write_case(spec, d)
        for rel in ("system/controlDict", "system/fvSchemes", "system/fvSolution",
                    "0/U", "constant/momentumTransport", "kinematik_attitude.json"):
            assert os.path.isfile(os.path.join(case, rel)), rel
        # forceCoeffs function object present
        with open(os.path.join(case, "system/controlDict")) as f:
            assert "forceCoeffs" in f.read()


def test_openfoam_inlet_velocity_rotates_with_yaw():
    ux0, uy0, _ = OpenFOAMSolver._inlet_velocity(Attitude(yaw_deg=0, speed_ms=20))
    ux5, uy5, _ = OpenFOAMSolver._inlet_velocity(Attitude(yaw_deg=10, speed_ms=20))
    assert uy0 == pytest.approx(0.0, abs=1e-9)
    assert uy5 < 0                       # yaw to the right => negative y inlet component
    assert ux5 < ux0                     # streamwise component reduced


def test_openfoam_run_without_binary_raises_not_fakes():
    b = OpenFOAMSolver(application="simpleFoam_definitely_not_installed")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SolverUnavailable):
            b.run_case(CaseSpec(Attitude(), "car.stl"), d)


def test_openfoam_parses_forcecoeffs_file():
    b = OpenFOAMSolver()
    spec = CaseSpec(Attitude(yaw_deg=0), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        case = b.write_case(spec, d)
        pp = os.path.join(case, "postProcessing", "forceCoeffs", "0")
        os.makedirs(pp, exist_ok=True)
        with open(os.path.join(pp, "coefficient.dat"), "w") as f:
            f.write("# Time Cd Cs Cl CmRoll CmPitch CmYaw\n")
            for it in range(100):
                cl = 2.60 + 0.5 * math.exp(-it / 10.0)   # settles to 2.60
                f.write(f"{it} 1.05 0.0 {cl:.5f} 0 0.1 0\n")
        res = b.read_result(spec, d)
    assert res.converged is True
    assert res.c_lift == pytest.approx(-2.60, abs=2e-2)   # up-positive -> down-negative
    assert res.c_drag == pytest.approx(1.05, abs=1e-3)


# --------------------------------------------------------------------------- #
#  Submitter honesty
# --------------------------------------------------------------------------- #
def test_local_submitter_captures_failures_without_aborting():
    # a backend that always raises on run -> all holes, sweep still completes
    sub = LocalSubmitter()
    b = StarCCMSolver()
    specs = [CaseSpec(Attitude(yaw_deg=y), "car.stl") for y in (0, 2)]
    with tempfile.TemporaryDirectory() as d:
        results = sub.submit_all(b, specs, d)
    assert len(results) == 2
    assert all(not r.ok and r.error for r in results)


def test_slurm_submitter_writes_sbatch_and_refuses_without_target():
    sub = SlurmSSHSubmitter()    # no ssh_target
    b = OpenFOAMSolver()
    specs = [CaseSpec(Attitude(yaw_deg=y), "car.stl") for y in (0, 2)]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SolverUnavailable):
            sub.submit_all(b, specs, d)
        assert os.path.isfile(os.path.join(d, "run_array.sbatch"))
        assert os.path.isfile(os.path.join(d, "cases.txt"))


# --------------------------------------------------------------------------- #
#  Lap-sim coupling (the "B" payoff)
# --------------------------------------------------------------------------- #
def test_provider_falls_back_to_scalar_without_map():
    p = AeroProvider(reference_area_m2=1.0, fallback_cl_a=2.5, fallback_cd_a=1.2)
    cl_a, cd_a = p.cla_cda(Attitude(yaw_deg=5))
    assert (cl_a, cd_a) == (2.5, 1.2)
    assert p.is_mapped() is False


def test_provider_uses_map_when_present():
    b = ReferenceAeroModel()
    with tempfile.TemporaryDirectory() as d:
        results = [b.run_case(CaseSpec(Attitude(yaw_deg=y), "c.stl", reference_area_m2=1.0), d)
                   for y in (0, 8)]
    amap = AeroMap.from_results(results, reference_area_m2=1.0)
    p = AeroProvider(reference_area_m2=1.0, aero_map=amap)
    cl_straight, cd_straight = p.cla_cda(Attitude(yaw_deg=0))
    cl_yaw, cd_yaw = p.cla_cda(Attitude(yaw_deg=8))
    # downforce (cl_a, positive) should DROP with yaw; drag should rise
    assert cl_yaw < cl_straight
    assert cd_yaw > cd_straight
    assert p.is_mapped() is True


def test_estimate_attitude_signs():
    att = estimate_attitude(speed_ms=20, lat_g=1.0, long_g=-1.0)
    assert att.roll_deg > 0          # lateral g rolls the car
    assert att.pitch_deg < 0         # braking dives the nose


def test_get_backend_aliases():
    assert isinstance(get_backend("openfoam"), OpenFOAMSolver)
    assert isinstance(get_backend("star"), StarCCMSolver)
    # "ansys" / "fluent" now resolve to the self-contained in-house verification
    # solver (computes the coefficient internally, writes a Fluent deck to verify).
    assert isinstance(get_backend("ansys"), FluentVerificationSolver)
    assert isinstance(get_backend("fluent"), FluentVerificationSolver)
    # the pure external pass-through stub is still reachable explicitly
    assert isinstance(get_backend("fluent-stub"), FluentSolver)
    assert isinstance(get_backend("reference"), ReferenceAeroModel)


# --------------------------------------------------------------------------- #
#  Real geometry -> attitude link (attitude_from_dynamics)
# --------------------------------------------------------------------------- #
def _veh_with_springs():
    from suspension import (SuspensionKinematics, Hardpoints, VehicleDynamics,
                            VehicleParams, default_tire)
    kin = SuspensionKinematics(Hardpoints.default())
    p = VehicleParams()
    # turn on the spring-rate path so ride rates (and thus pitch/heave) are derivable
    p.use_spring_rates = True
    return VehicleDynamics(p, front_kin=kin, rear_kin=kin, tire=default_tire())


def test_attitude_from_dynamics_reads_real_roll():
    from suspension.aero import attitude_from_dynamics
    veh = _veh_with_springs()
    att, info = attitude_from_dynamics(veh, lat_g=1.2, long_g=0.0, speed_ms=20)
    # roll must come from the dynamics model, not a flat gradient
    assert info["roll_source"].startswith("dynamics")
    assert att.roll_deg > 0           # cornering rolls the car
    # and it should scale with lateral g
    att2, _ = attitude_from_dynamics(veh, lat_g=0.6, long_g=0.0, speed_ms=20)
    assert att.roll_deg > att2.roll_deg


def test_attitude_roll_sign_follows_turn_direction():
    from suspension.aero import attitude_from_dynamics
    veh = _veh_with_springs()
    left, _ = attitude_from_dynamics(veh, lat_g=1.0, long_g=0.0, speed_ms=20)
    right, _ = attitude_from_dynamics(veh, lat_g=-1.0, long_g=0.0, speed_ms=20)
    assert left.roll_deg > 0 and right.roll_deg < 0
    assert left.roll_deg == pytest.approx(-right.roll_deg, rel=1e-6)


def test_attitude_braking_pitches_nose_down():
    from suspension.aero import attitude_from_dynamics
    veh = _veh_with_springs()
    braking, info = attitude_from_dynamics(veh, lat_g=0.0, long_g=1.2, speed_ms=20)
    assert info["pitch_source"].startswith("dynamics")
    assert braking.pitch_deg < 0      # nose down under braking
    accel, _ = attitude_from_dynamics(veh, lat_g=0.0, long_g=-1.2, speed_ms=20)
    assert accel.pitch_deg > 0        # nose up under acceleration


def test_attitude_pitch_left_neutral_without_ride_rate():
    # default VehicleParams has use_spring_rates False -> no ride rate -> honest neutral
    from suspension import (SuspensionKinematics, Hardpoints, VehicleDynamics,
                            VehicleParams, default_tire)
    from suspension.aero import attitude_from_dynamics
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin, tire=default_tire())
    att, info = attitude_from_dynamics(veh, lat_g=0.0, long_g=1.0, speed_ms=20)
    assert att.pitch_deg == 0.0
    assert info["pitch_source"] == "neutral"
    assert "pitch_note" in info       # explains WHY, doesn't silently fake it


def test_attitude_ride_height_drops_with_aero_downforce():
    from suspension.aero import attitude_from_dynamics, AeroProvider
    veh = _veh_with_springs()
    prov = AeroProvider(reference_area_m2=1.0, fallback_cl_a=3.0, fallback_cd_a=1.2)
    low, info = attitude_from_dynamics(veh, lat_g=0.0, long_g=0.0, speed_ms=20,
                                       static_ride_mm=80.0, aero_provider=prov)
    assert info["ride_source"] == "static - aero heave"
    assert low.ride_height_mm < 80.0          # downforce squashes the platform
    # faster => more downforce => more heave => lower ride height
    lower, info2 = attitude_from_dynamics(veh, lat_g=0.0, long_g=0.0, speed_ms=28,
                                          static_ride_mm=80.0, aero_provider=prov)
    assert info2["aero_heave_mm"] > info["aero_heave_mm"]
    assert lower.ride_height_mm < low.ride_height_mm


def test_full_chain_geometry_to_aero_coeffs():
    # hardpoints -> dynamics attitude -> aero map query -> cl_a/cd_a, no faked step
    from suspension.aero import (attitude_from_dynamics, AeroProvider, AeroMap,
                                 ReferenceAeroModel, CaseSpec, Attitude)
    import tempfile
    veh = _veh_with_springs()
    b = ReferenceAeroModel()
    with tempfile.TemporaryDirectory() as d:
        results = [b.run_case(CaseSpec(Attitude(roll_deg=r, speed_ms=20),
                                       "c.stl", reference_area_m2=1.0), d)
                   for r in (0.0, 2.0, 4.0)]
    amap = AeroMap.from_results(results, reference_area_m2=1.0)
    prov = AeroProvider(reference_area_m2=1.0, aero_map=amap)
    att, _ = attitude_from_dynamics(veh, lat_g=1.0, long_g=0.0, speed_ms=20)
    cl_a, cd_a = prov.cla_cda(att)
    assert cl_a > 0 and cd_a > 0      # a real, attitude-indexed aero number falls out


# --------------------------------------------------------------------------- #
#  snappyHexMesh dictionary generation
# --------------------------------------------------------------------------- #
def test_mesher_writes_full_toolchain():
    from suspension.aero import SnappyMesher, MeshParams, CaseSpec, Attitude
    import tempfile, os
    mp = MeshParams()
    mesher = SnappyMesher(mp)
    spec = CaseSpec(Attitude(roll_deg=2, ride_height_mm=25), "car.stl",
                    reference_length_m=1.55)
    with tempfile.TemporaryDirectory() as d:
        case = os.path.join(d, spec.case_name())
        os.makedirs(case)
        allmesh = mesher.write(spec, case)
        for rel in ("system/blockMeshDict", "system/surfaceFeatureExtractDict",
                    "system/snappyHexMeshDict", "system/meshQualityDict",
                    "system/decomposeParDict", "Allmesh",
                    "kinematik_mesh_attitude.json"):
            assert os.path.isfile(os.path.join(case, rel)), rel
        assert allmesh.endswith("Allmesh")


def test_mesher_applies_roll_and_ride_geometry_side():
    from suspension.aero import SnappyMesher, MeshParams, CaseSpec, Attitude
    import tempfile, os
    spec = CaseSpec(Attitude(roll_deg=3.5, ride_height_mm=20, pitch_deg=1, yaw_deg=5),
                    "car.stl", reference_length_m=1.55)
    with tempfile.TemporaryDirectory() as d:
        case = os.path.join(d, spec.case_name())
        os.makedirs(case)
        SnappyMesher(MeshParams()).write(spec, case)
        snappy = open(os.path.join(case, "system/snappyHexMeshDict")).read()
        # roll goes into the snappy transform; pitch/yaw must NOT be in the mesh dict
        assert "3.5" in snappy and "axisAngle" in snappy
        manifest = open(os.path.join(case, "kinematik_mesh_attitude.json")).read()
        assert '"roll_deg": 3.5' in manifest
        assert '"pitch_deg": 1' in manifest and '"yaw_deg": 5' in manifest
        # ride height 20mm => car shifts down 10mm (-0.010 m) from the 30mm nominal
        assert "-0.01000" in manifest


def test_mesher_domain_scales_with_car_length():
    from suspension.aero import SnappyMesher, MeshParams, CaseSpec, Attitude
    import tempfile, os
    mp = MeshParams(domain_behind=6.0)
    with tempfile.TemporaryDirectory() as d:
        for L in (1.0, 2.0):
            spec = CaseSpec(Attitude(), "car.stl", reference_length_m=L)
            case = os.path.join(d, f"L{L}")
            os.makedirs(case)
            SnappyMesher(mp).write(spec, case)
            bm = open(os.path.join(case, "system/blockMeshDict")).read()
            # downstream extent = 6 * L should appear as a vertex coord
            assert f"{6.0*L:.4f}" in bm


def test_parse_checkmesh_honest_about_missing_log():
    from suspension.aero import parse_checkmesh
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert parse_checkmesh(d) is None      # no log => None, never a guess


def test_parse_checkmesh_reads_real_count():
    from suspension.aero import parse_checkmesh
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "log.checkMesh"), "w") as f:
            f.write("Mesh stats\n    points: 9000000\n    cells: 23456789\n")
        assert parse_checkmesh(d) == 23456789


def test_openfoam_with_meshparams_emits_snappy():
    from suspension.aero import OpenFOAMSolver, MeshParams, CaseSpec, Attitude
    import tempfile, os
    b = OpenFOAMSolver(mesh_params=MeshParams())
    spec = CaseSpec(Attitude(roll_deg=1), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        case = b.write_case(spec, d)
        assert os.path.isfile(os.path.join(case, "system/snappyHexMeshDict"))
        # without meshing requested, no snappy dict
    b2 = OpenFOAMSolver()
    with tempfile.TemporaryDirectory() as d:
        case = b2.write_case(spec, d)
        assert not os.path.exists(os.path.join(case, "system/snappyHexMeshDict"))


def test_openfoam_refuses_to_solve_without_staged_stl():
    from suspension.aero import OpenFOAMSolver, MeshParams, CaseSpec, Attitude, SolverUnavailable
    import tempfile
    b = OpenFOAMSolver(mesh_params=MeshParams())
    spec = CaseSpec(Attitude(), "car.stl")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SolverUnavailable):
            b.run_case(spec, d)         # STL not staged => refuse, don't solve on no mesh
