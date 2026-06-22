# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Physics sanity tests for the flexible-body / compliance extension.

These pin the conventions and catch regressions in the new stack: the finite-
element core (flex.py), the member load-path solver (loadpath.py), and the
compliance coupling that re-solves the corner under load (compliance.py).

Like the kinematics suite these aren't a full cross-check against a commercial
FE/MBD solver — that's a great PR — but they nail the closed-form cases the
engine MUST reproduce (a bar's EA/L, a cantilever's 3EI/L^3, a Guyan series
reduction), the equilibrium residuals, and the signs of compliance steer/camber.

Run:  python -m pytest tests/  (or just: python tests/test_compliance.py)
"""
import numpy as np
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import (
    SuspensionKinematics, Hardpoints, VehicleDynamics, VehicleParams,
    MATERIALS, tube_section, axial_stiffness_tube,
    FlexElement, FlexMesh, guyan_condense, CondensedFlexBody,
    load_flex_body, read_mnf,
    WheelLoad, solve_member_forces,
    MemberStiffness, CompliantCorner, corner_wheel_load,
)
import suspension as _s


# --------------------------------------------------------------------------- #
#  Finite-element core: closed-form stiffness
# --------------------------------------------------------------------------- #
def test_tube_section_matches_annulus_formula():
    od, wall = 19.05, 0.9
    A, I, J = tube_section(od, wall)
    idia = od - 2 * wall
    A_exp = np.pi / 4.0 * (od**2 - idia**2)
    I_exp = np.pi / 64.0 * (od**4 - idia**4)
    assert abs(A - A_exp) / A_exp < 1e-12
    assert abs(I - I_exp) / I_exp < 1e-12
    assert abs(J - 2 * I_exp) / (2 * I_exp) < 1e-12   # thin tube: J = 2 I


def test_bar_axial_stiffness_is_EA_over_L():
    """A single axial bar between two interface nodes must give k = E A / L."""
    mat = "Steel 4130"
    E = MATERIALS[mat].E
    L = 350.0
    od, wall = 20.0, 1.0
    A, _, _ = tube_section(od, wall)
    nodes = {"a": (0, 0, 0), "b": (L, 0, 0)}
    el = [FlexElement("a", "b", kind="bar", material=mat, od_mm=od, wall_mm=wall)]
    body = FlexMesh(nodes, el, {"a": "a", "b": "b"}).condense()
    k = body.relative_axial_stiffness("a", "b")
    assert abs(k - E * A / L) / (E * A / L) < 1e-9


def test_guyan_series_two_beams_axial():
    """Two beams in series, condensed to the ends, must give E A / L_total.

    (Series validation uses BEAM elements: an axial bar chain leaves the interior
    node's transverse DOFs unconstrained, a deliberately singular case the code
    rejects — see flex.py.)
    """
    mat = "Steel 4130"
    E = MATERIALS[mat].E
    od, wall = 18.0, 1.2
    A, _, _ = tube_section(od, wall)
    L1, L2 = 150.0, 200.0
    nodes = {"a": (0, 0, 0), "m": (L1, 0, 0), "b": (L1 + L2, 0, 0)}
    el = [FlexElement("a", "m", kind="beam", material=mat, od_mm=od, wall_mm=wall),
          FlexElement("m", "b", kind="beam", material=mat, od_mm=od, wall_mm=wall)]
    body = FlexMesh(nodes, el, {"a": "a", "b": "b"}).condense()
    k = body.relative_axial_stiffness("a", "b")
    k_exp = E * A / (L1 + L2)
    assert abs(k - k_exp) / k_exp < 1e-6


def test_cantilever_tip_stiffness_3EI_over_L3():
    """A beam grounded at one end has lateral tip stiffness 3 E I / L^3."""
    mat = "Steel 4130"
    E = MATERIALS[mat].E
    od, wall = 25.0, 2.0
    _, I, _ = tube_section(od, wall)
    L = 300.0
    nodes = {"root": (0, 0, 0), "tip": (L, 0, 0)}
    el = [FlexElement("root", "tip", kind="beam", material=mat, od_mm=od, wall_mm=wall)]
    body = FlexMesh(nodes, el, {"root": "root", "tip": "tip"}).condense()
    # ground the root fully; push the tip in y
    k_tip = body.grounded_stiffness("tip", grounded=["root"], direction=(0, 1, 0))
    k_exp = 3.0 * E * I / L**3
    assert abs(k_tip - k_exp) / k_exp < 1e-3


def test_guyan_condense_raises_on_singular_master():
    """A bar chain with a free interior node -> singular Kss -> clear error."""
    nodes = {"a": (0, 0, 0), "m": (100.0, 0, 0), "b": (200.0, 0, 0)}
    el = [FlexElement("a", "m", kind="bar", od_mm=20, wall_mm=1),
          FlexElement("m", "b", kind="bar", od_mm=20, wall_mm=1)]
    mesh = FlexMesh(nodes, el, {"a": "a", "b": "b"})
    raised = False
    try:
        mesh.condense()
    except (np.linalg.LinAlgError, ValueError):
        raised = True
    assert raised, "expected a singular/ill-conditioned reduction to be rejected"


# --------------------------------------------------------------------------- #
#  Flex-body import / export
# --------------------------------------------------------------------------- #
def test_reduced_schema_roundtrips_verbatim():
    """A pre-reduced superelement (MNF-equivalent) loads and round-trips."""
    mat = "Steel 4130"
    od, wall = 20.0, 1.0
    L = 250.0
    nodes = {"a": (0, 0, 0), "b": (L, 0, 0)}
    el = [FlexElement("a", "b", kind="bar", material=mat, od_mm=od, wall_mm=wall)]
    body = FlexMesh(nodes, el, {"a": "a", "b": "b"}).condense()
    d = body.to_dict()
    assert d["type"] == "reduced"
    body2 = load_flex_body(d)
    k1 = body.relative_axial_stiffness("a", "b")
    k2 = body2.relative_axial_stiffness("a", "b")
    assert abs(k1 - k2) / k1 < 1e-12


def test_binary_mnf_raises_actionable_error():
    """A binary .mnf must raise NotImplementedError, not silently guess."""
    with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as fh:
        fh.write(b"\x00\x01\x02MNF\xff\xfe garbage binary header \x00\x00")
        path = fh.name
    try:
        raised = False
        try:
            read_mnf(path)
        except NotImplementedError as exc:
            raised = True
            assert "reduced" in str(exc).lower() or "export" in str(exc).lower()
        assert raised, "binary MNF should raise NotImplementedError"
    finally:
        os.unlink(path)


def test_flex_body_feeds_member_stiffness():
    """A condensed FEA body used as a member's stiffness matches its EA/L."""
    mat = "Steel 4130"
    E = MATERIALS[mat].E
    od, wall = 16.0, 1.0
    A, _, _ = tube_section(od, wall)
    L = 280.0
    nodes = {"out": (0, 0, 0), "in": (L, 0, 0)}
    el = [FlexElement("out", "in", kind="bar", material=mat, od_mm=od, wall_mm=wall)]
    body = FlexMesh(nodes, el, {"out": "out", "in": "in"}).condense()
    ms = MemberStiffness(flex_body=body, node_out="out", node_in="in")
    k = ms.axial_stiffness(L)   # length arg ignored on the FE path
    assert abs(k - E * A / L) / (E * A / L) < 1e-9


# --------------------------------------------------------------------------- #
#  Member load-path solver
# --------------------------------------------------------------------------- #
def test_member_force_equilibrium_residual_small():
    kin = SuspensionKinematics(Hardpoints.default())
    state = kin.static
    load = WheelLoad(Fx=0.0, Fy=-2800.0, Fz=2000.0, Mz=0.0)
    mf = solve_member_forces(kin, state, load)
    assert mf.residual < 1e-6, f"equilibrium residual too large: {mf.residual}"


def test_pure_vertical_load_reacts_vertically():
    """A pure vertical patch load: member axial forces on the upright sum to -Fz z."""
    kin = SuspensionKinematics(Hardpoints.default())
    state = kin.static
    Fz = 1000.0
    mf = solve_member_forces(kin, state, WheelLoad(Fx=0, Fy=0, Fz=Fz, Mz=0))
    total = np.zeros(3)
    for m, T in mf.forces.items():
        total += T * mf.axes[m]          # force the member applies to the upright
    # the links must react the patch vertical load (equal/opposite through upright)
    assert abs(total[0]) < 1e-5 * Fz
    assert abs(total[1]) < 1e-5 * Fz
    assert abs(abs(total[2]) - Fz) < 1e-4 * Fz


# --------------------------------------------------------------------------- #
#  Compliance coupling
# --------------------------------------------------------------------------- #
def test_rigid_path_unchanged_at_zero_load():
    """Zero load must reproduce the rigid corner exactly (no spurious compliance)."""
    hp = Hardpoints.default()
    cc = CompliantCorner.uniform_tube(hp)
    res = cc.solve(WheelLoad(0, 0, 0, 0))
    assert abs(res.compliance_toe) < 1e-9
    assert abs(res.compliance_camber) < 1e-9
    assert res.converged


def test_tie_rod_stretch_changes_toe():
    """Make ONLY the tie rod compliant: a lateral load must move toe, camber ~0."""
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    state = kin.static
    load = WheelLoad(Fx=0.0, Fy=-3000.0, Fz=2000.0, Mz=0.0)
    # tie-rod only, deliberately soft so the effect is unambiguous
    stiff = {"TR": MemberStiffness(k_direct=200.0)}
    cc = CompliantCorner(hp, stiff)
    res = cc.solve(load)
    assert res.converged
    assert abs(res.compliance_toe) > 1e-3, "soft tie rod should produce compliance steer"
    assert abs(res.compliance_camber) < abs(res.compliance_toe), \
        "tie-rod compliance should move toe far more than camber"


def test_lower_arm_compliance_changes_camber():
    """Soft lower arms under a lateral load must change camber."""
    hp = Hardpoints.default()
    load = WheelLoad(Fx=0.0, Fy=-3000.0, Fz=2000.0, Mz=0.0)
    stiff = {"LF": MemberStiffness(k_direct=400.0),
             "LR": MemberStiffness(k_direct=400.0)}
    cc = CompliantCorner(hp, stiff)
    res = cc.solve(load)
    assert res.converged
    assert abs(res.compliance_camber) > 1e-3, "soft lower arm should change camber"


def test_softer_tabs_increase_compliance_steer():
    """Adding chassis-tab compliance in series must increase compliance toe."""
    hp = Hardpoints.default()
    load = WheelLoad(Fx=0.0, Fy=-2800.0, Fz=2000.0, Mz=0.0)
    stiff_tube = CompliantCorner.uniform_tube(hp).solve(load)
    stiff_tabs = CompliantCorner.uniform_tube(hp, k_tab=8000.0).solve(load)
    assert abs(stiff_tabs.compliance_toe) > abs(stiff_tube.compliance_toe), \
        "series tab compliance should add to, not reduce, compliance steer"


def test_full_compliance_solve_15g_is_physical():
    """The headline 1.5 g front-outer case converges with sane magnitudes."""
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    veh = VehicleDynamics(VehicleParams(), front_kin=kin)
    res = veh.corner_compliance(1.5)
    assert res is not None
    assert res.converged
    # deflections must be small but non-zero, and toe/camber sub-degree for steel tube
    defl = max(abs(v) for v in res.member_deflection.values())
    assert 0 < defl < 5.0, f"implausible member deflection {defl} mm"
    assert abs(res.compliance_toe) < 1.0
    assert abs(res.compliance_camber) < 1.0
    # the loaded lower legs should be in compression in a corner
    assert res.member_forces["LF"] < 0 or res.member_forces["LR"] < 0


def test_corner_wheel_load_uses_real_load_transfer():
    """corner_wheel_load Fz must match the dynamics load-transfer outer-front load."""
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    veh = VehicleDynamics(VehicleParams(), front_kin=kin)
    loads, _ = veh.lateral_load_transfer(1.5)
    wl = corner_wheel_load(veh, "front", 1.5, outer=True)
    assert abs(wl.Fz - loads.fr) < 1e-6
    assert wl.Fy < 0, "outer cornering force points inboard (-y) in this corner model"


# --------------------------------------------------------------------------- #
#  Package wiring
# --------------------------------------------------------------------------- #
def test_public_api_and_version():
    assert _s.__version__ == "0.21.0"
    for name in ("CompliantCorner", "load_flex_body", "FlexMesh",
                 "MemberStiffness", "WheelLoad", "corner_wheel_load",
                 "StructuralTireModel", "ReferenceTireModel", "FTireModel",
                 "CDTireModel", "WheelState", "TireOutput", "run_cosim_maneuver"):
        assert hasattr(_s, name), f"missing public export: {name}"


# --------------------------------------------------------------------------- #
#  Non-linear joint compliance (bushings / rod ends / spherical bearings)
# --------------------------------------------------------------------------- #
from suspension import JointCompliance
from suspension.joints import JointCompliance as _JC


def test_joint_force_displacement_inverse_roundtrips():
    """force(displacement(F)) == F for every joint kind, tension and compression."""
    joints = [
        _JC.linear(5000.0),
        _JC.cubic(1500.0, 12000.0),
        _JC.bilinear(2000.0, 8000.0, 0.3),
        _JC.freeplay(0.05, 120000.0, k_lash=2000.0),
        _JC.rubber_bushing(),
        _JC.polyurethane_bushing(),
        _JC.spherical_bearing(),
        _JC.tabular([-1, -0.2, 0, 0.2, 1], [-5000, -400, 0, 400, 5000]),
    ]
    for jc in joints:
        for F in (-3000.0, -250.0, 0.0, 175.0, 4200.0):
            d = jc.displacement(F)
            F2 = jc.force(d)
            assert abs(F2 - F) < 1e-3 * max(1.0, abs(F)) + 1e-6, (jc.kind, F, F2)


def test_cubic_joint_is_progressively_stiffening():
    """A rubber-style cubic joint's tangent rate must rise with displacement."""
    jc = _JC.cubic(1000.0, 5000.0)
    assert jc.tangent_stiffness(0.0) < jc.tangent_stiffness(0.5) < jc.tangent_stiffness(1.0)
    # and softer than its high-displacement rate near zero (the 'soft off-centre')
    assert abs(jc.force(0.1)) < abs(jc.force(0.2)) / 2 * 1.0 + abs(jc.force(0.2))


def test_freeplay_gives_lash_before_taking_load():
    """Within the lash band a spherical bearing barely reacts; beyond it stiffens."""
    jc = _JC.freeplay(lash_mm=0.10, k=100000.0, k_lash=500.0)
    # a small displacement inside the band carries only the tiny contact load
    f_in_band = jc.force(0.05)
    assert abs(f_in_band) <= 0.10 * 500.0 + 1e-9
    # the engaged rate outside the band is the stiff k
    assert abs(jc.tangent_stiffness(0.2) - 100000.0) < 1e-6
    # at a given load the lash adds displacement vs a no-lash bearing
    d_lash = jc.displacement(1000.0)
    d_nolash = _JC.linear(100000.0).displacement(1000.0)
    assert d_lash > d_nolash


def test_series_deflection_adds_link_and_joints():
    """axial_deflection = link give + both joint gives (springs in series)."""
    ms = MemberStiffness(k_direct=20000.0,
                         joint_in=_JC.linear(8000.0),
                         joint_out=_JC.linear(50000.0))
    F, L = 1500.0, 300.0
    expect = F / 20000.0 + F / 8000.0 + F / 50000.0
    assert abs(ms.axial_deflection(F, L) - expect) < 1e-9
    b = ms.deflection_breakdown(F, L)
    assert abs((b["link"] + b["joint_in"] + b["joint_out"]) - expect) < 1e-9


def test_no_joint_path_is_unchanged():
    """A MemberStiffness with no joints must equal the old force/stiffness behaviour."""
    ms = MemberStiffness(material="Steel 4130", od_mm=19.05, wall_mm=0.9)
    L = 300.0
    k = ms.axial_stiffness(L)
    for F in (-2000.0, 0.0, 3500.0):
        assert abs(ms.axial_deflection(F, L) - F / k) < 1e-12


def test_rigid_link_with_only_joints_flexes_through_joints():
    """No link stiffness given -> link is rigid, all give is in the joints."""
    ms = MemberStiffness(joint_in=_JC.linear(10000.0))
    assert abs(ms.axial_deflection(2000.0, 250.0) - 2000.0 / 10000.0) < 1e-12
    raised = False
    try:
        MemberStiffness().axial_deflection(100.0, 250.0)   # nothing defined
    except ValueError:
        raised = True
    assert raised


def test_softer_rubber_makes_more_compliance_steer_than_bearings():
    """Rubber tie-rod bushings must produce more compliance steer than rod ends."""
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    veh = VehicleDynamics(VehicleParams(), front_kin=kin)
    load = corner_wheel_load(veh, "front", 1.5, outer=True)
    rubber = CompliantCorner.with_bushings(
        hp, bushing=_JC.rubber_bushing(), rod_end=_JC.spherical_bearing())
    bearings = CompliantCorner.with_bushings(
        hp, bushing=_JC.spherical_bearing(), rod_end=_JC.spherical_bearing())
    r = rubber.solve(load)
    b = bearings.solve(load)
    assert r.converged and b.converged
    assert abs(r.compliance_toe) > abs(b.compliance_toe), \
        "soft rubber bushings should give more compliance steer than spherical bearings"


def test_joints_produce_track_compliance_shift():
    """Compliant joints move the contact patch laterally (track compliance)."""
    hp = Hardpoints.default()
    veh = VehicleDynamics(VehicleParams(), front_kin=SuspensionKinematics(hp))
    load = corner_wheel_load(veh, "front", 1.5, outer=True)
    res = CompliantCorner.with_bushings(hp, bushing=_JC.rubber_bushing()).solve(load)
    assert abs(res.contact_patch_lateral_shift_mm) > 1e-3


def test_damping_energy_scales_and_is_inert_statically():
    """Energy/cycle ~ amplitude^2 and ~ frequency; zero amplitude -> zero loss."""
    jc = _JC.rubber_bushing()
    assert jc.energy_loss_per_cycle(0.0, 15.0) == 0.0
    e1 = jc.energy_loss_per_cycle(0.3, 10.0)
    e2 = jc.energy_loss_per_cycle(0.6, 10.0)
    assert e2 > e1                              # grows with amplitude
    # purely viscous joint: energy must rise with frequency
    visc = _JC.linear(5000.0, c_viscous=2.0, loss_factor=0.0)
    assert visc.energy_loss_per_cycle(0.5, 20.0) > visc.energy_loss_per_cycle(0.5, 10.0)


def test_static_solve_ignores_damping():
    """Two corners identical but for damping must give the same static compliance."""
    hp = Hardpoints.default()
    veh = VehicleDynamics(VehicleParams(), front_kin=SuspensionKinematics(hp))
    load = corner_wheel_load(veh, "front", 1.5, outer=True)
    undamped = _JC.cubic(1500.0, 12000.0, c_viscous=0.0, loss_factor=0.0)
    damped = _JC.cubic(1500.0, 12000.0, c_viscous=5.0, loss_factor=0.3)
    r_u = CompliantCorner.with_bushings(hp, bushing=undamped).solve(load)
    r_d = CompliantCorner.with_bushings(hp, bushing=damped).solve(load)
    assert abs(r_u.compliance_toe - r_d.compliance_toe) < 1e-9


def test_linearize_matches_curve_tangent():
    """linearize() reports the curve's tangent stiffness at the working force."""
    jc = _JC.cubic(1000.0, 6000.0)
    F = 800.0
    lin = jc.linearize(about_force_N=F)
    d = jc.displacement(F)
    assert abs(lin["tangent_stiffness_N_per_mm"] - jc.tangent_stiffness(d)) < 1e-6


def test_invalid_joint_definitions_are_rejected():
    """Non-physical / non-invertible joint definitions must raise."""
    for bad in (lambda: _JC.cubic(1000.0, -5.0),          # softening cubic
                lambda: _JC.linear(-10.0),                # negative rate
                lambda: _JC.tabular([0, 1, 2], [0, 0, 5]),  # non-increasing force
                lambda: _JC.tabular([0, 0, 1], [0, 1, 2])): # non-increasing disp
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        assert raised


def test_tabular_linear_table_recovers_slope():
    """A straight F-δ table must behave like a linear joint of that slope."""
    k = 3000.0
    jc = _JC.tabular([-2, -1, 0, 1, 2], [-2 * k, -k, 0, k, 2 * k])
    assert abs(jc.force(0.5) - 0.5 * k) < 1e-6
    assert abs(jc.displacement(k) - 1.0) < 1e-6


def test_result_carries_joint_breakdown():
    """The solved result exposes the per-joint give breakdown for the report."""
    hp = Hardpoints.default()
    veh = VehicleDynamics(VehicleParams(), front_kin=SuspensionKinematics(hp))
    load = corner_wheel_load(veh, "front", 1.5, outer=True)
    res = CompliantCorner.with_bushings(
        hp, bushing=_JC.rubber_bushing(), rod_end=_JC.spherical_bearing()).solve(load)
    assert "TR" in res.member_joint_deflection
    keys = set(res.member_joint_deflection["TR"])
    assert {"link", "joint_in", "joint_out"} <= keys
    # the breakdown must sum to the member's reported total deflection
    b = res.member_joint_deflection["TR"]
    assert abs(sum(b.values()) - res.member_deflection["TR"]) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
