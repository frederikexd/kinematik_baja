# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the higher-fidelity in-house aero: the 3D source-panel (boundary-element)
potential-flow solver (suspension/aero/panel_method.py) and its integration into
FluentVerificationSolver via the `method` switch.

These pin the behaviour that makes it a genuine fidelity step rather than a relabel:

  1. it SOLVES on the real STL — geometry that does not exist is an honest
     PanelMethodUnavailable, never a fabricated number,
  2. GROUND EFFECT emerges from the physics — lowering ride height increases
     downforce magnitude monotonically (the image-panel system, not a tuned term),
  3. total drag is physical — a flat-plate friction term is added to the (near-zero,
     by d'Alembert) potential pressure drag,
  4. it is honestly labelled POTENTIAL fidelity, UNCORRELATED, with notes that say it
     does not resolve separation/stall/wake,
  5. FluentVerificationSolver dispatches method='analytic'|'panel'|'auto' correctly,
     'auto' uses the panel solve when geometry is present and falls back to the
     analytic surrogate (recording why) when it is not,
  6. the Fluent verification deck is still written in every mode.

Run:  python -m pytest tests/test_panel_method.py
"""
import os
import tempfile

import numpy as np
import pytest
import trimesh

from suspension.aero import (
    CaseSpec, Attitude, SolverFidelity,
    PanelMethodModel, PanelParams, PanelMethodUnavailable,
    FluentVerificationSolver,
)


# --------------------------------------------------------------------------- #
#  A refined plate STL we can actually solve (a box is only 12 triangles)
# --------------------------------------------------------------------------- #
def _plate_stl(z_lift_m=0.10):
    box = trimesh.creation.box(extents=[1.5, 0.6, 0.08])
    box = box.subdivide().subdivide().subdivide()       # -> hundreds of panels
    box.apply_translation([0.0, 0.0, z_lift_m])         # sit it above the road (z=0)
    d = tempfile.mkdtemp(prefix="panel_stl_")
    path = os.path.join(d, "plate.stl")
    box.export(path)
    return path


def _spec(stl, h=30.0, pitch=0.0, yaw=0.0, v=27.0):
    return CaseSpec(Attitude(pitch_deg=pitch, yaw_deg=yaw, ride_height_mm=h,
                             speed_ms=v),
                    stl, reference_area_m2=0.9, reference_length_m=1.5)


# --------------------------------------------------------------------------- #
#  1. Honest hole when there is no geometry
# --------------------------------------------------------------------------- #
def test_missing_geometry_is_an_honest_hole():
    m = PanelMethodModel()
    with pytest.raises(PanelMethodUnavailable):
        m.solve(_spec("does_not_exist.stl"))


def test_too_coarse_surface_refuses():
    # a raw box is 12 triangles — below the min_panels floor
    box = trimesh.creation.box(extents=[1.0, 0.5, 0.1])
    d = tempfile.mkdtemp(); path = os.path.join(d, "coarse.stl"); box.export(path)
    m = PanelMethodModel(PanelParams(min_panels=24))
    with pytest.raises(PanelMethodUnavailable):
        m.solve(_spec(path))


# --------------------------------------------------------------------------- #
#  2. Ground effect emerges from the image-panel physics
# --------------------------------------------------------------------------- #
def test_ground_effect_increases_downforce_as_ride_height_drops():
    stl = _plate_stl()
    m = PanelMethodModel(PanelParams(max_panels=2000, ground_effect=True))
    cls = [m.solve(_spec(stl, h=h)).c_lift for h in (80.0, 50.0, 30.0, 18.0)]
    # all downforce (negative), and magnitude grows monotonically as we get lower
    assert all(c < 0 for c in cls)
    mags = [abs(c) for c in cls]
    assert mags == sorted(mags), f"downforce should grow as ride height drops: {mags}"
    assert mags[-1] > mags[0] * 1.3      # a meaningful, not marginal, increase


def test_ground_effect_off_is_weaker_than_on():
    stl = _plate_stl()
    on = PanelMethodModel(PanelParams(max_panels=2000, ground_effect=True))
    off = PanelMethodModel(PanelParams(max_panels=2000, ground_effect=False))
    cl_on = on.solve(_spec(stl, h=18.0)).c_lift
    cl_off = off.solve(_spec(stl, h=18.0)).c_lift
    # the road image adds downforce; with it off there is less
    assert abs(cl_on) > abs(cl_off)


# --------------------------------------------------------------------------- #
#  3. Drag is physical: friction is added to the potential pressure drag
# --------------------------------------------------------------------------- #
def test_total_drag_includes_friction_and_is_positive():
    stl = _plate_stl()
    m = PanelMethodModel(PanelParams(max_panels=2000))
    r = m.solve(_spec(stl))
    assert r.c_drag is not None and r.c_drag > 0.0
    assert "Cd(friction)" in r.notes


# --------------------------------------------------------------------------- #
#  4. Honest labelling
# --------------------------------------------------------------------------- #
def test_provenance_is_potential_and_uncorrelated_and_candid():
    stl = _plate_stl()
    r = PanelMethodModel().solve(_spec(stl))
    prov = r.provenance
    assert prov.fidelity == SolverFidelity.POTENTIAL
    assert prov.is_correlated is False
    assert prov.cell_count is not None and prov.cell_count > 0   # panels recorded
    low = prov.notes.lower()
    assert "potential" in low
    assert "separation" in low or "wake" in low   # candid about what it misses


# --------------------------------------------------------------------------- #
#  5. FluentVerificationSolver method dispatch
# --------------------------------------------------------------------------- #
def test_analytic_method_is_geometry_insensitive():
    b = FluentVerificationSolver(method="analytic")
    wd = tempfile.mkdtemp()
    # even with a real STL, the analytic surrogate ignores it
    stl = _plate_stl()
    r = b.run_case(_spec(stl), wd)
    assert "analytic surrogate" in r.notes.lower()
    assert r.provenance.backend == "fluent"


def test_panel_method_uses_geometry_and_records_panels():
    b = FluentVerificationSolver(method="panel")
    stl = _plate_stl()
    r = b.run_case(_spec(stl), tempfile.mkdtemp())
    assert r.provenance.backend == "panel-method"
    assert r.provenance.cell_count and r.provenance.cell_count > 0


def test_panel_method_raises_without_geometry():
    b = FluentVerificationSolver(method="panel")
    with pytest.raises(PanelMethodUnavailable):
        b.run_case(_spec("car.stl"), tempfile.mkdtemp())


def test_auto_uses_panel_when_geometry_present():
    b = FluentVerificationSolver(method="auto")
    stl = _plate_stl()
    r = b.run_case(_spec(stl), tempfile.mkdtemp())
    assert r.provenance.backend == "panel-method"


def test_auto_falls_back_to_analytic_without_geometry_and_says_why():
    b = FluentVerificationSolver(method="auto")
    r = b.run_case(_spec("car.stl"), tempfile.mkdtemp())
    assert r.provenance.backend == "fluent"          # analytic provenance
    assert "panel solve unavailable" in r.notes.lower()
    assert r.c_lift is not None and r.c_lift < 0     # still a usable number


def test_bad_method_rejected():
    with pytest.raises(ValueError):
        FluentVerificationSolver(method="rans")


# --------------------------------------------------------------------------- #
#  6. The Fluent deck is still written in every mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,geom", [
    ("analytic", "car.stl"),
    ("auto", "car.stl"),
])
def test_fluent_deck_written_in_each_mode(method, geom):
    b = FluentVerificationSolver(method=method)
    wd = tempfile.mkdtemp()
    spec = _spec(geom)
    b.run_case(spec, wd)
    jou = os.path.join(wd, spec.case_name() + ".jou")
    assert os.path.isfile(jou)
    assert "ANSYS Fluent" in open(jou).read()


def test_fluent_deck_written_for_panel_mode_with_geometry():
    b = FluentVerificationSolver(method="panel")
    wd = tempfile.mkdtemp()
    spec = _spec(_plate_stl())
    b.run_case(spec, wd)
    jou = os.path.join(wd, spec.case_name() + ".jou")
    assert os.path.isfile(jou)
