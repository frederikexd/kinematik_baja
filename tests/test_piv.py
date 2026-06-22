# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for PIV — measured flow-field vs CFD correlation.

These pin the behaviour that makes the feature a faithful flow-correlation tool
rather than a pretty overlay:

  1. the cross-correlation reduction recovers a KNOWN velocity from synthetic
     particle frames — i.e. the pixel displacement -> physical velocity math is
     right, to within the real single-pass window-loss bias,
  2. the provenance physics gates are honest: a heavy tracer fails the Stokes check,
     a mistuned Δt fails the timing check, a fixed floor raises the ground warning,
  3. the field correlation overlays PIV on a CFD slice of the SAME plane and reports
     low error when they agree,
  4. the headline question works: when CFD runs ATTACHED where the measured flow
     SEPARATES, the separation IoU collapses and the verdict is FIELD MISMATCH with
     the correct "CFD attached where the car stalls" diagnostic,
  5. the honesty contract holds — non-overlapping planes/extents are holes, never
     extrapolated; the offline rig writes a real plan and refuses to fabricate frames.

Run:  python -m pytest tests/test_piv.py
"""
import math
import os
import tempfile

import numpy as np
import pytest

from suspension.aero import (
    Attitude, SheetOrientation, LaserSheetPlane, PIVProvenance, FramePair,
    VelocityField, PIVProcessor, AcquisitionPlan, OfflinePIVRig, RigUnavailable,
    CFDFieldSlice, correlate_field, separation_mask, GroundState,
    DEFAULT_FIELD_TOL,
)


# --------------------------------------------------------------------------- #
#  Synthetic particle-image helpers
# --------------------------------------------------------------------------- #
def _seed(H, W, n, rng, sigma=1.5):
    return rng.uniform(0, W, n), rng.uniform(0, H, n)


def _render(xs, ys, H, W, sigma=1.5):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W))
    for x, y in zip(xs, ys):
        img += np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return img


def _uniform_shift_pair(H, W, shift_x, shift_y, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = _seed(H, W, n, rng)
    a = _render(xs, ys, H, W)
    b = _render(xs + shift_x, ys + shift_y, H, W)
    return a, b


# --------------------------------------------------------------------------- #
#  1. The reduction recovers a known velocity (the core claim)
# --------------------------------------------------------------------------- #
def test_cross_correlation_recovers_known_velocity():
    H = W = 256
    shift_x, shift_y = 5.0, -2.0          # image-pixel shift between pulses
    a, b = _uniform_shift_pair(H, W, shift_x, shift_y)
    mpp = 5e-4                            # 0.5 mm/px
    dt_us = 80.0
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    prov = PIVProvenance("rig", plane, dt_us, mpp,
                         particle_diameter_um=1.0, freestream_ms=20.0, window_px=64)
    vf = PIVProcessor(window_px=64, overlap=0.5, peak_ratio_min=1.05).process(
        FramePair(a, b, dt_us), prov, attitude=Attitude(speed_ms=20.0))

    exp_u = shift_x * mpp / (dt_us * 1e-6)
    exp_v = -shift_y * mpp / (dt_us * 1e-6)    # image-y flips to plane-up
    mu = float(np.median(vf.u[vf.valid]))
    mv = float(np.median(vf.v[vf.valid]))
    assert vf.coverage() > 0.95
    # within the real single-pass window-loss bias (~couple %), not exact
    assert abs(mu - exp_u) / abs(exp_u) < 0.03
    assert abs(mv - exp_v) / abs(exp_v) < 0.05


def test_velocity_scales_with_dt_and_magnification():
    # Same pixel shift, double the magnification -> double the velocity (the m/px lever)
    H = W = 192
    a, b = _uniform_shift_pair(H, W, 4.0, 0.0)
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    dt_us = 100.0
    proc = PIVProcessor(window_px=64, overlap=0.5, peak_ratio_min=1.05)
    v1 = proc.process(FramePair(a, b, dt_us),
                      PIVProvenance("r", plane, dt_us, 5e-4, freestream_ms=20.0, window_px=64))
    v2 = proc.process(FramePair(a, b, dt_us),
                      PIVProvenance("r", plane, dt_us, 1e-3, freestream_ms=20.0, window_px=64))
    m1 = float(np.median(v1.u[v1.valid]))
    m2 = float(np.median(v2.u[v2.valid]))
    assert abs(m2 / m1 - 2.0) < 0.02


# --------------------------------------------------------------------------- #
#  2. Provenance physics gates are honest
# --------------------------------------------------------------------------- #
def test_stokes_flags_heavy_tracer():
    plane = LaserSheetPlane()
    light = PIVProvenance("r", plane, 50.0, 5e-4, particle_diameter_um=1.0,
                          freestream_ms=20.0)
    heavy = PIVProvenance("r", plane, 50.0, 5e-4, particle_diameter_um=50.0,
                          freestream_ms=20.0)
    assert light.seeding_ok()
    assert not heavy.seeding_ok()
    assert heavy.stokes_number() > light.stokes_number()
    assert "lags the flow" in heavy.status()


def test_timing_band_flags_mistuned_dt():
    plane = LaserSheetPlane()
    # window 32 px, freestream 20 m/s, 0.5 mm/px: healthy band is 4..16 px shift
    good = PIVProvenance("r", plane, dt_us=200.0, magnification_m_per_px=5e-4,
                         freestream_ms=20.0, window_px=32)   # shift = 8 px
    tiny = PIVProvenance("r", plane, dt_us=10.0, magnification_m_per_px=5e-4,
                         freestream_ms=20.0, window_px=32)   # shift = 0.4 px
    assert good.timing_ok()
    assert not tiny.timing_ok()
    assert "outside" in tiny.status()


def test_fixed_floor_raises_ground_warning():
    plane = LaserSheetPlane()
    prov = PIVProvenance("r", plane, 200.0, 5e-4, freestream_ms=20.0,
                         window_px=32, ground_state=GroundState.FIXED_FLOOR)
    assert "fixed/suction floor" in prov.status()


# --------------------------------------------------------------------------- #
#  3 & 4. Field correlation: agreement passes, attached-vs-separated fails loud
# --------------------------------------------------------------------------- #
def _synthetic_field(plane, attached=True, freestream=20.0, nx=24, ny=16):
    """
    Build a measured-style VelocityField on an XZ plane. Freestream flows in +x.
    If attached: u>0 everywhere. If separated near the floor (low z), u<0 in a patch.
    """
    xv = np.linspace(0.0, 0.5, nx)
    yv = np.linspace(0.0, 0.3, ny)
    xs, ys = np.meshgrid(xv, yv)
    u = np.full_like(xs, freestream)
    v = np.zeros_like(xs)
    if not attached:
        # reversed flow in the lower-rear patch (a separation bubble)
        bub = (xs > 0.3) & (ys < 0.1)
        u[bub] = -0.3 * freestream
    valid = np.ones_like(xs, dtype=bool)
    prov = PIVProvenance("rig", plane, 80.0, 5e-4, freestream_ms=freestream, window_px=32)
    return VelocityField(plane, xs, ys, u, v, valid,
                         attitude=Attitude(speed_ms=freestream), provenance=prov)


def _cfd_slice_from(field: VelocityField, attached=True, freestream=20.0):
    """A CFD slice on the same plane/grid, optionally with matching/!matching sep."""
    u = np.full_like(field.xs, freestream)
    v = np.zeros_like(field.xs)
    if not attached:
        bub = (field.xs > 0.3) & (field.ys < 0.1)
        u[bub] = -0.3 * freestream
    return CFDFieldSlice(field.plane, field.xs, field.ys, u, v,
                         attitude=field.attitude)


def test_field_matches_when_cfd_reproduces_flow():
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    piv = _synthetic_field(plane, attached=False)        # real flow separates
    cfd = _cfd_slice_from(piv, attached=False)           # CFD separates too, same place
    rep = correlate_field(piv, cfd)
    assert rep.n_compared > 0
    assert rep.within_tol
    assert rep.sep_iou > 0.99
    assert "FIELD MATCHES" in rep.summary


def test_attached_cfd_vs_separated_reality_fails_loud():
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    piv = _synthetic_field(plane, attached=False)        # real flow SEPARATES
    cfd = _cfd_slice_from(piv, attached=True)            # CFD runs ATTACHED
    rep = correlate_field(piv, cfd)
    assert not rep.within_tol
    assert rep.sep_iou < DEFAULT_FIELD_TOL["sep_iou_min"]
    assert rep.sep_piv_frac > rep.sep_cfd_frac
    assert "ATTACHED where the car" in rep.summary or "SEPARATION DISAGREES" in rep.summary


def test_separation_mask_flags_reversed_flow():
    u = np.array([[1.0, -0.5], [2.0, 3.0]])
    valid = np.array([[True, True], [True, False]])
    sep = separation_mask(u, valid)
    assert sep[0, 1] and not sep[0, 0]
    assert not sep[1, 1]      # invalid never marked separated


# --------------------------------------------------------------------------- #
#  5. Honesty contract: holes not extrapolation; rig refuses to fabricate
# --------------------------------------------------------------------------- #
def test_non_overlapping_planes_refuse_to_correlate():
    p_xz = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0)
    p_yz = LaserSheetPlane(SheetOrientation.YZ_CROSS, 1.0)
    piv = _synthetic_field(p_xz)
    cfd = CFDFieldSlice(p_yz, piv.xs, piv.ys, piv.u, piv.v)
    with pytest.raises(ValueError):
        correlate_field(piv, cfd)


def test_cfd_extent_miss_is_a_hole_not_extrapolation():
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0)
    piv = _synthetic_field(plane, attached=True)
    # CFD slice covers a disjoint region far from the PIV grid -> nothing to compare
    cfd = CFDFieldSlice(plane,
                        piv.xs + 100.0, piv.ys + 100.0,
                        np.full_like(piv.u, 20.0), np.zeros_like(piv.v))
    rep = correlate_field(piv, cfd)
    assert rep.n_compared == 0
    assert not rep.within_tol
    assert "nothing to compare" in rep.summary


def test_offline_rig_writes_plan_and_refuses_to_acquire():
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    plan = AcquisitionPlan(plane=plane, attitude=Attitude(speed_ms=20.0),
                           dt_us=80.0, magnification_m_per_px=5e-4,
                           n_pairs=200, particle_diameter_um=1.0,
                           window_px=32, freestream_ms=20.0)
    rig = OfflinePIVRig()
    workdir = tempfile.mkdtemp(prefix="kinematik_piv_test_")
    path = rig.write_plan(plan, workdir)
    assert os.path.exists(path)
    with open(path) as f:
        txt = f.read()
    assert "acquisition plan" in txt.lower()
    assert "dt_us" in txt
    with pytest.raises(RigUnavailable):
        rig.acquire(plan, workdir)


def test_acquisition_plan_makes_consistent_provenance():
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    plan = AcquisitionPlan(plane=plane, attitude=Attitude(speed_ms=27.0),
                           dt_us=120.0, magnification_m_per_px=4e-4,
                           freestream_ms=27.0, window_px=32)
    prov = plan.to_provenance("A2 Wind Shear")
    assert prov.dt_us == 120.0
    assert prov.freestream_ms == 27.0
    assert prov.plane is plane


# --------------------------------------------------------------------------- #
#  Regression guards for the two real bugs found wiring the field correlation
# --------------------------------------------------------------------------- #
def test_on_grid_handles_descending_row_axis():
    # The PIV plane-up flip leaves the row (y) axis DESCENDING. A resampler that
    # assumes ascending order scrambles every lookup, so pin node-exact recovery on a
    # descending-y grid: the resampled field must equal the source at coincident nodes.
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0)
    xv = np.linspace(0.0, 0.5, 12)
    yv = np.linspace(0.3, 0.0, 8)             # DESCENDING, as the processor produces
    xs, ys = np.meshgrid(xv, yv)
    u = xs * 10.0 + ys                        # a field that varies in both axes
    v = ys * 3.0
    cfd = CFDFieldSlice(plane, xs, ys, u, v)
    uq, vq, inside = cfd.on_grid(xs, ys)
    assert inside.all()
    assert np.allclose(uq, u, atol=1e-9)      # node-exact, no scrambling
    assert np.allclose(vq, v, atol=1e-9)


def test_recirculation_does_not_blow_up_angle_metric():
    # Inside a shared separation bubble the vectors point "backwards"; a ~180 deg
    # angle difference there is expected and must NOT fail an otherwise-matching field.
    # The separation IoU scores those windows instead. Build two fields that BOTH
    # separate in the same patch and agree elsewhere -> MATCH, angle metric small.
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, 0.0, 1.0)
    piv = _synthetic_field(plane, attached=False)
    cfd = _cfd_slice_from(piv, attached=False)
    rep = correlate_field(piv, cfd)
    assert rep.within_tol
    assert rep.angle_rms_deg < 1.0            # attached region agrees to <1 deg
    assert rep.sep_iou > 0.99                 # recirculation handled by IoU, not angle
