# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for suspension.harness — the 3-D wiring harness layer.

These exercise the exact deliverables the brief asks for:
  * route individual wires in 3-D car space and catch a bend tighter than the
    conductor can take and a connector entry with no strain relief;
  * clearance-check a routed wire against the same keep-out volumes the
    mount-point clash uses (a loom through the accumulator box);
  * derive the manufacturing artefacts straight off the geometry — exact cut
    length to the mm, a 1:1 length-true formboard, the automated BOM, and the
    exact copper mass + distribution (CG).

Every physical model (AWG copper area, 3-D arc length, copper density) is
analytic / tabular, so the numbers are reproducible and the asserts are exact-ish.
"""

import math
import numpy as np

from suspension.harness import (
    Connector, WireRun, HarnessLedger, check_harness,
    awg_area_mm2, awg_nominal_od_mm, polyline_length_mm,
    segment_lengths_mm, vertex_bend_radius_mm,
    CU_DENSITY_KG_M3,
)
from suspension.interfaces import Severity
from suspension.mountpoints import KeepOut


# --------------------------------------------------------------------------- #
#  AWG tables + geometry sanity
# --------------------------------------------------------------------------- #
def test_awg_area_known_and_interpolated():
    # AWG10 is a tabulated value
    assert math.isclose(awg_area_mm2(10), 5.261, rel_tol=1e-3)
    # area roughly halves every 3 gauges: AWG13 ~ half of AWG10
    assert awg_area_mm2(13) < awg_area_mm2(10)
    assert awg_area_mm2(13) > awg_area_mm2(16)
    # thicker wire (smaller AWG) has more copper
    assert awg_area_mm2(8) > awg_area_mm2(20)


def test_awg_nominal_od_monotonic():
    assert awg_nominal_od_mm(8) > awg_nominal_od_mm(20) > awg_nominal_od_mm(30)


def test_polyline_length_3d():
    p = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 12]], float)
    # 5 + 12 = 17
    assert math.isclose(polyline_length_mm(p), 17.0, rel_tol=1e-9)


def test_vertex_bend_radius_sharp_vs_gentle():
    sharp = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [10, 5, 0]], float)
    gentle = np.array([[0, 0, 0], [500, 0, 0], [1000, 50, 0]], float)
    rs = vertex_bend_radius_mm(sharp)
    rg = vertex_bend_radius_mm(gentle)
    assert float(np.min(rs)) < float(np.min(rg))


# --------------------------------------------------------------------------- #
#  Cut length to the millimetre — every term explicit
# --------------------------------------------------------------------------- #
def test_cut_length_terms_sum():
    # dead-straight run: no bend allowance, just routed + loop + strip(both ends)
    w = WireRun("w", "electrics", gauge_awg=18,
                path_mm=[(0, 0, 0), (1000, 0, 0)],
                service_loop_mm=40.0, strip_mm=10.0)
    routed = w.routed_length_mm()
    assert math.isclose(routed, 1000.0, rel_tol=1e-9)
    # straight => zero bend allowance
    assert math.isclose(w.cut_length_mm(),
                        1000.0 + 40.0 + 2 * 10.0, rel_tol=1e-9)


def test_cut_length_adds_bend_allowance_when_cornered():
    straight = WireRun("s", "e", gauge_awg=18,
                       path_mm=[(0, 0, 0), (1000, 0, 0)], strip_mm=0.0)
    cornered = WireRun("c", "e", gauge_awg=18,
                       path_mm=[(0, 0, 0), (500, 0, 0), (500, 500, 0)],
                       strip_mm=0.0)
    # cornered routed length == 1000 too, but with a real corner it gets a
    # positive bend allowance, so the cut length exceeds the routed length
    assert math.isclose(cornered.routed_length_mm(), 1000.0, rel_tol=1e-9)
    assert cornered.cut_length_mm() > cornered.routed_length_mm()
    assert math.isclose(straight.cut_length_mm(), 1000.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
#  Exact copper mass + distribution
# --------------------------------------------------------------------------- #
def test_copper_mass_exact():
    # 1 m of AWG10 (5.261 mm^2): mass = density * area * length
    w = WireRun("w", "e", gauge_awg=10,
                path_mm=[(0, 0, 0), (1000, 0, 0)], strip_mm=0.0, service_loop_mm=0.0)
    expect_g = CU_DENSITY_KG_M3 * (5.261e-6) * 1.0 * 1000.0
    assert math.isclose(w.copper_mass_g(), expect_g, rel_tol=1e-3)


def test_mass_distribution_cg_between_wires():
    hl = HarnessLedger()
    hl.set_wire(WireRun("a", "e", gauge_awg=10,
                        path_mm=[(0, 0, 0), (100, 0, 0)], strip_mm=0.0))
    hl.set_wire(WireRun("b", "e", gauge_awg=10,
                        path_mm=[(1000, 0, 0), (1100, 0, 0)], strip_mm=0.0))
    md = hl.mass_distribution()
    assert md["total_copper_g"] > 0
    # equal wires symmetric about x=550 -> CG x near 550
    assert 500.0 < md["harness_cg_mm"][0] < 600.0


def test_connector_without_mass_excluded_from_cg_and_listed():
    hl = HarnessLedger()
    hl.set_connector(Connector("C", "e", xyz_mm=(0, 0, 0), mass_g=None))
    hl.set_wire(WireRun("a", "e", gauge_awg=10,
                        path_mm=[(0, 0, 0), (100, 0, 0)], from_conn="C"))
    md = hl.mass_distribution()
    assert "C" in md["connectors_without_declared_mass"]


# --------------------------------------------------------------------------- #
#  Bend-radius + strain-relief findings (the bench scrap checks)
# --------------------------------------------------------------------------- #
def test_kinked_wire_is_bend_fail():
    hl = HarnessLedger()
    # sharp corner over very short legs -> tiny formed radius << min bend radius
    hl.set_wire(WireRun("kink", "electrics", gauge_awg=10,
                        path_mm=[(0, 0, 0), (5, 0, 0), (5, 5, 0), (1000, 5, 0)],
                        bundle_min_radius_mult=6.0, is_estimate=False))
    res = check_harness(hl, keepouts=[KeepOut("k", "chassis",
                                              lo_mm=(9000, 9000, 9000),
                                              hi_mm=(9100, 9100, 9100))])
    bends = [f for f in res.findings if f.check == "harness-bend"]
    assert any(f.severity == Severity.FAIL for f in bends)


def test_gentle_sweep_passes_bend():
    hl = HarnessLedger()
    hl.set_wire(WireRun("sweep", "electrics", gauge_awg=10,
                        path_mm=[(0, 0, 0), (500, 0, 0), (1000, 80, 0)],
                        bundle_min_radius_mult=6.0, is_estimate=False))
    res = check_harness(hl)
    bends = [f for f in res.findings if f.check == "harness-bend"]
    assert not any(f.severity == Severity.FAIL for f in bends)


def test_no_strain_relief_warns():
    hl = HarnessLedger()
    hl.set_connector(Connector("ECU", "electrics", xyz_mm=(0, 0, 0),
                               strain_relief_mm=25.0))
    # first segment is only 5 mm before a bend -> strain-relief WARN
    hl.set_wire(WireRun("w", "electrics", gauge_awg=18,
                        path_mm=[(0, 0, 0), (5, 0, 0), (5, 500, 0)],
                        from_conn="ECU", is_estimate=False))
    res = check_harness(hl)
    sr = [f for f in res.findings if f.check == "harness-strain-relief"]
    assert any(f.severity == Severity.WARN for f in sr)


# --------------------------------------------------------------------------- #
#  3-D clearance vs the SAME keep-outs the mount-points use
# --------------------------------------------------------------------------- #
def test_wire_through_keepout_is_clearance_fail():
    hl = HarnessLedger()
    hl.set_wire(WireRun("thru", "electrics", gauge_awg=14,
                        path_mm=[(0, 0, 0), (1000, 0, 0)], is_estimate=False))
    box = KeepOut("accumulator", "powertrain",
                  lo_mm=(400, -50, -50), hi_mm=(600, 50, 50))
    res = check_harness(hl, keepouts=[box])
    cl = [f for f in res.findings if f.check == "harness-clearance"]
    fails = [f for f in cl if f.severity == Severity.FAIL]
    assert fails
    # both owners named on the finding
    assert set(fails[0].subsystems) == {"electrics", "powertrain"}


def test_wire_clear_of_keepout_passes():
    hl = HarnessLedger()
    hl.set_wire(WireRun("clear", "electrics", gauge_awg=14,
                        path_mm=[(0, 0, 0), (1000, 0, 0)], is_estimate=False))
    box = KeepOut("box", "chassis", lo_mm=(400, 200, 200), hi_mm=(600, 300, 300))
    res = check_harness(hl, keepouts=[box])
    cl = [f for f in res.findings if f.check == "harness-clearance"]
    assert not any(f.severity == Severity.FAIL for f in cl)


def test_clearance_is_truly_3d_not_topdown():
    """A wire that routes OVER a keep-out box in z must clear it; the same route
    flattened into the box's z-band must foul it. Guards against a 2-D top-down
    projection masquerading as a 3-D clearance check."""
    box = KeepOut("accumulator", "powertrain",
                  lo_mm=(500, -50, -50), hi_mm=(700, 50, 50))
    # climbs to z=100, well above the box's z=50 top -> clears
    over = HarnessLedger()
    over.set_wire(WireRun("over", "electrics", gauge_awg=10,
                          path_mm=[(0, 0, 0), (600, 0, 100), (1200, 0, 0)],
                          is_estimate=False))
    res_over = check_harness(over, keepouts=[box])
    assert not any(f.check == "harness-clearance" and f.severity == Severity.FAIL
                   for f in res_over.findings)
    # same x-path but flat at z=0 -> straight through the box
    flat = HarnessLedger()
    flat.set_wire(WireRun("flat", "electrics", gauge_awg=10,
                          path_mm=[(0, 0, 0), (600, 0, 0), (1200, 0, 0)],
                          is_estimate=False))
    res_flat = check_harness(flat, keepouts=[box])
    assert any(f.check == "harness-clearance" and f.severity == Severity.FAIL
               for f in res_flat.findings)


def test_clearance_missing_without_keepouts():
    hl = HarnessLedger()
    hl.set_wire(WireRun("w", "electrics", gauge_awg=18,
                        path_mm=[(0, 0, 0), (100, 0, 0)]))
    res = check_harness(hl, keepouts=[])
    cl = [f for f in res.findings if f.check == "harness-clearance"]
    assert any(f.severity == Severity.MISSING for f in cl)


# --------------------------------------------------------------------------- #
#  Automated BOM
# --------------------------------------------------------------------------- #
def test_bom_rolls_up_wire_connectors_contacts():
    hl = HarnessLedger()
    hl.set_connector(Connector("ECU", "electrics", cavities=12,
                               part_number="DTM-12", mass_g=18.0))
    hl.set_connector(Connector("MOT", "powertrain", cavities=3,
                               part_number="AMP-3", mass_g=40.0))
    hl.set_wire(WireRun("p1", "electrics", gauge_awg=10,
                        path_mm=[(0, 0, 0), (1000, 0, 0)],
                        from_conn="ECU", to_conn="MOT"))
    hl.set_wire(WireRun("p2", "electrics", gauge_awg=10,
                        path_mm=[(0, 0, 0), (800, 0, 0)],
                        from_conn="ECU", to_conn="MOT"))
    bom = hl.bom()
    # two AWG10 conductors rolled into one gauge row
    awg10 = [r for r in bom["wire"] if r["gauge_awg"] == 10]
    assert awg10 and awg10[0]["conductors"] == 2
    # 2 wires * 2 ends each landed in known connectors = 4 contacts
    assert bom["contacts_total"] == 4
    # two distinct connector part numbers
    assert len(bom["connectors"]) == 2
    assert bom["total_copper_g"] > 0


# --------------------------------------------------------------------------- #
#  1:1 formboard is length-true (the manufacturing guarantee)
# --------------------------------------------------------------------------- #
def test_formboard_is_length_true():
    hl = HarnessLedger()
    w = WireRun("w", "electrics", gauge_awg=18,
                path_mm=[(0, 0, 0), (300, 0, 100), (300, 400, 100)],
                from_conn="A", to_conn="B")
    hl.set_wire(w)
    fb = hl.formboard()
    assert len(fb.branches) == 1
    pts = np.array(fb.branches[0].points_mm, float)
    flat_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    # the flat unfold preserves the 3-D routed arc length exactly
    assert math.isclose(flat_len, w.routed_length_mm(), rel_tol=1e-3)


def test_formboard_extent_nonneg_origin():
    hl = HarnessLedger()
    hl.set_wire(WireRun("w", "e", gauge_awg=18,
                        path_mm=[(0, 0, 0), (-300, -200, 0)]))
    fb = hl.formboard()
    # after the corner-origin shift, every point is >= 0
    for b in fb.branches:
        for (x, y) in b.points_mm:
            assert x >= -1e-6 and y >= -1e-6


# --------------------------------------------------------------------------- #
#  Persistence round-trip
# --------------------------------------------------------------------------- #
def test_harness_roundtrip():
    hl = HarnessLedger(ambient_c=45.0, clearance_warn_mm=12.0)
    hl.set_connector(Connector("ECU", "electrics", xyz_mm=(1.0, 2.0, 3.0),
                               cavities=8, part_number="DTM-08", mass_g=15.0,
                               is_estimate=False))
    hl.set_wire(WireRun("w", "electrics", gauge_awg=16,
                        path_mm=[(0, 0, 0), (100, 0, 0), (100, 100, 0)],
                        from_conn="ECU", net="lv", service_loop_mm=30.0,
                        is_estimate=False))
    d = hl.as_dict()
    hl2 = HarnessLedger.from_dict(d)
    assert math.isclose(hl2.ambient_c, 45.0)
    assert math.isclose(hl2.clearance_warn_mm, 12.0)
    assert "ECU" in hl2.connectors and "w" in hl2.wires
    w2 = hl2.wires["w"]
    assert w2.gauge_awg == 16 and len(w2.path_mm) == 3
    assert math.isclose(w2.cut_length_mm(), hl.wires["w"].cut_length_mm(), rel_tol=1e-9)


def test_full_gate_returns_all_artefacts():
    hl = HarnessLedger()
    hl.set_connector(Connector("ECU", "electrics", xyz_mm=(0, 0, 0), mass_g=18.0))
    hl.set_connector(Connector("MOT", "powertrain", xyz_mm=(1000, 0, 0), mass_g=40.0))
    hl.set_wire(WireRun("p", "electrics", gauge_awg=10,
                        path_mm=[(0, 0, 0), (500, 0, 0), (1000, 0, 0)],
                        from_conn="ECU", to_conn="MOT", is_estimate=False))
    res = check_harness(hl, keepouts=[KeepOut("k", "chassis",
                                              lo_mm=(0, 500, 0), hi_mm=(50, 600, 50))])
    assert res.cut_list and res.bom and res.mass and res.formboard is not None
    assert "FAIL" in res.summary() or "OK" in res.summary() or "WARN" in res.summary()
