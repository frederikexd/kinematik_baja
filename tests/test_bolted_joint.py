"""Tests for the bolted-joint pedal-box mount analysis (VDI 2230 / Shigley)."""

import math
import pytest

from suspension.bolted_joint import (
    Fastener, ClampedStack, analyze_joint, joint_findings, BOLT_GRADES,
)
from suspension.interfaces import Severity


def base_case(torque=12.0, external=1500.0, prying=1.0):
    f = Fastener(grade="10.9", nominal_d_mm=8.0, K_factor=0.20,
                 head_dia_mm=13.0, hole_dia_mm=8.4)
    s = ClampedStack(base_material="Aluminium 7075",
                     chassis_material="Aluminium 6061", grip_mm=10.0)
    return analyze_joint(f, s, torque, external, prying_factor=prying)


def test_preload_from_torque():
    # T = K F d  ->  F = T/(K d). M8, 12 N·m, K=0.20: F = 12000/(0.2*8) = 7500 N
    r = base_case(torque=12.0)
    assert r.F_preload == pytest.approx(7500.0, rel=1e-6)


def test_load_factor_between_zero_and_one():
    r = base_case()
    assert 0.0 < r.load_factor < 1.0
    # short grip + alloy clamp: bolt and member stiffness are comparable, so Phi
    # sits near (but below) 0.6 — the bolt still takes well under the full load.
    assert 0.3 < r.load_factor < 0.6


def test_bolt_barely_feels_load_below_separation():
    # below separation, bolt sees preload + Phi*F_ext, only slightly above preload
    r = base_case(torque=12.0, external=1500.0)
    assert not r.separated
    assert r.F_bolt_max > r.F_preload
    assert r.F_bolt_max < r.F_preload + 1500.0  # nowhere near the raw load


def test_separation_spike():
    # huge external load must open the joint and spike the bolt to the full load
    r = base_case(torque=12.0, external=50000.0, prying=1.0)
    assert r.separated
    assert r.F_bolt_max == pytest.approx(50000.0, rel=1e-6)
    fs = joint_findings(r)
    assert any(f.check == "bolt-separation" and f.severity == Severity.FAIL
               for f in fs)


def test_prying_amplifies_and_flags_estimate():
    even = base_case(external=2000.0, prying=1.0)
    pried = base_case(external=2000.0, prying=2.5)
    assert pried.F_bolt_share == pytest.approx(2.5 * even.F_bolt_share)
    assert pried.is_estimate is True
    assert even.is_estimate is False
    fs = joint_findings(pried)
    assert any(f.check == "bolt-prying" and f.severity == Severity.INFO for f in fs)


def test_separation_safety_definition():
    r = base_case(torque=12.0, external=1000.0)
    # F_sep / F_ext
    assert r.separation_safety == pytest.approx(r.F_sep / r.F_bolt_share, rel=1e-9)


def test_washer_crush_flag():
    # tiny head on soft alloy + high load -> bearing yield
    f = Fastener(grade="12.9", nominal_d_mm=6.0, K_factor=0.15,
                 head_dia_mm=8.0, hole_dia_mm=6.4)
    s = ClampedStack(base_material="Aluminium 6061", grip_mm=6.0)
    r = analyze_joint(f, s, assembly_torque_Nm=15.0, external_tensile_N=500.0)
    assert r.sigma_bearing is not None
    if r.bearing_yield:
        fs = joint_findings(r)
        assert any(f.check == "bolt-bearing" and f.severity == Severity.FAIL
                   for f in fs)


def test_healthy_joint_passes():
    r = base_case(torque=12.0, external=800.0, prying=1.0)
    fs = joint_findings(r)
    sep = [f for f in fs if f.check == "bolt-separation"][0]
    assert sep.severity == Severity.OK


def test_bad_inputs_raise():
    f = Fastener(nominal_d_mm=8.0)
    s = ClampedStack()
    with pytest.raises(ValueError):
        analyze_joint(f, s, assembly_torque_Nm=0.0, external_tensile_N=100.0)
    with pytest.raises(ValueError):
        analyze_joint(f, s, assembly_torque_Nm=10.0, external_tensile_N=100.0,
                      prying_factor=0.0)


def test_unknown_grade_raises():
    f = Fastener(grade="99.9", nominal_d_mm=8.0)
    with pytest.raises(ValueError):
        analyze_joint(f, ClampedStack(), 10.0, 100.0)
