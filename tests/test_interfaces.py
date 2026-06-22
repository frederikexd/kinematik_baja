# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the subsystem interface ledger (suspension/interfaces.py).

These pin the integration-checker behaviour the feature exists for:
  * the eight subsystems are all present, undeclared = MISSING (not a silent pass),
  * each consistency rule fires FAIL on a real conflict and OK when satisfied,
  * mass rolls up and the combined CG is mass-weighted correctly,
  * estimate provenance is always surfaced, and
  * the whole ledger round-trips through dict/JSON.

Run:  python -m pytest tests/test_interfaces.py
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension import interfaces as I
from suspension.interfaces import Severity, SubsystemInterface


def _full_ledger():
    led = I.blank_ledger()
    led.target_mass_kg = 230
    led.includes_driver_kg = 80
    led.chassis_envelope_mm = (1200, 600, 500)
    led.driveline_torque_limit_nm = 250
    led.lv_supply_capacity_w = 180
    led.accumulator_voltage_v = 12
    led.set(SubsystemInterface("front-suspension", mass_kg=28, cg_x_mm=900, cg_y_mm=0,
                               cg_z_mm=180, is_estimate=False))
    led.set(SubsystemInterface("drivetrain", mass_kg=42, cg_x_mm=1100, cg_y_mm=0,
                               cg_z_mm=200, peak_torque_nm=200, voltage_v=400))
    led.set(SubsystemInterface("chassis", mass_kg=32, cg_x_mm=900, cg_y_mm=0,
                               cg_z_mm=200, env_x_mm=1200, env_y_mm=600, env_z_mm=500,
                               is_estimate=False))
    return led


def test_all_subsystems_present_in_blank():
    led = I.blank_ledger()
    assert set(led.interfaces.keys()) == set(I.SUBSYSTEMS)
    assert len(I.SUBSYSTEMS) == 5


def test_undeclared_is_missing_not_pass():
    led = I.blank_ledger()
    f = led.check_all()
    # a totally empty ledger should report missing mass, not silently pass
    assert any(x.severity == Severity.MISSING for x in f)
    s = I.summarize(f)
    assert s["worst"] in ("missing", "info")


# --------------------------------------------------------------------------- #
#  Mass + CG
# --------------------------------------------------------------------------- #
def test_mass_rollup_and_weighted_cg():
    led = I.blank_ledger()
    led.set(SubsystemInterface("chassis", mass_kg=10, cg_x_mm=0, cg_y_mm=0, cg_z_mm=100))
    led.set(SubsystemInterface("drivetrain", mass_kg=30, cg_x_mm=0, cg_y_mm=0, cg_z_mm=300))
    r = led.mass_rollup()
    assert abs(r["total_kg"] - 40) < 1e-9
    # mass-weighted z = (10*100 + 30*300)/40 = 250
    assert abs(r["cg_mm"][2] - 250) < 1e-6


def test_mass_budget_accounts_for_driver():
    led = I.blank_ledger()
    led.target_mass_kg = 230
    led.includes_driver_kg = 80          # net target 150
    for s in ("chassis", "drivetrain"):
        led.set(SubsystemInterface(s, mass_kg=75))  # 150 total = exactly net target
    f = led.check_all()
    budget = [x for x in f if x.check == "mass-budget-total"][0]
    assert budget.severity == Severity.OK


def test_mass_over_budget_fails():
    led = I.blank_ledger()
    led.target_mass_kg = 100
    led.includes_driver_kg = 0
    led.set(SubsystemInterface("chassis", mass_kg=130))
    f = led.check_all()
    budget = [x for x in f if x.check == "mass-budget-total"][0]
    assert budget.severity == Severity.FAIL


def test_lateral_cg_offset_warns():
    led = I.blank_ledger()
    led.set(SubsystemInterface("data-acquisition", mass_kg=20, cg_x_mm=800,
                               cg_y_mm=120, cg_z_mm=150))  # 120 mm off centreline
    f = led.check_all()
    assert any(x.check == "cg-lateral" and x.severity == Severity.WARN for x in f)


# --------------------------------------------------------------------------- #
#  Envelope fit
# --------------------------------------------------------------------------- #
def test_envelope_fit_fail():
    led = I.blank_ledger()
    led.chassis_envelope_mm = (1000, 500, 400)
    led.set(SubsystemInterface("chassis", env_x_mm=1100, env_y_mm=300, env_z_mm=300))
    f = led.check_all()
    assert any(x.check == "envelope-fit" and x.severity == Severity.FAIL for x in f)


def test_envelope_fit_ok_when_inside():
    led = I.blank_ledger()
    led.chassis_envelope_mm = (1000, 500, 400)
    led.set(SubsystemInterface("chassis", env_x_mm=300, env_y_mm=200, env_z_mm=200))
    f = led.check_all()
    assert not any(x.check == "envelope-fit" and x.severity == Severity.FAIL for x in f)


# --------------------------------------------------------------------------- #
#  Thermal / cooling airflow
# --------------------------------------------------------------------------- #
def test_cooling_airflow_shortfall_fails():
    led = I.blank_ledger()
    led.total_cooling_airflow_cms = 0.2
    led.set(SubsystemInterface("drivetrain", cooling_airflow_cms=0.35))
    f = led.check_all()
    assert any(x.check == "cooling-airflow" and x.severity == Severity.FAIL for x in f)


def test_cooling_airflow_sufficient_ok():
    led = I.blank_ledger()
    led.total_cooling_airflow_cms = 0.5
    led.set(SubsystemInterface("drivetrain", cooling_airflow_cms=0.3))
    f = led.check_all()
    assert any(x.check == "cooling-airflow" and x.severity == Severity.OK for x in f)


# --------------------------------------------------------------------------- #
#  Electrical
# --------------------------------------------------------------------------- #
def test_lv_power_over_budget_fails():
    led = I.blank_ledger()
    led.lv_supply_capacity_w = 500
    led.lv_voltage_v = 24
    led.set(SubsystemInterface("data-acquisition", power_draw_w=400, voltage_v=24))
    led.set(SubsystemInterface("drivetrain", power_draw_w=200, voltage_v=24))
    f = led.check_all()
    assert any(x.check == "lv-power" and x.severity == Severity.FAIL for x in f)


def test_hv_voltage_mismatch_warns():
    led = I.blank_ledger()
    led.accumulator_voltage_v = 12
    led.set(SubsystemInterface("drivetrain", voltage_v=600))
    f = led.check_all()
    assert any(x.check == "hv-voltage" and x.severity == Severity.WARN for x in f)


# --------------------------------------------------------------------------- #
#  Driveline torque
# --------------------------------------------------------------------------- #
def test_driveline_torque_exceeded_fails():
    led = I.blank_ledger()
    led.driveline_torque_limit_nm = 250
    led.set(SubsystemInterface("drivetrain", peak_torque_nm=300))
    f = led.check_all()
    assert any(x.check == "driveline-torque" and x.severity == Severity.FAIL for x in f)


def test_driveline_torque_within_ok():
    led = I.blank_ledger()
    led.driveline_torque_limit_nm = 250
    led.set(SubsystemInterface("drivetrain", peak_torque_nm=200))
    f = led.check_all()
    assert any(x.check == "driveline-torque" and x.severity == Severity.OK for x in f)


def test_driveline_torque_missing_limit():
    led = I.blank_ledger()
    led.set(SubsystemInterface("drivetrain", peak_torque_nm=200))
    f = led.check_all()
    assert any(x.check == "driveline-torque" and x.severity == Severity.MISSING for x in f)


# --------------------------------------------------------------------------- #
#  Provenance, summary, findings_for, serialization
# --------------------------------------------------------------------------- #
def test_estimate_provenance_always_flagged():
    led = I.blank_ledger()
    led.set(SubsystemInterface("chassis", mass_kg=12, is_estimate=True))
    f = led.check_all()
    assert any(x.check == "data-provenance" for x in f)


def test_findings_for_subsystem_filter():
    led = _full_ledger()
    f = led.check_all()
    pt = I.findings_for(f, "drivetrain")
    assert all("drivetrain" in x.subsystems for x in pt)
    assert len(pt) >= 1


def test_summary_worst_is_fail_when_any_fail():
    led = I.blank_ledger()
    led.driveline_torque_limit_nm = 100
    led.set(SubsystemInterface("drivetrain", peak_torque_nm=300))
    s = I.summarize(led.check_all())
    assert s["worst"] == "fail"


def test_ledger_round_trips_through_json():
    led = _full_ledger()
    js = json.dumps(led.as_dict())
    led2 = I.IntegrationLedger.from_dict(json.loads(js))
    assert set(led2.interfaces.keys()) == set(led.interfaces.keys())
    # a tuple field survives the round trip
    assert led2.chassis_envelope_mm == (1200, 600, 500)
    # checks produce the same worst severity
    assert I.summarize(led2.check_all())["worst"] == I.summarize(led.check_all())["worst"]


def test_declared_fields_excludes_none_and_meta():
    it = SubsystemInterface("front-suspension", mass_kg=9, brake_torque_nm=900)
    df = it.declared_fields()
    assert "mass_kg" in df and "brake_torque_nm" in df
    assert "name" not in df and "is_estimate" not in df and "notes" not in df
    assert "power_draw_w" not in df    # was None


def test_interface_tuple_fields_round_trip():
    it = SubsystemInterface("chassis", downforce_n_at_v=(600, 15),
                            env_origin_mm=(0, 0, 300))
    it2 = SubsystemInterface.from_dict(json.loads(json.dumps(it.as_dict())))
    assert it2.downforce_n_at_v == (600, 15)
    assert it2.env_origin_mm == (0, 0, 300)


# --------------------------------------------------------------------------- #
#  Documentation: rationale, change log, report export
# --------------------------------------------------------------------------- #
def test_rationale_and_provenance_round_trip():
    it = SubsystemInterface("chassis", mass_kg=12, rationale="targeting 600 N",
                            owner="Dana", updated_by="Dana", updated_on="2026-06-13",
                            is_estimate=False)
    it2 = SubsystemInterface.from_dict(json.loads(json.dumps(it.as_dict())))
    assert it2.rationale == "targeting 600 N"
    assert it2.owner == "Dana" and it2.updated_on == "2026-06-13"
    # rationale/owner are documentation, not numeric channels
    assert "rationale" not in it2.declared_fields()


def test_diff_interfaces_reports_value_changes():
    old = SubsystemInterface("chassis", mass_kg=12,
                             downforce_n_at_v=(550, 15), is_estimate=True).as_dict()
    new = SubsystemInterface("chassis", mass_kg=14,
                             downforce_n_at_v=(600, 15), is_estimate=False)
    changes = I.diff_interfaces(old, new)
    joined = " | ".join(changes)
    assert "mass" in joined and "12" in joined and "14" in joined
    assert "downforce" in joined
    assert any("confirmed" in c for c in changes)   # estimate -> confirmed logged


def test_diff_interfaces_no_change_is_empty():
    it = SubsystemInterface("front-suspension", mass_kg=9, brake_torque_nm=900)
    assert I.diff_interfaces(it.as_dict(), it) == []


def test_diff_from_blank_lists_all_new_values():
    new = SubsystemInterface("chassis", mass_kg=6, cooling_airflow_cms=0.3)
    changes = I.diff_interfaces(None, new)
    assert any("mass" in c for c in changes)
    assert any("cooling airflow" in c for c in changes)


def test_build_interface_markdown_contains_key_sections():
    led = I.blank_ledger()
    led.target_mass_kg = 230
    led.set(SubsystemInterface("chassis", mass_kg=12, cg_x_mm=1000, cg_y_mm=0,
                               cg_z_mm=300, downforce_n_at_v=(600, 15),
                               rationale="skidpad balance", owner="Dana",
                               is_estimate=False))
    md = I.build_interface_markdown(led, team_name="Elbee", season="2026")
    assert "# Elbee — Subsystem Interface Contract" in md
    assert "Car-level budgets" in md
    assert "Combined mass" in md
    assert "Subsystem interfaces" in md
    assert "Integration checks" in md
    assert "skidpad balance" in md           # rationale rendered
    assert "downforce" in md                  # human label, not field name
    assert "downforce_n_at_v" not in md


def test_report_flags_estimates_and_missing():
    led = I.blank_ledger()
    led.set(SubsystemInterface("chassis", mass_kg=12, is_estimate=True))
    md = I.build_interface_markdown(led)
    assert "estimate" in md.lower()
    assert "Not yet declared" in md            # honest about incompleteness


def test_old_ledger_without_doc_fields_loads():
    # Regression for the deployed AttributeError: a ledger saved before the
    # rationale/owner/updated fields existed must reconstruct with safe defaults,
    # not blow up when those attributes are read.
    led = I.blank_ledger()
    led.set(SubsystemInterface("chassis", mass_kg=12, cg_x_mm=1000,
                               cg_y_mm=0, cg_z_mm=300))
    d = led.as_dict()
    for idict in d["interfaces"].values():       # strip the new fields
        for k in ("rationale", "owner", "updated_by", "updated_on"):
            idict.pop(k, None)
    led2 = I.IntegrationLedger.from_dict(d)
    it = led2.get("chassis")
    assert it.rationale == "" and it.owner == "" and it.updated_on == ""
    # report and checks still run on the upgraded-in-memory object
    assert "Interface Contract" in I.build_interface_markdown(led2)
    assert I.summarize(led2.check_all())["n"] > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
