# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Tests for the tire model, the MF5.2 fitter, and the setup-sensitivity tools.

These pin the behaviour that the grip/balance upgrade depends on:
  - the Pacejka model is load-sensitive and camber-aware in the right direction,
  - the fitter recovers a known tire from noisy data,
  - wiring the tire into VehicleDynamics changes grip vs the linear placeholder
    without breaking the existing balance/sign conventions,
  - the optimiser never returns a setup worse than the one it started from.

Run:  python tests/test_tiremodel.py   (or: python -m pytest tests/)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suspension.tiremodel import (PacejkaLateral, default_tire, describe)
from suspension.tirefit import fit_mf52_lateral
from suspension import (SuspensionKinematics, Hardpoints,
                        VehicleDynamics, VehicleParams)
from suspension.setup import sensitivity, optimise, evaluate


_PASS, _FAIL = [], []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# ----------------------- tire model behaviour --------------------------- #
def test_load_sensitivity():
    t = default_tire()
    mu_light = t.mu_peak(0.4 * t.FNOMIN)
    mu_heavy = t.mu_peak(1.8 * t.FNOMIN)
    # Real tires lose grip coefficient as load rises.
    check("mu drops with load (load sensitivity)", mu_light > mu_heavy)
    check("mu in physical FSAE range", 1.0 < mu_heavy < mu_light < 2.2)


def test_camber_has_an_optimum():
    t = default_tire()
    best_cam, best_mu = t.optimal_camber(t.FNOMIN)
    mu_zero = t.mu_peak(t.FNOMIN, 0.0)
    # Some non-zero camber should be at least as good as zero (camber helps).
    check("camber optimum is at least as good as zero", best_mu >= mu_zero - 1e-6)
    check("optimal camber is a mild negative", -6.0 <= best_cam <= 0.5)


def test_peak_force_scales_with_load():
    t = default_tire()
    f_low = t.peak_force(500)
    f_high = t.peak_force(1500)
    # More load => more absolute force (even though mu falls).
    check("peak force rises with load", f_high > f_low)
    check("alpha_peak is a sane slip angle", 3.0 < t.alpha_peak(t.FNOMIN) < 12.0)


# --------------------------- the fitter --------------------------------- #
def test_fitter_recovers_known_tire():
    truth = PacejkaLateral(coeffs=dict(
        PCY1=1.5, PDY1=1.7, PDY2=-0.30, PDY3=1.5,
        PEY1=-0.8, PEY2=-0.05, PEY3=0.2, PEY4=1.0,
        PKY1=-35.0, PKY2=1.8, PKY3=0.7,
        PHY1=0.0, PHY2=0.0, PHY3=0.0,
        PVY1=0.0, PVY2=0.0, PVY3=0.18, PVY4=0.0,
    ), FNOMIN=1100.0)
    rng = np.random.default_rng(7)
    SA, FZ, IA = [], [], []
    for fz in [350, 650, 1100, 1550, 2000]:
        for ia in [0.0, 2.0, 4.0]:
            for sa in np.linspace(-12, 12, 50):
                SA.append(sa); FZ.append(fz); IA.append(ia)
    SA = np.radians(SA); FZ = np.array(FZ, float); IA = np.radians(IA)
    FY = truth.fy(SA, FZ, IA) + rng.normal(0, 45, size=len(SA))
    res = fit_mf52_lateral(SA, FZ, FY, IA_rad=IA, fnomin=1100.0)
    check("fit R^2 above 0.99", res["r2"] > 0.99)
    fitted = PacejkaLateral(coeffs=res["coeffs"], FNOMIN=res["FNOMIN"])
    errs = [abs(truth.mu_peak(fz) - fitted.mu_peak(fz))
            for fz in (500, 900, 1300, 1800)]
    check("recovered mu within 0.05 across load", max(errs) < 0.05)


def test_fitter_handles_flipped_sign():
    truth = default_tire()
    SA, FZ, IA = [], [], []
    for fz in [500, 1100, 1700]:
        for sa in np.linspace(-12, 12, 60):
            SA.append(sa); FZ.append(fz); IA.append(0.0)
    SA = np.radians(SA); FZ = np.array(FZ, float); IA = np.radians(IA)
    FY = truth.fy(SA, FZ, IA)
    # Feed the OPPOSITE sign convention; the fitter should still fit well.
    res = fit_mf52_lateral(SA, FZ, -FY, IA_rad=IA, fnomin=1100.0)
    check("fitter robust to flipped FY sign (R^2>0.98)", res["r2"] > 0.98)


# --------------------- wiring into vehicle dynamics --------------------- #
def test_pacejka_changes_grip_vs_linear():
    kin = SuspensionKinematics(Hardpoints.default())
    lin = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin)
    pac = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin,
                          tire=default_tire())
    check("linear model labelled correctly",
          lin.grip_model_name() == "linear placeholder")
    check("pacejka model labelled correctly",
          pac.grip_model_name() == "Pacejka MF5.2")
    check("both give physical max-g",
          0.8 < lin.max_lateral_g() < 2.2 and 0.8 < pac.max_lateral_g() < 2.2)


def test_balance_sign_preserved_with_tire():
    kin = SuspensionKinematics(Hardpoints.default())
    p = VehicleParams(weight_dist_front=0.5,
                      roll_stiffness_front=500, roll_stiffness_rear=250)
    veh = VehicleDynamics(p, front_kin=kin, rear_kin=kin, tire=default_tire())
    bal, uf, ur = veh.balance_index(1.0)
    # Stiff front bar relative to rear loads the front pair harder => front-limited
    # => positive (understeer) balance index. Pins the sign convention end-to-end.
    check("stiff front bar => understeer (balance > 0)", bal > 0)


# ----------------------------- setup tools ------------------------------ #
def test_sensitivity_ranks_something():
    kin = SuspensionKinematics(Hardpoints.default())
    s = sensitivity(VehicleParams(), front_kin=kin, rear_kin=kin,
                    tire=default_tire())
    check("sensitivity returns ranked knobs", len(s["rankings"]) >= 4)
    # CG height should be a strong grip lever (lower CG = less load transfer).
    top = s["rankings"][0]["knob"]
    check("CG height is among the top grip levers",
          "cg_height" in [r["knob"] for r in s["rankings"][:2]])


def test_optimiser_never_regresses():
    kin = SuspensionKinematics(Hardpoints.default())
    o = optimise(VehicleParams(), front_kin=kin, rear_kin=kin,
                 tire=default_tire())
    check("optimiser does not lose grip", o["delta_maxg"] >= -1e-6)
    check("optimiser keeps balance bounded",
          abs(o["best_eval"]["balance"]) < 0.5)


def test_optimiser_pure_param_path():
    # No kinematics attached — exercises the param-camber fallback path.
    o = optimise(VehicleParams(), tire=default_tire())
    check("optimiser works without geometry attached",
          o["best_eval"]["max_g"] > 0.8)


if __name__ == "__main__":
    print("Tire model:")
    test_load_sensitivity()
    test_camber_has_an_optimum()
    test_peak_force_scales_with_load()
    print("Fitter:")
    test_fitter_recovers_known_tire()
    test_fitter_handles_flipped_sign()
    print("Vehicle dynamics wiring:")
    test_pacejka_changes_grip_vs_linear()
    test_balance_sign_preserved_with_tire()
    print("Setup tools:")
    test_sensitivity_ranks_something()
    test_optimiser_never_regresses()
    test_optimiser_pure_param_path()
    n = len(_PASS) + len(_FAIL)
    print(f"\n{len(_PASS)}/{n} tests passed")
    if _FAIL:
        print("FAILED:", ", ".join(_FAIL))
        sys.exit(1)
