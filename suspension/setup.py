# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Setup sensitivity & optimisation — spend your ONE tire set wisely.

An underfunded team's edge isn't more testing, it's better decisions before the
test. This module answers the two questions that actually move you up the results
sheet when you can't iterate on rubber:

    SENSITIVITY  — of every setup/geometry knob, which one buys the most grip (or
                   fixes the balance) per unit change? Rank them so you spend your
                   limited build/tune time on the levers that matter, not the ones
                   that feel important.

    OPTIMISE     — search the high-impact knobs for the combination that maximises
                   limit grip while holding balance in a target window (mild
                   understeer is fast and safe for an FSAE driver). This is a
                   coarse, transparent grid/coordinate search — not a black box —
                   so you can see *why* it picked what it picked and sanity-check it.

Everything runs on whatever grip model VehicleDynamics is carrying: the generic
default tire, or — once you've fitted it — YOUR TTC tire. The rankings are only as
trustworthy as the tire underneath them, which is exactly why fitting your tire
first is the highest-value thing you can do.

Design choice: we perturb a *copy* of the params/geometry and rebuild the model,
so nothing mutates the live setup. Each evaluation is one max_lateral_g + one
balance call — cheap enough to sweep dozens of points interactively.
"""

from __future__ import annotations

import copy
import numpy as np
from dataclasses import replace

from .kinematics import SuspensionKinematics, Hardpoints
from .dynamics import VehicleDynamics, VehicleParams


# Knobs we can sweep on VehicleParams, with a sensible step and unit label. These
# are the levers a suspension lead actually has between now and competition.
#
# IMPORTANT: the stiffness levers are SPRING RATES (N/mm) and ARB rates, NOT an
# abstract roll stiffness. A spring rate only becomes a wheel/roll rate THROUGH the
# corner's motion ratio (k_wheel = k_spring · MR²), so with geometry attached the
# optimiser explores the parts you actually fit and the rocker geometry decides what
# they do. Sweeping roll stiffness directly (the old behaviour) silently assumed a
# motion ratio of 1 and made the spring numbers meaningless — which is exactly the
# bug this fixes.
PARAM_KNOBS = {
    "weight_dist_front": dict(label="Front weight bias", unit="frac", step=0.02,
                              lo=0.42, hi=0.58),
    "cg_height":         dict(label="CG height", unit="mm", step=10.0,
                              lo=240.0, hi=360.0),
    "spring_rate_front": dict(label="Front spring rate", unit="N/mm",
                              step=5.0, lo=20.0, hi=120.0),
    "spring_rate_rear":  dict(label="Rear spring rate", unit="N/mm",
                              step=5.0, lo=20.0, hi=120.0),
    "arb_rate_front":    dict(label="Front anti-roll bar", unit="N·m/°",
                              step=25.0, lo=0.0, hi=400.0),
    "arb_rate_rear":     dict(label="Rear anti-roll bar", unit="N·m/°",
                              step=25.0, lo=0.0, hi=400.0),
    "static_camber_front":  dict(label="Front static camber", unit="°",
                                 step=0.5, lo=-4.0, hi=0.0),
    "static_camber_rear":   dict(label="Rear static camber", unit="°",
                                 step=0.5, lo=-4.0, hi=0.0),
}


def _build(params, front_kin, rear_kin, tire):
    return VehicleDynamics(params, front_kin=front_kin, rear_kin=rear_kin, tire=tire)


def _prepare(params: VehicleParams, front_kin, rear_kin):
    """
    Seed the camber params from the solved geometry (so the starting point matches
    the live car) and switch the grip model to use those params, so the setup tools
    can treat camber as a free lever. Also switch stiffness to SPRING-RATE mode so
    the optimiser's spring/ARB levers flow through the real motion ratio into roll
    stiffness (k_wheel = k_spring·MR²) instead of being ignored in favour of a
    directly-specified roll stiffness. Returns a copy — never mutates the caller's.
    """
    p = replace(params)
    if front_kin is not None:
        p = replace(p, static_camber_front=front_kin.static.camber)
    if rear_kin is not None:
        p = replace(p, static_camber_rear=rear_kin.static.camber)
    p = replace(p, use_param_camber=True)
    # Only switch to spring-rate-driven stiffness when geometry is attached on at
    # least one axle; otherwise there is no motion ratio to convert through and the
    # directly-specified roll stiffness remains the honest fallback.
    if front_kin is not None or rear_kin is not None:
        p = replace(p, use_spring_rates=True)
    return p


def evaluate(params: VehicleParams, front_kin=None, rear_kin=None, tire=None):
    """Grip + balance for a setup. The objective the rankings are built on."""
    veh = _build(params, front_kin, rear_kin, tire)
    max_g = veh.max_lateral_g()
    bal, uf, ur = veh.balance_index(min(1.2, max_g))
    return dict(max_g=max_g, balance=bal, util_front=uf, util_rear=ur)


def sensitivity(params: VehicleParams, front_kin=None, rear_kin=None, tire=None,
                knobs=None):
    """
    Central-difference sensitivity of grip and balance to each knob.

    Returns a list (sorted by |d max_g| descending) of dicts:
        knob, label, unit, step, d_maxg_per_step, d_balance_per_step
    so the lead can read straight off "camber front: +0.04 g per 0.5°, pushes
    balance -0.03" and prioritise accordingly.
    """
    knobs = knobs or list(PARAM_KNOBS.keys())
    params = _prepare(params, front_kin, rear_kin)
    base = evaluate(params, front_kin, rear_kin, tire)
    rows = []
    for name in knobs:
        spec = PARAM_KNOBS[name]
        step = spec["step"]
        cur = getattr(params, name)
        hi_p = replace(params, **{name: min(cur + step, spec["hi"])})
        lo_p = replace(params, **{name: max(cur - step, spec["lo"])})
        # actual span used (may be clipped at a bound)
        span = getattr(hi_p, name) - getattr(lo_p, name)
        if abs(span) < 1e-9:
            continue
        e_hi = evaluate(hi_p, front_kin, rear_kin, tire)
        e_lo = evaluate(lo_p, front_kin, rear_kin, tire)
        d_g = (e_hi["max_g"] - e_lo["max_g"]) / span * step
        d_b = (e_hi["balance"] - e_lo["balance"]) / span * step
        rows.append(dict(knob=name, label=spec["label"], unit=spec["unit"],
                         step=step, d_maxg_per_step=float(d_g),
                         d_balance_per_step=float(d_b),
                         current=float(cur)))
    rows.sort(key=lambda r: abs(r["d_maxg_per_step"]), reverse=True)
    return dict(base=base, rankings=rows)


def optimise(params: VehicleParams, front_kin=None, rear_kin=None, tire=None,
             knobs=None, target_balance=0.04, balance_tol=0.06, n_grid=5,
             passes=2):
    """
    Coordinate-descent search for max grip with balance held near target.

    target_balance : desired balance index (+ = mild understeer, the fast/safe
        default for an FSAE driver). Setups outside [target-tol, target+tol] are
        penalised so the optimiser doesn't chase grip into a snap-oversteer car.
    n_grid / passes : resolution and refinement. Coarse on purpose — this is a
        transparent search you can audit, not a global optimiser. It returns the
        best setup found AND the path, so you can see the trade it made.

    Returns dict: best_params (as dict of changed knobs), best_eval, improvement
    over the starting setup, and the search history.
    """
    knobs = knobs or list(PARAM_KNOBS.keys())
    params = _prepare(params, front_kin, rear_kin)

    def objective(p):
        e = evaluate(p, front_kin, rear_kin, tire)
        # grip is the prize; balance outside the window is a penalty in g-equivalent
        pen = 0.0
        if abs(e["balance"] - target_balance) > balance_tol:
            pen = 0.5 * (abs(e["balance"] - target_balance) - balance_tol)
        return e["max_g"] - pen, e

    cur = params
    cur_score, cur_eval = objective(cur)
    start_eval = cur_eval
    history = [dict(score=cur_score, **cur_eval)]

    for _ in range(passes):
        for name in knobs:
            spec = PARAM_KNOBS[name]
            grid = np.linspace(spec["lo"], spec["hi"], n_grid)
            best_val, best_score, best_eval = getattr(cur, name), cur_score, cur_eval
            for v in grid:
                trial = replace(cur, **{name: float(v)})
                s, e = objective(trial)
                if s > best_score:
                    best_val, best_score, best_eval = float(v), s, e
            cur = replace(cur, **{name: best_val})
            cur_score, cur_eval = best_score, best_eval
            history.append(dict(knob=name, set_to=best_val, score=cur_score,
                                **cur_eval))

    changed = {k: getattr(cur, k) for k in knobs
               if abs(getattr(cur, k) - getattr(params, k)) > 1e-9}
    return dict(
        best_params=changed,
        best_eval=cur_eval,
        start_eval=start_eval,
        delta_maxg=cur_eval["max_g"] - start_eval["max_g"],
        history=history,
    )
