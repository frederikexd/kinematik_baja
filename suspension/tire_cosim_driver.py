# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Driving a stateful StructuralTireModel from the transient solver.

`transient.py` evaluates the tyre as an *algebraic* block inside each RK4 substep:
it calls `self.tire.lateral.fy(...)` fresh, with no tyre memory except the slip-lag
state it integrates itself. That is correct for a stateless Pacejka, but it is
fundamentally incompatible with a stateful structural tyre, for two reasons:

  1. A structural/thermal tyre OWNS internal state (carcass nodes, gas/tread temps)
     that must advance once per real timestep, monotonically in time. RK4 evaluates
     the derivative four times per step at trial points; you must NOT advance the
     vendor tyre four times, or backwards, per logged step.
  2. The vendor model is the authority on Fz at the patch (from carcass deflection),
     whereas the RK4 tyre takes Fz as an input.

The clean, standard way to couple a stateful subsystem to an explicit ODE
integrator is a STAGGERED (co-simulation) scheme: hold the tyre forces constant as
an external input across one macro-step, advance the vehicle ODE over that step,
then advance the tyre state ONCE using the step-averaged wheel motion, and pass the
new forces to the next macro-step. That is exactly how ADAMS/Car couples FTire/CDTire
to the multibody solver. This module implements that staggered driver for KinematiK's
four corners, around the EXISTING `TransientSolver` primitives, without modifying it.

Honesty note: with the ReferenceTireModel backend the structural/thermal channels
come back None and are logged as absent — the driver does not invent them. They are
populated only when a real FTire/CDTire backend is bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tire_cosim import (StructuralTireModel, WheelState, TireOutput,
                         default_structural_tire)


# corner convention shared with transient.py
FL, FR, RL, RR = 0, 1, 2, 3
CORNER_NAMES = ("FL", "FR", "RL", "RR")


@dataclass
class CosimTireHistory:
    """Per-corner time history of the structural/thermal channels, when available."""
    t: np.ndarray
    Fx: np.ndarray            # (n,4)
    Fy: np.ndarray
    Fz: np.ndarray
    # structural/thermal — arrays of None-or-value, kept as object arrays so an
    # absent channel (reference backend) is represented honestly, not as zero.
    carcass_deflection_m: Optional[np.ndarray] = None
    gas_temp_c: Optional[np.ndarray] = None
    tread_temp_c: Optional[list] = None
    absent_channels: list = field(default_factory=list)
    backend_status: str = ""
    warnings: list = field(default_factory=list)


class CosimCornerSet:
    """
    Four independent StructuralTireModel instances (one per corner) advanced as a
    staggered co-sim. Holds the last-returned forces so the vehicle ODE can read
    them as a constant input across a macro-step, and exposes `advance()` to step
    all four tyre states once per macro-step from the step-averaged wheel motion.

    This is the object a co-sim transient loop carries instead of a single stateless
    tyre. It never raises; backend faults surface in `warnings`.
    """

    def __init__(self, backend_factory=None):
        factory = backend_factory or (lambda: default_structural_tire())
        self.corners: list[StructuralTireModel] = [factory() for _ in range(4)]
        self._last: list[TireOutput] = [TireOutput() for _ in range(4)]
        self.warnings: list[str] = []

    def reset(self, init: Optional[list[WheelState]] = None):
        for i, c in enumerate(self.corners):
            try:
                c.reset(init[i] if init else None)
            except Exception as e:
                self._warn(f"{CORNER_NAMES[i]} reset failed ({type(e).__name__}).")
        self._last = [TireOutput() for _ in range(4)]

    def _warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Last-known (Fx, Fy, Fz) per corner — the constant-input the ODE reads."""
        Fx = np.array([o.Fx for o in self._last])
        Fy = np.array([o.Fy for o in self._last])
        Fz = np.array([o.Fz for o in self._last])
        return Fx, Fy, Fz

    def advance(self, wheel_states: list[WheelState]) -> list[TireOutput]:
        """Advance all four tyre states ONE macro-step; cache and return outputs."""
        outs = []
        for i, c in enumerate(self.corners):
            try:
                o = c.step(wheel_states[i])
            except Exception as e:
                # contract says backends shouldn't raise, but defend anyway
                self._warn(f"{CORNER_NAMES[i]} step raised ({type(e).__name__}); "
                           f"held previous force.")
                o = self._last[i]
            outs.append(o)
        self._last = outs
        # surface backend warnings
        for i, c in enumerate(self.corners):
            for w in c.warnings():
                self._warn(f"{CORNER_NAMES[i]}: {w}")
        return outs

    def status(self) -> str:
        try:
            return self.corners[0].provenance().status()
        except Exception:
            return "unknown backend"

    def absent_channels(self) -> list[str]:
        return list(self._last[0].synthesized) if self._last else []


def run_cosim_maneuver(veh, kind: str = "step_steer",
                       backend_factory=None, params=None, **kw) -> dict:
    """
    Run one named transient manoeuvre with a STAGGERED structural-tyre co-sim.

    This reuses transient.py's manoeuvre builders and its solver for the vehicle
    ODE, but replaces the inline stateless tyre with a CosimCornerSet advanced once
    per macro-step. It demonstrates the boundary end-to-end and returns both the
    vehicle TransientResult and the tyre-side CosimTireHistory (whose structural/
    thermal arrays are populated only if a real backend is bound; with the reference
    backend they are honestly absent).

    Returns a dict {result, tire_history, backend_status, warnings}. Never raises.
    """
    from .transient import (TransientSolver, run_maneuver,  # local import: avoid cycle
                            step_steer_maneuver, snap_oversteer_maneuver,
                            brake_to_throttle_maneuver, curb_strike_maneuver)

    builders = {
        "step_steer": step_steer_maneuver,
        "snap_oversteer": snap_oversteer_maneuver,
        "brake_to_throttle": brake_to_throttle_maneuver,
        "curb_strike": curb_strike_maneuver,
    }
    try:
        builder = builders.get(kind)
        if builder is None:
            return dict(result=None, tire_history=None,
                        backend_status="", warnings=[f"unknown manoeuvre '{kind}'"])
        drv, road, t_end, u0, label = builder(**kw)

        # Vehicle ODE: we still use the existing solver to integrate the chassis,
        # but feed it the co-sim tyre forces by overriding its inner tyre block.
        sim = TransientSolver(veh, params=params)
        corners = CosimCornerSet(backend_factory=backend_factory)
        corners.reset()

        p = sim.p
        # macro-step = the solver's logged dt; tyre advances once per macro-step.
        # We wrap sim.algebraic so the tyre forces it uses come from the co-sim set,
        # advancing the tyre state with the step-averaged wheel motion.
        x_i, y_i = p.corner_xy()
        history = dict(t=[], Fx=[], Fy=[], Fz=[], carcass=[], gas=[], tread=[])

        orig_algebraic = sim.algebraic
        state_box = {"t_prev": 0.0}

        def cosim_algebraic(t, y, driver, roadinp):
            A = orig_algebraic(t, y, driver, roadinp)  # gives loads, slips, kinematics
            dt = max(t - state_box["t_prev"], 0.0) or p.dt
            # build per-corner WheelState from the solver's own algebraic quantities
            u = float(y[0]); v = float(y[1]); r = float(y[2])
            al = A["alpha_lag"]; Fz = A["Fz"]
            ws = []
            for i in range(4):
                vx_w = u - r * y_i[i]
                vy_w = v + r * x_i[i]
                ws.append(WheelState(
                    alpha=float(al[i]), kappa=0.0,
                    gamma=float(sim._cam_f if i in (FL, FR) else sim._cam_r) * float(np.sign(y_i[i])),
                    Fz=float(Fz[i]), v_x=vx_w, v_y=vy_w,
                    z_wheel=float(y[12 + i]), zdot_wheel=float(y[16 + i]),
                    dt=dt))
            outs = corners.advance(ws)
            # override the tyre forces in the algebraic dict with the co-sim result
            A["Fx"] = np.array([o.Fx for o in outs])
            A["Fy"] = np.array([o.Fy for o in outs])
            state_box["t_prev"] = t
            return A

        sim.algebraic = cosim_algebraic
        res = sim.run(t_end, driver=drv, road=road, u0=u0)
        res.meta["maneuver"] = label
        res.meta["cosim_backend"] = corners.status()

        th = CosimTireHistory(
            t=res.t, Fx=res.Fx, Fy=res.Fy, Fz=res.Fz,
            absent_channels=corners.absent_channels(),
            backend_status=corners.status(),
            warnings=list(set(res.warnings) | set(corners.warnings)))
        return dict(result=res, tire_history=th,
                    backend_status=corners.status(),
                    warnings=th.warnings)
    except Exception as e:
        return dict(result=None, tire_history=None, backend_status="",
                    warnings=[f"cosim manoeuvre '{kind}' failed "
                              f"({type(e).__name__}: {e})"])
