# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: the explicit transient time-step DAE solver vs the quasi-steady-state sim.

Runs the four named transient events — step steer, snap-oversteer + recovery,
brake-to-throttle pitch/dive, and a kerb strike — plus the transient-vs-QSS
corner settling comparison, and renders the millisecond-resolution traces that
the QSS point-mass model structurally cannot produce.

Run:  python demo_transient.py   ->  writes transient_demo.png and prints a summary.
"""

import os
import sys
import types
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_engine():
    """Load the engine modules directly, skipping the package __init__ (which
    imports trimesh for the CAD tools). The dynamics/tyre/transient stack needs
    only numpy, so the demo runs in a bare environment."""
    pkg = types.ModuleType("suspension")
    pkg.__path__ = [os.path.join(_ROOT, "suspension")]
    sys.modules.setdefault("suspension", pkg)
    names = ["kinematics", "tiremodel", "dynamics", "damper", "lapsim", "transient"]
    mods = {}
    for m in names:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"suspension.{m}"] = mod
        mods[m] = mod
    for m in names:
        path = os.path.join(_ROOT, "suspension", f"{m}.py")
        spec = importlib.util.spec_from_file_location(f"suspension.{m}", path)
        spec.loader.exec_module(mods[m])
    return mods


_M = _load_engine()
VehicleDynamics = _M["dynamics"].VehicleDynamics
VehicleParams = _M["dynamics"].VehicleParams
default_tire = _M["tiremodel"].default_tire
TR = _M["transient"]


def main():
    veh = VehicleDynamics(VehicleParams(), tire=default_tire())

    print("Running transient manoeuvres (explicit RK4 @ 1 ms)…")
    step = TR.run_maneuver(veh, "step_steer", steer_deg=4.0, u0=18.0)
    snap_c = TR.run_maneuver(veh, "snap_oversteer", recover=True)
    snap_u = TR.run_maneuver(veh, "snap_oversteer", recover=False)
    btt = TR.run_maneuver(veh, "brake_to_throttle", u0=25.0)
    curb = TR.run_maneuver(veh, "curb_strike", u0=20.0, curb_h=0.03, wheels=("FL", "RL"))
    settle = TR.transient_vs_qss_corner(veh, u0=16.0)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("KinematiK — explicit transient DAE solver: the unsteady behaviour QSS assumes away",
                 fontsize=13, fontweight="bold")

    # 1) Step steer: yaw rate overshoot then settle
    a = ax[0, 0]
    a.plot(step.t, np.degrees(step.r), color="#2f6bff", lw=1.5)
    a.axhline(np.degrees(step.r[-1]), color="#999", ls="--", lw=1,
              label=f"steady {np.degrees(step.r[-1]):.0f}°/s")
    a.set_title("Step steer — yaw-rate overshoot & settle")
    a.set_xlabel("time (s)"); a.set_ylabel("yaw rate (°/s)"); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # 2) Snap oversteer: spin vs recovery (sideslip)
    a = ax[0, 1]
    a.plot(snap_u.t, np.degrees(snap_u.beta), color="#ff4444", lw=1.5,
           label="uncaught → spins")
    a.plot(snap_c.t, np.degrees(snap_c.beta), color="#3ec46d", lw=1.5,
           label="feedback countersteer → caught")
    a.set_title("Snap oversteer — divergence vs recovery")
    a.set_xlabel("time (s)"); a.set_ylabel("body sideslip β (°)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # 3) Brake-to-throttle: pitch dive/squat oscillation
    a = ax[0, 2]
    a.plot(btt.t, np.degrees(btt.pitch), color="#a855f7", lw=1.5)
    a.axhline(0, color="#999", lw=0.8)
    a.set_title("Brake → throttle — pitch dive/squat")
    a.set_xlabel("time (s)"); a.set_ylabel("pitch (°)  (− dive / + squat)")
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(btt.t, btt.ax, color="#ff8c1a", lw=0.8, alpha=0.6)
    a2.set_ylabel("long. accel (g)", color="#ff8c1a")

    # 4) Curb strike: contact load spike + wheel lift
    a = ax[1, 0]
    a.plot(curb.t, curb.Fz[:, 0], color="#ff6fb5", lw=1.2, label="FL (struck)")
    a.plot(curb.t, curb.Fz[:, 1], color="#5ec8f2", lw=1.2, label="FR (untouched)")
    a.axhline(0, color="#999", lw=0.8)
    a.set_title("Kerb strike — unsprung hop, load spike & wheel lift")
    a.set_xlabel("time (s)"); a.set_ylabel("contact vertical load Fz (N)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # 5) Curb strike: suspension velocity (high-frequency)
    a = ax[1, 1]
    a.plot(curb.t, curb.susp_vel[:, 0], color="#ff6fb5", lw=1.0)
    a.set_title("Kerb strike — front-left suspension velocity")
    a.set_xlabel("time (s)"); a.set_ylabel("wheel vel (m/s, + bump)")
    a.grid(alpha=0.3)

    # 6) Transient vs QSS: lateral g rise/overshoot/settle
    a = ax[1, 2]
    res = settle.result
    a.plot(res.t, np.abs(res.ay), color="#2f6bff", lw=1.3, label="transient ay")
    a.axhline(settle.steady_ay_g, color="#3ec46d", ls="--", lw=1,
              label=f"transient steady {settle.steady_ay_g:.2f} g")
    a.axhline(settle.qss_max_ay_g, color="#ff4444", ls=":", lw=1,
              label=f"QSS max {settle.qss_max_ay_g:.2f} g")
    a.set_title("Transient vs QSS — the rise QSS skips")
    a.set_xlabel("time (s)"); a.set_ylabel("lateral g")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transient_demo.png")
    fig.savefig(out, dpi=110)
    print(f"\nSaved {out}")

    print("\nHeadline transient metrics (things QSS cannot report):")
    print(f"  step steer    : peak yaw {step.summary()['peak_yaw_rate_deg_s']:.0f}°/s, "
          f"settles to {np.degrees(step.r[-1]):.0f}°/s")
    print(f"  snap (uncaught): final sideslip {np.degrees(snap_u.beta[-1]):.0f}°  (spun)")
    print(f"  snap (caught)  : final sideslip {np.degrees(snap_c.beta[-1]):.1f}°  (recovered)")
    print(f"  brake→throttle : pitch {np.degrees(btt.pitch.min()):.2f}° dive → "
          f"{np.degrees(btt.pitch.max()):.2f}° squat")
    print(f"  kerb strike    : FL load {curb.Fz[:,0].min():.0f}–{curb.Fz[:,0].max():.0f} N "
          f"(wheel lift: {curb.summary()['wheel_lift']})")
    print(f"  corner settling: rise {settle.rise_time_s*1000:.0f} ms, "
          f"settle {settle.settle_time_s*1000:.0f} ms, "
          f"steady {settle.steady_ay_g:.2f} g vs QSS max {settle.qss_max_ay_g:.2f} g")


if __name__ == "__main__":
    main()
