# ============================================================================
#  KinematiK — Formula SAE suspension & vehicle dynamics toolkit
#  Demo: assemble and export the LIVE Formula-car 3D model (no Streamlit needed).
# ============================================================================

"""
Build the dynamic whole-car 3D model from corner geometry + vehicle params + a
populated integration ledger, and write it to interactive HTML. Run again with
different ledger numbers and the car visibly changes — that's the point.

    python demo_fullcar3d.py [output.html]

This uses exactly the renderer the Streamlit "FULL CAR 3D" tab uses
(suspension.fullcar3d.build_full_car_figure).
"""

import sys

from suspension.kinematics import Hardpoints, SuspensionKinematics
from suspension.dynamics import VehicleParams
from suspension.interfaces import blank_ledger, SubsystemInterface
from suspension.fullcar3d import build_full_car_figure, influence_summary


def demo_ledger():
    """A fully-declared car so every subsystem body shows up and is sized."""
    led = blank_ledger()
    led.set(SubsystemInterface(name="aerodynamics", downforce_n_at_v=(900, 20),
                               drag_n_at_v=(320, 20), mass_kg=12,
                               cg_x_mm=1450, cg_y_mm=0, cg_z_mm=260, is_estimate=False))
    led.set(SubsystemInterface(name="powertrain", peak_power_kw=80, peak_torque_nm=160,
                               mass_kg=40, cg_x_mm=1300, cg_y_mm=0, cg_z_mm=210))
    led.set(SubsystemInterface(name="cooling", cooling_airflow_cms=0.8,
                               heat_reject_w=9000, mass_kg=6,
                               cg_x_mm=950, cg_y_mm=0, cg_z_mm=180))
    led.set(SubsystemInterface(name="electrics", env_x_mm=300, env_y_mm=250,
                               env_z_mm=180, mass_kg=22, power_draw_w=400,
                               cg_x_mm=1050, cg_y_mm=0, cg_z_mm=160))
    led.set(SubsystemInterface(name="brakes", brake_torque_nm=1200, mass_kg=8,
                               cg_x_mm=775, cg_y_mm=0, cg_z_mm=220))
    led.set(SubsystemInterface(name="chassis", mass_kg=32,
                               cg_x_mm=900, cg_y_mm=0, cg_z_mm=270))
    led.set(SubsystemInterface(name="suspension", mass_kg=28,
                               cg_x_mm=775, cg_y_mm=0, cg_z_mm=200))
    led.set(SubsystemInterface(name="data-acquisition", mass_kg=2,
                               cg_x_mm=600, cg_y_mm=0, cg_z_mm=300))
    return led


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fullcar.html"
    hp = Hardpoints.default()
    vp = VehicleParams()
    SuspensionKinematics(hp)              # sanity-solve geometry
    led = demo_ledger()

    fig = build_full_car_figure(hp, vp, ledger=led)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print("Live influence of each subsystem on the model:")
    for r in influence_summary(vp, led):
        print(f"  {r['subsystem']:<16} [{r['status']:<7}] {r['detail']}")

    # The suspension follows the chosen ARCHITECTURE. Render the same car with a
    # different topology to show the corners change (members, not just labels).
    try:
        from suspension.topologies import example as topo_example
        from suspension.adapter import GenericKinematics
        for key, fname in (("macpherson_strut", "fullcar_macpherson.html"),
                           ("multilink", "fullcar_multilink.html")):
            kin = GenericKinematics(topo_example(key))
            f2 = build_full_car_figure(corner_front=kin, vp=vp, ledger=led,
                                       topology_label=key)
            f2.write_html(fname, include_plotlyjs="cdn")
            print(f"Wrote {fname}  (suspension = {key})")
    except Exception as e:
        print(f"(topology variants skipped: {e})")

    print("\nChange any number in demo_ledger() — or the topology — and re-run "
          "to see the car move.")


if __name__ == "__main__":
    main()
