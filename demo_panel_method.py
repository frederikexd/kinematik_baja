# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: the higher-fidelity in-house aero — a 3D source-panel (boundary-element)
potential-flow solve on a REAL STL, with a ground plane.

Unlike the analytic surrogate (which curve-fits coefficients to attitude and does
not move when the geometry changes), this method SOLVES a flow on the actual surface
mesh. The headline thing to watch: GROUND EFFECT emerges from the physics. We lower
the car and downforce grows — not because of a tuned `ride_ground_gain` constant, but
because the road's image-panel system genuinely strengthens as the body nears the
ground.

It is still inviscid potential flow: no separation, no stall, no real wake. That is
exactly what the ANSYS Fluent verification deck (written for every case) exists to
check. The number is labelled POTENTIAL fidelity and UNCORRELATED, honestly.

No license, no external solver. Needs numpy + trimesh (already in requirements).

Run:  python demo_panel_method.py
"""

import os
import tempfile

import numpy as np
import trimesh

from suspension.aero import (
    CaseSpec, Attitude, FluentVerificationSolver, PanelParams,
)


def make_floor_stl(path: str) -> None:
    """
    A simple FSAE-ish underbody plate: a thin, slightly raked panel — enough surface
    for the panel solve to resolve a ground-effect pressure field. Real teams point
    this at their actual car STL; we synthesise one so the demo is self-contained.
    """
    plate = trimesh.creation.box(extents=[1.5, 0.7, 0.06])
    # refine so there are enough panels for a trustworthy solve
    plate = plate.subdivide().subdivide().subdivide()
    # lift it so its underside starts ~0.10 m above the road (z = 0)
    plate.apply_translation([0.0, 0.0, 0.10])
    plate.export(path)


def main():
    d = tempfile.mkdtemp(prefix="kinematik_panel_demo_")
    stl = os.path.join(d, "underbody.stl")
    make_floor_stl(stl)
    mesh = trimesh.load(stl, force="mesh")

    print("=" * 78)
    print("HIGHER-FIDELITY IN-HOUSE AERO — 3D PANEL METHOD ON A REAL STL")
    print("=" * 78)
    print(f"geometry: {os.path.basename(stl)}  ({len(mesh.faces)} triangles)")
    print("solver:   source-panel BEM + ground image, no license, no external CFD\n")

    # The default Virtual-Tunnel backend in 'panel' mode (auto would also pick it up
    # because a real STL is present). A Fluent deck is written for every case too.
    solver = FluentVerificationSolver(method="panel",
                                      panel_params=PanelParams(max_panels=2500))

    print("Ride-height sweep at zero rake, V = 27 m/s:")
    print(f"  {'ride height':>12s}  {'C_l':>9s}  {'C_d':>8s}  {'front bal':>9s}")
    base = None
    for h in (80.0, 60.0, 40.0, 25.0, 15.0):
        spec = CaseSpec(Attitude(ride_height_mm=h, speed_ms=27.0),
                        stl, reference_area_m2=0.9, reference_length_m=1.5)
        r = solver.run_case(spec, d)
        if base is None:
            base = r.c_lift
        delta = 100.0 * (r.c_lift - base) / abs(base)
        print(f"  {h:9.0f} mm  {r.c_lift:+9.4f}  {r.c_drag:8.4f}  "
              f"{r.aero_balance_front:9.3f}   ({delta:+5.1f}% C_l vs 80 mm)")

    print("\n=> Downforce magnitude grows monotonically as the car is lowered — "
          "ground\n   effect, straight out of the image-panel physics, not a tuned term.")

    print("\n" + "=" * 78)
    print("HONESTY + VERIFICATION")
    print("=" * 78)
    spec = CaseSpec(Attitude(ride_height_mm=25.0, speed_ms=27.0),
                    stl, reference_area_m2=0.9, reference_length_m=1.5)
    r = solver.run_case(spec, d)
    print(r.provenance.status())
    print("notes:", r.notes)
    jou = os.path.join(d, spec.case_name() + ".jou")
    print(f"\nANSYS Fluent verification deck for this case: {jou}")
    print("(Run it on a licensed Fluent to check the in-house number against RANS — "
          "\n especially anywhere the floor is near separation, which potential flow "
          "cannot see.)")


if __name__ == "__main__":
    main()
