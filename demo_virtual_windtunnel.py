# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: the Virtual Tunnel Solver — now SELF-CONTAINED inside KinematiK.

There is no external solver to install and no multi-code choice to make. The
Virtual Tunnel Solver computes the aero coefficients IN-HOUSE (the analytic attitude
model) at every matched ride-height point, so you get a usable, honestly-labelled
number with nothing installed: no ANSYS Fluent, no license, no mesh. For every point
it ALSO writes a complete ANSYS Fluent journal — purely so you can independently
VERIFY the in-house number on your own Fluent install whenever you want. The deck is
a confirmation artefact, never a prerequisite.

The story:
  1. Map the physical aero map in the tunnel: C_l/C_d at swept front & rear ride
     heights. Here we fabricate a small but physically-shaped measured map.
  2. Build the matching Virtual Tunnel Solver and let KinematiK ANSWER each matched
     ride-height/speed point in-house, while writing an ANSYS Fluent deck per point.
  3. Correlate the in-house map against the physical tunnel map — the same
     like-for-like calibration loop as before, now runnable end-to-end with no
     external solver in the room.

Run:  python demo_virtual_windtunnel.py
"""

import os
import tempfile

from suspension.aero import windtunnel as wt
from suspension.aero import (
    get_backend, ride_heights_to_attitude, DEFAULT_MEMBER_NAMES,
)


def build_physical_map():
    """A small measured aero map: 3 front x 2 rear ride heights at one tunnel speed."""
    prov = wt.TunnelProvenance(
        facility="A2 Wind Shear (rolling road)",
        ground_state=wt.GroundState.MOVING_BELT,
        model_scale=1.0, blockage_corrected=True, reynolds=4.2e5,
        reference_area_m2=1.0, reference_length_m=1.55,
    )
    pm = wt.PhysicalAeroMap(prov, reference_area_m2=1.0, reference_length_m=1.55,
                            wheelbase_mm=1550.0)
    for front in (18.0, 25.0, 32.0):
        for rear in (40.0, 55.0):
            rh = wt.RideHeights(front, rear, speed_ms=27.0, wheelbase_mm=1550.0)
            # physically-shaped: lower & more rake => more downforce; slightly more drag
            cl = -2.95 + 0.012 * (front - 18.0) - 0.004 * (rear - 40.0)
            cd = 1.04 + 0.0015 * (front - 18.0) + 0.0010 * (rear - 40.0)
            bal = 0.43 + 0.0008 * (rear - front)
            pm.add_measurement(rh, c_lift=cl, c_drag=cd, aero_balance_front=bal)
    return pm


def main():
    pm = build_physical_map()
    print("=" * 78)
    print("PHYSICAL AERO MAP (the tunnel run)")
    print("=" * 78)
    print(pm.status())
    print(f"{len(pm)} measured points over front/rear ride height.\n")

    vwt = wt.VirtualWindTunnel(pm, geometry_path="car.stl", rho=1.225)
    print("=" * 78)
    print("VIRTUAL TUNNEL SOLVER — in-house answer + ANSYS Fluent deck to verify")
    print("=" * 78)
    print(vwt.plan())

    # Self-contained: the default Virtual Tunnel Solver is the in-house Fluent
    # backend. No external solver, license or mesh is needed to get a number. With a
    # real STL it solves a 3D panel/potential-flow field on the geometry (see
    # demo_panel_method.py); with the placeholder geometry here it uses the analytic
    # surrogate and says so.
    vts = get_backend("virtual-tunnel")
    print(f"\nDefault roster (no external solver): {DEFAULT_MEMBER_NAMES}")
    print("In-house method: 'auto' — 3D panel solve when a real STL is supplied, "
          "analytic surrogate otherwise.\n")

    outdir = tempfile.mkdtemp(prefix="kinematik_vts_demo_")
    specs = vwt.case_specs()

    # Answer every matched point in-house; the Fluent deck is written as a side effect.
    cfd_results = [vts.run_case(s, outdir) for s in specs]

    print(f"Answered {len(specs)} matched case(s) IN-HOUSE (no solver run). Each case "
          f"also has an ANSYS Fluent journal written for optional verification:")
    example = os.path.join(outdir, specs[0].case_name(), "fluent")
    files = os.listdir(example) if os.path.isdir(example) else []
    print(f"    {specs[0].case_name()}/fluent/  ->  {', '.join(files[:2])}")
    print("(Run any of those .jou files on a licensed ANSYS Fluent to confirm the "
          "in-house number; KinematiK needs nothing installed to give you the map.)\n")

    ex = cfd_results[0]
    print(f"e.g. {ex.attitude.label()}: in-house C_l {ex.c_lift:+.3f}, "
          f"C_d {ex.c_drag:+.3f}  (converged={ex.converged})")
    print(f"     provenance: {ex.provenance.status()}\n")

    print("=" * 78)
    print("CORRELATION — in-house map vs the physical tunnel map")
    print("=" * 78)
    rep = vwt.correlate(cfd_results)
    print(rep.summary)
    # Report which in-house method actually produced these numbers (the placeholder
    # "car.stl" here is not on disk, so 'auto' falls back to the analytic surrogate
    # and says so; supply a real STL to get the 3D panel solve — see
    # demo_panel_method.py).
    used_panel = any("panel solve" in (r.notes or "") for r in cfd_results)
    method_txt = ("the 3D panel / potential-flow solve on your STL" if used_panel
                  else "the analytic surrogate (no STL on disk for the panel solve)")
    print(f"\nNote: these numbers came from {method_txt}, openly labelled "
          "UNCORRELATED. Treat the absolute levels as provisional and the deltas as "
          "the trustworthy part — and run the written Fluent decks when you want a "
          "licensed-solver (RANS) check of any point, especially near separation, "
          "which potential flow cannot see.")


if __name__ == "__main__":
    main()
