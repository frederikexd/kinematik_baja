# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Generate a self-consistent set of DEMO wind-tunnel CSV files and validate that they
load back through KinematiK's own readers (so they are real, not invented).

It produces three things into an output folder:

  1. physical_aero_map.csv
       The measured tunnel deliverable: a front/rear ride-height sweep at one wind
       speed, in the column layout `AeroMap.from_csv` reads
       (roll_deg,pitch_deg,yaw_deg,ride_height_mm,speed_ms,c_lift,c_drag,c_side,
        aero_balance_front,converged). c_lift is NEGATIVE = downforce, KinematiK's
       convention. Load it with PhysicalAeroMap / AeroMap.from_csv.

  2. fluent_runs/<case_name>_coeffs.csv   (one per ride-height point)
       The per-case ANSYS Fluent verification export, in the layout
       `FluentVerificationSolver.read_fluent_csv` / `_read_simple_coeff_csv` read
       (Cl,Cd,Cs,CmPitch,converged). NOTE the vendor convention here: Cl is
       UP-POSITIVE, so downforce shows as a POSITIVE Cl — KinematiK flips the sign on
       read. These are made ~1-2% off the tunnel so the correlation is realistic.

  3. README.txt — what each file is and how to load it.

Run:  python gen_windtunnel_demo_csv.py [output_dir]
"""

import csv
import io
import math
import os
import sys

from suspension.aero import windtunnel as wt
from suspension.aero import (
    Attitude, CaseSpec, PhysicalAeroMap, VirtualWindTunnel,
    FluentVerificationSolver, ride_heights_to_attitude,
)


# --------------------------------------------------------------------------- #
#  The demo operating points (a small but real FRH x RRH map at one speed)
# --------------------------------------------------------------------------- #
WHEELBASE_MM = 1550.0
SPEED_MS = 27.0
FRONTS_MM = (18.0, 25.0, 32.0)
REARS_MM = (40.0, 55.0)
REF_AREA = 1.0
REF_LEN = 1.55


def physical_coeffs(front_mm: float, rear_mm: float):
    """
    A physically-shaped MEASURED map (the 'truth' the tunnel logged). Lower front and
    more rake => more downforce; a little more drag with both. c_lift negative = down.
    """
    cl = -2.95 + 0.012 * (front_mm - 18.0) - 0.004 * (rear_mm - 40.0)
    cd = 1.04 + 0.0015 * (front_mm - 18.0) + 0.0010 * (rear_mm - 40.0)
    bal = 0.43 + 0.0008 * (rear_mm - front_mm)
    return cl, cd, bal


def build_physical_map() -> PhysicalAeroMap:
    prov = wt.TunnelProvenance(
        facility="A2 Wind Shear (rolling road)",
        ground_state=wt.GroundState.MOVING_BELT,
        model_scale=1.0, blockage_corrected=True, reynolds=4.2e5,
        reference_area_m2=REF_AREA, reference_length_m=REF_LEN,
    )
    pm = PhysicalAeroMap(prov, reference_area_m2=REF_AREA,
                         reference_length_m=REF_LEN, wheelbase_mm=WHEELBASE_MM)
    for front in FRONTS_MM:
        for rear in REARS_MM:
            rh = wt.RideHeights(front, rear, speed_ms=SPEED_MS,
                                wheelbase_mm=WHEELBASE_MM)
            cl, cd, bal = physical_coeffs(front, rear)
            pm.add_measurement(rh, c_lift=cl, c_drag=cd, aero_balance_front=bal)
    return pm


# --------------------------------------------------------------------------- #
#  Writers
# --------------------------------------------------------------------------- #
def write_physical_map_csv(pm: PhysicalAeroMap, path: str) -> None:
    """Use the map's OWN serialiser so the file is exactly what from_csv reads."""
    with open(path, "w") as f:
        f.write(pm.to_csv())


def write_fluent_run_csvs(vwt: VirtualWindTunnel, outdir: str,
                          offset_scale: float = 1.018) -> list:
    """
    For every matched ride-height point, write a per-case Fluent verification export
    `<case_name>_coeffs.csv` with the vendor columns Cl,Cd,Cs,CmPitch,converged.

    Vendor convention: Cl UP-POSITIVE, so we report -c_lift (downforce -> +Cl). The
    `offset_scale` puts the 'Fluent' numbers ~1.8% off the tunnel so a correlation is
    realistic rather than trivially perfect.
    """
    os.makedirs(outdir, exist_ok=True)
    written = []
    for spec in vwt.case_specs():
        rh = wt.attitude_to_ride_heights(spec.attitude, WHEELBASE_MM)
        cl_truth, cd_truth, bal = physical_coeffs(rh.front_mm, rh.rear_mm)
        # licensed-solver numbers, slightly off the tunnel:
        cl_fluent_downneg = cl_truth * offset_scale          # our convention (neg=down)
        cd_fluent = cd_truth * (2.0 - offset_scale)          # drag a touch the other way
        cl_vendor = -cl_fluent_downneg                        # vendor up-positive
        cs = 0.0
        cm_pitch = 0.10 + 0.002 * (rh.rear_mm - rh.front_mm)  # a plausible pitching moment
        name = spec.case_name() + "_coeffs.csv"
        path = os.path.join(outdir, name)
        with open(path, "w") as f:
            w = csv.writer(f)
            w.writerow(["Cl", "Cd", "Cs", "CmPitch", "converged"])
            w.writerow([f"{cl_vendor:.4f}", f"{cd_fluent:.4f}",
                        f"{cs:.4f}", f"{cm_pitch:.4f}", 1])
        written.append((spec, path))
    return written


README = """\
KinematiK — DEMO WIND-TUNNEL CSV FILES
======================================

These files are a small, self-consistent example of the two CSV roles in the Virtual
Wind Tunnel workflow. They were generated by gen_windtunnel_demo_csv.py and validated
by loading them back through KinematiK's own readers.

1) physical_aero_map.csv  — THE MEASURED TUNNEL MAP
   A front/rear ride-height sweep at one wind speed ({n_pts} points). Columns:
       roll_deg,pitch_deg,yaw_deg,ride_height_mm,speed_ms,
       c_lift,c_drag,c_side,aero_balance_front,converged
   Sign convention: c_lift NEGATIVE = downforce (KinematiK convention).
   Load it:
       from suspension.aero import AeroMap
       amap = AeroMap.from_csv(open("physical_aero_map.csv").read(),
                               reference_area_m2=1.0, reference_length_m=1.55)
   (For the full tunnel-provenance object, rebuild a PhysicalAeroMap and
    add_measurement() per row — see demo_virtual_windtunnel.py.)

2) fluent_runs/<case>_coeffs.csv  — PER-CASE ANSYS FLUENT VERIFICATION EXPORTS
   One file per ride-height point, named exactly as KinematiK names the case, with
   the columns the Fluent verification deck asks you to export:
       Cl,Cd,Cs,CmPitch,converged
   Sign convention: VENDOR up-positive (Cl POSITIVE = downforce). KinematiK flips the
   sign to its own convention on read. These sit ~1.8% off the tunnel on purpose, so
   a correlation is realistic.
   Load one back against its case:
       from suspension.aero import FluentVerificationSolver, CaseSpec, Attitude
       b = FluentVerificationSolver()
       res = b.read_fluent_csv(spec, "fluent_runs")   # spec.case_name() must match

WHAT THESE ARE FOR
   The physical map is the ruler; the Fluent exports are the digital run at the SAME
   operating points. VirtualWindTunnel.correlate() pairs them like-for-like and tells
   you whether the solver reproduced the tunnel inside tolerance. See
   demo_load_windtunnel_csv.py for a runnable end-to-end load + correlate.
"""


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "windtunnel_demo_csv"
    os.makedirs(outdir, exist_ok=True)
    fluent_dir = os.path.join(outdir, "fluent_runs")

    pm = build_physical_map()
    vwt = VirtualWindTunnel(pm, geometry_path="car.stl", rho=1.225)

    # 1) physical map
    phys_path = os.path.join(outdir, "physical_aero_map.csv")
    write_physical_map_csv(pm, phys_path)

    # 2) per-case Fluent verification exports
    written = write_fluent_run_csvs(vwt, fluent_dir)

    # 3) readme
    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write(README.format(n_pts=len(pm)))

    # ---- validate everything round-trips through the real readers ---------- #
    from suspension.aero import AeroMap
    reloaded = AeroMap.from_csv(open(phys_path).read(),
                                reference_area_m2=REF_AREA,
                                reference_length_m=REF_LEN)
    assert len(reloaded) == len(pm), "physical map row count mismatch on reload"

    b = FluentVerificationSolver()
    for spec, path in written:
        res = b.read_fluent_csv(spec, fluent_dir)
        assert res.c_lift is not None and res.c_lift < 0, \
            f"reloaded Fluent CSV {path} did not give downforce"

    # ---- show the correlation these files produce -------------------------- #
    results = [b.read_fluent_csv(spec, fluent_dir) for spec, _ in written]
    rep = vwt.correlate(results)

    print(f"Wrote demo wind-tunnel CSVs to: {outdir}/")
    print(f"  physical_aero_map.csv          ({len(pm)} measured points)")
    print(f"  fluent_runs/                   ({len(written)} per-case verification CSVs)")
    print(f"  README.txt")
    print()
    print("Validation: all files reload through KinematiK's own readers. Example "
          "correlation of the Fluent exports vs the tunnel map:")
    print("  " + rep.summary)


if __name__ == "__main__":
    main()
