# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: LOAD the demo wind-tunnel CSVs and run the calibration loop.

This is the consumer side of gen_windtunnel_demo_csv.py. It shows the two CSV roles
being read back through KinematiK's own readers and correlated like-for-like:

  * physical_aero_map.csv          -> the measured tunnel map (the ruler)
  * fluent_runs/<case>_coeffs.csv  -> the per-case ANSYS Fluent verification exports

Point it at the folder the generator wrote (default: windtunnel_demo_csv).

Run:  python demo_load_windtunnel_csv.py [csv_dir]
"""

import csv
import os
import sys

from suspension.aero import windtunnel as wt
from suspension.aero import (
    PhysicalAeroMap, VirtualWindTunnel, FluentVerificationSolver,
)

WHEELBASE_MM = 1550.0
REF_AREA = 1.0
REF_LEN = 1.55


def load_physical_map(path: str) -> PhysicalAeroMap:
    """
    Rebuild a PhysicalAeroMap from the measured CSV. We read the rows directly so the
    map carries proper tunnel provenance (AeroMap.from_csv would load the numbers but
    not the facility/floor/blockage metadata).
    """
    prov = wt.TunnelProvenance(
        facility="A2 Wind Shear (rolling road)",
        ground_state=wt.GroundState.MOVING_BELT,
        model_scale=1.0, blockage_corrected=True, reynolds=4.2e5,
        reference_area_m2=REF_AREA, reference_length_m=REF_LEN,
    )
    pm = PhysicalAeroMap(prov, reference_area_m2=REF_AREA,
                         reference_length_m=REF_LEN, wheelbase_mm=WHEELBASE_MM)
    with open(path) as f:
        for row in csv.DictReader(f):
            front = float(row["ride_height_mm"])
            # recover rear from pitch (rake) so the point lands on the same key
            import math
            pitch = float(row["pitch_deg"])
            rear = front + WHEELBASE_MM * math.tan(math.radians(pitch))
            rh = wt.RideHeights(front, round(rear, 6), float(row["speed_ms"]),
                                WHEELBASE_MM)
            bal = row.get("aero_balance_front") or None
            pm.add_measurement(rh, c_lift=float(row["c_lift"]),
                               c_drag=float(row["c_drag"]),
                               aero_balance_front=float(bal) if bal else None)
    return pm


def main():
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "windtunnel_demo_csv"
    phys_path = os.path.join(csv_dir, "physical_aero_map.csv")
    fluent_dir = os.path.join(csv_dir, "fluent_runs")
    if not os.path.isfile(phys_path):
        sys.exit(f"No physical_aero_map.csv in {csv_dir}. Run "
                 f"gen_windtunnel_demo_csv.py {csv_dir} first.")

    pm = load_physical_map(phys_path)
    print("=" * 78)
    print("LOADED PHYSICAL AERO MAP")
    print("=" * 78)
    print(pm.status())
    print(f"{len(pm)} measured points.\n")

    vwt = VirtualWindTunnel(pm, geometry_path="car.stl", rho=1.225)
    b = FluentVerificationSolver()

    print("=" * 78)
    print("LOADED FLUENT VERIFICATION EXPORTS (per case) AND CORRELATED")
    print("=" * 78)
    results = []
    for spec in vwt.case_specs():
        res = b.read_fluent_csv(spec, fluent_dir)   # reads <case>_coeffs.csv
        results.append(res)
        rh = wt.attitude_to_ride_heights(spec.attitude, WHEELBASE_MM)
        print(f"  {rh.label():42s}  Fluent C_l {res.c_lift:+.3f}  C_d {res.c_drag:+.3f}")

    rep = vwt.correlate(results)
    print()
    print(rep.summary)


if __name__ == "__main__":
    main()
