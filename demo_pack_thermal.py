# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
demo_pack_thermal.py — the requested workflow, end to end.

Script a virtual lap (the car pulls hundreds of amps on every corner exit),
predict which lithium-ion cells climb toward their limit FIRST given an airflow
map, then let the tool pick the cooling-fan placement that keeps the hottest cell
coolest — all in software.

Run:  python demo_pack_thermal.py
"""

import numpy as np

from suspension import (
    SuspensionKinematics, Hardpoints, VehicleDynamics, VehicleParams,
    default_tire, EVParams,
    PackLayout, CellParams, AirflowParams, Fan,
    simulate_pack_thermal, optimize_fan_placement, fan_grid_candidates,
)
from suspension.lapsim import LapSimulator, LapSimParams, autocross_track


def ascii_heatmap(grid_c, title):
    lo, hi = float(np.nanmin(grid_c)), float(np.nanmax(grid_c))
    ramp = " .:-=+*#%@"
    print(f"\n{title}  ({lo:.1f}–{hi:.1f} °C)")
    for row in grid_c:
        line = "".join(
            ramp[min(int((c - lo) / max(hi - lo, 1e-6) * (len(ramp) - 1)),
                     len(ramp) - 1)]
            for c in row)
        print("  " + line)


def main():
    # --- the car + a multi-lap autocross (the current source) ---
    kin = SuspensionKinematics(Hardpoints.default())
    veh = VehicleDynamics(VehicleParams(), front_kin=kin, rear_kin=kin,
                          tire=default_tire())
    params = LapSimParams()
    lap = LapSimulator(veh, params).simulate(autocross_track(laps=8))
    print(f"Lap: {lap.lap_time:.2f} s/lap, {lap.top_speed:.1f} m/s top speed")

    # --- the pack: a 6×14 grid is the MODULE we map thermally. The WHOLE pack is
    #     84S6P (~302 V, sets the current the lap demands; per-cell current is
    #     pack/6). Hot-tent ambient. ---
    layout = PackLayout(rows=6, cols=14, series=84, parallel=6,
                        cell=CellParams(), ambient_c=32.0)
    print(f"Pack: {layout.n_cells} cells, "
          f"{layout.pack_nominal_v:.0f} V nominal, ambient {layout.ambient_c:.0f}°C")

    ev = EVParams()
    air = AirflowParams()

    # --- baseline: NO fan. Which cells cook first? ---
    base = simulate_pack_thermal(lap, params, layout=layout, fans=[],
                                 airflow=air, ev=ev, n_laps=8)
    print(f"\nNO FAN: hottest cell r{base.hottest_cell_rc[0]}"
          f"c{base.hottest_cell_rc[1]} reaches {base.hottest_peak_c:.1f}°C; "
          f"{base.breach_count} cell(s) over limit.")
    ascii_heatmap(base.peak_grid_c(), "Peak cell temperature — NO FAN")
    print("\n  " + base.provenance.split(".")[0] + ".")

    # --- let the software choose where the fan goes ---
    cands = fan_grid_candidates(layout, nx=4, ny=2, cfm=160.0, throw_mm=90.0)
    study = optimize_fan_placement(lap, params, cands, layout=layout,
                                   airflow=air, ev=ev, n_laps=8)
    print("\n" + study.summary())

    best = study.best
    ascii_heatmap(best.result.peak_grid_c(),
                  "Peak cell temperature — BEST FAN PLACEMENT")
    drop = base.hottest_peak_c - best.hottest_peak_c
    saved = base.breach_count - best.breach_count
    print(f"\nBest placement: hottest cell {drop:+.1f}°C, and cells over their "
          f"limit cut from {base.breach_count} to {best.breach_count} "
          f"({saved} cells saved) — chosen in software, before cutting metal.")
    print("Conduction ties the very hottest cell to its neighbours, so the win "
          "shows up as FEWER cells breaching, not a big peak drop — and that is "
          "the cell-survival number that actually matters.")


if __name__ == "__main__":
    main()
