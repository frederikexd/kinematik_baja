# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: surface pressure taps — raw transducer volts -> C_p on the wing -> RMSE vs CFD.

`demo_virtual_windtunnel.py` correlates the integrated COEFFICIENTS; `demo_piv.py`
correlates the off-body flow FIELD. This demo does the on-surface story the run is
uniquely able to tell: it takes the raw matrix of pressure-transducer voltages a
tunnel run actually produces and turns it into a non-dimensional C_p distribution
mapped onto the wing — so you can SEE where the wing is loaded and whether it has
stalled — then RMSEs that distribution against the CFD surface, tap for tap.

The story, on a two-element front wing:
  1. Off the DAQ comes a wall of volts — one column per surface tap, one row per
     sample. KinematiK reduces it to C_p = (p - p_inf)/q against the run's real
     dynamic pressure, mapped onto each tap's (element, x/c, surface).
  2. We read the main element as a wing: its suction peak, its sectional loading,
     and — the thing the coefficient correlation cannot see — whether the suction
     surface is recovering pressure (attached) or sitting on a flat plateau
     (separated/stalled).
  3. We overlay two CFD surface scans on the SAME taps:
       A — CFD that reproduces the measured C_p  -> MATCHED (low RMSE)
       B — CFD that keeps the flap attached where the tunnel shows it stalled ->
           MISMATCHED, with the worst tap pointing right at the stalled aft chord,
           even though the two could integrate to a similar C_l.

No DAQ, no transducer, no solver needed: the raw volts are synthesised from a known
C_p distribution so you can watch the reduction recover it exactly.

Run:  python demo_pressure_tap.py
"""

import numpy as np

from suspension.aero import (
    WingSurface, TapLocation, TapCalibration, ScanProvenance, RawPressureScan,
    CFDSurfaceCp, correlate_cp,
)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# Run conditions
RHO, V, P_INF = 1.225, 25.0, 0.0
SENS, ZERO = 1000.0, 0.05            # 1000 Pa/V transducer span, 0.05 V wind-off tare
Q = 0.5 * RHO * V * V


def synth_volts(cp_value, n=200, seed=0):
    """Volts a transducer would log for a target C_p, with a little turbulence."""
    p = P_INF + cp_value * Q
    v_mean = p / SENS + ZERO
    return v_mean + np.random.default_rng(seed).normal(0.0, 0.0015, n)


# --------------------------------------------------------------------------- #
#  A two-element front wing: main recovers nicely, flap is STALLED (flat aft)
# --------------------------------------------------------------------------- #
xc = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
main_suction  = [-2.6, -2.1, -1.6, -1.1, -0.6, -0.2]   # healthy recovery toward 0
main_pressure = [0.7, 0.45, 0.25, 0.10, 0.0, -0.05]
flap_suction  = [-2.5, -2.32, -2.30, -2.29, -2.31, -2.30]  # FLAT plateau aft = stalled
flap_pressure = [0.5, 0.3, 0.15, 0.05, 0.0, -0.05]

taps, cals, cols, cp_true = [], {}, [], {}
seed = 0
for i, x in enumerate(xc):
    for elem, surf, series in (
        ("main", WingSurface.SUCTION, main_suction),
        ("main", WingSurface.PRESSURE, main_pressure),
        ("flap1", WingSurface.SUCTION, flap_suction),
        ("flap1", WingSurface.PRESSURE, flap_pressure),
    ):
        tid = f"{elem}_{surf.value[0]}{i}"
        taps.append(TapLocation(tid, elem, x, surface=surf))
        cals[tid] = TapCalibration(SENS, ZERO, is_calibrated=True, saturation_v=10.0)
        cp_true[tid] = series[i]
        cols.append(synth_volts(series[i], seed=seed)); seed += 1

# one deliberately uncalibrated channel — it must come back as a HOLE, not a 0
taps.append(TapLocation("main_s_dead", "main", 0.50, surface=WingSurface.SUCTION))
cals["main_s_dead"] = TapCalibration(SENS, ZERO, is_calibrated=False)
cols.append(np.full(200, 1.0))

volts = np.column_stack(cols)


banner("1. RAW SCAN OFF THE DAQ")
scan = RawPressureScan(volts, taps, cals, attitude=None)
print(f"  {scan.n_samples} samples x {scan.n_taps} taps of transducer voltage "
      f"(a {volts.size:,}-number matrix nobody can read by eye)")

prov = ScanProvenance(facility="A2 Wind Shear", rho=RHO, speed_ms=V,
                      p_static_inf_pa=P_INF, sample_rate_hz=1000.0,
                      sample_seconds=8.0, reynolds=2.4e5)
cp = scan.to_cp(prov)

banner("2. REDUCED TO C_p, MAPPED ONTO THE WING")
print("  " + cp.status())
print(f"  (the dead channel reduced to a hole: C_p[main_s_dead] = {cp.cp['main_s_dead']})")
for elem in ("main", "flap1"):
    xcs, cps, _ = cp.chordwise(elem, WingSurface.SUCTION)
    peak_cp, peak_x = cp.suction_peak(elem)
    cn = cp.normal_load_coefficient(elem)
    print(f"\n  [{elem}] suction-surface C_p(x/c):")
    print("    x/c : " + "  ".join(f"{x:5.2f}" for x in xcs))
    print("    C_p : " + "  ".join(f"{c:5.2f}" for c in cps))
    print(f"    suction peak C_p={peak_cp:.2f} at x/c={peak_x:.2f};  "
          f"sectional C_n={cn:.2f}")
    v = cp.stall_indicator(elem)
    flag = "STALLED" if v.stalled else "attached"
    print(f"    -> {flag}: {v.note}")

banner("3a. CORRELATE vs CFD THAT REPRODUCES THE TUNNEL  (expect MATCHED)")
rng = np.random.default_rng(7)
cfd_good = CFDSurfaceCp.from_pairs(
    {tid: cp_true[tid] + rng.normal(0, 0.02) for tid in cp_true},
    backend="OpenFOAM", turbulence_model="kOmegaSST")
report_a = correlate_cp(cp, cfd_good)
print("  " + report_a.summary)

banner("3b. CORRELATE vs CFD THAT KEEPS THE FLAP ATTACHED  (expect MISMATCH on the flap)")
# Same as the tunnel everywhere EXCEPT the flap aft chord, where this CFD predicts a
# healthy recovery (attached) instead of the measured stalled plateau.
cfd_bad_pairs = {}
attached_recovery = {0: -3.0, 1: -2.3, 2: -1.7, 3: -1.1, 4: -0.6, 5: -0.2}
for tid in cp_true:
    if tid.startswith("flap1_s"):
        i = int(tid[-1])
        cfd_bad_pairs[tid] = attached_recovery[i]      # CFD thinks the flap flows
    else:
        cfd_bad_pairs[tid] = cp_true[tid] + rng.normal(0, 0.02)
cfd_bad = CFDSurfaceCp.from_pairs(cfd_bad_pairs, backend="OpenFOAM",
                                  turbulence_model="kOmegaSST (no transition model)")
report_b = correlate_cp(cp, cfd_bad)
print("  " + report_b.summary)
print("\n  Per-tap residuals on the flap suction surface (CFD - phys):")
for r in report_b.residuals:
    if r.paired and r.tap.element == "flap1" and r.tap.surface == WingSurface.SUCTION:
        print(f"    {r.tap.label():42s} ΔC_p = {r.residual:+.3f}")

banner("WHY THIS MATTERS")
print("  Both CFD runs could integrate to nearly the same flap C_l — the attached")
print("  CFD just trades a deep stalled plateau for a shallower attached curve. The")
print("  coefficient correlation in windtunnel.py would not flinch. The C_p RMSE")
print("  does: it localises the disagreement to the flap's aft chord, which is")
print("  exactly where the real wing has separated and the simulated one has not.")
