# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: the live wind-tunnel front end — connect a Virtual Instrument to an under-
floor multi-axis force balance and electronic pressure scanners, sample at kHz,
filter the fan/vibration noise, and stream CLEAN raw forces (F_x,F_y,F_z) and raw
static pressures.

Where the other aero demos START is where this one ENDS. `demo_pressure_tap.py`
begins from a finished matrix of volts; `demo_virtual_windtunnel.py` begins from a
finished C_l/C_d. This demo is the step before either: it is what the test engineer
actually does first — bolt the car to the balance, plumb the taps into the scanners,
connect a VI, and pull clean numbers off contaminated hardware.

The story:
  1. The car sits on a 6-component balance whose bridges CROSS-COUPLE: a pure drag
     load leaks into the lift channel. We decouple it through the interaction matrix.
  2. The whole rig vibrates at the fan blade-pass tone (137 Hz here) plus broadband
     turbulence. A naive average leaves the tone in the mean as a bias; the VI
     notches it out before averaging and REPORTS how much it removed.
  3. Out comes a clean BalanceReading (F_x,F_y,F_z with uncertainty) and a
     RawPressureScan that reduces, with the SAME run's q, to a C_p field on the wing.

No tunnel, no DAQ driver: a synthetic backend builds the contaminated raw streams
from a KNOWN truth (clearly flagged synthetic) so you can watch the VI recover it.
Swap SyntheticDAQ for a nidaqmx/Scanivalve/Chell backend and the same VI runs the
real tunnel.

Run:  python demo_virtual_instrument.py
"""

import numpy as np

from suspension.aero import (
    BalanceCalibration, ForceBalanceSpec, DAQChassis, ScannerVendor,
    PressureScannerSpec, TapLocation, TapCalibration, WingSurface,
    VibrationFilter, StreamingVibrationFilter, AcquisitionSpec, SyntheticDAQ,
    VirtualInstrument,
)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
#  Run conditions and the (unknown-to-the-VI) ground truth
# --------------------------------------------------------------------------- #
RHO, V = 1.225, 25.0
Q = 0.5 * RHO * V * V
FAN_HZ = 137.0           # fan blade-pass tone contaminating everything

TRUE_LOADS = {           # what the balance "should" read, in Newtons / N·m
    "Fx": 180.0,         # drag
    "Fy": -25.0,         # side force
    "Fz": -950.0,        # lift (negative = 950 N of downforce)
    "Mx": 0.0, "My": 12.0, "Mz": 0.0,
}

# A two-element-ish chordwise rake of suction taps with known C_p (so we can see the
# reduction recover it). C_p -> Pa via q.
TAP_CP = {"t00": -3.1, "t01": -2.4, "t02": -1.6, "t03": -0.9, "t04": -0.4}
TAPS = [TapLocation(f"t0{i}", element="main", x_over_c=xc, surface=WingSurface.SUCTION)
        for i, xc in enumerate([0.05, 0.20, 0.40, 0.65, 0.90])]
TRUE_PRESSURES = {t.tap_id: TAP_CP[t.tap_id] * Q for t in TAPS}


# --------------------------------------------------------------------------- #
#  Build the rig
# --------------------------------------------------------------------------- #
banner("RIG — balance with real cross-talk + a Scanivalve scanner")

# 6x6 interaction matrix: diagonal sensitivities plus genuine off-diagonal cross-talk
M = np.diag([220.0, 220.0, 450.0, 60.0, 60.0, 60.0])
M[2, 0] = 35.0     # drag bridge bleeds into the LIFT load — the classic trap
M[4, 0] = 9.0      # and into pitch
balance_cal = BalanceCalibration(matrix=M, is_calibrated=True, serial="B6-001",
                                 saturation_v=10.0)

cals = {t.tap_id: TapCalibration(sensitivity_pa_per_v=1500.0, is_calibrated=True,
                                 saturation_v=5.0, serial=t.tap_id) for t in TAPS}
scanner = PressureScannerSpec(vendor=ScannerVendor.SCANIVALVE, port_taps=TAPS,
                              calibrations=cals, serial="ZOC33-A")

chassis = DAQChassis(sample_rate_hz=2000.0, model="NI cDAQ (synthetic)")

backend = SyntheticDAQ(TRUE_LOADS, TRUE_PRESSURES, fan_hz=FAN_HZ, fan_amp_v=0.06,
                       turb_v=0.008, balance_cal=balance_cal, seed=7)

vi = VirtualInstrument(facility="A2 Wind Shear", chassis=chassis,
                       balance=ForceBalanceSpec(calibration=balance_cal,
                                                mount="under-floor sting", serial="B6-001"),
                       scanners=[scanner], backend=backend)
print(vi.status())


# --------------------------------------------------------------------------- #
#  Acquire WITHOUT the vibration filter — see the bias
# --------------------------------------------------------------------------- #
banner("1) Naive acquisition — no fan/vibration filter")
spec_raw = AcquisitionSpec(seconds=10.0, speed_ms=V, rho=RHO)   # no vibration filter
r_raw = vi.acquire_forces(spec_raw)
print(r_raw.status())
print(f"   true downforce = {-TRUE_LOADS['Fz']:.0f} N; the unfiltered window leaves "
      "the fan tone in the mean.")


# --------------------------------------------------------------------------- #
#  Acquire WITH the filter — clean forces
# --------------------------------------------------------------------------- #
banner("2) Filtered acquisition — notch the 137 Hz fan tone, then decouple + average")
vib = VibrationFilter(sample_rate_hz=2000.0, fan_blade_pass_hz=FAN_HZ,
                      structural_hz=[260.0], n_harmonics=3, aero_cutoff_hz=30.0)
spec = AcquisitionSpec(seconds=10.0, speed_ms=V, rho=RHO,
                       p_static_inf_pa=0.0, vibration=vib)
forces = vi.acquire_forces(spec)
print(forces.status())
print(f"\n   F_x (drag)  = {forces.Fx:+8.1f} N   (true {TRUE_LOADS['Fx']:+.1f})")
print(f"   F_y (side)  = {forces.Fy:+8.1f} N   (true {TRUE_LOADS['Fy']:+.1f})")
print(f"   F_z (lift)  = {forces.Fz:+8.1f} N   (true {TRUE_LOADS['Fz']:+.1f})")
print(f"   downforce   = {forces.downforce_N():+8.1f} N   (true {-TRUE_LOADS['Fz']:+.1f})")
print("\n   The interaction matrix removed the drag->lift cross-talk that a "
      "diagonal-only\n   balance would have reported as ~30 N of phantom lift.")


# --------------------------------------------------------------------------- #
#  Same acquisition -> raw pressures -> C_p on the wing
# --------------------------------------------------------------------------- #
banner("3) Raw static pressures -> C_p field on the wing (same run's q)")
scan = vi.acquire_pressures(spec)
cp = scan.to_cp(vi.scan_provenance(spec))
print(cp.status())
xc, cps, ids = cp.chordwise("main", WingSurface.SUCTION)
print("\n   x/c    C_p(meas)   C_p(true)")
for x, c, tid in zip(xc, cps, ids):
    print(f"   {x:0.2f}    {c:+7.3f}    {TAP_CP[tid]:+7.3f}")
peak_cp, peak_xc = cp.suction_peak("main")
print(f"\n   suction peak C_p = {peak_cp:+.2f} at x/c = {peak_xc:.2f} — the VI recovered "
      "the\n   known distribution out of volts + fan tone + turbulence.")

banner("4) Real-time form — same VI, swap to a streaming biquad cascade")
# The FFT filter in step 2 needs the whole window before it emits a sample (offline
# reduction). A live VI must emit a filtered sample the instant the chassis clocks
# one in — that is a causal biquad cascade. It satisfies the SAME apply() contract,
# so the swap is one line in the AcquisitionSpec; the VI code path is untouched.
spec_rt = AcquisitionSpec(
    seconds=10.0, speed_ms=V, rho=RHO,
    vibration=StreamingVibrationFilter(sample_rate_hz=2000.0, fan_blade_pass_hz=FAN_HZ,
                                       structural_hz=[260.0], n_harmonics=3,
                                       aero_cutoff_hz=30.0))
forces_rt = vi.acquire_forces(spec_rt)
print(forces_rt.status())
print(f"\n   FFT-filtered F_z       = {forces.Fz:+8.1f} N")
print(f"   streaming-filtered F_z = {forces_rt.Fz:+8.1f} N   (same answer, real-time path)")

# And the genuine sample-by-sample primitive a live loop calls on one bridge:
rt = StreamingVibrationFilter(2000.0, fan_blade_pass_hz=FAN_HZ, aero_cutoff_hz=30.0)
rt.reset(0.0)
raw_one_bridge = backend.read_balance(spec_rt, chassis)[:, 2]   # the lift bridge, volts
live = [rt.process_sample(float(v)) for v in raw_one_bridge[:5]]
print("\n   live VI loop — process_sample() per arriving sample (lift bridge, first 5):")
print("   " + "  ".join(f"{v:+.4f}" for v in live) + "  ...  (settling transient)")

banner("Done — clean F_x/F_y/F_z and raw P_static, streamed from contaminated hardware")
