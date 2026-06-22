<!--
  KinematiK — Formula SAE suspension & vehicle dynamics toolkit
  Created by Frederik Thio. Copyright (c) 2026 Frederik Thio.
  Open source. Original author: Frederik Thio, creator of KinematiK.
-->

# PIV flow-field validation (`suspension/aero/piv.py`)

Particle Image Velocimetry support for KinematiK's aero package. Where the Virtual
Wind Tunnel (`windtunnel.py`) correlates the integrated **coefficients** (C_l, C_d,
balance) against the tunnel, PIV correlates the **flow field itself** — so you can
answer the question coefficients can't: *does the CFD separate where the real flow
separates?*

A front wing or undertray can hit its downforce number in CFD with attached flow and
stall in reality. The coefficient correlation passes; the car then behaves nothing
like the sim mid-corner. PIV catches that before it costs a lap.

## What it does

You flood the test section with micron oil mist, fire a double-pulse laser sheet at
the car, and capture two frames a known microsecond-scale delay (Δt) apart. KinematiK
cross-correlates the frames in small interrogation windows to recover, per window, the
particle displacement in pixels; with the image magnification (m/px) and Δt that
becomes a **physical velocity vector** at that point in space, to sub-pixel accuracy.
Do it across the sheet and you have a measured 2D velocity field on the exact plane
your CFD wrote out — ready to overlay.

## The pipeline

```python
from suspension.aero import (
    Attitude, SheetOrientation, LaserSheetPlane, AcquisitionPlan,
    OfflinePIVRig, GroundState, PIVProcessor, CFDFieldSlice, correlate_field,
)

# 1. Plan the capture on the symmetry plane; the offline rig writes a real plan and
#    refuses to fabricate frames (capture them on your LaVision/DaVis-type rig).
plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, offset_m=0.0, thickness_mm=1.0)
plan = AcquisitionPlan(plane, Attitude(pitch_deg=1.0, ride_height_mm=25, speed_ms=27),
                       dt_us=80.0, magnification_m_per_px=5e-4, freestream_ms=27.0)
OfflinePIVRig().write_plan(plan, "/tmp/run")

# 2. Reduce the captured frame pair to a physical velocity field.
prov = plan.to_provenance("A2 Wind Shear", ground_state=GroundState.MOVING_BELT)
field = PIVProcessor(window_px=64, overlap=0.5).process(frame_pair, prov)

# 3. Overlay on a CFD slice of the SAME plane and correlate.
cfd = CFDFieldSlice(plane, xs, ys, u, v)         # exported from Star-CCM+/OpenFOAM
report = correlate_field(field, cfd)
print(report.summary)
```

Run the worked example: `python demo_piv.py`

## What it checks (and the honesty built in)

`correlate_field` reports three things over the windows valid in PIV **and** inside
the CFD extent (everything else is a hole, never extrapolated):

- **`mag_rms_pct`** — RMS in-plane speed error as % of freestream.
- **`angle_rms_deg`** — RMS flow-angle error, computed **only where both fields are
  attached** (inside a recirculation zone a ~180° difference is expected and is scored
  by the IoU below instead).
- **`sep_iou`** — separation-region intersection-over-union: high only when the
  simulated and measured flow detach in the *same place*. This is the headline number
  for your actual question, and a field can pass on magnitude/angle and still **fail**
  here — CFD attached where the car stalls.

Provenance is first-class. `PIVProvenance.status()` flags the three things that decide
whether a measured field is real data or its own setup error:

- **Seeding** — Stokes number check; a heavy tracer lags the flow and rounds off the
  separation you came to measure.
- **Timing** — the freestream particle shift must sit in the window-quarter band
  (`timing_ok()`); too small is noise, too large loses particles out of the window.
- **Ground state** — a fixed floor grows a wrong wall boundary layer; the measured
  underbody field is not ground-effect-true.

## Honesty / non-goals (same discipline as `cfd.py`, `windtunnel.py`)

- The laser/camera/seeder live **outside** KinematiK. `OfflinePIVRig.acquire()` raises
  `RigUnavailable` rather than inventing frames.
- The processor is a **single-pass** FFT cross-correlator and says so: it carries the
  real, well-known window-loss bias (a percent or two inside the quarter-window rule,
  growing with shift). Larger windows read more accurately; the multi-pass
  window-deformation upgrade has a seam but is not implemented, and the provenance
  claims no more than a single pass.
- A disagreement it reports is a real disagreement; a hole is a real hole.

Tests: `python -m pytest tests/test_piv.py` (14 tests — the reduction recovers a known
velocity, the provenance gates fire, attached-vs-separated fails loud, and the two
field-overlay bugs found while wiring it are regression-guarded).
