# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Demo: PIV flow-field validation — does the CFD separate where the real flow does?

`demo_virtual_windtunnel.py` checks the integrated COEFFICIENTS (C_l/C_d) against the
tunnel. This demo checks the FLOW FIELD itself, which coefficients can't see: a front
wing or undertray can hit its downforce number in CFD with attached flow and stall in
reality. PIV is how you catch that.

The story:
  1. Plan a laser-sheet capture on the car's symmetry plane (XZ), at a known
     attitude/speed, and write the acquisition plan a real rig executes. The offline
     rig refuses to fabricate frames — exactly like the CFD backends refuse to solve
     without a license.
  2. Reduce a pair of laser-pulse frames into a physical velocity field by FFT
     cross-correlation (here the frames are synthesised with a KNOWN flow that
     SEPARATES off the rear of the floor, so you can see the math recover it).
  3. Overlay that measured field on two CFD slices of the same plane:
       A — a CFD that also separates in the same place  -> FIELD MATCHES
       B — a CFD running attached over the floor         -> FIELD MISMATCH, with the
           "CFD attached where the car stalls" diagnostic and a collapsed
           separation IoU.

No laser, no camera, no solver needed: the plan is written for real and the reduction
+ correlation run on synthetic frames/fields you'd otherwise bring back from the lab
and the cluster.

Run:  python demo_piv.py
"""

import os
import tempfile

import numpy as np

from suspension.aero import (
    Attitude, SheetOrientation, LaserSheetPlane, PIVProvenance, FramePair,
    PIVProcessor, AcquisitionPlan, OfflinePIVRig, RigUnavailable, GroundState,
    CFDFieldSlice, correlate_field,
)


# --------------------------------------------------------------------------- #
#  Synthetic frames carrying a KNOWN separated flow (stand-in for the camera)
# --------------------------------------------------------------------------- #
def synth_separated_frames(H=400, W=600, n_particles=4000, dt_us=80.0,
                           mpp=5e-4, freestream=27.0, seed=1):
    """
    Render two laser-pulse frames of a flow that is fast and attached over most of the
    plane but SEPARATES (reverses) in a patch off the rear of the floor — the exact
    thing a front-wing/undertray PIV study goes looking for. Returns the frame pair
    and the true velocity field, so the demo can show the reduction recovering it.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n_particles)
    ys = rng.uniform(0, H, n_particles)

    # Build the true displacement field in pixels. Image x -> +x (rearward); image y
    # is flipped so low image-row = high z (above floor), high row = near floor.
    def true_shift(px, py):
        # freestream rightward shift everywhere
        sx = freestream * (dt_us * 1e-6) / mpp
        sy = 0.0
        # separation bubble: rear (px>0.6W) and near the floor (py>0.7H) -> reversed
        if px > 0.6 * W and py > 0.7 * H:
            sx = -0.3 * sx
        return sx, sy

    sx = np.zeros(n_particles)
    sy = np.zeros(n_particles)
    for i in range(n_particles):
        a, b = true_shift(xs[i], ys[i])
        sx[i] = a
        sy[i] = b

    def render(px, py, sigma=1.6):
        yy, xx = np.mgrid[0:H, 0:W]
        img = np.zeros((H, W))
        for x, y in zip(px, py):
            img += np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
        return img

    frame_a = render(xs, ys)
    frame_b = render(xs + sx, ys + sy)
    return FramePair(frame_a, frame_b, dt_us)


def cfd_slice(plane, xs, ys, freestream, separated):
    """A CFD slice on the SAME (xs,ys) grid as the PIV field; optionally separated."""
    u = np.full_like(xs, freestream)
    v = np.zeros_like(xs)
    if separated:
        # same rear/low-z patch reversed
        bub = (xs > xs.min() + 0.6 * (xs.max() - xs.min())) & \
              (ys < ys.min() + 0.3 * (ys.max() - ys.min()))
        u[bub] = -0.3 * freestream
    return CFDFieldSlice(plane, xs, ys, u, v)


def main():
    freestream = 27.0
    dt_us = 80.0
    mpp = 5e-4
    plane = LaserSheetPlane(SheetOrientation.XZ_SYMMETRY, offset_m=0.0, thickness_mm=1.0)
    attitude = Attitude(pitch_deg=1.0, ride_height_mm=25.0, speed_ms=freestream)

    print("=" * 78)
    print("1) PLAN THE CAPTURE — and watch the offline rig refuse to fake frames")
    print("=" * 78)
    plan = AcquisitionPlan(plane=plane, attitude=attitude, dt_us=dt_us,
                           magnification_m_per_px=mpp, n_pairs=200,
                           particle_diameter_um=1.0, window_px=32,
                           freestream_ms=freestream)
    print(plan.plan_text())
    rig = OfflinePIVRig()
    outdir = tempfile.mkdtemp(prefix="kinematik_piv_demo_")
    path = rig.write_plan(plan, outdir)
    print(f"Wrote acquisition plan: {path}")
    try:
        rig.acquire(plan, outdir)
    except RigUnavailable as e:
        print(f"rig.acquire() refused (correctly): {e}\n")

    print("=" * 78)
    print("2) REDUCE FRAMES -> PHYSICAL VELOCITY FIELD (FFT cross-correlation)")
    print("=" * 78)
    prov = plan.to_provenance("A2 Wind Shear (rolling road)",
                              ground_state=GroundState.MOVING_BELT)
    print(prov.status())
    pair = synth_separated_frames(dt_us=dt_us, mpp=mpp, freestream=freestream)
    piv = PIVProcessor(window_px=64, overlap=0.5, peak_ratio_min=1.05).process(
        pair, prov, attitude=attitude)
    print(piv.status())
    sp = piv.speed()
    print(f"Recovered freestream ~{np.nanmedian(sp[sp > 0.5 * freestream]):.1f} m/s "
          f"(set {freestream:.1f}); reversed-flow windows: "
          f"{int((piv.u < 0).sum())} (the separation bubble)\n")

    print("=" * 78)
    print("3a) OVERLAY ON CFD THAT ALSO SEPARATES — expect FIELD MATCHES")
    print("=" * 78)
    cfd_good = cfd_slice(plane, piv.xs, piv.ys, freestream, separated=True)
    repA = correlate_field(piv, cfd_good)
    print(repA.summary, "\n")

    print("=" * 78)
    print("3b) OVERLAY ON CFD RUNNING ATTACHED — expect FIELD MISMATCH")
    print("=" * 78)
    cfd_bad = cfd_slice(plane, piv.xs, piv.ys, freestream, separated=False)
    repB = correlate_field(piv, cfd_bad)
    print(repB.summary, "\n")

    print("Bottom line: the coefficient correlation in demo_virtual_windtunnel.py "
          "might pass both of these on total downforce. PIV is what tells the aero "
          "lead that case B's diffuser is stalled — before it costs a corner on track.")


if __name__ == "__main__":
    main()
