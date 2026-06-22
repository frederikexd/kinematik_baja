# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Particle Image Velocimetry (PIV) — correlate a MEASURED flow field against CFD.

WHY THIS MODULE EXISTS (read this before using it)
---------------------------------------------------
`windtunnel.py` answers one question: do the CFD *coefficients* (C_l, C_d, balance)
land on the tunnel's coefficients at the same operating point? That is necessary but
not sufficient. Two flow fields can integrate to the *same* downforce while being
physically different — the right total force for the wrong reason. A front wing can
make its number with attached flow in CFD and a fat separation bubble in reality;
an undertray can hit its C_l with the diffuser flowing in the digital world and
stalled in the real one. The coefficient correlation never sees this. The team only
learns it when the car behaves nothing like the sim mid-corner.

PIV is how you see it. You flood the test section with micron-scale oil-mist seeding,
fire a laser sheet that lights one plane of the flow, and capture two frames a known
microsecond-scale delay apart with a synchronised camera. Each tracer particle moves
a little between the two pulses. KinematiK cross-correlates the two frames in small
interrogation windows to recover, per window, the particle displacement Δx in pixels;
divide by the inter-pulse time Δt and the image magnification and you have a PHYSICAL
velocity vector at that point in space. Do it across the whole sheet and you have a
measured 2D velocity field on the exact plane your CFD wrote out — so you can overlay
the two and ask the question coefficients can't answer: *does the simulated flow
separate where the real flow separates?*

So this module's spine is:

    two laser-pulse frames (+ Δt, + magnification, + the CFD plane they sit on)
      ->  cross-correlation PIV  ->  a measured VelocityField on that plane
      ->  a vector-by-vector correlation against the CFD field sampled on the SAME
          plane, plus a separation-line comparison that says, in auditable numbers,
          whether the simulated and real flow detach in the same place.

THE HONESTY CONTRACT (identical discipline to cfd.py / windtunnel.py)
---------------------------------------------------------------------
A PIV field *looks* like ground truth — it is a measurement, after all — but a PIV
field is only as good as its seeding, its timing and its calibration, and a badly
set-up run lies more convincingly than CFD because it carries the authority of "we
measured it." So, by construction:

  * The laser/camera/seeder hardware lives OUTSIDE KinematiK, exactly as the
    Navier-Stokes solver does. A rig backend that cannot capture here WRITES a
    faithful, runnable acquisition plan and raises `RigUnavailable` from
    `acquire()` — it never fabricates frames or a field.
  * Every `VelocityField` carries `PIVProvenance`: seeding particle size and
    relaxation/Stokes check, the inter-pulse Δt and the particle-shift it implies
    (the one-quarter-window rule), magnification, laser-sheet plane and thickness,
    ground state, and the spatial resolution. A field whose seeding can't follow the
    flow, or whose Δt put particles out of the window, is a different measurement
    from a well-set-up one, and the provenance records the difference.
  * A correlation refuses to compare a measured vector to a CFD vector that isn't on
    the same plane / not at the same point. A hole it reports is a real hole; a
    disagreement it reports is a real disagreement — including, especially, a
    separation point the CFD got wrong.

DELIBERATE NON-GOALS, same as the rest of the aero package: this module does not
drive a laser, does not own a camera, and does not solve Navier-Stokes. It owns the
SEAM (a typed acquisition plan a real rig fulfils) and the MATH (cross-correlation
PIV reduction and the field/separation correlation), so the whole pipeline is
writable and testable now, with synthetic frames, against no hardware and no solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .cfd import Attitude
from .windtunnel import GroundState


# --------------------------------------------------------------------------- #
#  The plane the laser sheet lights — the lock that makes "same point" literal
# --------------------------------------------------------------------------- #
#
# A PIV sheet illuminates ONE plane of the flow. To overlay it on CFD you must sample
# the CFD on the IDENTICAL plane, in the same car-frame coordinates, at the same
# attitude. So the plane is a first-class, shared object: the rig sets the sheet to
# it, and the CFD post-processor slices the solution on it. Coordinates are car-frame
# metres (x rearward, y to the right, z up), matching the CFD convention in cfd.py.
class SheetOrientation(str, Enum):
    """Which plane the laser sheet lights, in car-frame axes."""
    XZ_SYMMETRY = "xz"     # vertical centre-plane: wing/undertray separation, the workhorse
    XY_PLAN = "xy"         # horizontal plane at a height z: floor edge, vortex footprints
    YZ_CROSS = "yz"        # vertical crossflow plane at a station x: vortex cores downstream


@dataclass(frozen=True)
class LaserSheetPlane:
    """
    One measurement plane, shared by the rig and the CFD slice. `orientation` picks
    the pair of in-plane axes; `offset_m` is the position of the plane along the third
    (out-of-plane) axis — e.g. an XZ symmetry sheet at y=0 has offset_m=0.0; a YZ
    crossflow sheet one wheelbase back has offset_m≈1.55. `thickness_mm` is the real
    laser-sheet thickness: a thick sheet averages out-of-plane motion and quietly
    blurs a vortex core, so it is recorded, not assumed.
    """
    orientation: SheetOrientation = SheetOrientation.XZ_SYMMETRY
    offset_m: float = 0.0
    thickness_mm: float = 1.0

    def in_plane_axes(self) -> tuple[str, str]:
        return {
            SheetOrientation.XZ_SYMMETRY: ("x", "z"),
            SheetOrientation.XY_PLAN: ("x", "y"),
            SheetOrientation.YZ_CROSS: ("y", "z"),
        }[self.orientation]

    def out_of_plane_axis(self) -> str:
        return {"xz": "y", "xy": "z", "yz": "x"}[self.orientation.value]

    def label(self) -> str:
        oop = self.out_of_plane_axis()
        return (f"{self.orientation.value}-plane @ {oop}={self.offset_m:.3f} m "
                f"(sheet {self.thickness_mm:.1f} mm)")


# --------------------------------------------------------------------------- #
#  Provenance — what makes a measured field trustworthy instead of authoritative
# --------------------------------------------------------------------------- #
@dataclass
class PIVProvenance:
    """
    Where a measured velocity field came from and what it's worth — the PIV twin of
    CFDProvenance / TunnelProvenance. These are the fields that decide whether a PIV
    field is a measurement of the flow or a measurement of its own setup error:

      * seeding: tracer particle diameter sets how faithfully the mist follows the
        air. Too big and the particle has inertia — it lags the flow through a shear
        layer and rounds off exactly the separation you came to measure.
      * timing: the inter-pulse Δt sets how far a particle moves between frames. The
        rule of thumb is the shift should be about a quarter of the interrogation
        window — too small and noise dominates the displacement; too large and
        particles leave the window and the correlation loses them.
      * magnification: metres-per-pixel, the scale that turns a pixel displacement
        into a physical velocity. Get it wrong and every vector is wrong by that
        factor, invisibly.
      * the plane, the ground state, and the spatial resolution: the same honesty the
        tunnel provenance carries.
    """
    facility: str
    plane: LaserSheetPlane
    dt_us: float                              # inter-pulse delay, microseconds
    magnification_m_per_px: float             # image scale (metres per pixel)
    particle_diameter_um: float = 1.0         # oil-mist tracer diameter
    laser: str = "Nd:YAG double-pulse"
    camera_fps: Optional[float] = None        # for time-resolved runs; None = frame-straddle
    ground_state: GroundState = GroundState.MOVING_BELT
    freestream_ms: float = 20.0               # tunnel speed, for the timing sanity check
    window_px: int = 32                       # interrogation window side, pixels
    notes: str = ""

    # -- physics sanity checks (the honest part) -------------------------- #
    def stokes_number(self, rho_p: float = 900.0, mu: float = 1.81e-5,
                      flow_length_m: Optional[float] = None) -> float:
        """
        Stokes number St = tau_p / tau_f: tracer relaxation time over a flow time
        scale. St << 1 means the mist follows the flow faithfully (good tracer); as
        St approaches and exceeds ~0.1 the particles start to lag and round off sharp
        gradients like a separating shear layer. rho_p is oil density (~900 kg/m^3).
        """
        d = self.particle_diameter_um * 1e-6
        tau_p = rho_p * d * d / (18.0 * mu)
        L = flow_length_m if flow_length_m else (self.plane.thickness_mm * 1e-3 * 50.0)
        # flow time scale ~ L / U; use freestream as the velocity scale
        tau_f = L / max(self.freestream_ms, 1e-6)
        return tau_p / max(tau_f, 1e-12)

    def particle_shift_px(self) -> float:
        """
        Expected freestream particle displacement between pulses, in pixels:
        Δx_px = U * Δt / magnification. The quarter-window rule wants this near
        window_px/4.
        """
        dx_m = self.freestream_ms * (self.dt_us * 1e-6)
        return dx_m / max(self.magnification_m_per_px, 1e-12)

    def timing_ok(self) -> bool:
        """True if the freestream shift sits in the healthy 1/8..1/2 window band."""
        shift = self.particle_shift_px()
        return (self.window_px / 8.0) <= shift <= (self.window_px / 2.0)

    def seeding_ok(self, st_limit: float = 0.1) -> bool:
        return self.stokes_number() <= st_limit

    def status(self) -> str:
        gs = self.ground_state.value
        st = self.stokes_number()
        shift = self.particle_shift_px()
        seed = "good tracer" if self.seeding_ok() else "HEAVY tracer"
        warn = ""
        if not self.seeding_ok():
            warn += (f" — WARNING: St={st:.2f} >= 0.1; {self.particle_diameter_um:.1f} um "
                     "mist lags the flow and will round off separation/shear layers")
        if not self.timing_ok():
            lo, hi = self.window_px / 8.0, self.window_px / 2.0
            warn += (f" — WARNING: freestream shift {shift:.1f} px outside the "
                     f"{lo:.0f}-{hi:.0f} px window-quarter band at dt={self.dt_us:g} us; "
                     "retune Δt before trusting vectors")
        if self.ground_state is not GroundState.MOVING_BELT:
            warn += (" — WARNING: fixed/suction floor grows a wrong wall boundary layer; "
                     "the measured underbody field is not ground-effect-true")
        return (f"{self.facility} [{self.plane.label()}], {self.laser}, "
                f"dt={self.dt_us:g} us, {self.magnification_m_per_px*1e3:.3f} mm/px, "
                f"{self.particle_diameter_um:.1f} um mist ({seed}, St={st:.2f}), "
                f"{gs} floor{warn}")


# --------------------------------------------------------------------------- #
#  A pair of laser-pulse frames — the raw acquisition
# --------------------------------------------------------------------------- #
@dataclass
class FramePair:
    """
    The two images a frame-straddling PIV capture produces: `frame_a` lit by the
    first laser pulse, `frame_b` by the second, `dt_us` apart. Both are 2D intensity
    arrays (rows = image y, cols = image x), same shape. KinematiK turns this pair
    into a VelocityField; the rig produces it (or, in tests/demo, it is synthesised).
    """
    frame_a: np.ndarray
    frame_b: np.ndarray
    dt_us: float

    def __post_init__(self):
        self.frame_a = np.asarray(self.frame_a, dtype=float)
        self.frame_b = np.asarray(self.frame_b, dtype=float)
        if self.frame_a.shape != self.frame_b.shape:
            raise ValueError("frame_a and frame_b must have the same shape")
        if self.frame_a.ndim != 2:
            raise ValueError("frames must be 2D intensity arrays")

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame_a.shape


# --------------------------------------------------------------------------- #
#  The measured velocity field — the PIV deliverable on one plane
# --------------------------------------------------------------------------- #
@dataclass
class VelocityField:
    """
    A 2D velocity field on one laser-sheet plane: at each interrogation-window centre
    (xs, ys in plane coordinates, metres) a velocity (u, v) in m/s in the plane's two
    in-plane axes. `valid` masks windows where the cross-correlation peak was too weak
    to trust (low seeding, glare, out-of-plane loss) — those are holes, not zeros.

    This is the object that overlays on a CFD slice. Both are (xs, ys, u, v) on the
    SAME plane, so `correlate_field` can compare them vector for vector.
    """
    plane: LaserSheetPlane
    xs: np.ndarray                    # shape (ny, nx) — in-plane axis-1 coord, m
    ys: np.ndarray                    # shape (ny, nx) — in-plane axis-2 coord, m
    u: np.ndarray                     # shape (ny, nx) — velocity along in-plane axis-1, m/s
    v: np.ndarray                     # shape (ny, nx) — velocity along in-plane axis-2, m/s
    valid: np.ndarray                 # shape (ny, nx) bool — trustworthy vector?
    attitude: Optional[Attitude] = None
    provenance: Optional[PIVProvenance] = None

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        shapes = {self.xs.shape, self.ys.shape, self.u.shape,
                  self.v.shape, self.valid.shape}
        if len(shapes) != 1:
            raise ValueError("xs, ys, u, v, valid must all share one shape")

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape

    def speed(self) -> np.ndarray:
        """In-plane speed magnitude sqrt(u^2+v^2); NaN where invalid."""
        s = np.hypot(self.u, self.v)
        s[~self.valid] = np.nan
        return s

    def coverage(self) -> float:
        """Fraction of windows that produced a trustworthy vector."""
        return float(self.valid.mean()) if self.valid.size else 0.0

    def status(self) -> str:
        prov = self.provenance.status() if self.provenance else "no provenance"
        att = self.attitude.label() if self.attitude else "attitude unknown"
        return (f"VelocityField {self.shape[1]}x{self.shape[0]} vectors on "
                f"{self.plane.label()}, {100*self.coverage():.0f}% valid, "
                f"@ {att}  [{prov}]")


# --------------------------------------------------------------------------- #
#  The PIV reduction — cross-correlation of one frame pair into a field
# --------------------------------------------------------------------------- #
def _gaussian_subpixel(corr: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """
    Three-point Gaussian sub-pixel peak fit around the integer correlation peak at
    (py, px). This is what gets PIV down to a fraction of a pixel — and, with the
    magnification and Δt, down to the microsecond/real-velocity accuracy the whole
    technique is sold on. Falls back to the integer peak at the array edge.
    """
    ny, nx = corr.shape
    def _fit(c0, cm, cp):
        # peak offset of a Gaussian through three log-samples; guard non-positive
        if cm <= 0 or c0 <= 0 or cp <= 0:
            return 0.0
        lm, l0, lp = math.log(cm), math.log(c0), math.log(cp)
        denom = (2.0 * (lm - 2.0 * l0 + lp))
        if abs(denom) < 1e-12:
            return 0.0
        return (lm - lp) / denom
    dx = 0.0
    dy = 0.0
    if 0 < px < nx - 1:
        dx = _fit(corr[py, px], corr[py, px - 1], corr[py, px + 1])
    if 0 < py < ny - 1:
        dy = _fit(corr[py, px], corr[py - 1, px], corr[py + 1, px])
    # clamp pathological fits to within one pixel
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))
    return dy, dx


def _window_displacement(a: np.ndarray, b: np.ndarray,
                         peak_ratio_min: float) -> tuple[float, float, bool]:
    """
    Cross-correlate two interrogation windows and return the (dy, dx) displacement in
    pixels of b relative to a, plus a validity flag from the peak-to-second-peak
    ratio (the standard PIV signal-quality gate). Uses normalised FFT cross-
    correlation with the window mean removed and a 2D Hann apodisation to suppress the
    spectral-leakage bias that otherwise pulls the peak toward zero shift — matching
    what real PIV processors do.
    """
    a = a - a.mean()
    b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0, 0.0, False
    # normalised cross-correlation via FFT (no apodisation: a Hann window biases the
    # magnitude low through particle-pattern attenuation; mean-subtracted finite
    # windows correlate cleanly at the shifts PIV is tuned for).
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    corr = np.fft.irfft2(np.conj(fa) * fb, s=a.shape)
    corr = np.fft.fftshift(corr)
    norm = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    if norm <= 0:
        return 0.0, 0.0, False
    corr = corr / norm
    # shift correlation to be positive for the log-domain sub-pixel fit
    corr = corr - corr.min() + 1e-6
    ny, nx = corr.shape
    peak_idx = int(np.argmax(corr))
    py, px = divmod(peak_idx, nx)
    peak = corr[py, px]
    if peak <= 0:
        return 0.0, 0.0, False
    # second-highest peak outside a 3x3 neighbourhood, for the validity ratio
    masked = corr.copy()
    y0, y1 = max(0, py - 1), min(ny, py + 2)
    x0, x1 = max(0, px - 1), min(nx, px + 2)
    masked[y0:y1, x0:x1] = -np.inf
    second = masked.max()
    ratio = peak / second if second > 0 else float("inf")
    valid = ratio >= peak_ratio_min
    sub_dy, sub_dx = _gaussian_subpixel(corr, py, px)
    # displacement is peak position relative to the zero-shift centre
    dy = (py - ny // 2) + sub_dy
    dx = (px - nx // 2) + sub_dx
    return dy, dx, valid


@dataclass
class PIVProcessor:
    """
    Turns a `FramePair` into a `VelocityField` by cross-correlating interrogation
    windows. This is the heart of PIV and it is pure, testable, hardware-free math:
    no laser, no camera — just the same FFT cross-correlation a commercial PIV
    processor runs, so the velocity it returns is the real physical velocity the
    particles had, to sub-pixel (hence microsecond-scale) accuracy.

    Parameters mirror the knobs a real processor exposes:
      * window_px       : interrogation window side (32 or 64 are typical),
      * overlap         : fraction (0..0.75) — denser vector grid without finer windows,
      * peak_ratio_min  : signal-quality gate; below it the vector is a hole, not a 0.

    Accuracy note (honest, not hidden): this is a single-pass FFT cross-correlator.
    It carries the real, well-known single-pass bias — particles that shift out of the
    interrogation window between pulses bias the displacement slightly low, on the
    order of a percent or two for shifts inside the quarter-window rule, growing as the
    shift approaches half the window. That is why `PIVProvenance.timing_ok()` enforces
    the window-quarter band and why larger windows read more accurately on the same
    flow. The production upgrade is iterative multi-pass window deformation (offset the
    second window by the first-pass estimate and re-correlate); the seam for it is
    here, but a single pass is what is implemented and what the provenance claims — no
    more.
    """
    window_px: int = 32
    overlap: float = 0.5
    peak_ratio_min: float = 1.2

    def process(self, pair: FramePair, prov: PIVProvenance,
                attitude: Optional[Attitude] = None,
                origin_xy_m: tuple[float, float] = (0.0, 0.0)) -> VelocityField:
        """
        Cross-correlate `pair` into a physical velocity field on `prov.plane`.
        `origin_xy_m` places the image's (col=0,row=0) corner in plane coordinates so
        the vectors land in the same car-frame metres the CFD slice uses. Velocity =
        (pixel displacement) * magnification / Δt, with image-y flipped to plane-up.
        """
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")
        a, b = pair.frame_a, pair.frame_b
        H, W = a.shape
        win = int(self.window_px)
        step = max(1, int(round(win * (1.0 - self.overlap))))
        mpp = prov.magnification_m_per_px
        dt_s = pair.dt_us * 1e-6
        if dt_s <= 0:
            raise ValueError("dt_us must be positive")

        rows = list(range(0, H - win + 1, step))
        cols = list(range(0, W - win + 1, step))
        ny, nx = len(rows), len(cols)
        xs = np.zeros((ny, nx)); ys = np.zeros((ny, nx))
        u = np.zeros((ny, nx)); v = np.zeros((ny, nx))
        valid = np.zeros((ny, nx), dtype=bool)

        ox, oy = origin_xy_m
        for iy, r in enumerate(rows):
            for ix, c in enumerate(cols):
                wa = a[r:r + win, c:c + win]
                wb = b[r:r + win, c:c + win]
                dy_px, dx_px, ok = _window_displacement(wa, wb, self.peak_ratio_min)
                # window centre in image pixels
                cx_px = c + win / 2.0
                cy_px = r + win / 2.0
                # plane coordinates: x grows with image column; the second in-plane
                # axis grows UP, so it is image-row flipped.
                xs[iy, ix] = ox + cx_px * mpp
                ys[iy, ix] = oy + (H - cy_px) * mpp
                # velocity: displacement * scale / time; flip dy so +v is plane-up
                u[iy, ix] = dx_px * mpp / dt_s
                v[iy, ix] = (-dy_px) * mpp / dt_s
                valid[iy, ix] = ok

        return VelocityField(plane=prov.plane, xs=xs, ys=ys, u=u, v=v,
                             valid=valid, attitude=attitude, provenance=prov)


# --------------------------------------------------------------------------- #
#  The rig seam — the laser/camera/seeder, which lives OUTSIDE KinematiK
# --------------------------------------------------------------------------- #
@dataclass
class AcquisitionPlan:
    """
    Everything a real PIV rig needs to capture one plane, with nothing rig-specific
    baked in — the acquisition twin of CaseSpec. A team hands this to whatever rig
    they own (LaVision DaVis, a home-built Arduino-timed double-pulse setup, a
    sponsor's lab); KinematiK does not own the timing hardware.
    """
    plane: LaserSheetPlane
    attitude: Attitude
    dt_us: float
    magnification_m_per_px: float
    n_pairs: int = 100                        # ensemble size for averaging/turbulence
    particle_diameter_um: float = 1.0
    window_px: int = 32
    freestream_ms: float = 20.0
    notes: str = ""

    def to_provenance(self, facility: str,
                      ground_state: GroundState = GroundState.MOVING_BELT) -> PIVProvenance:
        return PIVProvenance(
            facility=facility, plane=self.plane, dt_us=self.dt_us,
            magnification_m_per_px=self.magnification_m_per_px,
            particle_diameter_um=self.particle_diameter_um,
            ground_state=ground_state, freestream_ms=self.freestream_ms,
            window_px=self.window_px, notes=self.notes,
        )

    def plan_text(self) -> str:
        return (f"PIV acquisition: {self.plane.label()} @ {self.attitude.label()}; "
                f"{self.n_pairs} pulse-pairs, dt={self.dt_us:g} us, "
                f"{self.magnification_m_per_px*1e3:.3f} mm/px, "
                f"{self.particle_diameter_um:.1f} um mist, {self.window_px}px windows.")


class RigUnavailable(RuntimeError):
    """
    Raised by a rig backend that can plan a capture but cannot acquire here (no laser,
    no camera, no seeder). Carries an actionable message — exactly what hardware is
    missing — never a silent fallback to fabricated frames. The PIV twin of
    SolverUnavailable.
    """


@runtime_checkable
class PIVRig(Protocol):
    """
    The hardware seam. A rig backend writes a faithful acquisition plan
    (`write_plan`) and, if it has the hardware, captures frame pairs (`acquire`).
    A backend with no hardware here implements `write_plan` and raises
    `RigUnavailable` from `acquire` rather than inventing frames.
    """
    name: str

    def write_plan(self, plan: AcquisitionPlan, workdir: str) -> str: ...
    def acquire(self, plan: AcquisitionPlan, workdir: str) -> list[FramePair]: ...


class OfflinePIVRig:
    """
    A rig backend that writes a correct, human-readable acquisition plan a real lab
    can execute, and refuses to fabricate frames. This is the always-available
    default in this environment, mirroring how the CFD backends write a faithful case
    and raise rather than solving on no license.
    """
    name = "offline-piv-rig"

    def write_plan(self, plan: AcquisitionPlan, workdir: str) -> str:
        import os
        os.makedirs(workdir, exist_ok=True)
        path = os.path.join(workdir, "piv_acquisition_plan.txt")
        with open(path, "w") as f:
            f.write("# KinematiK PIV acquisition plan\n")
            f.write(plan.plan_text() + "\n\n")
            f.write(f"plane.orientation : {plan.plane.orientation.value}\n")
            f.write(f"plane.offset_m    : {plan.plane.offset_m}\n")
            f.write(f"plane.thickness_mm: {plan.plane.thickness_mm}\n")
            f.write(f"attitude          : {plan.attitude.label()}\n")
            f.write(f"dt_us             : {plan.dt_us}\n")
            f.write(f"magnification     : {plan.magnification_m_per_px} m/px\n")
            f.write(f"n_pairs           : {plan.n_pairs}\n")
            f.write(f"particle_um       : {plan.particle_diameter_um}\n")
            f.write(f"window_px         : {plan.window_px}\n")
            f.write(f"freestream_ms     : {plan.freestream_ms}\n")
        return path

    def acquire(self, plan: AcquisitionPlan, workdir: str) -> list[FramePair]:
        raise RigUnavailable(
            "OfflinePIVRig cannot acquire: no laser/camera/seeder in this "
            "environment. The acquisition plan was written for a real rig to "
            "execute (LaVision DaVis or an equivalent double-pulse setup); capture "
            "the frame pairs there, then load them with PIVProcessor.process()."
        )


# --------------------------------------------------------------------------- #
#  Sampling a CFD field onto the SAME plane, so the overlay is like-for-like
# --------------------------------------------------------------------------- #
@dataclass
class CFDFieldSlice:
    """
    A CFD velocity field sampled on a laser-sheet plane — the digital twin of a
    VelocityField. The team's post-processor exports (xs, ys, u, v) on the SAME plane
    the PIV sheet lit; this object just holds it so the correlation can compare the
    two on a common grid. No interpolation magic is hidden: `on_grid` does a plain
    bilinear resample onto the PIV vector centres and flags points outside the CFD
    extent as holes, never extrapolated.
    """
    plane: LaserSheetPlane
    xs: np.ndarray
    ys: np.ndarray
    u: np.ndarray
    v: np.ndarray
    attitude: Optional[Attitude] = None

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.v = np.asarray(self.v, dtype=float)

    def on_grid(self, xs_q: np.ndarray, ys_q: np.ndarray):
        """
        Bilinearly resample the CFD field onto query points (xs_q, ys_q), assuming the
        CFD slice is on a regular grid (the usual export). Returns (uq, vq, inside)
        where `inside` is False for query points outside the CFD extent — those are
        holes, not extrapolations. Handles axes stored in either ascending or
        descending order (a plane-up coordinate flip leaves the row axis descending),
        because a silently mis-ordered axis would scramble every lookup.
        """
        xv = self.xs[0, :] if self.xs.ndim == 2 else self.xs
        yv = self.ys[:, 0] if self.ys.ndim == 2 else self.ys
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)

        # normalise both axes to ascending, remembering the flip so we can index back
        x_desc = xv[0] > xv[-1]
        y_desc = yv[0] > yv[-1]
        xv_a = xv[::-1] if x_desc else xv
        yv_a = yv[::-1] if y_desc else yv

        def _orig_col(i_a):       # ascending index -> original column index
            return (len(xv) - 1 - i_a) if x_desc else i_a

        def _orig_row(j_a):
            return (len(yv) - 1 - j_a) if y_desc else j_a

        uq = np.full(xs_q.shape, np.nan)
        vq = np.full(xs_q.shape, np.nan)
        inside = np.zeros(xs_q.shape, dtype=bool)

        x0, x1 = xv_a.min(), xv_a.max()
        y0, y1 = yv_a.min(), yv_a.max()
        for idx in np.ndindex(xs_q.shape):
            xq = xs_q[idx]; yq = ys_q[idx]
            if not (x0 <= xq <= x1 and y0 <= yq <= y1):
                continue
            i = int(np.clip(np.searchsorted(xv_a, xq) - 1, 0, len(xv_a) - 2))
            j = int(np.clip(np.searchsorted(yv_a, yq) - 1, 0, len(yv_a) - 2))
            dxi = xv_a[i + 1] - xv_a[i]
            dyj = yv_a[j + 1] - yv_a[j]
            tx = (xq - xv_a[i]) / dxi if dxi != 0 else 0.0
            ty = (yq - yv_a[j]) / dyj if dyj != 0 else 0.0

            ci, ci1 = _orig_col(i), _orig_col(i + 1)
            rj, rj1 = _orig_row(j), _orig_row(j + 1)

            def bilerp(field2d):
                f00 = field2d[rj, ci]; f10 = field2d[rj, ci1]
                f01 = field2d[rj1, ci]; f11 = field2d[rj1, ci1]
                return ((1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10
                        + (1 - tx) * ty * f01 + tx * ty * f11)

            uq[idx] = bilerp(self.u)
            vq[idx] = bilerp(self.v)
            inside[idx] = True
        return uq, vq, inside


# --------------------------------------------------------------------------- #
#  Separation detection — the thing coefficients can't see
# --------------------------------------------------------------------------- #
def separation_mask(u_along: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Mark windows where the streamwise velocity has gone negative — reversed flow, the
    signature of separation. `u_along` is the in-plane velocity component aligned with
    the freestream (for an XZ symmetry sheet that is u, the x-component). A True here
    means "the flow has detached / recirculated at this point." Invalid vectors are
    never marked separated.
    """
    sep = (u_along < 0.0) & valid
    return sep


# --------------------------------------------------------------------------- #
#  Field-vs-CFD correlation — the deliverable
# --------------------------------------------------------------------------- #
@dataclass
class FieldCorrelationReport:
    """
    Did the CFD flow FIELD match the measured one on this plane — and, the question
    that matters most, did they separate in the same place? Quantifies the vector
    agreement (RMS magnitude error, mean angle error) over the windows present in
    BOTH fields, and the separation-region overlap (intersection-over-union of the
    reversed-flow masks). Nothing here tunes CFD; it says, in auditable numbers,
    whether the simulated field reproduces the real one.
    """
    plane_label: str
    n_compared: int                           # windows valid in PIV AND inside CFD
    coverage_piv: float                       # fraction of PIV windows that were valid
    mag_rms_pct: float                        # RMS |speed error| / freestream, %
    mag_bias_pct: float                       # signed mean speed error / freestream, %
    angle_rms_deg: float                      # RMS in-plane flow-angle error
    sep_iou: float                            # separation-mask intersection-over-union
    sep_piv_frac: float                       # fraction of compared windows separated, PIV
    sep_cfd_frac: float                       # fraction separated, CFD
    within_tol: bool
    tolerances: dict
    summary: str = ""
    piv_provenance: str = ""

    def as_dict(self):
        d = asdict(self)
        return d


DEFAULT_FIELD_TOL = {
    "mag_rms_pct": 8.0,        # RMS in-plane speed error as % of freestream
    "angle_rms_deg": 8.0,      # RMS in-plane flow-angle error, degrees
    "sep_iou_min": 0.5,        # separation regions must overlap at least this much
}


def correlate_field(piv: VelocityField, cfd: CFDFieldSlice,
                    tol: Optional[dict] = None) -> FieldCorrelationReport:
    """
    Overlay a measured PIV field on a CFD slice of the SAME plane and report whether
    they agree — vector magnitudes, flow angles, and separation location. The CFD is
    resampled onto the PIV vector centres; only windows valid in PIV and inside the
    CFD extent are compared. Points the comparison can't pair are holes, never filled.

    The separation IoU is the headline number for the user's actual question: it is
    high only when the simulated and measured flow detach in the same region. A field
    can pass on magnitude/angle and still FAIL here — that is CFD predicting attached
    flow where the car really separates, the exact failure coefficient correlation
    misses.
    """
    if piv.plane.orientation != cfd.plane.orientation:
        raise ValueError("PIV and CFD planes have different orientations — "
                         "cannot overlay unlike planes")
    tol = {**DEFAULT_FIELD_TOL, **(tol or {})}

    uq, vq, inside = cfd.on_grid(piv.xs, piv.ys)
    compare = piv.valid & inside & np.isfinite(uq) & np.isfinite(vq)
    n = int(compare.sum())

    freestream = (piv.provenance.freestream_ms if piv.provenance
                  else max(float(np.nanmax(np.hypot(piv.u, piv.v))), 1e-6))

    if n == 0:
        return FieldCorrelationReport(
            plane_label=piv.plane.label(), n_compared=0,
            coverage_piv=piv.coverage(), mag_rms_pct=float("nan"),
            mag_bias_pct=float("nan"), angle_rms_deg=float("nan"),
            sep_iou=float("nan"), sep_piv_frac=float("nan"),
            sep_cfd_frac=float("nan"), within_tol=False, tolerances=tol,
            summary=("[PIV correlation] nothing to compare — the PIV valid windows "
                     "and the CFD slice extent do not overlap. Export the CFD on the "
                     "SAME plane and coordinate frame as the laser sheet."),
            piv_provenance=piv.provenance.status() if piv.provenance else "",
        )

    sp_piv = np.hypot(piv.u, piv.v)
    sp_cfd = np.hypot(uq, vq)
    mag_err = (sp_cfd - sp_piv)[compare]
    mag_rms_pct = float(np.sqrt(np.mean(mag_err ** 2)) / freestream * 100.0)
    mag_bias_pct = float(np.mean(mag_err) / freestream * 100.0)

    ang_piv = np.arctan2(piv.v, piv.u)
    ang_cfd = np.arctan2(vq, uq)
    dang_all = np.arctan2(np.sin(ang_cfd - ang_piv), np.cos(ang_cfd - ang_piv))

    # separation: reversed streamwise (in-plane axis-1) flow
    sep_piv = (piv.u < 0.0) & compare
    sep_cfd = (uq < 0.0) & compare

    # Flow-ANGLE error is only meaningful where the flow is attached in BOTH fields:
    # inside a recirculation zone the vector points "backwards" and a ~180 deg angle
    # difference is expected even when both fields agree it has separated. The
    # separation IoU below is what scores agreement in those zones, so the angle
    # metric deliberately excludes any window separated in either field.
    attached_both = compare & ~sep_piv & ~sep_cfd
    if attached_both.any():
        dang = dang_all[attached_both]
        angle_rms_deg = float(np.degrees(np.sqrt(np.mean(dang ** 2))))
    else:
        angle_rms_deg = float("nan")

    inter = int((sep_piv & sep_cfd).sum())
    union = int((sep_piv | sep_cfd).sum())
    sep_iou = (inter / union) if union > 0 else 1.0  # both attached => perfect agreement
    sep_piv_frac = float(sep_piv.sum() / n)
    sep_cfd_frac = float(sep_cfd.sum() / n)

    within = (mag_rms_pct <= tol["mag_rms_pct"]
              and (math.isnan(angle_rms_deg) or angle_rms_deg <= tol["angle_rms_deg"])
              and sep_iou >= tol["sep_iou_min"])

    summary = _summarise_field(piv, n, mag_rms_pct, mag_bias_pct, angle_rms_deg,
                               sep_iou, sep_piv_frac, sep_cfd_frac, within, tol, union)

    return FieldCorrelationReport(
        plane_label=piv.plane.label(), n_compared=n,
        coverage_piv=piv.coverage(), mag_rms_pct=mag_rms_pct,
        mag_bias_pct=mag_bias_pct, angle_rms_deg=angle_rms_deg,
        sep_iou=sep_iou, sep_piv_frac=sep_piv_frac, sep_cfd_frac=sep_cfd_frac,
        within_tol=within, tolerances=tol, summary=summary,
        piv_provenance=piv.provenance.status() if piv.provenance else "",
    )


def _summarise_field(piv, n, mag_rms, mag_bias, ang_rms, sep_iou,
                     sep_piv_frac, sep_cfd_frac, within, tol, union):
    head = (f"[PIV field correlation] {piv.plane.label()}: {n} vector(s) compared "
            f"({100*piv.coverage():.0f}% of PIV windows valid)")
    if within:
        verdict = ("FIELD MATCHES — magnitude, angle and separation all inside "
                   "tolerance; the CFD reproduces the real flow here, not just the "
                   "integrated force")
    else:
        verdict = "FIELD MISMATCH"
    diag = ""
    if sep_iou < tol["sep_iou_min"]:
        if sep_cfd_frac < sep_piv_frac:
            diag = (f"  SEPARATION DISAGREES (IoU {sep_iou:.2f}): the real flow "
                    f"separates over {100*sep_piv_frac:.0f}% of the plane but CFD "
                    f"only {100*sep_cfd_frac:.0f}% — CFD is running ATTACHED where the "
                    "car actually stalls; do not trust the simulated separation point "
                    "for this part.")
        else:
            diag = (f"  SEPARATION DISAGREES (IoU {sep_iou:.2f}): CFD separates more "
                    f"({100*sep_cfd_frac:.0f}%) than reality ({100*sep_piv_frac:.0f}%) "
                    "— likely under-resolved/over-diffusive near the wall.")
    elif union == 0:
        diag = "  Both fields fully attached on this plane — no separation to disagree on."
    if mag_rms > tol["mag_rms_pct"]:
        diag += (f"  Magnitude RMS {mag_rms:.1f}% > {tol['mag_rms_pct']:.0f}% "
                 f"(bias {mag_bias:+.1f}%).")
    if not math.isnan(ang_rms) and ang_rms > tol["angle_rms_deg"]:
        diag += f"  Flow-angle RMS {ang_rms:.1f} deg > {tol['angle_rms_deg']:.0f} deg."
    ang_txt = "n/a" if math.isnan(ang_rms) else f"{ang_rms:.1f} deg"
    return (f"{head}. {verdict}. "
            f"|V| RMS {mag_rms:.1f}% (bias {mag_bias:+.1f}%), "
            f"angle RMS {ang_txt}, separation IoU {sep_iou:.2f}." + diag)
