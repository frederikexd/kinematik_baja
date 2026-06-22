# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Non-linear joint compliance — the give in the bushings, rod ends and spherical
bearings that the rigid kinematics (and even the link-flex compliance layer)
pretends does not exist.

KinematiK's kinematics solver treats every joint as a mathematically perfect,
zero-play constraint. compliance.py relaxed that one step by letting the LINKS
stretch axially (E·A/L). This module relaxes it the rest of the way: every JOINT
connection can carry its own NON-LINEAR force-vs-displacement curve and a damping
coefficient, so the tool can model what real cars actually have —

  * rubber bushings        : soft, progressively-stiffening (cubic) rate + high loss;
  * polyurethane bushings  : stiffer, less progressive, lower loss;
  * spherical bearings /
    rod ends               : near-rigid but with a small free-play (lash) dead-band
                             — the micro-yield / clearance that lets toe and camber
                             move a fraction of a degree before the joint takes load;
  * arbitrary measured     : a tabulated F-δ curve straight off a rig test.

WHERE THE COMPLIANCE ACTS (and why this is defensible, not hand-waving)
----------------------------------------------------------------------
The member load-path solver (loadpath.py) idealises each link as a PIN-JOINTED
TWO-FORCE MEMBER: it carries — and therefore transmits through its end joints —
force only along its own axis. So each joint sees an axial force equal to the
member tension T, and gives a displacement δ_joint(T) ALONG THE LOAD LINE. A
joint is thus a non-linear spring in SERIES with its link:

    δ_total(T) = δ_link(T) + δ_joint_inboard(T) + δ_joint_outboard(T)

Series springs share the same force; their displacements add. That series total is
exactly the link-length change compliance.py already feeds back into the kinematics
solver, so secondary steering (tie-rod-joint give → toe) and track compliance
(wishbone-joint give → camber and lateral contact-patch shift) come straight out of
the existing solver with no new geometry code.

HONEST SCOPE
------------
  * Static / quasi-static, one load case, consistent with the two-force load path.
  * Joint give is modelled along the load line. A bushing's transverse (off-axis)
    give carries no load under the two-force idealisation, so it is not fed back as
    geometry — model it as a pickup-shift if you need it. This is stated, not hidden.
  * DAMPING does no work in a static solve (velocity = 0). It is stored as a
    first-class property and surfaced two honest ways: an energy-loss-per-cycle
    estimate for a given amplitude/frequency, and a `linearize()` export
    (tangent stiffness + effective viscous rate) that the transient DAE solver
    (transient.py) can consume. The static compliance number does not depend on it.

Units throughout: force N, displacement mm, stiffness N/mm, viscous damping
N·s/mm, loss factor dimensionless, frequency Hz.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

try:                                # PCHIP gives a monotone tabular curve
    from scipy.interpolate import PchipInterpolator
    _HAVE_PCHIP = True
except Exception:                   # pragma: no cover - scipy is a hard dep anyway
    _HAVE_PCHIP = False


_KINDS = ("linear", "cubic", "bilinear", "freeplay", "tabular")


@dataclass
class JointCompliance:
    """
    A non-linear axial force-displacement law (+ optional damping) for ONE joint.

    Build with the factory constructors below rather than the raw fields — they set
    a self-consistent, monotone curve:

        JointCompliance.linear(k)
        JointCompliance.cubic(k1, k3)
        JointCompliance.bilinear(k1, k2, knee_mm)
        JointCompliance.freeplay(lash_mm, k, k_lash=...)
        JointCompliance.tabular(disp_mm, force_N)
        JointCompliance.rubber_bushing(...)        # presets matching real parts
        JointCompliance.polyurethane_bushing(...)
        JointCompliance.spherical_bearing(...)

    The force law force(δ) is guaranteed monotone non-decreasing so displacement(F)
    is single-valued (the free-play dead-band maps F=0 → δ=0 by convention).

    Damping is a rate property: `c_viscous` (N·s/mm) and/or a structural
    `loss_factor` η (dimensionless). Both are inert in the static solve; see the
    module docstring and `energy_loss_per_cycle` / `linearize`.
    """
    kind: str = "linear"
    # parametric coefficients (meaning depends on `kind`)
    k1: float = 0.0          # primary rate (N/mm)
    k3: float = 0.0          # cubic hardening (N/mm^3), `cubic`
    k2: float = 0.0          # second-segment rate (N/mm), `bilinear`
    knee: float = 0.0        # segment break displacement (mm), `bilinear`
    lash: float = 0.0        # free-play half-width (mm), `freeplay`
    k_lash: float = 0.0      # in-band contact stiffness (N/mm), `freeplay`
    # tabular data (sorted, strictly increasing force)
    table_disp: Optional[np.ndarray] = None
    table_force: Optional[np.ndarray] = None
    # damping
    c_viscous: float = 0.0   # viscous damping coefficient (N·s/mm)
    loss_factor: float = 0.0 # structural/hysteretic loss factor η (-)
    label: str = ""

    # filled lazily for the tabular path
    _f_of_d: object = field(default=None, repr=False, compare=False)

    # --------------------------------------------------------------------- #
    #  Constructors
    # --------------------------------------------------------------------- #
    @staticmethod
    def linear(k: float, c_viscous: float = 0.0, loss_factor: float = 0.0,
               label: str = "linear") -> "JointCompliance":
        """A plain linear joint: F = k·δ (k in N/mm). The degenerate baseline."""
        if not (np.isfinite(k) and k > 0):
            raise ValueError("linear joint needs a positive finite rate k (N/mm).")
        return JointCompliance(kind="linear", k1=float(k),
                               c_viscous=float(c_viscous),
                               loss_factor=float(loss_factor), label=label)

    @staticmethod
    def cubic(k1: float, k3: float, c_viscous: float = 0.0,
              loss_factor: float = 0.0, label: str = "cubic") -> "JointCompliance":
        """
        Progressive (stiffening) rate: F = k1·δ + k3·δ³, with k1>0, k3≥0.
        The natural shape of a rubber/elastomer bushing — soft off-centre, firming
        up as it is squeezed. k1 is the small-displacement rate (N/mm); k3 the cubic
        hardening (N/mm³).
        """
        if not (np.isfinite(k1) and k1 > 0):
            raise ValueError("cubic joint needs a positive linear rate k1 (N/mm).")
        if k3 < 0:
            raise ValueError("cubic hardening k3 must be ≥ 0 (a softening joint is "
                             "not monotone / not physical here).")
        return JointCompliance(kind="cubic", k1=float(k1), k3=float(k3),
                               c_viscous=float(c_viscous),
                               loss_factor=float(loss_factor), label=label)

    @staticmethod
    def bilinear(k1: float, k2: float, knee_mm: float, c_viscous: float = 0.0,
                 loss_factor: float = 0.0, label: str = "bilinear") -> "JointCompliance":
        """
        Two-rate curve: rate k1 up to |δ| = knee_mm, then rate k2 (N/mm) beyond.
        Use k2 > k1 for a bushing that hits a bump-stop / packs out, or k2 < k1 for
        a joint that yields. Symmetric in tension/compression.
        """
        for nm, kv in (("k1", k1), ("k2", k2)):
            if not (np.isfinite(kv) and kv > 0):
                raise ValueError(f"bilinear joint needs positive rate {nm} (N/mm).")
        if not (np.isfinite(knee_mm) and knee_mm > 0):
            raise ValueError("bilinear knee_mm must be a positive displacement (mm).")
        return JointCompliance(kind="bilinear", k1=float(k1), k2=float(k2),
                               knee=float(knee_mm), c_viscous=float(c_viscous),
                               loss_factor=float(loss_factor), label=label)

    @staticmethod
    def freeplay(lash_mm: float, k: float, k_lash: float = 0.0,
                 c_viscous: float = 0.0, loss_factor: float = 0.0,
                 label: str = "freeplay") -> "JointCompliance":
        """
        Free-play (lash) dead-band then a stiff rate — the spherical-bearing / rod-end
        micro-yield model. Within ±lash_mm the joint carries almost no load (an
        optional small contact stiffness k_lash, N/mm, avoids a degenerate zero rate
        and represents the real residual contact); beyond the clearance it stiffens to
        k (N/mm). This is the give that lets toe/camber move a fraction of a degree
        before a "rigid" bearing actually takes the load.
        """
        if not (np.isfinite(lash_mm) and lash_mm >= 0):
            raise ValueError("freeplay lash_mm must be ≥ 0 (mm).")
        if not (np.isfinite(k) and k > 0):
            raise ValueError("freeplay needs a positive engaged rate k (N/mm).")
        if k_lash < 0:
            raise ValueError("freeplay in-band stiffness k_lash must be ≥ 0.")
        return JointCompliance(kind="freeplay", lash=float(lash_mm), k1=float(k),
                               k_lash=float(k_lash), c_viscous=float(c_viscous),
                               loss_factor=float(loss_factor), label=label)

    @staticmethod
    def tabular(disp_mm: Sequence[float], force_N: Sequence[float],
                c_viscous: float = 0.0, loss_factor: float = 0.0,
                label: str = "tabular") -> "JointCompliance":
        """
        An arbitrary measured curve. `disp_mm` and `force_N` are matched samples;
        both must be strictly increasing (a monotone joint) and span through the
        origin region so tension and compression are both represented. Interpolated
        with a monotone PCHIP spline and inverted the same way.
        """
        d = np.asarray(disp_mm, float)
        f = np.asarray(force_N, float)
        if d.shape != f.shape or d.ndim != 1 or d.size < 3:
            raise ValueError("tabular joint needs matching 1-D disp/force arrays of "
                             "length ≥ 3.")
        if np.any(np.diff(d) <= 0):
            raise ValueError("tabular displacement samples must be strictly increasing.")
        if np.any(np.diff(f) <= 0):
            raise ValueError("tabular force samples must be strictly increasing "
                             "(a monotone, invertible joint).")
        jc = JointCompliance(kind="tabular", table_disp=d, table_force=f,
                             c_viscous=float(c_viscous),
                             loss_factor=float(loss_factor), label=label)
        return jc                       # __post_init__ builds the interpolators

    # ---- named presets that map to the parts engineers actually fit ------- #
    @staticmethod
    def rubber_bushing(k_radial: float = 1500.0, hardening: float = 8.0,
                       loss_factor: float = 0.12, label: str = "rubber bushing"
                       ) -> "JointCompliance":
        """
        A representative rubber suspension bushing: soft off-centre radial rate
        `k_radial` (N/mm) with strong progressive hardening, and a high structural
        loss factor (rubber dissipates a lot). `hardening` scales the cubic term
        (k3 = hardening · k_radial, N/mm³). These are sensible defaults, not a part
        number — give your own from a rig curve via `cubic` or `tabular`.
        """
        return JointCompliance.cubic(k1=k_radial, k3=hardening * k_radial,
                                     loss_factor=loss_factor, label=label)

    @staticmethod
    def polyurethane_bushing(k_radial: float = 6000.0, hardening: float = 4.0,
                             loss_factor: float = 0.05,
                             label: str = "polyurethane bushing") -> "JointCompliance":
        """
        A polyurethane bushing: markedly stiffer than rubber, less progressive, and
        lower loss (poly is firmer and damps less). Defaults are representative.
        """
        return JointCompliance.cubic(k1=k_radial, k3=hardening * k_radial,
                                     loss_factor=loss_factor, label=label)

    @staticmethod
    def spherical_bearing(lash_mm: float = 0.05, k: float = 120000.0,
                          k_lash: float = 2000.0, loss_factor: float = 0.01,
                          label: str = "spherical bearing") -> "JointCompliance":
        """
        A spherical bearing / rod end: near-rigid engaged rate `k` (N/mm) with a
        small clearance `lash_mm` (typ. 0.02–0.10 mm of radial play, worn higher) and
        a soft in-band contact stiffness `k_lash`. The lash is the micro-yield that
        gives real cars a little toe/camber movement at low load that a perfect joint
        cannot. Very low loss.
        """
        return JointCompliance.freeplay(lash_mm=lash_mm, k=k, k_lash=k_lash,
                                        loss_factor=loss_factor, label=label)

    # --------------------------------------------------------------------- #
    #  Force law and its inverse
    # --------------------------------------------------------------------- #
    def force(self, disp_mm: float) -> float:
        """Axial force (N) the joint carries at displacement δ (mm, + = extension)."""
        d = float(disp_mm)
        s = np.sign(d)
        a = abs(d)
        if self.kind == "linear":
            return self.k1 * d
        if self.kind == "cubic":
            return self.k1 * d + self.k3 * d**3
        if self.kind == "bilinear":
            if a <= self.knee:
                return self.k1 * d
            return s * (self.k1 * self.knee + self.k2 * (a - self.knee))
        if self.kind == "freeplay":
            if a <= self.lash:
                return self.k_lash * d
            return s * (self.k_lash * self.lash + self.k1 * (a - self.lash))
        if self.kind == "tabular":
            return float(self._f_of_d(d))
        raise ValueError(f"unknown joint kind '{self.kind}'")

    def displacement(self, force_N: float) -> float:
        """Displacement δ (mm) at axial force F (N). Inverse of `force`."""
        F = float(force_N)
        if F == 0.0:
            return 0.0
        s = np.sign(F)
        a = abs(F)
        if self.kind == "linear":
            return F / self.k1
        if self.kind == "cubic":
            # one real root of k3 δ³ + k1 δ − F = 0; Newton from the linear seed
            d = F / self.k1
            for _ in range(60):
                g = self.k3 * d**3 + self.k1 * d - F
                dg = 3.0 * self.k3 * d**2 + self.k1
                step = g / dg
                d -= step
                if abs(step) < 1e-12:
                    break
            return d
        if self.kind == "bilinear":
            Fk = self.k1 * self.knee          # force at the knee
            if a <= Fk:
                return F / self.k1
            return s * (self.knee + (a - Fk) / self.k2)
        if self.kind == "freeplay":
            Fl = self.k_lash * self.lash      # force at clearance take-up
            if a <= Fl:
                return F / self.k_lash if self.k_lash > 0 else 0.0
            return s * (self.lash + (a - Fl) / self.k1)
        if self.kind == "tabular":
            return float(self._invert_tabular(F))
        raise ValueError(f"unknown joint kind '{self.kind}'")

    def tangent_stiffness(self, disp_mm: float) -> float:
        """Local slope dF/dδ (N/mm) at displacement δ — the linearised rate."""
        d = float(disp_mm)
        a = abs(d)
        if self.kind == "linear":
            return self.k1
        if self.kind == "cubic":
            return self.k1 + 3.0 * self.k3 * d**2
        if self.kind == "bilinear":
            return self.k1 if a <= self.knee else self.k2
        if self.kind == "freeplay":
            return self.k_lash if a <= self.lash else self.k1
        if self.kind == "tabular":
            return float(self._f_of_d.derivative()(d))
        raise ValueError(f"unknown joint kind '{self.kind}'")

    def secant_stiffness(self, force_N: float) -> float:
        """Secant rate F/δ (N/mm) at a given force — what the series sum 'sees'."""
        F = float(force_N)
        if F == 0.0:
            return self.tangent_stiffness(0.0)
        d = self.displacement(F)
        if d == 0.0:
            return np.inf
        return F / d

    # --------------------------------------------------------------------- #
    #  Damping surfaced honestly
    # --------------------------------------------------------------------- #
    def energy_loss_per_cycle(self, amplitude_mm: float, freq_hz: float) -> float:
        """
        Energy dissipated per oscillation cycle (N·mm = mJ) at sinusoidal amplitude
        `amplitude_mm` and frequency `freq_hz`, combining viscous and structural
        (hysteretic) damping:

            viscous     E = π · c · ω · X²          (ω = 2πf)
            structural  E = π · η · k_sec · X²

        with k_sec the secant stiffness at the amplitude. This is the honest place
        the damping shows up — the static compliance solve does not use it.
        """
        X = abs(float(amplitude_mm))
        if X == 0.0:
            return 0.0
        omega = 2.0 * np.pi * float(freq_hz)
        e_visc = np.pi * self.c_viscous * omega * X**2
        Fx = self.force(X)
        k_sec = abs(Fx / X) if X > 0 else self.tangent_stiffness(0.0)
        e_struct = np.pi * self.loss_factor * k_sec * X**2
        return float(e_visc + e_struct)

    def linearize(self, about_force_N: float = 0.0,
                  freq_hz: Optional[float] = None) -> dict:
        """
        Operating-point linearisation for a dynamic (transient) analysis: the tangent
        stiffness at the current force and an EQUIVALENT viscous rate that folds in
        the structural loss factor at `freq_hz` (c_eq = c + η·k/ω). Returns a dict the
        transient DAE solver can consume directly. Without `freq_hz` the structural
        part can't be turned into a viscous rate, so only the explicit viscous c is
        returned.
        """
        d = self.displacement(about_force_N)
        k_t = self.tangent_stiffness(d)
        c_eq = self.c_viscous
        if freq_hz is not None and freq_hz > 0 and self.loss_factor > 0:
            omega = 2.0 * np.pi * float(freq_hz)
            c_eq = self.c_viscous + self.loss_factor * k_t / omega
        return {"tangent_stiffness_N_per_mm": float(k_t),
                "viscous_c_N_s_per_mm": float(self.c_viscous),
                "equivalent_viscous_c_N_s_per_mm": float(c_eq),
                "loss_factor": float(self.loss_factor),
                "about_force_N": float(about_force_N),
                "displacement_mm": float(d)}

    # --------------------------------------------------------------------- #
    def _build_tabular(self):
        if not _HAVE_PCHIP:
            raise RuntimeError("tabular joints need scipy.interpolate.PchipInterpolator.")
        # Single source of truth: force as a monotone function of displacement.
        # displacement(F) numerically inverts THIS curve, so the two are exact
        # inverses (two independent splines would not be).
        self._f_of_d = PchipInterpolator(self.table_disp, self.table_force,
                                         extrapolate=True)
        self._d_lo = float(self.table_disp[0])
        self._d_hi = float(self.table_disp[-1])
        self._f_lo = float(self.table_force[0])
        self._f_hi = float(self.table_force[-1])
        deriv = self._f_of_d.derivative()
        self._slope_lo = float(deriv(self._d_lo))   # end tangents for extrapolation
        self._slope_hi = float(deriv(self._d_hi))

    def _invert_tabular(self, F: float) -> float:
        """Displacement at force F: bisection inside the table, linear slope outside."""
        if F <= self._f_lo:
            return self._d_lo + (F - self._f_lo) / self._slope_lo
        if F >= self._f_hi:
            return self._d_hi + (F - self._f_hi) / self._slope_hi
        lo, hi = self._d_lo, self._d_hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(self._f_of_d(mid)) < F:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi)

    def __post_init__(self):
        if self.kind == "tabular" and self._f_of_d is None \
                and self.table_disp is not None:
            self._build_tabular()

    def summary(self) -> dict:
        return {"kind": self.kind, "label": self.label,
                "k1_N_per_mm": self.k1, "k3_N_per_mm3": self.k3,
                "k2_N_per_mm": self.k2, "knee_mm": self.knee,
                "lash_mm": self.lash, "k_lash_N_per_mm": self.k_lash,
                "c_viscous_N_s_per_mm": self.c_viscous,
                "loss_factor": self.loss_factor}
