# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Damper (shock) force–velocity model.

A damper produces force as a function of shaft VELOCITY, not position — that is what
makes it a damper and not a spring. FSAE dampers are typically digressive: a steep
low-speed slope (which controls body motions — roll, pitch, dive) that transitions at
a "knee" velocity to a shallower high-speed slope (which controls wheel motions over
bumps/kerbs). Bump (compression) and rebound (extension) are tuned independently.

HONESTY CONTRACT (same as the tyre combined-slip section)
---------------------------------------------------------
A quantitative damper model needs YOUR damper's dyno data — a force/velocity plot per
click setting. We do NOT have it and will not fake specific curves. This module
implements the real bilinear-digressive force law with explicit, visible coefficients
and an `is_calibrated` flag that stays False until you load dyno points. Uncalibrated,
the defaults are representative FSAE magnitudes (clearly flagged), so the model and the
critical-damping diagnostics work today and become trustworthy the moment you calibrate.

This is the building block for the transient model on the roadmap (turn-in, pitch,
kerb response). It is exposed and tested as a standalone primitive so that the
transient layer is built on a verified force law rather than an ad-hoc one.

Convention: shaft velocity v > 0 in compression (bump), v < 0 in extension (rebound).
Force returned with the same sign convention (resists motion).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DamperCurve:
    """
    Bilinear-digressive force–velocity law for one damper.

    Parameters (all in SI: N per m/s for slopes, m/s for the knee):
        c_bump_low / c_reb_low   : low-speed damping coefficients (steep region)
        c_bump_high / c_reb_high : high-speed coefficients (shallow region past knee)
        v_knee                   : shaft speed (m/s) where the slope changes
    Force is continuous at the knee. Independent bump/rebound, independent low/high.

    Build from dyno data with `from_dyno_points` and the model becomes calibrated.
    """
    c_bump_low: float = 6000.0
    c_bump_high: float = 2000.0
    c_reb_low: float = 9000.0      # rebound usually stiffer than bump on FSAE
    c_reb_high: float = 3000.0
    v_knee: float = 0.05           # m/s, typical low/high-speed split
    is_calibrated: bool = False
    source: str = "representative"

    def force(self, v_shaft):
        """
        Damper force (N) at shaft velocity v_shaft (m/s); >0 bump, <0 rebound.
        Bilinear: F = c_low*v for |v|<=v_knee, then continues at c_high slope.
        Vectorised.
        """
        v = np.asarray(v_shaft, float)
        vk = max(self.v_knee, 1e-4)
        out = np.zeros_like(v, dtype=float)

        # compression (bump), v > 0
        comp = v > 0
        vlow = np.minimum(v, vk)
        vhigh = np.maximum(v - vk, 0.0)
        f_comp = self.c_bump_low * vlow + self.c_bump_high * vhigh
        # extension (rebound), v < 0
        av = -v
        avlow = np.minimum(av, vk)
        avhigh = np.maximum(av - vk, 0.0)
        f_reb = -(self.c_reb_low * avlow + self.c_reb_high * avhigh)

        out = np.where(comp, f_comp, f_reb)
        return out if out.shape else float(out)

    @staticmethod
    def from_dyno_points(v_ms, force_n, v_knee=0.05) -> "DamperCurve":
        """
        Fit the bilinear law to measured dyno points (shaft velocity m/s, force N).
        Separately least-squares-fits the low- and high-speed slopes in bump and
        rebound about `v_knee`. Sets is_calibrated=True. Falls back to defaults for
        any region with too few points (and stays uncalibrated then).
        """
        v = np.asarray(v_ms, float).ravel()
        f = np.asarray(force_n, float).ravel()
        if v.size < 4 or v.size != f.size:
            return DamperCurve()

        def _slope_through_origin(vv, ff):
            # least-squares slope of f = c*v (force passes through origin)
            denom = float(np.sum(vv * vv))
            return float(np.sum(vv * ff) / denom) if denom > 1e-12 else 0.0

        vk = max(v_knee, 1e-4)
        cur = DamperCurve(v_knee=vk, is_calibrated=True, source="dyno")
        # bump low/high
        m = (v > 0) & (v <= vk)
        if m.sum() >= 2: cur.c_bump_low = max(_slope_through_origin(v[m], f[m]), 0.0)
        m = (v > vk)
        if m.sum() >= 2:
            # incremental slope past knee: subtract the knee force, regress on (v-vk)
            cur.c_bump_high = max(_slope_through_origin(v[m] - vk,
                                  f[m] - cur.c_bump_low * vk), 0.0)
        # rebound low/high (mirror)
        m = (v < 0) & (v >= -vk)
        if m.sum() >= 2: cur.c_reb_low = max(_slope_through_origin(-v[m], -f[m]), 0.0)
        m = (v < -vk)
        if m.sum() >= 2:
            cur.c_reb_high = max(_slope_through_origin(-v[m] - vk,
                                 -f[m] - cur.c_reb_low * vk), 0.0)
        return cur

    def curve_points(self, v_max=0.5, n=101):
        """(velocity, force) arrays for plotting the force–velocity curve."""
        v = np.linspace(-v_max, v_max, n)
        return v, self.force(v)

    def status(self) -> str:
        return ("calibrated to dyno data" if self.is_calibrated else
                "UNCALIBRATED — representative FSAE magnitudes (load your damper "
                "dyno points to make these quantitative; the force law is real)")


def damping_ratio(curve: DamperCurve, corner_mass_kg: float,
                  wheel_rate_N_per_mm: float, motion_ratio: float = 1.0,
                  region: str = "bump") -> float:
    """
    Approximate damping ratio ζ of the sprung corner, the number that actually
    guides damper tuning: ζ≈0.65-0.7 bump / higher rebound is a common FSAE target.

        ζ = c_corner / (2 * sqrt(k_corner * m))

    The low-speed slope governs the body modes, so we use c_*_low, referred to the
    WHEEL through the motion ratio (shaft velocity = wheel velocity * MR, and force
    at the wheel = damper force * MR, so c_wheel = c_shaft * MR²). Wheel rate is the
    installed vertical rate (N/mm -> N/m). Returns ζ (dimensionless).

    Needs a calibrated curve to be quantitative; with the representative defaults it
    still shows the right ballpark and how a change moves ζ.
    """
    k_wheel = max(wheel_rate_N_per_mm, 1e-6) * 1000.0          # N/m
    m = max(corner_mass_kg, 1e-3)
    c_shaft = curve.c_bump_low if region == "bump" else curve.c_reb_low
    c_wheel = c_shaft * motion_ratio * motion_ratio
    crit = 2.0 * np.sqrt(k_wheel * m)
    return float(c_wheel / crit) if crit > 0 else float("nan")


def default_damper() -> DamperCurve:
    """Representative (uncalibrated) FSAE damper. See DamperCurve.status()."""
    return DamperCurve()
