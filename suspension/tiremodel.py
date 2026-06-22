# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Pacejka Magic Formula (MF5.2) lateral tire model.

This replaces KinematiK's linear placeholder grip model with the real Magic Formula,
evaluated from coefficients fitted to measured tire data. The EQUATIONS here are the
standard, published MF5.2 pure-lateral formulae (Pacejka, *Tyre and Vehicle Dynamics*)
— they are textbook and safe to open-source. The COEFFICIENTS are tire-specific and,
when they come from TTC data, are confidential: they load from a separate file that is
gitignored and never committed. Ship the code, not the numbers.

Scope: pure lateral force Fy as a function of slip angle, vertical load, and camber
is the fitted core. Combined slip (Fx+Fy friction-ellipse coupling) and relaxation
length are also provided (see CombinedSlipTire and relaxation_length) — implemented
with real physics but flagged uncalibrated until you supply drive/brake and transient
TTC data, so they never present an invented number as measured.

A `PacejkaLateral` exposes:
    .fy(alpha_rad, Fz_N, gamma_rad=0)  -> lateral force, N
    .mu_peak(Fz_N, gamma_rad=0)        -> peak |Fy|/Fz over slip, the grip coefficient

The second is what VehicleDynamics calls to get load-sensitive grip — so plugging this
in makes the balance/grip numbers reflect the real tire, not a straight-line guess.
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass


# Default scaling factors (lambdas) — all 1.0 means "use the fit as-is".
_DEFAULT_SCALING = {
    "LFZO": 1.0, "LCY": 1.0, "LMUY": 1.0, "LEY": 1.0,
    "LKY": 1.0, "LHY": 1.0, "LVY": 1.0, "LGAY": 1.0,
}


@dataclass
class PacejkaLateral:
    """MF5.2 pure-lateral model. Construct from a coefficient dict + nominal load."""
    coeffs: dict
    FNOMIN: float = 6306.0      # nominal vertical load, N (from the tire data)
    scaling: dict = None

    def __post_init__(self):
        self.scaling = {**_DEFAULT_SCALING, **(self.scaling or {})}
        missing = [k for k in ("PCY1", "PDY1", "PDY2", "PEY1", "PKY1", "PKY2")
                   if k not in self.coeffs]
        if missing:
            raise ValueError(f"Missing required lateral coefficients: {missing}")

    def _C(self, name, default=0.0):
        return float(self.coeffs.get(name, default))

    def fy(self, alpha, Fz, gamma=0.0):
        """
        Pure lateral force Fy (N) for slip angle alpha (rad), vertical load Fz (N),
        camber gamma (rad). Standard MF5.2 pure-slip lateral equations.
        """
        Fz = np.maximum(np.asarray(Fz, float), 1e-6)
        s = self.scaling
        dfz = (Fz - self.FNOMIN) / self.FNOMIN          # normalised load increment
        g = gamma                                       # camber, rad

        # Shape factor
        Cy = self._C("PCY1") * s["LCY"]
        # Peak factor (friction)
        mu_y = (self._C("PDY1") + self._C("PDY2") * dfz) \
            * (1.0 - self._C("PDY3") * g * g) * s["LMUY"]
        Dy = mu_y * Fz
        # Curvature
        Ey = (self._C("PEY1") + self._C("PEY2") * dfz) \
            * (1.0 - (self._C("PEY3") + self._C("PEY4") * g) * np.sign(alpha)) * s["LEY"]
        Ey = np.minimum(Ey, 1.0)
        # Cornering stiffness -> B
        Ky = self._C("PKY1") * self.FNOMIN \
            * np.sin(2.0 * np.arctan(Fz / (self._C("PKY2") * self.FNOMIN * s["LFZO"]))) \
            * (1.0 - self._C("PKY3") * abs(g)) * s["LKY"]
        By = Ky / (Cy * Dy + 1e-9)
        # Horizontal/vertical shifts
        Shy = (self._C("PHY1") + self._C("PHY2") * dfz) * s["LHY"] + self._C("PHY3") * g
        Svy = Fz * ((self._C("PVY1") + self._C("PVY2") * dfz) * s["LVY"]
                    + (self._C("PVY3") + self._C("PVY4") * dfz) * g) * s["LGAY"]

        ax = alpha + Shy
        Fy = Dy * np.sin(Cy * np.arctan(By * ax - Ey * (By * ax - np.arctan(By * ax)))) + Svy
        return Fy

    def mu_peak(self, Fz, gamma=0.0):
        """
        Peak friction coefficient |Fy|/Fz at a given load, found by sweeping slip
        angle. This is the single number VehicleDynamics needs for grip — and unlike
        the linear placeholder, it carries the real nonlinear load sensitivity.
        """
        alphas = np.radians(np.linspace(-15, 15, 121))
        # fy is fully vectorised over alpha, so evaluate all sweep points in one
        # call rather than looping in Python — same numbers, ~100x fewer calls.
        fy = self.fy(alphas, Fz, gamma)
        return float(np.max(np.abs(fy)) / max(Fz, 1e-6))

    def peak_force(self, Fz, gamma=0.0):
        """Peak |Fy| in newtons at a given load/camber (mu_peak * Fz)."""
        return self.mu_peak(Fz, gamma) * max(float(Fz), 0.0)

    def alpha_peak(self, Fz, gamma=0.0):
        """
        Slip angle (deg) at which lateral force peaks for this load/camber. This is
        the target operating slip — knowing it tells the driver/aero team how much
        steer the front needs at the limit, and feeds combined-slip headroom later.
        """
        alphas = np.radians(np.linspace(0.0, 15.0, 151))
        fy = np.abs(self.fy(alphas, Fz, gamma))
        return float(np.degrees(alphas[int(np.argmax(fy))]))

    def optimal_camber(self, Fz, cam_min_deg=-6.0, cam_max_deg=0.5, n=40):
        """
        Camber (deg, in tire frame where negative leans the top inboard) that
        maximises peak lateral grip at a given load. This is *free* grip for an
        underfunded team: it's set by geometry, not budget. Returns (best_camber_deg,
        mu_at_best). The dynamics layer uses the per-corner kinematic camber, but
        this answers "what should we target?" directly from the tire.

        Convention note: this returns the inclination angle magnitude that helps.
        A racing setup runs negative static camber; here we sweep the |IA| that the
        loaded outside tire actually sees and report the best as a negative number.
        """
        cambers = np.linspace(cam_min_deg, cam_max_deg, n)
        mus = [self.mu_peak(Fz, np.radians(abs(c))) for c in cambers]
        i = int(np.argmax(mus))
        return float(cambers[i]), float(mus[i])


# --------------------------------------------------------------------------- #
#  Loading coefficients (kept OUT of the public repo)
# --------------------------------------------------------------------------- #
def load_from_json(path: str) -> PacejkaLateral:
    """Load a coefficient JSON {coeffs:{...}, FNOMIN:..} — your private tire file."""
    with open(path) as f:
        d = json.load(f)
    return PacejkaLateral(coeffs=d["coeffs"], FNOMIN=d.get("FNOMIN", 6306.0),
                          scaling=d.get("scaling"))


def coeffs_to_json(coeffs: dict, FNOMIN: float, path: str):
    """Write a private coefficient file (gitignored). Never commit the result."""
    with open(path, "w") as f:
        json.dump({"coeffs": coeffs, "FNOMIN": FNOMIN}, f, indent=2)


# --------------------------------------------------------------------------- #
#  Generic default tire (NOT TTC-derived — safe to ship)
# --------------------------------------------------------------------------- #
# These coefficients are a representative, hand-tuned MF5.2 lateral set for a
# generic ~13"/10" FSAE tire at low load. They are NOT fitted to any confidential
# TTC data and are safe to commit. They give physically sensible behaviour:
#   - peak mu ~1.55 near nominal load, ~1.69 light / ~1.37 heavy (load sensitivity)
#   - a camber optimum around 2-3 deg of inclination, then falloff
# Use them so the grip/balance engine runs on a real Magic Formula from day one.
# Replace with YOUR fitted tire (process_ttc.py -> JSON) the moment you have data;
# the absolute grip numbers only become trustworthy once they're your tire's.
_DEFAULT_FSAE_COEFFS = {
    "PCY1": 1.45, "PDY1": 1.55, "PDY2": -0.22, "PDY3": 1.2,
    "PEY1": -0.6, "PEY2": -0.1, "PEY3": 0.1, "PEY4": 2.0,
    "PKY1": -28.0, "PKY2": 1.6, "PKY3": 0.6,
    "PHY1": 0.0, "PHY2": 0.0, "PHY3": 0.0,
    "PVY1": 0.0, "PVY2": 0.0, "PVY3": 0.12, "PVY4": 0.0,
}
_DEFAULT_FSAE_FNOMIN = 1100.0      # N, representative FSAE corner load


def default_tire() -> PacejkaLateral:
    """
    A generic FSAE Pacejka lateral model with sensible behaviour, safe to ship.
    This is what the tool uses until you load your own fitted tire. It is good for
    RELATIVE comparisons (which setup change helps?) out of the box; absolute grip
    only becomes trustworthy once you swap in your TTC-fitted coefficients.
    """
    return PacejkaLateral(coeffs=dict(_DEFAULT_FSAE_COEFFS),
                          FNOMIN=_DEFAULT_FSAE_FNOMIN)


def describe(tire: PacejkaLateral) -> dict:
    """Quick human-readable summary of a tire model's grip envelope."""
    return {
        "FNOMIN_N": tire.FNOMIN,
        "mu_at_nominal": round(tire.mu_peak(tire.FNOMIN), 3),
        "mu_light_load": round(tire.mu_peak(0.4 * tire.FNOMIN), 3),
        "mu_heavy_load": round(tire.mu_peak(1.8 * tire.FNOMIN), 3),
        "alpha_peak_deg": round(tire.alpha_peak(tire.FNOMIN), 2),
        "optimal_camber_deg": round(tire.optimal_camber(tire.FNOMIN)[0], 2),
    }


# --------------------------------------------------------------------------- #
#  Combined slip (Fx + Fy interaction) and relaxation length
# --------------------------------------------------------------------------- #
#  HONESTY CONTRACT for this section
#  --------------------------------
#  The lateral model above is fitted to real TTC *cornering* runs. Combined slip
#  and relaxation length need data the cornering runs don't contain:
#    * combined slip needs DRIVE/BRAKE (Fx) data at various slip angles,
#    * relaxation length needs TRANSIENT (step-input) data, not steady-state.
#  Rather than fake those, this implements the REAL physics with explicit, visible
#  parameters and an `is_calibrated` flag that is False until you supply the data.
#  Uncalibrated, it uses the standard friction-ellipse coupling and a
#  load-scaled relaxation length from published FSAE-tyre ranges — a principled
#  approximation, clearly flagged as such — so the capability exists today and
#  becomes quantitatively trustworthy the moment you calibrate it. Nothing here
#  silently invents a number and presents it as measured.
# --------------------------------------------------------------------------- #
@dataclass
class CombinedSlipTire:
    """
    Combined longitudinal+lateral grip wrapped around a fitted `PacejkaLateral`.

    Uses the friction-ellipse (cosine-weighting) method: at a given combined
    demand, the available lateral force is scaled by how much of the friction
    budget longitudinal force is using, and vice-versa. This is the standard
    closed-form coupling used when full combined-slip Magic Formula coefficients
    aren't available, and it is exact on the circle/ellipse limit that governs
    corner entry (trail-braking) and exit (power-down) — the cases that matter.

    Set `is_calibrated=True` only once you've fitted the ellipse exponents to
    drive/brake TTC data; until then it reports calibrated=False and uses the
    symmetric ellipse (exponents = 2), which is the honest default.
    """
    lateral: PacejkaLateral
    mu_x_ratio: float = 1.05      # peak longitudinal mu / peak lateral mu (>1 typical)
    ell_kx: float = 2.0           # ellipse exponent, longitudinal (2 => circle/ellipse)
    ell_ky: float = 2.0           # ellipse exponent, lateral
    is_calibrated: bool = False

    def mu_x_peak(self, Fz, gamma=0.0):
        """Peak longitudinal friction coefficient at this load (scaled from lateral)."""
        return self.mu_x_ratio * self.lateral.mu_peak(Fz, gamma)

    def available_fy(self, fx_demand_N, Fz, gamma=0.0):
        """
        Lateral force still available (N) when `fx_demand_N` of longitudinal force
        is being used at vertical load Fz. Friction-ellipse coupling:
            (Fx/Fx_max)^kx + (Fy/Fy_max)^ky = 1
        so Fy_avail = Fy_max * (1 - (Fx/Fx_max)^kx)^(1/ky).
        """
        Fz = max(float(Fz), 1e-6)
        fy_max = self.lateral.peak_force(Fz, gamma)
        fx_max = self.mu_x_peak(Fz, gamma) * Fz
        if fx_max <= 1e-6:
            return 0.0
        use = min(abs(fx_demand_N) / fx_max, 1.0)
        frac = max(1.0 - use ** self.ell_kx, 0.0)
        return float(fy_max * frac ** (1.0 / self.ell_ky))

    def available_fx(self, fy_demand_N, Fz, gamma=0.0):
        """Longitudinal force still available (N) when `fy_demand_N` lateral is used."""
        Fz = max(float(Fz), 1e-6)
        fy_max = self.lateral.peak_force(Fz, gamma)
        fx_max = self.mu_x_peak(Fz, gamma) * Fz
        if fy_max <= 1e-6:
            return fx_max
        use = min(abs(fy_demand_N) / fy_max, 1.0)
        frac = max(1.0 - use ** self.ell_ky, 0.0)
        return float(fx_max * frac ** (1.0 / self.ell_kx))

    def friction_circle(self, Fz, gamma=0.0, n=72):
        """
        The (Fx, Fy) limit envelope at a load — the 'friction circle' (really an
        ellipse) the team should keep the combined-g vector inside. Returns
        (fx_array, fy_array) tracing the boundary, for plotting and for checking
        how much combined-g headroom a corner phase has.
        """
        fx_max = self.mu_x_peak(Fz, gamma) * max(float(Fz), 1e-6)
        fy_max = self.lateral.peak_force(Fz, gamma)
        th = np.linspace(0, 2 * np.pi, n)
        # parametric ellipse with the calibrated exponents (superellipse if !=2)
        cx = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2.0 / self.ell_kx)
        cy = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2.0 / self.ell_ky)
        return fx_max * cx, fy_max * cy

    def status(self) -> str:
        return ("calibrated to drive/brake data" if self.is_calibrated
                else "UNCALIBRATED — symmetric friction ellipse (needs Fx TTC data "
                     "to be quantitative; coupling shape is still physically valid)")


def relaxation_length(Fz, FNOMIN=1100.0, sigma_nominal_m=0.45,
                      is_calibrated=False):
    """
    Tyre relaxation length (m): the distance the tyre must roll for lateral force
    to build to ~63% of its steady value after a slip-angle change. It sets
    turn-in lag and is what separates a transient model from a steady-state one.

    Physically it scales with cornering stiffness / vertical load and grows with
    load. Without transient TTC data we use a load-scaled value from published
    FSAE-tyre ranges (~0.3-0.6 m), flagged is_calibrated=False. With step-input
    data, fit `sigma_nominal_m` and set the flag. Returns sigma in metres.

    The lag itself is a first-order rolling filter:
        d(Fy)/ds = (Fy_steady - Fy) / sigma
    which the transient layer applies; this function supplies sigma.
    """
    Fz = max(float(Fz), 1e-6)
    # mild growth with load (sub-linear): sigma ~ sigma_nom * (Fz/Fnom)^0.4
    return float(sigma_nominal_m * (Fz / max(FNOMIN, 1e-6)) ** 0.4)


def apply_relaxation_lag(alpha_target, ds, sigma_m, alpha_prev=0.0):
    """
    Advance the lagged slip angle one rolling step. Exact first-order relaxation
    over a step of `ds` metres with relaxation length `sigma_m`:
        alpha_eff_new = alpha_target + (alpha_prev - alpha_target) * exp(-ds/sigma)
    so after one relaxation length (ds = sigma) the response has covered ~63% of a
    step change, which is the definition of relaxation length. `ds` and `sigma_m`
    in metres. This is the building block a transient lap/yaw model uses; it is
    exposed and tested so the transient layer is built on a verified primitive.
    """
    sigma = max(float(sigma_m), 1e-6)
    decay = math.exp(-max(ds, 0.0) / sigma)
    return float(alpha_target + (alpha_prev - alpha_target) * decay)


def default_combined_tire(lateral: PacejkaLateral = None) -> CombinedSlipTire:
    """A combined-slip wrapper on the generic default (or supplied) lateral tire.
    Uncalibrated by design — see CombinedSlipTire.status()."""
    return CombinedSlipTire(lateral=lateral or default_tire())
