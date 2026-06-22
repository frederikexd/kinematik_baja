# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Fit a full MF5.2 pure-lateral Magic Formula to measured tire data.

This is the bridge that turns raw TTC channels (slip angle, vertical load, camber,
lateral force) into the coefficient set that `tiremodel.PacejkaLateral` consumes —
so the grip/balance engine runs on YOUR tire instead of the generic default.

Why this is the competitive lever for an underfunded team
---------------------------------------------------------
You get one set of tires. You cannot test-and-iterate on rubber the way a funded
team can. So the entire edge is extracting maximum truth from the tire data you're
allowed (the FSAE Tire Test Consortium) and making every geometry/setup decision
against it BEFORE you commit the one set you can afford. A linear grip guess throws
away the load sensitivity and camber response that decide skidpad and the limit in
autocross. A fitted Magic Formula keeps them.

    raw TTC .mat  --process_ttc.load+clean-->  (SA, FZ, IA, FY) arrays
                  --fit_mf52_lateral-->        coeff dict + FNOMIN
                  --tiremodel.coeffs_to_json-> private JSON (gitignored)
                  --tiremodel.load_from_json-> live model in the tool

⚠ LICENSING: coefficients fitted from TTC data are confidential under the TTC
agreement. Write them to a gitignored file and never commit them. The fitter code
here is generic math and safe to ship; the numbers it produces are not.

The fit is a bounded nonlinear least-squares (scipy) over the pure-lateral
coefficients, seeded from the generic default so it converges from a sane start.
We fit Fy(alpha, Fz, gamma) directly against the measured points — the same
equation the live model evaluates — so what you fit is exactly what you run.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .tiremodel import PacejkaLateral, _DEFAULT_FSAE_COEFFS, _DEFAULT_FSAE_FNOMIN


# The lateral coefficients we fit, in order. A focused subset of MF5.2 that carries
# the physics that matters for steady-state cornering grip: shape, peak friction +
# load sensitivity, curvature, cornering stiffness, and camber response. Shifts
# (PHY*, PVY*) are fit too because they set camber thrust and any asymmetry.
_FIT_KEYS = [
    "PCY1",
    "PDY1", "PDY2", "PDY3",
    "PEY1", "PEY2", "PEY3", "PEY4",
    "PKY1", "PKY2", "PKY3",
    "PHY1", "PHY2", "PHY3",
    "PVY1", "PVY2", "PVY3", "PVY4",
]

# Physically-motivated bounds keep the fit from wandering into nonsense when the
# data is noisy or doesn't span enough load/camber. These are wide but sane.
_BOUNDS = {
    "PCY1": (1.0, 2.0),
    "PDY1": (0.5, 3.0), "PDY2": (-2.0, 0.5), "PDY3": (-10.0, 10.0),
    "PEY1": (-5.0, 1.0), "PEY2": (-2.0, 2.0), "PEY3": (-2.0, 2.0), "PEY4": (-20.0, 20.0),
    "PKY1": (-80.0, -5.0), "PKY2": (0.5, 5.0), "PKY3": (-2.0, 2.0),
    "PHY1": (-0.1, 0.1), "PHY2": (-0.1, 0.1), "PHY3": (-0.2, 0.2),
    "PVY1": (-0.5, 0.5), "PVY2": (-0.5, 0.5), "PVY3": (-1.0, 1.0), "PVY4": (-1.0, 1.0),
}


def _build_tire(free_vec, fnomin):
    coeffs = dict(_DEFAULT_FSAE_COEFFS)
    for k, v in zip(_FIT_KEYS, free_vec):
        coeffs[k] = float(v)
    return PacejkaLateral(coeffs=coeffs, FNOMIN=fnomin)


def _seed_zero_slope(fnomin):
    """Sign of dFy/dSA near zero for the seed/default model at nominal load — used
    to decide whether measured FY follows the same sign convention as the model."""
    seed = PacejkaLateral(coeffs=dict(_DEFAULT_FSAE_COEFFS), FNOMIN=fnomin)
    a = np.radians(1.0)
    return (seed.fy(a, fnomin, 0.0) - seed.fy(-a, fnomin, 0.0)) / (2 * a)


def fit_mf52_lateral(SA_rad, FZ_N, FY_N, IA_rad=None,
                     fnomin=None, weight_by_load=True, verbose=False):
    """
    Fit the MF5.2 pure-lateral coefficients to measured points.

    Parameters
    ----------
    SA_rad : slip angle, radians (TTC SA is degrees — convert before calling)
    FZ_N   : vertical load, newtons, positive magnitude
    FY_N   : lateral force, newtons (sign convention will be normalised internally)
    IA_rad : inclination/camber, radians. If None, assumed 0 (zero-camber fit).
    fnomin : nominal load. Defaults to the median measured |Fz| — a good FSAE choice
             since the median load is near the real operating point.
    weight_by_load : weight residuals by load so high-load points (where grip is
             scarce and decisions are made) aren't drowned by light-load samples.

    Returns
    -------
    dict with keys:
        coeffs  : the full coefficient dict for PacejkaLateral
        FNOMIN  : nominal load used
        rmse_N  : RMS force error over the fit set, newtons
        r2      : coefficient of determination on Fy
        n       : number of points fit
    """
    SA = np.asarray(SA_rad, float).ravel()
    FZ = np.abs(np.asarray(FZ_N, float).ravel())
    FY = np.asarray(FY_N, float).ravel()
    IA = (np.zeros_like(SA) if IA_rad is None
          else np.asarray(IA_rad, float).ravel())

    # Keep loaded, finite samples only.
    good = np.isfinite(SA) & np.isfinite(FZ) & np.isfinite(FY) & np.isfinite(IA) & (FZ > 50)
    SA, FZ, FY, IA = SA[good], FZ[good], FY[good], IA[good]
    if len(SA) < 50:
        raise ValueError(f"Only {len(SA)} usable points — need at least ~50 spanning "
                         "a range of slip and load to fit a Magic Formula.")

    # Sign convention: the MF5.2 form here (with negative PKY1) produces Fy of the
    # OPPOSITE sign to slip angle. TTC raw data sometimes follows the opposite
    # convention. We align the *data* to the *model's* convention by checking the
    # sign of the slope near zero slip and flipping FY only if it disagrees with the
    # model family — comparing against the seed model's own slope, not against SA.
    seed_slope = np.sign(_seed_zero_slope(fnomin if fnomin else float(np.median(FZ))))
    data_slope = np.sign(np.polyfit(SA, FY, 1)[0]) if np.std(SA) > 0 else 1.0
    if seed_slope != 0 and data_slope != 0 and seed_slope != data_slope:
        FY = -FY

    if fnomin is None:
        fnomin = float(np.median(FZ))

    w = np.sqrt(FZ / fnomin) if weight_by_load else np.ones_like(FZ)

    x0 = np.array([_DEFAULT_FSAE_COEFFS[k] for k in _FIT_KEYS], float)
    lo = np.array([_BOUNDS[k][0] for k in _FIT_KEYS], float)
    hi = np.array([_BOUNDS[k][1] for k in _FIT_KEYS], float)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def resid(x):
        tire = _build_tire(x, fnomin)
        pred = tire.fy(SA, FZ, IA)
        return (pred - FY) * w

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        max_nfev=4000, xtol=1e-12, ftol=1e-12, verbose=0)

    tire = _build_tire(sol.x, fnomin)
    pred = tire.fy(SA, FZ, IA)
    err = pred - FY
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((FY - FY.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    coeffs = dict(_DEFAULT_FSAE_COEFFS)
    for k, v in zip(_FIT_KEYS, sol.x):
        coeffs[k] = float(v)

    if verbose:
        print(f"  fit on {len(SA)} pts | FNOMIN={fnomin:.0f} N | "
              f"RMSE={rmse:.1f} N | R^2={r2:.4f}")

    return {"coeffs": coeffs, "FNOMIN": fnomin, "rmse_N": rmse,
            "r2": r2, "n": int(len(SA))}


def fit_from_ttc_channels(chans: dict, **kw):
    """
    Convenience wrapper for the channel dict that process_ttc.load_ttc/clean return.
    Expects keys FY, FZ, SA (deg), optionally IA (deg). Converts angles to radians.
    """
    if not all(k in chans for k in ("FY", "FZ", "SA")):
        raise ValueError("Need FY, FZ and SA channels to fit a lateral model.")
    sa = np.radians(chans["SA"])
    ia = np.radians(chans["IA"]) if "IA" in chans else None
    return fit_mf52_lateral(sa, chans["FZ"], chans["FY"], IA_rad=ia, **kw)
