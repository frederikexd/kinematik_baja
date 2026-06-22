# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
TTC tire-data processing — validate the grip model against measured tire data.

⚠ LICENSING: TTC (Tire Test Consortium) data is provided under a confidentiality
agreement. The raw .mat files AND any coefficients fitted from them must NOT be
committed to the public repository. This whole folder is gitignored. Keep the
.mat files and any output you produce here local to your machine / your team's
private storage. The open-source KinematiK ships only with the placeholder grip
model; the TTC-derived model loads separately from a local file that never enters
git. This is how you stay compliant while still using real data internally.

WHAT THIS DOES
--------------
Loads a TTC cornering .mat file, cleans it the standard way, and extracts the
load-sensitivity curve (peak friction coefficient mu vs vertical load Fz). That
curve is what validates — or corrects — KinematiK's grip model, which currently
assumes a straight line mu = mu_peak - load_sens * Fz.

Run:  python tire_tools/process_ttc.py path/to/your_cornering_file.mat
"""

from __future__ import annotations

import sys
import numpy as np
import scipy.io as sio


# TTC round files vary slightly in channel names across rounds. These are the
# common ones; the loader is tolerant and reports what it actually finds.
CHANNEL_ALIASES = {
    "FY": ["FY", "Fy"],
    "FZ": ["FZ", "Fz"],
    "SA": ["SA", "Sa", "slip_angle"],
    "IA": ["IA", "Ia", "camber"],
    "P":  ["P", "PRESS", "pressure"],
    "V":  ["V", "speed"],
    "FX": ["FX", "Fx"],
}


def _find_channel(mat: dict, names):
    for n in names:
        if n in mat:
            return np.asarray(mat[n]).squeeze()
    return None


def load_ttc(path: str) -> dict:
    """
    Load a TTC .mat cornering file into a dict of numpy channel arrays.
    Reports which expected channels were found so you can sanity-check the file.
    """
    raw = sio.loadmat(path)
    # strip MATLAB metadata keys
    raw = {k: v for k, v in raw.items() if not k.startswith("__")}
    chans = {}
    for canon, aliases in CHANNEL_ALIASES.items():
        arr = _find_channel(raw, aliases)
        if arr is not None:
            chans[canon] = arr.astype(float).ravel()
    found = sorted(chans.keys())
    missing = [c for c in ("FY", "FZ", "SA") if c not in chans]
    print(f"Loaded {path}")
    print(f"  channels found: {found}")
    if missing:
        print(f"  ⚠ missing essential channels {missing} — is this a cornering file? "
              f"(raw keys present: {sorted(raw.keys())[:12]})")
    n = min(len(v) for v in chans.values()) if chans else 0
    # truncate all channels to common length (rig files occasionally differ by 1)
    for k in chans:
        chans[k] = chans[k][:n]
    print(f"  samples: {n}")
    return chans


def clean(chans: dict, drop_warmup_frac=0.05) -> dict:
    """
    Standard cleanup: drop the first few percent (conditioning/warmup), and remove
    obviously bad samples (non-finite, zero/low vertical load where the tire is off
    the road). This is the step student analyses most often skip — without it the
    grip curve is contaminated by warmup and airborne samples.
    """
    n = len(chans["FZ"])
    start = int(n * drop_warmup_frac)
    out = {k: v[start:] for k, v in chans.items()}
    fz = out["FZ"]
    # TTC FZ is negative (downward) in SAE; work with magnitude
    fz_mag = np.abs(fz)
    good = np.isfinite(fz_mag) & (fz_mag > 100)   # tire actually loaded (>100 N)
    for k in out:
        if np.all(np.isfinite(out[k])) or out[k].shape == good.shape:
            out[k] = out[k][good]
    return out


def grip_vs_load(chans: dict, ia_window=(-0.5, 0.5), p_target=None, p_tol=10.0,
                 n_bins=12) -> tuple:
    """
    Extract the load-sensitivity curve: peak |FY/FZ| (friction coefficient mu) as a
    function of vertical load Fz, at roughly constant camber and pressure.

    Filtering to one camber/pressure is essential — mixing them averages away the
    very effect you're trying to measure. Returns (fz_bins_N, mu_peak_per_bin).
    """
    fy = np.abs(chans["FY"])
    fz = np.abs(chans["FZ"])
    mu = np.divide(fy, fz, out=np.zeros_like(fy), where=fz > 1)

    mask = np.ones(len(fz), dtype=bool)
    if "IA" in chans:
        mask &= (chans["IA"] >= ia_window[0]) & (chans["IA"] <= ia_window[1])
    if p_target is not None and "P" in chans:
        mask &= np.abs(chans["P"] - p_target) <= p_tol

    fz, mu = fz[mask], mu[mask]
    if len(fz) < 20:
        raise ValueError("Too few samples after filtering — loosen the camber/"
                         "pressure window or check the file.")

    # bin by load, take the peak mu in each bin (the tire's capability at that load)
    edges = np.linspace(fz.min(), fz.max(), n_bins + 1)
    centers, peaks = [], []
    for i in range(n_bins):
        sel = (fz >= edges[i]) & (fz < edges[i + 1])
        if sel.sum() >= 5:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            peaks.append(np.percentile(mu[sel], 95))   # 95th pct ≈ peak, robust to noise
    return np.array(centers), np.array(peaks)


def fit_linear_load_sensitivity(fz_bins, mu_peaks):
    """
    Fit KinematiK's model form mu = mu_peak0 - load_sens * Fz to the measured curve.
    Returns (mu_peak0, load_sens) — the two numbers KinematiK's grip model uses.
    These are TTC-derived: keep them OUT of the public repo.
    """
    A = np.vstack([np.ones_like(fz_bins), -fz_bins]).T
    (mu0, load_sens), *_ = np.linalg.lstsq(A, mu_peaks, rcond=None)
    return float(mu0), float(load_sens)


def compare_to_model(fz_bins, mu_peaks, mu_peak_model, load_sens_model):
    """Print how the current KinematiK grip model compares to measured TTC data."""
    model_mu = mu_peak_model - load_sens_model * fz_bins
    err = mu_peaks - model_mu
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print("\n  Fz (N)   measured mu   model mu   error")
    for f, m, mm in zip(fz_bins, mu_peaks, model_mu):
        print(f"  {f:6.0f}    {m:8.3f}    {mm:7.3f}   {m - mm:+.3f}")
    print(f"\n  RMSE of current model vs TTC data: {rmse:.4f}")
    fit_mu0, fit_ls = fit_linear_load_sensitivity(fz_bins, mu_peaks)
    print(f"\n  Best-fit to YOUR tire (use these in VehicleParams, keep them private):")
    print(f"    mu_peak     = {fit_mu0:.3f}")
    print(f"    tire_load_sens = {fit_ls:.6f}   (1/N)")
    return rmse, fit_mu0, fit_ls


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python process_ttc.py your_cornering_file.mat [out.json]")
        print("  With no out.json: prints the linear load-sensitivity comparison.")
        print("  With out.json:    ALSO fits a full MF5.2 lateral model and writes it,")
        print("                    ready to load straight into KinematiK's tire model.")
        return
    chans = load_ttc(sys.argv[1])
    if not all(c in chans for c in ("FY", "FZ")):
        print("\nCan't proceed without FY and FZ. If this is a drive/brake file, "
              "use a cornering file instead.")
        return
    chans = clean(chans)
    # KinematiK's current defaults, for the comparison
    MODEL_MU_PEAK = 1.55
    MODEL_LOAD_SENS = 0.00018
    try:
        fz_bins, mu_peaks = grip_vs_load(chans)
    except ValueError as e:
        print(f"\n{e}")
        return
    compare_to_model(fz_bins, mu_peaks, MODEL_MU_PEAK, MODEL_LOAD_SENS)

    # ---- Full MF5.2 fit (the real upgrade) ---------------------------------
    # The linear comparison above is a sanity check. The thing the tool actually
    # runs on is a full Magic Formula fitted to ALL the points, not just the
    # binned peaks — it captures the slip-curve shape and camber response, not
    # only load sensitivity. Fit it and (optionally) write the private JSON.
    if "SA" not in chans:
        print("\nNo slip-angle (SA) channel found — can't fit the full Magic "
              "Formula (need force vs slip). The linear summary above still stands.")
        return
    try:
        from suspension.tirefit import fit_from_ttc_channels
    except Exception as e:
        print(f"\nCouldn't import the MF5.2 fitter ({e}). Run this from the project "
              "root so the `suspension` package is importable.")
        return

    print("\nFitting full MF5.2 lateral model to all points ...")
    try:
        res = fit_from_ttc_channels(chans, verbose=True)
    except ValueError as e:
        print(f"  fit skipped: {e}")
        return
    print(f"  R^2 = {res['r2']:.4f}  |  RMSE = {res['rmse_N']:.0f} N  "
          f"|  points = {res['n']}")
    if res["r2"] < 0.9:
        print("  ⚠ R^2 below 0.9 — the fit is loose. Check that the file is a clean "
              "cornering sweep with a good spread of load and slip, and that warmup "
              "was trimmed. A loose fit means loose grip numbers downstream.")

    if len(sys.argv) >= 3:
        out = sys.argv[2]
        from suspension.tiremodel import coeffs_to_json
        coeffs_to_json(res["coeffs"], res["FNOMIN"], out)
        print(f"\n  ✓ Wrote fitted tire to {out}")
        print(f"    Load it in the app's tire panel, or in code:")
        print(f"      from suspension.tiremodel import load_from_json")
        print(f"      tire = load_from_json('{out}')")
        print(f"    ⚠ This file is TTC-derived — keep it OUT of git (it's gitignored).")
    else:
        print("\n  (Pass an output path to write the fitted tire JSON, e.g. "
              "`python process_ttc.py file.mat my_tire.json`.)")

    print("\nReminder: these numbers are TTC-derived — do NOT commit them or the "
          ".mat files to the public repo.")


if __name__ == "__main__":
    main()
