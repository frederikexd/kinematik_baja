# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Multi-team integration: check ANY subteam's part against the chassis.

This generalises the suspension chassis check so every Elbee subteam — aero,
brakes, cooling, data-acq, electrics, powertrain — can validate their part against
the shared chassis before anyone manufactures. The workflow is identical no matter
which team you're on:

    1. load the chassis once (the shared reference body)
    2. load your part (caliper, radiator, battery box, wing mount, ...)
    3. get back: do they collide, how much clearance, and where

The engineering point for a team that can't out-spend USC: rework is the tax you
pay for not integrating in CAD. A richer team can afford to cut a part twice. We
catch the interference before the first cut. That's the equaliser.

Everything is mm, in whatever shared coordinate frame the team agrees on (use the
chassis origin). Parts are positioned with a manual offset/rotation since CAD
exports rarely share an origin out of the box.

This module has NO dependency on the suspension kinematics — it's pure geometry,
so it works for any rigid part. The suspension corner is just one special case
(handled in chassis.py, which sweeps a moving linkage); static parts use this.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import trimesh
from dataclasses import dataclass


# Elbee subteam registry — matches the Discord channels. Each team's parts get
# tagged so a future packaging view can colour/group by team.
#  Elbee Racing — Baja SAE subsystem teams.
#
#  Rebased from the original FSAE-EV team list. Baja SAE is a single-make
#  combustion off-road buggy: there is no aero package, no cooling pack and no
#  HV electrics, so those teams are gone. The drivetrain is a CVT + gearbox
#  (not a traction motor + accumulator). Suspension is split front/rear because
#  on this car the front corner also carries the steering, and the front and
#  rear corners are genuinely different mechanisms (the rear is usually a
#  trailing/semi-trailing or multi-link, the front a double wishbone with a
#  steering rack). Front suspension + steering and rear suspension being the
#  historical year-after-year failure point is exactly why they get their own
#  owners here.
TEAMS = {
    "drivetrain":       {"color": "#ff4444", "label": "Drivetrain"},
    "front-suspension": {"color": "#ff6fb5", "label": "Front Suspension + Steering"},
    "rear-suspension":  {"color": "#ffd93b", "label": "Rear Suspension"},
    "chassis":          {"color": "#a855f7", "label": "Chassis"},
    "data-acquisition": {"color": "#3ec46d", "label": "Data acquisition"},
}

# Back-compat alias map: old FSAE keys that other modules / saved projects may
# still reference are folded onto the nearest Baja owner so nothing KeyErrors.
TEAM_ALIASES = {
    "suspension":   "front-suspension",
    "powertrain":   "drivetrain",
    "electrics":    "drivetrain",
    "brakes":       "front-suspension",
    "aerodynamics": "chassis",
    "cooling":      "drivetrain",
}


def resolve_team(key):
    """Map any (possibly legacy FSAE) team key onto a current Baja team key."""
    if key in TEAMS:
        return key
    return TEAM_ALIASES.get(key, "chassis")


# --------------------------------------------------------------------------- #
#  Loading any part / body
# --------------------------------------------------------------------------- #
def load_part(path: str, offset=(0.0, 0.0, 0.0), scale=1.0,
              rotate_deg=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    """
    Load a part from STEP / STL / OBJ / GLB into a single Trimesh, positioned in
    the shared frame: scale first, then rotate (XYZ Euler, degrees), then translate.

    Raises ValueError with a human-readable message for the common failure modes
    (unreadable file, empty geometry, degenerate mesh) so the UI can tell the user
    what's actually wrong instead of surfacing a raw library traceback.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".step", ".stp", ".stl", ".obj", ".glb"):
        raise ValueError(
            f"Unsupported file type '{ext}'. Use STEP (.step/.stp), STL, OBJ, or GLB. "
            f"SolidWorks .sldprt isn't supported — export it as STEP first.")

    try:
        if ext in (".step", ".stp"):
            import cascadio
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                glb = tmp.name
            cascadio.step_to_glb(path, glb, tol_linear=0.5, tol_angular=0.5)
            scene = trimesh.load(glb, force="scene")
            os.unlink(glb)
        else:
            scene = trimesh.load(path, force="scene")
    except Exception as e:
        raise ValueError(
            f"Couldn't read the file — it may be corrupted or not a valid {ext} file. "
            f"(details: {e})")

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(
                "No 3D geometry found in the file. If it's a STEP assembly, make sure "
                "it contains solid bodies, not just sketches or reference geometry.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The file loaded but contains no usable surface mesh.")

    if scale != 1.0:
        mesh.apply_scale(scale)
    rx, ry, rz = np.radians(rotate_deg)
    if any((rx, ry, rz)):
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")
        mesh.apply_transform(R)
    mesh.apply_translation(np.asarray(offset, float))

    # Gentle unit sanity check: FSAE parts in mm are typically 10–3000 mm across.
    # A bounding box under ~5 mm usually means the file is in metres (needs scale 1000).
    size = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if size < 5.0:
        raise ValueError(
            f"The part is only {size:.2f} units across, which is suspiciously small — "
            f"the file is probably in metres. Set scale to 1000 to convert to mm.")
    return mesh


def part_summary(mesh: trimesh.Trimesh) -> dict:
    lo, hi = mesh.bounds
    return {
        "triangles": int(len(mesh.faces)),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "size_mm": (hi - lo).tolist(),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "watertight": bool(mesh.is_watertight),
    }


# --------------------------------------------------------------------------- #
#  Interference check — part vs reference (chassis)
# --------------------------------------------------------------------------- #
def interference_check(part: trimesh.Trimesh, reference: trimesh.Trimesh,
                       warn_mm=5.0, sample_target=4000):
    """
    Check a (static) part against a reference body (the chassis).

    Strategy: sample points across the part surface and query their signed distance
    to the reference. Negative signed distance (point inside reference solid) =
    interpenetration = hard collision. Small positive = too tight. We also test the
    reverse (reference sampled against part) is unnecessary for the verdict — a
    collision shows up from either side, and sampling the part is enough because the
    part is what's being placed.

    Returns verdict + min clearance + the worst-offending region of the part (its
    centroid), so the team knows WHERE to move the part.
    """
    from trimesh.proximity import ProximityQuery

    # Quick reject: if bounding boxes are farther apart than the warn band, it's
    # definitely clear and we can skip sampling. Within warn_mm we must sample to
    # get the true surface clearance and catch TIGHT cases.
    pa_lo, pa_hi = part.bounds
    rf_lo, rf_hi = reference.bounds
    gap = np.maximum(rf_lo - pa_hi, pa_lo - rf_hi)
    bbox_gap = float(np.max(gap))
    if bbox_gap > warn_mm:
        return {
            "verdict": "CLEAR",
            "min_clearance_mm": bbox_gap,
            "collision_fraction": 0.0,
            "worst_point": None,
            "method": "bbox_reject",
        }

    # Sample the part surface.
    n = min(sample_target, max(800, len(part.faces)))
    pts, _ = trimesh.sample.sample_surface(part, n)

    pq = ProximityQuery(reference)
    signed = pq.signed_distance(pts)     # + inside reference solid = collision
    clearance = -signed                  # + = real gap outside reference

    min_clear = float(np.min(clearance))
    collide_mask = clearance < 0
    collision_fraction = float(np.mean(collide_mask))

    if collision_fraction > 0:
        worst = pts[np.argmin(clearance)]
    else:
        worst = pts[np.argmin(clearance)]

    verdict = ("COLLISION" if min_clear < 0
               else "TIGHT" if min_clear < warn_mm
               else "CLEAR")
    return {
        "verdict": verdict,
        "min_clearance_mm": min_clear,
        "collision_fraction": collision_fraction,
        "worst_point": worst.tolist(),
        "method": "surface_sample",
        "n_samples": int(n),
    }


# --------------------------------------------------------------------------- #
#  Mass / packaging bookkeeping (cross-team)
# --------------------------------------------------------------------------- #
@dataclass
class PartRecord:
    team: str
    name: str
    mass_g: float | None
    centroid: tuple
    bbox_size: tuple


def part_record(team: str, name: str, mesh: trimesh.Trimesh,
                density_kg_m3: float | None = None) -> PartRecord:
    """
    Build a cross-team record. If a material density is given and the mesh is
    watertight, estimate mass from volume — useful for the weight budget that an
    underfunded team has to police obsessively (lightest reliable car wins).
    """
    centroid = mesh.center_mass if mesh.is_watertight else mesh.centroid
    lo, hi = mesh.bounds
    mass_g = None
    if density_kg_m3 is not None and mesh.is_watertight:
        vol_m3 = mesh.volume * 1e-9          # mm^3 -> m^3
        mass_g = vol_m3 * density_kg_m3 * 1000.0
    return PartRecord(team=team, name=name, mass_g=mass_g,
                      centroid=tuple(float(c) for c in centroid),
                      bbox_size=tuple(float(s) for s in (hi - lo)))
