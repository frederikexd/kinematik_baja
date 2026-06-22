# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Higher-fidelity in-house aero: a 3D source-panel (boundary-element) potential-flow
solver with a ground plane, run on the team's actual STL geometry.

WHY THIS MODULE EXISTS (read this before trusting its numbers)
--------------------------------------------------------------
The default in-house backend (`ReferenceAeroModel`) is an analytic SURROGATE: a
handful of FSAE-plausible sensitivities curve-fitted to attitude. It is honest about
being a stand-in, but its coefficients come from the fit, not from the car's shape —
change the geometry and the numbers do not move. That is fine for plumbing and
trends, useless for "does this new floor make more downforce".

This module is the genuine fidelity step between that surrogate and an external RANS
solve. It SOLVES a flow on the real surface mesh:

  * It reads the STL (via trimesh), places the car at the requested attitude
    (yaw/pitch fold into the onset flow; roll + ride height move the body over the
    road), and treats each triangle as a constant-strength SOURCE panel.
  * It enforces flow tangency (zero normal velocity) on every panel, giving a dense
    linear system A·sigma = -Vinf·n that is solved once per attitude.
  * GROUND EFFECT is modelled correctly, not tuned: an IMAGE of every panel is
    reflected through the road plane (z=0), so the road is an exact streamline. Lower
    the car and the image interaction strengthens — ground effect emerges from the
    physics, it is not a `ride_ground_gain` constant.
  * Surface pressure comes from the computed tangential velocity through Bernoulli
    (Cp = 1 - (V/Vinf)^2). Lift and the pressure (form) part of drag are the surface
    integral of Cp·n. A flat-plate turbulent SKIN-FRICTION estimate is added so total
    C_d is realistic rather than the near-zero a pure potential solve would give.

WHAT IT RESOLVES — AND WHAT IT HONESTLY DOES NOT
------------------------------------------------
Potential flow is inviscid and attached by assumption. So this method captures, from
geometry: ground effect, the pressure field of attached flow, the downforce trend
with rake and ride height, and induced effects. It does NOT capture viscous
SEPARATION, a real turbulent wake, stall, or vortex shedding — exactly the things a
RANS/DES solve and the written ANSYS Fluent deck exist to check. Its fidelity is
therefore labelled `POTENTIAL` (a resolved potential field, well above the analytic
surrogate, well below RANS), and `is_correlated=False`. The provenance says all of
this on every result. Trust deltas between geometries far more than absolute levels,
and correlate against the tunnel/Fluent before reporting an absolute number.

DELIBERATE NON-GOALS: no meshing of a volume, no Navier–Stokes, no turbulence model.
It is a surface BEM on the geometry the team already supplies, sized to run in
seconds so a sweep is interactive.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from .cfd import (
    Attitude, CaseSpec, CoeffResult, CFDProvenance, SolverFidelity,
)


# --------------------------------------------------------------------------- #
#  Tunables for the panel solve (sane FSAE-scale defaults)
# --------------------------------------------------------------------------- #
@dataclass
class PanelParams:
    """
    Knobs for the panel solve. Defaults trade a little accuracy for an interactive
    runtime so a whole ride-height sweep stays in the seconds range.

    max_panels        : the STL is decimated to at most this many triangles before
                        solving (the linear system is dense and O(N^2) memory, so
                        this is the cost lever). None => use the mesh as-is.
    ground_effect     : reflect an image of every panel through z=0 so the road is an
                        exact streamline. The whole point of a ground-effect car.
    road_plane_z_m    : the road height in the STL's own coordinates (default 0.0).
    kin_viscosity     : air kinematic viscosity for the skin-friction Reynolds number.
    laminar_fraction  : fraction of the body length assumed laminar before transition
                        (cuts the flat-plate Cf slightly; 0.0 = fully turbulent).
    min_panels        : below this many usable triangles the geometry is too coarse to
                        trust a solve, and the caller should fall back.
    """
    max_panels: Optional[int] = 4000
    ground_effect: bool = True
    road_plane_z_m: float = 0.0
    kin_viscosity: float = 1.5e-5
    laminar_fraction: float = 0.05
    min_panels: int = 24


class PanelMethodUnavailable(RuntimeError):
    """
    Raised when the panel solve cannot run for a SPECIFIC, reportable reason —
    geometry file missing/empty, trimesh not installed, the surface too coarse, or
    the linear solve failing. Carries an actionable message so the caller can fall
    back to the analytic surrogate transparently rather than fabricating a number.
    """


# --------------------------------------------------------------------------- #
#  The solver
# --------------------------------------------------------------------------- #
class PanelMethodModel:
    """
    A 3D constant-source-panel potential-flow model with a ground image, evaluated on
    the supplied STL. Implements the same `provenance()/write_case/run_case/
    read_result` shape as the other in-house backends, but it is normally used as the
    PHYSICS ENGINE inside a higher-level backend (it does not, by itself, write any
    solver deck). `FluentVerificationSolver(method="panel")` wraps it and adds the
    Fluent verification journal.

    Sign convention (matches cfd.py): c_lift NEGATIVE = downforce.
    """
    name = "panel-method"

    def __init__(self, params: Optional[PanelParams] = None):
        self.params = params or PanelParams()

    # -- provenance -------------------------------------------------------- #
    def provenance(self, n_panels: Optional[int] = None) -> CFDProvenance:
        note = (
            "In-house 3D source-panel (boundary-element) potential-flow solve on the "
            "actual STL, with a ground-image plane. Resolves the attached-flow "
            "pressure field and ground effect FROM GEOMETRY (not a curve fit), so "
            "geometry deltas are meaningful. Inviscid by construction: it does NOT "
            "resolve viscous separation, stall, or the real turbulent wake — that is "
            "what the written ANSYS Fluent deck / a RANS solve is for. Skin friction "
            "is a flat-plate estimate. Correlate against the tunnel/Fluent before "
            "trusting absolute levels; trust deltas more than levels."
        )
        if n_panels:
            note = f"{note} [{n_panels} panels]"
        return CFDProvenance(
            backend=self.name,
            fidelity=SolverFidelity.POTENTIAL,
            is_correlated=False,
            turbulence_model="none (inviscid potential flow + flat-plate friction)",
            cell_count=n_panels,
            notes=note,
        )

    # -- the public physics entry point ------------------------------------ #
    def solve(self, spec: CaseSpec) -> CoeffResult:
        """
        Solve one attitude on the STL and return its coefficients. Raises
        PanelMethodUnavailable (never a fabricated number) if the geometry cannot be
        loaded or is too coarse to solve.
        """
        import numpy as np

        centroids, normals, areas, length_ref = self._load_panels(spec)
        n = len(areas)

        # Onset flow: yaw about +z, pitch about +y, unit magnitude (coeffs are
        # non-dimensional, so we work at |Vinf| = 1 and scale out cleanly).
        vinf = _freestream_unit(spec.attitude)

        # Influence matrix: normal velocity at panel i induced by unit source on
        # panel j (plus its ground image), in a point-source approximation evaluated
        # at panel centroids. A[i,j] = n_i · (u_ij + u_image_ij).
        A = self._influence_matrix(centroids, normals, areas)

        # RHS: cancel the onset normal velocity on every panel (flow tangency).
        rhs = -(normals @ vinf)

        # Solve A·sigma = rhs (least-squares for robustness on imperfect STLs).
        try:
            sigma, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError as e:                  # noqa: BLE001
            raise PanelMethodUnavailable(f"panel linear solve failed: {e}")

        # Surface velocity = onset + induced; pressure from Bernoulli.
        v_ind = self._induced_velocity(centroids, normals, areas, sigma)
        v_surf = v_ind + vinf[None, :]
        # remove the normal component numerically (tangency is enforced only approx.)
        vn = np.einsum("ij,ij->i", v_surf, normals)
        v_tan = v_surf - vn[:, None] * normals
        speed2 = np.einsum("ij,ij->i", v_tan, v_tan)
        cp = 1.0 - speed2                                   # |Vinf| = 1

        # Force coefficient = -(1/Aref) ∮ Cp n dA, then split into lift/drag/side.
        # Aref/Lref come from the spec so the result is comparable to tunnel/Fluent.
        aref = max(spec.reference_area_m2, 1e-9)
        f = -(cp * areas)[:, None] * normals                # per-panel force vector
        F = f.sum(axis=0) / aref                            # [Fx, Fy, Fz] coeff

        # Drag is along the onset flow; lift is vertical (z); side is lateral (y).
        c_drag_pressure = float(F @ vinf)
        c_lift = float(F[2])                                # +z up => negative = downforce
        c_side = float(F[1])

        # Skin friction: flat-plate turbulent estimate over the wetted area, added to
        # the pressure drag so total C_d is physical (potential flow alone gives ~0).
        c_drag_friction = self._friction_cd(spec, areas, aref, length_ref)
        c_drag = c_drag_pressure + c_drag_friction

        # Aero balance: fraction of downforce ahead of the body mid-length. Use the
        # panel x-centroids and their vertical load to split front/rear.
        front = self._aero_balance(centroids, cp, areas, normals)

        return CoeffResult(
            attitude=spec.attitude,
            c_lift=c_lift, c_drag=c_drag, c_side=c_side,
            c_pitch=None,
            aero_balance_front=front,
            converged=True,                 # a direct solve, not an iteration
            force_monitor_range=0.0,
            provenance=self.provenance(n_panels=n),
            notes=(f"panel solve: {n} panels, Cd(pressure)={c_drag_pressure:+.3f} "
                   f"+ Cd(friction)={c_drag_friction:.3f}"),
        )

    # -- CFDSolver-shaped convenience (physics only; no deck) -------------- #
    def run_case(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        return self.solve(spec)

    def read_result(self, spec: CaseSpec, workdir: str) -> CoeffResult:
        return self.solve(spec)

    # ------------------------------------------------------------------ #
    #  Geometry loading + attitude placement
    # ------------------------------------------------------------------ #
    def _load_panels(self, spec: CaseSpec):
        """
        Load the STL, place it at the attitude (roll + ride height move the body;
        yaw/pitch are in the onset flow), decimate to the panel budget, and return
        per-panel centroids, unit normals, areas and a reference length. Raises
        PanelMethodUnavailable for any geometry problem so the caller can fall back.
        """
        import numpy as np

        path = spec.geometry_path
        if not path or not os.path.isfile(path):
            raise PanelMethodUnavailable(
                f"geometry '{path}' not found on disk — the panel method needs a real "
                "STL/surface mesh. Falling back to the analytic estimate.")
        try:
            import trimesh
        except Exception as e:                              # noqa: BLE001
            raise PanelMethodUnavailable(f"trimesh not available: {e}")

        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as e:                              # noqa: BLE001
            raise PanelMethodUnavailable(f"could not load geometry '{path}': {e}")
        if mesh is None or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
            raise PanelMethodUnavailable(f"geometry '{path}' has no triangles to solve")

        # Decimate to the panel budget to keep the dense solve interactive.
        mp = self.params
        if mp.max_panels and len(mesh.faces) > mp.max_panels:
            try:
                mesh = mesh.simplify_quadric_decimation(mp.max_panels)
            except Exception:                              # noqa: BLE001
                # decimation is best-effort; if it fails we solve the full mesh
                pass

        # Place the body at attitude: roll about +x, then ride-height translate in z.
        a = spec.attitude
        T = trimesh.transformations
        roll = T.rotation_matrix(math.radians(a.roll_deg), [1.0, 0.0, 0.0])
        mesh.apply_transform(roll)
        dz = (a.ride_height_mm - 30.0) / 1000.0            # 30 mm nominal, lower = down
        mesh.apply_translation([0.0, 0.0, dz])

        centroids = np.asarray(mesh.triangles_center, dtype=float)
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)

        # Drop degenerate panels (zero area / nan normal).
        good = (areas > 1e-12) & np.isfinite(normals).all(axis=1)
        centroids, normals, areas = centroids[good], normals[good], areas[good]
        if len(areas) < mp.min_panels:
            raise PanelMethodUnavailable(
                f"only {len(areas)} usable panels (< {mp.min_panels}); surface too "
                "coarse for a trustworthy panel solve")

        length_ref = float(spec.reference_length_m) if spec.reference_length_m else \
            float(centroids[:, 0].ptp() or 1.0)
        return centroids, normals, areas, length_ref

    # ------------------------------------------------------------------ #
    #  Source-panel influence (point-source approx + ground image)
    # ------------------------------------------------------------------ #
    def _influence_matrix(self, centroids, normals, areas):
        """
        A[i,j] = normal velocity at panel i from a unit constant source on panel j
        (area-weighted point source at its centroid), plus the contribution of j's
        IMAGE reflected through the road plane when ground_effect is on. Self-term is
        the standard +1/2 source jump on the panel's own normal.
        """
        import numpy as np

        c = centroids
        # r_ij = c_i - c_j  (vector from source j to field point i)
        diff = c[:, None, :] - c[None, :, :]               # (N, N, 3)
        A = self._normal_vel_kernel(diff, normals, areas)

        # Self-influence: a source panel induces +1/2 (area-scaled) on its own normal.
        np.fill_diagonal(A, 0.5)

        if self.params.ground_effect:
            zr = self.params.road_plane_z_m
            c_img = c.copy()
            c_img[:, 2] = 2.0 * zr - c_img[:, 2]           # reflect sources through z=zr
            diff_img = c[:, None, :] - c_img[None, :, :]
            A = A + self._normal_vel_kernel(diff_img, normals, areas)
        return A

    @staticmethod
    def _normal_vel_kernel(diff, normals, areas):
        """n_i · u(r_ij) for a point source of strength = area_j, u = r / (4π |r|^3)."""
        import numpy as np
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + 1e-12  # softened to avoid blowup
        inv = 1.0 / (4.0 * math.pi * np.power(r2, 1.5))
        # velocity vector field = diff * inv * area_j ; dot with field-point normal n_i
        ndotd = np.einsum("ik,ijk->ij", normals, diff)
        return ndotd * inv * areas[None, :]

    @staticmethod
    def _induced_velocity(centroids, normals, areas, sigma):
        """Full induced velocity vector at each centroid (for the surface speed)."""
        import numpy as np
        c = centroids
        diff = c[:, None, :] - c[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + 1e-9
        inv = (sigma * areas)[None, :] / (4.0 * math.pi * np.power(r2, 1.5))
        v = np.einsum("ij,ijk->ik", inv, diff)
        return v

    # ------------------------------------------------------------------ #
    #  Closures: skin friction + aero balance
    # ------------------------------------------------------------------ #
    def _friction_cd(self, spec, areas, aref, length_ref) -> float:
        """
        Flat-plate turbulent skin-friction drag referenced to Aref. Cf via the
        Schlichting 1/7-power correlation Cf = 0.074 Re^-0.2, scaled by wetted/ref
        area, lightly discounted for a laminar run length. Speed scales out of the
        coefficient but Re (hence Cf) depends on it, so we use the case speed.
        """
        import numpy as np
        v = max(float(spec.attitude.speed_ms), 1e-3)
        re = v * length_ref / max(self.params.kin_viscosity, 1e-12)
        cf = 0.074 / (re ** 0.2) if re > 1.0 else 0.01
        cf *= (1.0 - 0.85 * self.params.laminar_fraction)   # small transition discount
        wetted = float(np.sum(areas))
        return cf * wetted / max(aref, 1e-9)

    @staticmethod
    def _aero_balance(centroids, cp, areas, normals) -> Optional[float]:
        """
        Fraction of vertical aero load carried ahead of the body mid-length. Uses the
        per-panel vertical pressure load (Cp · area · n_z); returns None if there is
        effectively no vertical load to split.
        """
        import numpy as np
        x = centroids[:, 0]
        x_mid = 0.5 * (x.min() + x.max())
        load = -(cp * areas) * normals[:, 2]               # +ve = downforce contribution
        total = float(np.sum(load))
        if abs(total) < 1e-9:
            return None
        front = float(np.sum(load[x > x_mid]))             # +x is nose-forward
        frac = front / total
        return float(min(max(frac, 0.0), 1.0))


def _freestream_unit(att: Attitude):
    """Unit onset-flow vector with yaw (about +z) and pitch (about +y) folded in."""
    import numpy as np
    yaw = math.radians(att.yaw_deg)
    pitch = math.radians(att.pitch_deg)
    ux = math.cos(yaw) * math.cos(pitch)
    uy = -math.sin(yaw)
    uz = math.cos(yaw) * math.sin(pitch)
    v = np.array([ux, uy, uz], dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n > 1e-12 else 1.0)
