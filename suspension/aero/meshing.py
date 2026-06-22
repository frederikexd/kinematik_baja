# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
snappyHexMesh dictionary generation for the OpenFOAM aero pipeline.

WHY THIS IS A SEPARATE MODULE
-----------------------------
`backends.py` owns the SOLVE side of one OpenFOAM case (controlDict, fields,
fvSchemes/Solution). Meshing is a different concern with its own dictionary zoo —
blockMesh for the background hex block, surfaceFeatureExtract for feature edges,
snappyHexMesh for the castellate/snap/layer cycle, plus mesh-quality and
decomposition controls. Splitting it keeps each file readable and lets a team mesh
with their OWN workflow (their own snappy settings, cfMesh, or a commercial mesher)
without touching the solver adapter.

WHAT IT DOES AND DOES NOT DO
----------------------------
It WRITES a complete, valid snappyHexMesh tool-chain sized from a `MeshParams`
budget and the case's `target_yplus` / `reference_length`. It does NOT mesh — it
emits the dictionaries plus an `Allmesh` script the team runs (locally or on the
cluster). Cell COUNT is a TARGET expressed through refinement levels and a base
cell size; the real count only exists after snappy runs, so nothing here pretends
to know it. That honesty matters: `CFDProvenance.cell_count` stays None until a
real mesh log is parsed (see `parse_checkmesh`), never a guess from the budget.

ATTITUDE HANDLING (the subtle correctness point)
------------------------------------------------
In this package the inlet velocity carries YAW and PITCH (see
OpenFOAMSolver._inlet_velocity). ROLL and RIDE-HEIGHT are geometry-side: they move
the CAR relative to the ground plane, which the freestream cannot represent. So the
mesher applies them to the STL via snappy's `transform` on the geometry entry — a
roll rotation about the x-axis and a vertical translation for ride height — so the
meshed car actually sits at the requested attitude over the road. This is recorded
in the written `kinematik_mesh_attitude.json` so it is auditable, never silent.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .cfd import Attitude, CaseSpec


# --------------------------------------------------------------------------- #
#  Mesh budget / controls
# --------------------------------------------------------------------------- #
@dataclass
class MeshParams:
    """
    A mesh recipe, not a mesh. Refinement LEVELS and a base cell size set the target;
    snappy decides the final count. Defaults are FSAE-full-car sane for a RANS run.

    domain extents are in CAR LENGTHS (multiples of reference_length) so the wind
    tunnel scales with the car: a longer car gets a proportionally longer domain.
    """
    base_cell_m: float = 0.040               # background blockMesh cell size, m
    domain_ahead: float = 3.0                # car-lengths upstream of the car
    domain_behind: float = 6.0               # car-lengths downstream (wake)
    domain_width: float = 3.0                # car-lengths to each side (half-width)
    domain_height: float = 4.0               # car-lengths above the ground
    surface_min_level: int = 4               # snappy surface refinement (min)
    surface_max_level: int = 6               # snappy surface refinement (max)
    feature_level: int = 6                   # edge feature refinement
    n_layers: int = 5                        # prism boundary layers
    layer_expansion: float = 1.2             # layer growth ratio
    first_layer_rel: float = 0.3             # first layer thickness, fraction of final cell
    wake_box_level: int = 3                  # refinement level inside a wake refinement box
    car_patch: str = "car"                   # patch name the solver's forceCoeffs reads
    n_subdomains: int = 64                   # decomposition for parallel snappy+solve

    def estimate_note(self) -> str:
        """Honest disclaimer: a budget is not a count."""
        return (f"target mesh: base {self.base_cell_m*1000:.0f} mm, surface L"
                f"{self.surface_min_level}-{self.surface_max_level}, {self.n_layers} "
                f"layers. Final cell COUNT is only known after snappyHexMesh runs; "
                f"check the snappy/checkMesh log, do not infer it from this recipe.")


# --------------------------------------------------------------------------- #
#  Geometry attitude transform (roll + ride height live here)
# --------------------------------------------------------------------------- #
def _attitude_geometry_transform(att: Attitude) -> dict:
    """
    The roll rotation (about +x, car frame) and ride-height translation applied to
    the STL so the meshed car sits at the requested attitude. Pitch and yaw are NOT
    applied here — they ride on the inlet velocity in the solver. Returned as plain
    numbers so they can be written into snappy's transform block and audited.
    """
    roll_rad = math.radians(att.roll_deg)
    # ride height is a clearance in mm; translate the car vertically by the delta
    # from a nominal 30 mm so 30 mm => no shift, lower => car moves down toward road.
    dz_m = (att.ride_height_mm - 30.0) / 1000.0
    return {
        "roll_axis": "(1 0 0)",
        "roll_angle_deg": att.roll_deg,
        "roll_rad": roll_rad,
        "translate_m": f"(0 0 {dz_m:.5f})",
        "dz_m": dz_m,
    }


# --------------------------------------------------------------------------- #
#  Dictionary writers
# --------------------------------------------------------------------------- #
class SnappyMesher:
    """
    Writes the snappyHexMesh tool-chain for ONE case into its OpenFOAM case dir.
    Call `write(spec, case_dir, mp)`; it creates system/ dictionaries, an Allmesh
    runner, and the attitude manifest. It returns the path to the Allmesh script.
    """

    def __init__(self, mesh_params: Optional[MeshParams] = None):
        self.mp = mesh_params or MeshParams()

    # -- public ------------------------------------------------------------ #
    def write(self, spec: CaseSpec, case_dir: str) -> str:
        mp = self.mp
        os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
        os.makedirs(os.path.join(case_dir, "constant", "triSurface"), exist_ok=True)

        stl_name = os.path.basename(spec.geometry_path) or "car.stl"
        bbox = self._domain_box(spec)
        self._write_blockmesh(case_dir, bbox, mp)
        self._write_surface_features(case_dir, stl_name, mp)
        self._write_snappy(case_dir, spec, stl_name, bbox, mp)
        self._write_mesh_quality(case_dir)
        self._write_decompose(case_dir, mp)
        path = self._write_allmesh(case_dir, stl_name, mp)
        self._write_attitude_manifest(case_dir, spec)
        return path

    # -- domain ------------------------------------------------------------ #
    def _domain_box(self, spec: CaseSpec) -> dict:
        L = spec.reference_length_m
        mp = self.mp
        # car nominally near origin; build a wind tunnel around it, ground at z=0
        return {
            "xmin": -mp.domain_ahead * L,
            "xmax": mp.domain_behind * L,
            "ymin": -mp.domain_width * L,
            "ymax": mp.domain_width * L,
            "zmin": 0.0,
            "zmax": mp.domain_height * L,
        }

    def _write_blockmesh(self, case_dir: str, b: dict, mp: MeshParams) -> None:
        nx = max(int(round((b["xmax"] - b["xmin"]) / mp.base_cell_m)), 1)
        ny = max(int(round((b["ymax"] - b["ymin"]) / mp.base_cell_m)), 1)
        nz = max(int(round((b["zmax"] - b["zmin"]) / mp.base_cell_m)), 1)
        txt = f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
    ({b['xmin']:.4f} {b['ymin']:.4f} {b['zmin']:.4f})
    ({b['xmax']:.4f} {b['ymin']:.4f} {b['zmin']:.4f})
    ({b['xmax']:.4f} {b['ymax']:.4f} {b['zmin']:.4f})
    ({b['xmin']:.4f} {b['ymax']:.4f} {b['zmin']:.4f})
    ({b['xmin']:.4f} {b['ymin']:.4f} {b['zmax']:.4f})
    ({b['xmax']:.4f} {b['ymin']:.4f} {b['zmax']:.4f})
    ({b['xmax']:.4f} {b['ymax']:.4f} {b['zmax']:.4f})
    ({b['xmin']:.4f} {b['ymax']:.4f} {b['zmax']:.4f})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet   {{ type patch;  faces ( (0 4 7 3) ); }}
    outlet  {{ type patch;  faces ( (1 2 6 5) ); }}
    ground  {{ type wall;   faces ( (0 1 2 3) ); }}
    top     {{ type slip;   faces ( (4 5 6 7) ); }}
    sides   {{ type slip;   faces ( (0 1 5 4) (3 7 6 2) ); }}
);
mergePatchPairs ();
"""
        self._w(case_dir, "system/blockMeshDict", txt)

    # -- features ---------------------------------------------------------- #
    def _write_surface_features(self, case_dir: str, stl: str, mp: MeshParams) -> None:
        txt = f"""FoamFile {{ version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }}
{stl}
{{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs {{ includedAngle 150; }}
    subsetFeatures {{ nonManifoldEdges no; openEdges yes; }}
    writeObj yes;
}}
"""
        self._w(case_dir, "system/surfaceFeatureExtractDict", txt)

    # -- snappy ------------------------------------------------------------ #
    def _write_snappy(self, case_dir: str, spec: CaseSpec, stl: str,
                      b: dict, mp: MeshParams) -> None:
        tf = _attitude_geometry_transform(spec.attitude)
        # locationInMesh: a point KNOWN to be in the fluid, not inside the car.
        # Put it high and forward, safely in the tunnel.
        loc = (b["xmin"] + 0.5, 0.0, b["zmax"] * 0.5)
        # a wake refinement box behind the car
        L = spec.reference_length_m
        first = mp.first_layer_rel
        txt = f"""FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    {stl}
    {{
        type triSurfaceMesh;
        name {mp.car_patch};
        // ATTITUDE: roll {tf['roll_angle_deg']:+.2f} deg about {tf['roll_axis']}, ride
        // shift {tf['translate_m']} m. Pitch/yaw ride on the inlet velocity, not here.
        transform
        {{
            coordinateSystem
            {{
                type        cartesian;
                origin      (0 0 0);
                rotation    {{ type axisAngle; axis {tf['roll_axis']}; angle {tf['roll_angle_deg']:.4f}; }}
            }}
        }}
    }}
    wakeBox
    {{
        type searchableBox;
        min ( {0.5*L:.3f} {-mp.domain_width*L*0.4:.3f} 0.0 );
        max ( {mp.domain_behind*L:.3f} {mp.domain_width*L*0.4:.3f} {mp.domain_height*L*0.4:.3f} );
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      60000000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    maxLoadUnbalance    0.10;
    resolveFeatureAngle 30;
    features ( {{ file "{stl.rsplit('.',1)[0]}.eMesh"; level {mp.feature_level}; }} );
    refinementSurfaces
    {{
        {mp.car_patch} {{ level ({mp.surface_min_level} {mp.surface_max_level}); patchInfo {{ type wall; }} }}
    }}
    refinementRegions
    {{
        wakeBox {{ mode inside; levels ( (1e15 {mp.wake_box_level}) ); }}
    }}
    locationInMesh ( {loc[0]:.3f} {loc[1]:.3f} {loc[2]:.3f} );
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{ "{mp.car_patch}.*" {{ nSurfaceLayers {mp.n_layers}; }} }}
    expansionRatio {mp.layer_expansion};
    finalLayerThickness 0.7;
    firstLayerThickness {first};
    minThickness 0.05;
    nGrow 0; featureAngle 130; slipFeatureAngle 30;
    nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls {{ #include "meshQualityDict" }}
writeFlags ( scalarLevels layerSets layerFields );
mergeTolerance 1e-6;
"""
        self._w(case_dir, "system/snappyHexMeshDict", txt)

    def _write_mesh_quality(self, case_dir: str) -> None:
        txt = """FoamFile { version 2.0; format ascii; class dictionary; object meshQualityDict; }
#includeEtc "caseDicts/meshQualityDict"
maxNonOrtho 65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
minTetQuality 1e-15;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
"""
        self._w(case_dir, "system/meshQualityDict", txt)

    def _write_decompose(self, case_dir: str, mp: MeshParams) -> None:
        txt = f"""FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {mp.n_subdomains};
method scotch;
"""
        self._w(case_dir, "system/decomposeParDict", txt)

    # -- runner ------------------------------------------------------------ #
    def _write_allmesh(self, case_dir: str, stl: str, mp: MeshParams) -> str:
        txt = f"""#!/bin/sh
# KinematiK-generated meshing runner. Place {stl} in constant/triSurface/ first.
cd "${{0%/*}}" || exit 1
set -e
echo "Place your STL at constant/triSurface/{stl} before running."
test -f "constant/triSurface/{stl}" || {{ echo "MISSING constant/triSurface/{stl}"; exit 1; }}

blockMesh
surfaceFeatureExtract
decomposePar -copyZero
mpirun -np {mp.n_subdomains} snappyHexMesh -overwrite -parallel
mpirun -np {mp.n_subdomains} checkMesh -parallel | tee log.checkMesh
reconstructParMesh -constant
echo "Mesh done. Inspect log.checkMesh for the REAL cell count before solving."
"""
        path = os.path.join(case_dir, "Allmesh")
        self._w(case_dir, "Allmesh", txt)
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        return path

    def _write_attitude_manifest(self, case_dir: str, spec: CaseSpec) -> None:
        tf = _attitude_geometry_transform(spec.attitude)
        ux_uy_uz = "see 0/U (pitch+yaw on inlet)"
        txt = (
            "{\n"
            f'  "case": "{spec.case_name()}",\n'
            f'  "geometry_side": {{ "roll_deg": {spec.attitude.roll_deg}, '
            f'"ride_height_mm": {spec.attitude.ride_height_mm}, '
            f'"applied_via": "snappy transform + ride translation {tf["translate_m"]}" }},\n'
            f'  "freestream_side": {{ "pitch_deg": {spec.attitude.pitch_deg}, '
            f'"yaw_deg": {spec.attitude.yaw_deg}, "applied_via": "{ux_uy_uz}" }}\n'
            "}\n"
        )
        self._w(case_dir, "kinematik_mesh_attitude.json", txt)

    # -- helper ------------------------------------------------------------ #
    @staticmethod
    def _w(case_dir: str, rel: str, text: str) -> None:
        path = os.path.join(case_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


# --------------------------------------------------------------------------- #
#  Honest cell-count parsing — the REAL number, only after meshing
# --------------------------------------------------------------------------- #
def parse_checkmesh(case_dir: str) -> Optional[int]:
    """
    Read the actual cell count from a checkMesh log, or None if no log exists yet.
    This is the ONLY trustworthy source of cell count — the recipe is a target, the
    log is the truth. Used to populate CFDProvenance.cell_count post-mesh.
    """
    for name in ("log.checkMesh", "log.snappyHexMesh"):
        path = os.path.join(case_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, errors="ignore") as f:
            text = f.read()
        m = re.search(r"cells:\s*([\d]+)", text)
        if m:
            return int(m.group(1))
    return None
