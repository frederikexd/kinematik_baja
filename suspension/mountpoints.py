# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Geometric mount-point layer — the CAD↔clash↔CG chain the integration ledger implied
but didn't close.

`interfaces.py` owns the *interface* between subsystems: each declares a bounding-box
envelope, a mass+CG, and a peak mount load, and the checker validates fit/budget/load.
What it does NOT yet own is the thing an aero member actually does day to day: move a
single wing mounting point a few millimetres and need to know, *immediately*,

    1. does that point now clash with — or run too close to — a chassis keep-out?
       (the clearance clash in the chassis engineer's master file), and
    2. what did it do to the car's mass/CG roll-up the vehicle-dynamics ledger uses?

This module adds exactly that, and nothing more. It is deliberately NOT a CAD kernel
and NOT an FEA tool — same non-goal the rest of KinematiK keeps. It works on explicit
points and explicit keep-out boxes the subteams declare, runs an analytic
point-to-box distance (zero dependencies beyond numpy), and emits the same typed
`Finding` objects the rest of the integration board already renders, so a clash shows
up in the existing UI with both owners named.

Provenance is preserved: a `MountPoint` carries who set it and whether it's an estimate,
and a clash on placeholder geometry is flagged as such rather than presented as final.

The CG half is not re-derived here — it delegates to `IntegrationLedger.mass_rollup()`,
the single source of truth — so "move a point" and "recompute CG" can't drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .interfaces import Finding, Severity, IntegrationLedger, SubsystemInterface


# --------------------------------------------------------------------------- #
#  Geometry primitives
# --------------------------------------------------------------------------- #
@dataclass
class MountPoint:
    """
    A single hardpoint in car coordinates (mm), owned by one subsystem and bolting
    onto another. This is the wing mount the aero member drags.

    Coordinate convention matches SubsystemInterface CG fields:
        x +rearward from front axle, y +right of centreline, z +up from ground.

    `min_clearance_mm` is the gap this point must keep from any keep-out volume it is
    NOT allowed to enter — i.e. a soft margin on top of hard interference. A point
    inside a keep-out is a FAIL (interference); a point closer than the clearance but
    outside is a WARN.
    """
    name: str
    xyz_mm: tuple                  # (x, y, z)
    owner_subsystem: str           # who owns/moves this point, e.g. "aerodynamics"
    mounts_on: str = "chassis"     # whose structure it attaches to
    min_clearance_mm: float = 5.0  # required gap from keep-outs it must avoid
    is_estimate: bool = True
    set_by: str = ""
    notes: str = ""

    def as_array(self) -> np.ndarray:
        return np.asarray(self.xyz_mm, dtype=float)

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "MountPoint":
        d = dict(d)
        if isinstance(d.get("xyz_mm"), list):
            d["xyz_mm"] = tuple(d["xyz_mm"])
        valid = MountPoint.__dataclass_fields__.keys()
        return MountPoint(**{k: v for k, v in d.items() if k in valid})


@dataclass
class KeepOut:
    """
    An axis-aligned volume another subsystem reserves — the chassis tube path, the
    driver's leg room, the cooling duct, the accumulator box. A mount point that lands
    inside it interferes; one within its owner's clearance band is too close.

    Defined by two opposite corners (min, max) in the same car coordinates as points.
    """
    name: str
    owner_subsystem: str            # whose master file reserves this volume
    lo_mm: tuple                    # (x,y,z) min corner
    hi_mm: tuple                    # (x,y,z) max corner
    is_estimate: bool = True
    notes: str = ""

    def __post_init__(self):
        lo = np.minimum(np.asarray(self.lo_mm, float), np.asarray(self.hi_mm, float))
        hi = np.maximum(np.asarray(self.lo_mm, float), np.asarray(self.hi_mm, float))
        self.lo_mm = tuple(lo.tolist())
        self.hi_mm = tuple(hi.tolist())

    def signed_distance_mm(self, p: np.ndarray) -> float:
        """
        Analytic signed distance from point p to this box (axis-aligned).
          > 0  : p is outside, value = nearest-surface gap
          == 0 : on the surface
          < 0  : p is inside, value = -(depth to nearest face)  [interference]
        Standard exact AABB SDF — no sampling, no dependencies beyond numpy.
        """
        lo = np.asarray(self.lo_mm, float)
        hi = np.asarray(self.hi_mm, float)
        # distance component outside the box on each axis
        d_out = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        outside = float(np.linalg.norm(d_out))
        if outside > 0.0:
            return outside
        # inside: negative of distance to the closest face
        inside = float(np.min(np.minimum(p - lo, hi - p)))
        return -inside

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "KeepOut":
        d = dict(d)
        for k in ("lo_mm", "hi_mm"):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        valid = KeepOut.__dataclass_fields__.keys()
        return KeepOut(**{k: v for k, v in d.items() if k in valid})


# --------------------------------------------------------------------------- #
#  The geometric ledger: points + keep-outs, with clash + CG propagation
# --------------------------------------------------------------------------- #
@dataclass
class GeometryLedger:
    """
    Holds the mount points the subteams move and the keep-out volumes their master
    files reserve, and runs the clash check that closes the CAD→clash half of the
    chain. Pairs with an IntegrationLedger for the clash→CG half.

    A point is NOT checked against a keep-out owned by its own subsystem, nor against
    the structure it legitimately mounts onto (you expect a wing bracket to touch the
    chassis tab it bolts to) UNLESS that keep-out is explicitly flagged hard. By
    default `mounts_on` is treated as an allowed contact; everything else is a
    keep-out the point must clear.
    """
    points: dict = field(default_factory=dict)     # name -> MountPoint
    keepouts: dict = field(default_factory=dict)   # name -> KeepOut

    def set_point(self, mp: MountPoint):
        self.points[mp.name] = mp

    def set_keepout(self, ko: KeepOut):
        self.keepouts[ko.name] = ko

    def move_point(self, name: str, xyz_mm: tuple, set_by: str = "") -> "MountPoint":
        """Move an existing point (the wing-mount drag). Returns the updated point."""
        mp = self.points[name]
        mp.xyz_mm = tuple(float(v) for v in xyz_mm)
        if set_by:
            mp.set_by = set_by
        return mp

    # ---- the clash check ---------------------------------------------------- #
    def check_clashes(self) -> list:
        """
        For every mount point, against every keep-out it is required to clear, emit a
        Finding. Interference (inside the box) is FAIL; closer than the point's
        clearance but outside is WARN; otherwise OK. Both owners are named.
        """
        out: list = []
        if not self.points:
            return out
        if not self.keepouts:
            out.append(Finding(
                "clash", Severity.MISSING,
                "Mount points are declared but no keep-out volumes exist to check "
                "them against — no clearance check is possible yet.",
                subsystems=sorted({mp.owner_subsystem for mp in self.points.values()})))
            return out

        for mp in self.points.values():
            p = mp.as_array()
            for ko in self.keepouts.values():
                # skip self-owned volumes and the structure this point legitimately
                # mounts onto (expected contact, not a clash)
                if ko.owner_subsystem == mp.owner_subsystem:
                    continue
                if ko.owner_subsystem == mp.mounts_on:
                    continue
                d = ko.signed_distance_mm(p)
                est = mp.is_estimate or ko.is_estimate
                tag = " (estimated geometry)" if est else ""
                pair = sorted({mp.owner_subsystem, ko.owner_subsystem})
                if d < 0.0:
                    out.append(Finding(
                        "clash-interference", Severity.FAIL,
                        f"{mp.owner_subsystem}'s '{mp.name}' is {(-d):.1f} mm INSIDE "
                        f"{ko.owner_subsystem}'s keep-out '{ko.name}' — hard "
                        f"interference{tag}. {ko.owner_subsystem}'s master file "
                        f"clashes here.",
                        subsystems=pair,
                        detail=dict(point=mp.name, keepout=ko.name,
                                    penetration_mm=-d, clearance_req_mm=mp.min_clearance_mm,
                                    estimate=est)))
                elif d < mp.min_clearance_mm - 1e-9:
                    out.append(Finding(
                        "clash-clearance", Severity.WARN,
                        f"{mp.owner_subsystem}'s '{mp.name}' clears "
                        f"{ko.owner_subsystem}'s '{ko.name}' by only {d:.1f} mm "
                        f"(needs {mp.min_clearance_mm:.0f} mm){tag}.",
                        subsystems=pair,
                        detail=dict(point=mp.name, keepout=ko.name,
                                    gap_mm=d, clearance_req_mm=mp.min_clearance_mm,
                                    estimate=est)))
        if not out:
            out.append(Finding(
                "clash", Severity.OK,
                f"All {len(self.points)} mount point(s) clear every keep-out with "
                f"required margin.",
                subsystems=sorted({mp.owner_subsystem for mp in self.points.values()})))
        return out

    def as_dict(self):
        return dict(points={k: v.as_dict() for k, v in self.points.items()},
                    keepouts={k: v.as_dict() for k, v in self.keepouts.items()})

    @staticmethod
    def from_dict(d) -> "GeometryLedger":
        d = d or {}
        gl = GeometryLedger()
        for k, v in (d.get("points") or {}).items():
            gl.set_point(MountPoint.from_dict(v))
        for k, v in (d.get("keepouts") or {}).items():
            gl.set_keepout(KeepOut.from_dict(v))
        return gl


# --------------------------------------------------------------------------- #
#  The propagation event — the "instantly flags ... and updates CG" one call
# --------------------------------------------------------------------------- #
@dataclass
class PropagationResult:
    """What a single mount-point move did, across both ledgers."""
    moved_point: str
    new_xyz_mm: tuple
    clash_findings: list            # list[Finding] from the geometry check
    cg_before_mm: Optional[tuple]
    cg_after_mm: Optional[tuple]
    cg_delta_mm: Optional[tuple]
    mass_total_kg: float
    notes: list = field(default_factory=list)

    @property
    def has_hard_clash(self) -> bool:
        return any(f.severity == Severity.FAIL for f in self.clash_findings)

    def summary(self) -> str:
        head = f"Moved {self.moved_point} → {tuple(round(v,1) for v in self.new_xyz_mm)} mm. "
        clash = ("HARD CLASH flagged. " if self.has_hard_clash
                 else "no interference. ")
        if self.cg_delta_mm is not None:
            dz = self.cg_delta_mm[2]
            cg = f"CG moved {dz:+.2f} mm in z (now {self.cg_after_mm[2]:.1f} mm)."
        else:
            cg = "CG not recomputable (mass/CG data incomplete)."
        return head + clash + cg


def propagate_mount_move(geom: GeometryLedger,
                         ledger: IntegrationLedger,
                         point_name: str,
                         new_xyz_mm: tuple,
                         set_by: str = "",
                         update_interface_cg: bool = False) -> PropagationResult:
    """
    The single action the original brief described: an aero member moves one wing
    mounting point, and we (1) re-run the clearance clash against the chassis master
    file's keep-outs and (2) re-roll the car CG in the vehicle-dynamics ledger — in
    one call, so they can never be out of sync.

    `update_interface_cg=True` additionally shifts the owning subsystem's declared CG
    by the same delta as the moved point, so the mass roll-up reflects geometry that
    actually moved. Off by default: a single bracket usually isn't the subsystem CG,
    and faking that link would be the false-confidence trap the codebase avoids. Turn
    it on when the point genuinely represents the part's mass location.
    """
    cg_before = ledger.mass_rollup().get("cg_mm")
    old = geom.points[point_name].as_array().copy()
    geom.move_point(point_name, new_xyz_mm, set_by=set_by)
    new = geom.points[point_name].as_array()

    notes: list = []
    if update_interface_cg:
        owner = geom.points[point_name].owner_subsystem
        iface = ledger.get(owner)
        if iface is not None and None not in (iface.cg_x_mm, iface.cg_y_mm, iface.cg_z_mm):
            shift = new - old
            iface.cg_x_mm += float(shift[0])
            iface.cg_y_mm += float(shift[1])
            iface.cg_z_mm += float(shift[2])
            ledger.set(iface)
            notes.append(f"{owner} CG shifted by {tuple(round(s,1) for s in shift)} mm "
                         f"with the point.")
        else:
            notes.append("update_interface_cg requested but owner has no declared CG "
                         "to shift — left unchanged.")

    roll = ledger.mass_rollup()
    cg_after = roll.get("cg_mm")
    cg_delta = (tuple(a - b for a, b in zip(cg_after, cg_before))
                if (cg_before and cg_after) else None)

    clashes = geom.check_clashes()
    return PropagationResult(
        moved_point=point_name, new_xyz_mm=tuple(new.tolist()),
        clash_findings=clashes, cg_before_mm=cg_before, cg_after_mm=cg_after,
        cg_delta_mm=cg_delta, mass_total_kg=roll.get("total_kg", 0.0), notes=notes)
