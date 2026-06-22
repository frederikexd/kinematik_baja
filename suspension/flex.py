# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Flexible bodies — finite-element compliance for the suspension links (ADAMS Flex–style).

WHY THIS EXISTS
---------------
The rest of KinematiK's kinematics solver (kinematics.py) assumes every control
arm, pushrod, tie rod and chassis tab is INFINITELY STIFF: link lengths are frozen
at static ride height and enforced as hard constraints. That is the right model for
geometry design, but it is a lie under load. An FSAE car cornering at ~1.5 g pushes
several kN through each wishbone leg and the tie rod; real steel/aluminium tubes and
the chassis tabs they bolt to deflect by tenths of a millimetre to a couple of
millimetres, and those deflections steer and camber the wheel — "compliance steer"
and "compliance camber". That is exactly the effect teams chase when the car tramlines
under brakes or the toe goes away mid-corner, and it is invisible to a rigid solver.

This module supplies the missing piece: the COMPLIANCE of a component, either

  (a) analytically, from material + tube section (k_axial = E·A/L) — the path a team
      with no FEA can use today, because they know their tube sizes; or

  (b) from a finite-element model of the actual part ("ADAMS Flex"): import a mesh of
      the A-arm / upright / chassis bracket, and we statically condense it down to the
      stiffness felt at its attachment (interface) points. That condensed boundary
      stiffness is exactly the static content an ADAMS/Car flexible body (a Craig–
      Bampton MNF) carries — the constraint-mode partition — and for the QUASI-STEADY
      cornering case it is the physically exact reduction. We do NOT fabricate the
      dynamic/modal content (the fixed-interface normal modes used for NVH/transient
      response); that needs the body's mass distribution and a transient solver, and
      inventing it would be the same false-confidence failure the rest of this codebase
      refuses. What we import and use is the part of the flexible-body data that
      actually governs load↔deflection in a sustained corner.

WHAT "STATIC CONDENSATION" MEANS HERE
-------------------------------------
Partition the assembled FE stiffness K into interface ("master", m) DOFs — the bolt
holes / ball-joint centres where the part connects to the rest of the car — and
internal ("slave", s) DOFs:

        | K_mm  K_ms | | u_m |   | f_m |
        | K_sm  K_ss | | u_s | = |  0  |     (no loads applied to internal nodes)

Eliminating u_s = -K_ss⁻¹ K_sm u_m gives the condensed (Guyan) stiffness

        K_red = K_mm - K_ms K_ss⁻¹ K_sm

which reproduces the EXACT static load–deflection of the full part at its interface
points. From K_red we read the quantity the corner model needs: how much the two
mounts of a link move apart per unit axial load (its real axial give), replacing the
E·A/L idealisation with the measured part.

ELEMENTS
--------
Suspension links are slender tubes, so the correct and honest finite element is a
3D beam/bar, not a solid tet mesh we couldn't validate. We assemble:
  * BAR  (truss)  — axial stiffness only, the right model for a rod-end-to-rod-end
                    link that carries load along its length;
  * BEAM (frame)  — full 3D Euler–Bernoulli element (axial + torsion + bending about
                    both axes), the right model for a welded/bonded tube structure
                    or a bracket loaded off-axis.
For brackets and uprights that are genuinely solid, import the reduced superelement
your FEA package already produces (from_superelement_*) — that is the standard ADAMS
Flex workflow and we consume it directly.

All lengths mm, forces N, so stiffness comes out N/mm; moduli are given in N/mm² (MPa),
which is the consistent unit set the rest of the tool uses.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
#  Materials (E, G in N/mm² = MPa; rho in kg/m³ for completeness)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Material:
    name: str
    E: float            # Young's modulus, MPa (N/mm²)
    G: float            # shear modulus, MPa
    rho: float = 0.0    # density, kg/m³ (not needed for static compliance)


# A small library covering the materials FSAE suspension links are actually made of.
MATERIALS = {
    "Steel 4130": Material("Steel 4130", E=205000.0, G=80000.0, rho=7850.0),
    "Steel mild": Material("Steel mild", E=200000.0, G=79000.0, rho=7850.0),
    "Aluminium 6061": Material("Aluminium 6061", E=68900.0, G=26000.0, rho=2700.0),
    "Aluminium 7075": Material("Aluminium 7075", E=71700.0, G=26900.0, rho=2810.0),
    "Titanium Ti-6Al-4V": Material("Titanium Ti-6Al-4V", E=113800.0, G=44000.0, rho=4430.0),
    # Carbon is laminate-dependent; this is a representative axial modulus for a
    # quasi-iso tube and is clearly a stand-in — a real CFRP link should be imported
    # as a condensed superelement from its laminate FEA, not modelled as isotropic.
    "Carbon (axial, representative)": Material("Carbon (axial, representative)",
                                               E=70000.0, G=5000.0, rho=1600.0),
}


def tube_section(od_mm: float, wall_mm: float) -> tuple:
    """
    Section properties of a round tube: (A, I, J) in mm², mm⁴, mm⁴.

    A   = cross-sectional area
    I   = second moment of area (same about both axes for a round tube)
    J   = polar second moment (= 2I for a thin/thick round tube)
    """
    if od_mm <= 0 or wall_mm <= 0 or 2 * wall_mm >= od_mm:
        raise ValueError(f"Invalid tube: OD={od_mm} wall={wall_mm} "
                         "(need OD>0, wall>0, 2·wall<OD).")
    ro = od_mm / 2.0
    ri = ro - wall_mm
    A = np.pi * (ro ** 2 - ri ** 2)
    I = np.pi / 4.0 * (ro ** 4 - ri ** 4)
    J = 2.0 * I
    return float(A), float(I), float(J)


def solid_rod_section(od_mm: float) -> tuple:
    """Section properties of a solid round rod: (A, I, J)."""
    ro = od_mm / 2.0
    A = np.pi * ro ** 2
    I = np.pi / 4.0 * ro ** 4
    J = 2.0 * I
    return float(A), float(I), float(J)


def axial_stiffness_tube(material: Material, length_mm: float,
                         od_mm: float, wall_mm: float) -> float:
    """
    Closed-form axial stiffness k = E·A/L (N/mm) of a straight tube link.

    This is the analytic compliance path: a team that knows its tube material and
    size gets a real, defensible link stiffness with no FEA at all. It is exact for
    a straight, axially-loaded prismatic member — which is what a rod-end suspension
    link is to first order.
    """
    A, _, _ = tube_section(od_mm, wall_mm)
    if length_mm <= 0:
        raise ValueError("Link length must be positive.")
    return float(material.E * A / length_mm)


# --------------------------------------------------------------------------- #
#  Element stiffness matrices (local frame)
# --------------------------------------------------------------------------- #
def _bar_local_K(E: float, A: float, L: float) -> np.ndarray:
    """6×6 axial-only (truss) element in its local frame (x along the bar)."""
    k = E * A / L
    K = np.zeros((6, 6))
    K[0, 0] = K[3, 3] = k
    K[0, 3] = K[3, 0] = -k
    return K


def _beam_local_K(E: float, G: float, A: float, Iy: float, Iz: float,
                  J: float, L: float) -> np.ndarray:
    """
    12×12 3D Euler–Bernoulli beam element in its local frame.

    DOF order per node: [u, v, w, θx, θy, θz]; node1 then node2. Local x is along
    the element, y and z are the two bending principal axes. For a round tube
    Iy == Iz and J == 2I, but the matrix is written for the general case.
    """
    K = np.zeros((12, 12))
    # axial
    a = E * A / L
    K[0, 0] += a; K[6, 6] += a; K[0, 6] += -a; K[6, 0] += -a
    # torsion
    t = G * J / L
    K[3, 3] += t; K[9, 9] += t; K[3, 9] += -t; K[9, 3] += -t
    # bending about z (deflection v, in local x-y plane)  -> DOFs v(1,7) θz(5,11)
    ez = E * Iz
    K[1, 1] += 12 * ez / L ** 3
    K[1, 5] += 6 * ez / L ** 2
    K[1, 7] += -12 * ez / L ** 3
    K[1, 11] += 6 * ez / L ** 2
    K[5, 5] += 4 * ez / L
    K[5, 7] += -6 * ez / L ** 2
    K[5, 11] += 2 * ez / L
    K[7, 7] += 12 * ez / L ** 3
    K[7, 11] += -6 * ez / L ** 2
    K[11, 11] += 4 * ez / L
    # symmetrise the z-bending block
    for i, j in [(5, 1), (7, 1), (11, 1), (7, 5), (11, 5), (11, 7)]:
        K[i, j] = K[j, i]
    # bending about y (deflection w, in local x-z plane)  -> DOFs w(2,8) θy(4,10)
    # note the sign convention difference vs the z block (θy couples to -w slope)
    ey = E * Iy
    K[2, 2] += 12 * ey / L ** 3
    K[2, 4] += -6 * ey / L ** 2
    K[2, 8] += -12 * ey / L ** 3
    K[2, 10] += -6 * ey / L ** 2
    K[4, 4] += 4 * ey / L
    K[4, 8] += 6 * ey / L ** 2
    K[4, 10] += 2 * ey / L
    K[8, 8] += 12 * ey / L ** 3
    K[8, 10] += 6 * ey / L ** 2
    K[10, 10] += 4 * ey / L
    for i, j in [(4, 2), (8, 2), (10, 2), (8, 4), (10, 4), (10, 8)]:
        K[i, j] = K[j, i]
    return K


def _element_transform(p1: np.ndarray, p2: np.ndarray, dofs: int,
                       ref: Optional[np.ndarray] = None) -> tuple:
    """
    Direction-cosine transform T (block-diagonal R) and length for an element.

    `dofs` is 3 (bar: translations only) or 6 (beam: translations + rotations).
    `ref` is an optional 'up' reference to fix the local y/z orientation of a beam;
    irrelevant for a bar and for a round (isotropic) tube's axial response.
    """
    d = np.asarray(p2, float) - np.asarray(p1, float)
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        raise ValueError("Zero-length element.")
    x = d / L
    # pick a stable local y perpendicular to x
    if ref is not None:
        up = np.asarray(ref, float)
    else:
        up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x, up)) > 0.99:           # x nearly parallel to up -> use another
        up = np.array([0.0, 1.0, 0.0])
    z = np.cross(x, up)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])                # rows = local axes in global coords
    nblk = 2 if dofs == 3 else 4            # blocks of 3 to fill
    T = np.zeros((dofs * 2, dofs * 2))
    for b in range(nblk):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return T, L


# --------------------------------------------------------------------------- #
#  Mesh container + assembly
# --------------------------------------------------------------------------- #
@dataclass
class FlexElement:
    n1: str                      # node id
    n2: str                      # node id
    kind: str = "beam"           # "beam" | "bar"
    material: str = "Steel 4130"
    # section: either tube (od/wall) or explicit (A, I, J)
    od_mm: Optional[float] = None
    wall_mm: Optional[float] = None
    A: Optional[float] = None
    I: Optional[float] = None
    J: Optional[float] = None

    def section_props(self) -> tuple:
        if self.A is not None and self.I is not None:
            J = self.J if self.J is not None else 2.0 * self.I
            return float(self.A), float(self.I), float(J)
        if self.od_mm is not None and self.wall_mm is not None:
            return tube_section(self.od_mm, self.wall_mm)
        if self.od_mm is not None:
            return solid_rod_section(self.od_mm)
        raise ValueError(f"Element {self.n1}-{self.n2} has no section "
                         "(give od/wall, od, or A/I/J).")


class FlexMesh:
    """
    A finite-element mesh of one component (an A-arm, a bracket, an upright…).

    nodes        : dict  node_id -> (x, y, z) in mm
    elements     : list of FlexElement
    interface    : dict  interface_name -> node_id   (the attachment points the rest
                   of the car connects to — ball joints, bolt holes, pickups)

    The mesh is homogeneous in element type: all 'bar' (a pin-jointed truss model)
    or all 'beam' (a frame model). Mixing the two in one assembly leaves the
    rotational DOFs of bar-only nodes unconstrained (a singular system); if you need
    a genuinely mixed/solid part, import its reduced superelement instead.
    """

    def __init__(self, nodes: dict, elements: list, interface: dict):
        self.nodes = {str(k): np.asarray(v, float) for k, v in nodes.items()}
        self.elements = list(elements)
        self.interface = {str(k): str(v) for k, v in interface.items()}
        self._validate()

    def _validate(self):
        if not self.elements:
            raise ValueError("Mesh has no elements.")
        kinds = {e.kind for e in self.elements}
        if kinds - {"bar", "beam"}:
            raise ValueError(f"Unknown element kind(s): {kinds - {'bar', 'beam'}}")
        if len(kinds) > 1:
            raise ValueError("Mesh mixes 'bar' and 'beam' elements. Use one type, "
                             "or import a reduced superelement for a mixed/solid part.")
        for e in self.elements:
            for n in (e.n1, e.n2):
                if str(n) not in self.nodes:
                    raise ValueError(f"Element references unknown node '{n}'.")
        for name, nid in self.interface.items():
            if nid not in self.nodes:
                raise ValueError(f"Interface '{name}' points at unknown node '{nid}'.")
        if not self.interface:
            raise ValueError("Mesh declares no interface nodes — there is nothing to "
                             "condense to.")

    @property
    def dofs_per_node(self) -> int:
        return 3 if self.elements[0].kind == "bar" else 6

    def assemble(self) -> tuple:
        """
        Assemble the global stiffness K and return (K, node_index, dofs_per_node).

        node_index maps node_id -> its block index (0..N-1); global DOF j of node i
        is i*dofs_per_node + j.
        """
        dofs = self.dofs_per_node
        node_ids = list(self.nodes.keys())
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids) * dofs
        K = np.zeros((N, N))

        for e in self.elements:
            mat = MATERIALS.get(e.material)
            if mat is None:
                raise ValueError(f"Unknown material '{e.material}'. Known: "
                                 f"{list(MATERIALS)}")
            A, I, J = e.section_props()
            p1, p2 = self.nodes[e.n1], self.nodes[e.n2]
            T, L = _element_transform(p1, p2, dofs)
            if e.kind == "bar":
                kl = _bar_local_K(mat.E, A, L)
            else:
                kl = _beam_local_K(mat.E, mat.G, A, I, I, J, L)
            kg = T.T @ kl @ T
            idx = []
            for nid in (e.n1, e.n2):
                base = node_index[nid] * dofs
                idx.extend(range(base, base + dofs))
            idx = np.array(idx)
            K[np.ix_(idx, idx)] += kg
        return K, node_index, dofs

    def condense(self) -> "CondensedFlexBody":
        """Assemble and Guyan-condense to the interface DOFs."""
        K, node_index, dofs = self.assemble()
        master_nodes = list(self.interface.values())
        master_dofs = []
        for nid in master_nodes:
            base = node_index[nid] * dofs
            master_dofs.extend(range(base, base + dofs))
        K_red = guyan_condense(K, master_dofs)
        names = list(self.interface.keys())
        xyz = np.array([self.nodes[self.interface[n]] for n in names])
        return CondensedFlexBody(names, xyz, K_red, dofs, source="mesh")


def guyan_condense(K: np.ndarray, master_dofs: list) -> np.ndarray:
    """
    Static (Guyan) condensation of K onto `master_dofs`.

        K_red = K_mm - K_ms K_ss⁻¹ K_sm

    Returns the condensed stiffness in the order of `master_dofs`. Raises if the
    internal partition K_ss is singular (an internal mechanism / unconstrained
    DOF), because the condensation would be meaningless — better a clear error than
    a silently wrong number.
    """
    n = K.shape[0]
    master = list(master_dofs)
    mset = set(master)
    slave = [i for i in range(n) if i not in mset]
    Kmm = K[np.ix_(master, master)]
    if not slave:
        return Kmm.copy()
    Kms = K[np.ix_(master, slave)]
    Ksm = K[np.ix_(slave, master)]
    Kss = K[np.ix_(slave, slave)]
    # Detect an ill-posed internal partition before inverting.
    try:
        cond = np.linalg.cond(Kss)
    except np.linalg.LinAlgError:
        cond = np.inf
    if not np.isfinite(cond) or cond > 1e14:
        raise ValueError(
            "Internal stiffness is singular/ill-conditioned (cond≈"
            f"{cond:.1e}). The mesh has an internal mechanism or an unconstrained "
            "DOF once the interface is fixed — check connectivity, or use beam "
            "elements (a bar leaves interior nodes free in shear/bending).")
    Kss_inv_Ksm = np.linalg.solve(Kss, Ksm)
    K_red = Kmm - Kms @ Kss_inv_Ksm
    return 0.5 * (K_red + K_red.T)          # clean tiny asymmetry from round-off


# --------------------------------------------------------------------------- #
#  Condensed flexible body — the object the corner model consumes
# --------------------------------------------------------------------------- #
@dataclass
class CondensedFlexBody:
    """
    A component reduced to the stiffness felt at its interface (attachment) nodes.

    This is the static equivalent of an ADAMS Flex body for load↔deflection: feed it
    the loads at its mounts and it returns how those mounts move. It is built either
    by condensing a mesh (FlexMesh.condense) or imported directly as a reduced
    superelement — the latter is what a Nastran/ANSYS CMS reduction (the matrices an
    MNF carries) gives you, and we use it verbatim.

    names : interface names, in the row order of K_red's node blocks
    xyz   : (M,3) interface node coordinates, mm
    K_red : (M·dofs, M·dofs) condensed stiffness, N/mm (and N·mm/rad for rotations)
    dofs  : DOFs per interface node (3 = translations only, 6 = +rotations)
    """
    names: list
    xyz: np.ndarray
    K_red: np.ndarray
    dofs: int = 6
    source: str = "reduced"

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, float)
        self.K_red = np.asarray(self.K_red, float)
        M = len(self.names)
        if self.K_red.shape != (M * self.dofs, M * self.dofs):
            raise ValueError(
                f"K_red shape {self.K_red.shape} doesn't match {M} interface nodes "
                f"× {self.dofs} DOF.")
        self._index = {n: i for i, n in enumerate(self.names)}

    # ---- DOF helpers --------------------------------------------------- #
    def _trans_dofs(self, name: str) -> np.ndarray:
        if name not in self._index:
            raise KeyError(f"Interface node '{name}' not in flex body "
                           f"{list(self.names)}.")
        base = self._index[name] * self.dofs
        return np.array([base, base + 1, base + 2])

    def _solve_free(self, f: np.ndarray) -> np.ndarray:
        """
        Solve K_red u = f for a free-floating (unconstrained) reduced body.

        K_red of a free part is singular (it has rigid-body modes), so a general f
        has no solution. But a SELF-EQUILIBRATED load (zero net force and moment —
        e.g. an equal-and-opposite axial pair across a link) is orthogonal to the
        rigid modes and yields a well-defined RELATIVE deflection. The minimum-norm
        least-squares solution (pinv) returns exactly that, with the arbitrary
        rigid-body component set to zero.
        """
        u, *_ = np.linalg.lstsq(self.K_red, f, rcond=None)
        return u

    # ---- the numbers the corner model needs --------------------------- #
    def relative_axial_stiffness(self, name_a: str, name_b: str) -> float:
        """
        Effective axial stiffness (N/mm) of the body between two interface nodes:
        the force needed to move the nodes apart by 1 mm ALONG the line joining them.

        This is the drop-in replacement for a link's E·A/L: it captures the real give
        of the actual part (bending of an A-arm under an off-axis ball-joint load,
        a compliant bracket, etc.), not just pure axial stretch.
        """
        pa = self.xyz[self._index[name_a]]
        pb = self.xyz[self._index[name_b]]
        axis = pb - pa
        L = np.linalg.norm(axis)
        if L < 1e-9:
            raise ValueError("Interface nodes coincide; no axis defined.")
        axis = axis / L
        f = np.zeros(self.K_red.shape[0])
        f[self._trans_dofs(name_a)] = -axis      # equal & opposite -> self-equilibrated
        f[self._trans_dofs(name_b)] = +axis
        u = self._solve_free(f)
        da = u[self._trans_dofs(name_a)]
        db = u[self._trans_dofs(name_b)]
        rel = float(np.dot(db - da, axis))       # relative opening along the axis
        if rel <= 1e-15:
            return np.inf                        # effectively rigid in this direction
        return 1.0 / rel

    def grounded_stiffness(self, name: str, grounded: list,
                           direction: np.ndarray) -> float:
        """
        Stiffness (N/mm) at `name` along `direction` when the `grounded` interface
        nodes are fully fixed. This is the model for a chassis tab / bracket: ground
        the rest of the chassis interface, push the pickup, measure how far it moves.
        """
        direction = np.asarray(direction, float)
        direction = direction / np.linalg.norm(direction)
        gset = set()
        for g in grounded:
            gset.update(self._node_all_dofs(g))
        free = [i for i in range(self.K_red.shape[0]) if i not in gset]
        if not free:
            raise ValueError("All DOFs grounded; nothing to push.")
        Kff = self.K_red[np.ix_(free, free)]
        f = np.zeros(self.K_red.shape[0])
        f[self._trans_dofs(name)] = direction
        uf = np.linalg.solve(Kff, f[free])
        u = np.zeros(self.K_red.shape[0])
        u[free] = uf
        disp = float(np.dot(u[self._trans_dofs(name)], direction))
        if disp <= 1e-15:
            return np.inf
        return 1.0 / disp

    def _node_all_dofs(self, name: str) -> list:
        base = self._index[name] * self.dofs
        return list(range(base, base + self.dofs))

    # ---- (de)serialisation --------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "type": "reduced",
            "dofs_per_node": self.dofs,
            "interface": [{"name": n, "xyz": self.xyz[i].tolist()}
                          for i, n in enumerate(self.names)],
            "K_condensed": self.K_red.tolist(),
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> "CondensedFlexBody":
        iface = d["interface"]
        names = [r["name"] for r in iface]
        xyz = np.array([r["xyz"] for r in iface], float)
        K = np.array(d["K_condensed"], float)
        dofs = int(d.get("dofs_per_node", 6))
        return CondensedFlexBody(names, xyz, K, dofs,
                                 source=d.get("source", "reduced"))


# --------------------------------------------------------------------------- #
#  Importers
# --------------------------------------------------------------------------- #
def _mesh_from_dict(d: dict) -> FlexMesh:
    nodes = {str(n["id"]): n["xyz"] for n in d["nodes"]}
    elems = []
    for e in d["elements"]:
        elems.append(FlexElement(
            n1=str(e["n1"]), n2=str(e["n2"]),
            kind=e.get("kind", "beam"),
            material=e.get("material", "Steel 4130"),
            od_mm=e.get("od_mm"), wall_mm=e.get("wall_mm"),
            A=e.get("A"), I=e.get("I"), J=e.get("J")))
    interface = {str(k): str(v) for k, v in d["interface"].items()}
    return FlexMesh(nodes, elems, interface)


def load_flex_body(path_or_dict) -> CondensedFlexBody:
    """
    Load a flexible body from a .flex.json file (or an already-parsed dict).

    Two accepted schemas:

      {"type": "mesh", "nodes": [{"id","xyz"}...],
       "elements": [{"n1","n2","kind","material","od_mm","wall_mm"|"A","I","J"}...],
       "interface": {"name": node_id, ...}}
        -> a full beam/bar mesh we assemble and Guyan-condense ourselves.

      {"type": "reduced", "dofs_per_node": 6,
       "interface": [{"name","xyz"}...], "K_condensed": [[...]]}
        -> a pre-reduced superelement: the interface stiffness an ADAMS Flex MNF
           (or any CMS reduction) carries. Used verbatim.

    Returns a CondensedFlexBody either way.
    """
    if isinstance(path_or_dict, (str, bytes)):
        with open(path_or_dict, "r") as fh:
            d = json.load(fh)
    else:
        d = dict(path_or_dict)
    kind = d.get("type", "mesh")
    if kind == "reduced":
        return CondensedFlexBody.from_dict(d)
    if kind == "mesh":
        return _mesh_from_dict(d).condense()
    raise ValueError(f"Unknown flex-body type '{kind}' (expected 'mesh' or 'reduced').")


def read_mnf(path: str) -> CondensedFlexBody:
    """
    Read an ADAMS Modal Neutral File (MNF) flexible body.

    HONEST SCOPE. A production ADAMS `.mnf` is a PROPRIETARY BINARY container, and
    its full content (interface nodes, generalised mass + stiffness, and the fixed-
    interface normal modes used for dynamics) is not something this open tool can
    faithfully parse byte-for-byte — and pretending to would risk feeding the corner
    model silently wrong numbers. What this reader supports is the PORTABLE form of
    the same data that every CMS/MNF workflow can export: an ASCII file (or the
    .flex.json above) carrying the interface nodes and the condensed interface
    stiffness — which is exactly the constraint-mode (static) content that governs
    load↔deflection in a sustained corner.

    If handed a binary MNF, this raises a clear, actionable error pointing at the
    export step, rather than guessing.
    """
    with open(path, "rb") as fh:
        head = fh.read(64)
    # ADAMS MNF binary files begin with an identifying token; treat anything that
    # isn't clean ASCII/JSON as the unsupported binary container.
    is_textual = True
    try:
        head.decode("ascii")
    except UnicodeDecodeError:
        is_textual = False
    if not is_textual or head.lstrip()[:1] not in (b"{", b"#", b"M", b"I", b"N"):
        raise NotImplementedError(
            "This looks like a binary ADAMS .mnf. KinematiK imports the flexible "
            "body's interface stiffness (the static / constraint-mode content that "
            "drives compliance in a sustained corner), not the proprietary binary "
            "container. Export the reduced superelement — the interface nodes and "
            "the condensed stiffness matrix (Craig–Bampton boundary stiffness / DMIG)"
            " — as JSON in the 'reduced' schema (see load_flex_body) or as the ASCII "
            "MNF this reader understands, and load that. The numbers are identical; "
            "only the packaging differs.")
    # ASCII path: accept JSON directly (the common, simplest portable form).
    with open(path, "r") as fh:
        txt = fh.read()
    try:
        return load_flex_body(json.loads(txt))
    except Exception as exc:
        raise NotImplementedError(
            "ASCII flexible-body file present but not in a schema this reader "
            "understands. Provide the 'reduced' JSON schema (interface nodes + "
            "K_condensed). Underlying error: " + str(exc))
