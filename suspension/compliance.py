# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Compliant kinematics — what the wheel actually does once the links can flex.

This is the module that breaks the rigid-body assumption. It couples three things
the rest of the tool already has, in a loop:

    geometry  →  member loads  →  deflections  →  geometry  →  …

  1. the rigid kinematic solver (kinematics.py) gives the corner geometry;
  2. the load-path solver (loadpath.py) resolves the contact-patch cornering load
     into the axial force in every link;
  3. each link's COMPLIANCE (flex.py — analytic E·A/L, or a condensed FEA body)
     turns that force into a length change;
  4. those length changes are fed back into the solver, which re-solves the loaded
     geometry. Because the deflected geometry slightly changes the member loads, the
     loop is iterated to convergence (usually 2–4 steps; deflections are sub-mm to a
     few mm and the load redistribution is small).

The output is the thing a rigid tool cannot give: the COMPLIANCE STEER (toe change
under load) and COMPLIANCE CAMBER (camber change under load) the car develops at a
given cornering load — e.g. how much toe the front axle loses to link and tab flex
at 1.5 g. That is a primary reason a real car doesn't behave like its kinematics
spreadsheet, and the number teams need when the car tramlines under brakes or the
balance moves with load.

Honest about scope: this is a QUASI-STATIC compliance analysis at one load case
(steady-state cornering load), built on the same idealisations as the load-path
solver (pin-jointed two-force members, axial compliance as the dominant term). It is
not a transient/NVH analysis and does not pretend to be.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .kinematics import SuspensionKinematics, Hardpoints, CornerState
from . import loadpath as lp
from . import flex as flexmod
from .joints import JointCompliance


# Map each load-path member to the length-delta key the solver understands.
# The pushrod (PR) loads the spring/rocker chain (a ride-rate effect), not a wishbone
# leg length, so it has no wishbone-length key — its deflection is reported, not fed
# back as a camber/toe-changing length change.
_MEMBER_TO_LENGTHKEY = {
    "UF": "upper_f", "UR": "upper_r",
    "LF": "lower_f", "LR": "lower_r",
    "TR": "tie",
}

# The two endpoints (outboard, inboard hardpoint field) of each member — used both
# for analytic length and to look up the matching interface nodes of an FEA body.
_MEMBER_ENDPOINTS = {
    "UF": ("upper_outer", "upper_front_inner"),
    "UR": ("upper_outer", "upper_rear_inner"),
    "LF": ("lower_outer", "lower_front_inner"),
    "LR": ("lower_outer", "lower_rear_inner"),
    "TR": ("tie_rod_outer", "tie_rod_inner"),
    "PR": ("pushrod_outer", "rocker_pushrod"),
}


@dataclass
class MemberStiffness:
    """
    Axial stiffness source for one suspension member (N/mm).

    Provide ONE of:
      k_direct  : a stiffness value you already have (N/mm);
      (material, od_mm, wall_mm) : analytic tube, k = E·A/L on the link's length;
      flex_body + (node_out, node_in) : a condensed FEA body (flex.py), using its
        real axial give between the two attachment nodes.

    Optionally add `k_tab` (N/mm), a chassis-tab/bracket stiffness in SERIES with the
    link (the mount flexes too): 1/k_total = 1/k_link + 1/k_tab.

    NON-LINEAR JOINTS (the bushings / rod ends / spherical bearings)
    ---------------------------------------------------------------
    Optionally attach a `JointCompliance` at either end of the member:
      joint_in  : the inboard (chassis-pickup) joint — typically a rubber or
                  polyurethane bushing;
      joint_out : the outboard joint — typically a spherical bearing / rod end.
    Each is a non-linear spring in SERIES with the link along the load line, so the
    member's total axial give under tension T is
        δ(T) = T/k_link(+tab) + joint_in.displacement(T) + joint_out.displacement(T).
    The link itself may be omitted (no k_direct/material/flex_body) to model a RIGID
    link whose only compliance is in its joints.
    """
    k_direct: Optional[float] = None
    material: Optional[str] = None
    od_mm: Optional[float] = None
    wall_mm: Optional[float] = None
    flex_body: Optional[flexmod.CondensedFlexBody] = None
    node_out: Optional[str] = None
    node_in: Optional[str] = None
    k_tab: Optional[float] = None
    joint_in: Optional[JointCompliance] = None
    joint_out: Optional[JointCompliance] = None

    def _has_link(self) -> bool:
        return (self.flex_body is not None or self.k_direct is not None
                or (self.material is not None and self.od_mm and self.wall_mm))

    def axial_stiffness(self, length_mm: float) -> float:
        if self.flex_body is not None:
            if not (self.node_out and self.node_in):
                raise ValueError("flex_body given without node_out/node_in names.")
            k = self.flex_body.relative_axial_stiffness(self.node_out, self.node_in)
        elif self.k_direct is not None:
            k = float(self.k_direct)
        elif self.material is not None and self.od_mm and self.wall_mm:
            mat = flexmod.MATERIALS.get(self.material)
            if mat is None:
                raise ValueError(f"Unknown material '{self.material}'.")
            k = flexmod.axial_stiffness_tube(mat, length_mm, self.od_mm, self.wall_mm)
        else:
            raise ValueError("MemberStiffness has no usable stiffness definition.")
        if self.k_tab is not None and np.isfinite(self.k_tab) and self.k_tab > 0:
            k = 1.0 / (1.0 / k + 1.0 / self.k_tab)   # series with the tab
        return float(k)

    def axial_deflection(self, force_N: float, length_mm: float) -> float:
        """
        Total axial give (mm) of the link AND its end joints in series under axial
        force `force_N` (N, + tension). The linear link (+ optional chassis tab) and
        each non-linear `JointCompliance` carry the same force; their displacements
        add. Reduces exactly to force/axial_stiffness when no joints are attached, so
        existing linear behaviour is unchanged.
        """
        d = 0.0
        if self._has_link():
            k_lin = self.axial_stiffness(length_mm)
            if np.isfinite(k_lin) and k_lin > 0:
                d += force_N / k_lin
        elif self.joint_in is None and self.joint_out is None:
            raise ValueError("MemberStiffness has neither a link stiffness nor a "
                             "joint — nothing to deflect.")
        if self.joint_in is not None:
            d += self.joint_in.displacement(force_N)
        if self.joint_out is not None:
            d += self.joint_out.displacement(force_N)
        return float(d)

    def deflection_breakdown(self, force_N: float, length_mm: float) -> dict:
        """Per-source axial give (mm): the link, the inboard joint, the outboard joint."""
        link = 0.0
        if self._has_link():
            k_lin = self.axial_stiffness(length_mm)
            if np.isfinite(k_lin) and k_lin > 0:
                link = force_N / k_lin
        return {
            "link": float(link),
            "joint_in": float(self.joint_in.displacement(force_N)) if self.joint_in else 0.0,
            "joint_out": float(self.joint_out.displacement(force_N)) if self.joint_out else 0.0,
        }

    def tangent_axial_stiffness(self, force_N: float, length_mm: float) -> float:
        """
        Operating-point axial stiffness (N/mm) of the link+joints series at the
        current force — the local slope a dynamic/linearised analysis should use.
        """
        flex = 0.0
        if self._has_link():
            k_lin = self.axial_stiffness(length_mm)
            if np.isfinite(k_lin) and k_lin > 0:
                flex += 1.0 / k_lin
        for jt in (self.joint_in, self.joint_out):
            if jt is not None:
                d = jt.displacement(force_N)
                k_t = jt.tangent_stiffness(d)
                flex += 1.0 / k_t if np.isfinite(k_t) and k_t > 0 else np.inf
        return float(1.0 / flex) if flex > 0 else np.inf


@dataclass
class CompliantResult:
    """Rigid vs compliant corner state at one load case, plus the load breakdown."""
    load: lp.WheelLoad
    rigid: CornerState
    compliant: CornerState
    member_forces: dict             # member -> axial force (N, + tension)
    member_deflection: dict         # member -> axial length change (mm, + = stretch)
    member_stiffness: dict          # member -> stiffness used (N/mm)
    iterations: int
    converged: bool
    note: str = ""
    member_joint_deflection: dict = field(default_factory=dict)
    # member -> {"link","joint_in","joint_out"} axial give breakdown (mm)

    # --- the headline deltas ------------------------------------------------ #
    @property
    def compliance_toe(self) -> float:
        """Toe change due to compliance (deg, + = toe-out). The 'compliance steer'."""
        return self.compliant.toe - self.rigid.toe

    @property
    def compliance_camber(self) -> float:
        """Camber change due to compliance (deg)."""
        return self.compliant.camber - self.rigid.camber

    @property
    def compliance_caster(self) -> float:
        return self.compliant.caster - self.rigid.caster

    @property
    def wheel_center_shift_mm(self) -> np.ndarray:
        return self.compliant.wheel_center - self.rigid.wheel_center

    @property
    def contact_patch_lateral_shift_mm(self) -> float:
        """Lateral (y) movement of the contact patch — track/scrub change under load."""
        return float(self.compliant.contact_patch[1] - self.rigid.contact_patch[1])

    def summary(self) -> dict:
        return {
            "compliance_toe_deg": self.compliance_toe,
            "compliance_camber_deg": self.compliance_camber,
            "compliance_caster_deg": self.compliance_caster,
            "contact_patch_lateral_shift_mm": self.contact_patch_lateral_shift_mm,
            "wheel_center_shift_mm": self.wheel_center_shift_mm.tolist(),
            "iterations": self.iterations,
            "converged": self.converged,
            "member_forces_N": {k: float(v) for k, v in self.member_forces.items()},
            "member_deflection_mm": {k: float(v) for k, v in self.member_deflection.items()},
            "member_joint_deflection_mm": {
                m: {k: float(v) for k, v in b.items()}
                for m, b in self.member_joint_deflection.items()},
            "note": self.note,
        }


class CompliantCorner:
    """
    A corner whose links can flex. Wraps a rigid SuspensionKinematics and a per-member
    stiffness map, and solves the loaded geometry under a contact-patch wheel load.
    """

    def __init__(self, hp: Hardpoints, stiffness: dict):
        """
        hp        : the (unloaded) hardpoint geometry.
        stiffness : dict member -> MemberStiffness for the members you want flexible.
                    Members omitted are treated as RIGID (infinite stiffness), so you
                    can study one link at a time.
        """
        self.hp = hp
        self.rigid_kin = SuspensionKinematics(hp)
        self.stiffness = dict(stiffness)

    # ------------------------------------------------------------------ #
    @staticmethod
    def uniform_tube(hp: Hardpoints, material: str = "Steel 4130",
                     od_mm: float = 19.05, wall_mm: float = 0.9,
                     tie_od_mm: Optional[float] = None,
                     tie_wall_mm: Optional[float] = None,
                     k_tab: Optional[float] = None,
                     bushing: Optional[JointCompliance] = None,
                     rod_end: Optional[JointCompliance] = None,
                     tie_bushing: Optional[JointCompliance] = None,
                     tie_rod_end: Optional[JointCompliance] = None,
                     ) -> "CompliantCorner":
        """
        Build a corner where every link is the same tube (the common FSAE case:
        '3/4 inch 4130, 0.9 mm wall'). The tie rod can be given its own size.
        `k_tab` optionally adds the same chassis-tab stiffness in series on every
        wishbone leg. This is the zero-FEA path — defensible link stiffness from
        material and tube size alone.

        Non-linear joints (optional): pass a `JointCompliance` to put the same joint
        on every matching connection —
          bushing     : inboard (chassis) joint on the four wishbone legs;
          rod_end     : outboard joint on the four wishbone legs;
          tie_bushing : inboard joint on the tie rod;
          tie_rod_end : outboard joint on the tie rod.
        NOTE on shared joints: each wishbone leg is treated as an independent two-
        force member, so `rod_end` is applied to BOTH legs of an A-arm. That is
        correct for the common FSAE design where each leg carries its own rod end
        into a separate tab; if your A-arm shares ONE outboard ball joint, assign the
        rod_end to a single leg instead (build the dict directly) to avoid double-
        counting that joint.
        """
        tie_od = tie_od_mm if tie_od_mm is not None else od_mm
        tie_wall = tie_wall_mm if tie_wall_mm is not None else wall_mm
        stiff = {}
        for m in ("UF", "UR", "LF", "LR"):
            stiff[m] = MemberStiffness(material=material, od_mm=od_mm,
                                       wall_mm=wall_mm, k_tab=k_tab,
                                       joint_in=bushing, joint_out=rod_end)
        stiff["TR"] = MemberStiffness(material=material, od_mm=tie_od, wall_mm=tie_wall,
                                      joint_in=tie_bushing, joint_out=tie_rod_end)
        return CompliantCorner(hp, stiff)

    # ------------------------------------------------------------------ #
    @staticmethod
    def with_bushings(hp: Hardpoints,
                      bushing: JointCompliance,
                      rod_end: Optional[JointCompliance] = None,
                      material: str = "Steel 4130",
                      od_mm: float = 19.05, wall_mm: float = 0.9,
                      ) -> "CompliantCorner":
        """
        Convenience for the production-style layout: rubber/poly BUSHINGS on every
        inboard pickup (and the tie-rod inner), with optional rod ends outboard. The
        links default to the usual 4130 tube. Equivalent to `uniform_tube` with the
        joint arguments filled in for both the wishbones and the tie rod.

        Example
        -------
        >>> b = JointCompliance.rubber_bushing()
        >>> r = JointCompliance.spherical_bearing(lash_mm=0.05)
        >>> cc = CompliantCorner.with_bushings(hp, bushing=b, rod_end=r)
        """
        return CompliantCorner.uniform_tube(
            hp, material=material, od_mm=od_mm, wall_mm=wall_mm,
            bushing=bushing, rod_end=rod_end,
            tie_bushing=bushing, tie_rod_end=rod_end)

    # ------------------------------------------------------------------ #
    def _member_length(self, member: str, kin: SuspensionKinematics,
                       state: CornerState) -> float:
        """Current length of a member from the solved state (for analytic E·A/L)."""
        out_f, in_f = _MEMBER_ENDPOINTS[member]
        # outboard point: prefer the solved state, fall back to hardpoints
        state_attr = {"upper_outer": "upper_outer", "lower_outer": "lower_outer",
                      "tie_rod_outer": "tie_rod_outer",
                      "pushrod_outer": "pushrod_outer"}.get(out_f)
        p_out = getattr(state, state_attr) if state_attr and getattr(state, state_attr, None) is not None \
            else np.asarray(getattr(kin.hp, out_f), float)
        p_in = np.asarray(getattr(kin.hp, in_f), float)
        return float(np.linalg.norm(np.asarray(p_out, float) - p_in))

    def solve(self, load: lp.WheelLoad, max_iter: int = 12,
              tol_deg: float = 1e-4) -> CompliantResult:
        """
        Solve the loaded corner geometry under a contact-patch wheel load.

        Returns a CompliantResult with the rigid and compliant states and the full
        force/deflection breakdown. Iterates the load↔geometry coupling to
        convergence on camber and toe.
        """
        rigid_state = self.rigid_kin.static
        kin = self.rigid_kin
        state = rigid_state
        length_deltas = {}
        member_forces = {}
        member_defl = {}
        member_k = {}
        member_joint_defl = {}
        converged = False
        it = 0
        note_parts = []

        for it in range(1, max_iter + 1):
            mf = lp.solve_member_forces(kin, state, load)
            if mf.note and it == 1:
                note_parts.append(mf.note)

            new_deltas = {}
            for member, ms in self.stiffness.items():
                if member not in _MEMBER_TO_LENGTHKEY and member != "PR":
                    continue
                T = mf.tension(member)
                L = self._member_length(member, kin, state)
                # total series give = link (+tab) + non-linear end joints
                delta = ms.axial_deflection(T, L)
                member_k[member] = ms.tangent_axial_stiffness(T, L)
                member_joint_defl[member] = ms.deflection_breakdown(T, L)
                member_defl[member] = delta
                member_forces[member] = T
                key = _MEMBER_TO_LENGTHKEY.get(member)
                if key is not None:
                    new_deltas[key] = delta
            # record pushrod force even though it has no length key
            if "PR" not in member_forces and mf.has_pushrod:
                member_forces["PR"] = mf.tension("PR")

            # re-solve loaded geometry
            kin = SuspensionKinematics(self.hp, length_deltas=new_deltas)
            new_state = kin.static

            d_cam = abs(new_state.camber - state.camber)
            d_toe = abs(new_state.toe - state.toe)
            state = new_state
            length_deltas = new_deltas
            if max(d_cam, d_toe) < tol_deg:
                converged = True
                break

        if not converged:
            note_parts.append(f"Compliance loop did not fully converge in {max_iter} "
                              "iterations (still useful, but treat the last digit with "
                              "care).")

        return CompliantResult(
            load=load, rigid=rigid_state, compliant=state,
            member_forces=member_forces, member_deflection=member_defl,
            member_stiffness=member_k, iterations=it, converged=converged,
            member_joint_deflection=member_joint_defl,
            note="  ".join(note_parts).strip())

    # ------------------------------------------------------------------ #
    def linearized_rates(self, load: lp.WheelLoad,
                         freq_hz: Optional[float] = None,
                         **solve_kw) -> dict:
        """
        Operating-point linearisation for a DYNAMIC analysis: solve the static
        compliance case at `load`, then report, per member, the tangent axial
        stiffness (link+joints in series, N/mm) and the per-joint linearised
        stiffness + equivalent viscous damping at that working force. This is the
        artifact the transient DAE solver (transient.py) consumes to give the joints
        rate-dependent behaviour the static solve cannot — damping does no work at
        zero velocity, so it lives here, not in the compliance numbers.

        `freq_hz` (optional) lets each joint fold its structural loss factor into an
        equivalent viscous coefficient (c_eq = c + η·k/ω).
        """
        res = self.solve(load, **solve_kw)
        out = {}
        for member, ms in self.stiffness.items():
            T = res.member_forces.get(member)
            if T is None:
                continue
            entry = {"force_N": float(T),
                     "tangent_axial_stiffness_N_per_mm":
                         float(res.member_stiffness.get(member, np.inf)),
                     "joints": {}}
            for end, jt in (("in", ms.joint_in), ("out", ms.joint_out)):
                if jt is not None:
                    entry["joints"][end] = jt.linearize(about_force_N=T,
                                                         freq_hz=freq_hz)
            out[member] = entry
        return out

    def damping_summary(self, load: lp.WheelLoad, amplitude_mm: float = 0.5,
                        freq_hz: float = 15.0, **solve_kw) -> dict:
        """
        Energy dissipated per cycle (N·mm = mJ) by each joint when the corner
        oscillates about the `load` working point at the given amplitude/frequency —
        the honest, scalar way joint damping shows up in a steady analysis. Sums the
        viscous and structural-loss contributions of every defined joint.
        """
        res = self.solve(load, **solve_kw)
        per_joint = {}
        total = 0.0
        for member, ms in self.stiffness.items():
            for end, jt in (("in", ms.joint_in), ("out", ms.joint_out)):
                if jt is None:
                    continue
                e = jt.energy_loss_per_cycle(amplitude_mm, freq_hz)
                per_joint[f"{member}_{end}"] = float(e)
                total += e
        return {"amplitude_mm": float(amplitude_mm), "freq_hz": float(freq_hz),
                "energy_per_cycle_mJ": per_joint,
                "total_energy_per_cycle_mJ": float(total)}


# --------------------------------------------------------------------------- #
#  Driving the compliance case from a vehicle cornering condition
# --------------------------------------------------------------------------- #
def corner_wheel_load(veh, axle: str, lateral_g: float,
                      outer: bool = True, long_g: float = 0.0) -> lp.WheelLoad:
    """
    Build the contact-patch WheelLoad on one tyre from a steady-state cornering case.

      veh        : a VehicleDynamics (for the load-transfer split)
      axle       : "front" or "rear"
      lateral_g  : sustained lateral acceleration (g)
      outer      : True for the loaded outer tyre (the one that matters for grip and
                   the worst compliance case), False for the inner
      long_g     : optional simultaneous longitudinal g (braking/traction)

    Vertical load comes from the real load-transfer model. The lateral force on the
    tyre is distributed by equal lateral-g utilisation (Fy = lateral_g · Fz), the
    standard steady-state assumption: summed over the car, ΣFy/ΣFz = lateral_g.
    """
    loads, _ = veh.lateral_load_transfer(lateral_g)
    if axle == "front":
        Fz = loads.fr if outer else loads.fl
    else:
        Fz = loads.rr if outer else loads.rl
    # lateral force toward the turn centre (centripetal). For the right-side corner
    # model, the outer wheel's cornering force points inboard (−y).
    return lp.WheelLoad(Fx=long_g * Fz, Fy=-lateral_g * Fz, Fz=Fz)
