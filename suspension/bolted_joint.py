# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Bolted-joint analysis for bracket feet — the pedal-box-base-onto-chassis check
the structures member actually needs before a hard-braking event lifts a foot off
its tabs.

WHY THIS MODULE IS *ANALYTIC* AND NOT "FEA"
-------------------------------------------
The rest of KinematiK keeps a strict non-goal: it is NOT a CAD kernel and NOT a
solid-FEA contact solver, and it refuses to fabricate results it cannot physically
defend (see flex.py, mountpoints.py). A full frictional-contact + bolt-pretension +
separation study on a real pedal-box base is a 3D solid nonlinear-contact FEA job
(Ansys Mechanical / Abaqus): it needs the actual meshed bracket, surface-to-surface
contact, and a solver. KinematiK has beam/bar elements and analytic stiffness only.
Faking that workflow on beam elements and printing a "the bolt will not separate"
green light on a BRAKE component would be exactly the false-confidence failure this
codebase exists to avoid — and the worst possible place to make it.

So this module does the part that is genuinely closed-form and standard, and is
honest about the one geometric input that truly needs a measurement or an FEA:

  * The MEMBER mechanics of a preloaded bolted joint — pretension from assembly
    torque, the joint stiffness ratio, how much of an external tensile load the bolt
    actually carries vs. how much simply unloads the clamped interface, the gap-
    opening (separation) load, and the bolt working stress — are all classical
    VDI 2230 / Shigley results. No FEA is required for these and inventing an FEA
    would not make them more correct.

  * The PRYING GEOMETRY — how a bending moment that tries to peel the front (or rear)
    edge of the base off its tabs gets reacted as a *higher* tensile force at the
    far-edge bolts than a simple "force / n_bolts" share — depends on the bracket's
    real stiffness and contact footprint. That is the genuine FEA-shaped part. We do
    NOT invent it. The user supplies a prying / load-distribution factor (a lever-arm
    ratio they can estimate by hand, read off a hand-calc, or lift from a real FEA),
    and every result that depends on it is flagged `is_estimate=True` and carries the
    factor in its detail so design judges — and the next member — can see the
    assumption rather than trusting a hidden number.

WHAT YOU GET
------------
Given a fastener spec, the clamped members, an assembly torque, and the external
load a braking event puts on the worst-loaded foot, this returns a `JointResult`
with:
    F_preload        clamp force from torque  (N)
    load_factor      Phi = k_b / (k_b + k_m), the bolt's share of external load
    F_bolt_max       peak bolt tensile force under load                       (N)
    sigma_bolt       bolt tensile stress on the stress area                   (MPa)
    F_sep            external tensile load at which the joint gaps open        (N)
    separated        bool — did the applied load exceed F_sep?
    sigma_bearing    contact pressure under the bolt-head/washer face         (MPa)
    bearing_yield    bool — does that pressure crush the base material?
plus the same typed `Finding` objects the integration board already renders, so a
marginal pedal-box mount shows up in the existing UI with an owner named.

UNITS: mm, N, MPa (N/mm²), N·mm. Consistent with the rest of KinematiK.

REFERENCES: VDI 2230 Part 1 (systematic calculation of high-duty bolted joints);
Shigley, *Mechanical Engineering Design*, bolted-joint chapter (joint constant C,
separation load, factor of safety against separation). These are the standard hand
methods; this module is a faithful, transparent implementation of them, not a
replacement for an FEA where one is actually required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from .interfaces import Finding, Severity


# --------------------------------------------------------------------------- #
#  Fastener library — the property classes / grades FSAE pedal-box hardware
#  actually uses. Proof and yield strengths in MPa; stress area in mm².
#  Stress area A_t is the standard ISO 898-1 tensile-stress area for the thread.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoltGrade:
    name: str
    proof_MPa: float      # proof strength (onset of permanent set), MPa
    yield_MPa: float      # 0.2% yield, MPa
    uts_MPa: float        # ultimate tensile, MPa


BOLT_GRADES = {
    # Metric property classes (ISO 898-1)
    "8.8":  BoltGrade("8.8",  proof_MPa=580.0,  yield_MPa=640.0,  uts_MPa=800.0),
    "10.9": BoltGrade("10.9", proof_MPa=830.0,  yield_MPa=940.0,  uts_MPa=1040.0),
    "12.9": BoltGrade("12.9", proof_MPa=970.0,  yield_MPa=1100.0, uts_MPa=1220.0),
    # A common aerospace/motorsport socket-head spec, treated as ~12.9-equivalent
    "A574-alloy": BoltGrade("A574-alloy", proof_MPa=970.0, yield_MPa=1100.0, uts_MPa=1220.0),
}


# Standard metric coarse threads: nominal d (mm) -> (pitch mm, tensile-stress area mm²)
# A_t from ISO 898-1. These cover the M5–M10 range pedal boxes live in.
METRIC_COARSE = {
    5.0:  (0.80, 14.2),
    6.0:  (1.00, 20.1),
    8.0:  (1.25, 36.6),
    10.0: (1.50, 58.0),
    12.0: (1.75, 84.3),
}


# --------------------------------------------------------------------------- #
#  Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Fastener:
    """
    A single bolt in the pattern. `nominal_d_mm` selects the thread (and stress area)
    from METRIC_COARSE unless `stress_area_mm2` is given explicitly. `K_factor` is the
    nut-factor in the torque–preload relation T = K · F · d (≈0.20 dry, ≈0.15 lightly
    lubricated, ≈0.12 with anti-seize — a real and large source of preload scatter,
    so it is exposed, not buried).
    """
    grade: str = "10.9"
    nominal_d_mm: float = 6.0
    stress_area_mm2: Optional[float] = None
    K_factor: float = 0.20
    # head/washer bearing geometry for the crush check:
    head_dia_mm: Optional[float] = None     # outer bearing diameter of head or washer
    hole_dia_mm: Optional[float] = None     # clearance hole the head bears around

    def stress_area(self) -> float:
        if self.stress_area_mm2 is not None:
            return float(self.stress_area_mm2)
        key = float(self.nominal_d_mm)
        if key not in METRIC_COARSE:
            raise ValueError(
                f"No built-in stress area for M{self.nominal_d_mm}; supply "
                f"stress_area_mm2 explicitly. Known: {sorted(METRIC_COARSE)}")
        return METRIC_COARSE[key][1]

    def bearing_area_mm2(self) -> Optional[float]:
        """Annular bearing area under the head/washer face, if geometry is given."""
        if self.head_dia_mm is None:
            return None
        do = float(self.head_dia_mm)
        di = float(self.hole_dia_mm) if self.hole_dia_mm is not None else self.nominal_d_mm * 1.1
        if do <= di:
            return None
        return math.pi / 4.0 * (do * do - di * di)


@dataclass
class ClampedStack:
    """
    The members squeezed between bolt head and nut/threads — here, the pedal-box base
    foot and the chassis mounting face. `grip_mm` is the total clamped thickness.

    Stiffnesses are the classical idealisation: the bolt is a rod E·A_t/L over its
    grip; the clamped members are Shigley's frustum approximation, reduced here to the
    standard closed-form k_m = E_m · d · A · exp(B · d / grip) (Wileman/Shigley fit,
    A=0.78715, B=0.62873 for the common case). Both are exposed so a team that has a
    condensed FEA stiffness for the foot can override `k_member_N_per_mm` directly.
    """
    base_material: str = "Aluminium 7075"   # the pedal-box base foot
    chassis_material: str = "Aluminium 6061"
    grip_mm: float = 8.0
    # bearing/crush limit of the SOFTER clamped face (the base). Conservatively the
    # material yield; many teams use a bearing allowance ~1.0–1.5× yield for ductile
    # alloys, so this is exposed too.
    base_bearing_allow_MPa: Optional[float] = None
    # explicit overrides (e.g. from flex.py condensation) — if given, used directly:
    k_bolt_N_per_mm: Optional[float] = None
    k_member_N_per_mm: Optional[float] = None


@dataclass
class JointResult:
    """Outcome of one preloaded-bolt check. All forces N, stresses MPa."""
    F_preload: float
    load_factor: float          # Phi = k_b/(k_b+k_m)
    F_bolt_share: float         # external tensile load reaching THIS bolt (after prying)
    F_bolt_max: float           # preload + bolt's elastic share of that load
    sigma_bolt: float
    sigma_proof: float
    F_sep: float                # external load at this bolt that opens the gap
    separation_safety: float    # F_sep / F_bolt_share  (>1 = stays closed)
    separated: bool
    sigma_bearing: Optional[float]
    bearing_allow: Optional[float]
    bearing_yield: Optional[bool]
    prying_factor: float
    is_estimate: bool
    notes: str = ""

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Stiffness helpers (Shigley / Wileman)
# --------------------------------------------------------------------------- #
def _bolt_stiffness(fastener: Fastener, grip_mm: float, E_bolt_MPa: float = 205000.0) -> float:
    At = fastener.stress_area()
    return E_bolt_MPa * At / grip_mm


def _member_stiffness(E_member_MPa: float, d_mm: float, grip_mm: float) -> float:
    # Wileman closed-form fit to the frustum model (Shigley Eq. for k_m)
    A, B = 0.78715, 0.62873
    return E_member_MPa * d_mm * A * math.exp(B * d_mm / grip_mm)


# --------------------------------------------------------------------------- #
#  The check
# --------------------------------------------------------------------------- #
def analyze_joint(
    fastener: Fastener,
    stack: ClampedStack,
    assembly_torque_Nm: float,
    external_tensile_N: float,
    *,
    prying_factor: float = 1.0,
    preload_fraction_of_proof_cap: float = 0.75,
) -> JointResult:
    """
    Analyse one preloaded bolt under an external tensile (separating) load.

    Parameters
    ----------
    assembly_torque_Nm : the torque the bolt is installed to. Preload F_i = T/(K·d).
    external_tensile_N : the tensile load this ONE bolt's region must react, BEFORE
        prying amplification — e.g. (prying moment)/(n_bolts · lever) or a per-bolt
        share of the foot's pull-off load.
    prying_factor : the genuine FEA-shaped input. 1.0 = load shares evenly with no
        prying; >1 = a bending moment levers a larger tensile force into the far-edge
        bolt than its even share (the pedal-box-base-peeling case). Supplying anything
        other than 1.0 marks the result `is_estimate=True`, because the true value
        comes from the bracket's stiffness/footprint (hand-calc lever ratio or FEA),
        which this analytic module cannot derive on its own.

    Returns a JointResult; see field docs. This is a VDI 2230 / Shigley hand method,
    transparently implemented — NOT a substitute for a contact FEA where the bracket
    geometry demands one.
    """
    from .flex import MATERIALS

    if assembly_torque_Nm <= 0:
        raise ValueError("assembly_torque_Nm must be > 0")
    if prying_factor <= 0:
        raise ValueError("prying_factor must be > 0")

    d = float(fastener.nominal_d_mm)
    At = fastener.stress_area()
    grade = BOLT_GRADES.get(fastener.grade)
    if grade is None:
        raise ValueError(f"Unknown bolt grade {fastener.grade!r}; "
                         f"known: {sorted(BOLT_GRADES)}")

    # --- preload from torque:  T = K · F_i · d  (T in N·mm) ------------------ #
    T_Nmm = assembly_torque_Nm * 1000.0
    F_preload = T_Nmm / (fastener.K_factor * d)

    # --- joint stiffnesses and load factor Phi = k_b/(k_b+k_m) -------------- #
    if stack.k_bolt_N_per_mm is not None:
        k_b = float(stack.k_bolt_N_per_mm)
    else:
        k_b = _bolt_stiffness(fastener, stack.grip_mm)
    if stack.k_member_N_per_mm is not None:
        k_m = float(stack.k_member_N_per_mm)
    else:
        E_soft = min(MATERIALS[stack.base_material].E,
                     MATERIALS[stack.chassis_material].E)
        k_m = _member_stiffness(E_soft, d, stack.grip_mm)
    Phi = k_b / (k_b + k_m)

    # --- external load actually seen by this bolt (prying amplifies it) ----- #
    F_ext = external_tensile_N * prying_factor

    # --- bolt working force & stress ---------------------------------------- #
    # Until separation, the bolt only sees Phi · F_ext on top of preload.
    # After separation the bolt carries the FULL external load (the spike).
    F_sep = F_preload / (1.0 - Phi) if Phi < 1.0 else float("inf")
    separated = F_ext > F_sep
    if not separated:
        F_bolt_max = F_preload + Phi * F_ext
    else:
        F_bolt_max = F_ext  # joint gapped: clamp gone, bolt takes it all
    sigma_bolt = F_bolt_max / At
    sigma_proof = grade.proof_MPa
    separation_safety = (F_sep / F_ext) if F_ext > 0 else float("inf")

    # --- preload sanity vs proof (installation, before external load) ------- #
    preload_stress = F_preload / At
    preload_cap = preload_fraction_of_proof_cap * sigma_proof

    # --- head/washer-face bearing (crush) check ----------------------------- #
    sigma_bearing = None
    bearing_allow = None
    bearing_yield = None
    Ab = fastener.bearing_area_mm2()
    if Ab is not None:
        # bearing pressure under the head uses the installed clamp the face carries;
        # before separation that is ~F_preload + Phi·F_ext at the head.
        F_head = F_bolt_max
        sigma_bearing = F_head / Ab
        if stack.base_bearing_allow_MPa is not None:
            bearing_allow = float(stack.base_bearing_allow_MPa)
        else:
            # default conservative allowance = base-material 0.2% proxy.
            # Aluminium alloys: use representative yields (not in flex Material,
            # which only carries E/G), so name them explicitly here.
            al_yield = {"Aluminium 6061": 276.0, "Aluminium 7075": 503.0,
                        "Steel 4130": 460.0, "Steel mild": 250.0}
            bearing_allow = al_yield.get(stack.base_material, 200.0)
        bearing_yield = sigma_bearing > bearing_allow

    notes = []
    if preload_stress > preload_cap:
        notes.append(
            f"Preload stress {preload_stress:.0f} MPa exceeds "
            f"{preload_fraction_of_proof_cap:.0%} of proof "
            f"({preload_cap:.0f} MPa) — torque spec over-tensions the bolt at install.")
    if separated:
        notes.append(
            "JOINT SEPARATED: external load exceeded the gap-opening load; the bolt "
            "now carries the full raw tensile load and sees a fatigue/peak spike.")

    return JointResult(
        F_preload=F_preload,
        load_factor=Phi,
        F_bolt_share=F_ext,
        F_bolt_max=F_bolt_max,
        sigma_bolt=sigma_bolt,
        sigma_proof=sigma_proof,
        F_sep=F_sep,
        separation_safety=separation_safety,
        separated=separated,
        sigma_bearing=sigma_bearing,
        bearing_allow=bearing_allow,
        bearing_yield=bearing_yield,
        prying_factor=prying_factor,
        is_estimate=(prying_factor != 1.0),
        notes="  ".join(notes),
    )


def joint_findings(result: JointResult, *, bolt_label: str = "pedal-box mount bolt",
                   owners=("structures", "chassis")) -> list:
    """
    Render a JointResult as the typed Finding objects the integration board already
    shows. Separation and bolt-overstress are FAIL; thin margins are WARN; an
    estimate-based result is flagged so it can't pass as final.
    """
    out: list = []
    owners = list(owners)

    # Separation — the headline failure mode for a peeling pedal-box base.
    if result.separated:
        out.append(Finding(
            "bolt-separation", Severity.FAIL,
            f"{bolt_label}: the clamped joint OPENS under the braking load "
            f"(external {result.F_bolt_share:.0f} N > gap-opening {result.F_sep:.0f} N). "
            f"The base lifts off its tab and the bolt takes the full raw load.",
            subsystems=owners,
            detail=result.as_dict()))
    elif result.separation_safety < 1.25:
        out.append(Finding(
            "bolt-separation", Severity.WARN,
            f"{bolt_label}: thin margin against joint separation "
            f"(safety {result.separation_safety:.2f}× — under the usual 1.25 target). "
            f"Raise preload/torque or add clamp area before the base starts to peel.",
            subsystems=owners,
            detail=result.as_dict()))
    else:
        out.append(Finding(
            "bolt-separation", Severity.OK,
            f"{bolt_label}: joint stays clamped "
            f"({result.separation_safety:.2f}× margin to separation).",
            subsystems=owners,
            detail=result.as_dict()))

    # Bolt stress vs proof.
    proof_use = result.sigma_bolt / result.sigma_proof
    if proof_use > 1.0:
        out.append(Finding(
            "bolt-stress", Severity.FAIL,
            f"{bolt_label}: bolt tensile stress {result.sigma_bolt:.0f} MPa exceeds "
            f"proof {result.sigma_proof:.0f} MPa — the bolt yields.",
            subsystems=owners, detail=result.as_dict()))
    elif proof_use > 0.85:
        out.append(Finding(
            "bolt-stress", Severity.WARN,
            f"{bolt_label}: bolt at {proof_use:.0%} of proof — little headroom.",
            subsystems=owners, detail=result.as_dict()))

    # Washer-face crush (preload relaxation / back-out risk).
    if result.bearing_yield:
        out.append(Finding(
            "bolt-bearing", Severity.FAIL,
            f"{bolt_label}: bearing pressure under the head/washer "
            f"{result.sigma_bearing:.0f} MPa exceeds the base allowable "
            f"{result.bearing_allow:.0f} MPa — the base crushes locally, preload "
            f"relaxes and the joint can back out on track. Add a washer or larger head.",
            subsystems=owners, detail=result.as_dict()))

    # Provenance: an estimate must not read as final.
    if result.is_estimate:
        out.append(Finding(
            "bolt-prying", Severity.INFO,
            f"{bolt_label}: result uses an ESTIMATED prying factor "
            f"{result.prying_factor:.2f}× (the bending-moment lever amplifying this "
            f"bolt's tensile share). This is the part a real contact FEA or a hand "
            f"lever-arm calc owns — treat as provisional until that factor is checked.",
            subsystems=owners, detail={"prying_factor": result.prying_factor}))

    return out
