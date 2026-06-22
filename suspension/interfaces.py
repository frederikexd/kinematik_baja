# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Subsystem interface ledger — the integration layer FSAE teams actually lack.

OptimumK, ANSYS and SolidWorks each go deep in ONE domain. What no team has is the
thing the eight subsystem channels in Discord imply: a place where the *interfaces
between* subsystems are owned and checked. Eight sub-teams each optimise in their own
tool, hand off a CAD file or a number, and the integration failures surface at
assembly or at competition — the radiator that doesn't fit the duct aero reserved,
the motor torque that exceeds what the upright was designed for, the eight "we're
~12 kg" estimates that sum to 18 kg over the number suspension used for load transfer.

This module is that ledger. Each subsystem declares, in typed fields:
    * what it NEEDS from the rest of the car (envelope, mounting points, supply
      current, cooling airflow, mass budget…), and
    * what it PROVIDES to the rest of the car (mass + CG, loads into mounts, heat
      rejected, power drawn, downforce/drag…).

The checker then validates consistency ACROSS declarations and, where the physics
already exists in KinematiK, against it: every subsystem's mass+CG flows into the
real load-transfer / lap-time engine, and mounting loads come from the kinematic
solver. The rest are constraint checks — fit, budget, current, heat, consistency.

DELIBERATE NON-GOAL: this does not simulate any subsystem. KinematiK cannot do CFD,
brake-thermal, chassis FEA or battery modelling, and faking those would be the exact
false-confidence failure the rest of the codebase refuses. Each subsystem's analysis
stays in the tool that does it properly; KinematiK owns the channels between them, and
flags every input whose value is a placeholder rather than a real number, so a green
board never means more than the data behind it.

Every check returns a structured Finding with a severity and a plain-language message
naming BOTH subsystems involved, so the conflict has an owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# The five Baja subsystems (matching the team channels). 'chassis' is the
# integrator most others hang off; the two suspension corners and the drivetrain
# are the historical failure-driven priorities. All are peers here.
SUBSYSTEMS = [
    "chassis", "drivetrain", "front-suspension", "rear-suspension",
    "data-acquisition",
]


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARN = "warning"      # a real conflict, fixable, won't stop the car today
    FAIL = "fail"         # a hard incompatibility — the parts do not go together
    MISSING = "missing"   # needed data not provided yet (not a pass and not a fail)


@dataclass
class Finding:
    """One consistency-check result. Names the subsystems so it has an owner."""
    check: str
    severity: Severity
    message: str
    subsystems: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# --------------------------------------------------------------------------- #
#  Interface declaration — what each subsystem needs and provides
# --------------------------------------------------------------------------- #
@dataclass
class SubsystemInterface:
    """
    One subsystem's declared interface. Every numeric field is Optional: None means
    "not declared yet", which the checker reports as MISSING (honestly distinct from
    a value of 0). `is_estimate` marks the whole declaration as placeholder data so
    the board can never imply more certainty than the team actually has.

    Not every field applies to every subsystem; teams fill what's relevant. The
    checker only fires a rule when the fields it needs are present.
    """
    name: str
    # --- mass & balance (applies to ALL; feeds the real physics) ---
    mass_kg: Optional[float] = None
    cg_x_mm: Optional[float] = None         # +rearward from front axle
    cg_y_mm: Optional[float] = None         # +right of centreline
    cg_z_mm: Optional[float] = None         # +up from ground
    # --- spatial envelope this subsystem occupies / requires (a bounding box) ---
    env_x_mm: Optional[float] = None        # length it needs
    env_y_mm: Optional[float] = None        # width
    env_z_mm: Optional[float] = None        # height
    env_origin_mm: Optional[tuple] = None   # (x,y,z) of the box's reference corner
    # --- mechanical loads it imposes into its mounts (peak) ---
    mount_load_n: Optional[float] = None    # peak force into the structure it bolts to
    mount_points: Optional[int] = None
    mounts_on: Optional[str] = None         # which subsystem carries it (e.g. "chassis")
    # --- electrical ---
    power_draw_w: Optional[float] = None    # continuous electrical draw
    peak_current_a: Optional[float] = None
    voltage_v: Optional[float] = None
    # --- thermal ---
    heat_reject_w: Optional[float] = None   # heat it dumps (needs cooling/airflow)
    cooling_airflow_cms: Optional[float] = None  # airflow it REQUIRES (m^3/s)
    max_temp_c: Optional[float] = None
    # --- powertrain / longitudinal ---
    peak_torque_nm: Optional[float] = None
    peak_power_kw: Optional[float] = None
    # --- aero ---
    downforce_n_at_v: Optional[tuple] = None  # (force_N, speed_m_s)
    drag_n_at_v: Optional[tuple] = None
    # --- brakes ---
    brake_torque_nm: Optional[float] = None   # per corner, peak
    # --- provenance & documentation ---
    is_estimate: bool = True
    rationale: str = ""          # WHY these numbers — the design justification judges ask for
    owner: str = ""              # who on the sub-team owns this interface
    updated_by: str = ""         # who last changed it
    updated_on: str = ""         # ISO date of last change (auto-stamped on edit)
    notes: str = ""

    def declared_fields(self) -> list:
        """Names of the numeric channels actually filled in (not None)."""
        skip = {"name", "is_estimate", "rationale", "owner",
                "updated_by", "updated_on", "notes"}
        return [k for k, v in asdict(self).items() if k not in skip and v is not None]

    def numeric_values(self) -> dict:
        """Just the declared numeric channels and their values (for change diffs)."""
        return {k: getattr(self, k) for k in self.declared_fields()}

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> "SubsystemInterface":
        # tuples survive a JSON round-trip as lists; normalise back
        d = dict(d)
        for k in ("env_origin_mm", "downforce_n_at_v", "drag_n_at_v"):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        valid = SubsystemInterface.__dataclass_fields__.keys()
        return SubsystemInterface(**{k: v for k, v in d.items() if k in valid})


# --------------------------------------------------------------------------- #
#  The ledger
# --------------------------------------------------------------------------- #
@dataclass
class IntegrationLedger:
    """
    Holds one interface per subsystem plus the shared car-level budgets/limits the
    checks validate against. Construct empty, then `set()` each subsystem's
    declaration. `check_all()` returns the full list of Findings.
    """
    target_mass_kg: float = 230.0          # whole-car mass target (incl. driver?)
    includes_driver_kg: float = 0.0        # driver mass already counted in target
    accumulator_voltage_v: float = 12.0    # Baja has no HV pack; small 12 V battery
    lv_voltage_v: float = 12.0             # 12 V bus (ignition / DAQ / lights)
    lv_supply_capacity_w: float = 180.0    # what the small battery/charging can deliver
    total_cooling_airflow_cms: float = 0.0 # airflow the cooling pkg can actually move
    chassis_envelope_mm: Optional[tuple] = None  # (x,y,z) interior the car offers
    upright_design_load_n: Optional[float] = None  # what suspension designed mounts for
    driveline_torque_limit_nm: Optional[float] = None  # what driveshaft/CV is rated for
    interfaces: dict = field(default_factory=dict)

    def set(self, iface: SubsystemInterface):
        self.interfaces[iface.name] = iface

    def get(self, name: str) -> Optional[SubsystemInterface]:
        return self.interfaces.get(name)

    # ---- physics bridge: mass roll-up + CG ---------------------------------- #
    def mass_rollup(self) -> dict:
        """
        Sum declared subsystem masses and compute the combined CG (mass-weighted),
        the single most important cross-subsystem number: it's what suspension's
        load-transfer model assumes, and the place eight optimistic estimates
        quietly blow the budget. Returns totals + which subsystems are missing mass.
        """
        total = 0.0
        mx = my = mz = 0.0
        have_cg = True
        declared, missing = [], []
        any_estimate = False
        for s in SUBSYSTEMS:
            it = self.interfaces.get(s)
            if it is None or it.mass_kg is None:
                missing.append(s)
                continue
            declared.append(s)
            total += it.mass_kg
            any_estimate = any_estimate or it.is_estimate
            if None in (it.cg_x_mm, it.cg_y_mm, it.cg_z_mm):
                have_cg = False
            else:
                mx += it.mass_kg * it.cg_x_mm
                my += it.mass_kg * it.cg_y_mm
                mz += it.mass_kg * it.cg_z_mm
        cg = None
        if have_cg and total > 0:
            cg = (mx / total, my / total, mz / total)
        return dict(total_kg=total, cg_mm=cg, declared=declared, missing=missing,
                    any_estimate=any_estimate,
                    target_kg=self.target_mass_kg,
                    delta_kg=total - self.target_mass_kg)

    def as_dict(self):
        d = {k: getattr(self, k) for k in (
            "target_mass_kg", "includes_driver_kg", "accumulator_voltage_v",
            "lv_voltage_v", "lv_supply_capacity_w", "total_cooling_airflow_cms",
            "chassis_envelope_mm", "upright_design_load_n",
            "driveline_torque_limit_nm")}
        d["interfaces"] = {k: v.as_dict() for k, v in self.interfaces.items()}
        return d

    @staticmethod
    def from_dict(d) -> "IntegrationLedger":
        d = dict(d)
        ifaces = d.pop("interfaces", {}) or {}
        for k in ("chassis_envelope_mm",):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        valid = IntegrationLedger.__dataclass_fields__.keys()
        led = IntegrationLedger(**{k: v for k, v in d.items()
                                   if k in valid and k != "interfaces"})
        for name, idict in ifaces.items():
            led.set(SubsystemInterface.from_dict(idict))
        return led

    # ---- the checks --------------------------------------------------------- #
    def check_all(self) -> list:
        findings: list = []
        findings += self._check_mass_budget()
        findings += self._check_cg_sanity()
        findings += self._check_envelopes()
        findings += self._check_thermal()
        findings += self._check_electrical()
        findings += self._check_driveline_torque()
        findings += self._check_mount_loads()
        findings += self._check_estimates_flagged()
        return findings

    def _check_mass_budget(self):
        r = self.mass_rollup()
        out = []
        # The car target may include a driver allowance the subsystem masses don't.
        # Compare subsystem mass against the target NET of any declared driver mass.
        target_net = self.target_mass_kg - max(self.includes_driver_kg, 0.0)
        delta_net = r["total_kg"] - target_net if r["declared"] else 0.0
        if r["missing"]:
            out.append(Finding(
                "mass-budget", Severity.MISSING,
                f"{len(r['missing'])} subsystem(s) have not declared a mass: "
                f"{', '.join(r['missing'])}. The car mass total — and therefore "
                f"suspension's load transfer and the lap sim — is incomplete.",
                subsystems=r["missing"], detail=dict(declared=r["declared"])))
        if r["declared"]:
            sev = Severity.FAIL if delta_net > 0.05 * target_net else (
                Severity.WARN if delta_net > 0 else Severity.OK)
            drv = (f" (target net of {self.includes_driver_kg:.0f} kg driver = "
                   f"{target_net:.0f} kg)") if self.includes_driver_kg > 0 else ""
            msg = (f"Declared subsystem mass {r['total_kg']:.1f} kg vs target "
                   f"{target_net:.0f} kg{drv} ({delta_net:+.1f} kg).")
            if r["missing"]:
                msg += " And not everyone has reported — the real total is higher."
            out.append(Finding("mass-budget-total", sev, msg,
                               subsystems=r["declared"],
                               detail=dict(total_kg=r["total_kg"],
                                           target_net_kg=target_net,
                                           delta_kg=delta_net)))
        return out

    def _check_cg_sanity(self):
        r = self.mass_rollup()
        if not r["cg_mm"]:
            return [Finding("cg", Severity.MISSING,
                            "Combined CG not computable — some subsystems gave mass "
                            "but no CG location. Suspension is using an assumed CG "
                            "height that nobody has confirmed against real layout.",
                            subsystems=["front-suspension", "rear-suspension"])]
        x, y, z = r["cg_mm"]
        out = [Finding("cg", Severity.INFO,
                       f"Combined CG (declared subsystems): x={x:.0f} y={y:.0f} "
                       f"z={z:.0f} mm. Feed CG height {z:.0f} mm into the vehicle "
                       f"model so load transfer matches the real build.",
                       subsystems=["front-suspension", "rear-suspension", "chassis"],
                       detail=dict(cg_x=x, cg_y=y, cg_z=z))]
        # lateral CG should be near centreline
        if abs(y) > 25.0:
            out.append(Finding("cg-lateral", Severity.WARN,
                               f"Combined CG is {y:+.0f} mm off centreline — that's a "
                               f"static lateral weight bias the suspension corner "
                               f"weights must account for.",
                               subsystems=["front-suspension", "rear-suspension", "chassis"]))
        return out

    def _check_envelopes(self):
        """Does each subsystem's required box fit inside the chassis interior?"""
        out = []
        env = self.chassis_envelope_mm
        if env is None:
            ch = self.interfaces.get("chassis")
            if ch and None not in (ch.env_x_mm, ch.env_y_mm, ch.env_z_mm):
                env = (ch.env_x_mm, ch.env_y_mm, ch.env_z_mm)
        if env is None:
            out.append(Finding("envelope", Severity.MISSING,
                               "Chassis has not declared the interior envelope it "
                               "offers, so no fit check is possible.",
                               subsystems=["chassis"]))
            return out
        for s in SUBSYSTEMS:
            it = self.interfaces.get(s)
            if it is None or None in (it.env_x_mm, it.env_y_mm, it.env_z_mm):
                continue
            need = (it.env_x_mm, it.env_y_mm, it.env_z_mm)
            if any(n > e + 1e-6 for n, e in zip(need, env)):
                out.append(Finding(
                    "envelope-fit", Severity.FAIL,
                    f"{s} needs a {need[0]:.0f}×{need[1]:.0f}×{need[2]:.0f} mm box "
                    f"but chassis offers {env[0]:.0f}×{env[1]:.0f}×{env[2]:.0f} mm — "
                    f"it does not fit.",
                    subsystems=[s, "chassis"],
                    detail=dict(need=need, have=env)))
        return out

    def _check_thermal(self):
        """Heat-rejecting subsystems vs the airflow cooling can actually move."""
        out = []
        airflow_have = self.total_cooling_airflow_cms
        cool = self.interfaces.get("cooling")
        if cool and cool.cooling_airflow_cms is not None:
            airflow_have = max(airflow_have, cool.cooling_airflow_cms)
        need = 0.0
        heat_sources = []
        for s in SUBSYSTEMS:
            it = self.interfaces.get(s)
            if it is None:
                continue
            if it.cooling_airflow_cms is not None and s != "cooling":
                need += it.cooling_airflow_cms
                heat_sources.append(s)
        if heat_sources:
            if airflow_have <= 0:
                out.append(Finding("cooling-airflow", Severity.MISSING,
                                   f"{', '.join(heat_sources)} require cooling airflow "
                                   f"({need:.3f} m³/s total) but cooling hasn't "
                                   f"declared what the package can move.",
                                   subsystems=heat_sources + ["cooling"]))
            elif need > airflow_have + 1e-9:
                out.append(Finding("cooling-airflow", Severity.FAIL,
                                   f"Cooling can move {airflow_have:.3f} m³/s but "
                                   f"{', '.join(heat_sources)} together need "
                                   f"{need:.3f} m³/s — under-cooled.",
                                   subsystems=heat_sources + ["cooling"],
                                   detail=dict(need=need, have=airflow_have)))
            else:
                out.append(Finding("cooling-airflow", Severity.OK,
                                   f"Cooling airflow {airflow_have:.3f} m³/s covers the "
                                   f"{need:.3f} m³/s required.",
                                   subsystems=heat_sources + ["cooling"]))
        # aero vs cooling: a radiator duct competes with the aero envelope — flag it
        aero = self.interfaces.get("aerodynamics")
        if aero and cool and cool.env_x_mm is not None and aero.is_estimate:
            out.append(Finding("aero-cooling-duct", Severity.INFO,
                               "Cooling occupies an envelope that aero must route air "
                               "to; confirm the radiator duct and aero package don't "
                               "claim the same volume.",
                               subsystems=["aerodynamics", "cooling"]))
        return out

    def _check_electrical(self):
        """Sum LV draws against supply; check HV components match accumulator V."""
        out = []
        lv_draw = 0.0
        lv_users = []
        for s in SUBSYSTEMS:
            it = self.interfaces.get(s)
            if it is None or it.power_draw_w is None:
                continue
            # treat draws at/near LV bus voltage as LV loads
            if it.voltage_v is None or abs(it.voltage_v - self.lv_voltage_v) < 5.0:
                lv_draw += it.power_draw_w
                lv_users.append(s)
        if lv_users:
            if lv_draw > self.lv_supply_capacity_w + 1e-6:
                out.append(Finding("lv-power", Severity.FAIL,
                                   f"LV loads ({', '.join(lv_users)}) draw "
                                   f"{lv_draw:.0f} W but the LV supply provides "
                                   f"{self.lv_supply_capacity_w:.0f} W.",
                                   subsystems=lv_users + ["data-acquisition"],
                                   detail=dict(draw_w=lv_draw,
                                               capacity_w=self.lv_supply_capacity_w)))
            else:
                out.append(Finding("lv-power", Severity.OK,
                                   f"LV draw {lv_draw:.0f} W within "
                                   f"{self.lv_supply_capacity_w:.0f} W supply.",
                                   subsystems=lv_users + ["data-acquisition"]))
        # HV voltage match
        pt = self.interfaces.get("drivetrain")
        if pt and pt.voltage_v is not None and pt.voltage_v > 60.0:
            if abs(pt.voltage_v - self.accumulator_voltage_v) > 0.05 * self.accumulator_voltage_v:
                out.append(Finding("hv-voltage", Severity.WARN,
                                   f"Powertrain expects {pt.voltage_v:.0f} V but the "
                                   f"accumulator is {self.accumulator_voltage_v:.0f} V "
                                   f"— inverter/motor spec mismatch.",
                                   subsystems=["drivetrain"]))
        return out

    def _check_driveline_torque(self):
        """Motor torque the powertrain claims vs what the driveline is rated for."""
        out = []
        pt = self.interfaces.get("drivetrain")
        if not pt or pt.peak_torque_nm is None:
            return out
        limit = self.driveline_torque_limit_nm
        if limit is None:
            sus = self.interfaces.get("suspension")
            limit = sus.peak_torque_nm if sus and sus.peak_torque_nm else None
        if limit is None:
            out.append(Finding("driveline-torque", Severity.MISSING,
                               f"Powertrain delivers {pt.peak_torque_nm:.0f} N·m peak "
                               f"but no driveline/CV torque rating is declared to "
                               f"check it against.",
                               subsystems=["powertrain", "suspension"]))
        elif pt.peak_torque_nm > limit + 1e-6:
            out.append(Finding("driveline-torque", Severity.FAIL,
                               f"Powertrain peak torque {pt.peak_torque_nm:.0f} N·m "
                               f"exceeds the {limit:.0f} N·m driveline rating — "
                               f"driveshaft/CV/upright will be overloaded.",
                               subsystems=["powertrain", "suspension"],
                               detail=dict(torque=pt.peak_torque_nm, limit=limit)))
        else:
            out.append(Finding("driveline-torque", Severity.OK,
                               f"Powertrain torque {pt.peak_torque_nm:.0f} N·m within "
                               f"the {limit:.0f} N·m driveline rating.",
                               subsystems=["powertrain", "suspension"]))
        return out

    def _check_mount_loads(self):
        """Loads a subsystem imposes vs what its carrier designed its mounts for."""
        out = []
        design = self.upright_design_load_n
        for s in SUBSYSTEMS:
            it = self.interfaces.get(s)
            if it is None or it.mount_load_n is None:
                continue
            carrier = it.mounts_on or "chassis"
            ref = design if (carrier == "suspension" and design is not None) else None
            if ref is not None and it.mount_load_n > ref + 1e-6:
                out.append(Finding("mount-load", Severity.FAIL,
                                   f"{s} imposes {it.mount_load_n:.0f} N into "
                                   f"{carrier} mounts, above the {ref:.0f} N they were "
                                   f"designed for.",
                                   subsystems=[s, carrier],
                                   detail=dict(load=it.mount_load_n, design=ref)))
            else:
                out.append(Finding("mount-load", Severity.INFO,
                                   f"{s} imposes {it.mount_load_n:.0f} N into {carrier} "
                                   f"({it.mount_points or '?'} mounts); confirm "
                                   f"{carrier} has designed for it.",
                                   subsystems=[s, carrier]))
        return out

    def _check_estimates_flagged(self):
        """Honesty pass: any check passing on placeholder data is only as good as it."""
        est = [s for s in SUBSYSTEMS
               if self.interfaces.get(s) and self.interfaces[s].is_estimate
               and self.interfaces[s].declared_fields()]
        if est:
            return [Finding("data-provenance", Severity.INFO,
                            f"{len(est)} subsystem(s) are using estimated/placeholder "
                            f"numbers: {', '.join(est)}. Every check above involving "
                            f"them is only as trustworthy as those estimates — "
                            f"replace with measured/CAD values before freezing.",
                            subsystems=est)]
        return []


# --------------------------------------------------------------------------- #
#  Summary helpers
# --------------------------------------------------------------------------- #
def summarize(findings: list) -> dict:
    """Counts by severity + the worst severity present, for a board-level badge."""
    order = [Severity.FAIL, Severity.WARN, Severity.MISSING, Severity.INFO, Severity.OK]
    counts = {s.value: 0 for s in order}
    for f in findings:
        counts[f.severity.value] += 1
    worst = next((s.value for s in order if counts[s.value] > 0), Severity.OK.value)
    return dict(counts=counts, worst=worst, n=len(findings))


def findings_for(findings: list, subsystem: str) -> list:
    """All findings that involve a given subsystem (for its own tab view)."""
    return [f for f in findings if subsystem in f.subsystems]


def blank_ledger() -> IntegrationLedger:
    """An empty ledger with every subsystem present but undeclared (all MISSING)."""
    led = IntegrationLedger()
    for s in SUBSYSTEMS:
        led.set(SubsystemInterface(name=s))
    return led


# --------------------------------------------------------------------------- #
#  Documentation: human labels, change log, and a report export
# --------------------------------------------------------------------------- #
import datetime as _dt

# Human-readable names + units for every interface channel, so the report and the
# change log read like English, not like field names.
FIELD_LABELS = {
    "mass_kg": ("mass", "kg"),
    "cg_x_mm": ("CG x (rearward)", "mm"),
    "cg_y_mm": ("CG y (right)", "mm"),
    "cg_z_mm": ("CG z (up)", "mm"),
    "env_x_mm": ("envelope length", "mm"),
    "env_y_mm": ("envelope width", "mm"),
    "env_z_mm": ("envelope height", "mm"),
    "mount_load_n": ("peak mount load", "N"),
    "mount_points": ("mount points", ""),
    "mounts_on": ("mounts on", ""),
    "power_draw_w": ("power draw", "W"),
    "peak_current_a": ("peak current", "A"),
    "voltage_v": ("voltage", "V"),
    "heat_reject_w": ("heat rejected", "W"),
    "cooling_airflow_cms": ("cooling airflow req", "m³/s"),
    "max_temp_c": ("max temperature", "°C"),
    "peak_torque_nm": ("peak torque", "N·m"),
    "peak_power_kw": ("peak power", "kW"),
    "downforce_n_at_v": ("downforce", "N @ m/s"),
    "drag_n_at_v": ("drag", "N @ m/s"),
    "brake_torque_nm": ("brake torque/corner", "N·m"),
}


def _fmt_val(field, v):
    if v is None:
        return "—"
    label, unit = FIELD_LABELS.get(field, (field, ""))
    if isinstance(v, (tuple, list)):
        body = "/".join(f"{x:g}" for x in v)
    elif isinstance(v, float):
        body = f"{v:g}"
    else:
        body = str(v)
    return f"{body}{(' ' + unit) if unit else ''}"


def diff_interfaces(old_dict, new_iface: SubsystemInterface) -> list:
    """
    Compare a previous interface snapshot (dict) against the current one and return
    a list of plain-language change strings, e.g.
        "mass: 12 kg → 14.5 kg", "downforce: — → 600/15 N @ m/s".
    Used to AUTO-LOG interface edits into the decision/handover record, so the
    documentation writes itself as the team works instead of being a write-up
    scramble before design event.
    """
    old = SubsystemInterface.from_dict(old_dict) if old_dict else SubsystemInterface(name=new_iface.name)
    changes = []
    tracked = list(FIELD_LABELS.keys())
    for f in tracked:
        ov, nv = getattr(old, f, None), getattr(new_iface, f, None)
        if ov != nv:
            label = FIELD_LABELS.get(f, (f, ""))[0]
            changes.append(f"{label}: {_fmt_val(f, ov)} → {_fmt_val(f, nv)}")
    # provenance changes worth recording
    if old.is_estimate != new_iface.is_estimate:
        changes.append("status: "
                       + ("estimate → confirmed" if not new_iface.is_estimate
                          else "confirmed → estimate"))
    if (new_iface.rationale or "").strip() and (old.rationale or "").strip() != (new_iface.rationale or "").strip():
        changes.append("rationale updated")
    return changes


def build_interface_markdown(ledger: IntegrationLedger,
                             team_name: str = "", season: str = "") -> str:
    """
    Render the whole subsystem interface contract as a clean Markdown document —
    design-event / design-report ready. Captures, per subsystem: the declared
    interface values, WHY (rationale), owner, provenance (estimate vs confirmed and
    when it was last touched), plus the car-level budgets, the combined mass/CG, and
    the current integration findings.

    This is the "documentation" half of the tool: it turns the interfaces the team
    has already agreed on into the written justification judges ask for, instead of
    that being a separate scramble. Honest by construction — it shows which numbers
    are still estimates and which checks are passing only on placeholder data.
    """
    today = _dt.date.today().isoformat()
    L = []
    hdr = team_name or "FSAE Team"
    L.append(f"# {hdr} — Subsystem Interface Contract")
    sub = f"_Generated {today}"
    if season:
        sub += f" · season {season}"
    L.append(sub + "_\n")
    L.append("This document is auto-generated from KinematiK's integration ledger. It "
             "records what each subsystem provides to and requires from the rest of the "
             "car, the reasoning behind those numbers, and the cross-subsystem "
             "consistency checks — the interface decisions and their justification in "
             "one place.\n")

    # Car-level contract
    L.append("## Car-level budgets & limits\n")
    L.append(f"- Mass target: **{ledger.target_mass_kg:.1f} kg** "
             f"(incl. {ledger.includes_driver_kg:.0f} kg driver allowance)")
    if ledger.chassis_envelope_mm:
        e = ledger.chassis_envelope_mm
        L.append(f"- Chassis interior envelope: **{e[0]:.0f} × {e[1]:.0f} × {e[2]:.0f} mm**")
    if ledger.driveline_torque_limit_nm:
        L.append(f"- Driveline torque rating: **{ledger.driveline_torque_limit_nm:.0f} N·m**")
    L.append(f"- LV bus: **{ledger.lv_voltage_v:.0f} V**, supply capacity "
             f"**{ledger.lv_supply_capacity_w:.0f} W**")
    L.append(f"- Battery: **{ledger.accumulator_voltage_v:.0f} V** (ignition / DAQ)")
    L.append("")

    # Combined mass + CG (the number that feeds the vehicle model)
    roll = ledger.mass_rollup()
    L.append("## Combined mass & centre of gravity\n")
    if roll["declared"]:
        L.append(f"- Declared subsystem mass: **{roll['total_kg']:.1f} kg** "
                 f"across {len(roll['declared'])} subsystem(s)")
        if roll["cg_mm"]:
            x, y, z = roll["cg_mm"]
            L.append(f"- Mass-weighted CG: **x={x:.0f}, y={y:.0f}, z={z:.0f} mm** "
                     f"(CG height {z:.0f} mm feeds load transfer & the lap sim)")
        if roll["missing"]:
            L.append(f"- ⚠ Not yet declared: {', '.join(roll['missing'])} — "
                     f"the real total is higher than the figure above")
        if roll["any_estimate"]:
            L.append("- ⚠ Some masses are still estimates; treat the total as provisional")
    else:
        L.append("_No subsystem masses declared yet._")
    L.append("")

    # Per-subsystem interfaces
    L.append("## Subsystem interfaces\n")
    for s in SUBSYSTEMS:
        it = ledger.get(s)
        L.append(f"### {s}\n")
        if it is None or not it.declared_fields():
            L.append("_No interface declared yet._\n")
            continue
        prov = "estimate" if it.is_estimate else "confirmed"
        meta = [f"status: **{prov}**"]
        if it.owner:
            meta.append(f"owner: {it.owner}")
        if it.updated_on:
            meta.append(f"last updated: {it.updated_on}"
                        + (f" by {it.updated_by}" if it.updated_by else ""))
        L.append("_" + " · ".join(meta) + "_\n")
        L.append("| Channel | Value |")
        L.append("|---|---|")
        for f in it.declared_fields():
            label, _ = FIELD_LABELS.get(f, (f, ""))
            L.append(f"| {label} | {_fmt_val(f, getattr(it, f))} |")
        L.append("")
        if (it.rationale or "").strip():
            L.append(f"**Why:** {it.rationale.strip()}\n")
        if (it.notes or "").strip():
            L.append(f"_Notes: {it.notes.strip()}_\n")

    # Integration findings
    findings = ledger.check_all()
    summary = summarize(findings)
    L.append("## Integration checks\n")
    L.append(f"Overall: **{summary['worst'].upper()}** — "
             f"{summary['counts']['fail']} conflict(s), "
             f"{summary['counts']['warning']} warning(s), "
             f"{summary['counts']['missing']} missing.\n")
    order = ["fail", "warning", "missing", "info", "ok"]
    for f in sorted(findings, key=lambda x: order.index(x.severity.value)):
        who = ", ".join(f.subsystems) if f.subsystems else "—"
        L.append(f"- **[{f.severity.value.upper()}]** {f.message} _({who})_")
    L.append("")
    L.append("---")
    L.append("_Generated by KinematiK. Numbers flagged as estimates and checks "
             "passing on placeholder data are marked as such — a clean board reflects "
             "only the data entered so far._")
    return "\n".join(L)
