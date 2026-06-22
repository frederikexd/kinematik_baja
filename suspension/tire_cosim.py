# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Structural tire co-simulation boundary — the seam where FTire / CDTire plug in.

WHY THIS MODULE EXISTS (read this before adding a backend)
----------------------------------------------------------
`tiremodel.py` gives KinematiK a *stateless force law*: call `fy(alpha, Fz, gamma)`
and a force comes back instantly, the same answer every time for the same inputs.
The whole tool is built on that contract — `dynamics._corner_force`, the QSS
`lapsim`, and the RK4 `transient` solver all treat the tyre as an algebraic block
re-evaluated from scratch each step. The only tyre *state* anywhere is the lagged
slip angle, and `transient.py` integrates that itself.

FTire (cosin) and CDTire (Fraunhofer) are a different animal entirely. They are
*stateful, structural* models: the carcass deformation, the contact-patch pressure
distribution, the inflation-gas temperature and the per-element tread temperature
are INTERNAL STATE THE TYRE MODEL OWNS and integrates over time. You don't ask
FTire "what's the force at this slip angle"; you hand it a wheel-centre motion and
a 3D road over a timestep, and it returns forces PLUS an evolved internal state.
They run at 1–10+ kHz, ship as compiled commercial libraries with their own
solvers and a fitted parameter file, and there is no open-source build to call.

So integration is NOT "swap in a better fy()". It is defining a co-simulation
boundary the transient solver can drive, and being honest that the structural and
thermal physics live OUTSIDE KinematiK, inside a licensed binary fitted to a real
tyre. This module is that boundary:

    * `StructuralTireModel`  — the Protocol every backend implements (the contract),
    * `WheelState` / `TireOutput` — the per-step exchange across the boundary,
    * `TireProvenance`       — where the parameters came from and what's trustworthy,
    * `ReferenceTireModel`   — a backend that wraps the EXISTING Pacejka so the whole
                               co-sim machinery is testable today, with NO backend
                               binary, while refusing to invent the structural and
                               thermal channels it cannot compute,
    * `FTireModel` / `CDTireModel` — adapter stubs that declare the exact binary API
                               they need and raise a clear, actionable error until a
                               license + parameter file is wired in.

THE HONESTY CONTRACT (why it is strict here, specifically)
----------------------------------------------------------
This is the part that makes the layer *recreate* ADAMS/Car rather than cosplay it.
ADAMS/Car's value with FTire/CDTire is a VALIDATED pipeline: the tyre parameters
were fitted by the vendor to physical rig data, and the co-sim is contracted to be
faithful to that file. The trust lives in PROVENANCE. A structural model that emits
a detailed 3D pressure field and a tread-temperature map from unvalidated defaults
is *more* dangerous than a single mu number, because the output looks like
measurement and nobody questions it — the exact false-confidence failure the rest
of this codebase (see `interfaces.py`) refuses.

Therefore, by construction:
  - Every output channel carries whether it is MEASURED-backend-derived or a
    placeholder, via `TireOutput.synthesized` and the model's `provenance`.
  - The reference backend returns `None` for carcass deformation, contact-patch
    pressure and tread temperature rather than a plausible-looking number. It does
    not have a structural mesh or a thermal network; it says so, it does not fake
    one. Code that needs those channels checks for `None` and degrades honestly.
  - `is_calibrated` is False for any backend running on anything other than a
    vendor parameter file fitted to the actual tyre, mirroring `DamperCurve` and
    `CombinedSlipTire`.

DELIBERATE NON-GOAL: this module does not implement a structural tyre model. Writing
a fake FTire would be the failure above. It owns the SEAM — a clean, typed, tested
protocol — so that a real binary drops in at one place, and so the transient solver
can be written against the stateful contract now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .tiremodel import PacejkaLateral, default_tire, relaxation_length


# --------------------------------------------------------------------------- #
#  Provenance — the thing that makes this trustworthy instead of dangerous
# --------------------------------------------------------------------------- #
class TireFidelity(str, Enum):
    """How physically complete a backend is — for honest UI labelling."""
    HANDLING = "handling"          # force law only (Pacejka-class): Fx/Fy/Mz
    STRUCTURAL = "structural"      # + carcass deformation, contact patch, enveloping
    THERMAL = "thermal"            # + gas/tread temperature, pressure build-up
    STRUCTURAL_THERMAL = "structural+thermal"   # FTire / CDTire full scope


@dataclass
class TireProvenance:
    """
    Where a tyre model's parameters came from and what they're worth — the single
    most important object in this module. A clean co-sim that ran on vendor-default
    parameters is NOT the same as one fitted to your tyre, and this records the
    difference so the board can never imply more than the data behind it.
    """
    backend: str                       # "reference-pacejka", "ftire", "cdtire"
    fidelity: TireFidelity
    is_calibrated: bool = False        # True only on a vendor file fitted to THIS tyre
    parameter_file: Optional[str] = None   # the .tir / FTire / CDTire property file
    fitted_to: str = ""                # what rig data the file was fitted to, if known
    notes: str = ""

    def status(self) -> str:
        """One-line human summary, in the style of DamperCurve.status()."""
        if self.is_calibrated:
            src = self.parameter_file or "a vendor parameter file"
            fit = f", fitted to {self.fitted_to}" if self.fitted_to else ""
            return (f"{self.backend} ({self.fidelity.value}), calibrated from "
                    f"{src}{fit}")
        return (f"{self.backend} ({self.fidelity.value}), UNCALIBRATED — running on "
                f"reference/default parameters, NOT a measured-tyre file; treat "
                f"absolute and structural/thermal outputs as placeholders")


# --------------------------------------------------------------------------- #
#  The per-step exchange across the boundary
# --------------------------------------------------------------------------- #
@dataclass
class WheelState:
    """
    Everything a stateful tyre needs to advance one step, expressed at the WHEEL
    CENTRE in the tyre's own frame. This is the input half of the co-sim contract.

    A handling-class backend uses only (alpha, kappa, Fz, gamma, omega, v...).
    A structural backend additionally uses the 3D road under the contact patch
    (`road_points`) and the wheel-centre vertical motion to deform its carcass;
    a thermal backend additionally uses speed/load history (carried in its own
    internal state) plus ambient/track temperature.
    """
    # --- kinematic state at the wheel centre ---
    alpha: float = 0.0                 # slip angle, rad
    kappa: float = 0.0                 # longitudinal slip ratio, -
    gamma: float = 0.0                 # inclination (camber) angle, rad
    Fz: float = 0.0                    # vertical load demand, N (handling backends)
    omega: float = 0.0                 # wheel spin rate, rad/s
    v_x: float = 0.0                   # forward speed at contact, m/s
    v_y: float = 0.0                   # lateral speed at contact, m/s
    # --- wheel-centre vertical (structural backends integrate the carcass) ---
    z_wheel: float = 0.0               # wheel-centre height vs static, m
    zdot_wheel: float = 0.0            # wheel-centre vertical velocity, m/s
    # --- 3D road under the patch (structural / enveloping; None => flat) ---
    # shape (k,3): points (x,y,z) in the contact region, tyre frame, metres.
    road_points: Optional[np.ndarray] = None
    # --- environment (thermal backends) ---
    ambient_temp_c: float = 25.0
    track_temp_c: float = 30.0
    inflation_pressure_pa: Optional[float] = None   # cold/set pressure; None => model default
    dt: float = 1.0e-3                 # step the backend should advance over, s


@dataclass
class TireOutput:
    """
    What a tyre backend returns after advancing one step — the output half of the
    contract. Forces/moments are always present. The STRUCTURAL and THERMAL channels
    are Optional and are None on any backend that does not actually compute them:
    a None here means "this model cannot tell you that", which is the honest answer,
    NOT zero and NOT a guess.

    `synthesized` lists, by name, any channel whose value is a placeholder rather
    than backend-physics — so a consumer (and the UI) can flag it the way
    CombinedSlipTire.status() flags the uncalibrated ellipse.
    """
    # --- forces & moments at the contact patch, tyre frame (always present) ---
    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0                    # actual vertical reaction (structural: from carcass)
    Mz: float = 0.0                    # aligning moment, N·m
    Mx: float = 0.0                    # overturning moment, N·m
    My: float = 0.0                    # rolling-resistance moment, N·m
    # --- structural channels (None unless the backend has a carcass model) ---
    carcass_deflection_m: Optional[float] = None       # radial deflection at patch centre
    contact_length_m: Optional[float] = None           # patch length
    contact_width_m: Optional[float] = None             # patch width
    pressure_distribution: Optional[np.ndarray] = None  # (ny,nx) patch pressure, Pa
    effective_radius_m: Optional[float] = None
    # --- thermal / pressure channels (None unless the backend has a thermal net) ---
    tread_temp_c: Optional[np.ndarray] = None           # per-band tread temperature
    carcass_temp_c: Optional[float] = None
    gas_temp_c: Optional[float] = None
    inflation_pressure_pa: Optional[float] = None       # current (hot) pressure
    # --- provenance of THIS sample ---
    synthesized: list[str] = field(default_factory=list)

    def is_structural(self) -> bool:
        return self.carcass_deflection_m is not None

    def is_thermal(self) -> bool:
        return self.gas_temp_c is not None or self.tread_temp_c is not None


# --------------------------------------------------------------------------- #
#  The contract every backend implements
# --------------------------------------------------------------------------- #
@runtime_checkable
class StructuralTireModel(Protocol):
    """
    The co-simulation contract. A backend is STATEFUL: `reset()` initialises its
    internal state (carcass at rest, gas at ambient), and each `step(WheelState)`
    advances that state by `WheelState.dt` and returns the resulting `TireOutput`.
    This is the FTire/CDTire interaction pattern (hand it motion + road over a step,
    get forces + evolved state back), reduced to what KinematiK actually drives.

    Backends MUST NOT raise from `step` — like the rest of the repo, they clamp and
    flag (via `TireOutput.synthesized` and `warnings()`), so a 10 000-step run is
    never taken down by one pathological sample.
    """
    def provenance(self) -> TireProvenance: ...
    def reset(self, state: Optional[WheelState] = None) -> None: ...
    def step(self, ws: WheelState) -> TireOutput: ...
    def warnings(self) -> list[str]: ...


# --------------------------------------------------------------------------- #
#  Reference backend: wraps the EXISTING Pacejka in the stateful contract
# --------------------------------------------------------------------------- #
class ReferenceTireModel:
    """
    A handling-fidelity backend that satisfies the co-sim contract using the tyre
    model KinematiK already has. Its job is to make the WHOLE boundary — the
    protocol, the transient driver, the tests — exercisable today with no external
    binary, and to be the fallback when no FTire/CDTire license is present.

    What it DOES (real physics, same as tiremodel.py):
        * Fx/Fy via the fitted Pacejka + friction-ellipse coupling,
        * lateral relaxation as a genuine internal state (the carcass lag), advanced
          per step — so even this backend is honestly *stateful* in the one channel
          the underlying physics supports.

    What it explicitly does NOT do, and will NOT fake:
        * carcass deformation field, contact-patch pressure distribution, enveloping
          over 3D road points  -> returned as None,
        * tread / gas temperature, pressure build-up                -> returned as None.
    A Pacejka model has no mesh and no thermal network; inventing those numbers is
    the precise false-confidence failure this layer exists to prevent. Consumers
    that need them must check for None and degrade honestly (or attach a real
    structural backend).
    """

    def __init__(self, lateral: Optional[PacejkaLateral] = None,
                 mu_x_ratio: float = 1.05, ell_kx: float = 2.0, ell_ky: float = 2.0):
        self.lateral = lateral or default_tire()
        self.mu_x_ratio = float(mu_x_ratio)
        self.ell_kx = float(ell_kx)
        self.ell_ky = float(ell_ky)
        self._alpha_lag = 0.0          # the one genuine internal state we own
        self._warnings: list[str] = []
        # channels this backend cannot produce — named once, attached to every sample
        self._absent = ["carcass_deflection_m", "contact_length_m", "contact_width_m",
                        "pressure_distribution", "tread_temp_c", "carcass_temp_c",
                        "gas_temp_c"]

    def provenance(self) -> TireProvenance:
        return TireProvenance(
            backend="reference-pacejka",
            fidelity=TireFidelity.HANDLING,
            is_calibrated=False,
            parameter_file=None,
            notes="Pacejka MF5.2 wrapped in the co-sim contract. Handling-only: no "
                  "structural or thermal channels — those are None by design, not "
                  "zero. Attach FTire/CDTire for those.")

    def reset(self, state: Optional[WheelState] = None) -> None:
        self._alpha_lag = float(state.alpha) if state is not None else 0.0
        self._warnings = []

    def _warn(self, msg: str):
        if msg not in self._warnings:
            self._warnings.append(msg)

    def warnings(self) -> list[str]:
        return list(self._warnings)

    def step(self, ws: WheelState) -> TireOutput:
        """Advance the slip-relaxation state one step and return handling forces."""
        try:
            Fz = max(float(ws.Fz), 0.0)
            if Fz <= 1.0:
                # no load -> no force; still advance the lag toward target
                self._relax(ws)
                out = TireOutput(Fx=0.0, Fy=0.0, Fz=Fz)
                out.synthesized = list(self._absent)
                return out

            self._relax(ws)
            gamma = float(ws.gamma)

            # pure lateral from the lagged slip, pure-ish long from kappa demand
            fy_pure = float(self.lateral.fy(self._alpha_lag, Fz, gamma))
            fy_max = float(self.lateral.peak_force(Fz, abs(gamma)))
            fx_max = self.mu_x_ratio * fy_max

            # longitudinal force from slip ratio, capped to the ellipse given lateral use
            # (a simple saturating map; the reference backend is handling-grade, the
            #  real Fx(kappa) curve is a structural/empirical backend's job)
            fx_demand = float(np.clip(ws.kappa, -1.0, 1.0)) * fx_max
            if fy_max > 1e-6:
                use_y = min(abs(fy_pure) / fy_max, 1.0)
                fx_avail = fx_max * max(1.0 - use_y ** self.ell_ky, 0.0) ** (1.0 / self.ell_kx)
            else:
                fx_avail = fx_max
            Fx = float(np.clip(fx_demand, -fx_avail, fx_avail))

            if fx_max > 1e-6:
                use_x = min(abs(Fx) / fx_max, 1.0)
                fy_avail = fy_max * max(1.0 - use_x ** self.ell_kx, 0.0) ** (1.0 / self.ell_ky)
            else:
                fy_avail = abs(fy_pure)
            Fy = float(np.clip(fy_pure, -fy_avail, fy_avail))

            out = TireOutput(Fx=Fx, Fy=Fy, Fz=Fz)
            # everything structural/thermal is honestly absent
            out.synthesized = list(self._absent)
            return out
        except Exception as e:
            self._warn(f"reference backend step failed ({type(e).__name__}); "
                       f"returned zero force for this sample.")
            out = TireOutput(Fx=0.0, Fy=0.0, Fz=max(float(ws.Fz), 0.0))
            out.synthesized = list(self._absent)
            return out

    def _relax(self, ws: WheelState):
        """Advance the lateral relaxation state — the one real internal state here."""
        try:
            sigma = max(relaxation_length(max(float(ws.Fz), 1.0)), 1e-3)
            Vx = max(abs(float(ws.v_x)), 0.3)
            dt = max(float(ws.dt), 0.0)
            # exact first-order step: d(al)/dt = (Vx/sigma)(target - al)
            decay = math.exp(-(Vx / sigma) * dt)
            self._alpha_lag = float(ws.alpha) + (self._alpha_lag - float(ws.alpha)) * decay
        except Exception:
            self._alpha_lag = float(ws.alpha)


# --------------------------------------------------------------------------- #
#  FTire / CDTire adapter stubs — the one place a real binary plugs in
# --------------------------------------------------------------------------- #
class _ExternalTireBackend:
    """
    Shared scaffolding for a real commercial backend. It declares the binary API
    KinematiK needs and the parameter file it must be given, validates what it can,
    and raises a single, actionable error if the license/binary/file isn't present.
    It does NOT pretend to compute anything — there is no fake structural physics
    here. Subclasses set the vendor specifics.
    """
    _vendor = "external"
    _fidelity = TireFidelity.STRUCTURAL_THERMAL
    # The interface KinematiK expects of the vendor's co-sim library, named so a
    # wrapper author knows exactly what to bind. (FTire ships a C/Fortran cosim
    # interface + the cosin/utilities Python bindings; CDTire ships an FMU / C API.)
    _required_binding = (
        "a callable that takes a wheel-centre state + 3D road over a timestep and "
        "returns forces, moments, and the evolved structural/thermal state "
        "(FTire: cosin co-sim interface or FMU; CDTire: FMU / C API)")

    def __init__(self, parameter_file: str, library_path: Optional[str] = None,
                 binding=None, fitted_to: str = ""):
        self.parameter_file = parameter_file
        self.library_path = library_path
        self._binding = binding             # the bound vendor co-sim callable
        self._fitted_to = fitted_to
        self._warnings: list[str] = []
        self._validate()

    def _validate(self):
        if self._binding is None:
            raise NotImplementedError(
                f"{self._vendor} backend is a stub: no co-sim binding was provided. "
                f"This module owns the integration SEAM; the {self._vendor} solver "
                f"itself is a licensed binary that is not bundled. To enable it, pass "
                f"`binding=` — {self._required_binding} — and a `parameter_file` "
                f"fitted to your tyre. Until then, use ReferenceTireModel (handling-"
                f"grade, honest about what it can't compute).")

    def provenance(self) -> TireProvenance:
        return TireProvenance(
            backend=self._vendor,
            fidelity=self._fidelity,
            # calibrated only once a real fitted file AND a live binding are present
            is_calibrated=self._binding is not None and bool(self.parameter_file),
            parameter_file=self.parameter_file,
            fitted_to=self._fitted_to)

    def reset(self, state: Optional[WheelState] = None) -> None:
        if self._binding is None:
            return
        # a real wrapper initialises the vendor state here (carcass at rest, gas at
        # the WheelState ambient/track temperature and set pressure)
        raise NotImplementedError(
            f"Bind {self._vendor}'s state-initialise entry point in a subclass.")

    def step(self, ws: WheelState) -> TireOutput:
        if self._binding is None:
            # never reached (constructor raises), but kept explicit for safety
            raise NotImplementedError(f"{self._vendor} backend not bound.")
        # a real wrapper marshals `ws` into the vendor call, advances dt, and maps
        # the returned forces + structural + thermal state into TireOutput. Because
        # those come from the vendor physics, NONE of them go in `synthesized`.
        raise NotImplementedError(
            f"Map {self._vendor}'s per-step co-sim call into TireOutput in a "
            f"subclass: forces/moments + carcass_deflection_m, pressure_distribution, "
            f"tread_temp_c, gas_temp_c, inflation_pressure_pa.")

    def warnings(self) -> list[str]:
        return list(self._warnings)


class FTireModel(_ExternalTireBackend):
    """
    Adapter for cosin's FTire (Flexible Structure Tire Model). Models the carcass as
    a flexible belt of mass nodes on a rim, so it resolves enveloping over short
    obstacles (kerbs, cleats), in-plane/out-of-plane carcass dynamics into the
    100s of Hz, and — with the thermal/TKC module — tread and gas temperature and
    pressure build-up. KinematiK drives it through the same `step(WheelState)`
    contract as everything else; this class declares the binding it needs and stays
    a stub (raising a clear error) until a license + `.tir`/FTire property file and
    the cosin co-sim binding are supplied.
    """
    _vendor = "ftire"
    _fidelity = TireFidelity.STRUCTURAL_THERMAL
    _required_binding = (
        "the cosin co-sim interface (FTire/cosim C/Fortran API, the cosin Python "
        "bindings, or an FTire FMU) — a callable advancing the belt+thermal state "
        "one step from wheel-centre motion and a 3D road")


class CDTireModel(_ExternalTireBackend):
    """
    Adapter for Fraunhofer ITWM's CDTire. A family (CDTire/3D being the structural
    shell-element model) that resolves carcass deformation, contact-patch pressure,
    and — with CDTire/Thermal — tread/gas temperature, aimed at ride, durability
    and NVH as well as handling. Same `step(WheelState)` contract; stays a stub
    until its FMU / C API and a fitted parameter file are wired in via `binding=`.
    """
    _vendor = "cdtire"
    _fidelity = TireFidelity.STRUCTURAL_THERMAL
    _required_binding = (
        "the CDTire FMU (via an FMI runtime) or its C API — a callable advancing the "
        "shell + thermal state one step from wheel-centre motion and a 3D road")


# --------------------------------------------------------------------------- #
#  Factory + helpers
# --------------------------------------------------------------------------- #
def make_tire_backend(kind: str = "reference", **kw) -> StructuralTireModel:
    """
    Build a backend by name. `kind` in {reference, thermal, ftire, cdtire}.
    `reference` is handling-only (no thermal/structural channels). `thermal` adds a
    lumped tread/carcass/gas thermal network on the same Pacejka core — real
    energy-balance physics, but UNCALIBRATED (every thermal channel flagged
    synthesized) until you supply temperature-swept data. The external backends
    (ftire, cdtire) require a `parameter_file=` and a `binding=`; without them they
    raise a clear, actionable error (by design — the seam is here, the binary is
    not). `reference` and `thermal` always work and need no binary.
    """
    kind = (kind or "reference").lower()
    if kind in ("reference", "pacejka", "default"):
        return ReferenceTireModel(lateral=kw.get("lateral"))
    if kind in ("thermal", "lumped-thermal", "lumped_thermal"):
        # Lumped 3-node thermal channel on the Pacejka core. Real energy-balance
        # physics, UNCALIBRATED parameters (flagged synthesized) until you supply
        # temperature-swept data. Imported lazily to avoid a module import cycle.
        from .tire_thermal import ThermalTireModel
        return ThermalTireModel(lateral=kw.get("lateral"),
                                params=kw.get("params"),
                                init_temp_c=kw.get("init_temp_c"))
    if kind == "ftire":
        return FTireModel(parameter_file=kw.get("parameter_file", ""),
                          library_path=kw.get("library_path"),
                          binding=kw.get("binding"),
                          fitted_to=kw.get("fitted_to", ""))
    if kind == "cdtire":
        return CDTireModel(parameter_file=kw.get("parameter_file", ""),
                           library_path=kw.get("library_path"),
                           binding=kw.get("binding"),
                           fitted_to=kw.get("fitted_to", ""))
    raise ValueError(f"Unknown tyre backend '{kind}'. "
                     f"Options: reference, thermal, ftire, cdtire.")


def default_structural_tire() -> StructuralTireModel:
    """The honest default: the handling-grade reference backend, no binary needed."""
    return ReferenceTireModel()
