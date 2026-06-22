# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Lumped-parameter tire thermal channel for the co-simulation boundary.

WHY THIS MODULE EXISTS (read this before trusting a number it produces)
-----------------------------------------------------------------------
`tire_cosim.py` is deliberately blunt about it: the ReferenceTireModel returns
`None` for tread/gas temperature, because a Pacejka force law has no thermal
network and inventing a temperature "looks like measurement and nobody questions
it" — the exact false-confidence failure the rest of the repo refuses. That is the
right default. It also leaves a real hole: KinematiK could not say *anything* about
how the tyre heats up over a lap, so every grip number is implicitly a single-
temperature snapshot, and the user has no way to reason about warm-up, thermal
degradation into a long run, or the front/rear temperature split that decides
balance late in a stint.

This module fills that hole the only honest way it can: with a TRANSPARENT,
first-principles **lumped thermal network** — not a fabricated lookup, not a
vendor surrogate. Three thermal masses per tyre (surface tread band(s), bulk
carcass, inflation gas) exchange energy through:

    * frictional heat generation at the contact patch  Q_fric = |F_slip · v_slip|
      (the sliding component of the tyre force times the sliding velocity),
    * cyclic/rolling-resistance hysteresis heating       Q_roll = My·omega-class loss,
    * conduction tread -> carcass and carcass -> gas,
    * forced convection tread -> air (speed-dependent h) and to the track surface,
    * the ideal-gas pressure rise from the gas-temperature change.

Every term is a textbook energy balance. The EQUATIONS are safe and physical. What
is NOT safe — and what makes this an UNCALIBRATED channel, flagged on every sample
in `TireOutput.synthesized` exactly like `CombinedSlipTire` flags its ellipse — is
the PARAMETERS: the masses, specific heats, the convection coefficients, the
contact-conduction coefficient to the track, and above all the grip-vs-temperature
law mu(T). Those cannot be known without **empirical, temperature-swept tyre data**
(a TTC run sweeping inflation/temperature, or a vendor thermal file). Without that
data the model gives you the right SHAPE — it warms up, it saturates, fronts and
rears diverge, a long run cooks the surface — with representative-but-invented
magnitudes. That is genuinely useful for relative setup work and completely unsafe
as an absolute temperature, and this module says so, loudly, in `provenance()`,
in `warnings()`, and per-channel.

THE HONESTY CONTRACT, APPLIED
-----------------------------
  * `ThermalTireModel` is a real `StructuralTireModel` backend (it satisfies the
    same `step(WheelState)->TireOutput` protocol), so it drops into the existing
    co-sim driver, the transient solver, and the four-corner set with no new wiring.
  * It DOES compute and return `tread_temp_c`, `carcass_temp_c`, `gas_temp_c`,
    `inflation_pressure_pa` — from the network above, integrated per step.
  * It puts EVERY one of those channel names in `TireOutput.synthesized` unless it
    was built from a calibrated thermal parameter set (`ThermalParams.calibrated`,
    default False). Synthesized here means "physically-shaped, parametrically
    invented" — the same status the rest of the repo uses for the friction ellipse.
  * `provenance().fidelity` is `TireFidelity.THERMAL`, and `is_calibrated` is False
    unless you pass a parameter set fitted to temperature-swept data.
  * The OPTIONAL grip feedback — letting the modelled temperature scale Pacejka
    grip through `mu(T)` — is OFF by default, and when on it adds `"mu_thermal"` to
    `synthesized`, because the mu(T) curve is the single most data-hungry, most
    abusable part: a plausible-looking warm-up grip gain with no measured peak
    temperature behind it is worse than no curve at all.

DELIBERATE NON-GOAL: this is not FTire/CDTire's thermal module. It is a 3-node
lumped network, not a per-element tread map over a structural belt. Where FTire
gives a calibrated finite-element temperature field, this gives a defensible
warm-up curve whose absolute level you must not trust until you calibrate it. The
FTire/CDTire seam in `tire_cosim.py` remains the place a calibrated thermal field
plugs in; this module is what KinematiK can honestly compute on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tiremodel import PacejkaLateral, default_tire, relaxation_length
from .tire_cosim import (TireFidelity, TireProvenance, WheelState, TireOutput,
                         ReferenceTireModel)


# --------------------------------------------------------------------------- #
#  Thermal parameter set — the part that needs temperature-swept data
# --------------------------------------------------------------------------- #
@dataclass
class ThermalParams:
    """
    The lumped thermal network's parameters. Defaults are REPRESENTATIVE values for
    a ~205/470-R13 FSAE-class slick on an aluminium rim — chosen so the warm-up
    curve has a sensible shape and time constant, NOT measured on any specific tyre.

    `calibrated` is the single most important field: leave it False (the default)
    and every thermal output this model produces is flagged `synthesized`. Set it
    True ONLY when every parameter below came from temperature-swept rig/track data
    for the actual tyre — that is what lets the co-sim board report a thermal number
    as measurement rather than as a physically-shaped guess.
    """
    # --- thermal masses (per tyre) ---
    n_bands: int = 3                    # tread bands across the width (inner/mid/outer)
    m_tread_kg: float = 1.2             # total tread-rubber mass that heats fast
    m_carcass_kg: float = 3.8           # carcass + belt thermal mass (slower)
    m_gas_kg: float = 0.035             # inflation gas mass (~air in the cavity)

    cp_tread: float = 1900.0            # J/(kg·K), rubber
    cp_carcass: float = 1100.0          # J/(kg·K), carcass composite (effective)
    cp_gas: float = 1005.0              # J/(kg·K), air at constant pressure

    # --- conduction (W/K) between lumped nodes ---
    k_tread_carcass: float = 18.0       # tread band <-> carcass
    k_carcass_gas: float = 6.0          # carcass <-> inflation gas

    # --- convection ---
    # forced convection coefficient to ambient air, modelled h = h0 + h1*v_x
    h_air0: float = 12.0                # W/(m²·K) at standstill
    h_air1: float = 5.5                 # W/(m²·K) per (m/s)
    area_tread_m2: float = 0.16         # tread surface exposed to air (per tyre)
    # contact conduction to the track surface while the patch is on the ground
    k_track: float = 22.0               # W/K, tread band <-> track (patch contact)
    area_carcass_m2: float = 0.35       # carcass shoulder/sidewall convective area

    # --- heat generation split ---
    # fraction of frictional sliding power that ends up in the tread (rest radiates
    # / goes to the road instantly); rolling hysteresis loss coefficient
    fric_to_tread: float = 0.72
    roll_resist_coeff: float = 0.018    # effective Crr feeding hysteresis heating
    eff_radius_m: float = 0.22          # rolling radius for the rolling-loss term

    # --- inflation gas (ideal-gas pressure rise) ---
    cold_pressure_pa: float = 83000.0   # set/cold pressure (gauge ~12 psi + atm)
    atm_pressure_pa: float = 101325.0

    # --- optional grip-vs-temperature law mu(T) (the most data-hungry part) ---
    # multiplicative scaling on Pacejka grip: 1.0 at T_opt, falling either side.
    # OFF by default; when enabled it is flagged synthesized as "mu_thermal".
    enable_mu_feedback: bool = False
    T_opt_c: float = 85.0               # temperature of peak grip
    mu_gain_per_C: float = 0.0042       # grip lost per °C away from optimum (cold side)
    mu_gain_per_C_hot: float = 0.0065   # steeper falloff above optimum (overheating)
    mu_floor: float = 0.55              # never scale grip below this fraction

    # --- provenance ---
    calibrated: bool = False
    fitted_to: str = ""

    def h_air(self, v_x: float) -> float:
        return self.h_air0 + self.h_air1 * max(abs(float(v_x)), 0.0)

    def mu_scale(self, T_c: float) -> float:
        """Grip multiplier from tread temperature. Identity unless mu feedback on."""
        if not self.enable_mu_feedback:
            return 1.0
        dT = float(T_c) - self.T_opt_c
        slope = self.mu_gain_per_C if dT < 0 else self.mu_gain_per_C_hot
        s = 1.0 - slope * abs(dT)
        return float(max(s, self.mu_floor))


def default_thermal_params() -> ThermalParams:
    """Representative, UNCALIBRATED parameter set (calibrated=False by design)."""
    return ThermalParams()


# --------------------------------------------------------------------------- #
#  The thermal backend
# --------------------------------------------------------------------------- #
class ThermalTireModel:
    """
    A THERMAL-fidelity co-sim backend: the existing handling physics (Pacejka +
    friction ellipse, via an embedded ReferenceTireModel) PLUS a 3-node lumped
    thermal network integrated per step. It returns the same forces the reference
    backend would, and additionally fills the thermal channels that backend leaves
    None — while flagging them `synthesized` unless built from calibrated params.

    State owned by this backend (advanced every `step`):
        * the slip-relaxation lag (delegated to the embedded ReferenceTireModel),
        * tread-band temperatures  T_tread[n_bands]   (°C),
        * carcass bulk temperature T_carcass          (°C),
        * inflation-gas temperature T_gas             (°C),
      from which the hot inflation pressure follows by the ideal-gas law.

    Optional grip feedback: if `params.enable_mu_feedback`, the modelled mean tread
    temperature scales the Pacejka grip through `params.mu_scale(T)` BEFORE the
    forces are computed for the step — closing the temperature->grip loop. This is
    off by default and flagged "mu_thermal" in `synthesized` when on, because the
    mu(T) curve is exactly what you cannot know without temperature-swept data.
    """

    _THERMAL_CHANNELS = ("tread_temp_c", "carcass_temp_c", "gas_temp_c",
                         "inflation_pressure_pa")

    def __init__(self, lateral: Optional[PacejkaLateral] = None,
                 params: Optional[ThermalParams] = None,
                 reference: Optional[ReferenceTireModel] = None,
                 init_temp_c: Optional[float] = None):
        self.lateral = lateral or default_tire()
        self.params = params or default_thermal_params()
        # the force law lives in the reference backend so we don't duplicate the
        # friction-ellipse coupling; we only override its grip via mu(T) when asked.
        self._ref = reference or ReferenceTireModel(lateral=self.lateral)
        self._warnings: list[str] = []
        self._init_temp = init_temp_c
        self._init_thermal_state(init_temp_c)

    # ---- state init -------------------------------------------------------- #
    def _init_thermal_state(self, init_temp_c: Optional[float]):
        p = self.params
        T0 = float(init_temp_c) if init_temp_c is not None else 25.0
        self.T_tread = np.full(int(max(p.n_bands, 1)), T0, dtype=float)
        self.T_carcass = T0
        self.T_gas = T0
        self._mu_scale_last = 1.0

    def reset(self, state: Optional[WheelState] = None) -> None:
        self._ref.reset(state)
        self._warnings = []
        if state is not None:
            # start the rubber at ambient/track, the natural cold condition
            T0 = 0.5 * (float(state.ambient_temp_c) + float(state.track_temp_c))
            self._init_thermal_state(T0)
        else:
            self._init_thermal_state(self._init_temp)

    # ---- provenance / warnings -------------------------------------------- #
    def provenance(self) -> TireProvenance:
        p = self.params
        if p.calibrated:
            note = ("3-node lumped thermal network (tread/carcass/gas) on a Pacejka "
                    "force core, calibrated to temperature-swept data.")
        else:
            note = ("3-node lumped thermal network (tread/carcass/gas) on a Pacejka "
                    "force core. UNCALIBRATED: equations are textbook energy "
                    "balance, but masses, heat-transfer coefficients and the mu(T) "
                    "law are representative defaults, NOT fitted to temperature-"
                    "swept tyre data. Treat temperatures as a physically-shaped "
                    "warm-up curve, good for relative/setup work, NOT as measured "
                    "degrees. Every thermal channel is flagged synthesized.")
        return TireProvenance(
            backend="lumped-thermal",
            fidelity=TireFidelity.THERMAL,
            is_calibrated=bool(p.calibrated),
            parameter_file=None,
            fitted_to=p.fitted_to,
            notes=note)

    def _warn(self, msg: str):
        if msg not in self._warnings:
            self._warnings.append(msg)

    def warnings(self) -> list[str]:
        w = list(self._ref.warnings()) + list(self._warnings)
        if not self.params.calibrated:
            self._uncal_warn(w)
        return w

    def _uncal_warn(self, w):
        msg = ("thermal channels are UNCALIBRATED (representative parameters, no "
               "temperature-swept data) — flagged synthesized; do not read as "
               "absolute temperature.")
        if msg not in w:
            w.append(msg)

    # ---- helpers ----------------------------------------------------------- #
    def mean_tread_c(self) -> float:
        return float(np.mean(self.T_tread))

    def hot_pressure_pa(self) -> float:
        """Ideal-gas pressure at the current gas temperature from the cold set."""
        p = self.params
        T_cold_k = 25.0 + 273.15
        T_hot_k = self.T_gas + 273.15
        abs_cold = p.cold_pressure_pa + p.atm_pressure_pa
        abs_hot = abs_cold * (T_hot_k / max(T_cold_k, 1.0))
        return float(abs_hot - p.atm_pressure_pa)

    # ---- the synthesized-channel list for a given sample ------------------ #
    def _synth(self) -> list[str]:
        if self.params.calibrated:
            base = []
        else:
            base = list(self._THERMAL_CHANNELS)
        # structural channels remain genuinely absent on this backend
        base += ["carcass_deflection_m", "contact_length_m", "contact_width_m",
                 "pressure_distribution"]
        if self.params.enable_mu_feedback and not self.params.calibrated:
            base.append("mu_thermal")
        return base

    # ---- the step ---------------------------------------------------------- #
    def step(self, ws: WheelState) -> TireOutput:
        """
        Advance forces AND the thermal network one step. Never raises: on any
        internal failure it holds the previous temperatures and flags a warning,
        mirroring the rest of the co-sim layer.
        """
        p = self.params
        try:
            # ---- 0. sanitise inputs: a non-finite wheel state must not poison the
            #         thermal integration. Clamp to finite, physical-ish bounds.
            def _finite(x, lo, hi, default=0.0):
                xf = float(x)
                if not math.isfinite(xf):
                    return default
                return float(min(max(xf, lo), hi))
            alpha = _finite(ws.alpha, -1.5, 1.5)
            kappa = _finite(ws.kappa, -1.0, 1.0)
            v_x_s = _finite(ws.v_x, 0.0, 150.0)
            dt = _finite(ws.dt, 0.0, 1.0)
            Fz = _finite(ws.Fz, 0.0, 1.0e5)
            T_amb = _finite(ws.ambient_temp_c, -40.0, 80.0, 25.0)
            T_track = _finite(ws.track_temp_c, -40.0, 120.0, 30.0)

            # ---- 1. grip feedback: scale the Pacejka grip by mu(T) if enabled ---
            mu_scale = p.mu_scale(self.mean_tread_c())
            self._mu_scale_last = mu_scale
            self._apply_mu_scale(mu_scale)

            # ---- 2. forces from the embedded handling backend ----------------
            out = self._ref.step(ws)

            self._restore_mu_scale()

            # ---- 3. heat generation this step --------------------------------
            v_x = v_x_s

            # sliding (friction) power: the slip velocities times the in-plane forces.
            # lateral slide speed ~ v_x * tan(alpha); long. slide speed ~ v_x * kappa.
            v_slide_y = v_x * math.tan(alpha)
            v_slide_x = v_x * kappa
            Q_fric = abs(out.Fy * v_slide_y) + abs(out.Fx * v_slide_x)   # W
            # rolling-resistance hysteresis: Crr * Fz * v_x  (W)
            Q_roll = p.roll_resist_coeff * Fz * v_x

            Q_gen = (Q_fric + Q_roll) * p.fric_to_tread                   # W into tread
            # split heat generation across the bands by load/slip emphasis. With a
            # single band it all goes to that band; with several, weight the slip
            # side slightly toward the loaded shoulder (camber/load proxy).
            band_w = self._band_weights(ws)
            Q_band = Q_gen * band_w                                       # W per band

            # ---- 4. conduction & convection per node -------------------------
            h_air = p.h_air(v_x)
            A_band = p.area_tread_m2 / len(self.T_tread)
            on_ground = Fz > 1.0

            dT_tread = np.zeros_like(self.T_tread)
            for i in range(len(self.T_tread)):
                Tt = self.T_tread[i]
                # generation
                q = Q_band[i]
                # conduction tread band -> carcass
                q -= p.k_tread_carcass / len(self.T_tread) * (Tt - self.T_carcass)
                # convection to air
                q -= h_air * A_band * (Tt - T_amb)
                # conduction to track through the patch (only while loaded)
                if on_ground:
                    q -= p.k_track / len(self.T_tread) * (Tt - T_track)
                C_band = p.m_tread_kg / len(self.T_tread) * p.cp_tread
                dT_tread[i] = q / max(C_band, 1e-6) * dt

            # carcass node
            q_c = 0.0
            q_c += p.k_tread_carcass * (self.mean_tread_c() - self.T_carcass)
            q_c -= p.k_carcass_gas * (self.T_carcass - self.T_gas)
            q_c -= h_air * p.area_carcass_m2 * (self.T_carcass - T_amb)
            C_c = p.m_carcass_kg * p.cp_carcass
            dT_carcass = q_c / max(C_c, 1e-6) * dt

            # gas node
            q_g = p.k_carcass_gas * (self.T_carcass - self.T_gas)
            C_g = p.m_gas_kg * p.cp_gas
            dT_gas = q_g / max(C_g, 1e-6) * dt

            # ---- 5. integrate (explicit Euler, clamped) ----------------------
            self.T_tread = np.clip(self.T_tread + dT_tread, -40.0, 350.0)
            self.T_carcass = float(np.clip(self.T_carcass + dT_carcass, -40.0, 300.0))
            self.T_gas = float(np.clip(self.T_gas + dT_gas, -40.0, 250.0))

            # ---- 6. fill the thermal channels on the output ------------------
            out.tread_temp_c = self.T_tread.copy()
            out.carcass_temp_c = self.T_carcass
            out.gas_temp_c = self.T_gas
            out.inflation_pressure_pa = self.hot_pressure_pa()
            out.synthesized = self._synth()
            return out

        except Exception as e:
            self._restore_mu_scale()
            self._warn(f"thermal step failed ({type(e).__name__}); held previous "
                       f"temperatures for this sample.")
            # still return whatever the handling backend can give, with held temps
            try:
                out = self._ref.step(ws)
            except Exception:
                out = TireOutput(Fx=0.0, Fy=0.0, Fz=max(float(ws.Fz), 0.0))
            out.tread_temp_c = self.T_tread.copy()
            out.carcass_temp_c = self.T_carcass
            out.gas_temp_c = self.T_gas
            out.inflation_pressure_pa = self.hot_pressure_pa()
            out.synthesized = self._synth()
            return out

    # ---- mu(T) feedback plumbing ------------------------------------------ #
    def _apply_mu_scale(self, scale: float):
        """
        Temporarily scale the embedded Pacejka's peak-grip lambda by mu(T). We touch
        the LMUY scaling factor, which the MF5.2 lateral peak multiplies through, so
        a hot/cold tyre makes proportionally less/more grip. Restored after the step.
        """
        if not self.params.enable_mu_feedback:
            return
        try:
            lat = self._ref.lateral
            self._mu_saved = float(lat.scaling.get("LMUY", 1.0))
            lat.scaling = {**lat.scaling, "LMUY": self._mu_saved * float(scale)}
        except Exception:
            self._mu_saved = None

    def _restore_mu_scale(self):
        if not self.params.enable_mu_feedback:
            return
        try:
            if getattr(self, "_mu_saved", None) is not None:
                self._ref.lateral.scaling = {**self._ref.lateral.scaling,
                                             "LMUY": self._mu_saved}
                self._mu_saved = None
        except Exception:
            pass

    def _band_weights(self, ws: WheelState) -> np.ndarray:
        """
        Distribute generated heat across tread bands. Camber and lateral slip load
        one shoulder more; we bias the outer band under positive load/slip so the
        across-width temperature SPREAD is represented (the thing a thermal camera
        shows). Pure shape — magnitudes are part of the uncalibrated parameter set.
        """
        n = len(self.T_tread)
        if n == 1:
            return np.array([1.0])
        # linear bias from slip+camber magnitude, normalised to sum 1
        a = float(ws.alpha); g = float(ws.gamma)
        if not math.isfinite(a):
            a = 0.0
        if not math.isfinite(g):
            g = 0.0
        bias = float(abs(math.tan(max(min(a, 1.5), -1.5))) + 0.5 * abs(g))
        bias = min(bias, 0.6)
        ramp = np.linspace(1.0 - bias, 1.0 + bias, n)
        ramp = np.clip(ramp, 0.05, None)
        return ramp / ramp.sum()


# --------------------------------------------------------------------------- #
#  Standalone warm-up simulation — the thing the UI / a test calls
# --------------------------------------------------------------------------- #
@dataclass
class ThermalRun:
    """Result of a constant-condition warm-up/soak simulation."""
    t: np.ndarray                       # s
    tread_c: np.ndarray                 # (n, n_bands)
    carcass_c: np.ndarray               # (n,)
    gas_c: np.ndarray                   # (n,)
    pressure_pa: np.ndarray             # (n,)
    Fy: np.ndarray                      # (n,)
    mu_scale: np.ndarray                # (n,) grip multiplier from mu(T) (1.0 if off)
    calibrated: bool
    status: str
    warnings: list = field(default_factory=list)

    def tread_mean_c(self) -> np.ndarray:
        return self.tread_c.mean(axis=1)


def simulate_warmup(model: Optional[ThermalTireModel] = None,
                    *,
                    alpha_deg: float = 3.0,
                    Fz: float = 1100.0,
                    v_x: float = 18.0,
                    gamma_deg: float = 1.0,
                    kappa: float = 0.0,
                    ambient_c: float = 25.0,
                    track_c: float = 32.0,
                    duration_s: float = 120.0,
                    dt: float = 5.0e-3) -> ThermalRun:
    """
    Drive a ThermalTireModel at a CONSTANT operating point and record the warm-up.
    This is the honest demonstration of the channel: hand it a steady cornering
    condition and watch the rubber climb from ambient to a thermal plateau, the gas
    pressure rise with it, and (if mu feedback is on) the grip track temperature.

    Defaults are a representative steady mid-corner. Returns a ThermalRun whose
    temperatures are flagged uncalibrated unless the model's params say otherwise.
    """
    m = model or ThermalTireModel()
    m.reset(WheelState(ambient_temp_c=ambient_c, track_temp_c=track_c))

    n = int(max(duration_s / dt, 1))
    t = np.arange(n) * dt
    n_bands = len(m.T_tread)
    tread = np.zeros((n, n_bands))
    carc = np.zeros(n)
    gas = np.zeros(n)
    press = np.zeros(n)
    fy = np.zeros(n)
    mus = np.zeros(n)

    ws = WheelState(alpha=math.radians(alpha_deg), Fz=Fz, v_x=v_x,
                    gamma=math.radians(gamma_deg), kappa=kappa,
                    ambient_temp_c=ambient_c, track_temp_c=track_c, dt=dt)
    for i in range(n):
        out = m.step(ws)
        tread[i] = out.tread_temp_c
        carc[i] = out.carcass_temp_c
        gas[i] = out.gas_temp_c
        press[i] = out.inflation_pressure_pa
        fy[i] = out.Fy
        mus[i] = m._mu_scale_last

    prov = m.provenance()
    return ThermalRun(t=t, tread_c=tread, carcass_c=carc, gas_c=gas,
                      pressure_pa=press, Fy=fy, mu_scale=mus,
                      calibrated=prov.is_calibrated, status=prov.status(),
                      warnings=m.warnings())


def psi(pa: float) -> float:
    """Pa (gauge) -> psi, for UI display."""
    return float(pa) / 6894.757
