# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Transient per-cell battery-pack thermal model — the answer to "which cell cooks
first, and where do I put the fan?"

WHY THIS MODULE EXISTS
----------------------
The EV layer (`ev_powertrain.py`) already turns a virtual lap into an ENERGY
number: net kWh per lap, regen recovered, laps-until-empty. That is the right
abstraction for sizing a pack and choosing an architecture. But it collapses the
whole pack into one scalar and the whole lap into one integral, and so it cannot
answer the question that actually melts an FSAE-EV car's day at competition:

    During the endurance run the car pulls hundreds of amps in bursts on every
    corner exit. Some lithium-ion cells sit in dead air behind a bus-bar and
    others sit in the fan's draught. WHICH cells climb toward their temperature
    limit first, and where should the cooling fan(s) go so none of them do?

That is a *transient, spatial* question. It needs:
  1. the lap's CURRENT vs TIME (not energy vs distance) — the I that drives I²R,
  2. a pack laid out as a GRID of cells with real positions,
  3. a per-cell energy balance: Joule self-heating, cell↔cell conduction, and
     cell↔coolant convection whose strength VARIES with position (the airflow
     map), so moving a fan actually changes the answer,
  4. an honest read-out of the hottest cells and when they breach a limit.

This module does exactly that and nothing it can't defend. It is the pack
analogue of `tire_thermal.py`: a lumped-capacitance network with explicit,
documented conductances, the SAME calibration/provenance contract (uncalibrated
→ every thermal output is flagged `synthesized`), and the SAME never-raise
discipline as the rest of KinematiK. It WRAPS the EV lap sim at the established
seam — it reads a `LapResult`/`EVRunResult` and never reaches into the solver.

HONEST SCOPE (the same contract as the rest of the tool)
--------------------------------------------------------
- The current trace is derived from the QSS tractive-power solution: P = F·v at
  the wheels, taken back through the drivetrain efficiency to the pack, divided
  by a pack-voltage model (nominal V with a simple OCV-vs-SOC sag). It is a
  planning-grade current history, not a measured logger trace. Flagged as such.
- Heat generation per cell is ohmic, I_cell²·R_internal, plus an optional
  entropic term (off by default — it needs cell datasheet dU/dT). R_internal
  rising with temperature is modelled with a small linear tempco. No
  electrochemical (P2D/SPM) model is claimed.
- The airflow map is a geometric coefficient field h(x,y) seeded from fan
  position(s), throw and a wake/shadow falloff. It captures "cells in the
  draught cool, cells in dead air don't" — the effect that makes fan PLACEMENT
  matter — without pretending to be CFD. A real duct map can be dropped in.
- Absolute temperatures are only trustworthy once `calibrated=True` (every node
  parameter from rig data for the actual cell + measured airflow). Until then
  the module is for RANKING layouts and finding the hot corner, which is robust
  to the absolute scale — and it says so, loudly, in every result.

Never raises. Every entry point returns a result object carrying `.warnings`
and a provenance flag, mirroring `EVRunResult` and `ThermalRun`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# These imports are type/parameter references only; the driver accepts duck-typed
# lap results so the module also loads standalone (engine-test style) without the
# package __init__ pulling in heavy optional deps.
try:  # pragma: no cover - import shape varies by load path
    from .lapsim import LapResult, LapSimParams
    from .ev_powertrain import EVParams, EVRunResult
except Exception:  # pragma: no cover
    LapResult = object        # type: ignore
    LapSimParams = object     # type: ignore
    EVParams = object         # type: ignore
    EVRunResult = object      # type: ignore


# --------------------------------------------------------------------------- #
#  Cell electro-thermal parameters
# --------------------------------------------------------------------------- #
@dataclass
class CellParams:
    """
    One lithium-ion cell's electro-thermal lump. Defaults are REPRESENTATIVE of a
    21700 NMC cell of the kind an FSAE-EV pack is built from — chosen so the
    self-heating curve has a sensible shape and time constant, NOT measured on any
    specific cell.

    `calibrated` is the single most important field, exactly as in `tire_thermal`:
    leave it False and every temperature this model emits is flagged `synthesized`.
    Set it True ONLY when every parameter below came from datasheet/rig data for
    the actual cell — that is what turns a predicted cell temperature from a
    physically-shaped guess into a number you can quote.
    """
    # --- thermal lump ---
    mass_kg: float = 0.070              # single 21700 cell mass, kg
    cp: float = 1000.0                  # J/(kg·K), effective cell heat capacity
    # --- electrical (heat source) ---
    r_internal_ohm: float = 0.022       # DC internal resistance per cell, ohm (~22 mOhm)
    r_tempco_per_c: float = -0.0010     # fractional dR/dT; warm cell ~slightly lower R
    nominal_v: float = 3.6              # nominal cell voltage, V
    # --- geometry for convection / conduction ---
    surface_area_m2: float = 0.0042     # convective surface of one 21700 can, m²
    contact_area_m2: float = 8.0e-5     # cell↔cell side-contact area for conduction, m²
    pitch_mm: float = 22.0              # centre-to-centre spacing in the grid, mm
    # --- limits (the thing we are trying not to hit) ---
    temp_limit_c: float = 60.0          # cell temperature at which we derate/abort
    temp_warn_c: float = 50.0           # softer flag before the hard limit
    # --- optional entropic heat (needs datasheet dU/dT) ---
    enable_entropic: bool = False
    dudt_v_per_k: float = 0.0           # entropic coefficient; 0 unless calibrated
    # --- provenance ---
    calibrated: bool = False
    fitted_to: str = ""

    def r_at(self, t_c: float) -> float:
        """Internal resistance at temperature t_c (clamped positive)."""
        r = self.r_internal_ohm * (1.0 + self.r_tempco_per_c * (float(t_c) - 25.0))
        return float(max(r, 1e-4))


def default_cell_params() -> CellParams:
    """Representative, UNCALIBRATED 21700-NMC parameter set (calibrated=False)."""
    return CellParams()


# --------------------------------------------------------------------------- #
#  Pack layout: a grid of cells with positions and a series/parallel topology
# --------------------------------------------------------------------------- #
@dataclass
class PackLayout:
    """
    The pack as a physical GRID of cells. `rows × cols` cells sit on a regular
    pitch so every cell has an (x, y) position in millimetres — that position is
    what lets airflow (and therefore fan placement) matter cell-by-cell.

    Electrically the pack is `series` groups of `parallel` cells (sNpP). Pack
    current splits evenly across the `parallel` cells of a group, and the same
    series current flows through every group — so the PER-CELL current that does
    the I²R heating is pack_current / parallel. The grid shape (rows, cols) is a
    PACKAGING choice independent of the electrical sNpP; by default we lay the
    n_cells = series*parallel cells into the grid row-major.
    """
    rows: int = 6
    cols: int = 14
    series: int = 84                    # series groups in the WHOLE pack (sets voltage)
    parallel: int = 6                   # cells per parallel group (splits pack current)
    cell: CellParams = field(default_factory=default_cell_params)
    ambient_c: float = 30.0             # cooling-air inlet temperature, °C (hot tent day)
    # The rows×cols grid is the MODULE we visualise thermally; the whole pack may
    # contain several such modules in series. `series` is the full-pack series
    # count (it sets pack voltage and therefore the current the lap demands);
    # `parallel` is how many cells share that current. Per-cell current is always
    # pack_current / parallel, independent of how many cells we draw in the grid.

    def __post_init__(self):
        self.rows = max(int(self.rows), 1)
        self.cols = max(int(self.cols), 1)
        self.series = max(int(self.series), 1)
        self.parallel = max(int(self.parallel), 1)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def pack_nominal_v(self) -> float:
        return self.series * self.cell.nominal_v

    def positions_mm(self) -> np.ndarray:
        """(n_cells, 2) array of cell centre positions in mm, row-major."""
        p = self.cell.pitch_mm
        xs = (np.arange(self.cols) + 0.5) * p
        ys = (np.arange(self.rows) + 0.5) * p
        gx, gy = np.meshgrid(xs, ys)            # (rows, cols)
        return np.column_stack([gx.ravel(), gy.ravel()])

    def extent_mm(self) -> tuple[float, float]:
        """(width, height) of the cell field in mm."""
        return self.cols * self.cell.pitch_mm, self.rows * self.cell.pitch_mm


# --------------------------------------------------------------------------- #
#  Fans + airflow map: the lever the user optimises
# --------------------------------------------------------------------------- #
@dataclass
class Fan:
    """
    A cooling fan as a source of convective cooling at a position in the pack
    plane. `x_mm, y_mm` is where the fan blows onto the cell field; `cfm` sets how
    much air it moves (→ peak convection coefficient); `throw_mm` sets how far the
    draught reaches before it decays (cells beyond the throw sit in progressively
    deader air). These are the geometric knobs a team actually has: where the fan
    bolts and how big it is.
    """
    x_mm: float
    y_mm: float
    cfm: float = 120.0                  # airflow, cubic feet per minute
    throw_mm: float = 90.0             # 1/e decay distance of the draught, mm
    label: str = ""


@dataclass
class AirflowParams:
    """
    Turns fan geometry into a per-cell convective coefficient field h(x,y).

    Model: each cell sees baseline still-air convection `h_floor`, PLUS a
    contribution from every fan that falls off with distance from the fan axis
    (Gaussian-ish wake, 1/e at the fan's throw) and scales with fan airflow. The
    field is normalised so a representative `cfm` at the fan face gives a sensible
    forced-convection coefficient `h_peak`. This is a geometric airflow PROXY, not
    CFD — it captures the only thing that matters for placement decisions: cells
    near a fan in its throw are well-cooled, cells in dead air are not.
    """
    h_floor: float = 8.0                # W/(m²·K) still-air natural convection
    h_peak: float = 75.0               # W/(m²·K) directly in a ~120 cfm draught
    ref_cfm: float = 120.0             # cfm that produces h_peak at the fan face
    # cross-cell shadowing: a cell downstream of many cooled cells gets warmer air
    enable_wake_warming: bool = True
    wake_warm_per_cell_c: float = 0.15  # air temp rise per upstream cell it passed, °C
    calibrated: bool = False            # set True only with measured airflow map

    def h_field(self, positions_mm: np.ndarray, fans: Sequence[Fan]) -> np.ndarray:
        """Per-cell convective coefficient h (W/m²·K) from all fans. Never raises."""
        try:
            pos = np.asarray(positions_mm, float)
            n = pos.shape[0]
            h = np.full(n, float(self.h_floor))
            for f in fans or []:
                try:
                    dx = pos[:, 0] - float(f.x_mm)
                    dy = pos[:, 1] - float(f.y_mm)
                    d = np.sqrt(dx * dx + dy * dy)
                    throw = max(float(f.throw_mm), 1e-3)
                    # Gaussian draught: 1/e at one throw length
                    shape = np.exp(-(d * d) / (throw * throw))
                    scale = max(float(f.cfm), 0.0) / max(self.ref_cfm, 1e-6)
                    h += (self.h_peak - self.h_floor) * scale * shape
                except Exception:
                    continue
            return np.clip(np.nan_to_num(h, nan=self.h_floor),
                           self.h_floor, None)
        except Exception:
            return np.full(int(np.asarray(positions_mm).shape[0]), self.h_floor)


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class PackThermalResult:
    """Outcome of a transient pack-thermal run over one virtual lap (or many)."""
    ok: bool
    rows: int
    cols: int
    time_s: np.ndarray                  # (n_t,) time grid, s
    temp_history_c: np.ndarray          # (n_t, n_cells) per-cell temperature, °C
    final_temp_c: np.ndarray            # (n_cells,) temperature at the end, °C
    peak_temp_c: np.ndarray             # (n_cells,) per-cell max over the run, °C
    time_to_warn_s: np.ndarray          # (n_cells,) first time cell passed temp_warn (nan if never)
    time_to_limit_s: np.ndarray         # (n_cells,) first time cell passed temp_limit (nan if never)
    pack_current_a: np.ndarray          # (n_t,) pack current trace used, A (drive +, regen -)
    hottest_cell_index: int             # row-major index of the worst cell
    hottest_cell_rc: tuple              # (row, col) of the worst cell
    hottest_peak_c: float               # peak temperature of the worst cell, °C
    any_cell_breached_limit: bool       # did ANY cell exceed temp_limit?
    breach_count: int                   # how many cells exceeded temp_limit
    synthesized: bool                   # True ⇒ outputs are physically-shaped, not measured
    provenance: str                     # one-line honesty flag
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def failed(rows: int, cols: int, warnings: list[str]) -> "PackThermalResult":
        n = max(rows * cols, 1)
        z1 = np.full(1, float("nan"))
        zc = np.full(n, float("nan"))
        return PackThermalResult(
            ok=False, rows=rows, cols=cols,
            time_s=z1.copy(), temp_history_c=np.full((1, n), float("nan")),
            final_temp_c=zc.copy(), peak_temp_c=zc.copy(),
            time_to_warn_s=zc.copy(), time_to_limit_s=zc.copy(),
            pack_current_a=z1.copy(), hottest_cell_index=-1,
            hottest_cell_rc=(-1, -1), hottest_peak_c=float("nan"),
            any_cell_breached_limit=False, breach_count=0,
            synthesized=True,
            provenance="run failed; no thermal field produced",
            warnings=list(warnings),
        )

    def temp_grid_c(self) -> np.ndarray:
        """Final temperatures reshaped to the (rows, cols) physical grid."""
        try:
            return self.final_temp_c.reshape(self.rows, self.cols)
        except Exception:
            return self.final_temp_c

    def peak_grid_c(self) -> np.ndarray:
        """Peak temperatures reshaped to the (rows, cols) physical grid."""
        try:
            return self.peak_temp_c.reshape(self.rows, self.cols)
        except Exception:
            return self.peak_temp_c


# --------------------------------------------------------------------------- #
#  Current trace: turn a virtual lap into pack-current-vs-time
# --------------------------------------------------------------------------- #
def pack_current_trace(lap,
                       lap_params,
                       pack_nominal_v: float,
                       inverter_motor_eff: float = 0.90,
                       regen_eff: float = 0.55,
                       regen_max_g: float = 0.35,
                       regen_enabled: bool = True,
                       sag_ohm: float = 0.0,
                       warn=lambda m: None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (time_s, pack_current_a) from a QSS lap trace. This is the transient
    bridge the EV energy integral never needed: instead of accumulating one kWh
    number, we keep the full current-vs-time history that drives I²R heating.

    Physics, per distance step (matching `_energy_from_trace` in ev_powertrain so
    the current integrates back to the same energy):
      - dt = ds / v_avg              (lap sim is distance-indexed; recover time)
      - accel: F = m·a + F_drag + F_roll ; P_wheel = F·v ; P_batt = P_wheel/eff
      - brake: regen captures up to regen_max_g of decel at regen_eff (negative I)
      - I = P_batt / V_pack          (optionally with an IR sag on V_pack)

    Returns time and current arrays of equal length (one sample per trace point).
    Never raises; non-finite points carry zero current rather than poisoning it.
    """
    p = lap_params
    g = float(getattr(p, "g", 9.81))
    mass = float(getattr(p, "mass", 280.0))
    rho = float(getattr(p, "rho", 1.225))
    cd_a = float(getattr(p, "cd_a", 1.2))
    rolling_g = float(getattr(p, "rolling_g", 0.015))
    v_min = float(getattr(p, "V_MIN", 0.5))
    eff = max(float(inverter_motor_eff), 1e-3)
    v_pack = max(float(pack_nominal_v), 1.0)

    try:
        v = np.asarray(lap.speed, float)
        d = np.asarray(lap.distance, float)
        lg = np.asarray(lap.long_g, float)
        n = v.size
        if n < 2:
            warn("Lap trace too short to build a current history.")
            return np.zeros(1), np.zeros(1)

        t = np.zeros(n)
        cur = np.zeros(n)
        for i in range(1, n):
            ds = d[i] - d[i - 1]
            if not (np.isfinite(ds) and ds > 0):
                t[i] = t[i - 1]
                continue
            vi = max(v[i], v_min)
            v_avg = max(0.5 * (v[i] + v[i - 1]), v_min)
            dt = ds / v_avg
            t[i] = t[i - 1] + dt
            a = (lg[i] if np.isfinite(lg[i]) else 0.0) * g
            f_drag = 0.5 * rho * cd_a * vi * vi
            f_roll = rolling_g * mass * g
            if a > 0:                                   # drawing
                f_trac = mass * a + f_drag + f_roll
                p_wheel = max(f_trac, 0.0) * vi
                p_batt = p_wheel / eff
                cur[i] = p_batt / v_pack
            elif a < 0 and regen_enabled:               # regen (negative current)
                a_regen = min(-a, regen_max_g * g)
                f_regen = mass * a_regen
                p_regen = f_regen * vi * regen_eff
                cur[i] = -(p_regen / v_pack)
            else:
                cur[i] = 0.0
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        return t, cur
    except Exception:
        warn("Current-trace construction failed; using zero current.")
        return np.zeros(1), np.zeros(1)


# --------------------------------------------------------------------------- #
#  The transient pack thermal solver
# --------------------------------------------------------------------------- #
class PackThermalModel:
    """
    Explicit transient lumped-capacitance network over a grid of cells.

    State: one temperature per cell. Per time step, each cell's energy balance is
        C_cell · dT/dt = Q_joule + Q_conduction(neighbours) + Q_convection(air)
    where
        Q_joule       = I_cell² · R_internal(T)   (+ optional entropic term)
        Q_conduction  = Σ k_cc · (T_neighbour − T_cell)   over 4-neighbours
        Q_convection  = h_cell · A_surface · (T_air_cell − T_cell)
    and h_cell comes from the airflow map (fan-position dependent) — which is the
    whole point: move the fan, change h_cell, change which cell cooks first.

    The integrator is forward-Euler with an automatically sub-divided step chosen
    to stay below the network's stability limit, so a coarse lap-trace dt never
    blows the solution up. Never raises.
    """

    def __init__(self,
                 layout: Optional[PackLayout] = None,
                 fans: Optional[Sequence[Fan]] = None,
                 airflow: Optional[AirflowParams] = None,
                 k_cell_cell: float = 0.35):
        self.layout = layout or PackLayout()
        self.fans = list(fans or [])
        self.airflow = airflow or AirflowParams()
        # cell↔cell conduction conductance (W/K) along the grid; bus-bar + can
        # contact + potting. A single representative number, documented & flagged.
        self.k_cc = float(k_cell_cell)
        self.warnings: list[str] = []

    def _warn(self, m: str):
        if m and m not in self.warnings:
            self.warnings.append(m)

    # ---- neighbour index list for the grid (4-connected) ---------------- #
    def _neighbours(self) -> list[tuple[int, int]]:
        R, C = self.layout.rows, self.layout.cols
        pairs = []
        for r in range(R):
            for c in range(C):
                i = r * C + c
                if c + 1 < C:
                    pairs.append((i, i + 1))          # right
                if r + 1 < R:
                    pairs.append((i, i + C))          # down
        return pairs

    # ---- the run -------------------------------------------------------- #
    def simulate(self,
                 time_s: np.ndarray,
                 pack_current_a: np.ndarray,
                 init_temp_c: Optional[float] = None,
                 n_laps: int = 1) -> PackThermalResult:
        """
        Integrate the pack temperature field over the supplied current history,
        optionally repeating it `n_laps` times (an endurance stint is the same lap
        current pattern back-to-back). Returns a full PackThermalResult.
        """
        lay = self.layout
        cell = lay.cell
        R, C = lay.rows, lay.cols
        n = lay.n_cells
        try:
            t_in = np.asarray(time_s, float).ravel()
            i_in = np.asarray(pack_current_a, float).ravel()
            if t_in.size < 2 or i_in.size < 2:
                return PackThermalResult.failed(
                    R, C, self.warnings + ["Current history too short to integrate."])
            m = min(t_in.size, i_in.size)
            t_in, i_in = t_in[:m], i_in[:m]
            # enforce a monotonic, finite time base
            t_in = np.nan_to_num(t_in, nan=0.0)
            i_in = np.nan_to_num(i_in, nan=0.0, posinf=0.0, neginf=0.0)

            n_laps = max(int(n_laps), 1)
            # stitch laps end-to-end (current pattern repeats; time accumulates)
            if n_laps > 1:
                lap_T = t_in[-1] - t_in[0]
                ts = [t_in]
                cs = [i_in]
                for k in range(1, n_laps):
                    ts.append(t_in + (t_in[-1] - t_in[0]) * 0 + ts[-1][-1] +
                              (t_in[1] - t_in[0]))
                    cs.append(i_in)
                t_in = np.concatenate(ts)
                i_in = np.concatenate(cs)

            positions = lay.positions_mm()
            h_cell = self.airflow.h_field(positions, self.fans)        # (n,)
            if self.airflow.enable_wake_warming:
                air_offset = self._wake_air_offset(h_cell)             # (n,)
            else:
                air_offset = np.zeros(n)
            if not (self.airflow.calibrated and cell.calibrated):
                synthesized = True
            else:
                synthesized = False

            # per-cell capacitance C = m·cp (J/K)
            C_cell = max(cell.mass_kg * cell.cp, 1e-6)
            A_surf = max(cell.surface_area_m2, 1e-9)
            neigh = self._neighbours()

            T0 = float(init_temp_c) if init_temp_c is not None else lay.ambient_c
            T = np.full(n, T0, float)
            T_air = lay.ambient_c + air_offset

            n_t = t_in.size
            temp_hist = np.empty((n_t, n), float)
            temp_hist[0] = T
            peak = T.copy()
            t_warn = np.full(n, np.nan)
            t_limit = np.full(n, np.nan)

            # per-cell current = pack current / parallel strings
            par = max(lay.parallel, 1)

            # forward-Euler with adaptive sub-steps for stability
            # stability needs dt < C / (Σ conductances out of a node)
            g_conv_max = float(np.max(h_cell)) * A_surf
            g_cond_node = 4.0 * self.k_cc
            g_out_max = g_conv_max + g_cond_node
            dt_stable = 0.5 * C_cell / max(g_out_max, 1e-9)

            for step in range(1, n_t):
                dt_macro = max(t_in[step] - t_in[step - 1], 0.0)
                I_pack = i_in[step]
                I_cell = I_pack / par
                if dt_macro <= 0.0:
                    temp_hist[step] = T
                    continue
                n_sub = max(int(math.ceil(dt_macro / dt_stable)), 1)
                # guard against pathological explosion of sub-steps
                if n_sub > 5000:
                    n_sub = 5000
                    self._warn("Thermal sub-stepping capped at 5000/step; "
                               "results near that step are coarse.")
                dt = dt_macro / n_sub
                for _ in range(n_sub):
                    # Joule heat per cell (W). entropic optional.
                    r = cell.r_at(float(np.mean(T)))  # mild: one R for the field
                    q = (I_cell * I_cell) * r * np.ones(n)
                    if cell.enable_entropic and cell.dudt_v_per_k != 0.0:
                        # entropic heat = I·T·dU/dT (sign depends on charge dir);
                        # small, and only meaningful when calibrated.
                        q = q + I_cell * (T + 273.15) * cell.dudt_v_per_k
                    # convection to (locally warmed) air
                    q_conv = h_cell * A_surf * (T_air - T)
                    # conduction across the grid
                    q_cond = np.zeros(n)
                    for (a, b) in neigh:
                        flow = self.k_cc * (T[b] - T[a])
                        q_cond[a] += flow
                        q_cond[b] -= flow
                    dT = (q + q_conv + q_cond) * (dt / C_cell)
                    T = T + dT
                # record at macro step
                T = np.nan_to_num(T, nan=T0)
                temp_hist[step] = T
                peak = np.maximum(peak, T)
                now = t_in[step]
                newly_warn = (T >= cell.temp_warn_c) & ~np.isfinite(t_warn)
                t_warn[newly_warn] = now
                newly_lim = (T >= cell.temp_limit_c) & ~np.isfinite(t_limit)
                t_limit[newly_lim] = now

            hottest = int(np.argmax(peak))
            breach_mask = peak >= cell.temp_limit_c
            prov = self._provenance(synthesized)

            return PackThermalResult(
                ok=True, rows=R, cols=C,
                time_s=t_in, temp_history_c=temp_hist,
                final_temp_c=T, peak_temp_c=peak,
                time_to_warn_s=t_warn, time_to_limit_s=t_limit,
                pack_current_a=i_in,
                hottest_cell_index=hottest,
                hottest_cell_rc=(hottest // C, hottest % C),
                hottest_peak_c=float(peak[hottest]),
                any_cell_breached_limit=bool(breach_mask.any()),
                breach_count=int(breach_mask.sum()),
                synthesized=synthesized, provenance=prov,
                warnings=list(self.warnings),
            )
        except Exception as exc:
            return PackThermalResult.failed(
                R, C, self.warnings + [f"pack thermal run crashed: {exc!r}"])

    # ---- airflow wake warming (downstream cells get pre-warmed air) ----- #
    def _wake_air_offset(self, h_cell: np.ndarray) -> np.ndarray:
        """
        Cells further from the fan tend to sit downstream of cells the air already
        passed, so their cooling air is slightly warmer. Approximate this as a
        per-cell air-temperature offset that grows with how poorly-cooled (low h)
        a cell is relative to the best-cooled cell.

        Crucially this is a SECOND-ORDER correction: it must never be large enough
        to make adding a fan worse than no fan. We therefore bound the total
        offset to a few °C and tie it to the cooling deficit, so a well-cooled
        cell (high h) sees ~0 offset and a dead-air cell sees at most the cap.
        """
        try:
            h = np.asarray(h_cell, float)
            hmax = float(np.max(h)) if h.size else 1.0
            if hmax <= self.airflow.h_floor + 1e-9:
                return np.zeros(h.size)        # no fan ⇒ no wake structure
            deficit = np.clip((hmax - h) / max(hmax, 1e-6), 0.0, 1.0)
            cap_c = max(float(self.airflow.wake_warm_per_cell_c), 0.0) * 6.0
            return deficit * cap_c
        except Exception:
            return np.zeros(self.layout.n_cells)

    def _provenance(self, synthesized: bool) -> str:
        if synthesized:
            return ("SYNTHESIZED: per-cell temperatures are physically-shaped "
                    "estimates from an uncalibrated lumped network + geometric "
                    "airflow proxy. Trustworthy for RANKING layouts and locating "
                    "the hot corner; absolute °C needs calibrated=True on both the "
                    "cell and the airflow map.")
        return ("CALIBRATED: cell thermal/electrical parameters and the airflow "
                "map are fitted to measured data; absolute temperatures are "
                "quotable within the fit's validity.")


# --------------------------------------------------------------------------- #
#  Driver: virtual lap → per-cell hot map (the headline entry point)
# --------------------------------------------------------------------------- #
def simulate_pack_thermal(lap,
                          lap_params,
                          layout: Optional[PackLayout] = None,
                          fans: Optional[Sequence[Fan]] = None,
                          airflow: Optional[AirflowParams] = None,
                          ev: Optional["EVParams"] = None,
                          k_cell_cell: float = 0.35,
                          init_temp_c: Optional[float] = None,
                          n_laps: Optional[int] = None) -> PackThermalResult:
    """
    The headline: take a virtual lap (a QSS `LapResult`) and predict, cell by cell,
    how hot the pack gets and WHICH cells reach their limit first under the given
    fan layout.

    Steps:
      1. derive the pack current-vs-time from the lap (pack_current_trace),
      2. integrate the per-cell transient thermal network over `n_laps` of it,
      3. return the per-cell peak/limit map + the worst cell.

    `lap` only needs `.speed`, `.distance`, `.long_g` (duck-typed), so this works
    with a real LapResult or any compatible trace. Never raises.
    """
    layout = layout or PackLayout()
    fans = list(fans or [])
    airflow = airflow or AirflowParams()

    # regen / efficiency: prefer the EVParams the rest of the EV layer uses
    eff = getattr(ev, "inverter_motor_eff", 0.90) if ev is not None else 0.90
    regen_eff = getattr(ev, "regen_eff", 0.55) if ev is not None else 0.55
    regen_max_g = getattr(ev, "regen_max_g", 0.35) if ev is not None else 0.35
    regen_on = getattr(ev, "regen_enabled", True) if ev is not None else True

    model = PackThermalModel(layout=layout, fans=fans, airflow=airflow,
                             k_cell_cell=k_cell_cell)

    laps = n_laps
    if laps is None:
        laps = int(getattr(lap, "meta", {}).get("laps", 1)) if hasattr(lap, "meta") else 1
        laps = max(laps, 1)

    t, cur = pack_current_trace(
        lap, lap_params, pack_nominal_v=layout.pack_nominal_v,
        inverter_motor_eff=eff, regen_eff=regen_eff, regen_max_g=regen_max_g,
        regen_enabled=regen_on, warn=model._warn)

    return model.simulate(t, cur, init_temp_c=init_temp_c, n_laps=laps)


# --------------------------------------------------------------------------- #
#  Fan-placement optimisation: script candidate layouts, rank on peak temp
# --------------------------------------------------------------------------- #
@dataclass
class FanPlacementCandidate:
    """One trial fan layout and the peak cell temperature it produced."""
    fans: list
    hottest_peak_c: float
    hottest_cell_rc: tuple
    breach_count: int
    result: PackThermalResult


@dataclass
class FanPlacementStudy:
    """Ranked fan-placement candidates for one car/lap/pack."""
    candidates: list                     # sorted best→worst on hottest_peak_c
    synthesized: bool
    provenance: str
    warnings: list = field(default_factory=list)

    @property
    def best(self) -> Optional[FanPlacementCandidate]:
        return self.candidates[0] if self.candidates else None

    def summary(self) -> str:
        if not self.candidates:
            return "No fan-placement candidates evaluated."
        lines = ["Fan-placement study (lower peak cell temp is better):"]
        for rank, c in enumerate(self.candidates, 1):
            where = ", ".join(
                f"({f.x_mm:.0f},{f.y_mm:.0f})mm" for f in c.fans) or "no fan"
            lines.append(
                f"  {rank}. peak {c.hottest_peak_c:6.1f}°C  "
                f"hot cell r{c.hottest_cell_rc[0]}c{c.hottest_cell_rc[1]}  "
                f"breaches={c.breach_count}  fans=[{where}]")
        if self.synthesized:
            lines.append("  NOTE: temperatures are SYNTHESIZED (uncalibrated); "
                         "the RANKING is the deliverable, not absolute °C.")
        return "\n".join(lines)


def optimize_fan_placement(lap,
                           lap_params,
                           candidate_fan_sets: Sequence[Sequence[Fan]],
                           layout: Optional[PackLayout] = None,
                           airflow: Optional[AirflowParams] = None,
                           ev: Optional["EVParams"] = None,
                           k_cell_cell: float = 0.35,
                           n_laps: Optional[int] = None) -> FanPlacementStudy:
    """
    Script a set of candidate fan layouts and rank them by the peak cell
    temperature they produce on the SAME virtual lap — the "optimise cooling fan
    placement in software" workflow. Each entry of `candidate_fan_sets` is a list
    of Fans (zero, one, or several). Returns a ranked study; never raises.

    Typical use: sweep a single fan across a grid of mounting positions, or
    compare "one big fan centre" vs "two small fans at the ends".
    """
    layout = layout or PackLayout()
    airflow = airflow or AirflowParams()
    cands: list[FanPlacementCandidate] = []
    warnings: list[str] = []
    synthesized = True
    prov = ""
    for fan_set in candidate_fan_sets or [[]]:
        res = simulate_pack_thermal(
            lap, lap_params, layout=layout, fans=list(fan_set),
            airflow=airflow, ev=ev, k_cell_cell=k_cell_cell, n_laps=n_laps)
        synthesized = res.synthesized
        prov = res.provenance
        for w in res.warnings:
            if w not in warnings:
                warnings.append(w)
        peak = res.hottest_peak_c if np.isfinite(res.hottest_peak_c) else float("inf")
        cands.append(FanPlacementCandidate(
            fans=list(fan_set), hottest_peak_c=peak,
            hottest_cell_rc=res.hottest_cell_rc,
            breach_count=res.breach_count, result=res))
    cands.sort(key=lambda c: (c.breach_count, c.hottest_peak_c))
    return FanPlacementStudy(candidates=cands, synthesized=synthesized,
                             provenance=prov, warnings=warnings)


# --------------------------------------------------------------------------- #
#  Convenience: a grid of single-fan candidate positions
# --------------------------------------------------------------------------- #
def fan_grid_candidates(layout: PackLayout,
                        nx: int = 3, ny: int = 2,
                        cfm: float = 120.0,
                        throw_mm: float = 90.0) -> list[list[Fan]]:
    """
    Build a default sweep: a single fan placed at each node of an nx×ny grid over
    the pack face, plus a no-fan baseline. Handy first pass for
    optimize_fan_placement.
    """
    w, h = layout.extent_mm()
    sets: list[list[Fan]] = [[]]  # baseline: no fan
    for iy in range(max(ny, 1)):
        for ix in range(max(nx, 1)):
            x = (ix + 0.5) * w / max(nx, 1)
            y = (iy + 0.5) * h / max(ny, 1)
            sets.append([Fan(x_mm=x, y_mm=y, cfm=cfm, throw_mm=throw_mm,
                             label=f"fan@({x:.0f},{y:.0f})")])
    return sets
