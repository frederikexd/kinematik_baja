# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
WORKED EXAMPLE: a conforming wrapper skeleton for a structural tire backend.

This is the analog of the STI-conforming wrapper a tire vendor ships for ADAMS/Car.
It shows EXACTLY how to bind a real FTire/CDTire co-sim call to KinematiK's
`StructuralTireModel` contract — frame/sign handling, the staggered-timing
discipline, and where each vendor output lands in `TireOutput`.

IMPORTANT — what this is and is NOT:
  * It is a TEMPLATE. The `_FakeStructuralBinding` below is a transparent stand-in
    that fabricates physically-shaped (but NOT validated) structural and thermal
    numbers, purely so you can run the wrapper end-to-end and see the data flow.
  * It is NOT a tire model. The fake binding is clearly labelled, and a backend
    built on it reports `is_calibrated=False` and names the fabricated channels.
    Do not ship results from the fake binding as if they were measured.
  * To make it real: replace `_FakeStructuralBinding` with the actual cosin/FMU
    call (see docs/tire_cosim_interface.md §7), keep the marshalling in
    `ExampleStructuralBackend.step` unchanged, and the channels you now fill from
    vendor physics drop out of `synthesized` automatically.

Run this file directly to see a step-steer co-sim driven by the example backend,
with the structural/thermal channels populated:

    python -m suspension.tire_cosim_ftire_example
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .tire_cosim import (WheelState, TireOutput, TireProvenance, TireFidelity,
                         _ExternalTireBackend)
from .tiremodel import PacejkaLateral, default_tire, relaxation_length


# --------------------------------------------------------------------------- #
#  A transparent stand-in for the vendor binding (REPLACE THIS)
# --------------------------------------------------------------------------- #
class _FakeStructuralBinding:
    """
    Stands in for the cosin co-sim callable / CDTire FMU. A real binding advances a
    flexible-carcass + thermal solver one step; this fakes physically-SHAPED outputs
    from the simple inputs so the wrapper is runnable. Every number it returns is a
    placeholder — the wrapper marks them all as synthesized.

    The real binding's `advance(...)` would take the marshalled wheel-centre state +
    3D road and return the vendor's force/moment/structural/thermal block. We keep
    the same call shape so swapping it in is a one-line change in the backend.
    """

    def __init__(self, lateral: PacejkaLateral):
        self.lateral = lateral
        # fabricated internal state, just enough to look stateful
        self._gas_temp_c = 25.0
        self._tread_temp_c = np.full(5, 25.0)   # 5 tread bands
        self._pressure_pa = 83_000.0            # ~12 psi cold

    def initialise(self, ws: Optional[WheelState]):
        amb = ws.ambient_temp_c if ws else 25.0
        self._gas_temp_c = amb
        self._tread_temp_c[:] = (ws.track_temp_c if ws else 30.0)
        self._pressure_pa = (ws.inflation_pressure_pa
                             if (ws and ws.inflation_pressure_pa) else 83_000.0)

    def advance(self, alpha, kappa, gamma, Fz, vx, omega, z_wheel,
                road_points, dt):
        """Fabricated one-step advance. REPLACE with the vendor co-sim call."""
        Fz = max(float(Fz), 0.0)
        # --- forces: use the real Pacejka so handling is at least sane ---
        fy = float(self.lateral.fy(alpha, max(Fz, 1.0), gamma)) if Fz > 1.0 else 0.0
        fy_max = float(self.lateral.peak_force(max(Fz, 1.0), abs(gamma))) if Fz > 1.0 else 0.0
        fx = float(np.clip(kappa, -1.0, 1.0)) * 1.05 * fy_max
        # --- FABRICATED structural channels (shape only, NOT validated) ---
        k_carcass = 200_000.0                       # pretend radial carcass rate, N/m
        deflection = Fz / k_carcass                 # m
        # crude enveloping bump if a road point lifts under the patch
        if road_points is not None and len(road_points):
            deflection += max(0.0, float(np.max(road_points[:, 2])))
        contact_len = 0.10 + 0.5 * deflection       # m, grows with deflection
        contact_wid = 0.18                          # m
        eff_radius = 0.228 - deflection             # m, ~9" wheel loaded radius
        # parabolic pressure bump, integrates to ~Fz (shape only)
        nx, ny = 12, 6
        xx = np.linspace(-1, 1, nx)[None, :]
        yy = np.linspace(-1, 1, ny)[:, None]
        shape = np.clip(1.0 - xx**2, 0, 1) * np.clip(1.0 - yy**2, 0, 1)
        area = max(contact_len * contact_wid, 1e-4)
        pressure = (Fz / area) * shape / max(shape.mean(), 1e-6)   # Pa
        # --- FABRICATED thermal channels (first-order heating toward a target) ---
        slip_power = abs(fy * math.tan(alpha)) + abs(fx * kappa)   # W-ish proxy
        target = 25.0 + 0.0015 * slip_power + 0.4 * (abs(vx))      # °C, made up
        a = min(dt / 8.0, 1.0)                                     # slow time const
        self._tread_temp_c += a * (target - self._tread_temp_c)
        self._gas_temp_c += a * (self._tread_temp_c.mean() - self._gas_temp_c)
        # ideal-gas-ish pressure build-up from gas temperature
        self._pressure_pa = 83_000.0 * (273.15 + self._gas_temp_c) / (273.15 + 25.0)
        return dict(
            Fx=fx, Fy=fy, Fz=Fz, Mz=0.0, Mx=0.0, My=0.0,
            carcass_deflection_m=deflection,
            contact_length_m=contact_len, contact_width_m=contact_wid,
            effective_radius_m=eff_radius, pressure_distribution=pressure,
            tread_temp_c=self._tread_temp_c.copy(),
            carcass_temp_c=float(self._tread_temp_c.mean() - 5.0),
            gas_temp_c=float(self._gas_temp_c),
            inflation_pressure_pa=float(self._pressure_pa),
        )


# --------------------------------------------------------------------------- #
#  The conforming wrapper
# --------------------------------------------------------------------------- #
class ExampleStructuralBackend(_ExternalTireBackend):
    """
    A conforming structural backend built on the fake binding. This is the file you
    copy to write a real FTire/CDTire wrapper: keep `step`'s marshalling and the
    `synthesized` bookkeeping, swap the binding.

    Because its channels come from the FAKE binding, it honestly reports
    is_calibrated=False and lists every fabricated channel in TireOutput.synthesized.
    A REAL binding fitted to your tire flips both: set the provenance file and remove
    channels from `_FABRICATED` as they become vendor-physics.
    """
    _vendor = "example-structural"
    _fidelity = TireFidelity.STRUCTURAL_THERMAL

    # channels the FAKE binding fabricates; a real binding empties this list as it
    # fills each one from validated vendor physics.
    _FABRICATED = ["carcass_deflection_m", "contact_length_m", "contact_width_m",
                   "pressure_distribution", "effective_radius_m",
                   "tread_temp_c", "carcass_temp_c", "gas_temp_c",
                   "inflation_pressure_pa"]

    def __init__(self, lateral: Optional[PacejkaLateral] = None,
                 parameter_file: str = "", fitted_to: str = ""):
        self._lateral = lateral or default_tire()
        # bind the (fake) vendor solver; a real wrapper passes the cosin/FMU handle
        binding = _FakeStructuralBinding(self._lateral)
        # bypass the _ExternalTireBackend stub-guard by supplying a binding
        super().__init__(parameter_file=parameter_file or "EXAMPLE-not-a-real-file",
                         binding=binding, fitted_to=fitted_to)

    def provenance(self) -> TireProvenance:
        return TireProvenance(
            backend=self._vendor, fidelity=self._fidelity,
            # FAKE binding => never calibrated, no matter what file string is set
            is_calibrated=False,
            parameter_file=self.parameter_file,
            notes="WORKED EXAMPLE on a fake binding. Structural/thermal channels are "
                  "fabricated placeholders (see TireOutput.synthesized). Replace the "
                  "binding with cosin FTire / CDTire FMU to make them real.")

    def reset(self, state: Optional[WheelState] = None) -> None:
        self._warnings = []
        try:
            self._binding.initialise(state)
        except Exception as e:
            self._warn(f"binding initialise failed ({type(e).__name__}).")

    def step(self, ws: WheelState) -> TireOutput:
        # ---- THE MARSHALLING (keep this when you swap in a real binding) ----
        try:
            r = self._binding.advance(
                alpha=float(ws.alpha), kappa=float(ws.kappa), gamma=float(ws.gamma),
                Fz=float(ws.Fz), vx=float(ws.v_x), omega=float(ws.omega),
                z_wheel=float(ws.z_wheel), road_points=ws.road_points,
                dt=max(float(ws.dt), 0.0))
        except Exception as e:
            self._warn(f"binding advance failed ({type(e).__name__}); zero force.")
            out = TireOutput(Fx=0.0, Fy=0.0, Fz=max(float(ws.Fz), 0.0))
            out.synthesized = list(self._FABRICATED)
            return out

        out = TireOutput(
            Fx=r.get("Fx", 0.0), Fy=r.get("Fy", 0.0), Fz=r.get("Fz", float(ws.Fz)),
            Mz=r.get("Mz", 0.0), Mx=r.get("Mx", 0.0), My=r.get("My", 0.0),
            carcass_deflection_m=r.get("carcass_deflection_m"),
            contact_length_m=r.get("contact_length_m"),
            contact_width_m=r.get("contact_width_m"),
            pressure_distribution=r.get("pressure_distribution"),
            effective_radius_m=r.get("effective_radius_m"),
            tread_temp_c=r.get("tread_temp_c"),
            carcass_temp_c=r.get("carcass_temp_c"),
            gas_temp_c=r.get("gas_temp_c"),
            inflation_pressure_pa=r.get("inflation_pressure_pa"),
        )
        # honesty bookkeeping: with the fake binding, all structural/thermal are
        # placeholders. A real binding sets _FABRICATED = [] (or a subset).
        out.synthesized = list(self._FABRICATED)
        return out


def example_backend_factory(lateral: Optional[PacejkaLateral] = None):
    """A factory for CosimCornerSet / run_cosim_maneuver."""
    return lambda: ExampleStructuralBackend(lateral=lateral)


# --------------------------------------------------------------------------- #
#  Runnable demo
# --------------------------------------------------------------------------- #
def _demo():
    from . import VehicleDynamics, VehicleParams
    from .tire_cosim_driver import run_cosim_maneuver

    veh = VehicleDynamics(VehicleParams(), tire=default_tire())
    out = run_cosim_maneuver(veh, kind="step_steer", steer_deg=4.0, t_end=1.5,
                             backend_factory=example_backend_factory())
    res, th = out["result"], out["tire_history"]
    print("backend:", out["backend_status"])
    if res is None or not res.ok:
        print("run failed:", out["warnings"]); return
    print(f"steps: {len(res.t)}   peak ay: {np.max(np.abs(res.ay)):.2f} g")
    print(f"channels flagged as placeholders: {th.absent_channels}")
    # show that structural/thermal came through the boundary on the example backend
    be = ExampleStructuralBackend()
    be.reset(WheelState(ambient_temp_c=20.0, track_temp_c=30.0))
    o = None
    for _ in range(300):
        o = be.step(WheelState(alpha=math.radians(4), Fz=1100, v_x=18, dt=1e-3))
    print(f"example structural sample after 0.3 s of slip:")
    print(f"  carcass deflection : {o.carcass_deflection_m*1000:.2f} mm")
    print(f"  contact patch      : {o.contact_length_m*1000:.0f} x "
          f"{o.contact_width_m*1000:.0f} mm")
    print(f"  tread temp (mean)  : {float(np.mean(o.tread_temp_c)):.1f} °C")
    print(f"  gas temp / pressure: {o.gas_temp_c:.1f} °C / "
          f"{o.inflation_pressure_pa/1000:.1f} kPa")
    print(f"  is_structural={o.is_structural()}  is_thermal={o.is_thermal()}")
    print(f"  (all of the above are FABRICATED — listed in synthesized: "
          f"{'pressure_distribution' in o.synthesized})")


if __name__ == "__main__":
    _demo()
