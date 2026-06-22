<!--
  KinematiK — Formula SAE suspension & vehicle dynamics toolkit
  Created by Frederik Thio. Copyright (c) 2026 Frederik Thio.
  Open source. Original author: Frederik Thio, creator of KinematiK.
-->

# KinematiK Tire Co-Simulation Interface

**Status:** stable contract · **Applies to:** `suspension/tire_cosim.py`, `suspension/tire_cosim_driver.py` · **Version:** 0.16.0

This is the interface contract a third-party structural tire model (FTire, CDTire, or
any other) must conform to in order to be driven by KinematiK's transient solver. It
is modelled on the role the **Standard Tire Interface (STI / TYDEX)** plays in
ADAMS/Car: the multibody solver and the tire model agree, channel by channel with
fixed units and sign conventions, on what crosses the boundary each step, and the
tire vendor supplies a wrapper that conforms. KinematiK owns this contract; the
vendor solver is a licensed binary that is not bundled. A conforming wrapper is the
single thing you write to plug a real model in.

If you only read one thing: implement the `StructuralTireModel` protocol
(`reset`, `step`, `provenance`, `warnings`), map your solver's per-step call into
the `WheelState` → `TireOutput` channel table below, and **never put a
vendor-physics channel into `TireOutput.synthesized`** — that list is only for
placeholders.

---

## 1. The contract

A backend is **stateful**. The lifecycle is:

```
backend = MyTireModel(parameter_file=..., binding=...)   # construct + validate license/file
backend.reset(initial_WheelState)                        # carcass at rest, gas at ambient
loop each macro-step:
    out = backend.step(wheel_state)                      # advance internal state by ws.dt, return forces+state
```

`step()` advances the model's internal state (carcass nodes, contact solver, thermal
network) by exactly `WheelState.dt` seconds and returns the resulting `TireOutput`.
This mirrors how FTire and CDTire are actually called: hand them wheel-centre motion
plus a 3D road over a timestep, get forces plus an evolved state back.

### Required methods

| Method | Contract |
|---|---|
| `provenance() -> TireProvenance` | Declares backend name, fidelity, calibration state, parameter-file lineage. Must report `is_calibrated=True` **only** when running on a vendor file fitted to the actual tire. |
| `reset(state=None) -> None` | Initialise internal state. With a `WheelState`, seed carcass at rest and the thermal network at `ambient_temp_c` / `track_temp_c` / `inflation_pressure_pa`. |
| `step(ws: WheelState) -> TireOutput` | Advance one step. **Must not raise** — clamp, flag via `warnings()` / `TireOutput.synthesized`, and return a valid object so a 10 000-step run is never taken down by one bad sample. |
| `warnings() -> list[str]` | Cumulative, de-duplicated backend warnings. |

### Built-in backends (no vendor binary)

Two backends ship in-tree and need no license — build either with
`make_tire_backend(kind)`:

| `kind` | Class | Fidelity | Fills | Honest gaps |
|---|---|---|---|---|
| `reference` | `ReferenceTireModel` | `HANDLING` | Fx/Fy/Mz via Pacejka + friction ellipse, lateral relaxation as real internal state | all structural **and** thermal channels are `None`, listed in `synthesized` |
| `thermal` | `ThermalTireModel` | `THERMAL` | the above **plus** `tread_temp_c` (per band), `carcass_temp_c`, `gas_temp_c`, hot `inflation_pressure_pa` from a 3-node lumped energy balance | structural channels still `None`; thermal channels are real physics but **uncalibrated** — flagged `synthesized` until `ThermalParams.calibrated=True` |

`ThermalTireModel` (`suspension/tire_thermal.py`) is the honest in-house answer to
"can KinematiK say anything about tyre temperature without a vendor thermal module?"
It integrates tread/carcass/gas nodes heated by frictional sliding power
(`|F·v_slip|`) and rolling hysteresis, cooled by speed-dependent convection to air
and conduction to the track, with an ideal-gas pressure rise. The **equations are
textbook**; the masses, heat-transfer coefficients and the optional grip-vs-
temperature law μ(T) are representative defaults, **not** fitted to your tyre — so a
true absolute temperature is impossible without temperature-swept TTC data, and the
backend says so on every sample. Read its *shape* (warm-up time, the camber-driven
inner/outer tread split, front/rear divergence over a stint, the pressure rise), not
its absolute degrees, until you calibrate it. Tests: `tests/test_tire_thermal.py`.

---

## 2. Frame & sign conventions

All channels are at the **wheel centre / contact patch in the tire's own frame**,
ISO/TYDEX-consistent and matching `transient.py`:

- **x** forward (+), **y** left (+), **z** up (+)
- `alpha` slip angle, rad. **Sign matches `tiremodel.PacejkaLateral`: positive
  `alpha` produces negative `Fy`.** A conforming wrapper must reproduce this sign or
  the yaw response inverts. (Verify with the round-trip check in §5.)
- `kappa` longitudinal slip ratio, dimensionless, drive (+) / brake (−).
- `gamma` inclination (camber) angle, rad. The driver passes an **antisymmetric**
  left/right `gamma` (via `sign(y_i)`) so static camber cancels in a straight line —
  see the note in `transient.algebraic`. Your model receives the per-corner signed
  value; do not re-symmetrise it.
- `Fz` (input) vertical **load demand**, N. For a handling backend this is the
  contact load to evaluate at. For a **structural** backend it is advisory — your
  carcass model computes the true reaction from deflection and returns it as
  `TireOutput.Fz` (which the driver then trusts).
- Moments: `Mz` aligning (+ about z), `Mx` overturning, `My` rolling resistance.

---

## 3. Input channel table — `WheelState`

| Field | Unit | Used by | Meaning |
|---|---|---|---|
| `alpha` | rad | all | slip angle (already lagged by the driver's relaxation state; a structural backend with its own relaxation should take the *target* — see §6) |
| `kappa` | – | all | longitudinal slip ratio |
| `gamma` | rad | all | inclination angle, signed per corner |
| `Fz` | N | handling | vertical load demand (advisory for structural) |
| `omega` | rad/s | structural/thermal | wheel spin rate (rolling, heat generation) |
| `v_x` | m/s | all | forward speed at contact (sets relaxation rate, slip) |
| `v_y` | m/s | structural | lateral speed at contact |
| `z_wheel` | m | structural | wheel-centre height vs static (carcass radial input) |
| `zdot_wheel` | m/s | structural | wheel-centre vertical velocity |
| `road_points` | (k,3) m | structural | 3D road under the patch, tire frame; `None` ⇒ flat. **This is the channel that makes enveloping over kerbs/cleats possible** — the reference backend ignores it because Pacejka has no carcass. |
| `ambient_temp_c` | °C | thermal | environment temperature |
| `track_temp_c` | °C | thermal | surface temperature |
| `inflation_pressure_pa` | Pa | thermal | cold/set pressure; `None` ⇒ model default |
| `dt` | s | all | the step to advance over (the macro-step; see §6) |

---

## 4. Output channel table — `TireOutput`

Forces and moments are **always** present. Structural and thermal channels are
`Optional` and **must be `None` if your backend does not compute them** — `None`
means "this model cannot tell you that," which is the honest answer, never zero and
never a guess.

| Field | Unit | Fidelity | Notes |
|---|---|---|---|
| `Fx`, `Fy`, `Fz` | N | all | contact forces; structural `Fz` is the carcass reaction |
| `Mz`, `Mx`, `My` | N·m | all (Mz handling+) | aligning / overturning / rolling-resistance moments |
| `carcass_deflection_m` | m | structural | radial deflection at patch centre |
| `contact_length_m`, `contact_width_m` | m | structural | patch dimensions |
| `pressure_distribution` | (ny,nx) Pa | structural | contact-patch pressure field |
| `effective_radius_m` | m | structural | loaded rolling radius |
| `tread_temp_c` | (nbands,) °C | thermal | per-band tread temperature |
| `carcass_temp_c`, `gas_temp_c` | °C | thermal | carcass / inflation-gas temperature |
| `inflation_pressure_pa` | Pa | thermal | **current (hot)** pressure — the build-up |
| `synthesized` | list[str] | all | names of channels that are placeholders. **A real vendor channel must NOT appear here.** |

`is_structural()` returns True iff `carcass_deflection_m is not None`;
`is_thermal()` iff a temperature channel is populated. The UI uses these plus
`provenance()` to label what's measured vs. placeholder.

---

## 5. Conformance checklist

A wrapper conforms when:

1. `isinstance(my_model, StructuralTireModel)` is True (the protocol is `runtime_checkable`).
2. `step()` never raises on `NaN`, negative `Fz`, or absurd inputs (see
   `test_reference_never_raises_on_garbage`).
3. **Sign round-trip:** positive `alpha` yields negative `Fy`; a left steer produces
   a left (+y) force and a left (+) yaw in a full-car run. If yaw diverges the wrong
   way, your `alpha`/`Fy` sign is flipped.
4. `provenance().is_calibrated` is True only on a fitted vendor file.
5. Every `TireOutput` field your model can't compute is `None` and listed in
   `synthesized`; every field it can compute is populated and **absent** from
   `synthesized`.
6. The co-sim driver runs all four named manoeuvres end-to-end
   (`run_cosim_maneuver` for `step_steer`, `snap_oversteer`, `brake_to_throttle`,
   `curb_strike`) without `result=None`.

Mirror the existing tests in `tests/test_tire_cosim.py` for your backend.

---

## 6. Co-simulation timing (important)

The driver in `tire_cosim_driver.py` uses a **staggered** scheme: tire forces are
held constant as an external input across one macro-step while the vehicle ODE is
integrated, then the tire state is advanced **once** per macro-step from the
step-averaged wheel motion. This is how ADAMS/Car couples FTire/CDTire to the
multibody solver, and it matters for two reasons:

- **Do not advance your internal state inside the RK4 sub-evaluations.** RK4 calls
  the vehicle derivative four times per step at trial points; the driver calls your
  `step()` once per *macro*-step, with `dt` = the macro-step. If your binding is
  wired to advance on every force query, you will integrate carcass/thermal state
  4× and partly backward in time per logged step.
- If your model runs faster than the vehicle macro-step (FTire often wants
  sub-0.1 ms internally), **sub-cycle inside your `step()`**: loop your own solver
  over `ws.dt` and return the end-of-step forces/state. Keep that internal; expose
  only the macro-step contract.

**Slip relaxation ownership.** The reference backend owns lateral relaxation as its
one honest internal state, and the driver passes it the lagged `alpha`. A real
structural model has its own carcass relaxation; if so, it should take the *unlagged
target* slip and not double-count the lag. Set this explicitly when you wire the
binding — it is the one place the two relaxation models can collide.

---

## 7. Per-vendor binding notes

Neither solver is bundled; both are licensed binaries. The wrapper you write binds
KinematiK's `step()` to the vendor's per-step co-sim entry point.

### FTire (cosin)
- **Binding:** the cosin co-sim interface — the FTire/cosim C/Fortran API, the cosin
  Python bindings, or an **FTire FMU** via an FMI runtime.
- **Parameter file:** the FTire property file (`.tir` with FTire blocks, or cosin's
  own format) fitted by cosin to your tire's rig data (cleat, footprint, thermal).
- **State init:** seed at `reset()` from `WheelState` ambient/track temperature and
  set pressure.
- **Channels it can fill:** all structural and thermal outputs, including
  `pressure_distribution` and `inflation_pressure_pa` build-up (with the TKC/thermal
  module licensed).
- Pass `binding=`, `parameter_file=`, and (optionally) `library_path=` to
  `FTireModel`. Without `binding` it raises a clear, actionable error by design.

### CDTire (Fraunhofer ITWM)
- **Binding:** the **CDTire FMU** (via an FMI runtime) or its C API. CDTire/3D is the
  structural shell-element variant; CDTire/Thermal adds the thermal network.
- **Parameter file:** the CDTire parameter set fitted by ITWM to your tire.
- **Channels:** carcass deformation, contact pressure (CDTire/3D); tread/gas
  temperature and pressure (CDTire/Thermal).
- Pass `binding=`, `parameter_file=` to `CDTireModel`; same refuse-until-bound
  behaviour as FTire.

A worked, runnable wrapper skeleton (a fake binding that fills the structural and
thermal channels so you can see exactly where vendor data lands, and which tests it
must pass) is in `suspension/tire_cosim_ftire_example.py`.

---

## 8. Honesty contract (why the `None`s are mandatory)

A structural model emitting a detailed 3D pressure field and a tread-temperature map
from unvalidated defaults is **more** dangerous than a single grip number, because
the output looks like measurement and nobody questions it. The trust in an
ADAMS/Car + FTire result lives entirely in provenance: which parameters came from a
fitted file, which co-sim settings are validated. This interface enforces that by
construction — `provenance().is_calibrated`, `TireOutput.synthesized`, and the `None`
rule — so a clean KinematiK board never implies more than the data behind it. A
wrapper that fills channels with plausible defaults instead of `None` defeats the
entire point of integrating a structural model and should fail review.
