<!--
  KinematiK — Formula SAE suspension & vehicle dynamics toolkit
  Created by Frederik Thio. Copyright (c) 2026 Frederik Thio.
  Open source. Original author: Frederik Thio, creator of KinematiK.
-->

# GGV Diagram Generator

`suspension/ggv.py` builds a **GGV diagram** — the car's steady-state
acceleration envelope (longitudinal g vs lateral g) at each forward speed. It is
the most useful single picture you can have before stepping up to full transient
vehicle modelling, and it answers the question you actually care about: *how does
a design change reshape what the car can do?*

It is built **on top of** the existing `VehicleDynamics` load-transfer + per-corner
Pacejka chain, so the inputs you named are the levers that move the diagram:
weight, CG height, weight distribution, roll-centre height, wheel/spring rate
(through the motion ratio), camber, and aero ClA/CdA.

## Quick start

```python
from suspension.ggv import quick_ggv

res = quick_ggv(
    mass=280, cg_height=300, weight_dist_front=0.47,
    track_front=1200, track_rear=1180,
    roll_stiffness_front=350, roll_stiffness_rear=300,
    static_camber_front=-1.5, static_camber_rear=-1.5,
    power_w=60_000, cl_a=2.5, cd_a=1.2,   # cl_a=cd_a=0 for a wingless car
)

res.max_lat_g      # (S,) peak pure-cornering g at each speed
res.max_accel_g    # (S,) peak pure-forward g
res.max_brake_g    # (S,) peak pure-braking g (positive)
res.long_g         # (S, N) longitudinal g at each (speed, direction)
res.lat_g          # (S, N) lateral g — the full envelope surface
res.speeds         # (S,) the speeds, m/s
res.warnings       # anything the model wants you to know (e.g. wheel lift)
```

## Full control (with real geometry + your fitted tire)

```python
from suspension.dynamics import VehicleDynamics, VehicleParams
from suspension.ggv import GGVGenerator, GGVParams
from suspension import tiremodel

vp = VehicleParams(mass=280, cg_height=300, weight_dist_front=0.47,
                   spring_rate_front=35, spring_rate_rear=35,
                   arb_rate_front=0, arb_rate_rear=0,
                   use_spring_rates=True)        # wheel rate = k_spring * MR^2
tire = tiremodel.load_from_json("my_ttc_tire.json")   # YOUR fitted Pacejka
veh = VehicleDynamics(vp, front_kin, rear_kin, tire) # kinematics -> motion ratio, camber

gp = GGVParams(power_w=60_000, drive_axle="rear",
               cl_a=2.5, cd_a=1.2, aero_balance_front=0.45)
res = GGVGenerator(veh, gp).generate()
```

With `use_spring_rates=True` and kinematics attached, the spring rate maps to the
wheel/roll rate **through the real motion ratio** — so you sweep the spring you
actually buy, not an abstract roll stiffness.

## "What does changing X do to my envelope?"

```python
from suspension.ggv import sweep_parameter

s = sweep_parameter(veh, gp, "cg_height", [250, 300, 350, 400],
                    speed=20.0, metric="max_lat_g")
#  -> {"values": [...], "metric": [...]}   lateral grip vs CG height at 20 m/s
```

`param` may be any field of `VehicleParams` (`cg_height`, `weight_dist_front`,
`roll_stiffness_front`, `spring_rate_front`, `static_camber_front`, ...) or of
`GGVParams` (`cl_a`, `cd_a`, `power_w`, ...). It is resolved automatically and
restored afterward. `metric` is `"max_lat_g"`, `"max_accel_g"`, or `"max_brake_g"`.

To sweep **camber** as a free lever, build the vehicle with
`use_param_camber=True` so `static_camber_*` is live rather than the solved
kinematic camber.

## What it does and doesn't claim

- **Relative is trustworthy out of the box.** With the shipped generic tire,
  "does lowering the CG widen the envelope?" is reliable. Absolute g numbers are
  only as good as the tire — swap in your TTC-fitted Pacejka and they become real
  for your tire.
- **The longitudinal axis** combines a tire friction-circle limit with the
  powertrain (power-limited traction) and brakes. The combined-load corners use
  the standard elliptic friction-circle blend between the axis limits; supply a
  `CombinedSlipTire` with measured drive/brake data to calibrate that blend.
- **Aero** adds downforce (more grip with speed) and drag (less top-end accel) —
  both visible in the speed trends. At very high downforce-to-weight the braking
  limit can read optimistically high because the simple model assumes all four
  tires brake at `mu·Fz` with matched bias; treat extreme high-speed braking g as
  an upper bound, not a guarantee.
- **Wheel lift is flagged, not hidden.** If an inner tire fully unloads at the
  cornering limit, the rigid load-transfer model has saturated and the grip
  number there is an artifact — you'll get a warning telling you to lower the CG
  or soften the bar rather than a silently inflated number.

## The generic tire's camber insensitivity

The shipped `default_tire()` has near-flat camber sensitivity by design (its
camber coefficients are placeholders). So a camber sweep on it will look flat —
that's the tire, not the plumbing. A real TTC fit carries meaningful camber
terms and the sweep will respond.

(Note: `sweep_parameter` automatically forces `use_param_camber=True` for the
duration of a `static_camber_*` sweep, so the lever is genuinely live even when
corner kinematics are attached — and restores the flag afterward. A flat camber
curve is therefore the tire's behaviour, never a geometry override.)

## In the app: the GGV DIAGRAM tab

A `GGV DIAGRAM` tab sits next to `LAP TIME`. It reuses the live vehicle (the
same geometry/setup/tire the rest of the app solved), so it reflects every
upstream change. It offers:

- a powertrain/aero panel with the same defaults as the Lap Sim tab;
- an optional combined-slip tire (μx/μy ratio + ellipse exponents), with an
  honest calibrated/uncalibrated status line;
- the GGV cross-sections plot (the diagram itself) and a capability-vs-speed
  plot;
- a design-input sweep ("what reshapes the envelope?") for CG height, weight
  distribution, camber, roll stiffness, ClA and power;
- a one-click cross-check against the Lap Sim (see below).

## Consistency with the lap sim

`GGVParams.from_powertrain(pt)` builds the GGV's longitudinal/aero inputs from a
`laptime.Powertrain`, so a GGV and the lap sim built from the same numbers share
one source of truth. `validate_against_laptime(veh, pt)` then compares their
axis limits directly:

```python
from suspension.ggv import validate_against_laptime
res = validate_against_laptime(veh, pt)   # pt is a laptime.Powertrain
res["ok"]            # True if every comparison is within rel_tol (default 6%)
res["max_reldiff"]   # worst relative difference
res["note"]          # explanation when the only divergence is the known one
```

In practice the two agree to **<0.1%** on lateral and acceleration across
downforce levels. The one deliberate exception: with a combined-slip tire whose
`mu_x_ratio > 1`, the braking limits diverge because `laptime._decel_long`
applies the plain lateral μ on the brake side (no `mu_x_ratio`), while the GGV
applies it (braking is longitudinal too). The validator detects this and reports
it as a laptime brake-side simplification rather than a GGV error.

## Combined-slip calibration

The GGV reuses the repo's existing `CombinedSlipTire` (`mu_x_ratio`, `ell_kx`,
`ell_ky`, `is_calibrated`) — the same object `laptime.Powertrain.combined_tire`
takes. Pass one via `GGVParams(combined_tire=...)` or `from_powertrain`:

- `mu_x_ratio` scales the longitudinal axis limits (accel and brake);
- `ell_kx`/`ell_ky` shape the combined-g corners (the friction "circle" becomes
  a superellipse — fatter exponents give more combined-g headroom for
  trail-braking and power-down);
- leave `is_calibrated=False` until you've fitted the exponents to drive/brake
  TTC data — the coupling shape is physically valid either way, but the numbers
  are only quantitative once calibrated.
