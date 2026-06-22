# Elbee Racing Baja — rebase notes

This tool is a Baja-SAE rebase of **KinematiK**, an open-source FSAE-EV suspension
& vehicle-dynamics studio by Frederik Thio (MIT licensed). The physics engine
(architecture-agnostic multibody kinematics, load transfer, grip balance, lap
sim, GGV, transient, compliance, bolted-joint) is unchanged and still validated
by the test suite; what changed is everything Baja-specific.

## What was rebased

**Subteams (one source of truth: `suspension/integration.py::TEAMS`).**
The 8 FSAE-EV teams became Elbee's 5: **Drivetrain, Front Suspension + Steering,
Rear Suspension, Chassis, Data Acquisition.** A `resolve_team()` alias map keeps
old saved projects from crashing. This cascades into Team Fit, Weight & Handover,
Lead Notes, Integration and the 3D model.

**Suspension/steering durability is the headline** (the year-after-year failure):
front and rear suspension are first-class owners, steering folds into the front
corner, and the Compliance tab (member deflection, bump-steer over travel,
bolted-joint separation) is foregrounded.

**Baja-correct physics defaults:**
- Drivetrain 3D body = engine + CVT + half-shafts (was traction motor + inverter);
  HV accumulator body removed.
- Lap Time / GGV default to ~7.5 kW, no downforce (ClA 0), dirt µ/crr, and a
  CVT+gearbox ratio map instead of a motor map.
- "Electronics (PCB)" tab reframed as **Data Acquisition** (DAQ/sensor harness,
  IPC-2221 heating / Onderdonk fusing / IR-drop brown-out — no HV).
- Interface ledger: 12 V electrics, no HV pack, Baja field sets; starter parts
  are engine/CVT/gearbox/logger; handover PDF shows a 12 V battery line.

**Branding:** page title, headers, README, source-file headers, requirements —
all Elbee Baja, attribution to the original author preserved under MIT.

## Validation
- `streamlit_app.py` boots clean via Streamlit AppTest: **0 exceptions.**
- Full test suite: **514 passing** (suite updated to the Baja contract;
  `rtree` is an optional dep for collision tests).

## Run it
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Known low-priority leftovers
A few deep aero/wind-tunnel sub-features in the VALIDATION tab still carry
downforce/wing framing internally. They're functional and isolated; not core to
Baja, so left as-is. Clean them if/when the team wants a Baja-pure validation tab.
