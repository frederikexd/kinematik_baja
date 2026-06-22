# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

from .kinematics import SuspensionKinematics, Hardpoints, CornerState

# Architecture-agnostic topology engine: a general multibody kinematics kernel,
# a library of parameterised topology templates (MacPherson, multi-link,
# trailing/semi-trailing arm, solid axle, twist-beam, heavy-truck steer linkage,
# plus a free-form builder for experimental layouts), and an adapter that runs
# the whole KinematiK pipeline on any of them.
from . import topology
from . import topologies
from .topology import (
    Point, Body, Constraint, Link, Coincident, OnLine, InPlane, Revolute,
    DriveZ, RackTranslation, AxleRoll, Mechanism, MechanismBuilder,
)
from .topologies import (
    double_wishbone, macpherson_strut, multilink, trailing_arm,
    semi_trailing_arm, solid_axle, twist_beam, truck_steer_linkage,
    from_links, TEMPLATES, list_templates, example,
)
from .adapter import GenericKinematics
from . import fullcar3d
from .fullcar3d import build_full_car_figure, influence_summary
from .dynamics import VehicleDynamics, VehicleParams, CornerLoads
from .tiremodel import (PacejkaLateral, default_tire, CombinedSlipTire,
                        default_combined_tire, relaxation_length)
from .ggv import (GGVGenerator, GGVParams, GGVResult,
                  sweep_parameter, quick_ggv)
from . import ggv
from . import chassis
from . import integration
from . import project
from . import tiremodel
from . import tirefit
from . import setup
from . import laptime
from . import correlation
from . import damper
from . import interfaces

# Flexible-body / compliance (ADAMS Flex-style) extension
from .flex import (
    Material, MATERIALS, tube_section, solid_rod_section,
    axial_stiffness_tube, FlexElement, FlexMesh, guyan_condense,
    CondensedFlexBody, load_flex_body, read_mnf,
)
from .loadpath import (
    WheelLoad, MemberForces, solve_member_forces,
    wheel_load_from_corner, MEMBERS,
)
from .compliance import (
    MemberStiffness, CompliantResult, CompliantCorner, corner_wheel_load,
)
from .joints import JointCompliance
from .bolted_joint import (
    BoltGrade, BOLT_GRADES, METRIC_COARSE,
    Fastener, ClampedStack, JointResult,
    analyze_joint, joint_findings,
)
from . import flex
from . import loadpath
from . import compliance
from . import joints
from . import bolted_joint

# Explicit high-frequency transient time-step DAE solver (the unsteady half of
# the lap: yaw/sideslip, pitch/dive, kerb strikes, snap-oversteer recovery).
from .transient import (
    TransientSolver, TransientParams, TransientResult, SettlingResult,
    DriverInput, RoadInput,
    step_steer_maneuver, snap_oversteer_maneuver, brake_to_throttle_maneuver,
    curb_strike_maneuver, run_maneuver, transient_vs_qss_corner,
)
from . import transient

# EV powertrain & energy layer (architecture comparison in seconds + kWh):
# single motor + diff vs two-motor axle split vs four-motor torque vectoring,
# with a regen/pack-energy budget. Wraps the QSS lap sim; never fakes the
# closed-loop yaw benefit QSS can't earn.
from .ev_powertrain import (
    Powertrain, EVParams, EVLapSimulator,
    EVRunResult, ArchitectureComparison,
)
from . import ev_powertrain

# Transient per-cell battery-pack thermal model (which cell cooks first, and
# where to put the fan). Wraps the EV lap sim at the energy seam: turns a virtual
# lap into a pack current-vs-time history, then time-steps a lumped-capacitance
# network over a grid of cells with a fan-position-dependent airflow map. Same
# calibration/never-raise contract as tire_thermal.
from .pack_thermal import (
    CellParams, default_cell_params, PackLayout,
    Fan, AirflowParams, PackThermalModel, PackThermalResult,
    pack_current_trace, simulate_pack_thermal,
    FanPlacementCandidate, FanPlacementStudy,
    optimize_fan_placement, fan_grid_candidates,
)
from . import pack_thermal

# Structural tire co-simulation boundary (the FTire / CDTire integration seam):
# a stateful tyre contract, a Pacejka-backed reference backend that refuses to
# fake structural/thermal channels, vendor adapter stubs, and a staggered co-sim
# driver around the transient solver.
from .tire_cosim import (
    StructuralTireModel, ReferenceTireModel, FTireModel, CDTireModel,
    WheelState, TireOutput, TireProvenance, TireFidelity,
    make_tire_backend, default_structural_tire,
)
from .tire_cosim_driver import (
    CosimCornerSet, CosimTireHistory, run_cosim_maneuver,
)
from .tire_thermal import (
    ThermalTireModel, ThermalParams, ThermalRun,
    default_thermal_params, simulate_warmup,
)
from . import tire_cosim
from . import tire_cosim_driver
from . import tire_cosim_ftire_example
from . import tire_thermal

# Aerodynamic CFD co-simulation boundary (OpenFOAM / STAR-CCM+ / Fluent seam).
# The aero analogue of tire_cosim: KinematiK owns the attitude sweep, the run
# orchestration and the aero map; the meshing and Navier-Stokes solve live outside,
# on the team's cluster with the team's license. Provenance is first-class.
from . import aero
from .aero import (
    Attitude, RunMatrix, CaseSpec, CoeffResult, CFDProvenance,
    SolverFidelity, CFDSolver, SolverUnavailable,
    ReferenceAeroModel, OpenFOAMSolver, StarCCMSolver, FluentSolver,
    LocalSubmitter, SlurmSSHSubmitter, SubmitResult,
    AeroMap, AeroQuery, AeroOrchestrator, OrchestratorReport,
    AeroProvider, estimate_attitude, attitude_from_dynamics,
    MeshParams, SnappyMesher, parse_checkmesh,
)
from .aero import get_backend as get_aero_backend
from . import mountpoints
from .mountpoints import (
    MountPoint, KeepOut, GeometryLedger, PropagationResult, propagate_mount_move,
)

# Electronics / PCB layer: copper-survival (IPC-2221 heating, Onderdonk fusing,
# IR-drop / ECU brown-out) and signal-integrity (differential-pair impedance +
# HV-aggressor coupling) checks, emitting the same Finding objects the rest of
# the integration board renders.
from . import electronics
from .electronics import (
    Trace, DiffPair, Aggressor, BoardLedger, BoardCheckResult,
    check_board, worst_case_currents, undeclared_loads,
    min_parallel_distance_mm, parallel_run_length_mm,
)
from . import harness
from .harness import (
    Connector, WireRun, HarnessLedger, HarnessCheckResult,
    Formboard, FormboardBranch, check_harness,
    awg_area_mm2, awg_nominal_od_mm,
)

__all__ = [
    "SuspensionKinematics", "Hardpoints", "CornerState",
    # architecture-agnostic topology engine
    "topology", "topologies", "GenericKinematics",
    "Point", "Body", "Constraint", "Link", "Coincident", "OnLine", "InPlane",
    "Revolute", "DriveZ", "RackTranslation", "AxleRoll", "Mechanism",
    "MechanismBuilder",
    "double_wishbone", "macpherson_strut", "multilink", "trailing_arm",
    "semi_trailing_arm", "solid_axle", "twist_beam", "truck_steer_linkage",
    "from_links", "TEMPLATES", "list_templates", "example",
    "VehicleDynamics", "VehicleParams", "CornerLoads",
    "PacejkaLateral", "default_tire", "CombinedSlipTire",
    "default_combined_tire", "relaxation_length",
    "chassis", "integration", "project", "tiremodel", "tirefit", "setup",
    "laptime", "correlation", "damper", "interfaces",
    # flexible-body extension
    "Material", "MATERIALS", "tube_section", "solid_rod_section",
    "axial_stiffness_tube", "FlexElement", "FlexMesh", "guyan_condense",
    "CondensedFlexBody", "load_flex_body", "read_mnf",
    "WheelLoad", "MemberForces", "solve_member_forces",
    "wheel_load_from_corner", "MEMBERS",
    "MemberStiffness", "CompliantResult", "CompliantCorner", "corner_wheel_load",
    "JointCompliance",
    "flex", "loadpath", "compliance", "joints",
    # transient time-step DAE solver
    "TransientSolver", "TransientParams", "TransientResult", "SettlingResult",
    "DriverInput", "RoadInput",
    "step_steer_maneuver", "snap_oversteer_maneuver", "brake_to_throttle_maneuver",
    "curb_strike_maneuver", "run_maneuver", "transient_vs_qss_corner",
    "transient",
    # EV powertrain & energy layer (architecture comparison in seconds + kWh)
    "Powertrain", "EVParams", "EVLapSimulator",
    "EVRunResult", "ArchitectureComparison", "ev_powertrain",
    # transient per-cell battery-pack thermal model (hot-cell map + fan placement)
    "CellParams", "default_cell_params", "PackLayout",
    "Fan", "AirflowParams", "PackThermalModel", "PackThermalResult",
    "pack_current_trace", "simulate_pack_thermal",
    "FanPlacementCandidate", "FanPlacementStudy",
    "optimize_fan_placement", "fan_grid_candidates", "pack_thermal",
    # structural tire co-simulation boundary (FTire / CDTire seam)
    "StructuralTireModel", "ReferenceTireModel", "FTireModel", "CDTireModel",
    "WheelState", "TireOutput", "TireProvenance", "TireFidelity",
    "make_tire_backend", "default_structural_tire",
    "CosimCornerSet", "CosimTireHistory", "run_cosim_maneuver",
    "tire_cosim", "tire_cosim_driver", "tire_cosim_ftire_example",
    # lumped-parameter tyre thermal channel (tread/carcass/gas energy balance)
    "ThermalTireModel", "ThermalParams", "ThermalRun",
    "default_thermal_params", "simulate_warmup", "tire_thermal",
    # aerodynamic CFD co-simulation boundary (OpenFOAM / STAR-CCM+ / Fluent seam)
    "Attitude", "RunMatrix", "CaseSpec", "CoeffResult", "CFDProvenance",
    "SolverFidelity", "CFDSolver", "SolverUnavailable",
    "ReferenceAeroModel", "OpenFOAMSolver", "StarCCMSolver", "FluentSolver",
    "get_aero_backend",
    "LocalSubmitter", "SlurmSSHSubmitter", "SubmitResult",
    "AeroMap", "AeroQuery", "AeroOrchestrator", "OrchestratorReport",
    "AeroProvider", "estimate_attitude", "attitude_from_dynamics",
    "MeshParams", "SnappyMesher", "parse_checkmesh", "aero",
    # geometric mount-point clash + CG propagation (CAD -> clash -> CG chain)
    "MountPoint", "KeepOut", "GeometryLedger",
    "PropagationResult", "propagate_mount_move",
    "mountpoints",
    # electronics / PCB layer (copper survival + signal integrity)
    "Trace", "DiffPair", "Aggressor", "BoardLedger", "BoardCheckResult",
    "check_board", "worst_case_currents", "undeclared_loads",
    # harness / 3-D loom (route, bend, clearance, formboard, BOM, copper mass)
    "Connector", "WireRun", "HarnessLedger", "HarnessCheckResult",
    "Formboard", "FormboardBranch", "check_harness",
    "awg_area_mm2", "awg_nominal_od_mm", "harness",
    "min_parallel_distance_mm", "parallel_run_length_mm",
    "electronics",
]
__version__ = "0.21.0"
