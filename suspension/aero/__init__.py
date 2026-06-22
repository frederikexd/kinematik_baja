# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
KinematiK aero co-simulation package.

The aero analogue of `tire_cosim`: a clean, typed, tested SEAM where an external
CFD solver (OpenFOAM / STAR-CCM+ / Fluent) plugs in, an orchestrator that sweeps
car attitude into an aero map, and a coupling that feeds that map back into the
existing point-mass lap sim. KinematiK owns the parameterisation, orchestration and
map; the meshing and the Navier–Stokes solve live OUTSIDE it, on the team's cluster
with the team's license. Provenance is first-class throughout — a CFD number is
never fabricated to fill a hole.

Two entry paths:
  * "A" — orchestrate runs:  AeroOrchestrator(backend, geometry).run(RunMatrix(...))
  * "B" — bring a map:        AeroMap.from_csv(text)  ->  AeroProvider(...)

Virtual Wind Tunnel (the CFD-validation path):
    A physical aero-map run in a wind tunnel exists mainly to CALIBRATE CFD: map the
    car over front/rear ride height, then run the IDENTICAL points in CFD and check
    that C_d/C_l/balance match. `windtunnel.py` owns that loop:
        from suspension.aero import (PhysicalAeroMap, TunnelProvenance, RideHeights,
                                     VirtualWindTunnel, StarCCMSolver)
        phys = PhysicalAeroMap(TunnelProvenance("A2"), reference_area_m2=1.0)
        phys.add_measurement(RideHeights(20, 40, 25.0), c_lift=-2.8, c_drag=1.05)
        vwt = VirtualWindTunnel(phys, "car.stl")
        specs = vwt.case_specs()           # exact same points, for Star-CCM+/TS-Auto
        # ... team runs the CFD, reads results back ...
        report = vwt.correlate(cfd_results)  # is k-omega SST calibrated to the tunnel?

Surface pressure taps (the where-is-it-loaded path):
    The coefficient correlation says how much downforce; the C_p field says WHERE it
    comes from and whether the wing has stalled. `pressure_tap.py` turns a run's raw
    transducer voltages into a non-dimensional C_p mapped onto the wing, then RMSEs it
    against the CFD surface tap-for-tap:
        from suspension.aero import (RawPressureScan, ScanProvenance, TapLocation,
                                     TapCalibration, CFDSurfaceCp, correlate_cp)
        scan = RawPressureScan(volts, taps, calibrations)   # off the DAQ
        cp   = scan.to_cp(ScanProvenance("A2", speed_ms=25))  # volts -> C_p on the wing
        print(cp.stall_indicator("main").note)              # where is it stalling?
        report = correlate_cp(cp, cfd_surface_cp)            # RMSE vs CFD surface C_p

Live acquisition (the off-the-hardware path, upstream of everything above):
    The car is bolted to an under-floor multi-axis force balance and skinned with
    hundreds of static pressure taps plumbed into electronic scanners (Scanivalve,
    Chell nanoDAQ), all clocked by a high-speed DAQ chassis. `daq.py` owns that front
    end: a Virtual Instrument binds the balance + scanners to the chassis through a
    backend, samples at kHz, NOTCHES the fan blade-pass tone and structural vibration
    out before averaging, decouples the balance through its interaction matrix, and
    streams CLEAN time-averaged raw forces (F_x, F_y, F_z) and a RawPressureScan that
    `to_cp` then turns into the C_p field above:
        from suspension.aero import (DAQChassis, BalanceCalibration, ForceBalanceSpec,
                                     PressureScannerSpec, ScannerVendor, VibrationFilter,
                                     AcquisitionSpec, SyntheticDAQ, VirtualInstrument)
        vi = VirtualInstrument(facility="A2", chassis=DAQChassis(2000.0),
                               balance=balance, scanners=[scanner], backend=backend)
        spec = AcquisitionSpec(seconds=10, speed_ms=25,
                               vibration=VibrationFilter(2000.0, fan_blade_pass_hz=137))
        forces, scan = vi.acquire(spec)         # clean F_x/F_y/F_z + raw P_static
        cp = scan.to_cp(vi.scan_provenance(spec))

Quick start (runnable today, no solver):
    from suspension.aero import (ReferenceAeroModel, AeroOrchestrator, RunMatrix)
    orch = AeroOrchestrator(ReferenceAeroModel(), "car.stl", reference_area_m2=1.0)
    print(orch.plan(RunMatrix(yaw_deg=[0,2,4,6])))     # cost preview
    report = orch.run(RunMatrix(yaw_deg=[0,2,4,6]), workdir="/tmp/sweep")
    amap = report.aero_map
"""

from .cfd import (
    Attitude, RunMatrix, CaseSpec, CoeffResult, CFDProvenance,
    SolverFidelity, CFDSolver, SolverUnavailable,
)
from .backends import (
    ReferenceAeroModel, FluentVerificationSolver, OpenFOAMSolver, StarCCMSolver,
    FluentSolver, TSAutoSolver, BACKENDS, get_backend,
)
from .panel_method import (
    PanelMethodModel, PanelParams, PanelMethodUnavailable,
)
from .ensemble import (
    EnsembleTunnelSolver, EnsembleResult, MemberOutcome, fused_results,
    DEFAULT_MEMBER_NAMES,
)
from .submit import (
    Submitter, LocalSubmitter, SlurmSSHSubmitter, SubmitResult,
)
from .aeromap import AeroMap, AeroQuery
from .orchestrator import AeroOrchestrator, OrchestratorReport
from .coupling import AeroProvider, estimate_attitude, attitude_from_dynamics
from .meshing import MeshParams, SnappyMesher, parse_checkmesh
from .windtunnel import (
    RideHeights, AeroMapGrid, GroundState, TunnelProvenance, PhysicalAeroMap,
    VirtualWindTunnel, PointCorrelation, TunnelCorrelationReport,
    ride_heights_to_attitude, attitude_to_ride_heights,
    downforce_to_clift, drag_to_cdrag, DEFAULT_TUNNEL_TOL,
)
from .piv import (
    SheetOrientation, LaserSheetPlane, PIVProvenance, FramePair, VelocityField,
    PIVProcessor, AcquisitionPlan, PIVRig, OfflinePIVRig, RigUnavailable,
    CFDFieldSlice, FieldCorrelationReport, correlate_field, separation_mask,
    DEFAULT_FIELD_TOL,
)
from .pressure_tap import (
    WingSurface, TapLocation, TapCalibration, ScanProvenance, RawPressureScan,
    CpField, StallVerdict, CFDSurfaceCp, TapResidual, CpCorrelationReport,
    correlate_cp, DEFAULT_CP_TOL,
)
from .daq import (
    BalanceAxis, BalanceCalibration, ScannerVendor, PressureScannerSpec,
    ForceBalanceSpec, DAQChassis, VibrationFilter, VibrationFilterReport,
    ChannelFilter, StreamingVibrationFilter,
    AcquisitionSpec, DAQProvenance, BalanceReading, DAQBackend, DAQUnavailable,
    OfflineDAQ, SyntheticDAQ, VirtualInstrument,
)

__all__ = [
    "Attitude", "RunMatrix", "CaseSpec", "CoeffResult", "CFDProvenance",
    "SolverFidelity", "CFDSolver", "SolverUnavailable",
    "ReferenceAeroModel", "FluentVerificationSolver", "OpenFOAMSolver",
    "StarCCMSolver", "FluentSolver",
    "TSAutoSolver", "BACKENDS", "get_backend",
    "PanelMethodModel", "PanelParams", "PanelMethodUnavailable",
    "EnsembleTunnelSolver", "EnsembleResult", "MemberOutcome", "fused_results",
    "DEFAULT_MEMBER_NAMES",
    "Submitter", "LocalSubmitter", "SlurmSSHSubmitter", "SubmitResult",
    "AeroMap", "AeroQuery",
    "AeroOrchestrator", "OrchestratorReport",
    "AeroProvider", "estimate_attitude", "attitude_from_dynamics",
    "MeshParams", "SnappyMesher", "parse_checkmesh",
    "RideHeights", "AeroMapGrid", "GroundState", "TunnelProvenance",
    "PhysicalAeroMap", "VirtualWindTunnel", "PointCorrelation",
    "TunnelCorrelationReport", "ride_heights_to_attitude",
    "attitude_to_ride_heights", "downforce_to_clift", "drag_to_cdrag",
    "DEFAULT_TUNNEL_TOL",
    "SheetOrientation", "LaserSheetPlane", "PIVProvenance", "FramePair",
    "VelocityField", "PIVProcessor", "AcquisitionPlan", "PIVRig",
    "OfflinePIVRig", "RigUnavailable", "CFDFieldSlice",
    "FieldCorrelationReport", "correlate_field", "separation_mask",
    "DEFAULT_FIELD_TOL",
    "WingSurface", "TapLocation", "TapCalibration", "ScanProvenance",
    "RawPressureScan", "CpField", "StallVerdict", "CFDSurfaceCp",
    "TapResidual", "CpCorrelationReport", "correlate_cp", "DEFAULT_CP_TOL",
    "BalanceAxis", "BalanceCalibration", "ScannerVendor", "PressureScannerSpec",
    "ForceBalanceSpec", "DAQChassis", "VibrationFilter", "VibrationFilterReport",
    "ChannelFilter", "StreamingVibrationFilter",
    "AcquisitionSpec", "DAQProvenance", "BalanceReading", "DAQBackend",
    "DAQUnavailable", "OfflineDAQ", "SyntheticDAQ", "VirtualInstrument",
]
