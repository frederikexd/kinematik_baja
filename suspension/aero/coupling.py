# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Lap-sim coupling — the "B" payoff. Turns an AeroMap into the cl_a / cd_a the
existing point-mass lap sim consumes, as a function of car attitude, while staying
100% backward-compatible: with no map, behaviour is identical to today's scalars.

The lap sim works in coefficient*area terms (cl_a, cd_a) and a force law
F = 1/2 rho (cl_a) v^2. A CFD map is in non-dimensional C_L/C_D referenced to a
frontal area A. So the bridge is simply:  cl_a = -C_L * A ,  cd_a = C_D * A
(the sign flips C_L's down-negative convention into the lap sim's downforce-positive
cl_a). Attitude (roll/pitch/yaw/ride-height) is estimated from the corner state the
sim already computes, so the SAME geometry-driven attitude that the suspension model
produces also indexes the aero — closing geometry -> attitude -> aero -> lap time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cfd import Attitude
from .aeromap import AeroMap


@dataclass
class AeroProvider:
    """
    Adapts an AeroMap to the lap sim's (cl_a, cd_a) interface. If `aero_map` is None
    it returns the fixed scalar fallback, so existing LapSimParams behaviour is
    preserved exactly. If a map is present, it queries it at the supplied attitude.

    reference_area_m2 must match the map's normalisation, or downforce will be wrong.
    """
    reference_area_m2: float
    fallback_cl_a: float = 2.5
    fallback_cd_a: float = 1.2
    aero_map: Optional[AeroMap] = None

    def cla_cda(self, attitude: Optional[Attitude] = None) -> tuple[float, float]:
        """Return (cl_a, cd_a) for the lap sim at a given attitude."""
        if self.aero_map is None or attitude is None or len(self.aero_map) == 0:
            return self.fallback_cl_a, self.fallback_cd_a
        q = self.aero_map.query(attitude)
        if q.c_lift is None or q.c_drag is None:
            return self.fallback_cl_a, self.fallback_cd_a
        cl_a = -q.c_lift * self.reference_area_m2     # down-negative C_L -> +cl_a
        cd_a = q.c_drag * self.reference_area_m2
        return cl_a, cd_a

    def scalar_for_speed(self, speed_ms: float,
                         roll_deg: float = 0.0, pitch_deg: float = 0.0,
                         yaw_deg: float = 0.0,
                         ride_height_mm: float = 30.0) -> tuple[float, float]:
        """Convenience for the QSS sim, which mostly knows speed (+ optional attitude)."""
        att = Attitude(roll_deg, pitch_deg, yaw_deg, ride_height_mm, speed_ms)
        return self.cla_cda(att)

    def is_mapped(self) -> bool:
        return self.aero_map is not None and len(self.aero_map) > 0

    def status(self) -> str:
        if not self.is_mapped():
            return (f"scalar aero (cl_a={self.fallback_cl_a}, cd_a={self.fallback_cd_a}); "
                    f"no aero map loaded")
        return self.aero_map.status()


def estimate_attitude(speed_ms: float, lat_g: float = 0.0, long_g: float = 0.0,
                      roll_grad_deg_per_g: float = 1.2,
                      pitch_grad_deg_per_g: float = 0.6,
                      static_ride_mm: float = 30.0,
                      ride_drop_mm_per_g_aero: float = 0.0) -> Attitude:
    """
    Crude attitude estimate from the sim's instantaneous lat/long g and speed — a
    placeholder bridge until the suspension model feeds real roll/pitch/heave in.
    Roll from lateral g, pitch from longitudinal g, yaw left at 0 (steady-state).
    Deliberately simple and clearly labelled; refine by wiring the kinematics
    roll/pitch output in here.
    """
    roll = roll_grad_deg_per_g * lat_g
    pitch = pitch_grad_deg_per_g * long_g      # braking (long_g<0) => nose down (pitch<0)
    return Attitude(roll_deg=roll, pitch_deg=pitch, yaw_deg=0.0,
                    ride_height_mm=static_ride_mm, speed_ms=speed_ms)


def _axle_wheel_rate(veh, axle: str) -> Optional[float]:
    """
    Vertical wheel-centre ride rate (N/mm) for one axle, from spring rate x MR^2 when
    the geometry is attached, mirroring dynamics._axle_roll_stiffness. Returns None
    if it cannot be derived honestly (no spring rates / no geometry).
    """
    p = getattr(veh, "p", None)
    if p is None:
        return None
    import numpy as _np
    if axle == "front":
        k_spring = getattr(p, "spring_rate_front", None)
        kin = getattr(veh, "front_kin", None)
    else:
        k_spring = getattr(p, "spring_rate_rear", None)
        kin = getattr(veh, "rear_kin", None)
    if not getattr(p, "use_spring_rates", False) or kin is None or k_spring is None:
        return None
    try:
        mr = kin.motion_ratio()
    except Exception:                       # noqa: BLE001
        return None
    if mr is None or not _np.isfinite(mr) or mr <= 0:
        return None
    return float(k_spring) * mr * mr        # N/mm at the contact patch


def attitude_from_dynamics(veh, lat_g: float, long_g: float, speed_ms: float,
                           static_ride_mm: float = 30.0,
                           yaw_deg: float = 0.0,
                           brake_bias_front: float = 0.65,
                           drive_bias_rear: float = 1.0,
                           aero_provider: "Optional[AeroProvider]" = None,
                           rho: float = 1.225) -> tuple[Attitude, dict]:
    """
    The real geometry -> attitude link. Reads roll, pitch and heave OFF the attached
    VehicleDynamics object instead of inventing them, closing the chain

        hardpoints -> kinematics -> load transfer/roll/pitch -> aero attitude -> lap time

    with no fabricated step. Where a quantity genuinely cannot be derived (e.g. no
    spring rates set, so no ride rate, so no pitch), it is left at its neutral value
    and flagged in the returned `info` dict, rather than guessed.

      * ROLL  — taken directly from veh.lateral_load_transfer(lat_g)["roll_angle"],
        which already uses real spring/MR/RC physics. Signed by turn direction via
        the sign of lat_g.
      * PITCH — from the longitudinal load transfer dW: front and rear suspension
        deflect by dW/ride_rate, REDUCED by the anti-dive / anti-squat fraction the
        geometry provides, and the front-minus-rear deflection over the wheelbase is
        the pitch angle. Braking (long_g>0) => nose down (negative pitch).
      * RIDE HEIGHT — static minus aero heave: total downforce at this speed over the
        combined ride rate pushes the platform down. Uses the aero_provider's own
        coefficients if supplied (self-consistent), else leaves height static.

    Returns (Attitude, info) where info records which channels were real vs neutral.
    """
    import numpy as _np
    p = getattr(veh, "p", None)
    info: dict = {"roll_source": "neutral", "pitch_source": "neutral",
                  "ride_source": "static"}

    # ---- roll: straight from the dynamics model ---- #
    roll_deg = 0.0
    try:
        _loads, rinfo = veh.lateral_load_transfer(abs(lat_g))
        roll_mag = float(rinfo.get("roll_angle", 0.0))
        if _np.isfinite(roll_mag):
            roll_deg = _np.sign(lat_g) * roll_mag if lat_g != 0 else 0.0
            info["roll_source"] = "dynamics(spring/MR/RC)"
    except Exception as e:                  # noqa: BLE001
        info["roll_error"] = str(e)

    # ---- pitch: load transfer / ride rate, reduced by anti-dive/squat ---- #
    pitch_deg = 0.0
    kf = _axle_wheel_rate(veh, "front")
    kr = _axle_wheel_rate(veh, "rear")
    if p is not None and kf and kr and long_g != 0.0:
        try:
            dW, _ = veh.longitudinal_load_transfer(long_g)   # N, +ve braking
            # anti-dive/squat REDUCE the geometric portion of the deflection
            if long_g > 0:      # braking: front compresses, rear extends; anti-dive resists
                ad = veh.anti_dive_pct(brake_bias_front=brake_bias_front)
                frac_f = 1.0 - (ad / 100.0 if _np.isfinite(ad) else 0.0)
                frac_r = 1.0
            else:               # accel: rear squats; anti-squat resists at the rear
                asq = veh.anti_squat_pct(drive_bias_rear=drive_bias_rear)
                frac_f = 1.0
                frac_r = 1.0 - (asq / 100.0 if _np.isfinite(asq) else 0.0)
            frac_f = min(max(frac_f, 0.0), 1.0)
            frac_r = min(max(frac_r, 0.0), 1.0)
            # deflection (mm): front loses dW under accel / gains under braking
            defl_f = (dW / kf) * frac_f      # +ve = front compresses (braking)
            defl_r = (-dW / kr) * frac_r     # rear extends under braking
            wheelbase_mm = float(p.wheelbase)
            # nose-down pitch is negative; front compressing relative to rear pitches nose down
            pitch_rad = _np.arctan2((defl_r - defl_f), wheelbase_mm)
            pitch_deg = _np.degrees(pitch_rad)
            info["pitch_source"] = "dynamics(loadtransfer/anti-dive)"
        except Exception as e:              # noqa: BLE001
            info["pitch_error"] = str(e)
    elif long_g != 0.0:
        info["pitch_note"] = ("no ride rate available (set use_spring_rates and "
                              "attach geometry) -> pitch left neutral")

    # ---- ride height: static minus aero heave ---- #
    ride_mm = static_ride_mm
    if aero_provider is not None and kf and kr:
        try:
            cl_a, _cd_a = aero_provider.scalar_for_speed(
                speed_ms, roll_deg=roll_deg, pitch_deg=pitch_deg,
                ride_height_mm=static_ride_mm)
            downforce_N = 0.5 * rho * cl_a * speed_ms * speed_ms   # cl_a already = -C_L*A
            heave_mm = downforce_N / (kf + kr)      # N / (N/mm) = mm directly
            ride_mm = max(static_ride_mm - heave_mm, 1.0)
            info["ride_source"] = "static - aero heave"
            info["aero_heave_mm"] = heave_mm
        except Exception as e:              # noqa: BLE001
            info["ride_error"] = str(e)

    att = Attitude(roll_deg=float(roll_deg), pitch_deg=float(pitch_deg),
                   yaw_deg=float(yaw_deg), ride_height_mm=float(ride_mm),
                   speed_ms=float(speed_ms))
    return att, info
