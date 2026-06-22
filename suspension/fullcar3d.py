# ============================================================================
#  KinematiK — Formula SAE suspension & vehicle dynamics toolkit
#  Dynamic full-vehicle 3D model (pure Python + Plotly).
#
#  A live Baja-buggy (tube-frame) assembled from the data every sub-team has already
#  entered. Edit a hardpoint, a spring rate, a wing's downforce, the battery
#  mass — and the body that subsystem owns visibly changes here, instantly,
#  because the figure is rebuilt from the same session state those tabs write.
# ============================================================================

"""
WHAT THIS DRAWS

A single Plotly 3D figure of an open tube-frame Baja SAE buggy, built from:

  * suspension geometry   (Hardpoints)          -> the four corners + tires
  * vehicle parameters    (VehicleParams)        -> wheelbase, track, CG, mass,
                                                    ride height (from spring rate)
  * the integration ledger (IntegrationLedger)   -> every other subsystem:
        aerodynamics -> none (Baja runs no wings; the aero body is disabled)
        powertrain   -> single-cylinder engine + CVT + half-shafts sized by power
        cooling      -> sidepod radiator ducts sized by required airflow
        electrics    -> small 12 V battery / DAQ box sized by its envelope+mass
        brakes       -> brake discs at each corner sized by brake torque
        chassis      -> the tubular space frame (lower floor rails, cockpit bay,
                        main & front roll hoops, side-impact bars, driver helmet)
        data-acq     -> a small logger pod (no meaningful envelope, shown small)

Every body is a real triangulated mesh (Mesh3d) or line set, positioned in the
kinematics frame (mm, SAE axes: x rear, y right, z up). Because the geometry is
recomputed from state on every Streamlit rerun, the "dynamic" requirement is met
structurally: there is no cached car. Each subsystem sees its own change and the
knock-on to the whole car (CG marker, mass readout) the moment it edits.

HOW A SUBSYSTEM'S NUMBERS BECOME GEOMETRY

Where a subsystem declares an explicit envelope box (env_x/y/z), we draw that box —
it is the literal thing they reserved. Where they declare a performance number but
no box (e.g. aero downforce, powertrain power), we map that number through a
documented, monotonic sizing law to a sensible body so the change is *visible* and
*directional* (more downforce -> bigger wing) without pretending to be CFD. Bodies
sized this way are labelled "(sized from <channel>)" so nobody mistakes the drawing
for analysis.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .kinematics import Hardpoints, SuspensionKinematics

# --------------------------------------------------------------------------- #
#  Palette — matched to the app's dark instrument styling.
# --------------------------------------------------------------------------- #
COLORS = dict(
    upper="#37e0d0", lower="#ffb02e", upright="#ffffff", tie="#ff5a52",
    push="#9b8cff", rocker="#5ad17a", spring="#ff9f43",
    wheel="#6f7d8c", tire="#15181c", tire_edge="#3a434c", rim="#23282e",
    # Matte-black tub with a hint of the photo's red/yellow livery accent.
    monocoque="#1d2024", nose="#23262b", livery="#d63a2f",
    frame="#0d1013", hoop="#0d1013", helmet="#e9edf1", helmet_band="#d63a2f",
    halo="#10141a",
    wing="#202428", wing_edge="#3a414a", endplate="#16191d",
    sidepod="#23262b", radiator="#ff6b5a", inlet="#0e1216",
    engine="#3a3a3f", motor="#454a52", airbox="#33373d",
    battery="#1f2a20", batt_edge="#5ad17a",
    brake="#c2410c", logger="#33373d",
    point="#e7ecf1", floor="#0c1014", cg="#ffd166",
    custom="#37e0d0", cad_neon="#1F8FFF",
)


# --------------------------------------------------------------------------- #
#  Mesh primitives
# --------------------------------------------------------------------------- #
def _box(cx, cy, cz, lx, ly, lz):
    """Axis-aligned box centred at (cx,cy,cz) with full extents (lx,ly,lz)."""
    hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], float)
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 5, 1), (0, 4, 5), (3, 2, 6), (3, 6, 7),
        (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4),
    ]
    i = [f[0] for f in faces]; j = [f[1] for f in faces]; k = [f[2] for f in faces]
    return v, np.array(i), np.array(j), np.array(k)


def _prism_xsection(profile_xy, x_positions, scales):
    """Loft a 2D cross-section (y,z) along x. scales: (sy,sz,oy,oz) per station."""
    prof = np.asarray(profile_xy, float)
    n = len(prof)
    rings = []
    for x, (sy, sz, oy, oz) in zip(x_positions, scales):
        rings.append(np.column_stack([
            np.full(n, x), prof[:, 0] * sy + oy, prof[:, 1] * sz + oz]))
    verts = np.vstack(rings)
    I, J, K = [], [], []
    for s in range(len(rings) - 1):
        b0, b1 = s * n, (s + 1) * n
        for a in range(n):
            b = (a + 1) % n
            v00, v01, v10, v11 = b0 + a, b0 + b, b1 + a, b1 + b
            I += [v00, v00]; J += [v01, v11]; K += [v11, v10]
    return verts, np.array(I), np.array(J), np.array(K)


def _ellipse_ring(n=20):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(np.cos(t), np.sin(t)) for t in th]


def _cylinder(center, axis, radius, length, n=24, cap=True):
    axis = np.asarray(axis, float); axis /= (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, ref); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.array([radius * (np.cos(t) * u + np.sin(t) * v) for t in th])
    c = np.asarray(center, float)
    a0, a1 = c - axis * (length / 2), c + axis * (length / 2)
    verts = np.vstack([a0 + rim, a1 + rim])
    I, J, K = [], [], []
    for a in range(n):
        b = (a + 1) % n
        I += [a, a]; J += [b, a + n]; K += [a + n, b + n]
    if cap:
        ci0, ci1 = len(verts), len(verts) + 1
        verts = np.vstack([verts, a0, a1])
        for a in range(n):
            b = (a + 1) % n
            I += [ci0]; J += [b]; K += [a]
            I += [ci1]; J += [a + n]; K += [b + n]
    return verts, np.array(I), np.array(J), np.array(K)


def chassis_cad_footprint(vp, custom_parts, ledger=None):
    """If a chassis CAD/estimate is present, return the footprint the rest of the
    car should attach to: dict(wb, length, width, height, cx, cy, cz, x_front,
    x_rear) in car SAE mm. Returns None if no chassis replacement exists.

    The surrounding dummy parts (wheels, wings, motor, etc.) are then placed
    against THIS footprint, so a real chassis pulls the whole car into a coherent
    assembly instead of leaving the dummies at the generic envelope.
    """
    import numpy as _np
    for cp in (custom_parts or []):
        if cp.get("replaces_subsys") != "chassis":
            continue
        mp = cp.get("mesh")
        # Determine the placed extents (length x, width y, height z) in car frame.
        if mp and mp.get("verts"):
            raw = _np.asarray(mp.get("size_mm") or [0, 0, 0], float)
            scale = float(cp.get("mesh_scale", 1.0) or 1.0)
            axis_map = cp.get("axis_map", "auto")
            # Reference slot for auto-orient: the generic monocoque anchor.
            try:
                ta = suggest_part_geometry_for(vp, "monocoque", ledger=ledger)
                ref = _np.array([ta["l_mm"], ta["w_mm"], ta["h_mm"]], float)
            except Exception:
                ref = _np.array([1550.0, 320.0, 300.0])
            if axis_map == "auto":
                so = _np.argsort(raw); to = _np.argsort(ref)
                ext = _np.zeros(3)
                for r in range(3):
                    ext[to[r]] = raw[so[r]]
            else:
                ext = raw
            # Apply the auto/explicit fit factor the renderer will use (middle axis).
            if cp.get("fit_to_envelope"):
                try:
                    so = _np.argsort(ext); tos = _np.argsort(ref)
                    paired = [ref[tos[r]] / (ext[so[r]] if ext[so[r]] > 1e-6 else 1.0)
                              for r in range(3)]
                    scale = scale * float(paired[1])
                except Exception:
                    pass
            ext = ext * scale
            L, W, H = float(ext[0]), float(ext[1]), float(ext[2])
        else:
            L = float(cp.get("l_mm", 0) or 0)
            W = float(cp.get("w_mm", 0) or 0)
            H = float(cp.get("h_mm", 0) or 0)
        if L <= 0:
            return None
        cx = float(cp.get("x_mm", 0) or 0)
        cy = float(cp.get("y_mm", 0) or 0)
        cz = float(cp.get("z_mm", 0) or 0)
        # Wheelbase ≈ a bit shorter than the hull length (axles sit inboard of
        # nose/tail). Use ~78% of hull length as a sane FSAE-ish ratio.
        wb = max(900.0, L * 0.78)
        return dict(wb=wb, length=L, width=W, height=H, cx=cx, cy=cy, cz=cz,
                    x_front=cx + wb / 2.0, x_rear=cx - wb / 2.0)
    return None


def _orient_part_mesh(verts, *, axis_map="z_up", yaw_deg=0.0, scale=1.0,
                      centre=(0.0, 0.0, 0.0), post_scale=None, axis_perm=None):
    """Place an imported CAD part's vertices into the car's SAE frame.

    verts come recentred on the part's own bbox centre (from chassis.load_part_mesh).
    We optionally remap axes (CAD up-axis -> car z-up), apply a yaw about car z,
    scale, then translate the part's centre to `centre`. Returns an (N,3) array.

        axis_map : "z_up"  CAD already z-up (no swap)
                   "y_up"  CAD is y-up (Y->Z, Z->-Y): common SolidWorks export
                   "x_up"  CAD is x-up (X->Z, Z->-X)
                   "auto"  no preset swap here; pair with axis_perm for auto-align
        axis_perm : optional (p0,p1,p2) permutation of source axes onto car
                    X,Y,Z — a CLEAN 90° re-labelling (never a distortion) used to
                    auto-align the part's longest side with the slot's longest.
        post_scale : optional (sx,sy,sz) applied in CAR axes AFTER orientation.
    """
    V = np.asarray(verts, float).reshape(-1, 3) * float(scale)
    if axis_perm is not None:
        V = V[:, list(axis_perm)]
    elif axis_map == "y_up":
        V = np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])
    elif axis_map == "x_up":
        V = np.column_stack([-V[:, 2], V[:, 1], V[:, 0]])
    if yaw_deg:
        a = np.radians(float(yaw_deg))
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        V = V @ R.T
    if post_scale is not None:
        # Stretch about the part's own centre (it's recentred at origin here).
        ctr0 = (V.max(axis=0) + V.min(axis=0)) / 2.0
        V = (V - ctr0) * np.asarray(post_scale, float) + ctr0
    return V + np.asarray(centre, float)


def _airfoil_section(n=14, thickness=0.12):
    """A closed 2D aerofoil-ish ring in (chord, thickness) coords, chord 0..1.

    Cambered teardrop: thicker near the leading edge, tapering to the trailing
    edge, so a lofted wing reads as a real element rather than a flat slab.
    """
    xs = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # cosine-spaced 0..1
    yt = thickness * (1.4845 * np.sqrt(np.clip(xs, 0, 1)) - 0.63 * xs
                      - 1.758 * xs**2 + 1.4215 * xs**3 - 0.5075 * xs**4)
    camber = 0.06 * (1 - (2 * xs - 1) ** 2)  # gentle single-element camber
    upper = np.column_stack([xs, camber + yt])
    lower = np.column_stack([xs[::-1], (camber - yt)[::-1]])
    return np.vstack([upper, lower])


def _wing_element(cx, cy, cz, chord, span, *, thickness=0.12, aoa_deg=-6.0,
                  n_sec=14):
    """A single aerofoil wing element centred at (cx,cy,cz), spanning in y.

    chord runs in x (fore-aft), span in y, with a small angle of attack rotating
    the section in the x-z plane. Returns mesh arrays ready for Mesh3d.
    """
    sec = _airfoil_section(n_sec, thickness)           # (m,2): chord, thick
    m = len(sec)
    a = np.deg2rad(aoa_deg)
    ca, sa = np.cos(a), np.sin(a)
    # section local -> (x,z): chord along x (centred), thickness along z, rotated
    chord_local = (sec[:, 0] - 0.5) * chord
    thick_local = sec[:, 1] * chord
    sx = chord_local * ca - thick_local * sa
    sz = chord_local * sa + thick_local * ca
    ys = np.array([cy - span / 2, cy + span / 2])
    rings = []
    for y in ys:
        rings.append(np.column_stack([cx + sx, np.full(m, y), cz + sz]))
    verts = np.vstack(rings)
    I, J, K = [], [], []
    for s in range(len(rings) - 1):
        b0, b1 = s * m, (s + 1) * m
        for aa in range(m):
            bb = (aa + 1) % m
            I += [b0 + aa, b0 + aa]; J += [b0 + bb, b1 + bb]; K += [b1 + bb, b1 + aa]
    return verts, np.array(I), np.array(J), np.array(K)


def _tube(p0, p1, radius, n=12):
    """A capped cylinder between two endpoints — for roll hoops and frame tubes."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    axis = p1 - p0
    length = np.linalg.norm(axis) + 1e-12
    center = (p0 + p1) / 2
    return _cylinder(center, axis, radius, length, n=n, cap=True)


def _swept_tube(points, radius, n=10):
    """A tube following a polyline of points — for the curved main roll hoop."""
    pts = [np.asarray(p, float) for p in points]
    V = []
    I = []
    J = []
    K = []
    base = 0
    for a in range(len(pts) - 1):
        v, i, j, k = _tube(pts[a], pts[a + 1], radius, n=n)
        V.append(v)
        I += list(i + base); J += list(j + base); K += list(k + base)
        base += len(v)
    return np.vstack(V), np.array(I), np.array(J), np.array(K)


def _sphere(center, radius, n=16):
    """A UV sphere — used for the driver's helmet."""
    c = np.asarray(center, float)
    u = np.linspace(0, np.pi, n)          # polar
    w = np.linspace(0, 2 * np.pi, n)      # azimuth
    U, W = np.meshgrid(u, w)
    x = c[0] + radius * np.sin(U) * np.cos(W)
    y = c[1] + radius * np.sin(U) * np.sin(W)
    z = c[2] + radius * np.cos(U)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    I, J, K = [], [], []
    cols = n
    for a in range(n - 1):
        for b in range(n - 1):
            v00 = a * cols + b
            v01 = a * cols + (b + 1)
            v10 = (a + 1) * cols + b
            v11 = (a + 1) * cols + (b + 1)
            I += [v00, v00]; J += [v01, v11]; K += [v11, v10]
    return verts, np.array(I), np.array(J), np.array(K)


# --------------------------------------------------------------------------- #
#  Corner geometry transforms
# --------------------------------------------------------------------------- #
def _corner_transform(p, *, mirror_y, lateral_scale, x_shift, y_center_ref):
    if p is None:
        return None
    q = np.array(p, float).copy()
    dy = (q[1] - y_center_ref) * lateral_scale
    q[1] = y_center_ref + (-dy if mirror_y else dy)
    q[0] = q[0] + x_shift
    return q


def _solved_corner_points(hp: Hardpoints, ride_drop_mm: float = 0.0):
    kin = SuspensionKinematics(hp)
    s = kin.static
    pts = dict(
        upper_front_inner=np.array(hp.upper_front_inner, float),
        upper_rear_inner=np.array(hp.upper_rear_inner, float),
        lower_front_inner=np.array(hp.lower_front_inner, float),
        lower_rear_inner=np.array(hp.lower_rear_inner, float),
        tie_rod_inner=np.array(hp.tie_rod_inner, float),
        upper_outer=np.array(s.upper_outer, float),
        lower_outer=np.array(s.lower_outer, float),
        tie_rod_outer=np.array(s.tie_rod_outer, float),
        wheel_center=np.array(s.wheel_center, float),
        contact_patch=np.array(s.contact_patch, float),
    )
    if hp.has_rocker():
        for kk in ("rocker_pivot", "rocker_pushrod", "rocker_spring", "spring_inner"):
            vv = getattr(hp, kk)
            if vv is not None:
                pts[kk] = np.array(vv, float)
        po = s.pushrod_outer if s.pushrod_outer is not None else hp.pushrod_outer
        if po is not None:
            pts["pushrod_outer"] = np.array(po, float)
    if ride_drop_mm:
        for kk in pts:
            pts[kk] = pts[kk] - np.array([0, 0, ride_drop_mm], float)
    return pts, s


# --------------------------------------------------------------------------- #
#  Topology-agnostic corner extractor
#
#  The full car must reflect whatever suspension ARCHITECTURE the team picked,
#  not just double wishbones. A double-wishbone corner is described by named
#  Hardpoints; every other topology (MacPherson, multi-link, trailing/semi-
#  trailing arm, solid axle, twist-beam, truck steer linkage, free-form) is
#  described by a GenericKinematics mechanism that reports its own member set via
#  render_segments(). This helper normalises BOTH into the same list of drawable
#  segments + the wheel centre / contact patch / camber the tire needs, so the
#  rest of the renderer is identical regardless of architecture.
# --------------------------------------------------------------------------- #

# Stable colour assignment for agnostic member labels, so the same link is the
# same colour on all four corners and across reruns.
_AGNOSTIC_PALETTE = [
    "#37e0d0", "#ffb02e", "#9b8cff", "#5ad17a", "#ff9f43",
    "#5cd2ff", "#ff7ab6", "#b6ff5a", "#ffd166", "#7d8893",
]


def _agnostic_color(label, registry):
    """Deterministic colour for a member label (its leading token), assigned on
    first sight and reused, so member 'L2' is always the same hue."""
    base = (label or "link").split()[0]
    if base not in registry:
        registry[base] = _AGNOSTIC_PALETTE[len(registry) % len(_AGNOSTIC_PALETTE)]
    return registry[base], base


# Map a subsystem name to the COLORS key whose hue best represents it, so a
# user-dropped custom part reads as "belonging to" that sub-team at a glance.
_SUBSYS_COLOR_KEY = {
    "aerodynamics": "wing", "brakes": "brake", "chassis": "monocoque",
    "cooling": "radiator", "electrics": "batt_edge", "powertrain": "motor",
    "suspension": "point", "data-acquisition": "logger",
}


def sub_color_key(subsys):
    """COLORS key for a subsystem's representative hue ('custom' if unknown)."""
    return _SUBSYS_COLOR_KEY.get(subsys, "custom")


def _is_wishbone_hardpoints(corner) -> bool:
    """True if `corner` is a double-wishbone Hardpoints (has the named fields)."""
    return isinstance(corner, Hardpoints)


def _extract_corner(corner, ride_drop_mm, color_registry):
    """Normalise a corner (Hardpoints OR GenericKinematics-like) into:
        dict(segments=[(p, q, label, color, group)],
             markers=[points...], wheel_center, contact_patch, camber)
    All points already lowered by ride_drop_mm.

    For wishbones we keep the named-link colour scheme (cyan upper, amber lower,
    etc). For any other topology we draw exactly the members render_segments()
    reports, coloured per-label, so a MacPherson shows a strut, a multi-link
    shows its links, a solid axle shows its Panhard rod — the real architecture.
    """
    drop = np.array([0, 0, ride_drop_mm], float)

    if _is_wishbone_hardpoints(corner):
        pts, s = _solved_corner_points(corner, ride_drop_mm)
        cam = getattr(corner, "static_camber", -1.5)
        # Named wishbone links -> fixed colours (matches the GEOMETRY 3D tab).
        segs = [
            (pts["upper_front_inner"], pts["upper_outer"], "Upper wishbone", COLORS["upper"], "upper"),
            (pts["upper_rear_inner"], pts["upper_outer"], "Upper wishbone", COLORS["upper"], "upper"),
            (pts["lower_front_inner"], pts["lower_outer"], "Lower wishbone", COLORS["lower"], "lower"),
            (pts["lower_rear_inner"], pts["lower_outer"], "Lower wishbone", COLORS["lower"], "lower"),
            (pts["lower_outer"], pts["upper_outer"], "Upright", COLORS["upright"], "upright"),
            (pts["tie_rod_inner"], pts["tie_rod_outer"], "Tie rod", COLORS["tie"], "tie"),
        ]
        po = pts.get("pushrod_outer"); rpv = pts.get("rocker_pivot")
        rpu = pts.get("rocker_pushrod"); rsp = pts.get("rocker_spring")
        spi = pts.get("spring_inner")
        if po is not None and rpu is not None:
            segs.append((po, rpu, "Pushrod", COLORS["push"], "push"))
        if rpv is not None and rpu is not None and rsp is not None:
            segs.append((rpv, rpu, "Rocker", COLORS["rocker"], "rocker"))
            segs.append((rpv, rsp, "Rocker", COLORS["rocker"], "rocker"))
        if rsp is not None and spi is not None:
            segs.append((rsp, spi, "Spring/damper", COLORS["spring"], "spring"))
        markers = [pts[k] for k in (
            "upper_front_inner", "upper_rear_inner", "lower_front_inner",
            "lower_rear_inner", "upper_outer", "lower_outer",
            "tie_rod_inner", "tie_rod_outer")]
        return dict(segments=segs, markers=markers,
                    wheel_center=pts["wheel_center"],
                    contact_patch=pts["contact_patch"], camber=cam)

    # ---- architecture-agnostic mechanism -------------------------------- #
    # `corner` quacks like GenericKinematics: render_segments(), named_points(),
    # static.wheel_center / contact_patch.
    raw = corner.render_segments()
    segs = []
    for p, q, label in raw:
        p = np.asarray(p, float) - drop
        q = np.asarray(q, float) - drop
        if label == "Wheel":
            continue  # the wheel hub line is drawn from wc/cp below
        color, base = _agnostic_color(label, color_registry)
        segs.append((p, q, label, color, base))
    named = corner.named_points()
    markers = [np.asarray(v, float) - drop for v in named.values()]
    st = corner.static
    wc = np.asarray(st.wheel_center, float) - drop
    cp = np.asarray(st.contact_patch, float) - drop
    cam = getattr(st, "camber", -1.5)
    return dict(segments=segs, markers=markers,
                wheel_center=wc, contact_patch=cp, camber=cam)


# --------------------------------------------------------------------------- #
#  Sizing laws
# --------------------------------------------------------------------------- #
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wing_span_chord(downforce_n, default_span, default_chord):
    """More downforce -> a visibly bigger wing. Monotonic around a 600 N ref."""
    if not downforce_n:
        return default_span, default_chord
    f = _clamp(downforce_n / 600.0, 0.45, 2.2)
    return (default_span * _clamp(f ** 0.3, 0.7, 1.4),
            default_chord * _clamp(f ** 0.5, 0.6, 1.8))


# --------------------------------------------------------------------------- #
#  Ledger helpers
# --------------------------------------------------------------------------- #
def _iface(led, name):
    if led is None:
        return None
    try:
        return led.get(name)
    except Exception:
        try:
            from .interfaces import SubsystemInterface
            if isinstance(led, dict):
                d = led.get("interfaces", {}).get(name)
                return SubsystemInterface.from_dict(d) if d else None
        except Exception:
            return None
    return None


def _g(it, attr, default=None):
    if it is None:
        return default
    v = getattr(it, attr, default)
    return default if v is None else v


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def build_full_car_figure(
    hp_front=None,
    vp=None,
    hp_rear=None,
    ledger=None,
    *,
    corner_front=None,
    corner_rear=None,
    topology_label: str | None = None,
    show_chassis=True, show_tires=True, show_floor=True,
    show_aero=False, show_powertrain=True, show_cooling=False,
    show_electrics=False, show_brakes=True, show_bodywork=True,
    highlight_subsystem: str | None = None,
    focus_subsystem: str | None = None,
    tire_width_mm: float = 180.0,
    part_overrides: dict | None = None,
    custom_parts: list | None = None,
    suppress_subsystems: set | None = None,
    suppress_parts: set | None = None,
    height: int = 720,
):
    """Assemble a live Baja-buggy 3D figure.

    The suspension reflects the chosen ARCHITECTURE. Pass the corner either as:
      * a double-wishbone `Hardpoints`  (via hp_front / hp_rear), or
      * any topology's solved kinematics (via corner_front / corner_rear) — a
        GenericKinematics-like object exposing render_segments(), named_points()
        and .static.wheel_center / .contact_patch.
    corner_* takes precedence over hp_* when both are given. This lets a
    MacPherson, multi-link, trailing-arm, solid-axle, twist-beam or free-form
    car render its real members instead of being forced into wishbones.

    PART OVERRIDES — user edits to size & position
    ----------------------------------------------
    `part_overrides` lets the user nudge the dimensions and location of any
    drawn part without changing the underlying engineering numbers. It is a
    dict keyed by the part's display name (exactly the `name=` each body is
    drawn with, e.g. "Engine + CVT", "Sidepod (cooling)", "Frame tube",
    "Main hoop", "Front hoop", "Tire", "Brake disc", "Driver",
    "Data logger", "Radiator core"). Each value is a dict with any of:

        dx, dy, dz : float   # translate the whole part, in mm (SAE axes)
        sx, sy, sz : float   # per-axis scale about the part's own centroid
        scale      : float   # uniform scale (applied if sx/sy/sz absent)

    The transform is applied at the single chokepoint every body passes
    through (the local `mesh`/`seg` helpers), so it covers every part — meshes
    and line members alike — and the per-subsystem bounding boxes used for
    click-to-zoom are computed AFTER the override, so the camera still frames
    the part where the user moved it. Missing keys default to no change
    (dx=dy=dz=0, scale=1), so an empty/None override leaves the car untouched.

    CUSTOM PARTS — "drop my part on the car"
    ----------------------------------------
    `custom_parts` lets a sub-team drop a part onto the car in REAL millimetres,
    straight off a spec sheet, with no scale-factor fiddling. It is a list of
    dicts, each:

        name : str            # label shown on the body and in the legend
        subsys : str          # which subsystem it belongs to (for colour +
                              #   click-to-zoom + spotlight); any of SUBSYSTEMS,
                              #   or None for a neutral grey "custom" body
        l_mm, w_mm, h_mm : float   # the part's real size: length(x) width(y) height(z)
        x_mm, y_mm, z_mm : float   # where its CENTRE sits in SAE car axes
                              #   (x: +forward from mid-wheelbase, y: +right of
                              #    centreline, z: +up from ground)
        shape : str           # "box" (default) or "cylinder" (l_mm = length
                              #   along x, w_mm = diameter)
        color : str           # optional hex; defaults to the subsystem colour

    Every custom part is a first-class body: it flows through the same `mesh`
    chokepoint as the built-in parts, so it honours part_overrides, the
    highlight spotlight, and — because its vertices are accrued under its
    subsystem — click-to-zoom frames it too. This is the path a powertrain lead
    uses to type "Radiator 289×124×34" and see it sit on the car immediately.
    """
    # Resolve the front/rear corner objects (architecture-agnostic).
    cf = corner_front if corner_front is not None else hp_front
    cr = corner_rear if corner_rear is not None else hp_rear
    if cf is None:
        cf = Hardpoints.default()
    if cr is None:
        cr = cf

    wb = float(getattr(vp, "wheelbase", 1550.0))
    tf = float(getattr(vp, "track_front", 1200.0))
    tr = float(getattr(vp, "track_rear", 1180.0))

    # If a real chassis CAD/estimate has been dropped in, re-base the whole car
    # on ITS footprint: wheels, wings, sidepods, motor etc. reposition to attach
    # to the real chassis, so the car reads as one coherent assembly (the "new
    # after chassis CAD upload" sketch) instead of dummies floating at the
    # generic envelope. Shift the fore/aft origin to the chassis centre too.
    _chassis_fp = None
    _x_origin = 0.0
    try:
        _chassis_fp = chassis_cad_footprint(vp, custom_parts, ledger=ledger)
    except Exception:
        _chassis_fp = None
    if _chassis_fp:
        wb = _chassis_fp["wb"]
        _tgt_track = _clamp(_chassis_fp["width"] + 520.0, 900.0, 1700.0)
        tf = tr = _tgt_track
        _x_origin = _chassis_fp["cx"]

    # Softer front spring -> more static sag -> body visibly lower. Cue, not a calc.
    kf = float(getattr(vp, "spring_rate_front", 35.0) or 35.0)
    ride_drop = _clamp((35.0 - kf) * 0.6, -12.0, 18.0)

    # Subsystems whose PROCEDURAL body is replaced by a user CAD/custom part:
    # skip drawing the stand-in loft/box so only the real geometry shows.
    _suppress = set(suppress_subsystems or ())
    # Individual part draw-names to suppress (per-PART replacement). A suppressed
    # subsystem also suppresses all of its catalog parts, for back-compat.
    _suppress_names = set(suppress_parts or ())
    for _k, _dn, _s, _c in PART_CATALOG:
        if _s in _suppress:
            _suppress_names.add(_dn)

    # Extract each axle's corner into a uniform, topology-independent description.
    color_registry = {}
    front_corner = _extract_corner(cf, ride_drop, color_registry)
    rear_corner = _extract_corner(cr, ride_drop, color_registry)

    y_center = 0.0
    scale_f = (tf / 2.0) / (abs(front_corner["contact_patch"][1] - y_center) or 1.0)
    scale_r = (tr / 2.0) / (abs(rear_corner["contact_patch"][1] - y_center) or 1.0)
    # Centre the axles on the chassis (or 0 when no chassis CAD is present), so a
    # real chassis pulls the wheels — and everything keyed off x_front/x_rear
    # (wings, sidepods, motor, nose, tail) — into alignment with it.
    x_front, x_rear = _x_origin + wb / 2.0, _x_origin - wb / 2.0

    fig = go.Figure()

    # ---- user part overrides (size & position) -------------------------- #
    # Resolve once: a part name -> (dx, dy, dz, sx, sy, sz). Scaling is about
    # the part's centroid so a part keeps its place while changing size; the
    # translate then moves it. Parts drawn in several pieces (a wing = elements
    # + endplates, a "Tire" = tire + rim across 4 corners) must scale about a
    # SHARED centroid, else the pieces fly apart — so we pin each part's scale
    # centre to the centroid of the FIRST batch of vertices seen under that
    # name, and reuse it for every later piece. Overrides are applied at the
    # mesh/seg chokepoint, before _accrue, so click-to-zoom frames the moved part.
    _ov = part_overrides or {}

    def _ov_for(name):
        o = _ov.get(name) if name else None
        if not o:
            return None
        dx = float(o.get("dx", 0.0) or 0.0)
        dy = float(o.get("dy", 0.0) or 0.0)
        dz = float(o.get("dz", 0.0) or 0.0)
        us = o.get("scale", 1.0)
        us = 1.0 if us is None else float(us)
        sx = float(o.get("sx", us) or us)
        sy = float(o.get("sy", us) or us)
        sz = float(o.get("sz", us) or us)
        if dx == dy == dz == 0.0 and sx == sy == sz == 1.0:
            return None
        return (dx, dy, dz, sx, sy, sz)

    _scale_centre: dict[str, np.ndarray] = {}
    # Parts drawn once per corner (4×): scale each instance about ITS OWN
    # centroid (so each tire/disc grows in place) rather than a shared car-wide
    # centre (which would splay the four apart). Translate still applies to all.
    _PER_INSTANCE = {"Tire", "Brake disc"}

    def _apply_ov(name, V):
        """Return V transformed by the named part's override (or V unchanged)."""
        spec = _ov_for(name)
        if spec is None:
            return V
        dx, dy, dz, sx, sy, sz = spec
        A = np.asarray(V, float)
        flat = A.reshape(-1, 3)
        if name in _PER_INSTANCE:
            c = flat.mean(axis=0)            # this instance's own centre
        else:
            c = _scale_centre.get(name)
            if c is None:
                c = flat.mean(axis=0)
                _scale_centre[name] = c
        out = (flat - c) * np.array([sx, sy, sz]) + c + np.array([dx, dy, dz])
        return out.reshape(A.shape)

    def op(subsys, base):
        if highlight_subsystem is None:
            return base
        return base if subsys == highlight_subsystem else base * 0.16

    def edge_op(subsys):
        if highlight_subsystem is None or subsys is None:
            return 1.0
        return 1.0 if subsys == highlight_subsystem else 0.14

    legend_done = set()
    _part_boxes = {}   # draw-name -> [lo(3), hi(3)] actual drawn box, for seamless
                       # replacement placement (CAD lands where placeholder was).

    # Per-subsystem point accumulator. Every body that belongs to a clickable
    # subsystem feeds its vertices here, so afterwards we know the bounding box
    # of each subsystem and can frame the camera on whichever one is clicked.
    subsys_pts: dict[str, list] = {}

    def _accrue(subsys, pts):
        if not subsys or pts is None:
            return
        bucket = subsys_pts.setdefault(subsys, [])
        bucket.extend(np.asarray(pts, float).reshape(-1, 3).tolist())

    def seg(p, q, color, w=5, name=None, group=None, subsys=None):
        if p is None or q is None:
            return
        # Override key: prefer the legend name, fall back to the group token so
        # unnamed members of a named part (hoop braces, wing mounts) move too.
        _ovk = name if (name and name in _ov) else (group if group in _ov else name)
        pq = _apply_ov(_ovk, np.array([p, q], float))
        p, q = pq[0], pq[1]
        _accrue(subsys, [p, q])
        fig.add_trace(go.Scatter3d(
            x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]],
            mode="lines", line=dict(color=color, width=w),
            opacity=edge_op(subsys), name=name, legendgroup=group,
            showlegend=name is not None, hoverinfo="skip",
            customdata=[subsys, subsys] if subsys else None))

    def mesh(verts, i, j, k, color, name, subsys, base_op=0.6, hover=None):
        # Record this part's actual box (its true drawn extent) keyed by name —
        # even if we then suppress it — so a replacement can land in EXACTLY the
        # same place its placeholder occupies. This is what makes CAD/sketch/
        # estimate parts and the procedural placeholders share one frame.
        try:
            _vv = _apply_ov(name, verts)
            _lo = _vv.min(axis=0); _hi = _vv.max(axis=0)
            if name in _part_boxes:
                _pb = _part_boxes[name]
                _pb[0] = np.minimum(_pb[0], _lo)
                _pb[1] = np.maximum(_pb[1], _hi)
            else:
                _part_boxes[name] = [_lo.copy(), _hi.copy()]
        except Exception:
            pass
        # A part replaced by a user CAD/sketch/estimate is suppressed by its
        # draw-name, so exactly that body disappears (not the whole subsystem).
        if name in _suppress_names:
            return
        once = name not in legend_done
        legend_done.add(name)
        verts = _apply_ov(name, verts)
        _accrue(subsys, verts)
        # customdata carries the clickable subsystem id on every vertex, so a
        # Streamlit selection event can read which part the user picked.
        cd = [subsys] * len(verts) if subsys else None
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=i, j=j, k=k,
            color=color, opacity=op(subsys, base_op), flatshading=True,
            name=name, showlegend=once, customdata=cd,
            hoverinfo="text" if hover else "skip", text=hover))

    def corner_name(base):
        if base in legend_done:
            return None
        legend_done.add(base)
        return base

    # ---- 1) suspension corners + tires + brake discs -------------------- #
    #  Each station reuses the SAME extracted corner description, transformed to
    #  its wheel position (mirror L/R, scale to axle track, shift fore/aft). The
    #  segments came from the chosen topology, so a MacPherson draws a strut, a
    #  multi-link draws its links, etc — the architecture is honoured everywhere.
    stations = [
        ("front", front_corner, scale_f, x_front, False),
        ("front", front_corner, scale_f, x_front, True),
        ("rear",  rear_corner,  scale_r, x_rear,  False),
        ("rear",  rear_corner,  scale_r, x_rear,  True),
    ]
    brake_tq = _g(_iface(ledger, "brakes"), "brake_torque_nm")

    def _xform(p, mirror, lat_scale, x_shift):
        return _corner_transform(p, mirror_y=mirror, lateral_scale=lat_scale,
                                 x_shift=x_shift, y_center_ref=y_center)

    for axle, corner, lat_scale, x_shift, mirror in stations:
        # Baja: front corners belong to Front Suspension + Steering, rear corners
        # to Rear Suspension, so spotlight / click-to-zoom resolve to the right
        # owning subteam instead of one merged "suspension".
        _susgrp = "front-suspension" if axle == "front" else "rear-suspension"
        # draw every member the topology reported
        for p, q, label, color, group in corner["segments"]:
            pT = _xform(p, mirror, lat_scale, x_shift)
            qT = _xform(q, mirror, lat_scale, x_shift)
            seg(pT, qT, color, 5, corner_name(label), group, _susgrp)

        wc = _xform(corner["wheel_center"], mirror, lat_scale, x_shift)
        cp = _xform(corner["contact_patch"], mirror, lat_scale, x_shift)
        # wheel hub line
        seg(cp, wc, COLORS["wheel"], 3, corner_name("Wheel hub"), "wheel", _susgrp)

        cam = np.deg2rad(corner["camber"])
        sign = -1.0 if mirror else 1.0
        axis = np.array([0.0, sign * np.cos(cam), np.sin(cam)])
        radius = abs(wc[2] - cp[2]) or 228.0
        if show_tires:
            tv, ti, tj, tk = _cylinder(wc, axis, radius, tire_width_mm, n=30)
            mesh(tv, ti, tj, tk, COLORS["tire"], "Tire", _susgrp,
                 base_op=0.95)
            # Rim: a slightly inset, lighter disc so the wheel reads as a wheel,
            # not a black drum — sits at ~62% of tire radius on the outboard face.
            rim_r = radius * 0.62
            rv, ri, rj, rk = _cylinder(wc, axis, rim_r, tire_width_mm * 0.9, n=24)
            mesh(rv, ri, rj, rk, COLORS["rim"], "Tire", _susgrp,
                 base_op=0.98)
        if show_brakes:
            disc_r = (radius * _clamp(0.62 + (brake_tq or 0) / 4000.0, 0.5, 0.85)
                      if brake_tq else radius * 0.62)
            dv, di, dj, dk = _cylinder(wc, axis, disc_r,
                                       max(8.0, tire_width_mm * 0.07), n=26)
            hv = ("Brake disc · r≈%.0f mm" % disc_r
                  + (" (sized from %.0f N·m)" % brake_tq if brake_tq else ""))
            mesh(dv, di, dj, dk, COLORS["brake"], "Brake disc", _susgrp,
                 base_op=0.9, hover=hv)

        mk = [_xform(m, mirror, lat_scale, x_shift) for m in corner["markers"]]
        if mk:
            _accrue(_susgrp, mk)
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in mk], y=[p[1] for p in mk], z=[p[2] for p in mk],
                mode="markers", marker=dict(size=3, color=COLORS["point"]),
                opacity=edge_op(_susgrp), showlegend=False, hoverinfo="skip",
                customdata=[_susgrp] * len(mk)))

    # z-extent + tire radius derived from the extracted corners (any topology).
    z_all = []
    for corner in (front_corner, rear_corner):
        for p, q, *_ in corner["segments"]:
            z_all += [p[2], q[2]]
        z_all += [corner["wheel_center"][2], corner["contact_patch"][2]]
    z_lo, z_hi = (min(z_all), max(z_all)) if z_all else (0.0, 300.0)
    tire_r = abs(front_corner["wheel_center"][2]
                 - front_corner["contact_patch"][2]) or 228.0
    inner_y_f = tf / 2.0 - tire_width_mm - 40
    inner_y_r = tr / 2.0 - tire_width_mm - 40

    # ---- 2) chassis: Baja SAE tubular space frame (roll cage) ----------- #
    #  Baja is an open tube-frame buggy (cf. the reference photo): a welded
    #  4130 space frame — lower floor rails, a cockpit bay, a curved MAIN hoop
    #  behind the driver's head, a front (A-pillar) hoop, lateral roof/floor
    #  cross-members, diagonal side-impact (door) bars, and front/rear nodes
    #  the suspension and engine hang off. NO body panels, NO pointed nosecone,
    #  NO wings. The driver's helmet shows in the open cockpit.
    #
    #  tub_w / tub_top / tub_bot / cz / hzz are still computed here because the
    #  cooling/powertrain/electrics sections downstream place bodies relative to
    #  the cockpit box.
    tub_w = _clamp(min(inner_y_f, inner_y_r) * 1.25, 180, 360)
    tub_bot = max(z_lo * 0.5, tire_r * 0.16)
    tub_top = tub_bot + _clamp(tire_r * 1.15, 220, 420)
    cz = (tub_top + tub_bot) / 2
    hzz = (tub_top - tub_bot) / 2
    if show_bodywork:
        ch_it = _iface(ledger, "chassis")
        tube_r = max(tire_r * 0.058, 12.0)        # ~25–32 mm OD frame tubing
        node_r = tube_r * 1.25
        hv = "Tubular space frame (4130)"
        if _g(ch_it, "mass_kg"):
            hv += " · %.1f kg" % _g(ch_it, "mass_kg")

        # Longitudinal stations of the frame (front bulkhead -> rear).
        fb_x = x_front + tire_r * 1.30          # front bulkhead (feet)
        dash_x = x_front - wb * 0.02            # front (dash) hoop
        seat_x = x_front - wb * 0.30            # main hoop (behind seat)
        rear_x = x_rear + tire_r * 0.10         # rear frame node
        hw = tub_w                              # half-width of the frame
        z0 = tub_bot                            # lower rail height
        z1 = tub_top                            # upper rail / hoop shoulder

        def frame_tube(p0, p1, r=None, name="Frame tube", hint=None):
            t = _tube(p0, p1, radius=(r or tube_r), n=10)
            mesh(t[0], t[1], t[2], t[3], COLORS["frame"], name, "chassis",
                 base_op=0.95, hover=hint)

        def frame_node(p):
            s = _sphere(p, node_r, n=12)
            mesh(s[0], s[1], s[2], s[3], COLORS["frame"], "Frame node",
                 "chassis", 0.96)

        # Lower floor rails (left & right), front bulkhead to rear node.
        for sgn in (-1, 1):
            y = sgn * hw
            frame_tube([fb_x, y, z0], [seat_x, y, z0], hint=hv)
            frame_tube([seat_x, y, z0], [rear_x, y, z0])
        # Lower cross-members (front, mid, rear).
        for xx in (fb_x, seat_x, rear_x):
            frame_tube([xx, -hw, z0], [xx, hw, z0])
        # Upper side rails along the cockpit (dash hoop shoulder -> main hoop).
        for sgn in (-1, 1):
            y = sgn * hw
            frame_tube([dash_x, y, z1 * 0.86], [seat_x, y, z1 * 0.92])
        # Diagonal side-impact (door) bars across the cockpit opening.
        for sgn in (-1, 1):
            y = sgn * hw
            frame_tube([dash_x, y, z0], [seat_x, y, z1 * 0.7], r=tube_r * 0.9)

        # MAIN roll hoop: a tall curved tube arching above/behind the driver.
        hoop_top = z1 + tire_r * 0.55
        main_hoop = [
            [seat_x, -hw, z0],
            [seat_x, -hw, cz],
            [seat_x - tire_r * 0.12, -hw * 0.6, hoop_top * 0.94],
            [seat_x - tire_r * 0.15, 0, hoop_top],
            [seat_x - tire_r * 0.12, hw * 0.6, hoop_top * 0.94],
            [seat_x, hw, cz],
            [seat_x, hw, z0],
        ]
        mh = _swept_tube(main_hoop, radius=tube_r, n=10)
        mesh(mh[0], mh[1], mh[2], mh[3], COLORS["hoop"], "Main hoop",
             "chassis", 0.96, "Main roll hoop")

        # Rear bracing: main hoop shoulders down to the rear frame nodes.
        for sgn in (-1, 1):
            frame_tube([seat_x - tire_r * 0.12, sgn * hw * 0.6, hoop_top * 0.94],
                       [rear_x, sgn * hw, z0], r=tube_r * 0.9,
                       name="Rear brace")
        # Roof cross tube tying the top of the main hoop laterally.
        frame_tube([seat_x - tire_r * 0.15, -hw * 0.6, hoop_top * 0.94],
                   [seat_x - tire_r * 0.15, hw * 0.6, hoop_top * 0.94],
                   r=tube_r * 0.85)

        # FRONT (dash / A-pillar) hoop, smaller, ahead of the cockpit.
        fh_top = z1 + tire_r * 0.12
        front_hoop = [
            [dash_x, -hw, z0],
            [dash_x, -hw, cz],
            [dash_x, -hw * 0.55, fh_top],
            [dash_x, 0, fh_top + tire_r * 0.05],
            [dash_x, hw * 0.55, fh_top],
            [dash_x, hw, cz],
            [dash_x, hw, z0],
        ]
        fhm = _swept_tube(front_hoop, radius=tube_r * 0.92, n=9)
        mesh(fhm[0], fhm[1], fhm[2], fhm[3], COLORS["frame"], "Front hoop",
             "chassis", 0.95, "Front (A-pillar) hoop")

        # Front-bay tubes: dash hoop forward to the front bulkhead (feet box).
        for sgn in (-1, 1):
            y = sgn * hw
            frame_tube([dash_x, y, z0], [fb_x, y, z0])
            frame_tube([dash_x, y, cz], [fb_x, y, z0 + hzz * 0.4],
                       r=tube_r * 0.85)
        frame_tube([fb_x, -hw, z0 + hzz * 0.4], [fb_x, hw, z0 + hzz * 0.4],
                   r=tube_r * 0.85, name="Front bulkhead")
        # A couple of frame nodes where the suspension picks up loads read.
        for xx in (fb_x, dash_x, seat_x, rear_x):
            for sgn in (-1, 1):
                frame_node([xx, sgn * hw, z0])

        # Driver: a helmet sphere seated in the open cockpit.
        cockpit_x = (dash_x + seat_x) / 2
        helmet_r = tub_w * 0.42
        helmet_z = z1 + helmet_r * 0.35
        hv_e = _sphere([cockpit_x, 0, helmet_z], helmet_r, n=16)
        mesh(hv_e[0], hv_e[1], hv_e[2], hv_e[3], COLORS["helmet"],
             "Driver", "chassis", 0.98, "Driver (helmet)")
        seg([cockpit_x - helmet_r, 0, helmet_z],
            [cockpit_x + helmet_r, 0, helmet_z],
            COLORS["helmet_band"], 6, None, "helmet", "chassis")


    # ---- 3) aerodynamics: multi-element wings + endplates --------------- #
    #  FSAE-style: a wide multi-element FRONT wing low and ahead of the front
    #  axle on endplates, and a tall multi-element REAR wing on twin endplates
    #  behind the rear axle. Element count/size still scale with declared
    #  downforce, so the aero team's number visibly grows the wing.
    if show_aero:
        aero_it = _iface(ledger, "aerodynamics")
        df = _g(aero_it, "downforce_n_at_v")
        df_n = df[0] if isinstance(df, (tuple, list)) and df else None

        def _elements(df_n):
            # more downforce -> more elements (2..4) and a touch more chord
            if not df_n:
                return 3
            return int(_clamp(2 + df_n / 500.0, 2, 4))

        # ---- FRONT wing: low, ahead of the front axle --------------------
        fw_span, fw_chord = _wing_span_chord(df_n, tf * 0.98, tire_r * 0.62)
        fw_x = x_front + tire_r * 1.62
        fw_z = tire_r * 0.32
        n_fe = _elements(df_n)
        hint_f = "Front wing" + (" (sized from %.0f N)" % df_n if df_n else "")
        for e in range(n_fe):
            ex = fw_x - e * fw_chord * 0.42
            ez = fw_z + e * fw_chord * 0.22
            ch = fw_chord * (0.7 + 0.12 * e)
            wv = _wing_element(ex, 0, ez, ch, fw_span,
                               thickness=0.11, aoa_deg=-8 - 4 * e)
            mesh(wv[0], wv[1], wv[2], wv[3], COLORS["wing"], "Front wing",
                 "aerodynamics", 0.9, hint_f if e == 0 else None)
        # endplates by the front tires
        for sgn in (-1, 1):
            ev, ei, ej, ek = _box(fw_x - fw_chord * 0.3, sgn * fw_span / 2,
                                   fw_z + fw_chord * 0.15,
                                   fw_chord * 1.5, 6, fw_chord * 1.1)
            mesh(ev, ei, ej, ek, COLORS["endplate"], "Front wing",
                 "aerodynamics", 0.9)
        # nose-to-wing pylons
        _mount_y = (tub_w * 0.3) if show_bodywork else (fw_span * 0.12)
        for sgn in (-1, 1):
            seg([fw_x, sgn * fw_span * 0.18, fw_z],
                [fw_x - fw_chord, sgn * _mount_y, fw_z + tire_r * 0.4],
                COLORS["wing_edge"], 4, None, "fw_mount", "aerodynamics")

        # ---- REAR wing: tall, behind the rear axle -----------------------
        rw_span, rw_chord = _wing_span_chord(df_n, tr * 0.82, tire_r * 0.72)
        rw_x = x_rear - tire_r * 1.55
        rw_z = z_hi + tire_r * 1.15
        n_re = _elements(df_n)
        hint_r = "Rear wing" + (" (sized from %.0f N)" % df_n if df_n else "")
        for e in range(n_re):
            ex = rw_x + e * rw_chord * 0.4
            ez = rw_z + e * rw_chord * 0.34
            ch = rw_chord * (0.8 + 0.1 * e)
            wv = _wing_element(ex, 0, ez, ch, rw_span,
                               thickness=0.12, aoa_deg=-12 - 5 * e)
            mesh(wv[0], wv[1], wv[2], wv[3], COLORS["wing"], "Rear wing",
                 "aerodynamics", 0.9, hint_r if e == 0 else None)
        # twin endplates
        for sgn in (-1, 1):
            ev, ei, ej, ek = _box(rw_x + rw_chord * 0.3, sgn * rw_span / 2,
                                   rw_z + rw_chord * 0.5,
                                   rw_chord * 2.0, 8, rw_chord * 2.2)
            mesh(ev, ei, ej, ek, COLORS["endplate"], "Rear wing",
                 "aerodynamics", 0.92)
        # rear-wing support struts up from the gearbox/tail
        for sgn in (-1, 1):
            seg([rw_x + rw_chord * 0.3, sgn * rw_span * 0.18, rw_z - rw_chord * 0.4],
                [rw_x + rw_chord * 1.2, sgn * rw_span * 0.12, z_hi * 0.7],
                COLORS["wing_edge"], 5, None, "rw_mount", "aerodynamics")

    # ---- 4) cooling: sidepods ------------------------------------------ #
    if show_cooling:
        cool_it = _iface(ledger, "cooling")
        airflow = _g(cool_it, "cooling_airflow_cms")
        heat = _g(cool_it, "heat_reject_w")
        f = _clamp((airflow or 0.4) / 0.4, 0.5, 2.2)
        pod_len = wb * 0.34 * _clamp(f ** 0.4, 0.7, 1.5)
        pod_h = tire_r * 0.7 * _clamp(f ** 0.4, 0.7, 1.4)
        pod_w = 110 * _clamp(f ** 0.5, 0.7, 1.6)
        pod_x = -wb * 0.05
        for sgn in (-1, 1):
            pod_y = sgn * (min(inner_y_f, inner_y_r) * 0.95)
            v, i, j, k = _box(pod_x, pod_y, tire_r * 0.65, pod_len, pod_w, pod_h)
            hv = "Sidepod / radiator duct"
            if airflow:
                hv += " (sized from %.2f m³/s)" % airflow
            if heat:
                hv += " · rejects %.0f W" % heat
            mesh(v, i, j, k, COLORS["sidepod"], "Sidepod (cooling)", "cooling", 0.7, hv)
            rv, ri, rj, rk = _box(pod_x + pod_len / 2, pod_y, tire_r * 0.65,
                                  8, pod_w * 0.8, pod_h * 0.8)
            mesh(rv, ri, rj, rk, COLORS["radiator"], "Radiator core", "cooling", 0.85)

    # ---- 5) drivetrain: Baja engine + CVT + half-shafts ----------------- #
    #  Baja SAE is single-make combustion: a ~10 hp Briggs & Stratton sits at the
    #  rear with a CVT (primary on the crank, secondary on the gearbox input) and
    #  a reduction gearbox driving the rear wheels through half-shafts. Sizing
    #  tracks the declared power/torque so the drivetrain team's number drives
    #  the body, exactly as the EV motor used to.
    if show_powertrain:
        pt_it = _iface(ledger, "drivetrain")
        pkw = _g(pt_it, "peak_power_kw")
        ptq = _g(pt_it, "peak_torque_nm")
        ex, ey, ez = _g(pt_it, "env_x_mm"), _g(pt_it, "env_y_mm"), _g(pt_it, "env_z_mm")
        if ex and ey and ez:
            blk_l, blk_w, blk_h = ex, ey, ez
            sized = "(declared envelope)"
        else:
            # Baja spec engine ≈ 7.5 kW; scale gently around that.
            f = _clamp((pkw or 7.5) / 7.5, 0.6, 1.8)
            blk_l = wb * 0.15 * _clamp(f ** 0.4, 0.7, 1.4)
            blk_w = min(inner_y_r, 150) * 1.1
            blk_h = tire_r * 0.85 * _clamp(f ** 0.3, 0.8, 1.3)
            sized = ("(sized from %.1f kW)" % pkw if pkw else "")
        mot_x = x_rear + tire_r * 1.05
        # Engine block: an upright box (single-cylinder IC engine, not a motor).
        ev, ei, ej, ek = _box(mot_x, 0, tire_r * 0.9, blk_l, blk_w * 0.7, blk_h)
        hv = "Engine (IC) " + sized + (" · %.0f N·m" % ptq if ptq else "")
        mesh(ev, ei, ej, ek, COLORS["engine"], "Engine + CVT",
             "drivetrain", 0.92, hv)
        # CVT housing: a flat cylinder on the driver's-left of the block.
        cc = _cylinder([mot_x, blk_w * 0.55, tire_r * 0.9], [0, 1, 0],
                       radius=blk_h * 0.5, length=blk_w * 0.5, n=22)
        mesh(cc[0], cc[1], cc[2], cc[3], COLORS["motor"], "Engine + CVT",
             "drivetrain", 0.9, "CVT (primary + secondary)")
        # Half-shafts to the rear wheels.
        for sgn in (-1, 1):
            seg([mot_x, sgn * blk_w * 0.45, tire_r * 0.9],
                [x_rear, sgn * tr / 2 * 0.78, tire_r],
                "#8d99a6", 5, None, "drive", "drivetrain")

    # ---- 6) (removed) EV accumulator — Baja has no HV pack -------------- #
    #  The original FSAE-EV build drew a tractive accumulator here. Baja SAE
    #  runs a small LV battery for ignition/DAQ only, which lives in the
    #  data-acquisition / electrics packaging rather than as a structural body,
    #  so no large box is drawn.
    if False and show_electrics:
        el_it = _iface(ledger, "electrics")
        ex, ey, ez = _g(el_it, "env_x_mm"), _g(el_it, "env_y_mm"), _g(el_it, "env_z_mm")
        emass, pwr = _g(el_it, "mass_kg"), _g(el_it, "power_draw_w")
        bl = bw = bh = 0
        sized = ""
        if ex and ey and ez:
            bl, bw, bh, sized = ex, ey, ez, "(declared envelope)"
        elif emass:
            side = (_clamp(emass, 2, 40) * 1.6e6) ** (1 / 3)
            bl, bw, bh = side * 1.4, side * 1.1, side * 0.7
            sized = "(sized from %.1f kg)" % emass
        else:
            bl, bw, bh = wb * 0.16, min(inner_y_r, 160) * 1.2, tire_r * 0.55
            sized = "(placeholder)"
        if bl:
            bx = x_rear + tire_r * 2.6
            v, i, j, k = _box(bx, 0, tire_r * 0.55, bl, bw, bh)
            hv = "Battery " + sized + (" · %.0f W" % pwr if pwr else "")
            mesh(v, i, j, k, COLORS["battery"], "Accumulator", "drivetrain", 0.85, hv)

    # ---- 7) data-acquisition: logger pod ------------------------------- #
    daq_it = _iface(ledger, "data-acquisition")
    _daq_mass = _g(daq_it, "mass_kg") if daq_it is not None else None
    v, i, j, k = _box(x_front - wb * 0.1, -tf * 0.18, tire_r * 1.05, 80, 60, 40)
    _daq_hv = ("Data-acquisition logger · %.1f kg" % _daq_mass if _daq_mass
               else "Data-acquisition logger (placeholder — declare mass in INTEGRATION)")
    mesh(v, i, j, k, COLORS["logger"], "Data logger", "data-acquisition", 0.85, _daq_hv)

    # ---- 8) CG marker from mass roll-up (ledger + custom parts) --------- #
    # The gold CG diamond reflects EVERYTHING declared: each subsystem's mass/CG
    # from the ledger PLUS any custom/CAD part that carries a mass. As a team
    # replaces stand-ins with real parts (each with its own mass), the CG slides
    # to the real number — the whole car re-balances as the build firms up.
    cg_h = float(getattr(vp, "cg_height", 0.0) or 0.0)
    wdist = float(getattr(vp, "weight_dist_front", 0.5) or 0.5)
    cg_x = x_rear + wdist * (x_front - x_rear)
    cg_y = 0.0
    cg_label = "CG (params)"
    _cg_terms = []   # (mass, x_car, y_car, z) tuples in car SAE frame
    _ledger_mass = 0.0
    if ledger is not None:
        try:
            roll = ledger.mass_rollup()
            if roll.get("cg_mm"):
                gx, gy, gz = roll["cg_mm"]
                _ledger_mass = float(roll.get("total_kg", 0.0) or 0.0)
                # ledger gx is +rearward from front axle -> car frame x is x_front-gx
                _cg_terms.append((_ledger_mass, x_front - gx, gy, gz))
        except Exception:
            pass
    # Fold in custom/CAD parts that declare a mass, at their placed centre.
    _cust_mass = 0.0
    for _cpm in (custom_parts or []):
        try:
            _m = float(_cpm.get("mass_kg", 0) or 0)
            if _m <= 0:
                continue
            _cx2 = float(_cpm.get("x_mm", 0) or 0)
            _cy2 = float(_cpm.get("y_mm", 0) or 0)
            _cz2 = float(_cpm.get("z_mm", 0) or 0)
            # If this part is auto-fitted, its drawn centre is the envelope home.
            if _cpm.get("fit_to_envelope") and _cpm.get("subsys") not in (None, "(custom / unassigned)"):
                try:
                    _tg = suggest_part_geometry(vp, _cpm["subsys"], ledger=ledger)
                    _cx2, _cy2, _cz2 = _tg["x_mm"], _tg["y_mm"], _tg["z_mm"]
                except Exception:
                    pass
            _cg_terms.append((_m, _cx2, _cy2, _cz2))
            _cust_mass += _m
        except Exception:
            continue
    if _cg_terms:
        _tot = sum(t[0] for t in _cg_terms) or 1.0
        cg_x = sum(t[0] * t[1] for t in _cg_terms) / _tot
        cg_y = sum(t[0] * t[2] for t in _cg_terms) / _tot
        cg_h = sum(t[0] * t[3] for t in _cg_terms) / _tot
        if _cust_mass > 0 and _ledger_mass > 0:
            cg_label = "CG (%.0f kg: %.0f declared + %.0f parts)" % (
                _tot, _ledger_mass, _cust_mass)
        elif _cust_mass > 0:
            cg_label = "CG (%.0f kg from parts)" % _tot
        else:
            cg_label = "CG (declared %.0f kg)" % _tot
    if cg_h > 0:
        fig.add_trace(go.Scatter3d(
            x=[cg_x], y=[cg_y], z=[cg_h], mode="markers+text",
            marker=dict(size=8, color=COLORS["cg"], symbol="diamond"),
            text=[cg_label], textposition="top center",
            textfont=dict(color=COLORS["cg"], size=11),
            name="Centre of gravity", hoverinfo="text"))

    # ---- ground plane -------------------------------------------------- #
    if show_floor:
        pad = max(tf, tr) * 0.8
        xs2 = [x_rear - tire_r * 2.0 - pad, x_front + tire_r * 2.2 + pad]
        ys2 = [-max(tf, tr) / 2 - pad, max(tf, tr) / 2 + pad]
        gx, gy = np.meshgrid(xs2, ys2)
        fig.add_trace(go.Surface(
            x=gx, y=gy, z=np.zeros_like(gx) - ride_drop, showscale=False,
            opacity=0.22, colorscale=[[0, COLORS["floor"]], [1, COLORS["floor"]]],
            hoverinfo="skip", name="Ground", showlegend=False))

    # ---- 9) custom parts: user-dropped bodies in real millimetres ------- #
    # A sub-team can drop any part onto the car straight off a spec sheet: real
    # L×W×H in mm at a real centre, no scale factors. Each becomes a first-class
    # body through the same `mesh` chokepoint, so it inherits highlight dimming,
    # part_overrides and — via _accrue under its subsystem — click-to-zoom.
    # A part flagged `provisional` is a stand-in for a part whose CAD hasn't
    # arrived yet: it draws faint and hatched-amber so nobody mistakes a guess
    # for a confirmed body, but still lets dependent packaging work continue.
    _cyl_default = None
    for cp in (custom_parts or []):
        try:
            nm = str(cp.get("name") or "Custom part").strip() or "Custom part"
            sub = cp.get("subsys") or None
            if sub == "(custom / unassigned)":
                sub = None
            l = float(cp.get("l_mm", 0) or 0)
            w = float(cp.get("w_mm", 0) or 0)
            h = float(cp.get("h_mm", 0) or 0)
            cx = float(cp.get("x_mm", 0) or 0)
            cy = float(cp.get("y_mm", 0) or 0)
            cz = float(cp.get("z_mm", 0) or 0)
            _has_mesh_early = bool(cp.get("mesh") and cp["mesh"].get("verts"))
            if not _has_mesh_early and (
                    l <= 0 or w <= 0 or (h <= 0 and cp.get("shape", "box") == "box")):
                continue
            prov = bool(cp.get("provisional"))
            mesh_payload = cp.get("mesh")
            has_mesh = bool(mesh_payload and mesh_payload.get("verts")
                            and mesh_payload.get("faces"))
            if prov:
                # A waiting-on-CAD stand-in: amber, see-through, clearly a guess.
                col = cp.get("color") or COLORS["cg"]
                base_op = 0.30
                nm_draw = nm if nm.endswith("(awaiting CAD)") else nm + " (awaiting CAD)"
                hov = "%s — PROVISIONAL stand-in, %.0f×%.0f×%.0f mm @ (x %.0f, y %.0f, z %.0f)" % (
                    nm, l, w, h, cx, cy, cz)
            else:
                # CAD mesh parts glow neon blue so real geometry is unmistakable
                # against the matte procedural placeholders. Sketch/estimate boxes
                # keep their subsystem hue.
                if has_mesh:
                    col = cp.get("color") or COLORS["cad_neon"]
                else:
                    col = cp.get("color") or COLORS.get(sub_color_key(sub), COLORS["custom"])
                base_op = 0.97 if has_mesh else 0.82
                nm_draw = nm
                _kind = "CAD mesh" if has_mesh else "%.0f×%.0f×%.0f mm" % (l, w, h)
                hov = "%s — %s @ (x %.0f, y %.0f, z %.0f)" % (nm, _kind, cx, cy, cz)
            shape = cp.get("shape", "box")
            if has_mesh:
                # Draw the ACTUAL imported geometry, oriented + placed on the car.
                faces = np.asarray(mesh_payload["faces"], int)
                _mesh_scale = float(cp.get("mesh_scale", 1.0) or 1.0)
                _cx, _cy, _cz = cx, cy, cz
                _axis_perm = None
                _axis_map = cp.get("axis_map", "auto")
                _raw = np.asarray(mesh_payload.get("size_mm") or [l, w, h], float)

                # ORIENTATION REFERENCE: the slot shape we align the part TO. Use
                # the replaced placeholder's actual box if we're hiding it, else
                # the chosen snap part's anchor. Available even when NOT auto-
                # sizing, so auto-orient still rotates a real-size part correctly.
                _ref_dims = None
                _ref_ctr = None
                _pk = cp.get("replaces_part") or cp.get("snap_part")
                _dn = cp.get("replaces_drawname")
                try:
                    if _dn and _dn in _part_boxes:
                        _lo, _hi = _part_boxes[_dn]
                        _ref_dims = np.asarray(_hi, float) - np.asarray(_lo, float)
                        _ref_ctr = (np.asarray(_hi, float) + np.asarray(_lo, float)) / 2.0
                    elif _pk:
                        _ta = part_anchor(vp, _pk, ledger=ledger)
                        _ref_dims = np.array([_ta["l_mm"], _ta["w_mm"], _ta["h_mm"]], float)
                        _ref_ctr = np.array([_ta["x_mm"], _ta["y_mm"], _ta["z_mm"]], float)
                    elif sub:
                        _ta = suggest_part_geometry(vp, sub, ledger=ledger)
                        _ref_dims = np.array([_ta["l_mm"], _ta["w_mm"], _ta["h_mm"]], float)
                        _ref_ctr = np.array([_ta["x_mm"], _ta["y_mm"], _ta["z_mm"]], float)
                except Exception:
                    _ref_dims = _ref_ctr = None

                # AUTO-ORIENT: clean 90° axis permutation so the part's longest
                # side lines up with the slot's longest side. Never a stretch, so
                # never distorted — and it fixes the "stood vertical" rotation.
                if _axis_map == "auto" and _ref_dims is not None:
                    src_order = np.argsort(_raw)         # short, mid, long
                    tgt_order = np.argsort(_ref_dims)
                    perm = [0, 0, 0]
                    for rank in range(3):
                        perm[tgt_order[rank]] = int(src_order[rank])
                    _axis_perm = tuple(perm)
                    _ext = _raw[list(_axis_perm)]
                else:
                    V0 = _orient_part_mesh(
                        mesh_payload["verts"], axis_map=_axis_map,
                        yaw_deg=float(cp.get("yaw_deg", 0.0) or 0.0),
                        scale=1.0, centre=(0.0, 0.0, 0.0))
                    _ext = V0.max(axis=0) - V0.min(axis=0)

                # SIZE + PLACE.
                if cp.get("fit_to_envelope") and _ref_dims is not None:
                    # Uniform auto-size into the slot, and snap to the slot centre.
                    # Because auto-orient has already matched the part's axes to the
                    # slot's (longest->longest), the per-axis ratios are comparable.
                    # "fit"  : scale so the part's LONGEST axis matches the slot's
                    #          matching axis — fills the primary extent (a chassis
                    #          spans the wheelbase) without distortion.
                    # "fill" : same idea but allow growing to the largest ratio so
                    #          even the stubbier axes reach the envelope.
                    try:
                        _so = np.argsort(_ext)              # short, mid, long (src)
                        _to = np.argsort(_ref_dims)         # short, mid, long (slot)
                        # Pair the part's axes with the slot's by rank.
                        _paired = [(_ref_dims[_to[r]] /
                                    (_ext[_so[r]] if _ext[_so[r]] > 1e-6 else 1.0))
                                   for r in range(3)]
                        if cp.get("fit_mode") == "fill":
                            # Fill the primary extent (longest axis); may overflow
                            # the slot on stubbier axes — for parts that should look
                            # substantial along their main direction.
                            _factor = float(_paired[2])
                        else:
                            # Balanced default: the MIDDLE ratio. Big enough to read
                            # as the real part, but won't blow far past the slot on
                            # the tight axes. True proportions always preserved.
                            _factor = float(_paired[1])
                        _mesh_scale = _mesh_scale * _factor
                        if _ref_ctr is not None:
                            _cx, _cy, _cz = (float(_ref_ctr[0]), float(_ref_ctr[1]),
                                             float(_ref_ctr[2]))
                    except Exception:
                        pass
                # else: keep the real CAD size × mesh_scale, at the user's x/y/z.
                V = _orient_part_mesh(
                    mesh_payload["verts"],
                    axis_map=_axis_map,
                    yaw_deg=float(cp.get("yaw_deg", 0.0) or 0.0),
                    scale=_mesh_scale,
                    centre=(_cx, _cy, _cz), axis_perm=_axis_perm)
                mesh(V, faces[:, 0], faces[:, 1], faces[:, 2],
                     col, nm_draw, sub, base_op, hov)
            elif shape == "cylinder":
                v, ii, jj, kk = _cylinder((cx, cy, cz), (1, 0, 0),
                                          radius=w / 2.0, length=l)
                mesh(v, ii, jj, kk, col, nm_draw, sub, base_op, hov)
            else:
                v, ii, jj, kk = _box(cx, cy, cz, l, w, h)
                mesh(v, ii, jj, kk, col, nm_draw, sub, base_op, hov)
            # On a box stand-in, also draw its wireframe edges so its true extent
            # is legible through the transparency — the packaging team is reading
            # a box, and a faint mesh alone is hard to judge.
            if prov and not has_mesh and shape != "cylinder":
                E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
                for a, b in E:
                    seg(v[a], v[b], col, w=3, subsys=sub)
        except Exception:
            # A malformed custom part must never take down the whole car view.
            continue

    # ---- camera: zoom to the focused subsystem, if one is clicked ------- #
    # When focus_subsystem is set we re-aim the camera at that part's bounding
    # box centre and pull the eye in proportionally, so clicking a part reads as
    # an automatic zoom. With no focus we keep the standard wide establishing
    # shot of the whole car.
    # uirevision token: constant while the focus is unchanged (so the user's
    # rotation is preserved across reruns), and distinct per focused part (so a
    # new click is allowed to re-aim the camera). "wide" is the no-focus shot.
    camera_revision = "wide"
    scene_camera = dict(eye=dict(x=1.8, y=-1.7, z=1.05))
    if focus_subsystem and subsys_pts.get(focus_subsystem):
        camera_revision = "focus:%s" % focus_subsystem
        pts = np.asarray(subsys_pts[focus_subsystem], float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ctr = (lo + hi) / 2.0

        # Aspect mode is "data", so camera coordinates are normalised against the
        # full scene span on each axis. Express the focus centre in that space and
        # bring the eye close along the standard viewing direction.
        all_pts = np.asarray([p for b in subsys_pts.values() for p in b], float)
        smin, smax = all_pts.min(axis=0), all_pts.max(axis=0)
        span = np.where((smax - smin) == 0, 1.0, (smax - smin))
        c_norm = (ctr - (smin + smax) / 2.0) / span  # centred, normalised

        # how big the part is relative to the whole car -> how hard we zoom
        part_span = (hi - lo)
        frac = float(np.clip(np.max(part_span / span), 0.04, 0.9))
        dist = 0.55 + frac * 1.4  # closer for small parts, backed off for big

        dir_unit = np.array([1.0, -0.95, 0.6])
        dir_unit = dir_unit / np.linalg.norm(dir_unit)
        eye = c_norm + dir_unit * dist
        scene_camera = dict(
            center=dict(x=float(c_norm[0]), y=float(c_norm[1]), z=float(c_norm[2])),
            eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        # Make rotating the car the primary mouse gesture: left-drag orbits the
        # scene (turntable keeps "up" sensible), scroll zooms, right-drag pans.
        dragmode="turntable",
        # Preserve the user's manual orbit/zoom across Streamlit reruns. Plotly
        # keeps the current camera as long as uirevision is unchanged; we only
        # bump it (via camera_revision) when we deliberately re-aim the camera
        # on a focus change, so a click-to-zoom still moves but ordinary reruns
        # (and the user's own rotation) don't snap the view back.
        uirevision=camera_revision,
        scene=dict(
            xaxis=dict(title="x (rear ←→ front)", backgroundcolor="#0e1216",
                       gridcolor="#1d242c", color="#8d99a6"),
            yaxis=dict(title="y (right)", backgroundcolor="#0e1216",
                       gridcolor="#1d242c", color="#8d99a6"),
            zaxis=dict(title="z (up)", backgroundcolor="#0e1216",
                       gridcolor="#1d242c", color="#8d99a6"),
            aspectmode="data", camera=scene_camera,
            dragmode="turntable"),
        font=dict(family="JetBrains Mono", color="#cdd6df", size=10),
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), itemsizing="constant"))
    # Expose the real drawn boxes (centre+size in car mm) so the UI can fit a CAD
    # to the ACTUAL placeholder a part occupies, not just the rough anchor.
    try:
        fig._part_boxes = {
            nm: dict(
                centre=[float((lo[i] + hi[i]) / 2.0) for i in range(3)],
                size=[float(hi[i] - lo[i]) for i in range(3)])
            for nm, (lo, hi) in _part_boxes.items()}
    except Exception:
        fig._part_boxes = {}
    return fig


# --------------------------------------------------------------------------- #
#  Live influence summary
# --------------------------------------------------------------------------- #
def influence_summary(vp, ledger, topology_label: str | None = None) -> list:
    rows = []
    def add(sys, status, detail):
        rows.append(dict(subsystem=sys, status=status, detail=detail))

    pt = _iface(ledger, "drivetrain")
    pkw = _g(pt, "peak_power_kw")
    add("drivetrain", "sized" if (pkw or _g(pt, "env_x_mm")) else "default",
        ("%.1f kW → engine + CVT size" % pkw) if pkw else "no power/envelope → nominal engine")

    _arch = (topology_label + " · ") if topology_label else ""
    add("front-suspension", "live",
        "%strack F %.0f mm · wheelbase %.0f mm · spring %.0f N/mm" % (
            _arch, getattr(vp, "track_front", 0),
            getattr(vp, "wheelbase", 0), getattr(vp, "spring_rate_front", 0)))
    add("rear-suspension", "live",
        "track R %.0f mm · spring %.0f N/mm" % (
            getattr(vp, "track_rear", 0), getattr(vp, "spring_rate_rear", 0)))

    ch = _iface(ledger, "chassis")
    cm = _g(ch, "mass_kg")
    add("chassis", "live", ("%.1f kg space frame" % cm) if cm else "space frame (no mass declared)")

    if ledger is not None:
        try:
            roll = ledger.mass_rollup()
            add("ALL", "rollup",
                "declared %.1f kg vs target %.0f kg (Δ %+.1f kg)" % (
                    roll["total_kg"], roll["target_kg"], roll["delta_kg"])
                + ("; CG live" if roll.get("cg_mm") else "; CG needs all masses+positions"))
        except Exception:
            pass
    return rows


def custom_part_fit(vp, part: dict) -> dict:
    """Plain-language fit check of a user-dropped part against the car envelope.

    Not a collision solver — that lives in the INTEGRATION tab against declared
    volumes. This is the quick "does my radiator even fit between the wheels and
    inside the floor-to-hoop height" read a sub-team wants the instant they drop
    a part on the car, expressed in real mm of clearance (negative = pokes out).

    Returns dict(status in {ok, tight, over}, messages:list[str], clearances:dict).
    """
    wb = float(getattr(vp, "wheelbase", 1550.0))
    tf = float(getattr(vp, "track_front", 1200.0))
    tr = float(getattr(vp, "track_rear", 1180.0))
    track = min(tf, tr)
    # A representative usable height: ground up to roughly the main-hoop top.
    z_ceiling = 1150.0

    l = float(part.get("l_mm", 0) or 0)
    w = float(part.get("w_mm", 0) or 0)
    h = float(part.get("h_mm", 0) or 0)
    cx = float(part.get("x_mm", 0) or 0)
    cy = float(part.get("y_mm", 0) or 0)
    cz = float(part.get("z_mm", 0) or 0)

    # Car spans x in [-wb/2, +wb/2] (mid-wheelbase origin), y in [-track/2, track/2].
    x_lo, x_hi = cx - l / 2, cx + l / 2
    y_lo, y_hi = cy - w / 2, cy + w / 2
    z_lo, z_hi = cz - h / 2, cz + h / 2

    clr = {
        "front of front axle": (wb / 2) - x_hi,
        "behind rear axle": x_lo - (-wb / 2),
        "right of track": (track / 2) - y_hi,
        "left of track": y_lo - (-track / 2),
        "above hoop height": z_ceiling - z_hi,
        "below ground": z_lo - 0.0,
    }
    msgs, status = [], "ok"
    for where, mm in clr.items():
        if mm < 0:
            status = "over"
            msgs.append("pokes out %s by %.0f mm" % (where, -mm))
        elif mm < 25:
            if status != "over":
                status = "tight"
            msgs.append("only %.0f mm clear %s" % (mm, where))
    if not msgs:
        msgs.append("sits inside the wheelbase, track and hoop-height envelope")
    return dict(status=status, messages=msgs, clearances=clr)


def part_dims_from_mesh(summary: dict) -> dict:
    """Pull a part's real L×W×H (mm) out of a loaded-CAD mesh summary.

    `summary` is what chassis.mesh_summary() returns (bbox_min/bbox_max/size_mm).
    The mesh sits in whatever frame it was exported in; we only take its overall
    bounding-box extents, mapped to the car's L(x)/W(y)/H(z) so a stand-in can be
    replaced by the part's true size the instant the CAD lands. Returns the size
    and the bbox so the caller can also recover a sensible default centre.
    """
    sz = summary.get("size_mm") or [0.0, 0.0, 0.0]
    lo = summary.get("bbox_min") or [0.0, 0.0, 0.0]
    hi = summary.get("bbox_max") or [0.0, 0.0, 0.0]
    l, w, h = (abs(float(sz[0])), abs(float(sz[1])), abs(float(sz[2])))
    ctr = [(float(lo[i]) + float(hi[i])) / 2.0 for i in range(3)]
    return dict(l_mm=l, w_mm=w, h_mm=h, centre_mm=ctr, size_mm=[l, w, h])


def reconcile_part(guess: dict, real: dict, tol_mm: float = 8.0,
                   tol_frac: float = 0.05) -> dict:
    """Compare a stand-in guess against the part that finally arrived.

    This is the catch for the exact handoff failure the team keeps hitting — the
    CAD that "is a mirror of the one I originally got", or that turns out a
    different size than everyone packaged around. We diff the three extents and
    also test for a swapped/mirrored aspect (the part's dimensions present but in
    a different order), and return a plain-language verdict so the dependent team
    learns BEFORE build that their guess was off.

    status: "match"      guess was right within tolerance
            "resize"     same part, different size — repackage around real dims
            "mirrored"   extents look transposed/swapped — likely a mirror/wrong
                         orientation handoff; check handedness before cutting
            "new"        nothing was packaged here yet (no guess to compare)
    """
    def trip(d):
        return [abs(float(d.get(k, 0) or 0)) for k in ("l_mm", "w_mm", "h_mm")]

    g, r = trip(guess or {}), trip(real or {})
    if max(g) <= 0:
        return dict(status="new", deltas=[r[0], r[1], r[2]],
                    messages=["No stand-in was here — placing the real part."])

    deltas = [r[i] - g[i] for i in range(3)]

    def within(a, b):
        return abs(a - b) <= max(tol_mm, tol_frac * max(a, b, 1.0))

    axiswise_ok = all(within(g[i], r[i]) for i in range(3))
    if axiswise_ok:
        return dict(status="match", deltas=deltas,
                    messages=["Real part matches your stand-in within tolerance "
                              "— your packaging holds."])

    # Same multiset of extents but assigned to different axes -> mirror/swap.
    if sorted(round(x, 1) for x in g) == sorted(round(x, 1) for x in r) and \
            not axiswise_ok:
        return dict(status="mirrored", deltas=deltas,
                    messages=["Same dimensions, different axes — this looks "
                              "mirrored or rotated vs your stand-in. Confirm "
                              "handedness/orientation before committing."])
    if sorted(within(gv, rv) for gv, rv in zip(sorted(g), sorted(r))) and \
            all(within(gv, rv) for gv, rv in zip(sorted(g), sorted(r))):
        return dict(status="mirrored", deltas=deltas,
                    messages=["Extents match but on different axes — likely a "
                              "mirror/orientation swap. Check before cutting."])

    msgs = []
    for ax, dv in zip(("L", "W", "H"), deltas):
        if abs(dv) > max(tol_mm, tol_frac * max(g[("L", "W", "H").index(ax)], 1.0)):
            msgs.append("%s %+.0f mm" % (ax, dv))
    return dict(status="resize", deltas=deltas,
                messages=["Real part differs from your stand-in: "
                          + ", ".join(msgs) + ". Repackage around the real size."])


def suggest_part_geometry(vp, subsys: str, ledger=None) -> dict:
    """Propose a TARGET size (x/y/z mm) + position for a part nobody has sized yet.

    The deeper version of the missing-part stall: a team is blocked not because a
    CAD is late but because they have *no idea* what the part should be, so they
    can't even guess. This gives them a number to design toward — dimensions the
    car can actually accommodate, using the same FSAE-typical proportions the
    full-car renderer already sizes each subsystem body with, scaled to THIS
    car's wheelbase / track / tyre size. It is an envelope to strive for, not a
    spec: "build it to roughly this and it will package."

    Any dimension the subsystem has already declared in the ledger (env_x/y/z) is
    honoured and passed straight back, so a partial declaration is completed
    rather than overwritten. Returns:

        l_mm, w_mm, h_mm     suggested extents (x, y, z)
        x_mm, y_mm, z_mm     a centre where that subsystem usually lives
        shape                "box" or "cylinder"
        basis                plain-language reason for each axis (what constrains it)
        from_declared        list of axes that came from the team's own declaration
    """
    wb = float(getattr(vp, "wheelbase", 1550.0))
    tf = float(getattr(vp, "track_front", 1200.0))
    tr = float(getattr(vp, "track_rear", 1180.0))
    # Approximate tyre radius and usable interior half-width the renderer uses.
    tire_r = 228.0
    interior_w = max(120.0, min(tf, tr) / 2.0 - 180.0 - 40.0) * 2.0  # full width
    x_front, x_rear = wb / 2.0, -wb / 2.0

    it = _iface(ledger, subsys) if ledger is not None else None
    dec = (_g(it, "env_x_mm"), _g(it, "env_y_mm"), _g(it, "env_z_mm"))

    # Per-subsystem TYPICAL envelope + home position, mirroring the renderer.
    S = subsys
    shape = "box"
    if S == "drivetrain":
        l, w, h = wb * 0.16, min(interior_w, 240.0), tire_r * 0.95
        x, y, z = x_rear + tire_r * 1.05, 0.0, tire_r * 0.9
        basis = ["L from wheelbase (rear engine bay)",
                 "W ≤ interior track width",
                 "H ~ engine + CVT stack height"]
    elif S == "data-acquisition":
        l, w, h = 160.0, 120.0, 80.0
        x, y, z = -wb * 0.16, 0.0, tire_r * 1.1
        basis = ["compact logger box", "fits beside the driver", "above the floor"]
    elif S == "chassis":
        l, w, h = wb * 0.5, min(interior_w * 1.1, 320.0), tire_r * 1.6
        x, y, z = -wb * 0.08, 0.0, tire_r * 0.9
        basis = ["L = central frame length", "W = interior width", "H = frame depth"]
    elif S == "front-suspension":
        l, w, h = 300.0, 200.0, 300.0
        x, y, z = x_front, tf / 2.0 - 120.0, tire_r
        basis = ["upright + steering package", "inboard of the front wheel", "corner height"]
    elif S == "rear-suspension":
        l, w, h = 300.0, 200.0, 300.0
        x, y, z = x_rear, tr / 2.0 - 120.0, tire_r
        basis = ["upright + trailing-arm package", "inboard of the rear wheel", "corner height"]
    else:
        l, w, h = 200.0, 150.0, 120.0
        x, y, z = 0.0, 0.0, tire_r
        basis = ["generic packaging box", "centred", "mid-height"]

    # Honour anything the team already declared: complete, don't overwrite.
    from_declared = []
    out_l, out_w, out_h = float(l), float(w), float(h)
    if dec[0]:
        out_l = float(dec[0]); from_declared.append("L")
    if dec[1]:
        out_w = float(dec[1]); from_declared.append("W")
    if dec[2]:
        out_h = float(dec[2]); from_declared.append("H")

    return dict(l_mm=round(out_l, 0), w_mm=round(out_w, 0), h_mm=round(out_h, 0),
                x_mm=round(float(x), 0), y_mm=round(float(y), 0),
                z_mm=round(float(z), 0), shape=shape, basis=basis,
                from_declared=from_declared)


# --------------------------------------------------------------------------- #
#  Per-PART replacement registry
# --------------------------------------------------------------------------- #
# Every body the car draws is individually replaceable by a CAD / sketch /
# estimate. This catalog maps each legend part to: the draw-name the renderer
# uses (so we can suppress exactly that body), its subsystem (colour + zoom),
# and an envelope+home so a dropped part auto-fits where THAT part belongs.
# `key` is a stable id used in session-state and suppression sets.
# SIMPLIFIED to seven subsystems. Each is replaceable as ONE unit by a CAD /
# sketch / estimate. Replacing a subsystem hides ALL of its procedural bodies
# (listed in `drawnames`) so only the real geometry shows there, while every
# OTHER subsystem stays on screen as a dummy suggestion — wheels, wings, driver,
# CG, etc. — so the user always sees their part in the context of a full car.
SUBSYSTEM_CATALOG = [
    # key                display name                    draw-names this subsystem owns
    ("chassis",          "Chassis",                      ["Frame tube", "Main hoop", "Front hoop", "Driver"]),
    ("drivetrain",       "Drivetrain",                   ["Engine + CVT"]),
    ("front-suspension", "Front Suspension + Steering",  ["Tire", "Upright", "Wheel hub",
                                                          "Upper wishbone", "Lower wishbone",
                                                          "Tie rod", "Pushrod", "Rocker",
                                                          "Spring/damper", "Brake disc"]),
    ("rear-suspension",  "Rear Suspension",              ["Tire", "Upright", "Wheel hub",
                                                          "Upper wishbone", "Lower wishbone",
                                                          "Pushrod", "Rocker", "Spring/damper"]),
]
SUBSYS_DRAWNAMES = {k: dn for k, _d, dn in SUBSYSTEM_CATALOG}
SUBSYS_DISPLAY = {k: d for k, d, _dn in SUBSYSTEM_CATALOG}

# Kept for the renderer's internal anchor/box lookups (per representative body).
PART_CATALOG = [
    ("monocoque",       "Space frame",         "chassis",          False),
    ("roll_hoop",       "Main hoop",           "chassis",          False),
    ("driver",          "Driver",              "chassis",          False),
    ("motor",           "Engine + CVT",        "drivetrain",       False),
    ("tire",            "Tire",                "front-suspension", True),
    ("brake_disc",      "Brake disc",          "front-suspension", True),
    ("upright",         "Upright",             "front-suspension", True),
]
# draw-name -> key, for suppression (the renderer suppresses by draw-name).
PART_DRAWNAME_BY_KEY = {k: dn for k, dn, _s, _c in PART_CATALOG}
PART_KEY_BY_DRAWNAME = {dn: k for k, dn, _s, _c in PART_CATALOG}
PART_SUBSYS_BY_KEY = {k: s for k, _dn, s, _c in PART_CATALOG}

# A representative body per subsystem, used to size/anchor a replacement.
SUBSYS_ANCHOR_PART = {
    "chassis": "monocoque", "drivetrain": "motor",
    "front-suspension": "tire", "rear-suspension": "tire",
}


def subsystem_catalog():
    """Public list of (key, display_name, [draw-names]) for the simplified UI."""
    return list(SUBSYSTEM_CATALOG)


def part_catalog():
    """Public list of (key, display_name, subsystem, is_corner) — internal."""
    return list(PART_CATALOG)


def suggest_part_geometry_for(vp, part_key: str, ledger=None) -> dict:
    """Per-PART target envelope + home position (finer than per-subsystem).

    Falls back to the subsystem suggestion, then refines for parts that are
    smaller than their whole subsystem (a roll hoop is not the whole chassis).
    """
    sub = PART_SUBSYS_BY_KEY.get(part_key, "chassis")
    base = suggest_part_geometry(vp, sub, ledger=ledger)
    wb = float(getattr(vp, "wheelbase", 1550.0))
    tf = float(getattr(vp, "track_front", 1200.0))
    tr = float(getattr(vp, "track_rear", 1180.0))
    tire_r = 228.0
    x_front, x_rear = wb / 2.0, -wb / 2.0

    def out(l, w, h, x, y, z, shape="box", basis=None):
        return dict(l_mm=round(l, 0), w_mm=round(w, 0), h_mm=round(h, 0),
                    x_mm=round(x, 0), y_mm=round(y, 0), z_mm=round(z, 0),
                    shape=shape, basis=basis or base["basis"],
                    from_declared=base.get("from_declared", []))

    if part_key == "front_wing":
        return out(tire_r * 0.65, tf * 0.98, 80, x_front + tire_r * 1.6, 0, tire_r * 0.32,
                   basis=["chord", "≈ front track", "element stack"])
    if part_key == "rear_wing":
        return out(tire_r * 0.7, tr * 0.92, 120, x_rear - tire_r * 0.8, 0, tire_r * 1.15,
                   basis=["chord", "≈ rear track", "tall element stack"])
    if part_key == "roll_hoop":
        return out(60, min(tf, tr) * 0.55, tire_r * 1.7, x_rear + wb * 0.30, 0, tire_r * 1.2,
                   basis=["tube thickness", "shoulder width", "above the driver"])
    if part_key == "driver":
        return out(280, 300, 600, x_front - wb * 0.16, 0, tire_r * 1.4,
                   basis=["torso", "shoulders", "seated height"])
    if part_key == "data_logger":
        return out(160, 120, 80, -wb * 0.16, 0, tire_r * 1.1,
                   basis=["compact box", "beside driver", "above floor"])
    if part_key == "tire":
        return out(360, 200, 360, x_front, tf / 2.0, tire_r,
                   shape="cylinder", basis=["diameter", "section width", "diameter"])
    if part_key == "brake_disc":
        return out(40, 280, 280, x_front, tf / 2.0 - 40, tire_r * 0.9,
                   shape="cylinder", basis=["disc thickness", "Ø", "Ø at wheel"])
    if part_key == "upright":
        return out(180, 160, 260, x_front, tf / 2.0 - 110, tire_r,
                   basis=["upright depth", "width", "hub-to-arm"])
    # monocoque, sidepod, radiator, motor, accumulator: subsystem suggestion fits.
    return base


# Single source of truth for WHERE and HOW BIG each part is. Both the placeholder
# renderer (via _placeholder_box, drawn when a part has no replacement) and the
# replacement auto-fit read this, so a CAD/sketch/estimate lands in EXACTLY the
# same box its placeholder occupied — every part integrates on one coordinate
# system. Honours ledger env_* declarations through suggest_part_geometry_for.
def part_anchor(vp, part_key: str, ledger=None) -> dict:
    """Canonical (centre + size + shape) box for a catalog part, in car SAE mm."""
    return suggest_part_geometry_for(vp, part_key, ledger=ledger)

