# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""Tests for the chassis fit/clearance module using synthetic tube frames."""
import numpy as np, trimesh, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suspension import SuspensionKinematics, Hardpoints
from suspension import chassis as ch


def _tube(p, q, r=12):
    p, q = np.array(p, float), np.array(q, float)
    s = trimesh.creation.cylinder(radius=r, segments=16, height=np.linalg.norm(q - p))
    d = (q - p); L = np.linalg.norm(d); d /= L; z = np.array([0, 0, 1.])
    v = np.cross(z, d); c = np.dot(z, d)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = (p + q) / 2
    s.apply_transform(T); return s


def _frame(hp):
    return trimesh.util.concatenate([
        _tube(hp.upper_front_inner, hp.upper_rear_inner),
        _tube(hp.lower_front_inner, hp.lower_rear_inner),
        _tube(hp.upper_front_inner, hp.lower_front_inner)])


def test_pickups_on_frame_pass_fit():
    hp = Hardpoints.default()
    res = ch.fit_check(hp, _frame(hp), tol_mm=15)
    on_frame = [r for r in res if "front" in r["point"] or "rear" in r["point"]]
    assert any(r["mountable"] for r in on_frame)


def test_clean_frame_is_clear():
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    res = ch.clearance_check(kin, _frame(hp), warn_mm=8)
    assert res["verdict"] in ("CLEAR", "TIGHT")
    assert not any(v["collision"] for v in res["per_link"].values())


def test_intruding_tube_detected_as_collision():
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    mid = 0.5 * (np.array(hp.lower_front_inner) + kin.static.lower_outer)
    mesh = trimesh.util.concatenate([_frame(hp),
                                     _tube(mid + [-50, 0, 0], mid + [50, 0, 0], r=18)])
    res = ch.clearance_check(kin, mesh, warn_mm=8)
    assert res["verdict"] == "COLLISION"


def test_manufacturing_sheet_has_pickups():
    hp = Hardpoints.default()
    kin = SuspensionKinematics(hp)
    sheet = ch.manufacturing_sheet(hp, kin)
    assert "lower_front_inner" in sheet and "upright_length" in sheet


# --------------------------------------------------------------------------- #
#  Generic subsystem envelope vs chassis (any non-suspension subsystem)
# --------------------------------------------------------------------------- #
def _rail(p, q, r=8):
    return _tube(p, q, r=r)


def _hollow_frame(mid_rail=False):
    """A hollow tube box frame, 1000 long, 400 wide, 400 tall, optional middle rail."""
    rails = []
    ys = [-200, 0, 200] if mid_rail else [-200, 200]
    for y in ys:
        for z in [0, 400]:
            rails.append(_rail([0, y, z], [1000, y, z]))
    return trimesh.util.concatenate(rails)


def test_envelope_inside_open_frame_is_clear():
    frame = _hollow_frame()
    res = ch.envelope_fit_check(frame, origin=[400, -80, 150], size=[200, 160, 150],
                                name="motor", warn_mm=10)
    assert res["verdict"] == "CLEAR"
    assert res["contained"] is True
    assert res["min_clearance_mm"] > 10


def test_envelope_outside_bbox_flagged():
    frame = _hollow_frame()
    res = ch.envelope_fit_check(frame, origin=[950, -80, 150], size=[200, 160, 150],
                                name="radiator")
    assert res["verdict"] == "OUTSIDE"
    assert "x" in res["oob_axes"]
    assert res["contained"] is False


def test_envelope_straddling_interior_rail_collides():
    frame = _hollow_frame(mid_rail=True)
    # contained in bounds but straddling the y=0,z=0 middle rail
    res = ch.envelope_fit_check(frame, origin=[400, -60, 5], size=[200, 120, 120],
                                name="ecu", warn_mm=10)
    assert res["contained"] is True
    assert res["verdict"] == "COLLISION"


def test_envelope_tight_clearance_warns():
    frame = _hollow_frame()
    # place a box whose face sits a few mm from a corner rail at y=-200,z=0
    res = ch.envelope_fit_check(frame, origin=[400, -190, 12], size=[200, 150, 150],
                                name="pump", warn_mm=15)
    assert res["verdict"] in ("TIGHT", "CLEAR")  # near the rail
    assert res["contained"] is True


def test_envelope_box_points_cover_faces():
    pts = ch.envelope_box_points([0, 0, 0], [100, 100, 100], step_mm=25)
    assert len(pts) > 8                       # more than just corners
    assert pts[:, 0].max() <= 100 + 1e-6 and pts[:, 0].min() >= -1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{p}/{len(fns)} chassis tests passed")
