# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""Tests for the generic multi-team part-vs-chassis interference checker."""
import numpy as np, trimesh, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suspension import integration as ig


def _box(extents, at):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(at)
    return b


def test_part_inside_reference_collides():
    chassis = _box([600, 500, 400], [0, 0, 200])
    part = _box([200, 120, 300], [0, 0, 200])
    assert ig.interference_check(part, chassis)["verdict"] == "COLLISION"


def test_part_outside_is_clear():
    chassis = _box([600, 500, 400], [0, 0, 200])
    part = _box([200, 120, 300], [0, 500, 200])
    assert ig.interference_check(part, chassis)["verdict"] == "CLEAR"


def test_part_just_outside_is_tight():
    chassis = _box([600, 500, 400], [0, 0, 200])
    part = _box([100, 100, 100], [0, 303, 200])
    assert ig.interference_check(part, chassis, warn_mm=5)["verdict"] == "TIGHT"


def test_collision_fraction_bounds():
    chassis = _box([600, 500, 400], [0, 0, 200])
    part = _box([200, 120, 300], [0, 0, 200])
    f = ig.interference_check(part, chassis)["collision_fraction"]
    assert 0.0 <= f <= 1.0


def test_all_baja_teams_registered():
    expected = {"drivetrain", "front-suspension", "rear-suspension",
                "chassis", "data-acquisition"}
    assert expected == set(ig.TEAMS.keys())
    for v in ig.TEAMS.values():
        assert v["color"].startswith("#") and len(v["color"]) == 7


def test_legacy_team_keys_resolve():
    # Old FSAE keys from saved projects must still map onto a Baja owner.
    assert ig.resolve_team("suspension") == "front-suspension"
    assert ig.resolve_team("powertrain") == "drivetrain"
    assert ig.resolve_team("chassis") == "chassis"


def test_mass_estimate_from_density():
    part = _box([100, 100, 100], [0, 0, 0])
    rec = ig.part_record("cooling", "block", part, density_kg_m3=2700)
    assert rec.mass_g is not None and rec.mass_g > 0


def test_load_part_rejects_sldprt():
    try:
        ig.load_part("model.sldprt")
        assert False, "should have rejected sldprt"
    except ValueError as e:
        assert "sldprt" in str(e).lower() or "step" in str(e).lower()


def test_load_part_rejects_tiny_metre_mesh():
    import tempfile, os as _os
    b = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    p = _os.path.join(tempfile.gettempdir(), "tiny_test.stl"); b.export(p)
    try:
        ig.load_part(p)
        assert False, "should have flagged metre-scale mesh"
    except ValueError as e:
        assert "metre" in str(e).lower() or "small" in str(e).lower()
    finally:
        _os.unlink(p)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{p}/{len(fns)} integration tests passed")
