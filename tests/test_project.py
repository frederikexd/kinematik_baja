# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""Tests for the weight budget, decision log, persistence, and report."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suspension import project as pj


def _store():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd); os.unlink(path)
    return pj.ProjectStore(path)


def test_weight_total_and_qty():
    s = _store()
    s.add_weight(pj.WeightItem("suspension", "upright", mass_g=850, qty=4))
    assert abs(s.total_mass_kg() - 3.4) < 1e-6


def test_budget_over_flag():
    s = _store()
    s.target_mass_kg = 1.0
    s.add_weight(pj.WeightItem("powertrain", "engine", mass_g=42000, qty=1))
    assert s.budget_status()["over_budget"]


def test_persistence_roundtrip():
    s = _store()
    s.add_weight(pj.WeightItem("brakes", "caliper", mass_g=600, qty=4))
    s.add_decision(pj.Decision("cooling", "moved radiator", "interference at droop"))
    s.save()
    s2 = pj.ProjectStore(s.path)
    assert len(s2.weights) == 1 and len(s2.decisions) == 1
    os.unlink(s.path)


def test_mass_by_team_sorted():
    s = _store()
    s.add_weight(pj.WeightItem("a", "x", mass_g=1000))
    s.add_weight(pj.WeightItem("b", "y", mass_g=5000))
    teams = list(s.mass_by_team().keys())
    assert teams[0] == "b"


def test_cad_mass_estimate():
    g = pj.estimate_mass_g(350000, "Aluminium 6061")
    assert g is not None and 900 < g < 1000


def test_cad_mass_unknown_material():
    assert pj.estimate_mass_g(350000, "Other / custom") is None


def test_handover_markdown_contains_sections():
    s = _store()
    s.add_decision(pj.Decision("suspension", "raised RC", "less roll"))
    md = pj.build_handover_markdown(s, geometry={"caster_deg": 5.1})
    assert "Weight budget" in md and "Design decisions" in md and "raised RC" in md


def test_pdf_renders():
    s = _store()
    md = pj.build_handover_markdown(s)
    out = os.path.join(tempfile.gettempdir(), "t_handover.pdf")
    pj.render_pdf(md, out)
    assert os.path.getsize(out) > 1000
    os.unlink(out)


def test_note_addressing_and_broadcast():
    s = _store()
    s.add_note(pj.Note(from_team="suspension", to_team="brakes", message="recheck caliper"))
    s.add_note(pj.Note(from_team="cooling", to_team="all", message="radiator moved"))
    assert s.open_note_count("brakes") == 2  # direct + broadcast
    assert s.open_note_count("aerodynamics") == 1  # broadcast only
    os.unlink(s.path) if os.path.exists(s.path) else None


def test_note_resolve_reopen():
    s = _store()
    s.add_note(pj.Note(from_team="electrics", to_team="brakes", message="wiring route"))
    nid = s.notes[0].id
    s.resolve_note(nid)
    assert s.notes[0].status == "resolved"
    assert s.open_note_count("brakes") == 0
    s.reopen_note(nid)
    assert s.open_note_count("brakes") == 1


def test_notes_persist():
    s = _store()
    s.add_note(pj.Note(from_team="aero", to_team="all", message="x", urgent=True))
    s.save()
    s2 = pj.ProjectStore(s.path)
    assert len(s2.notes) == 1 and s2.notes[0].urgent
    os.unlink(s.path)


def test_open_notes_in_handover():
    s = _store()
    s.add_note(pj.Note(from_team="suspension", to_team="brakes",
                       message="upright moved", status="open"))
    md = pj.build_handover_markdown(s)
    assert "Open cross-team items" in md and "upright moved" in md


def test_search_freetext():
    s = _store()
    s.add_decision(pj.Decision("suspension", "Raised roll centre", "less body roll", tags="roll-centre"))
    s.add_decision(pj.Decision("cooling", "Moved radiator", "interference", tags="packaging"))
    assert [d.title for d in s.search_decisions(query="roll")] == ["Raised roll centre"]
    assert [d.title for d in s.search_decisions(query="interference")] == ["Moved radiator"]


def test_search_team_filter():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x"))
    s.add_decision(pj.Decision("cooling", "B", "y"))
    assert [d.title for d in s.search_decisions(team="suspension")] == ["A"]


def test_search_tag_filter():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x", tags="roll-centre, front"))
    s.add_decision(pj.Decision("suspension", "B", "y", tags="rear"))
    assert [d.title for d in s.search_decisions(tag="front")] == ["A"]


def test_search_combined():
    s = _store()
    s.add_decision(pj.Decision("suspension", "Rear ARB", "rotate on entry", tags="rear"))
    s.add_decision(pj.Decision("cooling", "Rear duct", "airflow", tags="rear"))
    res = s.search_decisions(query="rotate", team="suspension")
    assert [d.title for d in res] == ["Rear ARB"]


def test_all_decision_tags():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x", tags="roll-centre, front"))
    s.add_decision(pj.Decision("cooling", "B", "y", tags="packaging"))
    assert s.all_decision_tags() == ["front", "packaging", "roll-centre"]


def test_pluggable_backend_persists():
    # An in-memory backend stands in for Supabase: data must survive a new store.
    class MemoryBackend:
        store = {}
        def read(self): return dict(MemoryBackend.store)
        def write(self, payload): MemoryBackend.store = dict(payload)
    b = MemoryBackend()
    s = pj.ProjectStore(path="unused.json", backend=b)
    s.add_decision(pj.Decision("suspension", "persisted", "via backend"))
    s.save()
    # a brand-new store reading the same backend should see the decision
    s2 = pj.ProjectStore(path="unused.json", backend=b)
    assert len(s2.decisions) == 1 and s2.decisions[0].title == "persisted"


def test_json_backend_roundtrip():
    import tempfile, os as _os
    fd, p = tempfile.mkstemp(suffix=".json"); _os.close(fd); _os.unlink(p)
    b = pj.JSONFileBackend(p)
    s = pj.ProjectStore(path=p, backend=b)
    s.add_note(pj.Note(from_team="aero", to_team="all", message="hi"))
    s.save()
    s2 = pj.ProjectStore(path=p, backend=b)
    assert len(s2.notes) == 1
    _os.unlink(p)


def test_auto_backend_defaults_to_json(monkeypatch=None):
    # With no Supabase env vars, the auto backend is the JSON file backend.
    import os as _os
    _os.environ.pop("SUPABASE_URL", None)
    _os.environ.pop("SUPABASE_KEY", None)
    b = pj._auto_backend("whatever.json")
    assert isinstance(b, pj.JSONFileBackend)


def test_search_part_filter():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x", part="front upright"))
    s.add_decision(pj.Decision("suspension", "B", "y", part="rear hub"))
    assert [d.title for d in s.search_decisions(part="upright")] == ["A"]


def test_all_decision_parts():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x", part="front upright"))
    s.add_decision(pj.Decision("cooling", "B", "y", part="radiator"))
    assert s.all_decision_parts() == ["front upright", "radiator"]


def test_part_in_freetext_search():
    s = _store()
    s.add_decision(pj.Decision("suspension", "A", "x", part="front upright"))
    assert [d.title for d in s.search_decisions(query="upright")] == ["A"]


def test_demo_build_never_uses_supabase():
    # DEMO BUILD: Supabase is intentionally disconnected. Even with credentials
    # present in the environment, _auto_backend must return the local JSON
    # backend and must NOT attempt any connection or flag a degraded state.
    import os as _os
    _os.environ["SUPABASE_URL"] = "https://bad.invalid"
    _os.environ["SUPABASE_KEY"] = "badkey"
    try:
        b = pj._auto_backend("fallback.json")
        assert isinstance(b, pj.JSONFileBackend)
        # Not a degraded fallback — it's the deliberate, clean local backend.
        assert getattr(b, "degraded_reason", None) is None
    finally:
        _os.environ.pop("SUPABASE_URL", None)
        _os.environ.pop("SUPABASE_KEY", None)


def test_save_is_failsafe_on_backend_error():
    # Regression: a backend write failure (e.g. a remote Postgres/Supabase APIError)
    # must NOT propagate out of save() and crash the app — it returns False and
    # records the reason instead.
    class _BrokenBackend:
        def read(self):
            raise FileNotFoundError()
        def write(self, payload):
            raise RuntimeError("postgrest APIError: relation does not exist")
    s = pj.ProjectStore("unused.json", backend=_BrokenBackend())
    s.add_decision(pj.Decision(team="integration", title="x", rationale="y"))
    ok = s.save()                      # must not raise
    assert ok is False
    assert s.save_error and "Could not write" in s.save_error


def test_save_succeeds_on_local_json(tmp_path=None):
    import tempfile, os as _os
    d = tempfile.mkdtemp()
    p = _os.path.join(d, "proj.json")
    s = pj.ProjectStore(p, backend=pj.JSONFileBackend(p))
    s.add_decision(pj.Decision(team="suspension", title="ok", rationale="z"))
    assert s.save() is True
    assert s.save_error is None


def test_note_seen_by_roundtrips_and_defaults():
    """A note's read receipts survive save/load, and legacy rows without the
    field load cleanly as an empty dict."""
    from dataclasses import asdict
    n = pj.Note(from_team="suspension", to_team="aero", message="hi", author="Alice")
    assert n.seen_by == {}
    # round-trip through the dict form used by the backend
    assert pj.Note(**asdict(n)).seen_by == {}
    # legacy row persisted before seen_by existed
    legacy = {"from_team": "a", "to_team": "b", "message": "m",
              "author": "Bob", "status": "open", "ts": "2026-01-01T00:00:00", "id": "x"}
    assert pj.Note(**legacy).seen_by == {}
    # defensive coercion of a bad type
    assert pj.Note(from_team="a", to_team="b", message="m", seen_by=None).seen_by == {}


def test_mark_note_seen_records_and_excludes_author():
    import tempfile, os as _os
    d = tempfile.mkdtemp()
    p = _os.path.join(d, "proj.json")
    s = pj.ProjectStore(p, backend=pj.JSONFileBackend(p))
    s.add_note(pj.Note(from_team="suspension", to_team="aero", message="m1", author="Alice"))
    s.add_note(pj.Note(from_team="aero", to_team="suspension", message="m2", author="Bob"))

    # Bob opens the tab: he should be stamped on Alice's note but NOT his own.
    assert s.mark_note_seen("Bob") is True
    assert "Bob" in s.notes[0].seen_by
    assert "Bob" not in s.notes[1].seen_by

    # Idempotent: opening again changes nothing (so no needless backend write).
    assert s.mark_note_seen("Bob") is False

    # Receipts persist across a reload from the same backend.
    s.save()
    s2 = pj.ProjectStore(p, backend=pj.JSONFileBackend(p))
    assert "Bob" in s2.notes[0].seen_by

    # An empty viewer label is ignored.
    assert s2.mark_note_seen("") is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{p}/{len(fns)} project tests passed")
