# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Project memory: weight budget, decision log, and handover report.

Two problems this solves for an underfunded team:

  WEIGHT BUDGET — the lightest reliable car is one of the few advantages money
  can't buy. But a budget that lives in one senior's spreadsheet dies when they
  graduate. Here it's a tracked, per-team running total against a target, with
  mass either estimated from CAD volume + material or entered by hand.

  HANDOVER — every year a team loses the *reasoning* behind its car: why the roll
  centre is where it is, why the battery box moved, what didn't work. Incomplete
  handover is how a team repeats last year's mistakes. So decisions are logged as
  they happen, and a one-click report bundles geometry + parts + weight + decisions
  into something next year's team can actually read.

Everything persists to JSON on disk (project.json) so it survives between sessions
and can be committed to the repo — the tool itself becomes the record, not a
person's memory. The report renders to Markdown, PDF, and JSON ("all of the above").
"""

from __future__ import annotations

import os
import json
import datetime as _dt
from dataclasses import dataclass, asdict, field

# Common FSAE materials, kg/m^3 — for CAD-volume mass estimates.
MATERIALS = {
    "Aluminium 6061": 2700, "Aluminium 7075": 2810, "Steel 4130": 7850,
    "Steel mild": 7850, "Titanium Ti-6Al-4V": 4430, "Carbon fibre (laminate)": 1600,
    "CFRP sandwich": 800, "ABS": 1040, "Nylon (3D print)": 1150,
    "PLA": 1240, "Magnesium": 1740, "Copper": 8960, "Other / custom": None,
}

DEFAULT_PROJECT = "project.json"


# --------------------------------------------------------------------------- #
#  Records
# --------------------------------------------------------------------------- #
@dataclass
class WeightItem:
    team: str
    name: str
    mass_g: float
    source: str = "manual"        # "manual" | "cad_estimate"
    material: str = ""
    qty: int = 1
    note: str = ""

    @property
    def total_g(self) -> float:
        return self.mass_g * self.qty


@dataclass
class Decision:
    team: str
    title: str
    rationale: str
    date: str = ""
    author: str = ""
    tags: str = ""
    part: str = ""               # the part/system this decision concerns (e.g. "front upright")

    def __post_init__(self):
        if not self.date:
            self.date = _dt.date.today().isoformat()


@dataclass
class Note:
    """
    A cross-team note between engineering leads. The point isn't chat — it's
    keeping interfaces from going stale. A note addressed to a specific team with
    an open/resolved status is a tracked action item, not a message that scrolls
    away in Discord. That's the difference that stops two finished parts not fitting.
    """
    from_team: str
    to_team: str                 # a team key, or "all"
    message: str
    author: str = ""
    is_request: bool = False     # asks the to_team to do something
    urgent: bool = False
    status: str = "open"         # "open" | "resolved"
    ts: str = ""
    id: str = ""
    # Read receipts: who has opened the Lead Notes tab and seen this note.
    # Keyed by a viewer label (the lead's name, or a session id if unnamed) ->
    # ISO timestamp of first view. Lets the *poster* see "Seen by ..." so they
    # know the note actually reached other leads, not just that it saved.
    seen_by: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.ts:
            self.ts = _dt.datetime.now().isoformat(timespec="seconds")
        if not self.id:
            self.id = _dt.datetime.now().strftime("%Y%m%d%H%M%S%f")
        # Tolerate older rows persisted before seen_by existed / wrong types.
        if not isinstance(self.seen_by, dict):
            self.seen_by = {}


# --------------------------------------------------------------------------- #
#  Storage backends — where the project memory actually lives
# --------------------------------------------------------------------------- #
class JSONFileBackend:
    """Default backend: a local JSON file. Perfect for laptops and tests."""

    def __init__(self, path: str):
        self.path = path
        self.degraded_reason = None   # set if we fell back from a failed Supabase

    def read(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {}

    def write(self, payload: dict):
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)


class SupabaseBackend:
    """
    Persists the whole project as a single JSON row in a Supabase (Postgres) table,
    so it survives restarts on ephemeral hosts like Streamlit Cloud.

    Expects a table named `kinematik_project` with columns:
        id   text  (primary key)
        data jsonb
    and these set in the environment / Streamlit secrets:
        SUPABASE_URL, SUPABASE_KEY
    A single row keyed by `project_id` (default "elbee") holds the team's data.
    Concurrency is last-write-wins, which is fine for a team of a few editors.
    """

    TABLE = "kinematik_project"

    def __init__(self, url: str, key: str, project_id: str = "elbee"):
        from supabase import create_client
        self.client = create_client(url, key)
        self.project_id = project_id

    def read(self) -> dict:
        resp = (self.client.table(self.TABLE)
                .select("data").eq("id", self.project_id).execute())
        rows = resp.data or []
        return rows[0]["data"] if rows else {}

    def write(self, payload: dict):
        self.client.table(self.TABLE).upsert(
            {"id": self.project_id, "data": payload}).execute()


def _read_credential(name: str):
    """Resolve a credential from either real environment variables or Streamlit
    Cloud secrets. Streamlit secrets (the TOML box in Settings) populate
    `st.secrets`, NOT `os.environ`, so an env-only lookup misses them and the app
    silently falls back to ephemeral local storage. Check both. Importing
    streamlit here (not at module top) keeps this module usable in plain
    scripts/tests with no Streamlit installed."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        # st.secrets behaves like a dict; .get avoids raising if the key is absent.
        secret = st.secrets.get(name)
        if secret:
            return str(secret)
    except Exception:
        pass
    return None


def _auto_backend(path: str):
    """
    DEMO BUILD — Supabase is intentionally disconnected.

    This standalone build always uses the local JSON file backend and never
    attempts any network connection, regardless of whether SUPABASE_URL /
    SUPABASE_KEY are set in the environment or in Streamlit secrets. Everything
    you do in the session is held in memory and written to a local project.json;
    nothing leaves the laptop.

    (The cloud-sync path is still present in SupabaseBackend below if you ever
    want to re-enable it — just restore the credential check that used to live
    here.)
    """
    return JSONFileBackend(path)


# --------------------------------------------------------------------------- #
#  Project store
# --------------------------------------------------------------------------- #
class ProjectStore:
    """
    The team's persistent project memory: weights, decisions, notes.

    Storage is pluggable. By default it reads/writes a local JSON file (great for
    running on a laptop or for tests). If a Supabase backend is configured (via
    environment variables on the deployed app), it persists to a hosted Postgres
    database instead — which survives restarts on ephemeral hosts like Streamlit
    Cloud, where the local filesystem is wiped. The rest of the app doesn't change:
    it calls .load() and .save() the same way regardless of backend.
    """

    # Class-level defaults: guarantee these attributes resolve even if __init__
    # is interrupted partway (a lazy optional-import failure, an exception in
    # load(), or a half-built instance returned from a cache). The render path
    # reads store.geometry / store.board unconditionally, so a missing attribute
    # turns into a redacted AttributeError on the deployed app.
    geometry = None
    board = None

    def __init__(self, path: str = DEFAULT_PROJECT, backend=None):
        self.path = path
        self.team_name = "Elbee Racing"
        self.season = str(_dt.date.today().year)
        self.target_mass_kg = 230.0
        self.weights: list[WeightItem] = []
        self.decisions: list[Decision] = []
        self.notes: list[Note] = []
        # Geometric mount-point / keep-out ledger (lazy import to avoid a hard
        # numpy dependency for callers that only touch weights/decisions/notes).
        # Defensive: never let an optional import failure leave the store without
        # the attribute (render_mountpoint_clash reads store.geometry
        # unconditionally — a bare import here turns a missing dep into an
        # AttributeError at `geom = store.geometry`).
        try:
            from .mountpoints import GeometryLedger
            self.geometry = GeometryLedger()
        except Exception:
            self.geometry = None
        # Electronics / PCB ledger (traces, differential pairs, aggressor nets) —
        # the copper-survival + signal-integrity board. Same lazy-import rationale.
        # Defensive: never let an optional import failure leave the store without
        # the attribute (the render path reads store.board unconditionally).
        try:
            from .electronics import BoardLedger
            self.board = BoardLedger()
        except Exception:
            self.board = None
        # Harness ledger (3-D routed wire runs + connectors) — the physical loom
        # in car space: bend radius, strain relief, clearance, and the
        # manufacturing roll-ups (cut length, formboard, BOM, copper mass). Same
        # lazy-import + defensive-default rationale as the board above.
        try:
            from .harness import HarnessLedger
            self.harness = HarnessLedger()
        except Exception:
            self.harness = None
        self.load_error = None
        self.save_error = None
        # Pick a backend: explicit > auto-detected Supabase > local JSON file.
        self.backend = backend or _auto_backend(path)
        self.load()

    def _payload(self) -> dict:
        return {
            "team_name": self.team_name,
            "season": self.season,
            "target_mass_kg": self.target_mass_kg,
            "weights": [asdict(w) for w in self.weights],
            "decisions": [asdict(x) for x in self.decisions],
            "notes": [asdict(n) for n in self.notes],
            "geometry": self.geometry.as_dict() if self.geometry else {},
            "board": self.board.as_dict() if self.board else {},
            "harness": self.harness.as_dict() if getattr(self, "harness", None) else {},
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    def _apply(self, d: dict):
        if not d:
            return
        self.team_name = d.get("team_name", self.team_name)
        self.season = d.get("season", self.season)
        self.target_mass_kg = d.get("target_mass_kg", self.target_mass_kg)
        self.weights = [WeightItem(**w) for w in d.get("weights", [])]
        self.decisions = [Decision(**x) for x in d.get("decisions", [])]
        self.notes = [Note(**n) for n in d.get("notes", [])]
        geom = d.get("geometry")
        if geom:
            from .mountpoints import GeometryLedger
            self.geometry = GeometryLedger.from_dict(geom)
        board = d.get("board")
        if board:
            from .electronics import BoardLedger
            self.board = BoardLedger.from_dict(board)
        harness = d.get("harness")
        if harness:
            from .harness import HarnessLedger
            self.harness = HarnessLedger.from_dict(harness)

    # ----------------------------- io ---------------------------------- #
    def load(self):
        try:
            d = self.backend.read()
        except FileNotFoundError:
            return  # fresh local project, nothing saved yet — expected
        except Exception as e:
            # A genuine read failure (corrupt file, DB error) shouldn't be hidden.
            self.load_error = f"Could not read saved project data: {e}"
            return
        self._apply(d)

    def save(self):
        """Persist the project. Fail-safe: a storage backend error (e.g. a remote
        Supabase/Postgres misconfiguration) is recorded on `self.save_error` and
        returns False rather than raising, so a save side-effect can never crash the
        caller. Returns True on success."""
        try:
            self.backend.write(self._payload())
            self.save_error = None
            return True
        except Exception as e:
            self.save_error = f"Could not write project data: {e}"
            return False

    def as_json(self) -> str:
        return json.dumps({
            "team_name": self.team_name, "season": self.season,
            "target_mass_kg": self.target_mass_kg,
            "weights": [asdict(w) for w in self.weights],
            "decisions": [asdict(x) for x in self.decisions],
            "notes": [asdict(n) for n in self.notes],
            "geometry": self.geometry.as_dict() if getattr(self, "geometry", None) else {},
            "board": self.board.as_dict() if getattr(self, "board", None) else {},
            "harness": self.harness.as_dict() if getattr(self, "harness", None) else {},
        }, indent=2)

    # -------------------------- mutations ------------------------------ #
    def add_weight(self, item: WeightItem):
        self.weights.append(item)

    def add_decision(self, dec: Decision):
        self.decisions.append(dec)

    def search_decisions(self, query="", team=None, tag=None, part=None):
        """
        Find decisions by free-text query (matches title + rationale + author + part),
        optional team, tag, and part filters. Returns newest-first. This is the
        'written but findable' layer — the whole point of the handover tool is that
        next year can locate the reasoning in seconds, including by which part it's about.
        """
        q = (query or "").strip().lower()
        out = []
        for d in self.decisions:
            if team and d.team != team:
                continue
            if tag and tag.lower() not in (d.tags or "").lower():
                continue
            if part and part.lower() not in (getattr(d, "part", "") or "").lower():
                continue
            if q:
                haystack = f"{d.title} {d.rationale} {d.author} {d.tags} {getattr(d, 'part', '')}".lower()
                if q not in haystack:
                    continue
            out.append(d)
        return sorted(out, key=lambda d: d.date, reverse=True)

    def all_decision_parts(self):
        """Unique, sorted list of parts/systems referenced across decisions."""
        parts = set()
        for d in self.decisions:
            p = (getattr(d, "part", "") or "").strip()
            if p:
                parts.add(p)
        return sorted(parts)

    def all_decision_tags(self):
        """Unique, sorted list of tags used across decisions (split on commas)."""
        tags = set()
        for d in self.decisions:
            for t in (d.tags or "").split(","):
                t = t.strip()
                if t:
                    tags.add(t)
        return sorted(tags)

    def add_note(self, note: Note):
        self.notes.append(note)

    def resolve_note(self, note_id: str):
        for n in self.notes:
            if n.id == note_id:
                n.status = "resolved"

    def reopen_note(self, note_id: str):
        for n in self.notes:
            if n.id == note_id:
                n.status = "open"

    def mark_note_seen(self, viewer: str, exclude_author: bool = True) -> bool:
        """Record that `viewer` has now seen the notes addressed to them.

        Stamps every note this viewer can see (i.e. not ones they authored, when
        exclude_author is set) with a first-seen timestamp. Returns True if any
        note was newly stamped, so the caller knows whether a save is worthwhile.
        A viewer is a stable label — the lead's typed name, or a session id when
        they haven't given one.
        """
        if not viewer:
            return False
        changed = False
        for n in self.notes:
            if exclude_author and n.author and n.author == viewer:
                continue
            if viewer not in n.seen_by:
                n.seen_by[viewer] = _dt.datetime.now().isoformat(timespec="seconds")
                changed = True
        return changed

    def notes_for(self, team: str, include_all=True):
        """Notes addressed to a team (and 'all' broadcasts), newest first."""
        out = [n for n in self.notes
               if n.to_team == team or (include_all and n.to_team == "all")]
        return sorted(out, key=lambda n: n.ts, reverse=True)

    def open_note_count(self, team: str):
        return sum(1 for n in self.notes_for(team) if n.status == "open")

    def remove_weight(self, idx: int):
        if 0 <= idx < len(self.weights):
            self.weights.pop(idx)

    # --------------------- geometry mutations -------------------------- #
    def set_mount_point(self, mp):
        """Add or replace a mount point in the geometry ledger."""
        self.geometry.set_point(mp)

    def set_keepout(self, ko):
        """Add or replace a keep-out volume in the geometry ledger."""
        self.geometry.set_keepout(ko)

    def remove_mount_point(self, name: str):
        self.geometry.points.pop(name, None)

    def remove_keepout(self, name: str):
        self.geometry.keepouts.pop(name, None)

    def move_mount(self, ledger, name: str, xyz_mm, set_by: str = "",
                   update_interface_cg: bool = False):
        """
        Move a mount point and propagate: re-run the clearance clash and re-roll the
        CG through the supplied IntegrationLedger, in one call. Returns the
        PropagationResult. Does NOT auto-save — the caller decides when to persist.
        """
        from .mountpoints import propagate_mount_move
        return propagate_mount_move(
            self.geometry, ledger, name, xyz_mm, set_by=set_by,
            update_interface_cg=update_interface_cg)

    def clash_findings(self):
        """Current clash/clearance findings for the stored geometry."""
        return self.geometry.check_clashes()

    # ---------------------- electronics / PCB board -------------------- #
    def _ensure_board(self):
        """Lazily create the board ledger if an old payload or import gap left it
        unset, so callers can always rely on store.board being present."""
        if getattr(self, "board", None) is None:
            from .electronics import BoardLedger
            self.board = BoardLedger()
        return self.board

    def set_trace(self, tr):
        """Add or replace a copper trace in the board ledger."""
        self._ensure_board().set_trace(tr)

    def set_pair(self, dp):
        """Add or replace a differential pair in the board ledger."""
        self._ensure_board().set_pair(dp)

    def set_aggressor(self, ag):
        """Add or replace an aggressor (noisy) net in the board ledger."""
        self._ensure_board().set_aggressor(ag)

    def remove_trace(self, name: str):
        self._ensure_board().traces.pop(name, None)

    def remove_pair(self, name: str):
        self._ensure_board().pairs.pop(name, None)

    def remove_aggressor(self, name: str):
        self._ensure_board().aggressors.pop(name, None)

    def board_check(self, ledger=None, scenario=None):
        """Run the full pre-fab board gate (copper survival + signal integrity).
        Returns a BoardCheckResult; does NOT auto-save."""
        from .electronics import check_board
        return check_board(self._ensure_board(), ledger=ledger, scenario=scenario)

    # ---------------------- harness / 3-D loom ------------------------- #
    def _ensure_harness(self):
        """Lazily create the harness ledger if an old payload or import gap left
        it unset, so callers can always rely on store.harness being present."""
        if getattr(self, "harness", None) is None:
            from .harness import HarnessLedger
            self.harness = HarnessLedger()
        return self.harness

    def set_wire(self, w):
        """Add or replace a routed wire run in the harness ledger."""
        self._ensure_harness().set_wire(w)

    def set_connector(self, c):
        """Add or replace a connector in the harness ledger."""
        self._ensure_harness().set_connector(c)

    def remove_wire(self, name: str):
        self._ensure_harness().remove_wire(name)

    def remove_connector(self, name: str):
        self._ensure_harness().remove_connector(name)

    def harness_check(self):
        """Run the full pre-cut harness gate (bend radius + strain relief +
        3-D clearance) and roll up cut list / BOM / mass / formboard. The
        keep-outs come from this project's own geometry ledger, so the loom is
        checked against the very volumes the mount-points clash against. Returns
        a HarnessCheckResult; does NOT auto-save."""
        from .harness import check_harness
        keepouts = []
        geom = getattr(self, "geometry", None)
        if geom is not None:
            keepouts = list(getattr(geom, "keepouts", {}).values())
        return check_harness(self._ensure_harness(), keepouts=keepouts)

    # --------------------------- queries ------------------------------- #
    def total_mass_kg(self) -> float:
        return sum(w.total_g for w in self.weights) / 1000.0

    def mass_by_team(self) -> dict:
        out: dict[str, float] = {}
        for w in self.weights:
            out[w.team] = out.get(w.team, 0.0) + w.total_g / 1000.0
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def budget_status(self) -> dict:
        total = self.total_mass_kg()
        over = total - self.target_mass_kg
        return {
            "total_kg": total,
            "target_kg": self.target_mass_kg,
            "delta_kg": over,
            "over_budget": over > 0,
            "pct_of_target": (total / self.target_mass_kg * 100.0)
            if self.target_mass_kg else 0.0,
        }


# --------------------------------------------------------------------------- #
#  CAD mass estimate
# --------------------------------------------------------------------------- #
def estimate_mass_g(volume_mm3: float, material: str) -> float | None:
    rho = MATERIALS.get(material)
    if rho is None or volume_mm3 is None:
        return None
    return (volume_mm3 * 1e-9) * rho * 1000.0   # mm^3 -> m^3 -> kg -> g


# --------------------------------------------------------------------------- #
#  Handover report
# --------------------------------------------------------------------------- #
def build_handover_markdown(store: ProjectStore,
                            geometry: dict | None = None,
                            extra_notes: str = "") -> str:
    """
    Assemble the full handover report as Markdown. `geometry` is an optional dict
    of the current suspension setup (static alignment, key metrics) so the report
    captures the design state, not just the admin data.
    """
    b = store.budget_status()
    today = _dt.date.today().isoformat()
    L = []
    L.append(f"# {store.team_name} — Handover Report")
    L.append(f"_Season {store.season} · generated {today}_\n")
    L.append("This report is auto-generated from the KinematiK project file. It "
             "captures the car's design state, weight budget, and the reasoning behind "
             "key decisions so next year's team starts from knowledge, not a blank page.\n")

    # Weight budget
    L.append("## Weight budget\n")
    status = "OVER BUDGET" if b["over_budget"] else "within budget"
    L.append(f"- Target mass: **{b['target_kg']:.1f} kg**")
    L.append(f"- Current total: **{b['total_kg']:.1f} kg** "
             f"({b['pct_of_target']:.0f}% of target — {status})")
    L.append(f"- Delta: **{b['delta_kg']:+.1f} kg**\n")
    if store.mass_by_team():
        L.append("| Subteam | Mass (kg) |")
        L.append("|---|---|")
        for team, kg in store.mass_by_team().items():
            L.append(f"| {team} | {kg:.2f} |")
        L.append("")
    if store.weights:
        L.append("### Logged parts\n")
        L.append("| Team | Part | Qty | Mass each (g) | Total (g) | Source |")
        L.append("|---|---|---|---|---|---|")
        for w in store.weights:
            L.append(f"| {w.team} | {w.name} | {w.qty} | {w.mass_g:.0f} | "
                     f"{w.total_g:.0f} | {w.source} |")
        L.append("")

    # Suspension / geometry state
    if geometry:
        L.append("## Suspension design state\n")
        for k, v in geometry.items():
            if isinstance(v, float):
                L.append(f"- {k}: {v:.2f}")
            else:
                L.append(f"- {k}: {v}")
        L.append("")

    # Decision log
    L.append("## Design decisions & rationale\n")
    if not store.decisions:
        L.append("_No decisions logged yet. Log them as you go — this is the section "
                 "next year's team will thank you for._\n")
    else:
        for d in sorted(store.decisions, key=lambda x: x.date):
            head = f"### {d.title}  \n"
            meta = f"_{d.team} · {d.date}"
            if d.author:
                meta += f" · {d.author}"
            if d.tags:
                meta += f" · {d.tags}"
            meta += "_\n"
            L.append(head + meta)
            L.append(d.rationale + "\n")

    if extra_notes.strip():
        L.append("## Additional notes\n")
        L.append(extra_notes.strip() + "\n")

    # Open cross-team items — unresolved interfaces next year must not lose
    open_notes = [n for n in store.notes if n.status == "open"]
    if open_notes:
        L.append("## Open cross-team items\n")
        L.append("_Unresolved interface notes carried into handover — these are loose "
                 "ends the next team needs to close._\n")
        L.append("| From | To | Note | Urgent |")
        L.append("|---|---|---|---|")
        for n in sorted(open_notes, key=lambda x: x.ts):
            u = "yes" if n.urgent else ""
            msg = n.message.replace("|", "/")
            L.append(f"| {n.from_team} | {n.to_team} | {msg} | {u} |")
        L.append("")

    L.append("---")
    L.append("_Generated by the Elbee Racing Baja suspension & integration studio._")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  SAAD per-subsystem documentation
#
#  Competition teams (e.g. CSULB) document each part in a "Standard Archived and
#  Accurate Documentation" (SAAD) format: a cover sheet with a WW-XYY-ZZZ part
#  number, a three-phase table of contents, and a fixed set of prompted sections
#  per phase. This produces ONE such document per subsystem, pre-filling the
#  cover table, weight budget, and any logged design decisions for that subteam,
#  and leaving the standard section prompts in place for the team to answer in
#  paragraph form (deleting the prompts as they go, per the template's
#  instructions). One document per subsystem keeps Elbee's docs drawer matching
#  the format the strongest Baja programs archive in.
# --------------------------------------------------------------------------- #

# Subsystem numeric ID for the WW-XYY-ZZZ part number (the "X" digit / "YY"
# subcategory live in the "XYY" field). These match the standard Baja subsystem
# identification scheme: 000 Data Acq, 100 Brakes, 200 Chassis/Ergo, 300
# Drivetrain, 400 Front Suspension, 500 Rear Suspension.
SAAD_SUBSYSTEM_IDS = {
    "data-acquisition": "000",
    "chassis":          "200",
    "drivetrain":       "300",
    "front-suspension": "400",
    "rear-suspension":  "500",
}

# The club objectives every section is asked to correlate against.
SAAD_CLUB_OBJECTIVES = [
    "Meet Rule Requirements", "Serviceability", "Manufacturability",
    "Vehicle Integration", "Cost", "Performance",
]


def saad_part_number(team_key: str, season: str, subcategory: str = "00",
                     id_number: str = "000") -> str:
    """Build a WW-XYY-ZZZ part number.

    WW  = last two digits of the competition year (from the season string)
    X   = subsystem identification digit (first digit of the subsystem ID)
    YY  = subcategory within subsystem (caller-supplied, default "00")
    ZZZ = ID number (caller-supplied, default "000")
    """
    sid = SAAD_SUBSYSTEM_IDS.get(team_key, "900")  # 900 = unmapped/other
    ww = "".join(ch for ch in str(season) if ch.isdigit())[-2:] or "26"
    x = sid[0]
    yy = (str(subcategory) + "00")[:2]
    zzz = (str(id_number) + "000")[:3]
    return f"{ww}-{x}{yy}-{zzz}"


def build_saad_markdown(store: ProjectStore, team_key: str,
                        team_label: str | None = None,
                        geometry: dict | None = None,
                        component_name: str = "Name Of Component",
                        subsystem_lead: str = "",
                        team_members: str = "") -> str:
    """
    Assemble a single subsystem's documentation in the SAAD template format:
    cover sheet, three-phase table of contents, and the standard prompted
    sections for each phase. Any design decisions / weights logged for this
    subteam are pre-filled so the document starts from the team's real data
    rather than a blank page; the section prompts remain so the subteam can
    answer them in paragraph form and delete the prompts as instructed.
    """
    label = team_label or team_key
    today = _dt.date.today().isoformat()
    part_no = saad_part_number(team_key, store.season)

    # Decisions logged against this subteam — the real content to seed sections.
    team_decisions = sorted(
        [d for d in store.decisions if d.team == team_key],
        key=lambda x: x.date)
    team_weights = [w for w in store.weights if w.team == team_key]
    team_mass_kg = sum(w.total_g for w in team_weights) / 1000.0
    open_notes = [n for n in store.notes
                  if n.status == "open" and n.to_team in (team_key, "all")]

    L: list[str] = []

    # ---- Cover sheet ------------------------------------------------------- #
    L.append(f"# {label}")
    L.append("_Society of Automotive Engineers · Elbee Racing Baja_")
    L.append("_Standard Archived and Accurate Documentation (SAAD)_")
    L.append(f"_Season {store.season} · generated {today}_\n")

    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| Part Number | {part_no} |")
    L.append(f"| Component | {component_name} |")
    L.append(f"| Subsystem | {label} |")
    L.append(f"| Subsystem Lead | {subsystem_lead or '—'} |")
    L.append(f"| Team Members | {team_members or '—'} |")
    L.append("")
    L.append("_Update this document throughout development, not just at the end. "
             "Date each update. Answer the prompts in paragraph form and delete "
             "the prompts once answered._\n")

    # ---- Table of contents ------------------------------------------------- #
    L.append("## Table of Contents\n")
    L.append("**Phase 1: Problem Identification & Concept Development (R&D)** — "
             "Introduction; Background & Previous Problems; Objectives & "
             "Constraints; Design Requirements & Rules; Proposed Improvements; "
             "Implementation Plan; Impact & Benefits; Vehicle Integration; "
             "Timeframe; Results; References.")
    L.append("**Phase 2: Design and Analysis** — CAD Design; FEA Analysis; "
             "Manufacturing.")
    L.append("**Phase 3: Testing, Evaluation, Validation, and Iteration** — "
             "Testing & Validation; Trade-offs & Performance Evaluation; Future "
             "Improvements; Next Steps.\n")

    L.append("**Club objectives** (correlate each section to these): "
             + "; ".join(SAAD_CLUB_OBJECTIVES) + ".\n")

    # Helper: emit a section heading + its prompts, plus any seeded data.
    def section(title: str, prompts: list[str], seed: list[str] | None = None):
        L.append(f"### {title}")
        L.append("_Planned: MM/DD/YY–MM/DD/YY · Actual: MM/DD/YY–MM/DD/YY_\n")
        if seed:
            L.extend(seed)
            L.append("")
        for p in prompts:
            L.append(f"- {p}")
        L.append("")

    # ---- Phase 1 ----------------------------------------------------------- #
    L.append("## Phase 1: Problem Identification & Concept Development (R&D)\n")

    section("Introduction", [
        "Define the subsystem or vehicle component being researched.",
        "Outline component objectives and the problems to address, correlating "
        "these to the club objectives.",
    ], seed=[f"This document covers the **{label}** subsystem of the Elbee "
             f"Racing Baja vehicle (part number {part_no})."])

    bg_seed = None
    if team_decisions:
        bg_seed = ["_Prior design decisions logged for this subsystem:_"]
        for d in team_decisions:
            part = f" ({d.part})" if d.part else ""
            bg_seed.append(f"- **{d.title}**{part} — {d.date}: {d.rationale}")
    section("Background & Previous Problems", [
        "Describe the current designs, components, or configurations on the Baja vehicle.",
        "Give the reasoning behind previous design decisions (cost, "
        "manufacturability, time).",
        "What improvements or changes are you aiming to achieve?",
        "What challenges or problems resulted from previous design decisions?",
        "Competitive benchmarking: what are other teams or industry leaders "
        "doing differently? Cite who and why.",
        "Use annotated images of current designs and competitor solutions.",
    ], seed=bg_seed)

    section("Objectives & Constraints", [
        "State what you aim to analyze, test, investigate, or improve.",
        "List project objectives and constraints in logical order.",
        "Quantify all new research.",
    ])

    section("Design Requirements & Rules", [
        "Define quantifiable performance metrics from previous designs.",
        "List relevant constraints and rules, citing rule sections directly, "
        "and why each pertains to this topic.",
        "Define expected loads and how they are accounted for in the design.",
    ])

    section("Proposed Improvements", [
        "Brainstorm and describe potential innovations or modifications in detail.",
        "Evaluate trade-offs in cost, weight, and performance, with predicted results.",
        "Support proposals with data-driven methods (simulations, prior testing).",
        "Rank innovations by measurable benchmarks (feasibility, cost-benefit, impact).",
    ])

    section("Implementation Plan", [
        "Explain how the objectives and constraints will be achieved.",
        "List required resources (software, materials, tools); confirm club "
        "access or propose alternatives.",
        "Identify challenges (supply chain, software limits) and collaboration "
        "opportunities with other subsystems.",
        "Weigh in-house fabrication vs outsourcing against the shared, limited budget.",
    ])

    impact_seed = None
    if team_weights:
        impact_seed = [f"_Current logged mass for {label}: "
                       f"**{team_mass_kg:.2f} kg** across {len(team_weights)} "
                       f"part(s)._"]
    section("Impact & Benefits", [
        "Explain the importance of the research and expected benefits.",
        "Highlight how the changes address current issues with the vehicle.",
        "Provide measurable goals for testing and performance improvement.",
    ], seed=impact_seed)

    vi_seed = None
    if open_notes:
        vi_seed = ["_Open cross-subsystem interface items affecting this subsystem:_"]
        for n in sorted(open_notes, key=lambda x: x.ts):
            u = " **[urgent]**" if n.urgent else ""
            vi_seed.append(f"- From {n.from_team}: {n.message}{u}")
    section("Vehicle Integration", [
        "Identify how changes here could affect other subsystems "
        "(e.g. suspension arm changes → chassis mounting tabs).",
        "Determine whether the changes require adjustments to other components.",
    ], seed=vi_seed)

    section("Timeframe", [
        "Outline a timeline with key phases and milestones "
        "(Phase 1 R&D; Phase 2 Design & Analysis; Phase 3 Testing & Iteration).",
    ])

    section("Results", [
        "Summarize research findings and assess feasibility for implementation.",
        "Highlight key insights, next steps, and areas for further research.",
        "Quantify the benefits and how they enhance previous designs.",
        "Align with club objectives; outline how subsystem performance will be "
        "measured in testing or competition.",
        "State whether the component needs a complete redesign or the extent of "
        "modification.",
    ])

    section("References", [
        "List all references cited (videos, papers, websites, etc.).",
    ])

    # ---- Phase 2 ----------------------------------------------------------- #
    L.append("## Phase 2: Design and Analysis\n")

    geo_seed = None
    if geometry:
        geo_seed = ["_Current design state captured from the studio:_"]
        for k, v in geometry.items():
            if isinstance(v, float):
                geo_seed.append(f"- {k}: {v:.2f}")
            else:
                geo_seed.append(f"- {k}: {v}")
    section("CAD Design", [
        "How did the design evolve during this phase?",
        "What tools and references developed the CAD model?",
        "What is the design intent, and how does the design fulfil the Phase 1 "
        "engineering requirements?",
    ], seed=geo_seed)

    section("FEA Analysis", [
        "How was the FEA set up to validate the design?",
        "What boundary conditions were applied, and why (fixed points, forces, loads)?",
        "What are the results; why do they make sense; how do they align with "
        "engineering principles (stress distribution, safety factors)?",
        "Were results consistent with expectations? If not, what changed and why?",
    ])

    section("Manufacturing", [
        "How feasible is this design to manufacture?",
        "What manufacturing methods were considered and chosen, and why?",
        "What challenges arose during manufacturing and how were they resolved?",
        "What is the order of operations?",
        "Include CAD models, FEA screenshots, and photos of manufacturing.",
    ])

    # ---- Phase 3 ----------------------------------------------------------- #
    L.append("## Phase 3: Testing, Evaluation, Validation, and Iteration\n")

    section("Testing & Validation", [
        "How has the design been tested (physical tests, simulations)?",
        "Do test results match the expected performance from FEA and CAD?",
        "Were additional adjustments or redesigns needed based on testing?",
    ])

    section("Trade-offs & Performance Evaluation", [
        "Key benefits of the current design (performance, weight, etc.).",
        "Potential flaws or drawbacks (complexity, cost).",
        "Are the trade-offs worth it given the overall project goals?",
    ])

    section("Future Improvements", [
        "What challenges remain, and what changes could further optimize the design?",
        "What is the plan for future iterations or adjustments?",
    ])

    section("Next Steps", [
        "What immediate actions are needed next (further testing, redesign)?",
        "What are the key upcoming milestones in your timeline?",
        "Include images of testing setups, results, and ongoing work.",
    ])

    L.append("---")
    L.append(f"_SAAD document for {label} · part {part_no} · generated by the "
             "Elbee Racing Baja suspension & integration studio. This internal "
             "document is for final part documentation and will be printed and "
             "stored in the shop documentation drawer once completed._")
    return "\n".join(L)


def build_all_saad_markdown(store: ProjectStore, team_labels: dict,
                            geometry: dict | None = None) -> str:
    """Concatenate a SAAD document for every subsystem into one file, page-broken."""
    parts = []
    for team_key, label in team_labels.items():
        # Geometry only belongs in the suspension subsystems' design state.
        geo = geometry if team_key in ("front-suspension", "rear-suspension") else None
        parts.append(build_saad_markdown(store, team_key, team_label=label,
                                         geometry=geo))
    return "\n\n---\n\n".join(parts)


def render_pdf(markdown_text: str, out_path: str):
    """Render the handover Markdown to a clean PDF via reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#0f6e56"), spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceBefore=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13)

    flow = []
    table_buf = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                for r in table_buf if "---" not in r]
        if rows:
            t = Table(rows, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1f5ee")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f6f6f6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 6))
        table_buf = []

    for line in markdown_text.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            table_buf.append(s)
            continue
        flush_table()
        if not s:
            flow.append(Spacer(1, 4))
        elif s.startswith("# "):
            flow.append(Paragraph(s[2:], h1))
        elif s.startswith("## "):
            flow.append(Paragraph(s[3:], h2))
        elif s.startswith("### "):
            flow.append(Paragraph(s[4:].replace("  ", ""), h3))
        elif s.startswith("- "):
            txt = s[2:].replace("**", "<b>", 1)
            txt = txt.replace("**", "</b>", 1) if "<b>" in txt else txt
            flow.append(Paragraph("• " + txt, body))
        elif s.startswith("---"):
            flow.append(Spacer(1, 6))
        else:
            txt = s.replace("**", "<b>", 1)
            txt = txt.replace("**", "</b>", 1) if "<b>" in txt else txt
            txt = txt.replace("_", "<i>", 1)
            txt = txt.replace("_", "</i>", 1) if "<i>" in txt else txt
            flow.append(Paragraph(txt, body))
    flush_table()

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    doc.build(flow)
    return out_path
