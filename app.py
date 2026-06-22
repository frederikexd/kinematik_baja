# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
KinematiK — open-source Formula SAE suspension design studio.

Edit double-wishbone hardpoints live and watch the kinematics (camber gain, bump
steer, caster, KPI, scrub) and the vehicle-level consequences (roll-centre
migration, lateral load transfer, grip balance) update together. Built for the
FSAE garage where OptimumK / ADAMS budgets don't reach.

Run:  streamlit run app.py
"""

import json
import os
import tempfile
import datetime as _datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from suspension import (
    SuspensionKinematics, Hardpoints,
    VehicleDynamics, VehicleParams,
    MATERIALS, MemberStiffness, CompliantCorner,
    load_flex_body, corner_wheel_load, WheelLoad,
    GenericKinematics, list_templates, example as topo_example,
)
from suspension import compliance as compliance_mod
from suspension import flex as flex_mod
from suspension import chassis as chassis_mod
from suspension import integration as integ_mod
from suspension import project as project_mod
from suspension import tiremodel as tire_mod
from suspension import setup as setup_mod
from suspension import laptime as lap_mod
from suspension import correlation as corr_mod
from suspension import damper as damper_mod
from suspension import interfaces as interfaces_mod
from suspension import transient as transient_mod
from suspension import ggv as ggv_mod
from suspension import tire_thermal as thermal_mod
from suspension import units as units_mod

# --------------------------------------------------------------------------- #
#  Cached compute layer.
#
#  Streamlit re-executes this entire script top-to-bottom on EVERY widget
#  interaction, and it runs the body of every `with tab:` block regardless of
#  which tab is visible. Without caching, that meant the thermal warm-up
#  integration (~30 s) and the GGV envelope (~2 s) were recomputed on every
#  rerun — that is why the app "took forever". These wrappers memoize the heavy
#  physics on their actual inputs, so they only recompute when something that
#  affects the result changes. Args are kept hashable (dicts/tuples/floats) so
#  cache_data can key on them.
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _cached_thermal_warmup(coeffs, fnomin, enable_mu, cold_pa,
                           alpha_deg, Fz, v_x, gamma_deg,
                           ambient_c, track_c, duration_s, dt):
    _lt = tire_mod.PacejkaLateral(coeffs=dict(coeffs), FNOMIN=fnomin)
    _tp = thermal_mod.ThermalParams(enable_mu_feedback=bool(enable_mu),
                                    cold_pressure_pa=float(cold_pa))
    _tm = thermal_mod.ThermalTireModel(lateral=_lt, params=_tp)
    return thermal_mod.simulate_warmup(
        model=_tm, alpha_deg=float(alpha_deg), Fz=float(Fz), v_x=float(v_x),
        gamma_deg=float(gamma_deg), ambient_c=float(ambient_c),
        track_c=float(track_c), duration_s=float(duration_s), dt=float(dt))


st.set_page_config(page_title="KinematiK · FSAE Suspension Studio",
                   page_icon="◢", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
#  Aesthetic: technical instrument panel. Dark carbon, amber/cyan telemetry,
#  monospace data, a single high-contrast accent. No generic dashboard look.
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#0b0d10; --panel:#13171c; --panel2:#171c22;
  --line:#262d36; --ink:#e7ecf1; --dim:#8d99a6;
  --amber:#ffb02e; --cyan:#37e0d0; --red:#ff5a52; --grid:#1d242c;
}
.stApp{ background:
  radial-gradient(1200px 600px at 80% -10%, #14202655 0%, transparent 60%),
  var(--bg); color:var(--ink); }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
h1,h2,h3,h4{ font-family:'Archivo',sans-serif!important; letter-spacing:-.02em; }
body, p, span, div, label{ font-family:'Archivo',sans-serif; }
.mono, .stMetric, code{ font-family:'JetBrains Mono',monospace!important; }

.brand{ display:flex; align-items:baseline; gap:.6rem; border-bottom:1px solid var(--line);
        padding-bottom:.5rem; margin-bottom:.2rem;}
.brand .mark{ font-family:'Archivo'; font-weight:800; font-size:2.1rem;
        background:linear-gradient(90deg,var(--amber),var(--cyan)); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; }
.brand .sub{ color:var(--dim); font-family:'JetBrains Mono'; font-size:.78rem; letter-spacing:.18em; text-transform:uppercase;}

.card{ background:linear-gradient(180deg,var(--panel2),var(--panel));
       border:1px solid var(--line); border-radius:14px; padding:1.0rem 1.1rem; }
.metric{ display:flex; flex-direction:column; gap:.15rem; padding:.7rem .9rem;
         border:1px solid var(--line); border-radius:12px; background:var(--panel2);}
.metric .v{ font-family:'JetBrains Mono'; font-weight:600; font-size:1.45rem; line-height:1; }
.metric .k{ color:var(--dim); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;}
.metric .u{ color:var(--dim); font-size:.85rem; font-weight:400;}
.tag{ display:inline-block; font-family:'JetBrains Mono'; font-size:.7rem; padding:.18rem .5rem;
      border-radius:6px; border:1px solid var(--line); color:var(--dim);}
.good{ color:var(--cyan); border-color:#1f4d49;}
.warn{ color:var(--amber); border-color:#5a4317;}
.bad{ color:var(--red); border-color:#5a2422;}
.stTabs [data-baseweb="tab-list"]{ gap:2px; }
.stTabs [data-baseweb="tab"]{ background:var(--panel); border:1px solid var(--line);
      border-bottom:none; border-radius:10px 10px 0 0; color:var(--dim); font-family:'JetBrains Mono'; font-size:.8rem;}
.stTabs [aria-selected="true"]{ color:var(--ink); background:var(--panel2); border-color:#34507c;}
.hint{ color:var(--dim); font-size:.82rem; }
hr{ border-color:var(--line);}
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono'!important;}

/* Buttons and download buttons — dark theme (Streamlit defaults render white) */
.stButton > button, .stDownloadButton > button{
  background:var(--panel2)!important;
  color:var(--ink)!important;
  border:1px solid var(--line)!important;
  border-radius:10px!important;
  font-family:'JetBrains Mono',monospace!important;
  font-size:.82rem!important;
  font-weight:600!important;
  transition:border-color .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:var(--amber)!important;
  background:#1b222a!important;
  color:var(--amber)!important;
}
.stButton > button:active, .stDownloadButton > button:active{ background:#11161b!important; }
.stButton > button:focus, .stDownloadButton > button:focus{
  box-shadow:none!important; border-color:var(--amber)!important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div{
  background:var(--panel2)!important; color:var(--ink)!important; border-color:var(--line)!important;
}
.stFileUploader > div{ background:var(--panel2)!important; border-color:var(--line)!important; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1216",
    font=dict(family="JetBrains Mono, monospace", color="#cdd6df", size=11),
    margin=dict(l=55, r=20, t=40, b=45),
    xaxis=dict(gridcolor="#1d242c", zerolinecolor="#33414e"),
    yaxis=dict(gridcolor="#1d242c", zerolinecolor="#33414e"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
AMBER, CYAN, RED, DIM = "#ffb02e", "#37e0d0", "#ff5a52", "#8d99a6"


# --------------------------------------------------------------------------- #
#  State
# --------------------------------------------------------------------------- #
def init_state():
    if "hp" not in st.session_state:
        st.session_state.hp = Hardpoints.default().as_dict()
    if "vp" not in st.session_state:
        st.session_state.vp = VehicleParams().__dict__.copy()
    # Tire model: start on the generic default so grip/balance run on a real Magic
    # Formula from the first load. Replaced by a TTC-fitted tire when one is loaded.
    if "tire_coeffs" not in st.session_state:
        dt = tire_mod.default_tire()
        st.session_state.tire_coeffs = dict(dt.coeffs)
        st.session_state.tire_fnomin = dt.FNOMIN
        st.session_state.tire_source = "Generic FSAE default (not your tire)"
        st.session_state.tire_is_default = True
    # Subsystem interface ledger — the cross-team integration contract.
    if "ledger" not in st.session_state:
        st.session_state.ledger = interfaces_mod.blank_ledger().as_dict()

init_state()

POINTS = [
    ("upper_front_inner", "Upper wishbone · front inner (chassis)"),
    ("upper_rear_inner",  "Upper wishbone · rear inner (chassis)"),
    ("lower_front_inner", "Lower wishbone · front inner (chassis)"),
    ("lower_rear_inner",  "Lower wishbone · rear inner (chassis)"),
    ("upper_outer",       "Upper ball joint (upright)"),
    ("lower_outer",       "Lower ball joint (upright)"),
    ("tie_rod_inner",     "Tie rod · inner (rack)"),
    ("tie_rod_outer",     "Tie rod · outer (upright)"),
    ("wheel_center",      "Wheel centre"),
    ("contact_patch",     "Contact patch"),
]

# Optional pushrod / rocker pickups. When all are present the tool reports the REAL
# motion ratio (k_wheel = k_spring·MR²) instead of the direct-acting proxy.
ROCKER_POINTS = [
    ("pushrod_outer",  "Pushrod · outer (on wishbone/upright)"),
    ("rocker_pivot",   "Rocker · pivot"),
    ("rocker_axis",    "Rocker · pivot axis (direction)"),
    ("rocker_pushrod", "Rocker · pushrod pickup"),
    ("rocker_spring",  "Rocker · spring pickup"),
    ("spring_inner",   "Spring/damper · chassis mount"),
]


def metric(label, value, unit="", cls=""):
    # Convert the displayed number + unit into the user's chosen unit system.
    # The model stays metric; only this presentation layer converts. Compound
    # units (e.g. "°/10mm", "N/mm @35") get bespoke handling.
    if unit in ("°/10mm",) or str(unit).startswith("N/mm"):
        value, unit = units_mod.convert_compound(str(value), unit)
    else:
        value = units_mod.convert_value_str(str(value), unit)
        unit = units_mod.label(unit)
    return f"""<div class="metric"><span class="k">{label}</span>
    <span class="v {cls}">{value}<span class="u"> {unit}</span></span></div>"""


PROJECT_PATH = os.path.join(os.getcwd(), "project.json")


def log_decision_now(team, title, rationale, author="auto"):
    """Append a decision to the persistent store from any tab.

    Fail-safe: a logging convenience must NEVER take down the app. If the backend
    write fails (e.g. a remote Supabase/Postgres backend is misconfigured or
    unreachable), swallow the error, record it quietly, and return False so the
    caller can fall back. Returns True on success.
    """
    try:
        st_ = project_mod.ProjectStore(PROJECT_PATH)
        st_.add_decision(project_mod.Decision(
            team=team, title=title, rationale=rationale, author=author,
            tags="auto-captured"))
        st_.save()
        return True
    except Exception as e:
        try:
            st.session_state.setdefault("_log_errors", [])
            st.session_state["_log_errors"].append(str(e))
        except Exception:
            pass
        return False


# --------------------------------------------------------------------------- #
#  Live cross-team note notifications
#
#  The Lead Notes tab persists a note to project.json (or Supabase), but that
#  only helps the *next* lead who happens to open that one tab and rerun the
#  page. A note another lead needs to see could sit unseen for hours. This block
#  closes that gap: every session polls the shared store on a short interval and,
#  whenever a note appears that this session hasn't seen yet (and didn't write
#  itself), it fires a toast — on whatever tab the user is currently looking at —
#  and bumps an unread badge on the LEAD NOTES tab. So one lead posting a note
#  notifies everyone else who has the platform open.
# --------------------------------------------------------------------------- #
NOTE_POLL_SECONDS = 10


def _load_notes_from_disk():
    """Read notes straight from the shared backend, bypassing the per-session
    cached store, so we see what *other* sessions have written. Never raises."""
    try:
        fresh = project_mod.ProjectStore(PROJECT_PATH)
        return list(fresh.notes)
    except Exception:
        return []


def _note_line(n):
    """One-line human summary of a note for a toast."""
    frm = integ_mod.TEAMS.get(n.from_team, {}).get("label", n.from_team)
    if n.to_team == "all":
        to = "all teams"
    else:
        to = integ_mod.TEAMS.get(n.to_team, {}).get("label", n.to_team)
    who = f" ({n.author})" if getattr(n, "author", "") else ""
    flags = ""
    if getattr(n, "urgent", False):
        flags += " ⚠ URGENT"
    if getattr(n, "is_request", False):
        flags += " · action requested"
    msg = (n.message or "").strip()
    if len(msg) > 140:
        msg = msg[:137] + "…"
    return f"📝 **{frm} → {to}**{who}{flags}\n\n{msg}"


def poll_note_notifications():
    """Detect notes posted since this session last looked and notify the user.

    Tracks the set of note ids already seen by THIS session. On the very first
    run we seed the baseline silently (so a brand-new visitor isn't flooded with
    toasts for the whole history). After that, any unseen note that wasn't
    authored in this session triggers a toast and increments the unread badge.
    """
    notes = _load_notes_from_disk()
    seen = st.session_state.get("_notes_seen_ids")
    my_session = st.session_state.get("_my_posted_note_ids", set())

    if seen is None:
        # First load this session: establish baseline, don't toast history.
        st.session_state["_notes_seen_ids"] = {n.id for n in notes}
        st.session_state.setdefault("_notes_unread", 0)
        return

    new_notes = [n for n in notes
                 if n.id not in seen and n.id not in my_session]
    if new_notes:
        # Oldest first so toasts read in chronological order.
        for n in sorted(new_notes, key=lambda x: x.ts):
            try:
                st.toast(_note_line(n), icon="📝")
            except Exception:
                pass
        st.session_state["_notes_unread"] = (
            st.session_state.get("_notes_unread", 0) + len(new_notes))

    # Update the high-water mark to everything currently on disk.
    st.session_state["_notes_seen_ids"] = {n.id for n in notes}


# A fragment lets the poll loop run on its own short timer WITHOUT re-executing
# the whole (expensive) app body. run_every makes idle sessions actually pick up
# notes other leads post, instead of only noticing on the next manual click.
try:
    _NOTE_FRAGMENT = st.fragment(run_every=NOTE_POLL_SECONDS)

    @_NOTE_FRAGMENT
    def _note_notification_fragment():
        poll_note_notifications()
        unread = st.session_state.get("_notes_unread", 0)
        if unread:
            st.markdown(
                f"<div style='position:sticky;top:0;z-index:50;'>"
                f"<span class='tag warn'>📝 {unread} new lead note"
                f"{'s' if unread != 1 else ''} — open LEAD NOTES</span></div>",
                unsafe_allow_html=True)
    _HAVE_NOTE_FRAGMENT = True
except Exception:
    # Older Streamlit without run_every: fall back to a once-per-rerun poll.
    _HAVE_NOTE_FRAGMENT = False

    def _note_notification_fragment():
        poll_note_notifications()


# --------------------------------------------------------------------------- #
#  Sidebar — geometry editor
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="brand"><span class="mark">◢ KinematiK</span></div>',
                unsafe_allow_html=True)

    _UNIT_LABELS = {"metric": "Metric (SI)", "us": "US / Imperial"}
    _unit_sys = st.radio(
        "Units",
        ["metric", "us"],
        index=["metric", "us"].index(st.session_state.get("unit_system", "metric")),
        format_func=lambda k: _UNIT_LABELS[k],
        horizontal=True,
        help="Switch all displayed values and input fields between metric "
             "(mm, kg, N, m/s, °C …) and US/Imperial (in, lb, lbf, mph, °F …). "
             "The underlying model always computes in SI; only the display and "
             "input units change.")
    st.session_state.unit_system = _unit_sys
    _US = (_unit_sys == "us")
    # Per-quantity unit labels for input widgets (track the active system).
    _U_LEN = units_mod.label("mm")
    _U_MASS = units_mod.label("kg")
    _U_RATE = units_mod.label("N/mm")
    _U_TORQ = units_mod.label("N·m")

    _TOPO_LABELS = {
        "double_wishbone": "Double wishbone (full editor)",
        "macpherson_strut": "MacPherson strut",
        "multilink": "Multi-link (5-link)",
        "trailing_arm": "Trailing arm",
        "semi_trailing_arm": "Semi-trailing arm",
        "solid_axle": "Solid axle (Panhard)",
        "twist_beam": "Twist-beam",
        "truck_steer_linkage": "Heavy-truck steer linkage",
        "from_links": "Experimental / free-form",
    }
    _topo_keys = list(_TOPO_LABELS.keys())
    topo_choice = st.selectbox(
        "Suspension topology",
        _topo_keys,
        format_func=lambda k: _TOPO_LABELS.get(k, k),
        index=_topo_keys.index(st.session_state.get("topology", "double_wishbone")),
        help="Double-wishbone exposes the full live hardpoint editor. Every other "
             "architecture is solved by the architecture-agnostic multibody engine "
             "from a representative parameter set and feeds the same vehicle-level "
             "balance analysis.")
    st.session_state.topology = topo_choice
    if topo_choice != "double_wishbone":
        st.caption("Agnostic engine · this topology drives the same RC / anti-dive / "
                   "balance pipeline as the wishbone path.")

    st.markdown(f'<div class="sub" style="color:#8d99a6;font-family:JetBrains Mono;font-size:.7rem;letter-spacing:.18em;margin-bottom:.6rem;">HARDPOINT EDITOR · {_U_LEN} · SAE x-rear y-right z-up</div>', unsafe_allow_html=True)

    _is_wishbone = (topo_choice == "double_wishbone")
    if not _is_wishbone:
        st.info("The full hardpoint editor below applies to the double-wishbone "
                "model. The selected topology uses its built-in parameterised "
                "template; switch back to double-wishbone to edit pickups live.")


    colA, colB = st.columns(2)
    if colA.button("↺ Reset", width='stretch'):
        st.session_state.hp = Hardpoints.default().as_dict()
        st.rerun()
    preset = colB.selectbox("Preset", ["Front (default)", "Low roll-centre",
                                       "High anti-dive"], label_visibility="collapsed")

    st.markdown("###### Design intent")
    c1, c2 = st.columns(2)
    st.session_state.hp["static_camber"] = c1.number_input(
        "Static camber °", value=float(st.session_state.hp.get("static_camber", -1.5)),
        step=0.1, format="%.2f")
    st.session_state.hp["static_toe"] = c2.number_input(
        "Static toe °", value=float(st.session_state.hp.get("static_toe", 0.0)),
        step=0.05, format="%.2f")

    st.markdown("###### Pickup coordinates")
    _coord_step = 0.1 if _US else 2.0
    _coord_fmt = "%.2f" if _US else "%.1f"
    for key, label in POINTS:
        with st.expander(label, expanded=False):
            v = st.session_state.hp[key]
            cols = st.columns(3)
            nv = []
            for i, ax in enumerate("xyz"):
                _disp = cols[i].number_input(
                    f"{ax} ({_U_LEN})", value=units_mod.from_metric(float(v[i]), "mm"),
                    step=_coord_step, key=f"{key}_{ax}",
                    format=_coord_fmt, label_visibility="visible")
                nv.append(units_mod.to_metric(_disp, "mm"))
            st.session_state.hp[key] = nv

    st.markdown("###### Pushrod / rocker")
    rocker_on = st.checkbox(
        "Pushrod-actuated (real motion ratio)",
        value=bool(st.session_state.hp.get("pushrod_outer") is not None),
        help="When on, the motion ratio and wheel rate come from the actual "
             "bell-crank geometry. When off, a direct-acting proxy is used and "
             "reported spring→wheel rates are only indicative.")
    if rocker_on:
        # Seed rocker points from the default if the project doesn't carry them.
        _def = Hardpoints.default().as_dict()
        for key, label in ROCKER_POINTS:
            if st.session_state.hp.get(key) is None:
                st.session_state.hp[key] = _def[key]
        attach = st.selectbox(
            "Pushrod mounts on", ["lower", "upper", "upright"],
            index=["lower", "upper", "upright"].index(
                st.session_state.hp.get("pushrod_attach", "lower")))
        st.session_state.hp["pushrod_attach"] = attach
        for key, label in ROCKER_POINTS:
            with st.expander(label, expanded=False):
                v = st.session_state.hp[key]
                cols = st.columns(3)
                nv = []
                for i, ax in enumerate("xyz"):
                    _disp = cols[i].number_input(
                        f"{ax} ({_U_LEN})",
                        value=units_mod.from_metric(float(v[i]), "mm"),
                        step=_coord_step, key=f"{key}_{ax}",
                        format="%.2f", label_visibility="visible")
                    nv.append(units_mod.to_metric(_disp, "mm"))
                st.session_state.hp[key] = nv
    else:
        # Clear rocker points so has_rocker() is False and the proxy is used.
        for key, _ in ROCKER_POINTS:
            st.session_state.hp[key] = None

    st.markdown("---")
    st.markdown("###### Vehicle")
    vp = st.session_state.vp
    if _US:
        _m_lo, _m_hi = units_mod.from_metric(180, "kg"), units_mod.from_metric(360, "kg")
        _m_disp = st.slider(f"Mass + driver ({_U_MASS})", round(_m_lo), round(_m_hi),
                            round(units_mod.from_metric(float(vp["mass"]), "kg")))
        vp["mass"] = units_mod.to_metric(_m_disp, "kg")
        _cg_lo, _cg_hi = units_mod.from_metric(200, "mm"), units_mod.from_metric(400, "mm")
        _cg_disp = st.slider(f"CG height ({_U_LEN})", round(_cg_lo, 1), round(_cg_hi, 1),
                             round(units_mod.from_metric(float(vp["cg_height"]), "mm"), 1))
        vp["cg_height"] = units_mod.to_metric(_cg_disp, "mm")
    else:
        vp["mass"] = st.slider("Mass + driver (kg)", 180, 360, int(vp["mass"]))
        vp["cg_height"] = st.slider("CG height (mm)", 200, 400, int(vp["cg_height"]))
    vp["weight_dist_front"] = st.slider("Front weight (%)", 40, 60,
                                        int(vp["weight_dist_front"] * 100)) / 100

    st.markdown("###### Springs & roll stiffness")
    use_springs = st.checkbox(
        "Drive roll stiffness from spring rates × motion ratio",
        value=bool(vp.get("use_spring_rates", False)),
        help="On: axle roll stiffness = spring rate × MR² (+ ARB), using the live "
             "rocker geometry. This is the physically correct path and is what the "
             "optimiser uses. Off: type roll stiffness directly (legacy).")
    vp["use_spring_rates"] = use_springs
    if use_springs:
        s1, s2 = st.columns(2)
        _sf = s1.number_input(
            f"Spring F ({_U_RATE})",
            value=units_mod.from_metric(float(vp.get("spring_rate_front", 35.0)), "N/mm"),
            step=units_mod.from_metric(2.5, "N/mm"))
        vp["spring_rate_front"] = units_mod.to_metric(_sf, "N/mm")
        _sr = s2.number_input(
            f"Spring R ({_U_RATE})",
            value=units_mod.from_metric(float(vp.get("spring_rate_rear", 35.0)), "N/mm"),
            step=units_mod.from_metric(2.5, "N/mm"))
        vp["spring_rate_rear"] = units_mod.to_metric(_sr, "N/mm")
        a1, a2 = st.columns(2)
        _af = a1.number_input(
            f"ARB F ({_U_TORQ}/°)",
            value=units_mod.from_metric(float(vp.get("arb_rate_front", 0.0)), "N·m"),
            step=units_mod.from_metric(10.0, "N·m"))
        vp["arb_rate_front"] = units_mod.to_metric(_af, "N·m")
        _ar = a2.number_input(
            f"ARB R ({_U_TORQ}/°)",
            value=units_mod.from_metric(float(vp.get("arb_rate_rear", 0.0)), "N·m"),
            step=units_mod.from_metric(10.0, "N·m"))
        vp["arb_rate_rear"] = units_mod.to_metric(_ar, "N·m")
    else:
        cc1, cc2 = st.columns(2)
        _rf = cc1.number_input(f"Roll stiff F ({_U_TORQ}/°)",
                               value=units_mod.from_metric(float(vp["roll_stiffness_front"]), "N·m"),
                               step=units_mod.from_metric(10.0, "N·m"))
        vp["roll_stiffness_front"] = units_mod.to_metric(_rf, "N·m")
        _rr = cc2.number_input(f"Roll stiff R ({_U_TORQ}/°)",
                               value=units_mod.from_metric(float(vp["roll_stiffness_rear"]), "N·m"),
                               step=units_mod.from_metric(10.0, "N·m"))
        vp["roll_stiffness_rear"] = units_mod.to_metric(_rr, "N·m")


# Apply presets (simple variations on the default)
def apply_preset(name, hp):
    hp = dict(hp)
    if name == "Low roll-centre":
        hp["lower_front_inner"][2] = 95
        hp["lower_rear_inner"][2] = 95
    elif name == "High anti-dive":
        # Steepen the forward-and-up convergence of the side-view wishbone pivot
        # axes (raise the front pickups, lower the rears) so the side-view swing
        # arm shortens and anti-dive rises from ~26% (default) to ~40%.
        hp["upper_front_inner"][2] = 305
        hp["lower_front_inner"][2] = 135
        hp["upper_rear_inner"][2] = 285
        hp["lower_rear_inner"][2] = 108
    return hp

hp_dict = apply_preset(preset, st.session_state.hp)


# --------------------------------------------------------------------------- #
#  Solve
# --------------------------------------------------------------------------- #
try:
    _topo = st.session_state.get("topology", "double_wishbone")
    if _topo == "double_wishbone":
        hp = Hardpoints.from_dict(hp_dict)
        kin = SuspensionKinematics(hp)
    else:
        mech = topo_example(_topo)
        kin = GenericKinematics(mech)
        hp = kin.hp
    # Build the live tire model from session state (default or TTC-fitted).
    _tire = tire_mod.PacejkaLateral(coeffs=dict(st.session_state.tire_coeffs),
                                    FNOMIN=st.session_state.tire_fnomin)
    # Only pass VehicleParams fields the dataclass knows about (forward/backward
    # compatible if an old saved project carries extra/missing keys).
    _vp_fields = set(VehicleParams.__dataclass_fields__.keys())
    _vp_kwargs = {k: v for k, v in st.session_state.vp.items() if k in _vp_fields}
    veh = VehicleDynamics(VehicleParams(**_vp_kwargs),
                          front_kin=kin, rear_kin=kin, tire=_tire)
    # Steer-DOF linkages (e.g. truck steering) have a limited vertical-travel
    # envelope; sweep a narrower band so the studio stays usable.
    _span = 15 if _topo == "truck_steer_linkage" else 30
    sweep = kin.sweep(-_span, _span, 41)
    solve_ok = all(s.converged for s in sweep)
    if not solve_ok and not kin.static.converged:
        st.error("Solver could not converge at the static ride height for this "
                 "topology. Check the template parameters.")
        st.stop()
except Exception as e:
    st.error(f"Solver failed for this geometry: {e}")
    st.stop()

if not solve_ok:
    st.warning("Some travel positions did not converge for this topology; "
               "results outside the converged band may be incomplete.")

st.markdown('<div class="brand"><span class="mark">◢ KinematiK</span>'
            f'<span class="sub">{_TOPO_LABELS.get(_topo, _topo)} · agnostic engine · open source</span></div>',
            unsafe_allow_html=True)

s = kin.static
mid = veh.lateral_load_transfer(1.2)[1]

# headline metrics
def gain(metric_fn):
    a = metric_fn(kin.solve_at_travel(-10))
    b = metric_fn(kin.solve_at_travel(10))
    return (b - a) / 20.0  # per mm

camber_gain = gain(lambda st_: st_.camber)
bump_steer = gain(lambda st_: st_.toe)

cols = st.columns(6)
items = [
    ("Static camber", f"{s.camber:+.2f}", "°", ""),
    ("Camber gain", f"{camber_gain*10:+.2f}", "°/10mm",
     "good" if camber_gain < 0 else "warn"),
    ("Bump steer", f"{bump_steer*10:+.3f}", "°/10mm",
     "good" if abs(bump_steer*10) < 0.1 else "warn"),
    ("Caster", f"{s.caster:+.1f}", "°", ""),
    ("KPI", f"{s.kpi:+.1f}", "°", ""),
    ("Scrub radius", f"{s.scrub_radius:+.0f}", "mm",
     "good" if abs(s.scrub_radius) < 25 else "warn"),
]
for c, (k, v, u, cls) in zip(cols, items):
    c.markdown(metric(k, v, u, cls), unsafe_allow_html=True)

if not solve_ok:
    st.markdown('<span class="tag bad">⚠ linkage does not close over full travel — '
                'check wishbone lengths</span>', unsafe_allow_html=True)

# Motion ratio + anti-dive/anti-squat row. MR is REAL when a rocker is defined;
# otherwise a clearly-labelled direct-acting proxy. Anti-dive uses this (front)
# corner's side-view geometry against the vehicle CG/wheelbase.
_mr = kin.motion_ratio()
_mr_real = kin.motion_ratio_is_real()
_spring_demo = float(st.session_state.vp.get("spring_rate_front", 35.0))
_wr = kin.wheel_rate(_spring_demo)
_ad = kin.anti_dive_pct(st.session_state.vp.get("cg_height", 300.0),
                        st.session_state.vp.get("wheelbase", 1550.0))
_as = kin.anti_squat_pct(st.session_state.vp.get("cg_height", 300.0),
                         st.session_state.vp.get("wheelbase", 1550.0))
mcols = st.columns(6)
mitems = [
    ("Motion ratio", f"{_mr:.3f}" if np.isfinite(_mr) else "—", "spring/wheel",
     "good" if _mr_real else "warn"),
    ("MR source", "rocker" if _mr_real else "proxy", "",
     "good" if _mr_real else "warn"),
    ("Wheel rate", f"{_wr:.1f}" if np.isfinite(_wr) else "—",
     f"N/mm @{_spring_demo:.0f}", ""),
    ("Anti-dive", f"{_ad:+.0f}" if np.isfinite(_ad) else "—", "%",
     "good" if (np.isfinite(_ad) and 0 <= _ad <= 50) else "warn"),
    ("Anti-squat", f"{_as:+.0f}" if np.isfinite(_as) else "—", "%",
     "good" if (np.isfinite(_as) and 0 <= _as <= 60) else "warn"),
    ("SVA length", f"{kin.side_view_swing_arm_length():.0f}"
     if np.isfinite(kin.side_view_swing_arm_length()) else "∞", "mm", ""),
]
for c, (k, v, u, cls) in zip(mcols, mitems):
    c.markdown(metric(k, v, u, cls), unsafe_allow_html=True)
if not _mr_real:
    st.markdown('<span class="tag warn">motion ratio is a direct-acting proxy — '
                'enable “Pushrod-actuated” in the sidebar and enter your rocker '
                'geometry for real spring→wheel rates</span>', unsafe_allow_html=True)

st.write("")
with st.expander("👋 New here? Start here (30-second tour)", expanded=False):
    st.markdown("""
**What KinematiK is:** an architecture-agnostic suspension studio (born FSAE
double-wishbone, now a general multibody platform) plus a shared team workspace.
It turns *any* suspension geometry into live kinematics and vehicle-level balance,
runs grip / lap-time / transient analysis on top, and keeps a searchable record of
*why* the team made its design decisions so that knowledge doesn't vanish at graduation.

**Pick your topology first:** the sidebar **Suspension topology** selector switches
the whole studio between double-wishbone (full live hardpoint editor), MacPherson
strut, multi-link, trailing / semi-trailing arm, solid axle, twist-beam, a
heavy-truck steering linkage, or an experimental free-form layout. Every topology
feeds the *same* analysis pipeline below.

**The tabs, in order:**
- **Kinematics** — camber gain, bump steer, caster, KPI, scrub, motion ratio vs travel.
- **Roll & Load Transfer** — roll-centre height & migration, lateral load-transfer split.
- **Grip Balance** — limit understeer/oversteer from the load-sensitive tire model.
- **Geometry 3D** — live 3D view of the linkage; non-wishbone topologies draw their own member set.
- **Compliance (Flex)** — member axial deflection → compliance steer/camber (double-wishbone member set; switch back to double-wishbone to use it).
- **Team Fit** — load the chassis once, load your part, get a collision/clearance verdict before you cut anything.
- **Weight & Handover** — log decisions, track weight, build the next-team handover record.
- **Lead Notes** — leave notes for another subteam.
- **Tire & Grip** — fit your tire from TTC data so grip/balance run on *your* rubber, not a generic default.
- **Setup Optimiser** — which change actually buys grip, given one set of tires.
- **Lap Time** — turn grip and balance into the score that matters: lap time.
- **GGV Diagram** — the combined accel/brake/cornering envelope at every speed, on your live tire; sweep CG, camber, ClA, power to see what reshapes it, and cross-check it against the Lap Sim.
- **Transient** — explicit high-frequency time-step solver for the unsteady stuff (turn-in, kerbs, dampers).
- **Validation** — correlation against logged/track data so a sim result is believable.
- **Integration** — CAD/tool interchange, plus the suspension-vs-chassis clearance-through-travel check.
- **Electronics (PCB)** — copper-survival and signal-integrity checks for the EV harness/ECU: trace heating, fusing, IR-drop brown-out, diff-pair impedance and HV coupling.

**The one habit that makes this worth it:** log your decisions as you make them —
especially the things that *didn't* work. It takes ten seconds with the templates,
and it's the difference between next year starting ahead or relearning everything.
    """)

_tabs = st.tabs(
    ["  KINEMATICS  ", "  ROLL & LOAD TRANSFER  ", "  GRIP BALANCE  ",
     "  GEOMETRY 3D  ", "  COMPLIANCE (FLEX)  ", "  TEAM FIT  ",
     "  WEIGHT & HANDOVER  ", "  LEAD NOTES  ",
     "  TIRE & GRIP  ", "  SETUP OPTIMISER  ", "  LAP TIME  ", "  GGV DIAGRAM  ", "  TRANSIENT  ",
     "  VALIDATION  ", "  INTEGRATION  ", "  ELECTRONICS (PCB)  "])
# Map the existing tab variable names onto the new (merged) tab order so the tab
# bodies below don't all need renumbering. SUSPENSION vs CHASSIS is no longer a
# top-level tab — its CAD fit/clearance check now lives inside the merged
# INTEGRATION tab (tab13) as a sub-view, rendered by render_suspension_vs_chassis().
# tab5c is the flexible-body compliance view (ADAMS Flex-style).
# tab_tr is the explicit transient time-step solver (the unsteady half of the lap).
# tab_pcb is the electronics layer: copper-survival (IPC-2221 heating, Onderdonk
# fusing, IR-drop / ECU brown-out) + signal-integrity (diff-pair impedance and
# HV-aggressor coupling), rendered by render_pcb_board().
(tab1, tab2, tab3, tab4, tab5c, tab6, tab7, tab8, tab9, tab10, tab11, tab_ggv,
 tab_tr, tab12, tab13, tab_pcb) = _tabs

# Global live notifier: polls the shared store and toasts every session when any
# lead posts a note, on whatever tab they're currently viewing. Runs on its own
# timer (fragment) so it doesn't re-run the whole app. Rendered outside the tab
# blocks so the toast + unread banner reach all users everywhere in the UI.
_note_notification_fragment()


travels = [st_.travel for st_ in sweep]

# ----------------------------- TAB 1 --------------------------------------- #
with tab1:
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=travels, y=[st_.camber for st_ in sweep],
                  mode="lines", line=dict(color=CYAN, width=3), name="Camber"))
    fig.update_layout(**PLOT_LAYOUT, title="Camber vs wheel travel",
                      xaxis_title="travel (mm, + bump)", yaxis_title="camber (°)",
                      height=340)
    c1.plotly_chart(fig, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=travels, y=[st_.toe for st_ in sweep],
                   mode="lines", line=dict(color=AMBER, width=3), name="Toe"))
    fig2.update_layout(**PLOT_LAYOUT, title="Bump steer (toe vs travel)",
                       xaxis_title="travel (mm, + bump)", yaxis_title="toe (°, + out)",
                       height=340)
    c2.plotly_chart(fig2, width='stretch')

    c3, c4 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=travels, y=[st_.scrub_radius for st_ in sweep],
                   mode="lines", line=dict(color="#9b8cff", width=3)))
    fig3.update_layout(**PLOT_LAYOUT, title="Scrub radius vs travel",
                       xaxis_title="travel (mm)", yaxis_title="scrub (mm)", height=320)
    c3.plotly_chart(fig3, width='stretch')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=travels, y=[st_.caster for st_ in sweep],
                   mode="lines", line=dict(color="#62d27a", width=3)))
    fig4.update_layout(**PLOT_LAYOUT, title="Caster vs travel",
                       xaxis_title="travel (mm)", yaxis_title="caster (°)", height=320)
    c4.plotly_chart(fig4, width='stretch')

    st.markdown('<p class="hint">Camber gain should be negative in bump so the '
                'outside wheel keeps its contact patch flat as the car rolls. Aim to '
                'keep bump steer under ~0.1°/10 mm — non-zero toe change with travel '
                'steers the car over bumps and under load.</p>', unsafe_allow_html=True)

# ----------------------------- TAB 2 --------------------------------------- #
with tab2:
    rc_heights = []
    for st_ in sweep:
        kin._tmp = st_
        rc_heights.append(veh.roll_center_height(kin, veh.p.track_front))
    # roll-centre vs travel needs a per-state RC; approximate via IC migration
    rc_static = veh.roll_center_height(kin, veh.p.track_front)

    c1, c2 = st.columns([1.3, 1])
    # load transfer vs lateral g
    gs = np.linspace(0, 1.8, 30)
    fl, fr, rl, rr = [], [], [], []
    for g in gs:
        ld, _ = veh.lateral_load_transfer(g)
        fl.append(ld.fl); fr.append(ld.fr); rl.append(ld.rl); rr.append(ld.rr)
    figL = go.Figure()
    figL.add_trace(go.Scatter(x=gs, y=fr, name="Front outer", line=dict(color=CYAN, width=3)))
    figL.add_trace(go.Scatter(x=gs, y=fl, name="Front inner", line=dict(color=CYAN, width=1.5, dash="dot")))
    figL.add_trace(go.Scatter(x=gs, y=rr, name="Rear outer", line=dict(color=AMBER, width=3)))
    figL.add_trace(go.Scatter(x=gs, y=rl, name="Rear inner", line=dict(color=AMBER, width=1.5, dash="dot")))
    figL.update_layout(**PLOT_LAYOUT, title="Tire vertical load vs lateral g",
                       xaxis_title="lateral acceleration (g)", yaxis_title="vertical load (N)",
                       height=380)
    c1.plotly_chart(figL, width='stretch')

    info = veh.lateral_load_transfer(1.2)[1]
    c2.markdown(metric("Roll-centre F", f"{info['rc_front']:.0f}", "mm"), unsafe_allow_html=True)
    c2.markdown(metric("Roll-centre R", f"{info['rc_rear']:.0f}", "mm"), unsafe_allow_html=True)
    c2.markdown(metric("Body roll @1.2g", f"{info['roll_angle']:.2f}", "°",
                       "good" if info['roll_angle'] < 2.5 else "warn"), unsafe_allow_html=True)
    c2.markdown(metric("Front LLT @1.2g", f"{info['ltd_front']:.0f}", "N"), unsafe_allow_html=True)
    c2.markdown(metric("Rear LLT @1.2g", f"{info['ltd_rear']:.0f}", "N"), unsafe_allow_html=True)

    # Roll-centre migration through travel — the honest picture vs a static number.
    mt, mrc = veh.roll_center_migration(kin, veh.p.track_front, -30, 30, 21)
    figM = go.Figure()
    figM.add_trace(go.Scatter(x=mt, y=mrc, mode="lines",
                              line=dict(color="#9b8cff", width=3)))
    figM.update_layout(**PLOT_LAYOUT, title="Roll-centre height migration vs travel",
                       xaxis_title="travel (mm, + bump)", yaxis_title="RC height (mm)",
                       height=300)
    st.plotly_chart(figM, width='stretch')
    _rc_swing = max(mrc) - min(mrc) if all(np.isfinite(mrc)) else float("nan")
    st.markdown(f'<p class="hint">Across ±30 mm of travel the front roll centre moves '
                f'{_rc_swing:.0f} mm. Large RC migration means the load-transfer balance '
                f'shifts as the car heaves and rolls — a flatter curve is generally more '
                f'predictable. The load-transfer numbers above use the static RC; this '
                f'plot shows how much that assumption drifts under travel.</p>',
                unsafe_allow_html=True)

    st.markdown(f'<p class="hint">Roll centre sits {rc_static:.0f} mm above ground at '
                'the front. A higher RC reduces body roll but adds jacking and lateral '
                'scrub; most FSAE cars keep it 20–60 mm. The geometric/elastic split of '
                'load transfer is what you tune with bar stiffness and RC height to set '
                'the balance.</p>', unsafe_allow_html=True)
    st.markdown('<p class="hint" style="border-left:2px solid #5a4317;padding-left:10px;">'
                '<b>Steady-state model.</b> These numbers assume sustained cornering at '
                'the given lateral g — they capture the car loaded and balanced mid-corner, '
                'but not transient load: turn-in, trail-braking, kerb strikes, or damper '
                'behaviour. Use it for balance and geometry tuning, not for transient '
                'response.</p>', unsafe_allow_html=True)

# ----------------------------- TAB 3 --------------------------------------- #
with tab3:
    max_g = veh.max_lateral_g()
    bal, uf, ur = veh.balance_index(min(1.2, max_g))
    verdict = ("NEUTRAL", "good") if abs(bal) < 0.03 else \
              (("UNDERSTEER", "warn") if bal > 0 else ("OVERSTEER", "bad"))

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric("Max lateral grip", f"{max_g:.2f}", "g"), unsafe_allow_html=True)
    c2.markdown(metric("Balance", verdict[0], "", verdict[1]), unsafe_allow_html=True)
    c3.markdown(metric("Front/rear util", f"{uf:.2f}/{ur:.2f}", ""), unsafe_allow_html=True)

    _model = veh.grip_model_name()
    _is_default = st.session_state.get("tire_is_default", True)
    if _model == "Pacejka MF5.2" and not _is_default:
        st.markdown(f'<p class="hint" style="border-left:2px solid #2c6b3f;'
                    f'padding-left:10px;">Grip is running on the <b>Pacejka MF5.2</b> '
                    f'model fitted to <b>your tire</b> ({st.session_state.tire_source}). '
                    f'These absolute grip numbers reflect measured rubber.</p>',
                    unsafe_allow_html=True)
    elif _model == "Pacejka MF5.2":
        st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                    'padding-left:10px;">Grip is running on the <b>Pacejka MF5.2</b> '
                    'model with the <b>generic default tire</b>. Good for comparing '
                    'setups; load your TTC-fitted tire in the TIRE &amp; GRIP tab for '
                    'absolute numbers you can trust.</p>', unsafe_allow_html=True)

    gs = np.linspace(0.3, max(max_g + 0.2, 1.0), 30)
    bidx = []
    for g in gs:
        b, _, _ = veh.balance_index(g)
        bidx.append(b)
    figB = go.Figure()
    figB.add_trace(go.Scatter(x=gs, y=bidx, line=dict(color=AMBER, width=3),
                              fill="tozeroy", fillcolor="rgba(255,176,46,.08)"))
    figB.add_hline(y=0, line_color=DIM, line_dash="dash")
    figB.update_layout(**PLOT_LAYOUT,
                       title="Handling balance vs lateral g  (+ understeer / − oversteer)",
                       xaxis_title="lateral acceleration (g)", yaxis_title="balance index",
                       height=380)
    st.plotly_chart(figB, width='stretch')
    st.markdown('<p class="hint">Balance index compares how hard each axle is working. '
                'Positive means the front saturates first (push/understeer), negative '
                'means the rear lets go first (oversteer). Shift it with roll-stiffness '
                'distribution, RC heights, and weight distribution in the sidebar.</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="hint" style="border-left:2px solid #5a4317;padding-left:10px;">'
                '<b>Steady-state.</b> Balance is computed at sustained cornering with '
                'the Pacejka load-sensitive, camber-aware grip model — good for '
                'comparing setups and predicting limit balance, but not transient '
                'response (turn-in, trail-braking, kerbs, dampers). The grip number is '
                'only as trustworthy as the tire it runs on — fit yours from TTC data '
                'in the TIRE &amp; GRIP tab.</p>',
                unsafe_allow_html=True)

# ----------------------------- TAB 4 --------------------------------------- #
with tab4:
    fig3d = go.Figure()

    def seg(p, q, color, w=6, name=None):
        fig3d.add_trace(go.Scatter3d(
            x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]],
            mode="lines", line=dict(color=color, width=w),
            name=name, showlegend=name is not None))

    H = hp
    st0 = kin.static
    if _topo == "double_wishbone":
        # wishbones
        seg(H.upper_front_inner, st0.upper_outer, CYAN, name="Upper wishbone")
        seg(H.upper_rear_inner, st0.upper_outer, CYAN)
        seg(H.lower_front_inner, st0.lower_outer, AMBER, name="Lower wishbone")
        seg(H.lower_rear_inner, st0.lower_outer, AMBER)
        seg(st0.lower_outer, st0.upper_outer, "#ffffff", 7, name="Upright / kingpin")
        seg(H.tie_rod_inner, st0.tie_rod_outer, RED, 4, name="Tie rod")
        seg(st0.contact_patch, st0.wheel_center, "#6f7d8c", 3, name="Wheel")

        pts = {k: getattr(H, k) for k, _ in POINTS}
        fig3d.add_trace(go.Scatter3d(
            x=[p[0] for p in pts.values()], y=[p[1] for p in pts.values()],
            z=[p[2] for p in pts.values()], mode="markers",
            marker=dict(size=4, color="#e7ecf1"), name="Hardpoints",
            text=list(pts.keys()), hoverinfo="text+x+y+z"))
    else:
        # Architecture-agnostic rendering: every member the mechanism reports.
        _palette = [CYAN, AMBER, RED, "#9b8cff", "#5ad17a", "#ff9f43", "#37e0d0"]
        _shown = set()
        for i, (p, q, label) in enumerate(kin.render_segments()):
            if label == "Wheel":
                seg(p, q, "#6f7d8c", 3, name="Wheel")
                continue
            base = label.split()[0] if label else f"link{i}"
            nm = None if base in _shown else label
            _shown.add(base)
            seg(p, q, _palette[i % len(_palette)], 5, name=nm)
        named = kin.named_points()
        fig3d.add_trace(go.Scatter3d(
            x=[p[0] for p in named.values()], y=[p[1] for p in named.values()],
            z=[p[2] for p in named.values()], mode="markers",
            marker=dict(size=4, color="#e7ecf1"), name="Points",
            text=list(named.keys()), hoverinfo="text+x+y+z"))

    fig3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="x (rear)", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
            yaxis=dict(title="y (right)", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
            zaxis=dict(title="z (up)", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9))),
        font=dict(family="JetBrains Mono", color="#cdd6df", size=10),
        height=560, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3d, width='stretch')

# --------------------------------------------------------------------------- #
# ----- SUSPENSION vs CHASSIS (now a section of the merged INTEGRATION tab) ----- #
def _render_envelope_vs_chassis(subsys, led):
    """Static envelope-fit check of a non-suspension subsystem against the chassis."""
    it = led.get(subsys)
    have_env = it is not None and None not in (
        getattr(it, "env_x_mm", None), getattr(it, "env_y_mm", None),
        getattr(it, "env_z_mm", None))
    st.markdown(f'<p class="hint">Static fit check for <b>{subsys}</b>: does its '
                'envelope sit inside the chassis interior and clear the frame tubes? '
                'Unlike suspension, a statically-mounted subsystem doesn\'t move, so '
                'this checks the bounding box, not a travel sweep. The envelope comes '
                'from the subsystem\'s declaration in the cross-subsystem ledger — set '
                'it there if it\'s blank.</p>', unsafe_allow_html=True)

    if not have_env:
        st.markdown(f'<span class="tag warn">{subsys} has no envelope declared — add '
                    'env X/Y/Z (and optionally an origin) in the cross-subsystem '
                    'ledger first.</span>', unsafe_allow_html=True)
        return

    size = (float(it.env_x_mm), float(it.env_y_mm), float(it.env_z_mm))
    default_origin = getattr(it, "env_origin_mm", None)
    st.markdown("###### Envelope placement (min corner, mm in the chassis frame)")
    pc = st.columns(3)
    ox = pc[0].number_input("origin x", value=float(default_origin[0]) if default_origin else 0.0,
                            step=10.0, key=f"env_ox_{subsys}")
    oy = pc[1].number_input("origin y", value=float(default_origin[1]) if default_origin else 0.0,
                            step=10.0, key=f"env_oy_{subsys}")
    oz = pc[2].number_input("origin z", value=float(default_origin[2]) if default_origin else 0.0,
                            step=10.0, key=f"env_oz_{subsys}")
    _es = [units_mod.from_metric(size[i], "mm") for i in range(3)]
    _eu = units_mod.label("mm")
    _ep = 2 if units_mod.is_us() else 0
    st.caption(f"Envelope size: {_es[0]:.{_ep}f} × {_es[1]:.{_ep}f} × {_es[2]:.{_ep}f} {_eu} "
               "(from the ledger). Place its min corner above to match where it mounts.")

    up = st.file_uploader("Chassis CAD", type=["step", "stp", "stl", "obj", "glb"],
                          label_visibility="collapsed", key=f"env_cad_{subsys}")
    oc1, oc2, oc3, oc4 = st.columns(4)
    off_x = oc1.number_input("offset x (mm)", value=0.0, step=10.0, key=f"env_offx_{subsys}")
    off_y = oc2.number_input("offset y (mm)", value=0.0, step=10.0, key=f"env_offy_{subsys}")
    off_z = oc3.number_input("offset z (mm)", value=0.0, step=10.0, key=f"env_offz_{subsys}")
    cad_scale = oc4.number_input("scale (m→mm = 1000)", value=1.0, step=1.0, key=f"env_scale_{subsys}")

    if up is None:
        st.markdown('<p class="hint" style="padding-top:.5rem;">Waiting for a chassis '
                    'file (STEP most reliable). Same CAD as the suspension check — the '
                    'envelope is placed in the same frame.</p>', unsafe_allow_html=True)
        return

    import tempfile as _tf
    suffix = "." + up.name.split(".")[-1]
    with _tf.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(up.getbuffer())
        cad_path = f.name
    try:
        with st.spinner(f"Loading chassis and fitting the {subsys} envelope…"):
            mesh = chassis_mod.load_chassis(cad_path, offset=(off_x, off_y, off_z),
                                            scale=cad_scale)
            summ = chassis_mod.mesh_summary(mesh)
            res = chassis_mod.envelope_fit_check(mesh, (ox, oy, oz), size,
                                                 name=subsys, warn_mm=8.0)

        vmap = {"CLEAR": ("good", "Envelope fits and clears the frame"),
                "TIGHT": ("warn", "Within 8 mm of a tube — review before mounting"),
                "COLLISION": ("bad", "Envelope intersects the frame — reposition or resize"),
                "OUTSIDE": ("bad", "Envelope pokes outside the chassis interior")}
        vc = vmap.get(res["verdict"], ("warn", res["verdict"]))
        st.markdown(f'<div class="metric" style="margin:.4rem 0;">'
                    f'<span class="k">{subsys.upper()} FIT VERDICT</span>'
                    f'<span class="v {vc[0]}">{res["verdict"]}'
                    f'<span class="u"> · {vc[1]}</span></span></div>',
                    unsafe_allow_html=True)

        mc = st.columns(3)
        mc[0].markdown(metric("Min clearance to frame", f"{res['min_clearance_mm']:.1f}", "mm",
                              vc[0]), unsafe_allow_html=True)
        mc[1].markdown(metric("Contained in chassis", "yes" if res["contained"] else "no",
                              "", "good" if res["contained"] else "bad"), unsafe_allow_html=True)
        oob = ", ".join(res["oob_axes"]) if res["oob_axes"] else "—"
        mc[2].markdown(metric("Outside on axes", oob, "", "bad" if res["oob_axes"] else "good"),
                       unsafe_allow_html=True)

        # 3D overlay: chassis mesh + subsystem box surface
        st.markdown(f"###### {subsys} envelope overlaid on chassis")
        box_pts = chassis_mod.envelope_box_points((ox, oy, oz), size, step_mm=25.0)
        fig = go.Figure()
        vx, vy, vz = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k,
                      color="#5a6b7a", opacity=0.30, name="Chassis", flatshading=True))
        boxcolor = {"CLEAR": "#37e0d0", "TIGHT": AMBER, "COLLISION": RED, "OUTSIDE": RED}.get(
            res["verdict"], "#9b8cff")
        fig.add_trace(go.Scatter3d(x=box_pts[:, 0], y=box_pts[:, 1], z=box_pts[:, 2],
                      mode="markers", marker=dict(size=2.5, color=boxcolor),
                      name=f"{subsys} envelope"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="x", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                yaxis=dict(title="y", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                zaxis=dict(title="z", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                aspectmode="data", camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9))),
            font=dict(family="JetBrains Mono", color="#cdd6df", size=10),
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
        st.plotly_chart(fig, width='stretch')

        st.markdown(f'<p class="hint">Chassis: {summ["triangles"]:,} triangles, '
                    f'{summ["size_mm"][0]:.0f}×{summ["size_mm"][1]:.0f}×{summ["size_mm"][2]:.0f} mm. '
                    'If the envelope sits in the wrong place, adjust the origin or the '
                    'CAD offset so the frames align.</p>', unsafe_allow_html=True)

        if res["verdict"] in ("COLLISION", "TIGHT", "OUTSIDE"):
            sug = (f"{subsys}: envelope {res['verdict'].lower()} vs chassis "
                   f"(min clearance {res['min_clearance_mm']:.1f} mm"
                   + (f", outside on {', '.join(res['oob_axes'])}" if res["oob_axes"] else "")
                   + "). Flagged before mounting.")
            note = st.text_area("Decision note (edit before logging)", value=sug,
                                height=80, key=f"env_note_{subsys}")
            if st.button("＋ Log this to handover", key=f"env_log_{subsys}"):
                ok = log_decision_now(subsys, f"{subsys} {res['verdict'].lower()} vs chassis",
                                      note, author="INTEGRATION")
                st.success("Logged to handover." if ok else
                           "Couldn't write to the handover log (backend offline) — verdict still stands.")
    except Exception as e:
        st.error(f"Could not process the chassis file: {e}")
    finally:
        try:
            os.unlink(cad_path)
        except Exception:
            pass


def render_suspension_vs_chassis():
    _IFm = interfaces_mod
    _led_cad = _IFm.IntegrationLedger.from_dict(st.session_state.ledger)
    # Any physical subsystem can be checked against the chassis. Suspension is
    # special — it MOVES, so it gets a swept-clearance check. Every other physical
    # subsystem is statically mounted, so it gets an envelope-fit check (does the
    # part's bounding box sit inside the frame and clear the tubes). Data-acquisition
    # is excluded: it's wiring/loggers with no meaningful rigid envelope to fit.
    _CAD_SUBSYSTEMS = [s for s in _IFm.SUBSYSTEMS if s != "data-acquisition"]
    subsys = st.selectbox("Which subsystem to check against the chassis?",
                          _CAD_SUBSYSTEMS,
                          index=_CAD_SUBSYSTEMS.index("suspension"),
                          key="cad_subsystem")

    if subsys != "suspension":
        _render_envelope_vs_chassis(subsys, _led_cad)
        return

    st.markdown('<p class="hint">Load the team\'s chassis CAD (STEP or STL) to check '
                'two things before you cut tube: do the inboard pickups land on the '
                'frame (fit), and does the moving linkage clear the chassis through '
                'full travel (clearance). Coordinates must share the suspension origin '
                '— use the offset boxes to align the CAD if needed.</p>',
                unsafe_allow_html=True)

    up = st.file_uploader("Chassis CAD", type=["step", "stp", "stl", "obj", "glb"],
                          label_visibility="collapsed")
    oc1, oc2, oc3, oc4 = st.columns(4)
    off_x = oc1.number_input("offset x (mm)", value=0.0, step=10.0)
    off_y = oc2.number_input("offset y (mm)", value=0.0, step=10.0)
    off_z = oc3.number_input("offset z (mm)", value=0.0, step=10.0)
    cad_scale = oc4.number_input("scale (m→mm = 1000)", value=1.0, step=1.0)

    if up is None:
        st.markdown('<p class="hint" style="padding-top:.5rem;">Waiting for a chassis '
                    'file. Don\'t have the CAD handy? Export it from your assembly as '
                    'STEP — that\'s the most reliable format here.</p>',
                    unsafe_allow_html=True)
    else:
        import tempfile as _tf
        suffix = "." + up.name.split(".")[-1]
        with _tf.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(up.getbuffer())
            cad_path = f.name
        try:
            with st.spinner("Loading chassis and sweeping the linkage…"):
                mesh = chassis_mod.load_chassis(
                    cad_path, offset=(off_x, off_y, off_z), scale=cad_scale)
                summ = chassis_mod.mesh_summary(mesh)
                fit = chassis_mod.fit_check(hp, mesh, tol_mm=12.0)
                clr = chassis_mod.clearance_check(kin, mesh, warn_mm=8.0)

            verdict = clr["verdict"]
            vcolor = {"CLEAR": ("good", "Linkage clears the chassis"),
                      "TIGHT": ("warn", "Clearance below 8 mm — review before fab"),
                      "COLLISION": ("bad", "Linkage hits the chassis — fix geometry")}[verdict]
            st.markdown(f'<div class="metric" style="margin:.4rem 0;">'
                        f'<span class="k">CLEARANCE VERDICT</span>'
                        f'<span class="v {vcolor[0]}">{verdict}'
                        f'<span class="u"> · {vcolor[1]}</span></span></div>',
                        unsafe_allow_html=True)

            cL, cR = st.columns(2)
            with cL:
                st.markdown("###### Inboard pickup fit")
                for r in fit:
                    tag = "good" if r["mountable"] else "bad"
                    note = "on frame" if r["mountable"] else "off frame"
                    st.markdown(metric(r["label"], f"{r['distance_mm']:.1f}",
                                       f"mm · {note}", tag), unsafe_allow_html=True)
            with cR:
                st.markdown("###### Link clearance (min over travel)")
                order = sorted(clr["per_link"].items(),
                               key=lambda kv: kv[1]["min_clearance_mm"])
                for link, v in order:
                    tag = ("bad" if v["collision"] else
                           "warn" if v["warning"] else "good")
                    label = link.replace("_", " ")
                    st.markdown(metric(label, f"{v['min_clearance_mm']:.1f}", "mm", tag),
                                unsafe_allow_html=True)

            # 3D overlay: chassis mesh + swept linkage
            st.markdown("###### Linkage swept through travel, overlaid on chassis")
            pts, names = chassis_mod.sweep_link_points(kin, -30, 30, 11)
            fig = go.Figure()
            vx, vy, vz = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
            i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
            fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k,
                          color="#5a6b7a", opacity=0.35, name="Chassis",
                          flatshading=True))
            names_arr = np.array(names)
            palette = {"upper_wishbone_front": CYAN, "upper_wishbone_rear": CYAN,
                       "lower_wishbone_front": AMBER, "lower_wishbone_rear": AMBER,
                       "upright": "#ffffff", "tie_rod": RED, "wheel_spindle": "#9b8cff"}
            for link in np.unique(names_arr):
                m = names_arr == link
                fig.add_trace(go.Scatter3d(
                    x=pts[m, 0], y=pts[m, 1], z=pts[m, 2], mode="markers",
                    marker=dict(size=2, color=palette.get(link, "#888")),
                    name=link.replace("_", " ")))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(title="x", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    yaxis=dict(title="y", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    zaxis=dict(title="z", backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    aspectmode="data", camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9))),
                font=dict(family="JetBrains Mono", color="#cdd6df", size=10),
                height=520, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
            st.plotly_chart(fig, width='stretch')

            st.markdown(f'<p class="hint">Chassis mesh: {summ["triangles"]:,} triangles, '
                        f'bounding box {summ["size_mm"][0]:.0f}×{summ["size_mm"][1]:.0f}×'
                        f'{summ["size_mm"][2]:.0f} mm. If the linkage and chassis look '
                        f'misaligned above, adjust the offset boxes so the origins match.</p>',
                        unsafe_allow_html=True)

            if verdict in ("COLLISION", "TIGHT"):
                worst = clr["worst_link"].replace("_", " ")
                if verdict == "COLLISION":
                    sug = (f"Suspension: {worst} hits the chassis through travel "
                           f"(worst {clr['worst_clearance_mm']:.0f} mm). Geometry "
                           f"adjusted / flagged before cutting tube.")
                else:
                    sug = (f"Suspension: {worst} clears the chassis by only "
                           f"{clr['worst_clearance_mm']:.1f} mm at full travel — tight. "
                           f"Reviewed before fabrication.")
                st.markdown('<p class="hint" style="margin-top:.4rem;">⚑ Worth recording '
                            'for handover:</p>', unsafe_allow_html=True)
                edited = st.text_area("Decision note (edit before logging)",
                                      value=sug, height=80, key="autocap_susp")
                if st.button("＋ Log this to handover", key="autocap_susp_btn"):
                    log_decision_now("suspension",
                                     f"Suspension {verdict.lower()} vs chassis",
                                     edited, author="INTEGRATION")
                    st.success("Logged to project.json — visible in WEIGHT & HANDOVER.")

            sheet = chassis_mod.manufacturing_sheet(hp, kin)
            st.download_button("⬇ Manufacturing pickup schedule (.csv)", sheet,
                               file_name="kinematik_pickups.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process the chassis file: {e}")
        finally:
            try:
                os.unlink(cad_path)
            except Exception:
                pass

# --------- MOUNT-POINT CLASH (section of the merged INTEGRATION tab) -------- #
def render_mountpoint_clash():
    """
    The CAD→clash→CG chain, live: an aero member drags a single wing mounting point
    and we re-run the point-vs-keep-out clearance check (flagging the chassis master
    file clash) and re-roll the car CG through the integration ledger — in one call,
    persisted to the project store so it survives a restart.
    """
    from suspension.mountpoints import MountPoint, KeepOut, propagate_mount_move
    _MP_EMOJI = {"aerodynamics": "💛", "brakes": "🧡", "chassis": "💜", "cooling": "🩵",
                 "data-acquisition": "💚", "electrics": "💙", "powertrain": "❤️",
                 "suspension": "🩷"}
    _SUBS = ["aerodynamics", "brakes", "chassis", "cooling",
             "data-acquisition", "electrics", "powertrain", "suspension"]

    st.markdown('<p class="hint">This is the move an aero member actually makes: drag '
                'a single <b>wing mounting point</b> and see, immediately, whether it '
                'now <b>clashes with a chassis keep-out</b> (the clearance check the '
                'chassis engineer would otherwise catch at assembly) and what it does '
                'to the car <b>CG</b> the vehicle-dynamics model uses. Points and '
                'keep-outs persist with the project. It is a <i>geometric</i> check on '
                'declared points and boxes — not a CAD kernel — so a point that bolts '
                'onto a structure is allowed to touch it; everything else it must '
                'clear.</p>', unsafe_allow_html=True)

    store = project_mod.ProjectStore(PROJECT_PATH)
    geom = store.geometry
    if geom is None:
        st.error("The mount-point / keep-out geometry layer is unavailable in "
                 "this deployment (its numpy-backed module failed to import). "
                 "Other tabs work; reinstall the app dependencies (numpy) to "
                 "restore the clash check.")
        return

    # ---- editor: keep-outs and mount points ---- #
    ec = st.columns(2)
    with ec[0]:
        st.markdown("###### Keep-out volumes (reserved by a subsystem's master file)")
        with st.expander("Add / replace a keep-out", expanded=not geom.keepouts):
            kname = st.text_input("Name", key="ko_name", value="main-hoop-tube")
            kowner = st.selectbox("Owned by", _SUBS, index=_SUBS.index("chassis"),
                                  key="ko_owner")
            kl = st.columns(3)
            lo = (kl[0].number_input("lo x", value=1380.0, key="ko_lox"),
                  kl[1].number_input("lo y", value=-180.0, key="ko_loy"),
                  kl[2].number_input("lo z", value=480.0, key="ko_loz"))
            kh = st.columns(3)
            hi = (kh[0].number_input("hi x", value=1430.0, key="ko_hix"),
                  kh[1].number_input("hi y", value=180.0, key="ko_hiy"),
                  kh[2].number_input("hi z", value=1050.0, key="ko_hiz"))
            kest = st.checkbox("Estimated geometry", value=False, key="ko_est")
            if st.button("Save keep-out", key="ko_save"):
                store.set_keepout(KeepOut(kname, kowner, lo_mm=lo, hi_mm=hi,
                                          is_estimate=kest))
                store.save()
                st.rerun()
        for name, ko in list(geom.keepouts.items()):
            est = " · est" if ko.is_estimate else ""
            kc = st.columns([5, 1])
            kc[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(ko.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{ko.owner_subsystem}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'{tuple(round(v) for v in ko.lo_mm)} → {tuple(round(v) for v in ko.hi_mm)} mm</span></div>',
                unsafe_allow_html=True)
            if kc[1].button("✕", key=f"ko_del_{name}"):
                store.remove_keepout(name); store.save(); st.rerun()

    with ec[1]:
        st.markdown("###### Mount points (the hardpoints a subteam moves)")
        with st.expander("Add / replace a mount point", expanded=not geom.points):
            pname = st.text_input("Name", key="mp_name", value="rear-wing-upper-mount")
            powner = st.selectbox("Owned by", _SUBS,
                                  index=_SUBS.index("aerodynamics"), key="mp_owner")
            pmounts = st.selectbox("Mounts onto", _SUBS,
                                   index=_SUBS.index("chassis"), key="mp_mounts")
            pc = st.columns(3)
            pxyz = (pc[0].number_input("x", value=1350.0, key="mp_x"),
                    pc[1].number_input("y", value=120.0, key="mp_y"),
                    pc[2].number_input("z", value=900.0, key="mp_z"))
            pclr = st.number_input("Required clearance (mm)", 0.0, 100.0,
                                   value=8.0, key="mp_clr")
            pest = st.checkbox("Estimated geometry", value=False, key="mp_est")
            if st.button("Save mount point", key="mp_save"):
                store.set_mount_point(MountPoint(pname, xyz_mm=pxyz,
                                                 owner_subsystem=powner,
                                                 mounts_on=pmounts,
                                                 min_clearance_mm=pclr,
                                                 is_estimate=pest))
                store.save()
                st.rerun()
        for name, mp in list(geom.points.items()):
            est = " · est" if mp.is_estimate else ""
            pc = st.columns([5, 1])
            pc[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(mp.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{mp.owner_subsystem} '
                f'→ {mp.mounts_on}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'{tuple(round(v) for v in mp.xyz_mm)} mm · clr {mp.min_clearance_mm:.0f}</span></div>',
                unsafe_allow_html=True)
            if pc[1].button("✕", key=f"mp_del_{name}"):
                store.remove_mount_point(name); store.save(); st.rerun()

    # ---- the live "drag a point" propagation ---- #
    if geom.points:
        st.markdown("###### Move a mount point — propagate clash + CG in one step")
        led = interfaces_mod.IntegrationLedger.from_dict(st.session_state.ledger)
        mc = st.columns([2, 3, 1])
        which = mc[0].selectbox("Point", list(geom.points), key="mv_which")
        cur = geom.points[which].xyz_mm
        nx = mc[1].columns(3)
        new_xyz = (nx[0].number_input("→ x", value=float(cur[0]), key="mv_x"),
                   nx[1].number_input("→ y", value=float(cur[1]), key="mv_y"),
                   nx[2].number_input("→ z", value=float(cur[2]), key="mv_z"))
        also_cg = st.checkbox(
            "This point is the subsystem's mass location (shift its CG with the move)",
            value=False, key="mv_cg")
        if mc[2].button("Move →", key="mv_go"):
            res = propagate_mount_move(geom, led, which, new_xyz,
                                       set_by="integration-tab",
                                       update_interface_cg=also_cg)
            if also_cg:
                st.session_state.ledger = led.as_dict()
            store.save()
            st.session_state["_mp_last"] = res

        res = st.session_state.get("_mp_last")
        if res is not None and res.moved_point == which:
            st.markdown(f'<p class="hint">{res.summary()}</p>', unsafe_allow_html=True)

    # ---- the clash board (same render idiom as the ledger findings) ---- #
    st.markdown("###### Clash board")
    findings = geom.check_clashes()
    # check_clashes() returns an empty list ONLY when there are no mount points at
    # all (a fresh project). Without this guard the board would render as a bare
    # header with nothing beneath it — indistinguishable from a broken tab. Show an
    # explicit empty state so the board always "appears".
    if not findings:
        if not geom.points and not geom.keepouts:
            _msg = ("No mount points or keep-out volumes declared yet. Add at least "
                    "one mount point and one keep-out above (the defaults in the "
                    "forms are a ready-to-save starting pair), then the clash board "
                    "checks every point against every keep-out it must clear.")
        elif not geom.points:
            _msg = ("Keep-out volumes are declared but no mount points exist to check "
                    "against them yet — add a mount point above.")
        else:
            _msg = ("No clashes to report.")
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:4px 0;">'
            f'<span class="tag">EMPTY</span> '
            f'<span style="font-size:.92rem;color:#8d99a6">{_msg}</span></div>',
            unsafe_allow_html=True)
    _SEV_CLS = {"fail": "bad", "warning": "warn", "missing": "warn",
                "info": "", "ok": "good"}
    order = ["fail", "warning", "missing", "info", "ok"]
    for f in sorted(findings, key=lambda x: order.index(x.severity.value)):
        cls = _SEV_CLS.get(f.severity.value, "")
        who = " ↔ ".join(f"{_MP_EMOJI.get(x,'')}{x}" for x in f.subsystems) if f.subsystems else ""
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:4px 0;">'
            f'<span class="tag {cls}">{f.severity.value.upper()}</span> '
            f'<b>{f.check}</b> &nbsp;<span style="color:#8d99a6;font-size:.8rem">{who}</span><br>'
            f'<span style="font-size:.92rem">{f.message}</span></div>',
            unsafe_allow_html=True)


def render_pcb_board():
    """
    The electronics layer, live: size copper traces and route the CAN differential
    pair, then run the pre-fab gate — does the worst simultaneous load (brake light
    + both cooling fans at once) melt a trace, overheat it, or brown out the ECU,
    and does the CAN pair run too close to the HV motor-controller net? Traces,
    pairs and aggressors persist with the project, and the worst-case currents are
    rolled up from the integration ledger's declared peak currents — the same
    single source of truth the LV/HV check uses.
    """
    from suspension.electronics import Trace, DiffPair, Aggressor

    _MP_EMOJI = {"aerodynamics": "💛", "brakes": "🧡", "chassis": "💜", "cooling": "🩵",
                 "data-acquisition": "💚", "electrics": "💙", "powertrain": "❤️",
                 "suspension": "🩷", "ecu": "🖥️"}
    _SUBS = ["aerodynamics", "brakes", "chassis", "cooling",
             "data-acquisition", "electrics", "powertrain", "suspension"]
    # Only subsystems that can actually put an electrical load on a board trace
    # belong in the "Loads on <trace>" picker. Aero/chassis/suspension draw no
    # board current, so offering them just invites a nonsensical scenario.
    _LOAD_SUBS = ["brakes", "cooling", "data-acquisition", "electrics", "powertrain"]

    st.markdown(
        '<p class="hint">This is the check an electrical member runs the afternoon '
        'before sending a board to fab: under the worst <b>simultaneous load</b> '
        '(brake light + both cooling fans firing at once), do the thin copper '
        'traces <b>physically melt</b> (Onderdonk fusing), overheat past the board '
        'derate ceiling (IPC-2221), or drop enough voltage to push the rail below '
        'the <b>ECU brown-out</b> threshold and reset the car? And does a CAN '
        '<b>differential pair</b> route close enough to the switching '
        '<b>HV motor-controller</b> net to pick up its noise? It is an '
        '<i>analytic</i> screening check on declared traces and routes — not a PCB '
        'CAD kernel and not a field solver — so the impedance and coupling numbers '
        'are labelled estimates, and the true eye/reflection that need a field '
        'solver are reported as <i>not computed</i> rather than invented.</p>',
        unsafe_allow_html=True)

    store = project_mod.ProjectStore(PROJECT_PATH)
    board = store._ensure_board()
    led = interfaces_mod.IntegrationLedger.from_dict(st.session_state.ledger)

    # ---- board-level brown-out / thermal context ---- #
    st.markdown("###### Board context (rail + ambient the checks run against)")
    bc = st.columns(5)
    board.rail_nominal_v = bc[0].number_input("ECU rail (V)", 1.0, 60.0,
                                              value=float(board.rail_nominal_v), key="pcb_rail")
    board.ecu_brownout_v = bc[1].number_input("Brown-out (V)", 0.5, 60.0,
                                              value=float(board.ecu_brownout_v), key="pcb_bo")
    board.ambient_c = bc[2].number_input("Ambient (°C)", -20.0, 150.0,
                                         value=float(board.ambient_c), key="pcb_amb")
    board.max_trace_temp_c = bc[3].number_input("Trace ceiling (°C)", 50.0, 200.0,
                                               value=float(board.max_trace_temp_c), key="pcb_ceil")
    board.fuse_safety_factor = bc[4].number_input("Fuse safety ×", 1.0, 10.0,
                                                 value=float(board.fuse_safety_factor), key="pcb_sf")

    # ---- editors: traces, pairs, aggressors ---- #
    ec = st.columns(3)
    with ec[0]:
        st.markdown("###### Copper traces (power / signal nets you size)")
        with st.expander("Add / replace a trace", expanded=not board.traces):
            tn = st.text_input("Name", key="tr_name", value="main_feed")
            tnet = st.text_input("Net", key="tr_net", value="lv_rail")
            town = st.selectbox("Owned by", _SUBS, index=_SUBS.index("electrics"), key="tr_own")
            tfeed = st.text_input("Feeds", key="tr_feed", value="ecu")
            tg = st.columns(3)
            tw = tg[0].number_input("Width (mm)", 0.05, 50.0, value=0.30, key="tr_w")
            toz = tg[1].number_input("Copper (oz)", 0.25, 6.0, value=1.0, key="tr_oz")
            tl = tg[2].number_input("Length (mm)", 1.0, 2000.0, value=100.0, key="tr_l")
            text = st.checkbox("Outer layer (external)", value=True, key="tr_ext")
            test = st.checkbox("Estimated geometry", value=False, key="tr_est")
            if st.button("Save trace", key="tr_save"):
                store.set_trace(Trace(name=tn, net=tnet, owner_subsystem=town,
                                      feeds=tfeed, width_mm=tw, copper_oz=toz,
                                      length_mm=tl, is_external=text, is_estimate=test))
                store.save(); st.rerun()
        for name, tr in list(board.traces.items()):
            est = " · est" if tr.is_estimate else ""
            row = st.columns([5, 1])
            row[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(tr.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{tr.owner_subsystem} → {tr.feeds}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'{tr.width_mm:.2f} mm · {tr.copper_oz:.0f} oz · {tr.length_mm:.0f} mm · '
                f'fuses @ {tr.fusing_current_a(ambient_c=board.ambient_c):.1f} A</span></div>',
                unsafe_allow_html=True)
            if row[1].button("✕", key=f"tr_del_{name}"):
                store.remove_trace(name); store.save(); st.rerun()

    with ec[1]:
        st.markdown("###### Differential pairs (CAN H/L routes)")
        with st.expander("Add / replace a pair", expanded=not board.pairs):
            pn = st.text_input("Name", key="dp_name", value="CAN")
            pown = st.selectbox("Owned by", _SUBS, index=_SUBS.index("electrics"), key="dp_own")
            pg = st.columns(3)
            pw = pg[0].number_input("Trace w (mm)", 0.05, 5.0, value=0.20, key="dp_w")
            psp = pg[1].number_input("Spacing (mm)", 0.05, 5.0, value=0.20, key="dp_s")
            ph = pg[2].number_input("Dielectric h (mm)", 0.05, 5.0, value=0.20, key="dp_h")
            pg2 = st.columns(2)
            peps = pg2[0].number_input("eps_r", 1.0, 15.0, value=4.3, key="dp_eps")
            ptz = pg2[1].number_input("Target Zdiff (Ω)", 20.0, 200.0, value=120.0, key="dp_tz")
            ppath = st.text_input("Route (x,y; x,y; …)", key="dp_path", value="0,0; 60,0")
            pest = st.checkbox("Estimated routing", value=False, key="dp_est")
            if st.button("Save pair", key="dp_save"):
                pts = _parse_path(ppath)
                store.set_pair(DiffPair(name=pn, owner_subsystem=pown, trace_w_mm=pw,
                                        spacing_mm=psp, height_mm=ph, eps_r=peps,
                                        target_z0_ohm=ptz, path_mm=pts, is_estimate=pest))
                store.save(); st.rerun()
        for name, dp in list(board.pairs.items()):
            est = " · est" if dp.is_estimate else ""
            z = dp.differential_z0_ohm()
            row = st.columns([5, 1])
            row[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(dp.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{dp.owner_subsystem}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'~{z:.0f} Ω diff (est) · target {dp.target_z0_ohm:.0f} Ω · {len(dp.path_mm)} pts</span></div>',
                unsafe_allow_html=True)
            if row[1].button("✕", key=f"dp_del_{name}"):
                store.remove_pair(name); store.save(); st.rerun()

    with ec[2]:
        st.markdown("###### Aggressor nets (noisy HV traces to avoid)")
        with st.expander("Add / replace an aggressor", expanded=not board.aggressors):
            an = st.text_input("Name", key="ag_name", value="INV")
            anet = st.text_input("Net", key="ag_net", value="hv_inverter")
            aown = st.selectbox("Owned by", _SUBS, index=_SUBS.index("powertrain"), key="ag_own")
            ag2 = st.columns(2)
            asw = ag2[0].number_input("Switched V", 1.0, 1000.0, value=400.0, key="ag_v")
            aedge = ag2[1].number_input("Edge (V/ns)", 0.1, 100.0, value=5.0, key="ag_e")
            apath = st.text_input("Route (x,y; x,y; …)", key="ag_path", value="0,0.3; 60,0.3")
            aest = st.checkbox("Estimated routing", value=False, key="ag_est")
            if st.button("Save aggressor", key="ag_save"):
                pts = _parse_path(apath)
                store.set_aggressor(Aggressor(name=an, net=anet, owner_subsystem=aown,
                                              sw_voltage_v=asw, edge_v_per_ns=aedge,
                                              path_mm=pts, is_estimate=aest))
                store.save(); st.rerun()
        for name, ag in list(board.aggressors.items()):
            est = " · est" if ag.is_estimate else ""
            row = st.columns([5, 1])
            row[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(ag.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{ag.owner_subsystem} · {ag.net}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'{ag.sw_voltage_v:.0f} V · {ag.edge_v_per_ns:.0f} V/ns · {len(ag.path_mm)} pts</span></div>',
                unsafe_allow_html=True)
            if row[1].button("✕", key=f"ag_del_{name}"):
                store.remove_aggressor(name); store.save(); st.rerun()

    # ---- the simultaneous-load scenario ---- #
    st.markdown("###### Worst-case simultaneous load — which subsystems fire at once, onto which trace")
    st.markdown(
        '<p class="hint">Each trace can carry several loads at once. Pick the '
        'subsystems whose peak currents sum onto it (e.g. cooling + cooling + '
        'brakes = both fans and the brake light). The per-load amps come from the '
        '<b>integration ledger</b> (each subsystem\'s declared peak current), not '
        'retyped here — so this can\'t drift from what TEAM FIT says.</p>',
        unsafe_allow_html=True)
    scenario = {}
    if board.traces:
        for name, tr in board.traces.items():
            # Drop any stale stored selection no longer in the allowed loads
            # (e.g. aerodynamics from before the picker was restricted), or the
            # widget errors. Use key only — passing default= alongside an existing
            # key makes Streamlit reset the selection to [] on rerun.
            _skey = f"pcb_scn_{name}"
            if _skey in st.session_state:
                st.session_state[_skey] = [
                    s for s in st.session_state[_skey] if s in _LOAD_SUBS]
            picks = st.multiselect(
                f"Loads on '{name}'", _LOAD_SUBS,
                key=_skey,
                help="Choose the same subsystem twice (e.g. two fans) by it appearing once "
                     "per declared load — duplicates are summed.")
            if picks:
                scenario[name] = picks
    else:
        st.caption("Add at least one trace to define a load scenario.")

    # ---- the pre-fab gate ---- #
    st.markdown("###### Pre-fab board check")
    res = store.board_check(ledger=led, scenario=scenario)
    if res.findings:
        st.markdown(f'<p class="hint">{res.summary()}</p>', unsafe_allow_html=True)
    else:
        st.caption("Nothing to check yet — declare traces/pairs and a load scenario above.")

    _SEV_CLS = {"fail": "bad", "warning": "warn", "missing": "warn",
                "info": "", "ok": "good"}
    order = ["fail", "warning", "missing", "info", "ok"]
    for f in sorted(res.findings, key=lambda x: order.index(x.severity.value)):
        cls = _SEV_CLS.get(f.severity.value, "")
        who = " ↔ ".join(f"{_MP_EMOJI.get(x,'')}{x}" for x in f.subsystems) if f.subsystems else ""
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:4px 0;">'
            f'<span class="tag {cls}">{f.severity.value.upper()}</span> '
            f'<b>{f.check}</b> &nbsp;<span style="color:#8d99a6;font-size:.8rem">{who}</span><br>'
            f'<span style="font-size:.92rem">{f.message}</span></div>',
            unsafe_allow_html=True)

    # ---- honest SI detail: what's analytic vs what needs a field solver ---- #
    if board.pairs:
        st.markdown("###### Signal-integrity detail (analytic vs field-solver)")
        which = st.selectbox("Pair", list(board.pairs), key="pcb_si_which")
        d = board.si_detail(which)
        sc = st.columns(2)
        sc[0].markdown(
            f"**Computed (analytic, IPC-2141 + edge coupling)**\n\n"
            f"- single-ended Z₀: {d['single_ended_z0_ohm']:.1f} Ω\n"
            f"- differential Z₀: {d['differential_z0_ohm']:.1f} Ω")
        sc[1].markdown(
            "**Not computed — needs a field solver / SPICE**\n\n"
            "- eye height: *not computed*\n"
            "- insertion loss: *not computed*\n"
            "- reflection coeff: *not computed*\n"
            "- coupled noise voltage: *not computed*")
        st.caption(d["note"])


def _parse_path3d(s: str):
    """Parse 'x,y,z; x,y,z; …' into a list of (x,y,z) float tuples. Tolerant of
    stray whitespace, trailing separators, and 2-D points (z defaults to 0)."""
    pts = []
    for chunk in str(s).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                x = float(parts[0]); y = float(parts[1])
                z = float(parts[2]) if len(parts) >= 3 else 0.0
                pts.append((x, y, z))
            except ValueError:
                continue
    return pts


def _formboard_svg(fb, width_px: int = 720, height_px: int = 360) -> str:
    """Render a Formboard as an inline 1:1-proportioned SVG nail-board drawing.
    Branches are drawn as polylines (length-true), connector nodes as labelled
    pads. The drawing is scaled to fit the panel but the proportions are the true
    flat layout — the shape a fabricator pins the loom out on."""
    ext_w, ext_h = fb.extent_mm
    ext_w = max(ext_w, 1.0); ext_h = max(ext_h, 1.0)
    pad = 28
    sx = (width_px - 2 * pad) / ext_w
    sy = (height_px - 2 * pad) / ext_h
    s = min(sx, sy)  # uniform scale -> proportions preserved (true 1:1 shape)

    def tx(p):
        return pad + p[0] * s, pad + (ext_h - p[1]) * s  # flip y for screen

    palette = ["#4aa3df", "#e1683c", "#5cb85c", "#b18ad6", "#d9a441",
               "#48b8b8", "#d76a8a", "#7d8c99"]
    parts = [f'<svg viewBox="0 0 {width_px} {height_px}" '
             f'style="width:100%;height:auto;background:#0e1419;'
             f'border:1px solid var(--line);border-radius:8px;">']
    # scale bar (100 mm)
    bar = 100.0 * s
    parts.append(f'<line x1="{pad}" y1="{height_px-12}" x2="{pad+bar}" '
                 f'y2="{height_px-12}" stroke="#8d99a6" stroke-width="2"/>'
                 f'<text x="{pad}" y="{height_px-16}" fill="#8d99a6" '
                 f'font-size="10">100 mm</text>')
    for i, b in enumerate(fb.branches):
        col = palette[i % len(palette)]
        pts = " ".join(f"{tx(p)[0]:.1f},{tx(p)[1]:.1f}" for p in b.points_mm)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                     f'stroke-width="2.4" stroke-linejoin="round" '
                     f'stroke-linecap="round" opacity="0.92"/>')
        # label at midpoint
        if b.points_mm:
            mid = b.points_mm[len(b.points_mm) // 2]
            mx, my = tx(mid)
            parts.append(f'<text x="{mx+4:.1f}" y="{my-4:.1f}" fill="{col}" '
                         f'font-size="10" font-weight="600">{b.wire} '
                         f'(AWG{b.gauge_awg})</text>')
    for name, (x, y) in fb.nodes.items():
        nx, ny = tx((x, y))
        parts.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="5" fill="#e8edf2" '
                     f'stroke="#0e1419" stroke-width="1.5"/>'
                     f'<text x="{nx+7:.1f}" y="{ny+3:.1f}" fill="#e8edf2" '
                     f'font-size="10" font-weight="700">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_harness():
    """
    The 3-D wiring harness, live: route every conductor through the same car
    coordinates the suspension geometry uses, catch a bend tighter than the wire
    can take, a connector with no strain relief, and a loom that fouls a reserved
    keep-out — then, before a single wire is cut, read off the exact cut length to
    the millimetre, the 1:1 formboard, the automated BOM, and the exact copper
    mass and where it sits on the car.
    """
    from suspension.harness import Connector, WireRun

    _MP_EMOJI = {"aerodynamics": "💛", "brakes": "🧡", "chassis": "💜", "cooling": "🩵",
                 "data-acquisition": "💚", "electrics": "💙", "powertrain": "❤️",
                 "suspension": "🩷", "ecu": "🖥️"}
    _SUBS = ["aerodynamics", "brakes", "chassis", "cooling",
             "data-acquisition", "electrics", "powertrain", "suspension"]

    st.markdown("---")
    st.markdown("### 🧵 Harness — the physical loom in 3-D car space")
    st.markdown(
        '<p class="hint">The board above is the copper <i>on the PCB</i>. This is '
        'the copper <i>between the boxes</i> — the loom you lay into the chassis. '
        'Route each conductor as a 3-D polyline through the <b>same car '
        'coordinates</b> the suspension mount-points and keep-outs live in, so a '
        'wire that would foul a wishbone or the accumulator box shows up as a '
        'clearance FAIL on the same board a mount clash does. It catches the two '
        'things that actually scrap a loom on the bench — a <b>bend tighter than '
        'the wire can take</b> (kinks the conductor) and a connector entry with no '
        '<b>strain-relief</b> straight length — then derives, <i>before you cut a '
        'single wire</i>: the exact <b>cut length</b> to the millimetre, a 1:1 '
        '<b>formboard</b>, the automated <b>BOM</b>, and the exact <b>copper mass '
        'and its distribution</b>. It measures the route you declare; it is not a '
        'cable-flex solver, so unsupported sag under vibration is reported as '
        '<i>not computed</i> rather than invented.</p>',
        unsafe_allow_html=True)

    store = project_mod.ProjectStore(PROJECT_PATH)
    harness = store._ensure_harness()
    geom = getattr(store, "geometry", None)
    keepouts = list(getattr(geom, "keepouts", {}).values()) if geom else []

    hc = st.columns(3)
    harness.ambient_c = hc[0].number_input("Ambient (°C)", -20.0, 150.0,
                                           value=float(harness.ambient_c), key="hn_amb")
    harness.clearance_warn_mm = hc[1].number_input(
        "Clearance WARN (mm)", 0.0, 100.0,
        value=float(harness.clearance_warn_mm), key="hn_cw")
    harness.clearance_fail_mm = hc[2].number_input(
        "Clearance FAIL (mm)", 0.0, 50.0,
        value=float(harness.clearance_fail_mm), key="hn_cf")

    if keepouts:
        st.caption(f"Routing against {len(keepouts)} keep-out volume(s) from the "
                   f"geometry ledger: " +
                   ", ".join(f"{_MP_EMOJI.get(getattr(k,'owner_subsystem','?'),'')}"
                             f"{k.name}" for k in keepouts))
    else:
        st.caption("No keep-out volumes declared yet — add them in the geometry / "
                   "mount-point tab to clearance-check the loom against reserved space.")

    ec = st.columns(2)
    # ---- connector editor ---- #
    with ec[0]:
        st.markdown("###### Connectors (harness end-points / branch nodes)")
        with st.expander("Add / replace a connector", expanded=not harness.connectors):
            cn = st.text_input("Name", key="cn_name", value="ECU")
            cown = st.selectbox("Owned by", _SUBS, index=_SUBS.index("electrics"),
                                key="cn_own")
            cg = st.columns(3)
            cx = cg[0].number_input("x (mm)", -5000.0, 5000.0, value=0.0, key="cn_x")
            cy = cg[1].number_input("y (mm)", -5000.0, 5000.0, value=0.0, key="cn_y")
            cz = cg[2].number_input("z (mm)", -5000.0, 5000.0, value=0.0, key="cn_z")
            cg2 = st.columns(3)
            ccav = cg2[0].number_input("Cavities", 1, 200, value=12, key="cn_cav")
            csr = cg2[1].number_input("Strain relief (mm)", 0.0, 200.0,
                                      value=25.0, key="cn_sr")
            cmass = cg2[2].number_input("Mass (g, 0=unknown)", 0.0, 2000.0,
                                        value=0.0, key="cn_mass")
            cpn = st.text_input("Part number", key="cn_pn", value="DTM-12")
            cest = st.checkbox("Estimated", value=False, key="cn_est")
            if st.button("Save connector", key="cn_save"):
                store.set_connector(Connector(
                    name=cn, owner_subsystem=cown, xyz_mm=(cx, cy, cz),
                    cavities=int(ccav), part_number=cpn, strain_relief_mm=csr,
                    mass_g=(None if cmass == 0.0 else cmass), is_estimate=cest))
                store.save(); st.rerun()
        for name, c in list(harness.connectors.items()):
            est = " · est" if c.is_estimate else ""
            mtxt = f"{c.mass_g:.0f} g" if c.mass_g is not None else "mass —"
            row = st.columns([5, 1])
            row[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(c.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">{c.owner_subsystem} · '
                f'{c.part_number or "—"}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'({c.xyz_mm[0]:.0f}, {c.xyz_mm[1]:.0f}, {c.xyz_mm[2]:.0f}) mm · '
                f'{c.cavities} cav · {mtxt} · SR {c.strain_relief_mm:.0f} mm</span></div>',
                unsafe_allow_html=True)
            if row[1].button("✕", key=f"cn_del_{name}"):
                store.remove_connector(name); store.save(); st.rerun()

    # ---- wire editor ---- #
    with ec[1]:
        st.markdown("###### Wire runs (3-D routed conductors)")
        conn_names = [""] + list(harness.connectors.keys())
        with st.expander("Add / replace a wire", expanded=not harness.wires):
            wn = st.text_input("Name", key="wr_name", value="MOT_PWR")
            wown = st.selectbox("Owned by", _SUBS, index=_SUBS.index("electrics"),
                                key="wr_own")
            wg = st.columns(3)
            wawg = wg[0].number_input("Gauge (AWG)", 8, 30, value=10, key="wr_awg")
            wnet = wg[1].text_input("Net", key="wr_net", value="hv_pwr")
            wod = wg[2].number_input("OD (mm, 0=AWG nom)", 0.0, 30.0,
                                     value=0.0, key="wr_od")
            wg2 = st.columns(2)
            wfrom = wg2[0].selectbox("From connector", conn_names, key="wr_from")
            wto = wg2[1].selectbox("To connector", conn_names, key="wr_to")
            wg3 = st.columns(3)
            wmult = wg3[0].number_input("Min bend ×OD", 1.0, 20.0,
                                        value=6.0, key="wr_mult")
            wloop = wg3[1].number_input("Service loop (mm)", 0.0, 1000.0,
                                        value=0.0, key="wr_loop")
            wstrip = wg3[2].number_input("Strip/end (mm)", 0.0, 100.0,
                                         value=8.0, key="wr_strip")
            wpath = st.text_input("3-D route (x,y,z; x,y,z; …)", key="wr_path",
                                  value="0,0,0; 40,0,0; 600,0,100; 1160,0,100; 1200,0,0")
            wcur = st.number_input("Carries current (A, 0=from net)", 0.0, 1000.0,
                                   value=0.0, key="wr_cur")
            west = st.checkbox("Estimated route", value=False, key="wr_est")
            if st.button("Save wire", key="wr_save"):
                pts = _parse_path3d(wpath)
                store.set_wire(WireRun(
                    name=wn, owner_subsystem=wown, gauge_awg=int(wawg),
                    path_mm=pts, from_conn=wfrom, to_conn=wto, net=wnet,
                    od_mm=(None if wod == 0.0 else wod),
                    bundle_min_radius_mult=wmult, service_loop_mm=wloop,
                    strip_mm=wstrip,
                    carries_current_a=(None if wcur == 0.0 else wcur),
                    is_estimate=west))
                store.save(); st.rerun()
        for name, w in list(harness.wires.items()):
            est = " · est" if w.is_estimate else ""
            row = st.columns([5, 1])
            row[0].markdown(
                f'<div style="border-left:3px solid var(--line);padding:4px 10px;margin:3px 0;">'
                f'{_MP_EMOJI.get(w.owner_subsystem,"")} <b>{name}</b> '
                f'<span style="color:#8d99a6;font-size:.8rem">AWG{w.gauge_awg} · '
                f'{w.net or "—"}{est}</span><br>'
                f'<span style="font-size:.82rem;color:#8d99a6">'
                f'{w.from_conn or "?"} → {w.to_conn or "?"} · '
                f'cut {w.cut_length_mm():.0f} mm · {w.copper_mass_g():.1f} g Cu · '
                f'min bend {w.min_bend_radius_mm:.0f} mm</span></div>',
                unsafe_allow_html=True)
            if row[1].button("✕", key=f"wr_del_{name}"):
                store.remove_wire(name); store.save(); st.rerun()

    # ---- the pre-cut gate ---- #
    st.markdown("###### Pre-cut harness check")
    res = store.harness_check()
    if res.findings:
        st.markdown(f'<p class="hint">{res.summary()}</p>', unsafe_allow_html=True)
    else:
        st.caption("Add connectors and at least one routed wire to run the check.")

    _SEV_CLS = {"fail": "bad", "warning": "warn", "missing": "warn",
                "info": "", "ok": "good"}
    order = ["fail", "warning", "missing", "info", "ok"]
    for f in sorted(res.findings, key=lambda x: order.index(x.severity.value)):
        cls = _SEV_CLS.get(f.severity.value, "")
        who = " ↔ ".join(f"{_MP_EMOJI.get(x,'')}{x}" for x in f.subsystems) if f.subsystems else ""
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:4px 0;">'
            f'<span class="tag {cls}">{f.severity.value.upper()}</span> '
            f'<b>{f.check}</b> &nbsp;<span style="color:#8d99a6;font-size:.8rem">{who}</span><br>'
            f'<span style="font-size:.92rem">{f.message}</span></div>',
            unsafe_allow_html=True)

    if not harness.wires:
        return

    # ---- manufacturing artefacts: cut list + BOM + mass + formboard ---- #
    st.markdown("###### Manufacturing roll-up — everything known before the first cut")
    mc = st.columns([3, 2])

    with mc[0]:
        st.markdown("**Cut list (to the millimetre)**")
        if res.cut_list:
            import pandas as pd
            df = pd.DataFrame(res.cut_list)[
                ["wire", "net", "gauge_awg", "from_conn", "to_conn",
                 "routed_mm", "bend_allowance_mm", "service_loop_mm",
                 "strip_both_ends_mm", "cut_length_mm"]]
            st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown("**Bill of materials (automated)**")
        bom = res.bom
        if bom.get("wire"):
            import pandas as pd
            st.caption("Wire by gauge")
            st.dataframe(pd.DataFrame(bom["wire"]), hide_index=True,
                         use_container_width=True)
        if bom.get("connectors"):
            import pandas as pd
            st.caption("Connectors")
            st.dataframe(pd.DataFrame(bom["connectors"]), hide_index=True,
                         use_container_width=True)
        st.markdown(
            f'<p class="hint">Totals: <b>{bom.get("total_wire_m",0):.2f} m</b> wire · '
            f'<b>{bom.get("total_copper_g",0):.0f} g</b> copper · '
            f'<b>{bom.get("contacts_total",0)}</b> crimp contacts.</p>',
            unsafe_allow_html=True)

    with mc[1]:
        st.markdown("**Copper mass & distribution**")
        md = res.mass
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:3px 0;">'
            f'<span style="font-size:.95rem"><b>{md.get("total_copper_g",0):.1f} g</b> '
            f'copper · <b>{md.get("total_harness_g",0):.1f} g</b> total harness</span><br>'
            f'<span style="font-size:.85rem;color:#8d99a6">harness CG: '
            f'{md.get("harness_cg_mm")} mm (x rearward, y right, z up)</span></div>',
            unsafe_allow_html=True)
        if md.get("connectors_without_declared_mass"):
            st.caption("Excluded from CG (mass not declared): " +
                       ", ".join(md["connectors_without_declared_mass"]))
        if md.get("per_wire"):
            import pandas as pd
            st.dataframe(pd.DataFrame(md["per_wire"])[
                ["wire", "gauge_awg", "copper_g", "total_g"]],
                hide_index=True, use_container_width=True)
        st.caption("Sag of unsupported runs under vibration: *not computed* — "
                   "needs a flexible-body solver; the route is measured, not solved.")

    # ---- the 1:1 formboard ---- #
    st.markdown("###### 1:1 Formboard (unfolded flat layout for the bench)")
    fb = res.formboard
    if fb and fb.branches:
        st.markdown(
            f'<p class="hint">The harness unfolded to a length-true 2-D nail-board: '
            f'every branch segment is the exact length of its 3-D run, so the '
            f'fabricator pins the loom out 1:1. Board extent '
            f'~{fb.extent_mm[0]:.0f} × {fb.extent_mm[1]:.0f} mm.</p>',
            unsafe_allow_html=True)
        st.markdown(_formboard_svg(fb), unsafe_allow_html=True)
    else:
        st.caption("Route at least one wire with ≥2 points to generate the formboard.")


def _parse_path(s: str):
    """Parse 'x,y; x,y; …' into a list of (x,y) float tuples. Tolerant of stray
    whitespace and trailing separators; returns [] on anything unparseable."""
    pts = []
    for chunk in str(s).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


# ----------------------------- TAB 5c (COMPLIANCE / FLEX) ------------------ #
with tab5c:
  if _topo != "double_wishbone":
    st.info("The flex / compliance solver models axial deflection of the "
            "double-wishbone member set (upper & lower arms + tie rod). For "
            "the selected topology, the architecture-agnostic engine reports "
            "its own member set on the GEOMETRY 3D tab; per-link compliance "
            "for arbitrary topologies is not yet wired into this tab. Switch "
            "to the double-wishbone model to use the flex analysis.")
  else:
    st.markdown('<p class="hint">The rigid model treats every link as infinitely '
                'stiff. Real control arms, pushrods and tie rods stretch under load, '
                'and the chassis tabs flex too — at 1.5 g that shows up as '
                '<b>compliance steer</b> and <b>compliance camber</b> you never asked '
                'for. This tab resolves the member loads at a cornering case, deflects '
                'each link by its axial stiffness, and re-solves the geometry to read '
                'off how much the wheel actually moves. Define link stiffness from tube '
                'size, or import a condensed FEA body (ADAMS&nbsp;Flex-style).</p>',
                unsafe_allow_html=True)

    csL, csR = st.columns([1, 1])
    with csL:
        st.markdown('<p class="hint" style="margin-bottom:4px;"><b>Load case</b></p>',
                    unsafe_allow_html=True)
        comp_g = st.slider("Lateral acceleration (g)", 0.5, 2.5, 1.5, 0.1,
                           key="comp_g")
        comp_axle = st.selectbox("Axle", ["front", "rear"], key="comp_axle")
        comp_long = st.slider("Longitudinal g (braking +, traction −)",
                              -1.5, 1.5, 0.0, 0.1, key="comp_long")
    with csR:
        st.markdown('<p class="hint" style="margin-bottom:4px;"><b>Link stiffness</b>'
                    '</p>', unsafe_allow_html=True)
        comp_mat = st.selectbox("Tube material", list(MATERIALS.keys()),
                                key="comp_mat")
        comp_od = st.number_input("Tube OD (mm)", 8.0, 40.0, 19.05, 0.05,
                                  key="comp_od")
        comp_wall = st.number_input("Tube wall (mm)", 0.4, 4.0, 0.9, 0.05,
                                    key="comp_wall")
        comp_use_tab = st.checkbox("Add chassis-tab compliance (series)",
                                   value=False, key="comp_use_tab")
        comp_ktab = None
        if comp_use_tab:
            comp_ktab = st.number_input("Tab stiffness per leg (N/mm)",
                                        500.0, 100000.0, 8000.0, 500.0,
                                        key="comp_ktab")

    # ---- Non-linear joints: bushings / rod ends / spherical bearings ----- #
    st.markdown('<p class="hint" style="margin-top:10px;border-left:2px solid #25506b;'
                'padding-left:10px;"><b>Joints (bushings / rod ends / spherical '
                'bearings).</b> The rigid model treats every joint as zero-play. Give '
                'the inboard pickups and outboard joints a real non-linear '
                'force–displacement curve — rubber or polyurethane bushings, or a '
                'spherical bearing with a little free-play (lash). On a tube car the '
                'joints, not the links, are usually the dominant give, and the tie-rod '
                'joint is what sets compliance steer.</p>',
                unsafe_allow_html=True)
    comp_use_joints = st.checkbox("Model joint compliance", value=False, key="cj_use")

    from suspension import JointCompliance as _JCui

    def _joint_from_ui(prefix, label, default_kind):
        kinds = ["None", "Rubber bushing", "Polyurethane bushing",
                 "Spherical bearing", "Custom (cubic)"]
        kind = st.selectbox(label, kinds, index=kinds.index(default_kind),
                            key=f"{prefix}_kind")
        if kind == "None":
            return None
        if kind in ("Rubber bushing", "Polyurethane bushing"):
            is_rub = kind.startswith("Rubber")
            c1, c2, c3 = st.columns(3)
            k = c1.number_input("radial rate k₁ (N/mm)", 100.0, 50000.0,
                                1500.0 if is_rub else 6000.0, 100.0, key=f"{prefix}_k")
            hard = c2.number_input("hardening ×k₁ (/mm²)", 0.0, 50.0,
                                   8.0 if is_rub else 4.0, 0.5, key=f"{prefix}_h")
            loss = c3.number_input("loss factor η", 0.0, 0.5,
                                   0.12 if is_rub else 0.05, 0.01, key=f"{prefix}_l")
            return _JCui.cubic(k, hard * k, loss_factor=loss, label=kind)
        if kind == "Spherical bearing":
            c1, c2, c3 = st.columns(3)
            lash = c1.number_input("lash (mm)", 0.0, 0.5, 0.05, 0.01,
                                   key=f"{prefix}_lash")
            k = c2.number_input("engaged k (N/mm)", 10000.0, 500000.0, 120000.0,
                                5000.0, key=f"{prefix}_k")
            loss = c3.number_input("loss factor η", 0.0, 0.2, 0.01, 0.01,
                                   key=f"{prefix}_l")
            return _JCui.spherical_bearing(lash_mm=lash, k=k, loss_factor=loss)
        c1, c2, c3 = st.columns(3)   # Custom (cubic)
        k1 = c1.number_input("k₁ (N/mm)", 50.0, 200000.0, 3000.0, 50.0, key=f"{prefix}_k1")
        k3 = c2.number_input("k₃ (N/mm³)", 0.0, 500000.0, 0.0, 100.0, key=f"{prefix}_k3")
        loss = c3.number_input("loss factor η", 0.0, 0.5, 0.05, 0.01, key=f"{prefix}_l")
        return _JCui.cubic(k1, k3, loss_factor=loss, label="custom")

    joint_in_ui = joint_out_ui = tie_in_ui = tie_out_ui = None
    if comp_use_joints:
        jc1, jc2 = st.columns(2)
        with jc1:
            st.markdown('<p class="hint"><b>Wishbone inboard</b> (chassis pickups)</p>',
                        unsafe_allow_html=True)
            joint_in_ui = _joint_from_ui("cj_wbi", "Inboard joint", "Rubber bushing")
        with jc2:
            st.markdown('<p class="hint"><b>Wishbone outboard</b> (ball joints)</p>',
                        unsafe_allow_html=True)
            joint_out_ui = _joint_from_ui("cj_wbo", "Outboard joint", "Spherical bearing")
        cj_tie_same = st.checkbox("Use the same joints on the tie rod", value=True,
                                  key="cj_tie_same")
        if cj_tie_same:
            tie_in_ui, tie_out_ui = joint_in_ui, joint_out_ui
        else:
            jt1, jt2 = st.columns(2)
            with jt1:
                tie_in_ui = _joint_from_ui("cj_ti", "Tie-rod inboard", "Rubber bushing")
            with jt2:
                tie_out_ui = _joint_from_ui("cj_to", "Tie-rod outboard", "Spherical bearing")

    # Optional FEA flex-body import (ADAMS Flex-equivalent). A loaded body replaces
    # the analytic tube stiffness for the members it names.
    st.markdown('<p class="hint" style="margin-top:10px;border-left:2px solid #25506b;'
                'padding-left:10px;"><b>FEA flex body (optional).</b> Import a condensed '
                'flexible body as <code>.flex.json</code> — either a beam/bar mesh we '
                'assemble and Guyan-reduce, or a pre-reduced superelement (the interface '
                'nodes + condensed stiffness an ADAMS&nbsp;Flex MNF carries). It is used '
                'for any member whose two endpoints you map to body nodes below.</p>',
                unsafe_allow_html=True)
    flex_up = st.file_uploader("Flexible body (.flex.json)", type=["json", "flex"],
                               key="comp_flex")
    flex_body = None
    flex_map = {}
    if flex_up is not None:
        try:
            import json as _json
            _d = _json.load(flex_up)
            flex_body = load_flex_body(_d)
            names = list(flex_body.names)
            st.success(f"Loaded flex body with interface nodes: {', '.join(names)}")
            fm1, fm2, fm3 = st.columns(3)
            fb_member = fm1.selectbox("Apply to member",
                                      ["UF", "UR", "LF", "LR", "TR"], key="fb_member")
            fb_out = fm2.selectbox("Outboard node", names, key="fb_out")
            fb_in = fm3.selectbox("Inboard node", names,
                                  index=min(1, len(names) - 1), key="fb_in")
            flex_map[fb_member] = (fb_out, fb_in)
        except NotImplementedError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not read flex body: {e}")

    # ---- build the compliant corner and solve --------------------------- #
    try:
        comp_kin = kin if comp_axle == "front" else kin   # same geometry both axles here
        stiff = {}
        for m in ("UF", "UR", "LF", "LR"):
            stiff[m] = MemberStiffness(material=comp_mat, od_mm=comp_od,
                                       wall_mm=comp_wall, k_tab=comp_ktab,
                                       joint_in=joint_in_ui, joint_out=joint_out_ui)
        stiff["TR"] = MemberStiffness(material=comp_mat, od_mm=comp_od,
                                      wall_mm=comp_wall,
                                      joint_in=tie_in_ui, joint_out=tie_out_ui)
        # overlay any FEA-backed members (keeping their joints)
        for m, (n_out, n_in) in flex_map.items():
            ji = tie_in_ui if m == "TR" else joint_in_ui
            jo = tie_out_ui if m == "TR" else joint_out_ui
            stiff[m] = MemberStiffness(flex_body=flex_body, node_out=n_out,
                                       node_in=n_in, joint_in=ji, joint_out=jo,
                                       k_tab=comp_ktab
                                       if (comp_use_tab and m != "TR") else None)
        corner = CompliantCorner(comp_kin.hp, stiff)
        load = corner_wheel_load(veh, comp_axle, comp_g, outer=True,
                                 long_g=comp_long)
        res = corner.solve(load)
    except Exception as e:
        st.error(f"Compliance solve failed: {e}")
        res = None

    if res is not None:
        st.markdown(f'<p class="hint" style="margin-top:14px;"><b>Compliance at '
                    f'{comp_g:.1f} g — {comp_axle} outer wheel.</b> '
                    f'Contact-patch load: Fz {load.Fz:.0f} N, Fy {load.Fy:.0f} N.'
                    f'</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        toe = res.compliance_toe
        cam = res.compliance_camber
        # FSAE rule of thumb: > ~0.15° compliance steer at the front is worth chasing.
        toe_cls = "good" if abs(toe) < 0.15 else ("warn" if abs(toe) < 0.4 else "bad")
        cam_cls = "good" if abs(cam) < 0.2 else ("warn" if abs(cam) < 0.5 else "bad")
        m1.markdown(metric("Compliance toe", f"{toe:+.3f}", "°", toe_cls),
                    unsafe_allow_html=True)
        m2.markdown(metric("Compliance camber", f"{cam:+.3f}", "°", cam_cls),
                    unsafe_allow_html=True)
        m3.markdown(metric("Patch lateral shift",
                           f"{res.contact_patch_lateral_shift_mm:+.2f}", "mm"),
                    unsafe_allow_html=True)
        m4.markdown(metric("Converged",
                           "yes" if res.converged else "NO",
                           f"{res.summary()['iterations']} it",
                           "good" if res.converged else "bad"),
                    unsafe_allow_html=True)

        # member force / deflection bar chart
        members = [m for m in ("UF", "UR", "LF", "LR", "TR", "PR")
                   if m in res.member_forces]
        forces = [res.member_forces[m] for m in members]
        defls = [res.member_deflection.get(m, float("nan")) for m in members]
        figF = make_subplots(specs=[[{"secondary_y": True}]])
        figF.add_trace(go.Bar(x=members, y=forces, name="Axial force (N)",
                              marker_color=CYAN, opacity=0.85), secondary_y=False)
        figF.add_trace(go.Scatter(x=members, y=defls, name="Deflection (mm)",
                                  mode="markers", marker=dict(color=AMBER, size=11)),
                       secondary_y=True)
        figF.update_layout(**PLOT_LAYOUT, title="Member axial force & deflection",
                           height=340, barmode="group")
        figF.update_yaxes(title_text="axial force (N, + tension)", secondary_y=False)
        figF.update_yaxes(title_text="deflection (mm, + stretch)", secondary_y=True)
        st.plotly_chart(figF, width='stretch')

        # ---- joint-specific views: where the give is, the curve, the damping --- #
        if comp_use_joints and res.member_joint_deflection:
            bd = [m for m in ("UF", "UR", "LF", "LR", "TR")
                  if m in res.member_joint_deflection]
            link_d = [res.member_joint_deflection[m]["link"] for m in bd]
            jin_d = [res.member_joint_deflection[m]["joint_in"] for m in bd]
            jout_d = [res.member_joint_deflection[m]["joint_out"] for m in bd]
            figB = go.Figure()
            figB.add_trace(go.Bar(x=bd, y=link_d, name="link", marker_color=DIM))
            figB.add_trace(go.Bar(x=bd, y=jin_d, name="inboard joint",
                                  marker_color=CYAN))
            figB.add_trace(go.Bar(x=bd, y=jout_d, name="outboard joint",
                                  marker_color=AMBER))
            figB.update_layout(**PLOT_LAYOUT, barmode="relative", height=320,
                               title="Where the give is — link vs joints (mm)")
            figB.update_yaxes(title_text="axial give (mm, + stretch)")
            st.plotly_chart(figB, width='stretch')

            jshow = tie_in_ui or joint_in_ui or joint_out_ui
            if jshow is not None:
                dd = np.linspace(-0.6, 0.6, 161)
                ff = [jshow.force(float(x)) for x in dd]
                figJ = go.Figure(go.Scatter(x=dd, y=ff, mode="lines",
                                            line=dict(color=CYAN, width=3)))
                figJ.update_layout(**PLOT_LAYOUT, height=300,
                                   title=f"Joint force–displacement curve "
                                         f"({jshow.label or jshow.kind})",
                                   xaxis_title="displacement (mm)",
                                   yaxis_title="force (N, + tension)")
                st.plotly_chart(figJ, width='stretch')

            try:
                dsum = corner.damping_summary(load, amplitude_mm=0.3, freq_hz=15.0)
                st.markdown(
                    f'<p class="hint">Joint damping at 0.3&nbsp;mm / 15&nbsp;Hz '
                    f'dissipates <b>~{dsum["total_energy_per_cycle_mJ"]:.1f}&nbsp;mJ '
                    f'per cycle</b> across all joints. Damping does no work in this '
                    f'steady solve — it is reported here and exported (via '
                    f'<code>linearized_rates</code>) for the transient model, where it '
                    f'actually bites.</p>', unsafe_allow_html=True)
            except Exception:
                pass

        # compliance steer vs lateral g sweep
        gs_c = np.linspace(0.2, max(2.0, comp_g), 18)
        toes, cams = [], []
        for g in gs_c:
            try:
                _ld = corner_wheel_load(veh, comp_axle, g, outer=True,
                                        long_g=comp_long)
                _r = corner.solve(_ld)
                toes.append(_r.compliance_toe if _r.converged else float("nan"))
                cams.append(_r.compliance_camber if _r.converged else float("nan"))
            except Exception:
                toes.append(float("nan")); cams.append(float("nan"))
        figS = go.Figure()
        figS.add_trace(go.Scatter(x=gs_c, y=toes, name="Compliance toe",
                                  line=dict(color=CYAN, width=3)))
        figS.add_trace(go.Scatter(x=gs_c, y=cams, name="Compliance camber",
                                  line=dict(color=AMBER, width=3)))
        figS.update_layout(**PLOT_LAYOUT,
                           title="Compliance steer / camber vs lateral g",
                           xaxis_title="lateral acceleration (g)",
                           yaxis_title="angle change (°)", height=320)
        st.plotly_chart(figS, width='stretch')

        sign_txt = ("toe-out" if toe > 0 else "toe-in")
        st.markdown(f'<p class="hint">At {comp_g:.1f} g this corner deflects into '
                    f'<b>{abs(toe):.3f}° {sign_txt}</b> and {cam:+.3f}° camber. '
                    f'Compliance steer changes the slip angle the tyre actually runs, '
                    f'so it shifts balance and can make the car feel vague or darty even '
                    f'when the kinematic bump steer is perfect. Stiffer tabs, larger '
                    f'tube, or a triangulated tie-rod pickup all pull it back toward '
                    f'zero.</p>', unsafe_allow_html=True)
        if abs(toe) < 1e-4:
            st.markdown('<p class="hint">Bare steel-tube axial stiffness is enormous, '
                        'so with rigid joints the compliance is tiny — which is the '
                        'honest answer. The dominant real-world give is in the joints '
                        'and chassis tabs: tick <b>"Model joint compliance"</b> to add '
                        'rubber/poly bushings or a spherical bearing with lash, tick '
                        '"chassis-tab compliance", or import an FEA body that captures '
                        'the bracket bending, to see realistic numbers.</p>',
                        unsafe_allow_html=True)
        st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                    'padding-left:10px;"><b>Steady-state, quasi-static.</b> This is the '
                    'car held at one cornering load — the constraint-mode (static) '
                    'content an MNF carries, which is what governs load↔deflection mid- '
                    'corner. It is not a transient or NVH model: no damper dynamics, no '
                    'modal response, no kerb strikes.</p>', unsafe_allow_html=True)

# ----------------------------- TAB 6 --------------------------------------- #
with tab6:
    st.markdown('<p class="hint">Any Elbee subteam: load the shared chassis once as '
                'the reference, then load your part (caliper, radiator, battery box, '
                'wing mount, ECU tray — anything). You get the same collision / tight / '
                'clear verdict suspension gets. <b>We can\'t out-spend USC, so we '
                'out-integrate them</b> — catch the interference here before the first '
                'cut, because rework is the tax for not integrating in CAD.</p>',
                unsafe_allow_html=True)

    tcol1, tcol2 = st.columns(2)
    team_keys = list(integ_mod.TEAMS.keys())
    team = tcol1.selectbox("Your subteam", team_keys,
                           format_func=lambda k: integ_mod.TEAMS[k]["label"])
    part_name = tcol2.text_input("Part name", value="my_part")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("###### Shared chassis (reference)")
        chassis_up = st.file_uploader("Chassis CAD", type=["step", "stp", "stl", "obj", "glb"],
                                      key="team_chassis", label_visibility="collapsed")
    with rc2:
        st.markdown(f"###### {integ_mod.TEAMS[team]['label']} part")
        part_up = st.file_uploader("Part CAD", type=["step", "stp", "stl", "obj", "glb"],
                                   key="team_part", label_visibility="collapsed")

    st.markdown("###### Position your part in the chassis frame")
    pc = st.columns(7)
    p_ox = pc[0].number_input("x mm", value=0.0, step=10.0, key="p_ox")
    p_oy = pc[1].number_input("y mm", value=0.0, step=10.0, key="p_oy")
    p_oz = pc[2].number_input("z mm", value=0.0, step=10.0, key="p_oz")
    p_rx = pc[3].number_input("rot x°", value=0.0, step=15.0, key="p_rx")
    p_ry = pc[4].number_input("rot y°", value=0.0, step=15.0, key="p_ry")
    p_rz = pc[5].number_input("rot z°", value=0.0, step=15.0, key="p_rz")
    p_scale = pc[6].number_input("scale", value=1.0, step=1.0, key="p_scale")

    if chassis_up is None or part_up is None:
        st.markdown('<p class="hint" style="padding-top:.5rem;">Load both the chassis '
                    'and your part to run the check. Only chassis and suspension have '
                    'CAD right now — as your team produces geometry, this works the same '
                    'way for you. Export STEP from your assembly for best results.</p>',
                    unsafe_allow_html=True)
    else:
        import tempfile as _tf
        def _save(uploaded):
            sfx = "." + uploaded.name.split(".")[-1]
            with _tf.NamedTemporaryFile(suffix=sfx, delete=False) as f:
                f.write(uploaded.getbuffer())
                return f.name
        ch_path = _save(chassis_up)
        pt_path = _save(part_up)
        try:
            with st.spinner("Loading geometry and checking interference…"):
                ref = integ_mod.load_part(ch_path)
                part = integ_mod.load_part(
                    pt_path, offset=(p_ox, p_oy, p_oz), scale=p_scale,
                    rotate_deg=(p_rx, p_ry, p_rz))
                res = integ_mod.interference_check(part, ref, warn_mm=5.0)
                psum = integ_mod.part_summary(part)

            vmap = {"CLEAR": ("good", "Part clears the chassis"),
                    "TIGHT": ("warn", "Under 5 mm — review before fab"),
                    "COLLISION": ("bad", "Part intersects the chassis — reposition")}
            tag, msg = vmap[res["verdict"]]
            st.markdown(f'<div class="metric" style="margin:.4rem 0;">'
                        f'<span class="k">INTERFERENCE VERDICT · {integ_mod.TEAMS[team]["label"].upper()}</span>'
                        f'<span class="v {tag}">{res["verdict"]}'
                        f'<span class="u"> · {msg}</span></span></div>',
                        unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(metric("Min clearance", f"{res['min_clearance_mm']:.1f}", "mm", tag),
                         unsafe_allow_html=True)
            mc2.markdown(metric("Part overlap", f"{res['collision_fraction']*100:.0f}", "%",
                                "bad" if res['collision_fraction'] > 0 else "good"),
                         unsafe_allow_html=True)
            mc3.markdown(metric("Part size",
                                f"{psum['size_mm'][0]:.0f}×{psum['size_mm'][1]:.0f}×{psum['size_mm'][2]:.0f}",
                                "mm"), unsafe_allow_html=True)

            if res["verdict"] in ("COLLISION", "TIGHT"):
                tlabel = integ_mod.TEAMS[team]["label"]
                if res["verdict"] == "COLLISION":
                    suggested = (f"{tlabel}: {part_name} intersects the chassis "
                                 f"(overlap {res['collision_fraction']*100:.0f}%, "
                                 f"worst point {res['min_clearance_mm']:.0f} mm inside). "
                                 f"Repositioned / flagged for redesign before fabrication.")
                else:
                    suggested = (f"{tlabel}: {part_name} clears the chassis by only "
                                 f"{res['min_clearance_mm']:.1f} mm — below the 5 mm "
                                 f"margin. Reviewed for clearance before fabrication.")
                st.markdown('<p class="hint" style="margin-top:.4rem;">⚑ This is worth '
                            'recording for handover — log it so next year knows the '
                            'constraint existed, and ping the team that owns what it '
                            'hits:</p>', unsafe_allow_html=True)
                edited = st.text_area("Decision note (edit before logging)",
                                      value=suggested, height=80, key="autocap_team")
                ncol = st.columns([1.6, 1.4, 1.4])
                notify_opts = ["(don't notify)"] + list(integ_mod.TEAMS.keys())
                default_idx = notify_opts.index("chassis") if "chassis" in notify_opts else 0
                notify_team = ncol[0].selectbox(
                    "Notify team", notify_opts, index=default_idx,
                    format_func=lambda k: k if k == "(don't notify)"
                    else integ_mod.TEAMS[k]["label"], key="notify_team")
                notify_urgent = ncol[1].checkbox("Mark urgent", key="notify_urgent",
                                                 value=(res["verdict"] == "COLLISION"))
                note_author = ncol[2].text_input("Your name", key="notify_author")
                if st.button("＋ Log to handover" +
                             (" & notify" if notify_team != "(don't notify)" else ""),
                             key="autocap_team_btn"):
                    _s = project_mod.ProjectStore(PROJECT_PATH)
                    _s.add_decision(project_mod.Decision(
                        team=team, title=f"{part_name} chassis {res['verdict'].lower()}",
                        rationale=edited, author="TEAM FIT", tags="auto-captured"))
                    posted = ""
                    if notify_team != "(don't notify)":
                        _s.add_note(project_mod.Note(
                            from_team=team, to_team=notify_team,
                            message=(f"{part_name} {res['verdict'].lower()} vs chassis "
                                     f"(min {res['min_clearance_mm']:.1f} mm). {edited}"),
                            author=note_author or "TEAM FIT",
                            is_request=True, urgent=notify_urgent))
                        posted = f" · note sent to {integ_mod.TEAMS[notify_team]['label']}"
                    _s.save()
                    st.success(f"Logged to handover{posted}.")

            # Auto-populate the weight budget from this part's CAD volume
            if psum.get("volume_mm3"):
                st.markdown('<p class="hint" style="margin-top:.4rem;">This part is '
                            'watertight, so its mass can be estimated from CAD volume — '
                            'log it straight into the weight budget:</p>',
                            unsafe_allow_html=True)
                awc = st.columns([1.6, 1, 1])
                aw_mat = awc[0].selectbox("Material", list(project_mod.MATERIALS.keys()),
                                          key="awmat")
                aw_qty = awc[1].number_input("Qty", value=1, min_value=1, step=1, key="awqty")
                est = project_mod.estimate_mass_g(psum["volume_mm3"], aw_mat)
                awc[2].markdown(metric("Est. mass each",
                                       f"{est:.0f}" if est else "—", "g"),
                                unsafe_allow_html=True)
                if est and st.button("＋ Add to weight budget", key="aw_btn"):
                    s_ = project_mod.ProjectStore(PROJECT_PATH)
                    s_.add_weight(project_mod.WeightItem(
                        team=team, name=part_name, mass_g=float(est), qty=int(aw_qty),
                        material=aw_mat, source="cad_estimate"))
                    s_.save()
                    st.success(f"Added {part_name} ({est:.0f} g × {aw_qty}) to the budget.")
                elif not est:
                    st.markdown('<p class="hint">Pick a material with a known density to '
                                'estimate mass (or use manual entry in WEIGHT & HANDOVER '
                                'for hollow/lattice parts).</p>', unsafe_allow_html=True)

            fig = go.Figure()
            for mesh, color, name, opac in [(ref, "#5a6b7a", "Chassis", 0.30),
                                            (part, integ_mod.TEAMS[team]["color"], part_name, 0.65)]:
                v = mesh.vertices
                f = mesh.faces
                fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                              i=f[:, 0], j=f[:, 1], k=f[:, 2],
                              color=color, opacity=opac, name=name, flatshading=True))
            if res["worst_point"] and res["verdict"] != "CLEAR":
                wp = res["worst_point"]
                fig.add_trace(go.Scatter3d(x=[wp[0]], y=[wp[1]], z=[wp[2]],
                              mode="markers", marker=dict(size=6, color=RED),
                              name="Worst point"))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    yaxis=dict(backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    zaxis=dict(backgroundcolor="#0e1216", gridcolor="#1d242c", color="#8d99a6"),
                    aspectmode="data", camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9))),
                font=dict(family="JetBrains Mono", color="#cdd6df", size=10),
                height=520, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(fig, width='stretch')
            st.markdown('<p class="hint">If the part is in the wrong place relative to '
                        'the chassis, adjust the offset and rotation above until it sits '
                        'where it mounts. The red dot marks the tightest/worst point so '
                        'you know which corner to move.</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not process the files: {e}")
        finally:
            for p in (ch_path, pt_path):
                try:
                    os.unlink(p)
                except Exception:
                    pass

# ----------------------------- TAB 7 --------------------------------------- #
with tab7:
    store = project_mod.ProjectStore(PROJECT_PATH)

    # Surface storage problems instead of silently losing data.
    _degraded = getattr(store.backend, "degraded_reason", None)
    if _degraded:
        st.error(f"⚠ {_degraded}")
    if getattr(store, "load_error", None):
        st.error(f"⚠ {store.load_error}")

    # Tell the user whether their data is persisting or session-only.
    _is_persistent = type(store.backend).__name__ == "SupabaseBackend"
    if _is_persistent:
        st.markdown('<span class="tag good">● persistent storage — data survives '
                    'restarts</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag warn">● local/session storage — set up Supabase '
                    'for permanent team data (see README)</span>', unsafe_allow_html=True)

    st.markdown('<p class="hint">The lightest reliable car is the advantage money '
                'can\'t buy — and the reasoning behind your design is the thing a team '
                'loses every graduation. This page is the persistent record: it saves '
                'to <code>project.json</code> in the project folder, so commit that file '
                'to the repo and the knowledge survives the handover.</p>',
                unsafe_allow_html=True)

    hcol1, hcol2, hcol3 = st.columns(3)
    store.team_name = hcol1.text_input("Team", value=store.team_name)
    store.season = hcol2.text_input("Season", value=store.season)
    store.target_mass_kg = hcol3.number_input("Target mass (kg)",
                                              value=float(store.target_mass_kg), step=5.0)

    b = store.budget_status()
    bcol = st.columns(3)
    bcol[0].markdown(metric("Current mass", f"{b['total_kg']:.1f}", "kg"), unsafe_allow_html=True)
    bcol[1].markdown(metric("Target", f"{b['target_kg']:.0f}", "kg"), unsafe_allow_html=True)
    bcol[2].markdown(metric("Delta", f"{b['delta_kg']:+.1f}", "kg",
                            "bad" if b["over_budget"] else "good"), unsafe_allow_html=True)

    if store.mass_by_team():
        figW = go.Figure()
        teams = list(store.mass_by_team().keys())
        masses = list(store.mass_by_team().values())
        colors = [integ_mod.TEAMS.get(t, {}).get("color", "#888") for t in teams]
        figW.add_trace(go.Bar(x=masses, y=teams, orientation="h",
                              marker_color=colors))
        figW.update_layout(**PLOT_LAYOUT, title="Mass by subteam (kg)",
                           height=max(220, 40 * len(teams)), xaxis_title="kg",
                           yaxis_title="")
        st.plotly_chart(figW, width='stretch')

    st.markdown("###### Log a part's mass")
    wc = st.columns([1.2, 1.4, 0.7, 1, 1.4, 1])
    w_team = wc[0].selectbox("Team", list(integ_mod.TEAMS.keys()),
                             format_func=lambda k: integ_mod.TEAMS[k]["label"], key="w_team")
    w_name = wc[1].text_input("Part name", key="w_name")
    w_qty = wc[2].number_input("Qty", value=1, min_value=1, step=1, key="w_qty")
    w_mass = wc[3].number_input("Mass each (g)", value=0.0, step=10.0, key="w_mass")
    w_mat = wc[4].selectbox("Material", list(project_mod.MATERIALS.keys()), key="w_mat")
    w_src = wc[5].selectbox("Source", ["manual", "cad_estimate"], key="w_src")
    if st.button("+ Add part", width='content'):
        if w_name and w_mass > 0:
            store.add_weight(project_mod.WeightItem(
                team=w_team, name=w_name, mass_g=float(w_mass), qty=int(w_qty),
                material=w_mat, source=w_src))
            store.save()
            st.rerun()
        else:
            st.warning("Enter a part name and a mass above zero.")

    if store.weights:
        st.markdown("###### Logged parts")
        for i, w in enumerate(store.weights):
            cc = st.columns([2, 3, 1, 1.5, 1.5, 0.8])
            cc[0].markdown(f"<span class='tag'>{integ_mod.TEAMS.get(w.team,{}).get('label',w.team)}</span>",
                           unsafe_allow_html=True)
            cc[1].write(w.name)
            cc[2].write(f"×{w.qty}")
            cc[3].write(f"{w.mass_g:.0f} g")
            cc[4].write(f"= {w.total_g/1000:.2f} kg")
            if cc[5].button("✕", key=f"del_{i}"):
                store.remove_weight(i)
                store.save()
                st.rerun()

    st.markdown("---")
    st.markdown("###### Log a design decision")
    st.markdown('<p class="hint">This is the section next year\'s team thanks you for. '
                'Write down <i>why</i>, not just what — the reasoning is what gets lost.</p>',
                unsafe_allow_html=True)

    # ---- Quick-add: one-tap templates to kill logging friction ----------
    QUICK_TEMPLATES = {
        "⚙ Geometry change": ("Geometry change", "changed-geometry",
                              "Changed [what] from [old] to [new] because [reason]. "
                              "Trade-off: [what it costs]."),
        "🔧 Material / part choice": ("Material choice", "material",
                              "Chose [material/part] for [component] because [reason]. "
                              "Considered [alternative] but [why not]."),
        "⚠ Interference found": ("Interference found", "interference",
                              "[Part] interferes with [what] at [condition]. "
                              "Resolved by [action] / flagged for [who]."),
        "🧪 Test result": ("Test result", "test",
                              "Tested [what]. Result: [outcome]. "
                              "Means we should [implication]."),
        "❌ Didn't work": ("Didn't work", "rejected",
                              "Tried [approach] for [goal]. Didn't work because [reason]. "
                              "Avoid repeating — instead [what to do]."),
    }
    st.markdown('<p class="hint" style="margin-bottom:.2rem;">Quick start — tap a '
                'template, then just fill in the brackets:</p>', unsafe_allow_html=True)
    qcols = st.columns(len(QUICK_TEMPLATES))
    for i, (label, (title, tag, body)) in enumerate(QUICK_TEMPLATES.items()):
        if qcols[i].button(label, key=f"qt_{i}", width='stretch'):
            # Seed the widget keys directly, before the widgets are created below.
            st.session_state["d_title"] = title
            st.session_state["d_tags"] = tag
            st.session_state["d_rationale"] = body
            st.rerun()

    dc = st.columns([1.2, 2, 1.2])
    d_team = dc[0].selectbox("Team", list(integ_mod.TEAMS.keys()),
                             format_func=lambda k: integ_mod.TEAMS[k]["label"], key="d_team")
    d_title = dc[1].text_input("Decision", key="d_title")
    d_author = dc[2].text_input("Author", key="d_author")
    d_rationale = st.text_area("Rationale — why this choice, what were the trade-offs",
                               key="d_rationale", height=90)
    tc = st.columns([1.4, 1.4])
    d_part = tc[0].text_input("Part / system (e.g. front upright, radiator)", key="d_part",
                              placeholder="what this decision is about")
    d_tags = tc[1].text_input("Tags (comma-separated)", key="d_tags",
                              placeholder="roll-centre, front, packaging…")
    if st.button("+ Log decision"):
        if d_title and d_rationale:
            store.add_decision(project_mod.Decision(
                team=d_team, title=d_title, rationale=d_rationale, author=d_author,
                tags=d_tags, part=d_part))
            store.save()
            for k in ("d_title", "d_tags", "d_rationale", "d_part"):
                st.session_state.pop(k, None)
            st.rerun()
        else:
            st.warning("Enter a decision title and rationale.")

    if store.decisions:
        st.markdown("###### Search the decision log")
        sc = st.columns([2.2, 1.2, 1.2, 1.2])
        d_query = sc[0].text_input("Search", key="dec_search",
                                   placeholder="search title, rationale, author, tags, part…",
                                   label_visibility="collapsed")
        team_opts = ["all teams"] + list(integ_mod.TEAMS.keys())
        d_fteam = sc[1].selectbox("Team", team_opts, key="dec_fteam",
                                  format_func=lambda k: "All teams" if k == "all teams"
                                  else integ_mod.TEAMS[k]["label"], label_visibility="collapsed")
        tag_opts = ["all tags"] + store.all_decision_tags()
        d_ftag = sc[2].selectbox("Tag", tag_opts, key="dec_ftag",
                                 format_func=lambda k: "All tags" if k == "all tags" else k,
                                 label_visibility="collapsed")
        part_opts = ["all parts"] + store.all_decision_parts()
        d_fpart = sc[3].selectbox("Part", part_opts, key="dec_fpart",
                                  format_func=lambda k: "All parts" if k == "all parts" else k,
                                  label_visibility="collapsed")

        results = store.search_decisions(
            query=d_query,
            team=None if d_fteam == "all teams" else d_fteam,
            tag=None if d_ftag == "all tags" else d_ftag,
            part=None if d_fpart == "all parts" else d_fpart)

        st.markdown(f"<p class='hint'>{len(results)} of {len(store.decisions)} "
                    f"decisions</p>", unsafe_allow_html=True)

        for d in results:
            meta = f"{integ_mod.TEAMS.get(d.team,{}).get('label',d.team)} · {d.date}"
            if d.author:
                meta += f" · {d.author}"
            dpart = getattr(d, "part", "") or ""
            if dpart:
                meta += f" · ⛭ {dpart}"
            auto = "<span class='tag good' style='margin-left:6px;'>auto-captured</span>" \
                if "auto" in (d.tags or "") else ""
            # render user tags as chips (excluding the internal auto-captured marker)
            chips = ""
            for t in (d.tags or "").split(","):
                t = t.strip()
                if t and t != "auto-captured":
                    chips += f"<span class='tag' style='margin-right:4px;'>{t}</span>"
            chip_row = f"<div style='margin-top:.3rem;'>{chips}</div>" if chips else ""
            st.markdown(f"<div class='card' style='margin:.3rem 0;'>"
                        f"<b>{d.title}</b>{auto}<br><span class='hint'>{meta}</span><br>"
                        f"<span style='font-size:.9rem;'>{d.rationale}</span>{chip_row}</div>",
                        unsafe_allow_html=True)
        if not results:
            st.markdown("<p class='hint'>No decisions match — try a broader search or "
                        "clear the filters.</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###### Export handover report")
    geo = {
        "static_camber_deg": s.camber, "static_toe_deg": s.toe,
        "caster_deg": s.caster, "kpi_deg": s.kpi,
        "scrub_radius_mm": s.scrub_radius,
        "roll_centre_front_mm": mid["rc_front"], "roll_centre_rear_mm": mid["rc_rear"],
        "max_lateral_g": veh.max_lateral_g(),
    }
    md = project_mod.build_handover_markdown(store, geometry=geo)
    ec = st.columns(3)
    ec[0].download_button("⬇ Handover (.md)", md, file_name="elbee_handover.md",
                          mime="text/markdown", width='stretch')
    ec[1].download_button("⬇ Project data (.json)", store.as_json(),
                          file_name="project.json", mime="application/json",
                          width='stretch')
    try:
        pdf_path = os.path.join(tempfile.gettempdir(), "elbee_handover.pdf")
        project_mod.render_pdf(md, pdf_path)
        with open(pdf_path, "rb") as f:
            ec[2].download_button("⬇ Handover (.pdf)", f.read(),
                                  file_name="elbee_handover.pdf",
                                  mime="application/pdf", width='stretch')
    except Exception as e:
        ec[2].markdown(f"<p class='hint'>PDF unavailable: {e}</p>", unsafe_allow_html=True)

# ----------------------------- TAB 8 --------------------------------------- #
with tab8:
    nstore = project_mod.ProjectStore(PROJECT_PATH)

    # The lead is looking at the notes — clear the unread badge for this session
    # and treat everything currently on disk as seen.
    st.session_state["_notes_unread"] = 0
    st.session_state["_notes_seen_ids"] = {n.id for n in nstore.notes}

    st.markdown('<p class="hint">Cross-team notes between leads — for keeping '
                'interfaces from going stale. Unlike Discord, a note here is addressed '
                'to a team, has an open/resolved status, and lives next to the work in '
                '<code>project.json</code>. <b>The way you out-integrate a richer team '
                'is by never letting two finished parts surprise each other.</b></p>',
                unsafe_allow_html=True)

    team_keys = list(integ_mod.TEAMS.keys())
    # Open-item summary across all teams
    open_counts = {t: nstore.open_note_count(t) for t in team_keys}
    open_counts = {t: c for t, c in open_counts.items() if c > 0}
    if open_counts:
        chips = " ".join(
            f"<span class='tag warn'>{integ_mod.TEAMS[t]['label']}: {c} open</span>"
            for t, c in open_counts.items())
        st.markdown(f"<div style='margin:.3rem 0 .6rem;'>{chips}</div>",
                    unsafe_allow_html=True)

    st.markdown("###### Post a note")
    pc = st.columns([1.2, 1.2, 1.2])
    n_from = pc[0].selectbox("From", team_keys,
                             format_func=lambda k: integ_mod.TEAMS[k]["label"], key="n_from")
    n_to = pc[1].selectbox("To", ["all"] + team_keys,
                           format_func=lambda k: "All teams" if k == "all"
                           else integ_mod.TEAMS[k]["label"], key="n_to")
    n_author = pc[2].text_input("Your name", key="n_author")
    n_msg = st.text_area("Note", key="n_msg", height=80,
                         placeholder="e.g. Upright moved 8 mm inboard — recheck caliper clearance")
    fc = st.columns([1, 1, 3])
    n_req = fc[0].checkbox("Requests action", key="n_req")
    n_urg = fc[1].checkbox("Urgent", key="n_urg")
    if st.button("Post note", key="n_post"):
        if n_msg.strip():
            _new_note = project_mod.Note(
                from_team=n_from, to_team=n_to, message=n_msg.strip(),
                author=n_author, is_request=n_req, urgent=n_urg)
            nstore.add_note(_new_note)
            _ok = nstore.save()
            # Remember this id as ours so the poller doesn't toast us our own
            # note, and mark it seen. Everyone ELSE's session will pick it up
            # off disk on their next poll (within NOTE_POLL_SECONDS) and toast.
            st.session_state.setdefault("_my_posted_note_ids", set()).add(_new_note.id)
            st.session_state.setdefault("_notes_seen_ids", set()).add(_new_note.id)
            _recipients = ("all teams" if n_to == "all"
                           else integ_mod.TEAMS.get(n_to, {}).get("label", n_to))
            if _ok:
                st.toast(f"Note posted — {_recipients} will be notified.", icon="✅")
            else:
                st.warning(
                    "Note saved in this session only — it could not be written to "
                    "shared storage, so other leads won't be notified. "
                    f"({getattr(nstore, 'save_error', 'unknown error')}) "
                    "Check the Supabase config so notes sync across users.")
            st.rerun()
        else:
            st.warning("Write a note before posting.")

    st.markdown("---")
    fcol1, fcol2 = st.columns([1.5, 3])
    view_team = fcol1.selectbox("Show notes for", ["all teams"] + team_keys,
                                format_func=lambda k: "All notes" if k == "all teams"
                                else integ_mod.TEAMS[k]["label"], key="n_view")
    show_resolved = fcol2.checkbox("Show resolved", value=False, key="n_showres")

    if view_team == "all teams":
        notes = sorted(nstore.notes, key=lambda n: n.ts, reverse=True)
    else:
        notes = nstore.notes_for(view_team)
    if not show_resolved:
        notes = [n for n in notes if n.status == "open"]

    if not notes:
        st.markdown('<p class="hint">No notes yet. When a check in TEAM FIT or '
                    'a suspension change affects another team, post a note here so '
                    'their lead sees it the next time they open the tool.</p>',
                    unsafe_allow_html=True)
    else:
        for n in notes:
            fclr = integ_mod.TEAMS.get(n.from_team, {}).get("color", "#888")
            tclr = integ_mod.TEAMS.get(n.to_team, {}).get("color", "#888") \
                if n.to_team != "all" else "#8d99a6"
            to_label = "All teams" if n.to_team == "all" \
                else integ_mod.TEAMS.get(n.to_team, {}).get("label", n.to_team)
            from_label = integ_mod.TEAMS.get(n.from_team, {}).get("label", n.from_team)
            badges = ""
            if n.urgent:
                badges += "<span class='tag bad'>urgent</span> "
            if n.is_request:
                badges += "<span class='tag warn'>action requested</span> "
            if n.status == "resolved":
                badges += "<span class='tag good'>resolved</span> "
            meta = f"{from_label} → {to_label} · {n.ts.replace('T',' ')[:16]}"
            if n.author:
                meta += f" · {n.author}"
            st.markdown(
                f"<div class='card' style='margin:.3rem 0; border-left:3px solid {fclr};'>"
                f"<div style='margin-bottom:.2rem;'>{badges}</div>"
                f"<span style='font-size:.95rem;'>{n.message}</span><br>"
                f"<span class='hint'>{meta}</span></div>", unsafe_allow_html=True)
            bc = st.columns([1, 6])
            if n.status == "open":
                if bc[0].button("Mark resolved", key=f"res_{n.id}"):
                    nstore.resolve_note(n.id)
                    nstore.save()
                    st.rerun()
            else:
                if bc[0].button("Reopen", key=f"reo_{n.id}"):
                    nstore.reopen_note(n.id)
                    nstore.save()
                    st.rerun()

# ----------------------------- TAB 9 --------------------------------------- #
# TIRE & GRIP — the competitive core. You get one set of tires; the edge is
# extracting every bit of truth from your tire data and running the whole grip/
# balance stack on it instead of a guess.
with tab9:
    st.markdown('<p class="hint">You can only afford <b>one set of tires</b>. The way '
                'you beat a team that can test rubber all year is to make every '
                'geometry and setup call against <b>your actual tire</b> before you '
                'commit it. This tab is where your tire lives — load a TTC-fitted '
                'model and the GRIP BALANCE and SETUP OPTIMISER tabs run on measured '
                'data, not a placeholder.</p>', unsafe_allow_html=True)

    _is_default = st.session_state.get("tire_is_default", True)
    badge_cls = "warn" if _is_default else "good"
    st.markdown(
        f"<div style='margin:.2rem 0 .8rem;'><span class='tag {badge_cls}'>"
        f"Active tire: {st.session_state.tire_source}</span></div>",
        unsafe_allow_html=True)

    live_tire = tire_mod.PacejkaLateral(coeffs=dict(st.session_state.tire_coeffs),
                                        FNOMIN=st.session_state.tire_fnomin)
    desc = tire_mod.describe(live_tire)
    m = st.columns(5)
    m[0].markdown(metric("μ @ nominal", f"{desc['mu_at_nominal']:.2f}", ""), unsafe_allow_html=True)
    m[1].markdown(metric("μ light load", f"{desc['mu_light_load']:.2f}", ""), unsafe_allow_html=True)
    m[2].markdown(metric("μ heavy load", f"{desc['mu_heavy_load']:.2f}", ""), unsafe_allow_html=True)
    m[3].markdown(metric("Peak slip", f"{desc['alpha_peak_deg']:.1f}", "°"), unsafe_allow_html=True)
    m[4].markdown(metric("Best camber", f"{desc['optimal_camber_deg']:.1f}", "°"), unsafe_allow_html=True)

    # ---- grip curves ----------------------------------------------------- #
    cc1, cc2 = st.columns(2)
    Fz = np.linspace(150, 2200, 60)
    mu = [live_tire.mu_peak(f) for f in Fz]
    figG = go.Figure()
    figG.add_trace(go.Scatter(x=Fz, y=mu, mode="lines", line=dict(color=CYAN, width=3)))
    figG.update_layout(**PLOT_LAYOUT, title="Load sensitivity — peak μ vs vertical load",
                       xaxis_title="vertical load (N)", yaxis_title="peak μ", height=320)
    cc1.plotly_chart(figG, width='stretch')

    cam = np.linspace(0, 5, 40)
    mu_c = [live_tire.mu_peak(live_tire.FNOMIN, np.radians(c)) for c in cam]
    figC = go.Figure()
    figC.add_trace(go.Scatter(x=cam, y=mu_c, mode="lines", line=dict(color=AMBER, width=3)))
    figC.update_layout(**PLOT_LAYOUT, title="Camber sensitivity — peak μ vs inclination",
                       xaxis_title="inclination (°)", yaxis_title="peak μ @ nominal load",
                       height=320)
    cc2.plotly_chart(figC, width='stretch')
    st.markdown('<p class="hint">Left: how fast grip falls as the tire is loaded — '
                'this is what makes load transfer cost you grip, and why a lower CG and '
                'softer springs help. Right: the camber the tire wants. The peak of '
                'this curve is free grip you set with geometry, not money — target it '
                'with your static camber and camber-gain.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###### Load YOUR fitted tire (from TTC data)")
    st.markdown('<p class="hint">Run <code>python process_ttc.py your_cornering.mat '
                'my_tire.json</code> to fit a Magic Formula to your TTC data, then '
                'upload <code>my_tire.json</code> here. It loads into the live engine '
                'immediately. <b>The .json is TTC-derived — keep it out of git.</b></p>',
                unsafe_allow_html=True)
    up = st.file_uploader("Fitted tire JSON", type=["json"], key="tire_json")
    lc1, lc2 = st.columns([1, 1])
    if up is not None:
        try:
            import json as _json
            d = _json.load(up)
            new_coeffs = d["coeffs"]
            new_fnom = float(d.get("FNOMIN", 1100.0))
            # validate it builds
            _t = tire_mod.PacejkaLateral(coeffs=new_coeffs, FNOMIN=new_fnom)
            _t.mu_peak(new_fnom)
            if lc1.button("✓ Use this tire", width='stretch'):
                st.session_state.tire_coeffs = dict(new_coeffs)
                st.session_state.tire_fnomin = new_fnom
                st.session_state.tire_source = f"TTC-fitted ({up.name})"
                st.session_state.tire_is_default = False
                log_decision_now("suspension", f"Loaded fitted tire {up.name}",
                                 "Grip/balance now run on measured TTC tire data.")
                st.rerun()
        except Exception as e:
            st.markdown(f"<p class='hint'>Couldn't read that tire file: {e}</p>",
                        unsafe_allow_html=True)
    if not _is_default:
        if lc2.button("↺ Revert to generic default", width='stretch'):
            dt = tire_mod.default_tire()
            st.session_state.tire_coeffs = dict(dt.coeffs)
            st.session_state.tire_fnomin = dt.FNOMIN
            st.session_state.tire_source = "Generic FSAE default (not your tire)"
            st.session_state.tire_is_default = True
            st.rerun()

    st.markdown('<p class="hint" style="border-left:2px solid #5a4317;padding-left:10px;">'
                'The generic default is hand-built to behave sensibly (load sensitivity, '
                'a camber optimum) but it is <b>not your tire</b> — use it for relative '
                'comparisons until you fit yours. Absolute grip numbers only become '
                'trustworthy once the tire above says "TTC-fitted".</p>',
                unsafe_allow_html=True)

    # ---- Combined slip (friction ellipse) -------------------------------- #
    st.markdown("###### Combined slip — the friction ellipse")
    st.markdown('<p class="hint">How much lateral grip is left while you brake or put '
                'power down. Built on the lateral tire above with friction-ellipse '
                'coupling. <b>Uncalibrated</b> until you fit it to drive/brake TTC data '
                '— the coupling shape is real physics; the exact exponents need your '
                'Fx data to be quantitative.</p>', unsafe_allow_html=True)
    try:
        _live_tire_cs = tire_mod.PacejkaLateral(
            coeffs=dict(st.session_state.tire_coeffs),
            FNOMIN=st.session_state.tire_fnomin)
        _ct = tire_mod.CombinedSlipTire(lateral=_live_tire_cs)
        _Fz_demo = float(st.session_state.tire_fnomin)
        fx_e, fy_e = _ct.friction_circle(_Fz_demo)
        figFE = go.Figure()
        figFE.add_trace(go.Scatter(x=fx_e, y=fy_e, mode="lines",
                                   line=dict(color=CYAN, width=2.5),
                                   name="grip limit"))
        figFE.update_layout(**PLOT_LAYOUT,
                            title=f"Combined grip envelope at Fz={_Fz_demo:.0f} N",
                            xaxis_title="longitudinal force Fx (N)",
                            yaxis_title="lateral force Fy (N)", height=340)
        figFE.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(figFE, width='stretch')
        st.markdown(f'<span class="tag warn">{_ct.status()}</span>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Combined-slip preview unavailable: {e}")

    # ---- Tire thermal channel (lumped tread/carcass/gas network) --------- #
    st.markdown("###### Tire thermal channel — warm-up, working range & pressure")
    st.markdown('<p class="hint">A true tire temperature cannot be computed without '
                '<b>empirical, temperature-swept tire data</b> — so this channel is '
                'built honestly: a 3-node lumped energy balance (tread / carcass / '
                'inflation gas) heated by frictional sliding and rolling hysteresis, '
                'cooled by convection to air and conduction to the track. The '
                '<b>equations are textbook physics</b>; the masses, heat-transfer '
                'coefficients and the grip-vs-temperature law are '
                '<b>representative defaults, NOT your tire</b>. Read the shape — '
                'warm-up time, the front/rear and across-width split, the pressure '
                'rise — not the absolute degrees. Every temperature here is flagged '
                'synthesized until you calibrate it to swept data.</p>',
                unsafe_allow_html=True)
    try:
        tcol = st.columns(4)
        _t_alpha = tcol[0].number_input("Slip angle (°)", 0.0, 12.0, value=4.0,
                                        step=0.5, key="therm_alpha")
        _t_fz = tcol[1].number_input("Vertical load (N)", 200.0, 3000.0,
                                     value=1300.0, step=50.0, key="therm_fz")
        _t_v = tcol[2].number_input("Speed (m/s)", 3.0, 45.0, value=20.0,
                                    step=1.0, key="therm_v")
        _t_dur = tcol[3].number_input("Run length (s)", 20.0, 600.0, value=150.0,
                                      step=10.0, key="therm_dur")
        tcol2 = st.columns(4)
        _t_cam = tcol2[0].number_input("Camber (°)", 0.0, 6.0, value=1.5,
                                       step=0.5, key="therm_cam")
        _t_amb = tcol2[1].number_input("Ambient (°C)", -5.0, 50.0, value=25.0,
                                       step=1.0, key="therm_amb")
        _t_trk = tcol2[2].number_input("Track surface (°C)", 0.0, 70.0, value=34.0,
                                       step=1.0, key="therm_trk")
        _t_mu = tcol2[3].checkbox("Couple grip to temp (μ(T))", value=False,
                                  key="therm_mu",
                                  help="Let the modelled tread temperature scale "
                                       "Pacejka grip. OFF by default — the μ(T) curve "
                                       "is the most data-hungry part and is flagged "
                                       "synthesized when on.")

        _t_cold_psi = tcol[0].number_input("Cold set pressure (psi)", 6.0, 35.0,
                                           value=12.0, step=0.5, key="therm_psi")

        _trun = _cached_thermal_warmup(
            coeffs=tuple(sorted(dict(st.session_state.tire_coeffs).items())),
            fnomin=st.session_state.tire_fnomin,
            enable_mu=bool(_t_mu),
            cold_pa=float(_t_cold_psi) * 6894.757,
            alpha_deg=float(_t_alpha), Fz=float(_t_fz), v_x=float(_t_v),
            gamma_deg=float(_t_cam), ambient_c=float(_t_amb), track_c=float(_t_trk),
            duration_s=float(_t_dur), dt=5.0e-3)

        _mean = _trun.tread_mean_c()
        tm_cols = st.columns(4)
        tm_cols[0].markdown(metric("Tread (plateau)", f"{_mean[-1]:.0f}", "°C"),
                            unsafe_allow_html=True)
        tm_cols[1].markdown(metric("Carcass", f"{_trun.carcass_c[-1]:.0f}", "°C"),
                            unsafe_allow_html=True)
        tm_cols[2].markdown(metric("Hot pressure", f"{thermal_mod.psi(_trun.pressure_pa[-1]):.1f}", "psi"),
                            unsafe_allow_html=True)
        # time to reach 90% of the plateau rise — a "warm-up time" proxy
        _rise = _mean - _mean[0]
        _target = 0.9 * _rise[-1] if abs(_rise[-1]) > 1e-6 else 0.0
        _idx = int(np.argmax(_rise >= _target)) if _target > 0 else 0
        _warm_s = _trun.t[_idx] if _target > 0 else 0.0
        tm_cols[3].markdown(metric("Warm-up (90%)", f"{_warm_s:.0f}", "s"),
                            unsafe_allow_html=True)

        # temperature traces
        figT = go.Figure()
        figT.add_trace(go.Scatter(x=_trun.t, y=_mean, mode="lines",
                                  line=dict(color=CYAN, width=3), name="tread (mean)"))
        figT.add_trace(go.Scatter(x=_trun.t, y=_trun.carcass_c, mode="lines",
                                  line=dict(color=AMBER, width=2), name="carcass"))
        figT.add_trace(go.Scatter(x=_trun.t, y=_trun.gas_c, mode="lines",
                                  line=dict(color=DIM, width=2, dash="dot"), name="gas"))
        # across-width band spread (inner/mid/outer) at the plateau
        if _trun.tread_c.shape[1] > 1:
            figT.add_trace(go.Scatter(x=_trun.t, y=_trun.tread_c[:, 0], mode="lines",
                                      line=dict(color=CYAN, width=1, dash="dot"),
                                      name="tread inner", opacity=0.5))
            figT.add_trace(go.Scatter(x=_trun.t, y=_trun.tread_c[:, -1], mode="lines",
                                      line=dict(color=RED, width=1, dash="dot"),
                                      name="tread outer", opacity=0.5))
        figT.update_layout(**PLOT_LAYOUT, title="Tire warm-up — lumped thermal network",
                           xaxis_title="time (s)", yaxis_title="temperature (°C)",
                           height=360)
        st.plotly_chart(figT, width='stretch')

        if bool(_t_mu):
            figMu = go.Figure()
            figMu.add_trace(go.Scatter(x=_trun.t, y=_trun.mu_scale, mode="lines",
                                       line=dict(color=RED, width=2.5)))
            figMu.update_layout(**PLOT_LAYOUT,
                                title="Grip multiplier μ(T) over the run "
                                      "(SYNTHESIZED — needs swept data)",
                                xaxis_title="time (s)",
                                yaxis_title="grip scale vs optimum", height=260)
            st.plotly_chart(figMu, width='stretch')

        st.markdown(f'<span class="tag warn">{_trun.status}</span>',
                    unsafe_allow_html=True)
        st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                    'padding-left:10px;">The across-width split (inner vs outer band) '
                    'is the same thing a tire pyrometer reads after a run — use it to '
                    'reason about camber, and the front/rear plateau split to reason '
                    'about balance late in a stint. The numbers are a physically-'
                    'shaped guess until the active tire above is calibrated to '
                    'temperature-swept data; then set <code>ThermalParams.calibrated'
                    '</code> and they stop being flagged.</p>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Thermal channel unavailable: {e}")

    # ---- Damper force-velocity ------------------------------------------- #
    st.markdown("###### Damper force–velocity (transient building block)")
    st.markdown('<p class="hint">Real bilinear-digressive damper law. <b>Uncalibrated</b> '
                'representative magnitudes until you load your dyno curve; the force law '
                'and the damping-ratio diagnostic are real. This is the primitive the '
                'transient (turn-in / pitch) model on the roadmap is built on.</p>',
                unsafe_allow_html=True)
    dmp_cols = st.columns(4)
    _cbl = dmp_cols[0].number_input("Bump low (N·s/m)", 0.0, 30000.0, value=6000.0, step=250.0)
    _crl = dmp_cols[1].number_input("Rebound low (N·s/m)", 0.0, 30000.0, value=9000.0, step=250.0)
    _cbh = dmp_cols[2].number_input("Bump high (N·s/m)", 0.0, 15000.0, value=2000.0, step=100.0)
    _crh = dmp_cols[3].number_input("Rebound high (N·s/m)", 0.0, 15000.0, value=3000.0, step=100.0)
    _dc = damper_mod.DamperCurve(c_bump_low=_cbl, c_reb_low=_crl,
                                 c_bump_high=_cbh, c_reb_high=_crh)
    _vv, _ff = _dc.curve_points(v_max=0.4)
    figD = go.Figure()
    figD.add_trace(go.Scatter(x=_vv, y=_ff, mode="lines",
                              line=dict(color=AMBER, width=2.5), name="damper"))
    figD.update_layout(**PLOT_LAYOUT, title="Damper force vs shaft velocity",
                       xaxis_title="shaft velocity (m/s)  +bump / −rebound",
                       yaxis_title="force (N)", height=320)
    st.plotly_chart(figD, width='stretch')
    try:
        _mr_demo = kin.motion_ratio() if kin.motion_ratio_is_real() else 1.0
        _corner_m = float(st.session_state.vp.get("mass", 300)) * 0.25
        _wr_demo = kin.wheel_rate(float(st.session_state.vp.get("spring_rate_front", 35.0))) \
            if kin.motion_ratio_is_real() else 30.0
        _zb = damper_mod.damping_ratio(_dc, _corner_m, _wr_demo, _mr_demo, "bump")
        _zr = damper_mod.damping_ratio(_dc, _corner_m, _wr_demo, _mr_demo, "rebound")
        zc = st.columns(2)
        zc[0].markdown(metric("Damping ratio ζ (bump)", f"{_zb:.2f}", "",
                              "good" if 0.5 <= _zb <= 0.8 else "warn"),
                       unsafe_allow_html=True)
        zc[1].markdown(metric("Damping ratio ζ (rebound)", f"{_zr:.2f}", "",
                              "good" if 0.6 <= _zr <= 1.1 else "warn"),
                       unsafe_allow_html=True)
        st.markdown(f'<span class="tag warn">{_dc.status()}</span>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Damping-ratio diagnostic unavailable: {e}")

# ----------------------------- TAB 10 -------------------------------------- #
# SETUP OPTIMISER — spend the one tire set wisely. Rank the levers by grip
# impact and search for the best setup, all on the live tire.
with tab10:
    st.markdown('<p class="hint">Which change actually buys grip? With one set of '
                'tires you cannot afford to chase the wrong lever. This ranks every '
                'setup knob by how much limit grip and balance it moves — on your live '
                'tire — then searches for the best combination at a target balance. '
                '<b>Out-integrate, don\'t out-spend: know the answer before you build '
                'it.</b></p>', unsafe_allow_html=True)

    base_vp = VehicleParams(**{k: v for k, v in st.session_state.vp.items()
                               if k in VehicleParams.__dataclass_fields__})

    sc1, sc2 = st.columns([1, 1])
    target_bal = sc1.slider("Target balance (+ understeer / − oversteer)",
                            -0.10, 0.15, 0.04, 0.01)
    bal_tol = sc2.slider("Balance tolerance", 0.02, 0.15, 0.06, 0.01)

    if st.button("▶ Rank levers & optimise", width='stretch'):
        st.session_state._run_opt = True

    if st.session_state.get("_run_opt"):
        with st.spinner("Sweeping setup space on the live tire…"):
            sens = setup_mod.sensitivity(base_vp, front_kin=kin, rear_kin=kin,
                                         tire=live_tire)
            opt = setup_mod.optimise(base_vp, front_kin=kin, rear_kin=kin,
                                     tire=live_tire, target_balance=target_bal,
                                     balance_tol=bal_tol)

        b = sens["base"]
        st.markdown("###### Current setup")
        bc = st.columns(3)
        bc[0].markdown(metric("Max grip", f"{b['max_g']:.3f}", "g"), unsafe_allow_html=True)
        _bv = ("NEUTRAL", "good") if abs(b["balance"]) < 0.03 else \
              (("UNDERSTEER", "warn") if b["balance"] > 0 else ("OVERSTEER", "bad"))
        bc[1].markdown(metric("Balance", _bv[0], "", _bv[1]), unsafe_allow_html=True)
        bc[2].markdown(metric("Balance index", f"{b['balance']:+.3f}", ""), unsafe_allow_html=True)

        st.markdown("###### Levers ranked by grip impact")
        st.markdown('<p class="hint">Read this as: change this knob by one step, get '
                    'this much grip and this much balance shift. Spend your build/tune '
                    'time top-down.</p>', unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td style='padding:4px 10px;'>{r['label']}</td>"
            f"<td style='padding:4px 10px;text-align:right;color:{'#62d27a' if r['d_maxg_per_step']>=0 else '#ff6b6b'};'>"
            f"{r['d_maxg_per_step']:+.4f} g</td>"
            f"<td style='padding:4px 10px;text-align:right;color:var(--dim);'>per {r['step']:g} {r['unit']}</td>"
            f"<td style='padding:4px 10px;text-align:right;'>{r['d_balance_per_step']:+.3f} bal</td></tr>"
            for r in sens["rankings"])
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:.92rem;'>"
            f"<tr style='color:var(--dim);border-bottom:1px solid var(--line);'>"
            f"<td style='padding:4px 10px;'>lever</td>"
            f"<td style='padding:4px 10px;text-align:right;'>grip / step</td>"
            f"<td></td><td style='padding:4px 10px;text-align:right;'>balance / step</td></tr>"
            f"{rows}</table>", unsafe_allow_html=True)

        st.markdown("###### Optimiser recommendation")
        oc = st.columns(3)
        oc[0].markdown(metric("Optimised grip", f"{opt['best_eval']['max_g']:.3f}", "g",
                              "good"), unsafe_allow_html=True)
        oc[1].markdown(metric("Grip gained", f"{opt['delta_maxg']:+.3f}", "g",
                              "good" if opt["delta_maxg"] > 0 else ""), unsafe_allow_html=True)
        oc[2].markdown(metric("Balance", f"{opt['best_eval']['balance']:+.3f}", ""),
                       unsafe_allow_html=True)

        if opt["best_params"]:
            _knob_lbl = {k: v["label"] for k, v in setup_mod.PARAM_KNOBS.items()}
            _knob_unit = {k: v["unit"] for k, v in setup_mod.PARAM_KNOBS.items()}
            recs = "".join(
                f"<tr><td style='padding:4px 10px;'>{_knob_lbl.get(k,k)}</td>"
                f"<td style='padding:4px 10px;text-align:right;'>{v:.2f} {_knob_unit.get(k,'')}</td></tr>"
                for k, v in opt["best_params"].items())
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-size:.92rem;'>"
                f"<tr style='color:var(--dim);border-bottom:1px solid var(--line);'>"
                f"<td style='padding:4px 10px;'>change</td>"
                f"<td style='padding:4px 10px;text-align:right;'>to</td></tr>"
                f"{recs}</table>", unsafe_allow_html=True)

            ac1, ac2 = st.columns([1, 2])
            if ac1.button("Apply to sidebar", width='stretch'):
                for k, v in opt["best_params"].items():
                    if k in ("static_camber_front", "static_camber_rear"):
                        continue  # camber is set by geometry; recommend, don't force
                    if k in st.session_state.vp:
                        st.session_state.vp[k] = v
                _cam_note = ""
                if "static_camber_front" in opt["best_params"]:
                    _cam_note = (f" Target front camber "
                                 f"{opt['best_params']['static_camber_front']:.1f}° via geometry.")
                log_decision_now("suspension", "Applied optimiser setup",
                                 f"Grip {opt['start_eval']['max_g']:.3f}→"
                                 f"{opt['best_eval']['max_g']:.3f} g at balance "
                                 f"{opt['best_eval']['balance']:+.3f}.{_cam_note}")
                st.session_state._run_opt = False
                st.rerun()
            ac2.markdown('<p class="hint">Camber targets are recommendations — set them '
                         'with static camber + camber-gain in your geometry, then check '
                         'the KINEMATICS tab. Everything else applies to the sidebar '
                         'directly.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="hint">Your current setup is already at the '
                        'optimiser\'s best within these bounds. Nice.</p>',
                        unsafe_allow_html=True)

        if _is_default:
            st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                        'padding-left:10px;">These rankings run on the <b>generic '
                        'default tire</b>. They show the right <i>directions</i>, but '
                        'load your TTC-fitted tire in the TIRE &amp; GRIP tab before '
                        'trusting the magnitudes — your tire\'s load and camber '
                        'sensitivity is exactly what sets which lever wins.</p>',
                        unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  TAB 11 — LAP TIME : turn the grip envelope into seconds
# --------------------------------------------------------------------------- #
with tab11:
    st.markdown('<p class="hint">Grip is a means; <b>lap time is the score.</b> This '
                'tab runs your <i>live</i> geometry, setup and tire around the FSAE '
                'skidpad and a representative autocross, so every change you make '
                'upstream reads out in <b>seconds</b> — the only currency at '
                'competition. A team that can\'t test rubber all year wins by knowing '
                'the lap-time consequence of a setup call <i>before</i> it freezes the '
                'build. Quasi-steady-state on the grip envelope you already trust.</p>',
                unsafe_allow_html=True)

    # Live dynamics model — same objects the rest of the app already solved.
    try:
        _veh_lap = veh
    except Exception:
        _veh_lap = None

    # Make sure a tire-backed model exists even if the user never opened TIRE & GRIP.
    try:
        _live_tire_lap = live_tire
    except NameError:
        _live_tire_lap = tire_mod.PacejkaLateral(
            coeffs=dict(st.session_state.tire_coeffs),
            FNOMIN=st.session_state.tire_fnomin)
    if _veh_lap is None:
        _veh_lap = VehicleDynamics(
            VehicleParams(**{k: v for k, v in st.session_state.vp.items()
                             if k in VehicleParams.__dataclass_fields__}),
            front_kin=kin, rear_kin=kin, tire=_live_tire_lap)

    # ---- Powertrain / aero inputs (all defaulted; safe to ignore) -------- #
    with st.expander("Powertrain & aero (defaults are sensible FSAE-EV values)",
                     expanded=False):
        pc = st.columns(4)
        pw = pc[0].number_input("Peak power (kW)", 10.0, 200.0,
                                value=80.0, step=5.0)
        tract = pc[1].number_input("Traction cap (N)", 500.0, 6000.0,
                                   value=2600.0, step=100.0)
        cda = pc[2].number_input("Drag CdA (m²)", 0.0, 3.0, value=1.10, step=0.05)
        cla = pc[3].number_input("Downforce ClA (m²)", 0.0, 6.0, value=2.60, step=0.1)
        pc2 = st.columns(4)
        drive = pc2[0].selectbox("Drive", ["rwd", "awd"], index=0)
        brake_g = pc2[1].number_input("Brake cap (g)", 0.5, 3.0, value=1.8, step=0.1)
        crr = pc2[2].number_input("Rolling res. crr", 0.005, 0.05,
                                  value=0.018, step=0.002, format="%.3f")
        eff = pc2[3].number_input("Drivetrain eff.", 0.5, 1.0, value=0.90, step=0.01)

        st.markdown("**Motor map** — replace the flat power cap with a real "
                    "torque/speed curve. The flat cap is the cruder model; the map "
                    "is strictly better when you have the numbers.")
        use_map = st.checkbox("Use a motor torque/speed map", value=False,
                              help="Enter your motor's peak torque, peak power and "
                                   "redline (from the datasheet). Builds a "
                                   "representative torque-plateau + constant-power "
                                   "curve — clearly flagged as representative, not a "
                                   "measured dyno pull.")
        _motor_map = None
        if use_map:
            mpc = st.columns(3)
            mt = mpc[0].number_input("Peak torque (N·m)", 20.0, 600.0, value=230.0, step=10.0)
            mp = mpc[1].number_input("Peak power (kW)", 10.0, 200.0, value=80.0, step=5.0)
            mr_in = mpc[2].number_input("Redline (rpm)", 3000.0, 20000.0, value=6000.0, step=500.0)
            mpc2 = st.columns(2)
            fd = mpc2[0].number_input("Final drive ratio", 1.0, 10.0, value=3.5, step=0.1)
            wr_ = mpc2[1].number_input("Loaded wheel radius (m)", 0.15, 0.30,
                                       value=0.20, step=0.005, format="%.3f")
            _motor_map = lap_mod.MotorMap.from_peak(mt, mp, mr_in, final_drive=fd,
                                                    wheel_radius_m=wr_)
            st.caption(f"Motor map source: {_motor_map.source} (from datasheet peaks; "
                       "for a measured curve construct MotorMap(rpm, torque_nm) in code).")

    _pt = lap_mod.Powertrain(power_kw=pw, max_tractive_n=tract, drivetrain_eff=eff,
                             cda=cda, cla=cla, crr=crr, drive=drive,
                             brake_g_cap=brake_g, motor_map=_motor_map)

    # ---- Track source: yardstick autocross, or YOUR GPS/cone layout ------- #
    st.markdown("###### Track")
    track_src = st.radio("Run on", ["Representative autocross",
                                    "Import GPS / cone CSV"], horizontal=True)
    ax_scale = 1.0
    _imported_xy = None
    if track_src == "Representative autocross":
        ax_scale = st.slider("Autocross lap scale (stretches the yardstick lap)",
                             0.6, 1.6, 1.0, 0.1)
    else:
        st.markdown('<p class="hint">Upload your actual layout — no more manual '
                    'segment entry. Centreline <code>x,y</code> (metres) or GPS '
                    '<code>lat,lon</code>; or cone rows '
                    '<code>left_x,left_y,right_x,right_y</code>. The lap then runs your '
                    'real course.</p>', unsafe_allow_html=True)
        tcol = st.columns(3)
        fmt = tcol[0].selectbox("CSV format", ["centreline x,y (m)",
                                               "GPS lat,lon", "cones L/R x,y"])
        width_m = tcol[1].number_input("Track width (m)", 2.0, 6.0, value=3.5, step=0.5)
        do_line = tcol[2].checkbox("Optimise racing line", value=True,
                                   help="Use the track width to straighten corners — "
                                        "reports the time gained vs the centreline.")
        tup = st.file_uploader("Track CSV", type=["csv"], key="track_csv")
        if tup is not None:
            try:
                import io as _io2
                raw = tup.getvalue().decode("utf-8", errors="replace")
                arr = np.genfromtxt(_io2.StringIO(raw), delimiter=",")
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.size and np.isnan(arr[0]).any():        # header row
                    arr = arr[1:]
                if fmt == "cones L/R x,y" and arr.shape[1] >= 4:
                    cx, cy = lap_mod.cones_to_centerline(arr[:, 0], arr[:, 1],
                                                         arr[:, 2], arr[:, 3])
                elif fmt == "GPS lat,lon" and arr.shape[1] >= 2:
                    cx, cy = lap_mod.latlon_to_xy(arr[:, 0], arr[:, 1])
                else:
                    cx, cy = arr[:, 0], arr[:, 1]
                _imported_xy = (np.asarray(cx, float), np.asarray(cy, float),
                                width_m, do_line)
                st.success(f"Loaded {len(cx)} points "
                           f"({np.hypot(np.diff(cx), np.diff(cy)).sum():.0f} m path).")
            except Exception as e:
                st.error(f"Couldn't parse that track CSV: {e}")

    if st.button("▶ Run lap-time sim", width='stretch'):
        st.session_state._run_lap = True

    if st.session_state.get("_run_lap"):
        with st.spinner("Driving your car around on the live tire…"):
            skid = lap_mod.skidpad_time(_veh_lap, _pt)
            if _imported_xy is not None:
                ix, iy, iw, iline = _imported_xy
                if iline:
                    _cmp = lap_mod.compare_line_vs_centerline(_veh_lap, ix, iy,
                                                              track_width_m=iw, pt=_pt)
                    track = _cmp["line_track"]
                    lap = _cmp["line_result"]
                    st.session_state._line_cmp = dict(
                        gained=_cmp["time_gained_s"],
                        center_t=_cmp["centerline_result"].lap_time_s,
                        line_t=_cmp["line_result"].lap_time_s,
                        lx=_cmp["line_x"], ly=_cmp["line_y"], cx=ix, cy=iy)
                else:
                    track = lap_mod.track_from_path(ix, iy, name="Imported", ds=1.0)
                    lap = lap_mod.simulate_lap(_veh_lap, track, _pt)
                    st.session_state.pop("_line_cmp", None)
            else:
                track = lap_mod.default_autocross(scale=ax_scale)
                lap = lap_mod.simulate_lap(_veh_lap, track, _pt)
                st.session_state.pop("_line_cmp", None)

        # Surface any safe-default warnings rather than hiding a bad data point.
        for r in (skid, lap):
            if r.warning:
                st.warning(f"⚠ {r.warning}")

        # ---- Skidpad ---- #
        st.markdown("###### FSAE skidpad (one timed circle)")
        skc = st.columns(3)
        _skt = f"{skid.lap_time_s:.3f}" if skid.ok and np.isfinite(skid.lap_time_s) else "—"
        skc[0].markdown(metric("Skidpad time", _skt, "s",
                               "good" if skid.ok else "bad"), unsafe_allow_html=True)
        skc[1].markdown(metric("Corner speed", f"{skid.avg_speed_ms:.1f}", "m/s"),
                        unsafe_allow_html=True)
        _sk_lat = (skid.avg_speed_ms ** 2) / (lap_mod.SKIDPAD_RADIUS_M * 9.81) \
            if skid.ok else 0.0
        skc[2].markdown(metric("Lateral", f"{_sk_lat:.2f}", "g"), unsafe_allow_html=True)

        # ---- Autocross / imported lap ---- #
        _lap_title = track.name if getattr(track, "name", "") else "Representative autocross"
        st.markdown(f"###### {_lap_title}")
        axc = st.columns(4)
        _axt = f"{lap.lap_time_s:.2f}" if lap.ok and np.isfinite(lap.lap_time_s) else "—"
        axc[0].markdown(metric("Lap time", _axt, "s",
                               "good" if lap.ok else "bad"), unsafe_allow_html=True)
        axc[1].markdown(metric("Avg speed", f"{lap.avg_speed_ms:.1f}", "m/s"),
                        unsafe_allow_html=True)
        axc[2].markdown(metric("Top speed", f"{lap.top_speed_ms:.1f}", "m/s"),
                        unsafe_allow_html=True)
        axc[3].markdown(metric("Min speed", f"{lap.min_speed_ms:.1f}", "m/s"),
                        unsafe_allow_html=True)

        # ---- Racing line vs centreline (only when an imported track was optimised) ---- #
        _lc = st.session_state.get("_line_cmp")
        if _lc and np.isfinite(_lc.get("gained", float("nan"))):
            st.markdown(metric("Racing line vs centreline",
                               f"{_lc['gained']:+.2f}", "s gained",
                               "good" if _lc["gained"] >= 0 else "warn"),
                        unsafe_allow_html=True)
            figRL = go.Figure()
            figRL.add_trace(go.Scatter(x=_lc["cx"], y=_lc["cy"], mode="lines",
                                       line=dict(color="#8d99a6", width=1.5, dash="dot"),
                                       name="centreline"))
            figRL.add_trace(go.Scatter(x=_lc["lx"], y=_lc["ly"], mode="lines",
                                       line=dict(color=CYAN, width=2.5),
                                       name="racing line"))
            figRL.update_layout(**PLOT_LAYOUT, title="Racing line (uses track width)",
                                xaxis_title="x (m)", yaxis_title="y (m)", height=360)
            figRL.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(figRL, width='stretch')
            st.markdown('<p class="hint">Curvature-optimal line within the track '
                        'width — straightens corners to raise minimum radius, hence '
                        'speed. It is a curvature-optimal (not fully-coupled '
                        'minimum-time) line; honest about the difference.</p>',
                        unsafe_allow_html=True)
        if _pt.uses_real_motor_map():
            st.markdown('<span class="tag good">motor map active (representative '
                        'curve)</span>', unsafe_allow_html=True)

        # Speed-vs-distance trace
        if lap.ok and lap.s and lap.v:
            figL = go.Figure()
            figL.add_trace(go.Scatter(x=lap.s, y=lap.v, mode="lines",
                                      line=dict(color=CYAN, width=2.5),
                                      name="speed"))
            figL.update_layout(**PLOT_LAYOUT, title="Speed around the lap",
                               xaxis_title="distance (m)", yaxis_title="speed (m/s)",
                               height=320)
            st.plotly_chart(figL, width='stretch')

        # Store last skidpad time so a delta can be shown after the next change.
        if skid.ok and np.isfinite(skid.lap_time_s):
            prev = st.session_state.get("_last_skidpad")
            if prev is not None and abs(prev - skid.lap_time_s) > 1e-4:
                d = skid.lap_time_s - prev
                _cls = "good" if d < 0 else "bad"
                st.markdown(
                    f"<span class='tag {_cls}'>Δ skidpad vs last run: "
                    f"{d:+.3f} s</span>", unsafe_allow_html=True)
            st.session_state._last_skidpad = skid.lap_time_s

        # Log it to the handover record so the reasoning survives.
        if lap.ok and np.isfinite(lap.lap_time_s):
            lc1, lc2 = st.columns([1, 2])
            if lc1.button("Log these times", width='stretch'):
                log_decision_now(
                    "suspension", "Lap-time prediction",
                    f"Skidpad {_skt}s, autocross {_axt}s on "
                    f"{'TTC tire' if not st.session_state.get('tire_is_default', True) else 'generic tire'} "
                    f"(power {pw:.0f}kW, ClA {cla:.2f}).")
                st.success("Logged to handover record.")
            lc2.markdown('<p class="hint">Tip: change a hardpoint or a setup lever, '
                         're-run, and watch the skidpad delta. That delta — in seconds '
                         '— is the number to defend a design decision with.</p>',
                         unsafe_allow_html=True)

        if st.session_state.get("tire_is_default", True):
            st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                        'padding-left:10px;">Running on the <b>generic default tire</b>. '
                        'Times are the right shape and rank setups correctly, but load '
                        'your TTC-fitted tire in TIRE &amp; GRIP before trusting the '
                        'absolute seconds.</p>', unsafe_allow_html=True)
        st.markdown('<p class="hint">Model: quasi-steady-state point mass on the live '
                    'grip envelope. Good for ranking and for skidpad (near closed-form); '
                    'on autocross it lands within a few percent — enough to choose '
                    'between setups, not to predict the absolute clock to the tenth.</p>',
                    unsafe_allow_html=True)


# ----------------------------- TAB 12 -------------------------------------- #
with tab12:
    st.markdown('<p class="hint">A sim only changes a decision if people <b>believe '
                'it</b> — and the honest way to earn that is to show it predicted '
                'something you measured. Load a real run and KinematiK reports the gap '
                'in plain numbers you can check by hand: how far off, and <i>which way</i> '
                'the model is biased. Nothing here tunes the model to fit — it tells you '
                'whether to trust the prediction, or which assumption to go hunt down. '
                'A matched sim wins the argument no stubborn opinion can.</p>',
                unsafe_allow_html=True)

    # Live model — same objects every other tab solved.
    try:
        _veh_val = veh
    except Exception:
        _veh_val = None
    if _veh_val is None:
        st.warning("Geometry/vehicle model isn't available — fix the linkage first.")
    else:
        def _verdict_tag(ok, data_error=False):
            if data_error:
                return '<span class="tag warn">could not correlate</span>'
            return ('<span class="tag good">✓ within tolerance — trust it</span>'
                    if ok else
                    '<span class="tag bad">✗ outside tolerance — find the wrong assumption</span>')

        def _channel_table(rep):
            rows = ""
            for c in rep.channels:
                cls = "good" if c.within_tol else ("warn" if c.verdict == "n/a" else "bad")
                epct = f"{c.error_pct:+.1f}%" if np.isfinite(c.error_pct) else "—"
                rows += (f"<tr><td style='padding:4px 10px'>{c.channel}</td>"
                         f"<td style='padding:4px 10px;text-align:right'>{c.measured:.4g}{corr_mod._u(c.unit)}</td>"
                         f"<td style='padding:4px 10px;text-align:right'>{c.predicted:.4g}{corr_mod._u(c.unit)}</td>"
                         f"<td style='padding:4px 10px;text-align:right'>{epct}</td>"
                         f"<td style='padding:4px 10px'><span class='tag {cls}'>{c.verdict}</span></td></tr>")
            return ("<table style='width:100%;border-collapse:collapse;font-size:.92rem;'>"
                    "<tr style='color:#8d99a6;font-size:.8rem'>"
                    "<td style='padding:4px 10px'>channel</td>"
                    "<td style='padding:4px 10px;text-align:right'>measured</td>"
                    "<td style='padding:4px 10px;text-align:right'>predicted</td>"
                    "<td style='padding:4px 10px;text-align:right'>error</td>"
                    "<td style='padding:4px 10px'>verdict</td></tr>"
                    f"{rows}</table>")

        vsub = st.radio("What did you measure?",
                        ["Skidpad", "Acceleration (75 m)", "Speed trace (datalogger)"],
                        horizontal=True)

        # ---------------------------- SKIDPAD --------------------------------- #
        if vsub == "Skidpad":
            st.markdown("Enter **either** the measured peak lateral g **or** your "
                        "timed-circle time — the other is derived so both are checked.")
            sc = st.columns(3)
            mode = sc[0].selectbox("I measured", ["peak lateral g", "timed-circle time (s)"])
            radius = sc[2].number_input("Circle radius (m)", 5.0, 12.0, value=9.125,
                                        step=0.125, format="%.3f",
                                        help="FSAE timed-circle path radius (centreline).")
            if mode == "peak lateral g":
                mg = sc[1].number_input("Measured peak lateral g", 0.5, 2.5,
                                        value=1.40, step=0.01)
                rep = corr_mod.correlate_skidpad(_veh_val, measured_g=mg, radius_m=radius)
            else:
                mt = sc[1].number_input("Measured circle time (s)", 3.0, 8.0,
                                        value=5.00, step=0.01)
                rep = corr_mod.correlate_skidpad(_veh_val, measured_time_s=mt, radius_m=radius)

            st.markdown(_verdict_tag(rep.overall_within_tol), unsafe_allow_html=True)
            st.markdown(_channel_table(rep), unsafe_allow_html=True)
            st.markdown(f'<p class="hint">{rep.summary}</p>', unsafe_allow_html=True)
            if st.button("Log this correlation to handover", key="log_skid"):
                log_decision_now("validation", "Skidpad correlation",
                                 rep.summary, author="validation")
                st.success("Logged.")

        # ------------------------- ACCELERATION ------------------------------- #
        elif vsub == "Acceleration (75 m)":
            st.markdown("Predicted 75 m time comes from the **live lap sim** on your "
                        "current car; enter your measured run to compare.")
            try:
                _live_tire_v = tire_mod.PacejkaLateral(
                    coeffs=dict(st.session_state.tire_coeffs),
                    FNOMIN=st.session_state.tire_fnomin)
                _pt_v = lap_mod.Powertrain()
                _acc = lap_mod.acceleration_time(_veh_val, _pt_v, distance_m=75.0)
                pred_t = float(_acc.lap_time_s)
                if _acc.warning:
                    st.warning(f"⚠ {_acc.warning}")
            except Exception as e:
                pred_t = float("nan")
                st.warning(f"Could not run the acceleration sim safely: {e}")

            ac = st.columns(2)
            mt = ac[0].number_input("Measured 75 m time (s)", 2.5, 8.0,
                                    value=4.00, step=0.01)
            ac[1].markdown(metric("Predicted", f"{pred_t:.3f}" if np.isfinite(pred_t) else "—",
                                  "s"), unsafe_allow_html=True)
            rep = corr_mod.correlate_acceleration(measured_time_s=mt, predicted_time_s=pred_t)
            st.markdown(_verdict_tag(rep.overall_within_tol), unsafe_allow_html=True)
            st.markdown(_channel_table(rep), unsafe_allow_html=True)
            st.markdown(f'<p class="hint">{rep.summary}</p>', unsafe_allow_html=True)

        # --------------------------- SPEED TRACE ------------------------------ #
        else:
            st.markdown("Upload a **CSV with two columns: distance, speed** (one row "
                        "per sample) from GPS or a wheel-speed log. The sim trace is "
                        "resampled onto your distance axis and compared point-for-point.")
            uc = st.columns(3)
            kmh = uc[0].checkbox("Speed is in km/h", value=False)
            track_kind = uc[1].selectbox("Compare against", ["Autocross lap", "Skidpad"])
            ax_scale_v = uc[2].slider("Autocross scale", 0.6, 1.6, 1.0, 0.1)
            up = st.file_uploader("Measured trace CSV (distance, speed)", type=["csv"])

            if up is not None:
                try:
                    import io as _io
                    raw = up.getvalue().decode("utf-8", errors="replace")
                    arr = np.genfromtxt(_io.StringIO(raw), delimiter=",",
                                        names=None, skip_header=0)
                    # tolerate a header row: if first row isn't numeric, retry skipping it
                    if arr.dtype.names is None and (arr.ndim != 2 or arr.shape[1] < 2):
                        arr = np.genfromtxt(_io.StringIO(raw), delimiter=",", skip_header=1)
                    if np.isnan(arr).all():
                        arr = np.genfromtxt(_io.StringIO(raw), delimiter=",", skip_header=1)
                    md = np.asarray(arr)[:, 0]
                    ms = np.asarray(arr)[:, 1]
                except Exception as e:
                    md = ms = None
                    st.error(f"Couldn't parse that CSV as two numeric columns: {e}")

                if md is not None:
                    try:
                        _live_tire_v = tire_mod.PacejkaLateral(
                            coeffs=dict(st.session_state.tire_coeffs),
                            FNOMIN=st.session_state.tire_fnomin)
                        _pt_v = lap_mod.Powertrain()
                        if track_kind == "Skidpad":
                            _lap_v = lap_mod.skidpad_time(_veh_val, _pt_v)
                        else:
                            _trk = lap_mod.default_autocross(scale=ax_scale_v)
                            _lap_v = lap_mod.simulate_lap(_veh_val, _trk, _pt_v)
                    except Exception as e:
                        _lap_v = None
                        st.warning(f"Lap sim could not produce a trace safely: {e}")

                    rep = corr_mod.correlate_speed_trace(
                        md, ms, lap_result=_lap_v, measured_speed_kmh=kmh)

                    if rep.trace is not None and rep.trace.ok:
                        tr = rep.trace
                        st.markdown(_verdict_tag(tr.within_tol), unsafe_allow_html=True)
                        mc = st.columns(4)
                        mc[0].markdown(metric("RMSE", f"{tr.rmse:.2f}", "m/s",
                                              "good" if tr.within_tol else "bad"),
                                       unsafe_allow_html=True)
                        mc[1].markdown(metric("Bias", f"{tr.bias:+.2f}", "m/s",
                                              "good" if abs(tr.bias_frac) <= rep.tolerances['trace_bias_frac'] else "warn"),
                                       unsafe_allow_html=True)
                        _r2 = f"{tr.r2:.3f}" if np.isfinite(tr.r2) else "n/a"
                        mc[2].markdown(metric("R²", _r2, ""), unsafe_allow_html=True)
                        mc[3].markdown(metric("Peak Δ", f"{tr.peak_speed_error:+.1f}", "m/s"),
                                       unsafe_allow_html=True)

                        figV = go.Figure()
                        figV.add_trace(go.Scatter(x=tr.distance, y=tr.measured,
                                                  name="measured", mode="lines"))
                        figV.add_trace(go.Scatter(x=tr.distance, y=tr.predicted,
                                                  name="predicted (sim)", mode="lines"))
                        figV.update_layout(**PLOT_LAYOUT, title="Speed vs distance — measured vs sim",
                                           xaxis_title="distance (m)", yaxis_title="speed (m/s)",
                                           height=380)
                        st.plotly_chart(figV, width='stretch')
                        st.markdown(f'<p class="hint">{rep.summary}</p>', unsafe_allow_html=True)
                        if st.button("Log this correlation to handover", key="log_trace"):
                            log_decision_now("validation", f"Speed-trace correlation ({track_kind})",
                                             rep.summary, author="validation")
                            st.success("Logged.")
                    else:
                        st.error(rep.summary)
            else:
                st.markdown('<p class="hint">No file yet. A two-column CSV like '
                            '<code>distance_m,speed_ms</code> with a header row is fine — '
                            'the parser skips a non-numeric header automatically.</p>',
                            unsafe_allow_html=True)

        st.markdown('<p class="hint">Tolerances are explicit and editable in '
                    '<code>suspension/correlation.py</code> (DEFAULT_TOL). They reflect '
                    'what a quasi-steady-state point-mass model on one tyre set can '
                    'credibly achieve — skidpad tightest, a noisy GPS trace loosest. '
                    'Tighten them and watch the verdict move; that transparency is the '
                    'point.</p>', unsafe_allow_html=True)


# ----------------------------- TAB 13 (merged INTEGRATION) ----------------- #
# INTEGRATION — suspension↔chassis CAD fit + the interface ledger across the
# eight sub-teams, combined into one tab.
with tab13:
    _iview = st.radio(
        "Integration view",
        ["Cross-subsystem ledger", "Subsystem ↔ chassis (CAD fit)",
         "Mount-point clash"],
        horizontal=True, label_visibility="collapsed", key="integration_view")

    _show_ledger = (_iview == "Cross-subsystem ledger")
    if _iview == "Subsystem ↔ chassis (CAD fit)":
        render_suspension_vs_chassis()
    elif _iview == "Mount-point clash":
        render_mountpoint_clash()

if _show_ledger:
  with tab13:
    st.markdown('<p class="hint">OptimumK, ANSYS and SolidWorks each go deep in '
                '<b>one</b> domain. What no team has is this: a place where the '
                '<b>interfaces between</b> subsystems are owned and checked. Each team '
                'enters what it <i>needs from</i> the car and what it <i>provides to</i> '
                'it; KinematiK flags the conflicts — the radiator that won\'t fit the '
                'duct, the motor torque that exceeds the driveline, the eight "~12 kg" '
                'estimates that blow the budget — <b>while they\'re still cheap to fix</b>. '
                'It does not simulate your subsystem (your own tool does that better); '
                'it owns the channels between them, and flags every placeholder number '
                'so a green board never means more than the data behind it. The '
                '<i>Suspension ↔ chassis</i> view above checks the geometric fit of '
                'any physical subsystem against the chassis CAD — suspension by swept '
                'clearance through travel, the rest by static envelope fit.</p>',
                unsafe_allow_html=True)

    _IF = interfaces_mod
    led = _IF.IntegrationLedger.from_dict(st.session_state.ledger)

    # ---- car-level shared limits the checks validate against -------------- #
    with st.expander("Car-level budgets & limits (the shared contract)", expanded=False):
        lc = st.columns(3)
        led.target_mass_kg = lc[0].number_input("Mass target (kg, incl. driver)",
                                                100.0, 400.0, value=float(led.target_mass_kg), step=5.0)
        led.includes_driver_kg = lc[1].number_input("of which driver (kg)",
                                                     0.0, 120.0, value=float(led.includes_driver_kg), step=5.0)
        led.driveline_torque_limit_nm = lc[2].number_input("Driveline torque rating (N·m)",
                                                            0.0, 1000.0,
                                                            value=float(led.driveline_torque_limit_nm or 0.0), step=10.0) or None
        lc2 = st.columns(3)
        led.lv_voltage_v = lc2[0].number_input("LV bus (V)", 6.0, 60.0, value=float(led.lv_voltage_v), step=1.0)
        led.lv_supply_capacity_w = lc2[1].number_input("LV supply capacity (W)", 0.0, 5000.0,
                                                        value=float(led.lv_supply_capacity_w), step=50.0)
        led.accumulator_voltage_v = lc2[2].number_input("Accumulator (V)", 0.0, 600.0,
                                                         value=float(led.accumulator_voltage_v), step=10.0)
        lc3 = st.columns(4)
        ex = lc3[0].number_input("Chassis interior X (mm)", 0.0, 3000.0,
                                 value=float((led.chassis_envelope_mm or (0, 0, 0))[0]), step=10.0)
        ey = lc3[1].number_input("interior Y (mm)", 0.0, 2000.0,
                                 value=float((led.chassis_envelope_mm or (0, 0, 0))[1]), step=10.0)
        ez = lc3[2].number_input("interior Z (mm)", 0.0, 2000.0,
                                 value=float((led.chassis_envelope_mm or (0, 0, 0))[2]), step=10.0)
        led.chassis_envelope_mm = (ex, ey, ez) if (ex and ey and ez) else None
        led.total_cooling_airflow_cms = lc3[3].number_input("Cooling airflow (m³/s)", 0.0, 5.0,
                                                            value=float(led.total_cooling_airflow_cms), step=0.05)

    # Which fields each subsystem typically declares — keeps each editor focused.
    # Every physical subsystem that the "Subsystem ↔ chassis (CAD fit)" view can
    # check against the chassis must be able to declare an envelope here — the fit
    # reads env_x/y/z straight from this ledger. Suspension is the only exception
    # (it gets a swept-clearance check, not a static box), and data-acquisition is
    # excluded from the CAD fit entirely, so neither needs env fields.
    FIELDSETS = {
        "common": ["mass_kg", "cg_x_mm", "cg_y_mm", "cg_z_mm"],
        "aerodynamics": ["env_x_mm", "env_y_mm", "env_z_mm"],
        "brakes": ["brake_torque_nm", "mount_load_n", "mount_points",
                   "peak_current_a",
                   "env_x_mm", "env_y_mm", "env_z_mm"],
        "chassis": ["env_x_mm", "env_y_mm", "env_z_mm"],
        "cooling": ["cooling_airflow_cms", "peak_current_a",
                    "env_x_mm", "env_y_mm", "env_z_mm"],
        "data-acquisition": ["power_draw_w", "voltage_v", "peak_current_a"],
        "electrics": ["power_draw_w", "voltage_v", "peak_current_a",
                      "env_x_mm", "env_y_mm", "env_z_mm"],
        "powertrain": ["peak_torque_nm", "peak_power_kw", "voltage_v",
                       "peak_current_a",
                       "cooling_airflow_cms", "heat_reject_w",
                       "env_x_mm", "env_y_mm", "env_z_mm"],
        "suspension": ["mount_load_n", "mount_points"],
    }
    FIELD_META = {
        "mass_kg": ("Mass (kg)", 0.0, 200.0, 0.5),
        "cg_x_mm": ("CG x — rearward (mm)", 0.0, 3000.0, 10.0),
        "cg_y_mm": ("CG y — right (mm)", -500.0, 500.0, 5.0),
        "cg_z_mm": ("CG z — up (mm)", 0.0, 1000.0, 5.0),
        "env_x_mm": ("Envelope X (mm)", 0.0, 3000.0, 10.0),
        "env_y_mm": ("Envelope Y (mm)", 0.0, 2000.0, 10.0),
        "env_z_mm": ("Envelope Z (mm)", 0.0, 2000.0, 10.0),
        "mount_load_n": ("Peak mount load (N)", 0.0, 50000.0, 100.0),
        "mount_points": ("# mount points", 0.0, 12.0, 1.0),
        "power_draw_w": ("Power draw (W)", 0.0, 5000.0, 10.0),
        "voltage_v": ("Voltage (V)", 0.0, 600.0, 1.0),
        "peak_current_a": ("Peak current (A)", 0.0, 600.0, 5.0),
        "heat_reject_w": ("Heat rejected (W)", 0.0, 50000.0, 100.0),
        "cooling_airflow_cms": ("Cooling airflow req (m³/s)", 0.0, 5.0, 0.05),
        "peak_torque_nm": ("Peak torque (N·m)", 0.0, 1000.0, 10.0),
        "peak_power_kw": ("Peak power (kW)", 0.0, 200.0, 5.0),
        "brake_torque_nm": ("Brake torque/corner (N·m)", 0.0, 5000.0, 50.0),
    }
    EMOJI = {"aerodynamics": "💛", "brakes": "🧡", "chassis": "💜", "cooling": "🩵",
             "data-acquisition": "💚", "electrics": "💙", "powertrain": "❤️",
             "suspension": "🩷"}

    st.markdown("###### Each subsystem's interface")
    st.caption("Fill what's relevant. Blank = not declared yet (reported as MISSING, "
               "not a silent pass). Untick “estimate” once a value is CAD/measured.")
    for s in _IF.SUBSYSTEMS:
        it = led.get(s) or _IF.SubsystemInterface(name=s)
        with st.expander(f"{EMOJI.get(s,'•')}  {s}", expanded=False):
            fields = FIELDSETS["common"] + [f for f in FIELDSETS.get(s, [])
                                            if f not in FIELDSETS["common"]]
            vals = {}
            cols = st.columns(4)
            for i, fld in enumerate(fields):
                label = FIELD_META[fld][0]
                cur = getattr(it, fld)
                # Blank-able text input: empty string == not declared (None), which is
                # distinct from a real 0. Robust for negative-valued fields (CG y) too.
                raw = cols[i % 4].text_input(
                    label, value=("" if cur is None else f"{cur:g}"),
                    key=f"if_{s}_{fld}", placeholder="—")
                raw = raw.strip()
                if raw == "":
                    vals[fld] = None
                else:
                    try:
                        vals[fld] = float(raw)
                    except ValueError:
                        vals[fld] = None
                        cols[i % 4].caption("⚠ not a number — ignored")
            est = st.checkbox("These are estimates / placeholders", value=bool(it.is_estimate),
                              key=f"if_{s}_est")
            rationale = st.text_area(
                "Why — design justification (this is what design-event judges ask for, "
                "and it goes straight into the interface report)",
                value=getattr(it, "rationale", "") or "", key=f"if_{s}_why", height=68)
            oc = st.columns(2)
            owner = oc[0].text_input("Owner", value=getattr(it, "owner", "") or "",
                                     key=f"if_{s}_owner",
                                     placeholder="who owns this interface")
            note = oc[1].text_input("Note (optional)", value=getattr(it, "notes", "") or "",
                                    key=f"if_{s}_note")
            new_it = _IF.SubsystemInterface(
                name=s, is_estimate=est, notes=note, rationale=rationale, owner=owner,
                updated_by=owner or getattr(it, "updated_by", ""),
                updated_on=getattr(it, "updated_on", ""),
                **{k: v for k, v in vals.items()})
            new_it.mounts_on = getattr(it, "mounts_on", None) or (
                "suspension" if s in ("brakes",) else "chassis")
            # Capture any change into an in-SESSION change log. We deliberately do
            # NOT write to the persistent backend on every edit: that fired a remote
            # DB round-trip inside the render loop (and a backend hiccup could crash
            # the app). Changes are batched and committed on demand below.
            try:
                _changes = _IF.diff_interfaces(it.as_dict(), new_it)
            except Exception:
                _changes = []
            if _changes:
                new_it.updated_on = _datetime.date.today().isoformat()
                entry = dict(
                    subsystem=s, when=new_it.updated_on, by=(owner or "—"),
                    changes=_changes,
                    why=rationale.strip() if rationale.strip() else "")
                pending = st.session_state.setdefault("_iface_changelog", [])
                sig = (s, tuple(_changes))
                if not any((e["subsystem"], tuple(e["changes"])) == sig for e in pending):
                    pending.append(entry)
            led.set(new_it)

    # persist edits back to session
    st.session_state.ledger = led.as_dict()

    # ---- run the checks ---- #
    findings = led.check_all()
    summary = _IF.summarize(findings)
    roll = led.mass_rollup()

    # board-level badge
    worst = summary["worst"]
    badge = {"fail": ("bad", "✗ INTEGRATION CONFLICTS"),
             "warning": ("warn", "⚠ WARNINGS"),
             "missing": ("warn", "◴ DATA MISSING"),
             "info": ("good", "ℹ INFO ONLY"),
             "ok": ("good", "✓ ALL CHECKS PASS")}.get(worst, ("warn", worst))
    st.markdown("###### Integration board")
    bc = st.columns(5)
    cnt = summary["counts"]
    bc[0].markdown(metric("Status", badge[1].split(' ', 1)[1] if ' ' in badge[1] else badge[1],
                          "", badge[0]), unsafe_allow_html=True)
    bc[1].markdown(metric("Conflicts", str(cnt["fail"]), "fail",
                          "bad" if cnt["fail"] else "good"), unsafe_allow_html=True)
    bc[2].markdown(metric("Warnings", str(cnt["warning"]), "", "warn" if cnt["warning"] else "good"),
                   unsafe_allow_html=True)
    bc[3].markdown(metric("Missing", str(cnt["missing"]), "", "warn" if cnt["missing"] else "good"),
                   unsafe_allow_html=True)
    _massbadge = "bad" if (roll["declared"] and roll["delta_kg"] - led.includes_driver_kg > 0) else "good"
    bc[4].markdown(metric("Declared mass", f"{roll['total_kg']:.0f}", "kg"),
                   unsafe_allow_html=True)

    # findings list
    _SEV_CLS = {"fail": "bad", "warning": "warn", "missing": "warn",
                "info": "", "ok": "good"}
    for f in sorted(findings, key=lambda x: ["fail", "warning", "missing", "info", "ok"].index(x.severity.value)):
        cls = _SEV_CLS.get(f.severity.value, "")
        who = " ↔ ".join(f"{EMOJI.get(x,'')}{x}" for x in f.subsystems) if f.subsystems else ""
        st.markdown(
            f'<div style="border-left:3px solid var(--line);padding:6px 12px;margin:4px 0;">'
            f'<span class="tag {cls}">{f.severity.value.upper()}</span> '
            f'<b>{f.check}</b> &nbsp;<span style="color:#8d99a6;font-size:.8rem">{who}</span><br>'
            f'<span style="font-size:.92rem">{f.message}</span></div>',
            unsafe_allow_html=True)

    # ---- close the loop with the real physics ---- #
    st.markdown("###### Feed the build back into the physics")
    if roll["cg_mm"]:
        cgz = roll["cg_mm"][2]
        total_with_driver = roll["total_kg"] + (led.includes_driver_kg or 0.0)
        cc = st.columns([2, 1])
        cc[0].markdown(f'<p class="hint">The declared build gives a combined mass of '
                       f'<b>{roll["total_kg"]:.1f} kg</b> (+{led.includes_driver_kg:.0f} kg '
                       f'driver = {total_with_driver:.1f} kg) and a CG height of '
                       f'<b>{cgz:.0f} mm</b>. This is the number suspension\'s load-transfer '
                       f'and the lap sim should be using — push it through so every other '
                       f'tab reflects the real car, not an assumption.</p>',
                       unsafe_allow_html=True)
        if cc[1].button("→ Use this mass & CG in the vehicle model",
                        width='stretch'):
            st.session_state.vp["mass"] = float(total_with_driver)
            st.session_state.vp["cg_height"] = float(cgz)
            _logged = log_decision_now(
                "integration", "Build mass/CG pushed to vehicle model",
                f"Subsystem ledger: {total_with_driver:.1f} kg total, "
                f"CG height {cgz:.0f} mm. Now driving load transfer & lap sim.",
                author="integration")
            st.success(f"Vehicle model updated: {total_with_driver:.1f} kg, "
                       f"CG {cgz:.0f} mm. Other tabs now use it."
                       + ("" if _logged else " (note: couldn't write to the handover "
                          "log — backend unavailable; the model change still applied.)"))
            st.rerun()
    else:
        st.markdown('<p class="hint">Once enough subsystems declare mass AND CG '
                    'location, the combined CG can be pushed straight into the vehicle '
                    'model here — closing the loop between the integration ledger and '
                    'the load-transfer/lap-time physics.</p>', unsafe_allow_html=True)

    # ---- pending change log (batched, committed on demand) ---- #
    _pending = st.session_state.get("_iface_changelog", [])
    if _pending:
        st.markdown("###### Pending change log")
        st.markdown(f'<p class="hint">{len(_pending)} interface change(s) captured this '
                    'session. They\'re held locally and written to the handover record '
                    'only when you commit — so editing never depends on the backend '
                    'being up.</p>', unsafe_allow_html=True)
        for e in _pending[-8:]:
            why = f" — <i>{e['why']}</i>" if e.get("why") else ""
            st.markdown(f'<div style="font-size:.86rem;color:#c9d3dd;padding:2px 0;">'
                        f'<b>{EMOJI.get(e["subsystem"],"")}{e["subsystem"]}</b> '
                        f'<span style="color:#8d99a6">{e["when"]} · {e["by"]}</span>: '
                        f'{"; ".join(e["changes"])}{why}</div>', unsafe_allow_html=True)
        pcols = st.columns([1, 1, 2])
        if pcols[0].button("✓ Commit to handover record", width='stretch'):
            ok = 0
            for e in _pending:
                body = "; ".join(e["changes"]) + (f"  [why: {e['why']}]" if e["why"] else "")
                if log_decision_now("integration", f"{e['subsystem']} interface updated",
                                    body, author=e["by"]):
                    ok += 1
            if ok == len(_pending):
                st.session_state["_iface_changelog"] = []
                st.success(f"Committed {ok} change(s) to the handover record.")
            else:
                st.warning(f"Committed {ok} of {len(_pending)}. The handover backend "
                           "rejected the rest (it may be misconfigured or offline) — "
                           "your edits are safe; try again or export the report instead.")
        if pcols[1].button("Discard pending", width='stretch'):
            st.session_state["_iface_changelog"] = []
            st.rerun()

    # surface any backend logging errors quietly, without having crashed
    if st.session_state.get("_log_errors"):
        with st.expander(f"⚠ {len(st.session_state['_log_errors'])} handover-log "
                         "write(s) failed this session", expanded=False):
            st.caption("The handover/decision store couldn't be written. This is a "
                       "backend/storage issue (e.g. Supabase credentials or table), "
                       "not a problem with your design data — everything on screen is "
                       "intact and the report export below still works.")
            st.code("\n".join(st.session_state["_log_errors"][-5:]))

    # ---- documentation export ---- #
    st.markdown("###### Documentation export")
    st.markdown('<p class="hint">The ledger doubles as living documentation. As each '
                'team locks numbers and writes the <b>why</b>, it\'s captured with owner '
                'and date; changes are batched into the pending log above and committed '
                'to the handover record on demand. Export the whole interface contract, '
                'rationale included, as a design-event-ready document — no write-up '
                'scramble before the report deadline, and no dependency on the backend '
                'being online.</p>',
                unsafe_allow_html=True)
    try:
        _team = project_mod.ProjectStore(PROJECT_PATH).team_name or "FSAE Team"
    except Exception:
        _team = "FSAE Team"
    _report_md = _IF.build_interface_markdown(led, team_name=_team)
    ec = st.columns([1, 1, 2])
    ec[0].download_button("📄 Download interface report (.md)", _report_md,
                          file_name="interface_contract.md", mime="text/markdown",
                          width='stretch')
    with ec[1]:
        if st.button("👁 Preview report", width='stretch'):
            st.session_state._show_iface_report = not st.session_state.get("_show_iface_report", False)
    if st.session_state.get("_show_iface_report"):
        st.markdown(_report_md)

    st.markdown('<p class="hint" style="border-left:2px solid #2a3340;padding-left:10px;">'
                'This is the edge OptimumK / ANSYS / SolidWorks don\'t give you: not '
                'deeper single-domain physics, but a live, checkable contract <i>between</i> '
                'domains — with mass and CG wired into the real vehicle model, every '
                'placeholder number flagged, and the whole thing exportable as the '
                'design justification judges ask for.</p>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Save / Load project — one file captures the whole session
# --------------------------------------------------------------------------- #
#  GGV DIAGRAM TAB — combined acceleration envelope vs speed
# --------------------------------------------------------------------------- #
with tab_ggv:
    st.markdown('<p class="hint">The <b>GGV diagram</b> is the car\'s combined '
                'acceleration envelope: at each speed, the boundary of longitudinal '
                'g (accel up / brake down) vs lateral g it can sustain. It is the '
                'one steady-state picture that shows how much combined grip you '
                'have, where you go power-limited vs grip-limited, and — the point '
                'for an underfunded team — <b>how a design change reshapes the whole '
                'envelope</b> before you cut a single tube. Built on the same live '
                'load-transfer + Pacejka chain as everything else, so CG height, '
                'roll-centre, wheel rate and camber are the levers that move it.</p>',
                unsafe_allow_html=True)

    # Reuse the live dynamics object the rest of the app already solved.
    _veh_ggv = veh

    # ---- Powertrain / aero (same defaults as the Lap Sim tab) ------------ #
    with st.expander("Powertrain & aero (defaults are sensible FSAE-EV values)",
                     expanded=False):
        gc = st.columns(4)
        g_pw = gc[0].number_input("Peak power (kW)", 10.0, 200.0, value=80.0,
                                  step=5.0, key="ggv_pw")
        g_tract = gc[1].number_input("Traction cap (N)", 500.0, 6000.0,
                                     value=2600.0, step=100.0, key="ggv_tract")
        g_cda = gc[2].number_input("Drag CdA (m²)", 0.0, 3.0, value=1.10,
                                   step=0.05, key="ggv_cda")
        g_cla = gc[3].number_input("Downforce ClA (m²)", 0.0, 6.0, value=2.60,
                                   step=0.1, key="ggv_cla")
        gc2 = st.columns(4)
        g_drive = gc2[0].selectbox("Drive", ["rwd", "awd"], index=0, key="ggv_drive")
        g_brake = gc2[1].number_input("Brake cap (g)", 0.5, 3.0, value=1.8,
                                      step=0.1, key="ggv_brake")
        g_crr = gc2[2].number_input("Rolling res. crr", 0.005, 0.05, value=0.018,
                                    step=0.002, format="%.3f", key="ggv_crr")
        g_eff = gc2[3].number_input("Drivetrain eff.", 0.5, 1.0, value=0.90,
                                    step=0.01, key="ggv_eff")

        st.markdown("**Combined-slip coupling** — how lateral and longitudinal "
                    "grip trade against each other (corner entry/exit). The "
                    "symmetric friction circle is the honest default; calibrate the "
                    "exponents and the longitudinal/lateral mu ratio once you have "
                    "drive/brake TTC data.")
        use_comb = st.checkbox("Use a combined-slip tire (friction ellipse)",
                               value=False, key="ggv_usecomb")
        _comb_tire = None
        if use_comb:
            cc = st.columns(3)
            mxr = cc[0].number_input("μx / μy ratio", 0.8, 1.4, value=1.05,
                                     step=0.05, key="ggv_mxr",
                                     help="Peak longitudinal grip / peak lateral "
                                          "grip. >1 is typical (tires put down more "
                                          "Fx than Fy).")
            kx = cc[1].number_input("ellipse exp. kx", 1.5, 3.0, value=2.0,
                                    step=0.1, key="ggv_kx")
            ky = cc[2].number_input("ellipse exp. ky", 1.5, 3.0, value=2.0,
                                    step=0.1, key="ggv_ky")
            is_cal = st.checkbox("These exponents are fitted to my Fx TTC data",
                                 value=False, key="ggv_iscal")
            _comb_tire = tire_mod.CombinedSlipTire(
                lateral=tire_mod.PacejkaLateral(
                    coeffs=dict(st.session_state.tire_coeffs),
                    FNOMIN=st.session_state.tire_fnomin),
                mu_x_ratio=mxr, ell_kx=kx, ell_ky=ky, is_calibrated=is_cal)
            st.caption("Status: " + _comb_tire.status())

    # Build the lap-sim Powertrain, then derive GGVParams from it so the GGV and
    # the Lap Sim tab share one source of truth.
    _pt_ggv = lap_mod.Powertrain(power_kw=g_pw, max_tractive_n=g_tract,
                                 drivetrain_eff=g_eff, cda=g_cda, cla=g_cla,
                                 crr=g_crr, drive=g_drive, brake_g_cap=g_brake,
                                 combined_tire=_comb_tire)
    _gp = ggv_mod.GGVParams.from_powertrain(_pt_ggv)

    # ---- Build the envelope --------------------------------------------- #
    _vmax_ui = st.slider("Top speed to chart (m/s)", 20.0, 45.0, 38.0, 1.0,
                         key="ggv_vmax")
    try:
        _speeds = np.linspace(5.0, _vmax_ui, 12)
        # Memoize the envelope: regenerate only when an input that affects it
        # changes. _veh_ggv / _gp aren't hashable, so key on a cheap signature
        # built from the underlying primitives instead.
        _ggv_sig = (
            round(float(_vmax_ui), 3),
            repr(st.session_state.vp),
            repr(dict(st.session_state.tire_coeffs)),
            float(st.session_state.tire_fnomin),
            round(float(g_pw), 3), round(float(g_tract), 3), round(float(g_eff), 4),
            round(float(g_cda), 4), round(float(g_cla), 4), round(float(g_crr), 5),
            g_drive, round(float(g_brake), 4),
            st.session_state.get("topology", "double_wishbone"),
            repr(hp_dict),
        )
        if st.session_state.get("_ggv_sig") == _ggv_sig and \
           st.session_state.get("_ggv_res") is not None:
            _res = st.session_state["_ggv_res"]
        else:
            _res = ggv_mod.GGVGenerator(_veh_ggv, _gp).generate(speeds=_speeds)
            st.session_state["_ggv_sig"] = _ggv_sig
            st.session_state["_ggv_res"] = _res
    except Exception as e:
        _res = None
        st.error(f"GGV generation failed: {e}")

    if _res is not None:
        for w in _res.warnings:
            st.warning(f"⚠ {w}")

        # headline metrics at a representative mid-corner speed
        _i_mid = int(np.argmin(np.abs(_res.speeds - 15.0)))
        mc = st.columns(4)
        mc[0].markdown(metric("Peak lateral", f"{np.nanmax(_res.max_lat_g):.2f}",
                              "g"), unsafe_allow_html=True)
        mc[1].markdown(metric("Peak accel", f"{np.nanmax(_res.max_accel_g):.2f}",
                              "g"), unsafe_allow_html=True)
        mc[2].markdown(metric("Peak braking", f"{np.nanmax(_res.max_brake_g):.2f}",
                              "g"), unsafe_allow_html=True)
        mc[3].markdown(metric("Grip model",
                              "Pacejka" if "Pacejka" in _res.grip_model else "linear",
                              ""), unsafe_allow_html=True)

        # ---- GGV cross-sections (the diagram itself) -------------------- #
        figG = go.Figure()
        _ns = len(_res.speeds)
        for i, v in enumerate(_res.speeds):
            if i % 2 and i != _ns - 1:
                continue  # thin the legend a little
            lon, lat = _res.long_g[i], _res.lat_g[i]
            # mirror lateral so both left and right cornering show
            x = np.concatenate([lat, -lat[::-1]])
            y = np.concatenate([lon, lon[::-1]])
            # colour ramp dim->cyan with speed
            t = i / max(_ns - 1, 1)
            col = f"rgba({int(55 + 200 * (1 - t))},{int(120 + 100 * t)},"\
                  f"{int(160 + 50 * t)},0.95)"
            figG.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                      line=dict(color=col, width=2),
                                      name=f"{v:.0f} m/s"))
        figG.update_layout(**PLOT_LAYOUT,
                           title="GGV cross-sections — combined g envelope by speed",
                           xaxis_title="lateral g", yaxis_title="longitudinal g  (+accel / −brake)",
                           height=460)
        figG.add_hline(y=0, line_color="#33414e", line_width=1)
        figG.add_vline(x=0, line_color="#33414e", line_width=1)
        st.plotly_chart(figG, width='stretch')
        st.markdown('<p class="hint">Each closed curve is the limit at one speed. '
                    'It grows downward (braking gains from aero downforce + drag) and '
                    'wider (more cornering grip with downforce) as speed rises; the '
                    'top flattens where the car runs out of power. Keep the combined-g '
                    'vector inside the curve for the current speed.</p>',
                    unsafe_allow_html=True)

        # ---- Capability vs speed --------------------------------------- #
        figC = go.Figure()
        figC.add_trace(go.Scatter(x=_res.speeds, y=_res.max_lat_g, mode="lines+markers",
                                  line=dict(color=CYAN, width=2.5), name="max lateral g"))
        figC.add_trace(go.Scatter(x=_res.speeds, y=_res.max_accel_g, mode="lines+markers",
                                  line=dict(color=AMBER, width=2.5), name="max accel g"))
        figC.add_trace(go.Scatter(x=_res.speeds, y=_res.max_brake_g, mode="lines+markers",
                                  line=dict(color=RED, width=2.5), name="max braking g"))
        figC.update_layout(**PLOT_LAYOUT, title="Capability vs speed",
                           xaxis_title="speed (m/s)", yaxis_title="g", height=320)
        st.plotly_chart(figC, width='stretch')

        # ---- "What does changing X do?" sweep -------------------------- #
        st.markdown("###### Design-input sweep — what reshapes the envelope?")
        sw = st.columns([2, 2, 1, 1])
        _param_opts = {
            "CG height (mm)": ("cg_height", [250, 300, 350, 400]),
            "Weight dist. front (frac)": ("weight_dist_front", [0.42, 0.46, 0.50, 0.54]),
            "Front static camber (°)": ("static_camber_front", [0.0, -1.0, -2.0, -3.0]),
            "Front roll stiffness (N·m/°)": ("roll_stiffness_front", [250, 350, 450, 550]),
            "Downforce ClA (m²)": ("cl_a", [0.0, 1.5, 2.5, 3.5]),
            "Peak power (W)": ("power_w", [40000, 60000, 80000, 100000]),
        }
        _param_label = sw[0].selectbox("Parameter", list(_param_opts.keys()),
                                       key="ggv_sweep_param")
        _metric_label = sw[1].selectbox(
            "Metric", ["max lateral g", "max accel g", "max braking g"],
            key="ggv_sweep_metric")
        _sweep_v = sw[2].number_input("at speed (m/s)", 5.0, 45.0, 20.0, 1.0,
                                      key="ggv_sweep_v")
        _metric_key = {"max lateral g": "max_lat_g", "max accel g": "max_accel_g",
                       "max braking g": "max_brake_g"}[_metric_label]
        if sw[3].button("Sweep", key="ggv_sweep_btn", width='stretch'):
            st.session_state._ggv_run_sweep = True
        if st.session_state.get("_ggv_run_sweep"):
            _pname, _pvals = _param_opts[_param_label]
            try:
                _sres = ggv_mod.sweep_parameter(_veh_ggv, _gp, _pname, _pvals,
                                                speed=_sweep_v, metric=_metric_key)
                figS = go.Figure()
                figS.add_trace(go.Scatter(x=_sres["values"], y=_sres["metric"],
                                          mode="lines+markers",
                                          line=dict(color=CYAN, width=2.5)))
                figS.update_layout(**PLOT_LAYOUT,
                                   title=f"{_metric_label} vs {_param_label} "
                                         f"@ {_sweep_v:.0f} m/s",
                                   xaxis_title=_param_label, yaxis_title=_metric_label,
                                   height=300)
                st.plotly_chart(figS, width='stretch')
                if _pname == "static_camber_front" and \
                        len(set(round(x, 3) for x in _sres["metric"])) <= 1:
                    st.markdown('<p class="hint">Camber looks flat because the '
                                '<b>generic tire is nearly camber-insensitive</b> by '
                                'design. Load your TTC-fitted tire (TIRE &amp; GRIP) — '
                                'a real fit carries camber terms and this curve will '
                                'respond.</p>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sweep failed: {e}")

        # ---- Cross-check against the Lap Sim --------------------------- #
        with st.expander("Cross-check: does the GGV agree with the Lap Sim?",
                         expanded=False):
            st.markdown('<p class="hint">Both the GGV and the Lap Sim run on the '
                        'same load-transfer + Pacejka chain, so their axis limits '
                        'should match. This button compares them directly — a '
                        'divergence means one has drifted and is worth chasing.</p>',
                        unsafe_allow_html=True)
            if st.button("Run cross-check", key="ggv_validate_btn"):
                try:
                    _vres = ggv_mod.validate_against_laptime(_veh_ggv, _pt_ggv)
                    _cls = "good" if _vres["ok"] else "warn"
                    st.markdown(metric("Max difference vs Lap Sim",
                                       f"{_vres['max_reldiff'] * 100:.2f}", "%", _cls),
                                unsafe_allow_html=True)
                    if _vres["ok"]:
                        st.success("Agrees with the Lap Sim within tolerance.")
                    else:
                        st.warning(_vres.get("note", _vres["reason"]))
                    # small comparison table
                    import pandas as _pd
                    _df = _pd.DataFrame({
                        "speed m/s": [round(x, 1) for x in _vres["speeds"]],
                        "lat GGV": [round(x, 3) for x in _vres["lat_ggv"]],
                        "lat Lap": [round(x, 3) for x in _vres["lat_lap"]],
                        "accel GGV": [round(x, 3) for x in _vres["accel_ggv"]],
                        "accel Lap": [round(x, 3) for x in _vres["accel_lap"]],
                        "brake GGV": [round(x, 3) for x in _vres["brake_ggv"]],
                        "brake Lap": [round(x, 3) for x in _vres["brake_lap"]],
                    })
                    st.dataframe(_df, width='stretch', hide_index=True)
                except Exception as e:
                    st.error(f"Cross-check failed: {e}")

        if st.session_state.get("tire_is_default", True):
            st.markdown('<p class="hint" style="border-left:2px solid #5a4317;'
                        'padding-left:10px;">Running on the <b>generic default tire</b>. '
                        'The envelope shape and the way it responds to setup changes are '
                        'right; load your TTC-fitted tire in TIRE &amp; GRIP before '
                        'quoting absolute g numbers.</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  TRANSIENT TAB — explicit high-frequency time-step DAE solver
# --------------------------------------------------------------------------- #
with tab_tr:
    st.markdown("#### ◢ TRANSIENT — explicit high-frequency time-step solver")
    st.markdown(
        '<p class="hint">The LAP TIME tab is <b>quasi-steady-state</b>: it assumes the '
        'car sits at a balanced equilibrium at every point and solves a speed profile. '
        'This solver integrates the full vehicle DAE <b>millisecond by millisecond</b> '
        '(explicit RK4 @ 1&nbsp;ms) on the <i>same</i> tyre, damper and geometry, so it '
        'shows what QSS assumes away: turn-in lag and yaw overshoot, snap-oversteer and '
        'the countersteer that catches it, pitch/dive through a brake&nbsp;→&nbsp;throttle '
        'transition, and kerb strikes (wheel hop, contact-load spikes, wheel lift).</p>',
        unsafe_allow_html=True)

    _veh_tr = veh  # the live model the rest of the app already solved

    def _trfig(title, xtitle, ytitle, height=320):
        f = go.Figure()
        f.update_layout(**PLOT_LAYOUT, title=title, xaxis_title=xtitle,
                        yaxis_title=ytitle, height=height)
        return f

    _MAN = [
        "Step steer (turn-in & yaw overshoot)",
        "Snap-oversteer + recovery",
        "Brake → throttle (pitch & dive)",
        "Kerb strike (wheel hop & lift)",
        "Transient vs QSS corner (the rise QSS skips)",
    ]
    mlabel = st.selectbox("Manoeuvre", _MAN, key="tr_maneuver")
    cc = st.columns(4)

    show_uncaught = False
    if mlabel.startswith("Step steer"):
        steer_deg = cc[0].number_input("Steer angle (°)", 0.5, 12.0, 4.0, 0.5,
                                       key="tr_ss_steer")
        u0 = cc[1].number_input("Entry speed (m/s)", 3.0, 40.0, 18.0, 1.0,
                                key="tr_ss_u0")
        kind, kw = "step_steer", dict(steer_deg=float(steer_deg), u0=float(u0))
    elif mlabel.startswith("Snap"):
        u0 = cc[0].number_input("Entry speed (m/s)", 5.0, 40.0, 16.0, 1.0,
                                key="tr_so_u0")
        steer_deg = cc[1].number_input("Corner steer (°)", 1.0, 8.0, 3.8, 0.2,
                                       key="tr_so_steer")
        brake_stab = cc[2].number_input("Trailing-brake stab (0–1)", 0.0, 1.0, 0.45,
                                        0.05, key="tr_so_bs")
        show_uncaught = cc[3].checkbox("Overlay uncaught spin", value=True,
                                       key="tr_so_unc")
        kind, kw = "snap_oversteer", dict(u0=float(u0), steer_deg=float(steer_deg),
                                          brake_stab=float(brake_stab), recover=True)
    elif mlabel.startswith("Brake"):
        u0 = cc[0].number_input("Entry speed (m/s)", 5.0, 40.0, 25.0, 1.0,
                                key="tr_bt_u0")
        kind, kw = "brake_to_throttle", dict(u0=float(u0))
    elif mlabel.startswith("Kerb"):
        u0 = cc[0].number_input("Speed (m/s)", 3.0, 40.0, 20.0, 1.0, key="tr_cb_u0")
        curb_h = cc[1].number_input("Kerb height (mm)", 5.0, 80.0, 30.0, 5.0,
                                    key="tr_cb_h") / 1000.0
        wsel = cc[2].selectbox("Wheels over kerb",
                               ["FL + RL (left side)", "FL only", "All four"],
                               key="tr_cb_w")
        wheels = {"FL + RL (left side)": ("FL", "RL"), "FL only": ("FL",),
                  "All four": ("FL", "FR", "RL", "RR")}[wsel]
        kind, kw = "curb_strike", dict(u0=float(u0), curb_h=float(curb_h),
                                       wheels=wheels)
    else:
        u0 = cc[0].number_input("Entry speed (m/s)", 5.0, 40.0, 16.0, 1.0,
                                key="tr_qs_u0")
        kind, kw = "_settling", dict(u0=float(u0))

    run = st.button("▶ Run transient simulation", type="primary", key="tr_run")
    if run:
        with st.spinner("Integrating the vehicle DAE at 1 ms… (a few seconds)"):
            try:
                if kind == "_settling":
                    sr = transient_mod.transient_vs_qss_corner(_veh_tr, u0=kw["u0"])
                    st.session_state["_tr_result"] = ("settling", sr, None, mlabel)
                elif kind == "snap_oversteer" and show_uncaught:
                    res = transient_mod.run_maneuver(_veh_tr, kind, **kw)
                    kw_u = dict(kw); kw_u["recover"] = False
                    res_u = transient_mod.run_maneuver(_veh_tr, kind, **kw_u)
                    st.session_state["_tr_result"] = (kind, res, res_u, mlabel)
                else:
                    res = transient_mod.run_maneuver(_veh_tr, kind, **kw)
                    st.session_state["_tr_result"] = (kind, res, None, mlabel)
            except Exception as e:
                st.session_state["_tr_result"] = ("error", str(e), None, mlabel)

    stored = st.session_state.get("_tr_result")
    if not stored:
        st.info("Pick a manoeuvre, set the inputs, and press **Run**. The solver "
                "reuses the tyre, damper and geometry from the rest of the app, so "
                "every setup change you make elsewhere shows up here too.")
    elif stored[0] == "error":
        st.error(f"Transient run failed: {stored[1]}")
    else:
        kind_done, res, res_u, label_done = stored
        st.caption(f"Showing: **{label_done}**")

        if kind_done == "settling":
            sr = res
            if not sr.ok:
                st.warning("Settling analysis returned a flagged result.")
            m = st.columns(5)
            m[0].metric("QSS max lat g", f"{sr.qss_max_ay_g:.2f}")
            m[1].metric("Transient steady", f"{sr.steady_ay_g:.2f} g")
            m[2].metric("Peak (overshoot)",
                        f"{sr.peak_ay_g:.2f} g", f"{sr.overshoot_pct:+.1f}%")
            m[3].metric("Rise time (90%)",
                        ("—" if not np.isfinite(sr.rise_time_s)
                         else f"{sr.rise_time_s*1000:.0f} ms"))
            m[4].metric("Settle (±5%)",
                        ("—" if not np.isfinite(sr.settle_time_s)
                         else f"{sr.settle_time_s*1000:.0f} ms"))
            rr = sr.result
            fig = _trfig("Lateral g — the rise QSS replaces with a single number",
                         "time (s)", "lateral g", height=360)
            fig.add_trace(go.Scatter(x=rr.t, y=np.abs(rr.ay), mode="lines",
                          line=dict(color=CYAN, width=2), name="transient ay"))
            fig.add_hline(y=sr.steady_ay_g, line=dict(color=AMBER, dash="dash"),
                          annotation_text="transient steady")
            fig.add_hline(y=sr.qss_max_ay_g, line=dict(color=RED, dash="dot"),
                          annotation_text="QSS max")
            st.plotly_chart(fig, width='stretch')
            st.caption("QSS reports the steady corner as one number. The transient "
                       "solver shows the car building up to it — the rise time, any "
                       "overshoot, and the settle — the unsettled phase QSS assumes "
                       "away. The steady value sits below the QSS limit because this "
                       "is a sub-limit corner, by construction.")
            warns = list(getattr(sr, "warnings", []) or [])
        else:
            s = res.summary()
            warns = list(res.warnings or [])
            if not res.ok:
                st.warning("Run hit a numerical limit and the trace was truncated — "
                           "metrics below are from what completed.")

            if kind_done == "step_steer":
                m = st.columns(4)
                m[0].metric("Peak yaw rate", f"{s.get('peak_yaw_rate_deg_s',0):.0f} °/s")
                m[1].metric("Steady yaw rate", f"{np.degrees(res.r[-1]):.0f} °/s")
                m[2].metric("Peak lateral g", f"{s.get('peak_ay_g',0):.2f}")
                m[3].metric("Peak body roll", f"{s.get('peak_roll_deg',0):.2f} °")
                g1, g2 = st.columns(2)
                f1 = _trfig("Yaw rate — overshoot then settle", "time (s)", "yaw rate (°/s)")
                f1.add_trace(go.Scatter(x=res.t, y=np.degrees(res.r), mode="lines",
                             line=dict(color=CYAN, width=2), name="yaw rate"))
                f1.add_hline(y=np.degrees(res.r[-1]), line=dict(color=DIM, dash="dash"),
                             annotation_text="steady")
                g1.plotly_chart(f1, width='stretch')
                f2 = _trfig("Lateral g & body roll", "time (s)", "lateral g")
                f2.add_trace(go.Scatter(x=res.t, y=res.ay, mode="lines",
                             line=dict(color=AMBER, width=2), name="lateral g"))
                f2.add_trace(go.Scatter(x=res.t, y=np.degrees(res.roll), mode="lines",
                             line=dict(color=RED, width=1.4), name="roll (°)", yaxis="y2"))
                f2.update_layout(yaxis2=dict(title="roll (°)", overlaying="y",
                                 side="right", gridcolor="#1d242c"))
                g2.plotly_chart(f2, width='stretch')

            elif kind_done == "snap_oversteer":
                m = st.columns(3)
                m[0].metric("Caught: final sideslip", f"{np.degrees(res.beta[-1]):.1f} °")
                if res_u is not None:
                    m[1].metric("Uncaught: final sideslip",
                                f"{np.degrees(res_u.beta[-1]):.0f} °", "spins", delta_color="inverse")
                m[2].metric("Peak yaw rate", f"{s.get('peak_yaw_rate_deg_s',0):.0f} °/s")
                f1 = _trfig("Body sideslip β — divergence vs recovery",
                            "time (s)", "sideslip β (°)", height=360)
                if res_u is not None:
                    f1.add_trace(go.Scatter(x=res_u.t, y=np.degrees(res_u.beta),
                                 mode="lines", line=dict(color=RED, width=2),
                                 name="uncaught → spins"))
                f1.add_trace(go.Scatter(x=res.t, y=np.degrees(res.beta), mode="lines",
                             line=dict(color="#3ec46d", width=2),
                             name="feedback countersteer → caught"))
                st.plotly_chart(f1, width='stretch')
                f2 = _trfig("Steer input (the catch) & yaw rate", "time (s)", "steer (°)")
                f2.add_trace(go.Scatter(x=res.t, y=np.degrees(res.steer), mode="lines",
                             line=dict(color=AMBER, width=1.6), name="steer (°)"))
                f2.add_trace(go.Scatter(x=res.t, y=np.degrees(res.r), mode="lines",
                             line=dict(color=CYAN, width=1.4), name="yaw rate (°/s)", yaxis="y2"))
                f2.update_layout(yaxis2=dict(title="yaw rate (°/s)", overlaying="y",
                                 side="right", gridcolor="#1d242c"))
                st.plotly_chart(f2, width='stretch')
                st.caption("Lift-off plus a trailing-brake stab unloads the rear; "
                           "uncaught it diverges into a spin, while a state-feedback "
                           "countersteer pulls the sideslip back toward zero — the "
                           "recovery a steady-state model can't represent because it "
                           "never lets the car leave equilibrium.")

            elif kind_done == "brake_to_throttle":
                m = st.columns(4)
                m[0].metric("Pitch dive", f"{np.degrees(res.pitch.min()):.2f} °")
                m[1].metric("Pitch squat", f"{np.degrees(res.pitch.max()):.2f} °")
                m[2].metric("Peak decel", f"{res.ax.min():.2f} g")
                m[3].metric("Peak accel", f"{res.ax.max():.2f} g")
                f1 = _trfig("Pitch — dive under braking, squat under power",
                            "time (s)", "pitch (°)  (− dive / + squat)", height=340)
                f1.add_trace(go.Scatter(x=res.t, y=np.degrees(res.pitch), mode="lines",
                             line=dict(color="#a855f7", width=2), name="pitch (°)"))
                f1.add_trace(go.Scatter(x=res.t, y=res.ax, mode="lines",
                             line=dict(color=AMBER, width=1.2), name="long. g", yaxis="y2"))
                f1.update_layout(yaxis2=dict(title="long. accel (g)", overlaying="y",
                                 side="right", gridcolor="#1d242c"))
                st.plotly_chart(f1, width='stretch')
                f2 = _trfig("Axle vertical load through the transition",
                            "time (s)", "axle load (N)")
                f2.add_trace(go.Scatter(x=res.t, y=res.Fz[:, 0] + res.Fz[:, 1],
                             mode="lines", line=dict(color=CYAN, width=1.6),
                             name="front axle"))
                f2.add_trace(go.Scatter(x=res.t, y=res.Fz[:, 2] + res.Fz[:, 3],
                             mode="lines", line=dict(color=RED, width=1.6),
                             name="rear axle"))
                st.plotly_chart(f2, width='stretch')
                st.caption("The sprung mass rocks forward (dive) then back (squat); the "
                           "digressive damper sets how fast the ringing settles. QSS has "
                           "no pitch degree of freedom, so this whole transient is "
                           "invisible to it.")

            elif kind_done == "curb_strike":
                m = st.columns(3)
                m[0].metric("Peak contact load", f"{s.get('max_Fz_N',0):.0f} N")
                m[1].metric("Min contact load", f"{s.get('min_Fz_N',0):.0f} N")
                m[2].metric("Wheel lift?", "yes" if s.get("wheel_lift") else "no")
                names = ["FL", "FR", "RL", "RR"]
                cols = [CYAN, AMBER, RED, "#3ec46d"]
                f1 = _trfig("Contact vertical load — spike & wheel lift",
                            "time (s)", "Fz (N)", height=340)
                for i in range(4):
                    f1.add_trace(go.Scatter(x=res.t, y=res.Fz[:, i], mode="lines",
                                 line=dict(color=cols[i], width=1.4), name=names[i]))
                f1.add_hline(y=0, line=dict(color=DIM, width=1))
                st.plotly_chart(f1, width='stretch')
                f2 = _trfig("Suspension (wheel) velocity — the high-frequency hop",
                            "time (s)", "wheel vel (m/s, + bump)")
                for i in range(4):
                    f2.add_trace(go.Scatter(x=res.t, y=res.susp_vel[:, i], mode="lines",
                                 line=dict(color=cols[i], width=1.2), name=names[i]))
                st.plotly_chart(f2, width='stretch')
                st.caption("The unsprung mass hops at ~15–20 Hz; the contact load spikes "
                           "well above static and can momentarily drop to zero (wheel "
                           "lift). A QSS point mass has no unsprung mass and cannot "
                           "represent this millisecond-scale event at all.")

        st.caption(f"Tyre: {res.meta.get('tire','n/a') if kind_done!='settling' else _veh_tr.grip_model_name()}"
                   if kind_done != "settling" else
                   f"Grip model: {_veh_tr.grip_model_name()}")
        if warns:
            with st.expander(f"⚠ {len(warns)} solver warning(s)"):
                for w in warns:
                    st.write("• " + str(w))
        st.markdown(
            '<p class="hint">Honest scope: this resolves the dominant transient modes '
            '(yaw/sideslip, heave/pitch/roll, four unsprung wheel-hops, lateral tyre '
            'relaxation). Longitudinal force is demanded and friction-ellipse-limited '
            'rather than spun up as full slip-ratio wheel states, and tyre thermal state '
            'and a closed-loop racing line are out of scope — flagged, not faked. '
            'Use QSS (LAP TIME) for the lap-time number; use this for the unsteady '
            'behaviour behind it.</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("#### Save / load your work")
st.markdown('<p class="hint">One file holds your whole session — geometry, vehicle '
            'setup, and the handover log (decisions, notes, weights). Save it to keep '
            'your progress or hand it to a teammate; load it to pick up exactly where '
            'you left off.</p>', unsafe_allow_html=True)

# Build the unified project bundle.
_store_for_save = project_mod.ProjectStore(PROJECT_PATH)
project_bundle = {
    "kinematik_version": "1.0",
    "saved": _datetime.datetime.now().isoformat(timespec="seconds"),
    "hardpoints": hp_dict,
    "vehicle": st.session_state.vp,
    "ledger": st.session_state.get("ledger"),
    "handover": json.loads(_store_for_save.as_json()),
}

sc1, sc2, sc3 = st.columns([1, 1, 1])
sc1.download_button("💾 Save project (.json)", json.dumps(project_bundle, indent=2),
                    file_name="kinematik_project.json", mime="application/json",
                    width='stretch')

# CSV of the sweep (tabular data — handy for report plots / Excel)
import io
buf = io.StringIO()
buf.write("travel_mm,camber_deg,toe_deg,caster_deg,kpi_deg,scrub_mm\n")
for st_ in sweep:
    buf.write(f"{st_.travel:.2f},{st_.camber:.4f},{st_.toe:.4f},"
              f"{st_.caster:.4f},{st_.kpi:.4f},{st_.scrub_radius:.3f}\n")
sc2.download_button("⬇ Sweep data (.csv)", buf.getvalue(),
                    file_name="kinematik_sweep.csv", mime="text/csv",
                    width='stretch')

with sc3:
    loaded = st.file_uploader("📂 Load project (.json)", type=["json"],
                              key="load_project", label_visibility="visible")
    if loaded is not None:
        try:
            data = json.load(loaded)
            if "hardpoints" in data:
                st.session_state.hp = data["hardpoints"]
            if "vehicle" in data:
                st.session_state.vp = data["vehicle"]
            if data.get("ledger"):
                st.session_state.ledger = data["ledger"]
            # restore handover data into the store
            if "handover" in data:
                _s = project_mod.ProjectStore(PROJECT_PATH)
                _s._apply(data["handover"])
                _s.save()
            st.success("Project loaded — geometry, vehicle, and handover restored.")
            if st.button("Apply loaded project"):
                st.rerun()
        except Exception as e:
            st.error(f"Couldn't read that project file: {e}")

# ----------------------------- TAB PCB (ELECTRONICS) ----------------------- #
with tab_pcb:
    render_pcb_board()
    render_harness()

st.markdown('<p class="hint" style="padding-top:.4rem;">Open source · MIT. Fork it, '
            'validate against your OptimumK model, send a PR. '
            '<i>Tip: on the hosted app, save your project before closing the tab — '
            'geometry tweaks aren\'t auto-saved the way the handover log is.</i></p>',
            unsafe_allow_html=True)
