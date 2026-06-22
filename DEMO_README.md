# Elbee Baja — Standalone Demo Build

This is the **Baja SAE** suspension & vehicle-dynamics studio, packaged to run on
its own with **no Supabase / cloud connection**. Everything stays on the laptop.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

It opens straight into the Baja studio (no team selector — this build is Baja
only). Pick your suspension topology in the sidebar and the whole studio follows.

## What "disconnected from Supabase" means here

- The storage backend is **forced to a local `project.json`** in this folder.
  Even if `SUPABASE_URL` / `SUPABASE_KEY` happen to be set on the demo machine,
  the app will **not** attempt any network connection.
- The status badge on the Weight & Handover tab reads *"local demo — saves to
  project.json (cloud sync off)"* instead of nagging to set up Supabase.
- Lead Notes save locally for the session (no cross-user sync, as expected for a
  single-laptop demo).
- The `supabase` package is commented out of `requirements.txt`, so you don't
  need it installed.

To re-enable cloud sync later, restore the credential check in
`suspension/project.py::_auto_backend` (the original logic is described in the
docstring there) and un-comment `supabase` in `requirements.txt`.

## Demo focus

The headline of this build is **suspension + steering durability** — the failure
mode that ends off-road runs (bent tie rods, sheared rod-ends, bump-steer, links
failing under course abuse). The Compliance / durability checks (member
deflection, bolted-joint separation, bump-steer over full travel) are
foregrounded, and the front/rear suspension subteams are first-class owners.
That's the competitive edge to walk the suspension lead through.

See `README.md` for the full feature tour.

---
Original engine by Frederik Thio (KinematiK, MIT). Baja rebase under MIT.
