"""
Cooperative Name Search & Duplicate Detection: Streamlit app
Run:  streamlit run app.py
Engine, features and models are the ones from the notebook (see coop_name_search.py).
Similarity is evidence, not proof of identity. Nothing here merges, deletes or approves records.
"""
import contextlib
import io
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import streamlit as st

import coop_name_search as core

HERE = Path(__file__).resolve().parent
OUT = core.OUT
MODEL_FILE = OUT / "name_search_model.joblib"
DECISIONS = ["", "Same", "Different", "Needs more information"]

st.set_page_config(page_title="Cooperative Name Search", page_icon="🔎", layout="wide")


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading search model...")
def load_engine(mtime: float):
    b = joblib.load(MODEL_FILE)
    return core.NameSearch(b["register"], b["vec"], b["model"]), b["model_name"], b["register"]


def icon(band: str) -> str:
    return "🔴" if band.startswith("LIKELY") else "🟠" if band == "REVIEW QUEUE" else "⚪"


def read_csv_safe(name: str):
    f = OUT / name
    return pd.read_csv(f) if f.exists() else None


def ss_queue() -> list:
    return st.session_state.setdefault("queue", [])


def pending_names_path() -> Path:
    OUT.mkdir(exist_ok=True, parents=True)
    return OUT / "pending_new_names.csv"


def append_pending_name(name: str, region: str = "", source: str = "search") -> None:
    path = pending_names_path()
    row = {
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "name": name.strip(),
        "region": region,
        "status": "pending_review",
    }
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and ((existing["name"].astype(str).str.strip().str.lower() == row["name"].lower()) &
                               (existing["region"].fillna("").astype(str).str.strip().str.lower() == row["region"].lower())).any():
        return
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def show_headline(res: pd.DataFrame):
    top = res.iloc[0]
    if top.match_prob >= core.BAND_LIKELY:
        st.error(f"**Likely duplicate:** closest existing name is **{top['name']}** ({top.region}), "
                 f"probability {top.match_prob:.1%}. An officer should confirm.")
    elif top.match_prob >= core.BAND_REVIEW:
        st.warning(f"**Possible duplicate, send to review:** **{top['name']}** ({top.region}), "
                   f"probability {top.match_prob:.1%}.")
    else:
        st.success(f"No close existing name found (best candidate {top.match_prob:.1%}). "
                   "This is evidence, not proof; an officer makes the decision.")


def results_table(res: pd.DataFrame):
    view = res.assign(flag=res.band.map(icon))[ ["flag", "coop_id", "name", "region", "match_prob", "band", "evidence"] ]
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "flag": st.column_config.TextColumn("", width="small"),
        "coop_id": "ID", "name": "Registered name", "region": "Region",
        "match_prob": st.column_config.ProgressColumn("Match probability", min_value=0.0, max_value=1.0, format="%.3f"),
        "band": "Confidence band", "evidence": "Evidence"})


# ----------------------------------------------------------------------------- guard: model present
engine = None
if MODEL_FILE.exists():
    try:
        engine, model_name, register = load_engine(MODEL_FILE.stat().st_mtime)
    except Exception as e:  # e.g. scikit-learn version mismatch
        st.warning(f"Saved model could not be loaded ({type(e).__name__}). Retrain it on the *Data & retrain* page.")

with st.sidebar:
    st.title("🔎 Coop Name Search")
    pages = ["Search", "Check a proposed name", "Review queue", "Dashboard", "Model performance",
             "Data & retrain", "About & responsible use"]
    page = st.radio("Go to", pages, label_visibility="collapsed")
    st.divider()
    if engine is not None:
        st.caption(f"**Register:** {len(register):,} names  \n**Model:** {model_name}  \n"
                   f"**Queued for review:** {len(ss_queue())}")
    st.caption("Similarity is evidence, not proof of identity. No automatic merge, delete or approval.")

if engine is None and page not in ("Data & retrain", "About & responsible use"):
    st.info("No trained model found. Open **Data & retrain** and train one.")
    st.stop()

# ----------------------------------------------------------------------------- Search
if page == "Search":
    st.header("Search the register")
    st.caption("Type a full or partial name, with typos if you like. Add the region when you know it.")

    def example():
        r = register.sample(1).iloc[0]
        st.session_state.q = core.add_noise(r["name"], random.Random(), force=True)
        st.session_state.region = r["region"]

    c1, c2, c3 = st.columns([3, 1.4, 1])
    q = c1.text_input("Name", key="q", placeholder="e.g. Mrima Zawadi Saccos Limited")
    regions = ["(not given)"] + sorted(r for r in register.region.dropna().unique() if r)
    region_sel = c2.selectbox("Region", regions, key="region")
    top = c3.slider("Results", 3, 15, 5)
    st.button("🎲 Try a random misspelled example", on_click=example)

    if q.strip():
        region = "" if region_sel == "(not given)" else region_sel
        res = engine.search(q, region, top=top)
        show_headline(res)
        results_table(res)
        if res.iloc[0].match_prob < core.BAND_REVIEW:
            if st.button("➕ Add this cooperative to local pending register"):
                append_pending_name(q, region, source="search")
                st.success(f"Saved '{q}' to the local pending register.")
        picks = st.multiselect("Send to review queue", res.coop_id.tolist(),
                               format_func=lambda i: f"{i} · {res.set_index('coop_id').loc[i, 'name']}")
        if st.button("Add selected to review queue", disabled=not picks):
            for i in picks:
                row = res[res.coop_id == i].iloc[0]
                ss_queue().append({"added": datetime.now().strftime("%Y-%m-%d %H:%M"), "query": q, "query_region": region,
                                   "candidate_id": i, "candidate_name": row["name"], "candidate_region": row.region,
                                   "match_prob": round(float(row.match_prob), 3), "band": row.band,
                                   "evidence": row.evidence, "officer_decision": ""})
            st.toast(f"Added {len(picks)} to the review queue")

# ----------------------------------------------------------------------------- Check a proposed name
elif page == "Check a proposed name":
    st.header("Is this proposed name available?")
    st.caption("Screens a new cooperative name against every registered name before registration.")
    name = st.text_input("Proposed name", placeholder="e.g. Tafala Women Cooperative Society")
    region = st.selectbox("Region (optional)", ["(not given)"] + sorted(r for r in register.region.dropna().unique() if r))
    if name.strip():
        res = engine.search(name, "" if region == "(not given)" else region, top=5)
        show_headline(res)
        results_table(res)
        if res.iloc[0].match_prob < core.BAND_REVIEW:
            if st.button("➕ Add this cooperative to local pending register"):
                append_pending_name(name, "" if region == "(not given)" else region, source="proposed_name")
                st.success(f"Saved '{name}' to the local pending register.")
        st.caption("Names that differ only by spelling, spacing, abbreviation or word order are treated as close.")

# ----------------------------------------------------------------------------- Review queue
elif page == "Review queue":
    st.header("Review queue")
    st.caption("An officer decides each pair: **Same**, **Different** or **Needs more information**. "
               "Recording a decision does not change the register.")
    src = st.radio("Show", ["Pairs I added from Search", "Sample scan of the register"], horizontal=True)
    if src.startswith("Pairs"):
        q = pd.DataFrame(ss_queue())
        if q.empty:
            st.info("Nothing queued yet. Add results from the Search page.")
            st.stop()
    else:
        q = read_csv_safe("review_queue_sample.csv")
        if q is None or q.empty:
            st.info("No sample queue available.")
            st.stop()
        q["officer_decision"] = q["officer_decision"].fillna("")
    edited = st.data_editor(
        q, hide_index=True, width="stretch", key=f"ed_{src}",
        disabled=[c for c in q.columns if c != "officer_decision"],
        column_config={"officer_decision": st.column_config.SelectboxColumn("Officer decision", options=DECISIONS),
                       "match_prob": st.column_config.ProgressColumn("Match probability", min_value=0.0, max_value=1.0, format="%.3f")})
    done = (edited.officer_decision != "").sum()
    st.progress(done / len(edited), text=f"{done} of {len(edited)} reviewed")
    c1, c2 = st.columns(2)
    if c1.button("💾 Save decisions to file"):
        path = OUT / "review_decisions.csv"
        stamped = edited.assign(saved_at=datetime.now().isoformat(timespec="seconds"))
        stamped.to_csv(path, mode="a", header=not path.exists(), index=False)
        st.success(f"Saved to {path.name}")
    c2.download_button("⬇️ Download as CSV", edited.to_csv(index=False), "review_queue.csv", "text/csv")

# ----------------------------------------------------------------------------- Dashboard
elif page == "Dashboard":
    st.header("Duplicate risk & data completeness")
    f1, f2 = st.columns(2)
    reg_f = f1.multiselect("Region", sorted(r for r in register.region.dropna().unique() if r))
    type_f = f2.multiselect("Society type", sorted(register.ctype.unique()))
    d = register.copy()
    if reg_f: d = d[d.region.isin(reg_f)]
    if type_f: d = d[d.ctype.isin(type_f)]

    scan = read_csv_safe("review_queue_sample.csv")
    pending = read_csv_safe("pending_new_names.csv")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registered names", f"{len(d):,}")
    k2.metric("Regions", d.region.replace("", pd.NA).nunique())
    k3.metric("Records with region", f"{(d.region.fillna('') != '').mean():.0%}")
    if scan is not None:
        k4.metric("Flagged in sample scan", f"{len(scan)} of 300", help="Names whose closest neighbour is at least in the review band")

    a, b = st.columns(2)
    a.subheader("Cooperatives per region")
    a.bar_chart(d.region.replace("", "(unknown)").value_counts())
    b.subheader("By society type")
    b.bar_chart(d.ctype.value_counts())

    if pending is not None and not pending.empty:
        st.subheader("Pending new cooperatives")
        st.dataframe(pending, hide_index=True, width="stretch")

    if scan is not None and not scan.empty:
        st.subheader("Duplicate risk from the sample scan")
        s = scan.merge(register[["coop_id", "region", "ctype"]], on="coop_id", how="left")
        if reg_f: s = s[s.region.isin(reg_f)]
        if type_f: s = s[s.ctype.isin(type_f)]
        x, y = st.columns(2)
        x.caption("Flagged pairs by confidence band")
        x.bar_chart(s.band.value_counts())
        y.caption("Flagged pairs by region")
        y.bar_chart(s.region.value_counts())
    st.caption("A sample scan checks 300 random names against the rest of the register; re-run it on the Data & retrain page.")

# ----------------------------------------------------------------------------- Model performance
elif page == "Model performance":
    st.header("Model performance")
    st.warning("Scores below come from **simulated** duplicates, so they show the pipeline works. "
               "Validate on officer-labelled real pairs before reporting accuracy.")
    m = read_csv_safe("model_metrics.csv")
    if m is not None:
        m = m.rename(columns={m.columns[0]: "model"}).set_index("model")
        st.subheader("Pair classification (held-out cooperatives)")
        st.dataframe(m, width="stretch")
        cols = st.multiselect("Compare metrics", [c for c in m.columns if c not in ("log_loss", "brier")],
                              default=[c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in m.columns])
        if cols: st.bar_chart(m[cols])
    r = read_csv_safe("retrieval_metrics.csv")
    if r is not None:
        st.subheader("Top-k retrieval for noisy queries")
        r = r.set_index("method")
        st.dataframe(r, width="stretch")
        st.bar_chart(r[["top1", "top3", "top5"]])
    fi = read_csv_safe("feature_importance.csv")
    if fi is not None:
        st.subheader("Which features matter")
        fi.columns = ["feature", "importance"]
        st.bar_chart(fi.set_index("feature")["importance"])
    imgs = [p for p in (OUT / "roc_curves.png", OUT / "roc_pr_curves.png") if p.exists()]
    img = max(imgs, key=lambda p: p.stat().st_mtime) if imgs else None
    if img:
        st.subheader("ROC and precision-recall curves")
        st.image(str(img))

# ----------------------------------------------------------------------------- Data & retrain
elif page == "Data & retrain":
    st.header("Data & retrain")
    st.write("Upload the approved, anonymised register (CSV with a **name** column and optionally **region**) to "
             "retrain on real names. Duplicate labels are still simulated from realistic noise.")
    st.download_button("Download CSV template", "name,region\nUmoja Kahawa AMCOS Ltd,Kagera\n", "register_template.csv")
    up = st.file_uploader("Register CSV", type="csv")
    use_synth = st.checkbox("Or retrain on the built-in synthetic register (from the Ushirika statistics)", value=up is None)
    reg_path = None
    if up is not None:
        df = pd.read_csv(up)
        df.columns = [c.strip().lower() for c in df.columns]
        if "name" not in df.columns:
            st.error("The CSV needs a column called `name`.")
            st.stop()
        df = df.dropna(subset=["name"]).drop_duplicates("name")
        st.write(f"{len(df):,} unique names. Preview:")
        st.dataframe(df.head(), hide_index=True)
        if len(df) > 4000:
            st.info("Registers above 4,000 names are subsampled for *training*; search still covers all names.")
        reg_path = HERE / "uploaded_register.csv"
        df.to_csv(reg_path, index=False)
    if st.button("🚀 Train", type="primary", disabled=(up is None and not use_synth)):
        args = SimpleNamespace(master=None, regional=str(HERE / "data" / "tanzania_ushirika_regional_analysis.csv"),
                               register=str(reg_path) if up is not None else None, scale=0.25)
        buf = io.StringIO()
        with st.spinner("Training three models with cross-validation (1-3 minutes)..."), contextlib.redirect_stdout(buf):
            core.train(args)
        load_engine.clear()
        st.success("Done. The new model is now active.")
        st.code(buf.getvalue()[-3500:])
    st.caption("Training overwrites the saved model and the metrics shown on the Model performance page.")

# ----------------------------------------------------------------------------- About
else:
    st.header("About & responsible use")
    st.markdown("""
**What this does.** Normalises a name, retrieves look-alikes with TF-IDF over character n-grams, then scores each
candidate pair with the best of three classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained on
18 similarity features: edit distance, Jaro-Winkler, token overlap, character n-grams, phonetic key, initials,
region and society type.

**Confidence bands.** ≥ 0.90 *likely duplicate*; 0.50-0.90 *review queue*; below 0.50 *unlikely*.

**Safeguards**
- Similarity is evidence, not proof of identity.
- No automatic delete, merge or approval; an officer records every decision.
- Original and normalised values are kept separate.
- Use only an approved, anonymised extract. Keep personal identifiers out of the register file.
- Test on local naming and transliteration variants before relying on results.

**Data note.** The Ushirika statistics contain no cooperative names, so the built-in register is synthetic.
""")


# ----------------------------------------------------------------------------- app-end: pending register support
pending_file = pending_names_path()
if pending_file.exists():
    pending_df = pd.read_csv(pending_file)
    if not pending_df.empty:
        st.sidebar.caption(f"**Pending additions:** {len(pending_df)}")


# End of file
"""
Cooperative Name Search & Duplicate Detection: Streamlit app
Run:  streamlit run app.py
Engine, features and models are the ones from the notebook (see coop_name_search.py).
Similarity is evidence, not proof of identity. Nothing here merges, deletes or approves records.
"""


# keep the original file content in place; the actual app is above
# this trailing block exists only to avoid accidental file duplication issues in older deployments

"""


# The file above was rewritten to include a local pending-additions feature for names not found in the register.
# This allows an officer to save a candidate new cooperative name without automatically inserting it into the main register.































































































































































































































































"""