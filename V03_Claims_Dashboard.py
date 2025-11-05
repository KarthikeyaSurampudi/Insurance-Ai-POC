# V03_Claims_Dashboard.py — Solidarity Motor Claims Dashboard (fixed & synced to V03 pipeline)
from __future__ import annotations
import os, json, shutil, subprocess, tempfile, sys, base64
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).parent.resolve()
PIPELINE_SCRIPT = "V03_Claims_Pipeline.py"  # pipeline entry

# === Azure OpenAI (kept same call shape/params as pipeline) ================
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY  = os.getenv("AZURE_API_KEY")
AZURE_API_VER  = os.getenv("AZURE_API_VER")
MODEL_NAME     = os.getenv("AZURE_MODEL")

import jwt
import os
from urllib.parse import urlparse, parse_qs

SECRET = os.getenv("APP_SECRET", "supersecret")

def verify_token():
    query = st.experimental_get_query_params()
    token = query.get("token", [None])[0]
    if not token:
        st.error("Unauthorized")
        st.stop()

    try:
        data = jwt.decode(token, SECRET, algorithms=["HS256"])
        st.session_state.user = data["username"]
    except Exception as e:
        print(e)
        st.error("Invalid or expired token")
        st.stop()

verify_token()
st.success(f"Welcome {st.session_state.user}!")


def llm_answer_with_context(question: str, context: str, *, max_completion_tokens: int = 3000) -> str:
    sys_prompt = (
        "You are an insurance claims assistant for Solidarity Bahrain. "
        "Answer using ONLY the supplied context (CSV rows and notes). "
        "If the context does not contain the answer, say so. Be precise. No speculation."
    )
    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Context:\n{context[:15000]}\n\nQuestion: {question}\nAnswer:"}
        ],
        "temperature": 1.0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "text"},
    }
    r = requests.post(
        f"{AZURE_ENDPOINT}/openai/deployments/{MODEL_NAME}/chat/completions?api-version={AZURE_API_VER}",
        headers={"api-key": AZURE_API_KEY},
        json=payload, timeout=90
    )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def llm_image_compare_damage(image_b64: str, notes: str, *, max_completion_tokens: int = 600) -> str:
    prompt = (
        "You are verifying car damage consistency. Using the notes and the image, "
        "respond with ONE sentence: Does the described damage align with what is visible? "
        "Say 'Yes, ...' or 'No, ...' with a brief reason. No speculation."
    )
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{image_b64}", "detail":"high"}},
                {"type":"text", "text": f"Notes:\n{notes[:4000]}"},
                {"type":"text", "text": prompt}
            ]
        }],
        "temperature": 1.0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "text"},
    }
    r = requests.post(
        f"{AZURE_ENDPOINT}/openai/deployments/{MODEL_NAME}/chat/completions?api-version={AZURE_API_VER}",
        headers={"api-key": AZURE_API_KEY},
        json=payload, timeout=90
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ---------------- Loaders ----------------
def _load_csv(kind: str, cid: str) -> pd.DataFrame | None:
    for fn in (f"{cid}_{kind}.csv", f"{kind}_{cid}.csv"):
        fp = BASE_DIR / fn
        if fp.exists():
            try: return pd.read_csv(fp)
            except Exception: return None
    return None

def load_master() -> pd.DataFrame:
    fp = BASE_DIR / "claim_processed_records.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame(columns=["claim_id","cost_tier"])

def load_key_values(cid): return _load_csv("key_values", cid)
def load_checks(cid):     return _load_csv("doc_checklist", cid)
def load_valids(cid):     return _load_csv("validations", cid)
def load_summary(cid) -> str:
    fp = BASE_DIR / f"summary_{cid}.txt";  return fp.read_text() if fp.exists() else ""
def load_acc_summ(cid) -> str:
    fp = BASE_DIR / f"accident_summary_{cid}.txt"
    txt = fp.read_text() if fp.exists() else ""
    return txt if txt.strip() else "_No police-report summary._"

# ------------- Run pipeline on a ZIP -------------
def run_pipeline_on_folder(zip_file):
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td, "claim.zip"); zpath.write_bytes(zip_file.getbuffer())
        shutil.unpack_archive(str(zpath), td)
        res = subprocess.run([sys.executable, PIPELINE_SCRIPT, td],
                             capture_output=True, text=True)
        if res.returncode != 0:
            st.error(f"Pipeline failed (exit {res.returncode})")
            if res.stdout.strip(): st.code(res.stdout)
            if res.stderr.strip(): st.code(res.stderr)
            return
        st.success("Pipeline finished ✔️. Refresh the claim list if needed.")

# === Minimal, safe UI theming (non-functional) =================================
AUGENT_PURPLE = "#3b0a5e"   # deep purple like augent.ai
AUGENT_GOLD   = "#d7a442"   # warm gold accent
BG_DARK       = "#0a0013"   # very dark background tone

def _load_logo_bytes() -> bytes | None:
    """Load Augent logo from local dir (preferred) or the mounted data path."""
    for p in (BASE_DIR / "augentLogo_2.png", Path("assets/augentLogo_2.png")):
        try:
            if Path(p).exists():
                return Path(p).read_bytes()
        except Exception:
            pass
    return None

logo_bytes = _load_logo_bytes()

# Page config (title + favicon if logo available)
st.set_page_config(
    page_title="Augent's AI assisted Intelligent claims processing.",
    page_icon=logo_bytes if logo_bytes else None,
    layout="wide"
)

# Light CSS to mimic augent.ai palette without altering Streamlit mechanics
st.markdown(
    f"""
    <style>
      /* font */
      @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
      html, body, [class*="css"]  {{
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
      }}
      /* top banner */
      .augent-hero {{
        background: linear-gradient(135deg, {BG_DARK} 0%, #160027 100%);
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 12px;
      }}
      .augent-title {{
        color: white; 
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
      }}
      .augent-sub {{
        color: rgba(255,255,255,0.80);
        font-size: 0.95rem;
        margin-top: 6px;
      }}
      /* accent pills / headings */
      .augent-pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #1a0b2a;
        background: linear-gradient(180deg, {AUGENT_GOLD}, #c59030);
        margin-right: 8px;
      }}
      /* tweak Streamlit accent color safely */
      :root {{
        --augent-gold: {AUGENT_GOLD};
        --augent-purple: {AUGENT_PURPLE};
      }}
      div.stButton > button, .stDownloadButton button, .st-emotion-cache-1vt4y43 {{
        border-radius: 8px;
      }}
      div.stButton > button:hover {{
        box-shadow: 0 0 0 3px rgba(215,164,66,0.25);
      }}
      /* sidebar title accent */
      section[data-testid="stSidebar"] .st-emotion-cache-1v0mbdj, 
      section[data-testid="stSidebar"] .stMarkdown h1, 
      section[data-testid="stSidebar"] .stMarkdown h2 {{
        color: {AUGENT_PURPLE};
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# Hero header with logo + title
with st.container():
    cols = st.columns([1, 6])
    with cols[0]:
        if logo_bytes:
            st.image(logo_bytes, caption=None, use_container_width=True)
    with cols[1]:
        st.markdown(
            f"""
            <div class="augent-hero">
              <div class="augent-pill">INSURANCE</div>
              <h1 class="augent-title">Augent's AI assisted Intelligent claims processing.</h1>
              <div class="augent-sub">Fast, accurate, and transparent claim triage and verification.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------- UI (original functionality preserved) -------------
st.sidebar.title("Augent - An Enterprise AI Platform")

with st.sidebar.expander("➕ Ingest new claim"):
    up = st.file_uploader("Upload claim ZIP", type=["zip"])
    if up and st.button("Run pipeline"):
        run_pipeline_on_folder(up)

def _discover_claim_ids():
    ids = set()
    for p in BASE_DIR.glob("*_doc_checklist.csv"): ids.add(p.stem.replace("_doc_checklist",""))
    for p in BASE_DIR.glob("summary_*.txt"):        ids.add(p.stem.replace("summary_",""))
    for p in BASE_DIR.glob("*_validations.csv"):    ids.add(p.stem.replace("_validations",""))
    for p in BASE_DIR.glob("*_assets"):             ids.add(p.stem.replace("_assets",""))
    return sorted(ids)

master = load_master()
if master.empty:
    disc = _discover_claim_ids()
    if not disc:
        st.sidebar.warning("No claims processed yet."); st.stop()
    master = pd.DataFrame({"claim_id": disc, "cost_tier": "Unknown"})

cid = st.sidebar.selectbox("Choose claim", master["claim_id"].astype(str))

TABS = ["Overview","Checklist","Key-Values","Validation","Accident","Claim Summary","Quality / Logs","APIs (JSON)","Chatbot","Image Analysis"]
tabs = st.tabs(TABS)

# Overview
with tabs[0]:
    st.header("📊 Portfolio Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Claims", len(master))

    # Try a few known numeric columns for an average
    series = None
    for c in ["estimate_total_aed","estimate_net_aed","amount_aed","net_total_aed"]:
        if c in master.columns:
            s = pd.to_numeric(master[c], errors="coerce")
            if s.notna().any(): series = s; break
    c2.metric("Avg Estimate (AED)", f"{series.mean():,.0f}" if series is not None else "–")

    split = master["cost_tier"].value_counts()
    c3.metric("Minor/Moderate/Total", "/".join(str(split.get(k,0)) for k in ["Minor","Moderate","Total-loss"]))
    st.bar_chart(split, use_container_width=True)
    st.dataframe(master, use_container_width=True)

# Checklist
with tabs[1]:
    st.header("📂 Required-documents checklist")
    chk_obj = load_checks(cid)
    if chk_obj is None or chk_obj.empty:
        st.info("Checklist CSV not found.")
    else:
        chk = chk_obj.copy()
        st.dataframe(chk, use_container_width=True)
        if {"Category","Present"}.issubset(chk.columns):
            miss = chk[~chk["Present"]]
            if not miss.empty: st.error("Missing ➜ " + ", ".join(miss["Category"].astype(str)))
            else: st.success("All mandatory docs present ✔️")

# Key-Values
with tabs[2]:
    st.header("🗂️ Key-values extracted")
    kv_obj = load_key_values(cid)
    if kv_obj is None or kv_obj.empty:
        st.info("No key/value table.")
    else:
        kv = kv_obj.copy()
        edited = st.data_editor(kv, hide_index=True, use_container_width=True)
        if st.button("💾 Save edits", key=f"save_{cid}"):
            pd.DataFrame(edited).to_csv(BASE_DIR / f"{cid}_key_values.csv", index=False)
            st.toast("Saved", icon="✅")

# Validation
with tabs[3]:
    st.header("🔍 Cross-doc validations")
    RULE_INFO = {
        "LICENCE_VALID_ON_ACC": "Licence must have been valid on the accident date.",
        "OWNER_NAME_MATCH": "Owner/holder/customer names should align across licence/registration/report.",
        "PLATE_IN_POLICE_MATCH_REG": "Plate in police report should match the vehicle registration.",
        "LICENCE_ON_REPORT_MATCH": "Licence number in police report equals the uploaded licence.",
        "OUR_PARTY_NOT_AT_FAULT": "At-fault plate on the report should NOT be the insured vehicle.",
        "VIN_PRESENT_ON_REG": "VIN/chassis number present on the registration (back).",
    }
    v_obj = load_valids(cid)
    if v_obj is None or v_obj.empty:
        st.info("No validation table found.")
    else:
        v = v_obj.copy()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("PASS", int((v["status"]=="PASS").sum()))
        c2.metric("WARN", int((v["status"]=="WARN").sum()))
        c3.metric("FAIL", int((v["status"]=="FAIL").sum()))
        c4.metric("MISSING", int((v["status"]=="MISSING").sum()))

        def _row_color(r):
            col = {"PASS":"#d9ead3","WARN":"#ffe599","FAIL":"#ffcccc"}.get(r["status"], "")
            return [f"background-color:{col}"]*len(r)
        st.dataframe(v.style.apply(_row_color, axis=1), use_container_width=True, hide_index=True)

        with st.expander("ℹ️ Rule details"):
            ch = st.selectbox("Select rule", v["rule"].unique())
            st.markdown(RULE_INFO.get(ch, "_No info._"))

# Accident
with tabs[4]:
    st.header("🚓 Police-report summary")
    st.markdown(load_acc_summ(cid))

# Claim Summary
with tabs[5]:
    st.header("📝 Claim Summary")
    chk_obj = load_checks(cid)
    vtab_obj = load_valids(cid)
    kv_obj = load_key_values(cid)

    chk = chk_obj if chk_obj is not None else pd.DataFrame(columns=["Category","Present","SourceDocs"])
    vtab = vtab_obj if vtab_obj is not None else pd.DataFrame(columns=["rule","status","details"])
    kv = kv_obj if kv_obj is not None else pd.DataFrame(columns=["Category","SourceDoc","Key","Value"])
    acc_sum = load_acc_summ(cid)

    def _badge(ok): return "✅" if ok else "⚠️"
    def is_present(prefix: str) -> bool:
        if chk.empty or "Category" not in chk or "Present" not in chk: return False
        mask = chk["Category"].astype(str).str.startswith(prefix)
        return bool(chk.loc[mask, "Present"].any())

    lic = is_present("licence.")
    reg = is_present("registration.")
    pid = is_present("id.")
    pol = False
    if not chk.empty and "Category" in chk and "Present" in chk:
        sub = chk[chk["Category"]=="police_report"]; pol = bool(not sub.empty and sub["Present"].any())

    def _get(k, cat=None):
        if kv.empty or "Key" not in kv or "Value" not in kv: return ""
        df = kv
        if cat and "Category" in kv: df = df[df["Category"]==cat]
        row = df[df["Key"].astype(str).str.lower()==k.lower()].head(1)
        return "" if row.empty else str(row["Value"].iloc[0])

    plate = _get("plate") or _get("plate_no") or _get("plate_number") or ""
    vin   = _get("vin") or _get("chassis_no") or _get("chassis_number") or ""
    make  = _get("make","registration")
    model = _get("model","registration")
    year  = _get("yom","registration") or _get("year","registration")
    holder= _get("holder_name","licence") or _get("name","licence")
    licno = _get("licence_number","licence") or _get("license_number","licence")
    issue = _get("first_issue_date","licence") or _get("issue_date_licence","licence")
    exp   = _get("expiry_date","licence") or _get("licence_expiry","licence")

    md = []
    md.append(f"**Claim ID:** `{cid}`")
    md.append("")
    md.append(f"{_badge(lic)} **Licence:** {'present' if lic else 'missing'}  •  {_badge(reg)} **Registration:** {'present' if reg else 'missing'}  •  {_badge(pid)} **ID:** {'present' if pid else 'missing'}  •  {_badge(pol)} **Police report:** {'present' if pol else 'missing'}")
    md.append("")
    md.append("**Vehicle**")
    veh_bits = []
    if make or model: veh_bits.append(f"{make} {model}".strip())
    if year: veh_bits.append(str(year))
    if plate: veh_bits.append(f"Plate `{plate}`")
    if vin: veh_bits.append(f"VIN `{vin}`")
    md.append("- " + (" • ".join(veh_bits) if veh_bits else "_No vehicle details extracted._"))
    md.append("")
    md.append("**Driver/Licence**")
    lic_bits = []
    if holder: lic_bits.append(holder)
    if licno:  lic_bits.append(f"Licence `{licno}`")
    if issue:  lic_bits.append(f"Issued {issue}")
    if exp:    lic_bits.append(f"Expires {exp}")
    md.append("- " + (" • ".join(lic_bits) if lic_bits else "_No licence details extracted._"))
    if pol:
        md.append("")
        md.append("**Accident (police)**")
        md.append("- " + (acc_sum.splitlines()[0] if acc_sum else "_No summary available_"))
    st.markdown("\n".join(md))

    with st.expander("Generate detailed LLM summary (max ~4000 tokens)"):
        if st.button("🧠 Produce claim narrative"):
            rows = []
            for _, r in chk.iterrows():
                rows.append(f"CHECK\t{r.get('Category','')}\t{r.get('Present','')}\t{r.get('SourceDocs','')}")
            for _, r in kv.iterrows():
                rows.append(f"KV\t{r.get('Category','')}\t{r.get('Key','')}\t{r.get('Value','')}\t{r.get('SourceDoc','')}")
            for _, r in vtab.iterrows():
                rows.append(f"VAL\t{r.get('rule','')}\t{r.get('status','')}\t{r.get('details','')}")
            if acc_sum: rows.append(f"POLICE\t{acc_sum}")
            ctx = "\n".join(rows)[:18000]
            try:
                narrative = llm_answer_with_context(
                    "Write a clear, business-friendly claim summary for internal handlers. Use ONLY the context; avoid speculation.",
                    ctx, max_completion_tokens=4000)
                st.success("Summary generated"); st.write(narrative)
            except Exception as e:
                st.error(f"LLM call failed: {e}")

# Logs
with tabs[6]:
    st.header("📈 Logs")
    logp = BASE_DIR / "insurance_ai.log"
    if logp.exists(): st.code("\n".join(logp.read_text().splitlines()[-200:]), language="log")
    else: st.info("No log file found.")

# APIs (JSON)
with tabs[7]:
    st.header("🧪 API responses (simulated)")
    def api_checklist_json():
        d = load_checks(cid)
        items = []
        if d is not None and not d.empty:
            for _, row in d.iterrows():
                cat = row.get("Category","")
                present = bool(row.get("Present", False))
                src = row.get("SourceDocs", "")
                src_str = "" if pd.isna(src) else str(src)
                source_docs = [s for s in src_str.split(";") if s]
                if isinstance(cat, str) and "." in cat:
                    doc, side = cat.split(".",1)
                else:
                    doc, side = str(cat), None
                items.append({"document": doc, "side": side, "present": present, "sourceDocs": source_docs})
        return {"claim_id": cid, "documents": items}

    def api_keyvalues_json():
        kv = load_key_values(cid)
        rows = []
        if kv is not None and not kv.empty:
            for cat, src, key, val in kv[["Category","SourceDoc","Key","Value"]].itertuples(index=False, name=None):
                rows.append({"category":cat, "source":src, "key":key, "value":val})
        return {"claim_id": cid, "key_values": rows}

    def api_validations_json():
        v = load_valids(cid)
        rows = []
        if v is not None and not v.empty:
            for rule, status, details in v[["rule","status","details"]].itertuples(index=False, name=None):
                rows.append({"rule":rule, "status":status, "details":details})
        return {"claim_id": cid, "validations": rows}

    def api_accident_json():
        return {"claim_id": cid, "accident_summary": load_acc_summ(cid)}

    c1,c2,c3,c4 = st.columns(4)
    if c1.button("Checklist JSON"): st.json(api_checklist_json())
    if c2.button("Key-Values JSON"): st.json(api_keyvalues_json())
    if c3.button("Validations JSON"): st.json(api_validations_json())
    if c4.button("Accident JSON"): st.json(api_accident_json())

# Chatbot
with tabs[8]:
    st.header("💬 Chatbot (answers from current claim artifacts)")
    _checks = load_checks(cid); checks = _checks if _checks is not None else pd.DataFrame()
    _keys   = load_key_values(cid); keys = _keys if _keys is not None else pd.DataFrame()
    _vals   = load_valids(cid); vals = _vals if _vals is not None else pd.DataFrame()
    summ = load_acc_summ(cid) or ""

    corpus = []
    if not checks.empty:
        for _, r in checks.iterrows():
            corpus.append(("checklist", f"{r.get('Category','')}: present={r.get('Present','')}, src={r.get('SourceDocs','')}"))
    if not keys.empty:
        for _, r in keys.iterrows():
            corpus.append(("key-values", f"[{r.get('Category','')}] {r.get('Key','')} = {r.get('Value','')} (src={r.get('SourceDoc','')})"))
    if not vals.empty:
        for _, r in vals.iterrows():
            corpus.append(("validation", f"{r.get('rule','')}: {r.get('status','')} — {r.get('details','')}"))
    if summ: corpus.append(("accident", summ))

    q = st.text_input("Ask about this claim…", "")
    if st.button("Ask") and q.strip():
        ql = q.lower().split()
        scored = []
        for src, txt in corpus:
            score = sum(1 for w in set(ql) if w and w in txt.lower())
            if score: scored.append((score, src, txt))
        scored.sort(reverse=True)
        if scored:
            st.subheader("Most relevant snippets")
            for s, src, txt in scored[:8]:
                st.markdown(f"- **{src}** (score {s}): {txt}")

        ctx_rows = []
        for _, r in (checks if not checks.empty else pd.DataFrame(columns=["Category","Present","SourceDocs"])).iterrows():
            ctx_rows.append(f"CHECK\t{r.get('Category','')}\t{r.get('Present','')}\t{r.get('SourceDocs','')}")
        for _, r in (keys if not keys.empty else pd.DataFrame(columns=["Category","SourceDoc","Key","Value"])).iterrows():
            ctx_rows.append(f"KV\t{r.get('Category','')}\t{r.get('Key','')}\t{r.get('Value','')}")
        for _, r in (vals if not vals.empty else pd.DataFrame(columns=["rule","status","details"])).iterrows():
            ctx_rows.append(f"VAL\t{r.get('rule','')}\t{r.get('status','')}\t{r.get('details','')}")
        if summ: ctx_rows.append(f"POLICE\t{summ}")

        try:
            answer = llm_answer_with_context(q.strip(), "\n".join(ctx_rows), max_completion_tokens=3000)
            st.subheader("Answer"); st.write(answer)
        except Exception as e:
            st.error(f"LLM call failed: {e}")

# Image Analysis
with tabs[9]:
    st.header("🖼️ Image Analysis — damage mentions vs. photo evidence")
    assets = BASE_DIR / f"{cid}_assets"
    tmp_kv = load_key_values(cid)
    kv = tmp_kv if tmp_kv is not None else pd.DataFrame(columns=["Category","SourceDoc","Key","Value"])
    if not assets.exists():
        st.info("No vehicle-damage assets found for this claim.")
    else:
        imgs = sorted([p for p in assets.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
        if not imgs:
            st.info("No vehicle-damage photos were exported.")
        else:
            cand = kv[kv["Key"].str.lower().str.contains("damage|dent|bumper|fender|head|tail|scratch|rear|front", regex=True, na=False)]
            notes = "\n".join(f"{r['Key']}: {r['Value']}" for _, r in cand.iterrows())[:4000] or "No explicit damage descriptions captured."
            chosen = st.selectbox("Choose damage photo", [p.name for p in imgs])
            c1, c2 = st.columns([1,1])
            with c1:
                sel = next(p for p in imgs if p.name == chosen)
                st.image(str(sel), caption=sel.name, use_container_width=True)
            with c2:
                if st.button("Analyse image"):
                    try:
                        b64 = base64.b64encode(sel.read_bytes()).decode()
                        ans = llm_image_compare_damage(b64, notes)
                        st.success(ans)
                    except Exception as e:
                        st.error(f"Vision call failed: {e}")

            st.divider()
            if st.button("Perform analysis on all photos"):
                out = []
                for p in imgs:
                    try:
                        b64 = base64.b64encode(p.read_bytes()).decode()
                        out.append((p.name, llm_image_compare_damage(b64, notes)))
                    except Exception as e:
                        out.append((p.name, f"ERROR: {e}"))
                st.write(pd.DataFrame(out, columns=["photo","assessment"]))
