#!/usr/bin/env python3
"""
Redrob MatchWise — Next-Gen Multi-Stage Ranking Dashboard
=========================================================
A premium Streamlit application that showcases the new 4-stage hybrid
ranking architecture (LambdaMART + Cross-Encoder).

Run locally:
    streamlit run app.py
"""

import csv
import io
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import offline feature engineering & taxonomy
from build_features import extract_features, FEATURE_NAMES, NUM_FEATURES, _ensure_embedder
from src.rank import (
    ATD_TAXONOMY,
    CORE_IR_SKILLS,
    compute_atd,
    compute_hea,
    generate_reasoning,
    load_honeypots,
)

# Import pipeline constants & module (module needed for mutable tuning knobs)
import rank_v2
from rank_v2 import (
    IDX_TIMELINE, IDX_INFLATION, IDX_SERVICE, IDX_GHOST, IDX_MAX_ATD,
    IDX_SINGULARITY, IDX_CORE_IR, IDX_DEPTH,
    apply_hard_filters,
    normalize_scores,
    load_lightgbm_model,
    ensemble_model_scores,
)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          PAGE CONFIG                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="MatchWise | Omni-Context Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Hide Default Streamlit Chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Sidebar Branding ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin: 0.75rem 0;
    }

    /* ── Sidebar Controls ── */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stToggle label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        padding-top: 0.25rem;
    }

    /* ── KPI Metric Cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(30,41,59,0.4));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56,189,248,0.4);
    }
    [data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    /* ── Candidate Card Container ── */
    .candidate-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.8) 100%);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .candidate-card:hover {
        border-color: rgba(56,189,248,0.35);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .candidate-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px;
        height: 100%;
        border-radius: 4px 0 0 4px;
    }
    .rank-gold::before   { background: linear-gradient(180deg, #f59e0b, #d97706); }
    .rank-silver::before { background: linear-gradient(180deg, #94a3b8, #64748b); }
    .rank-bronze::before { background: linear-gradient(180deg, #fb923c, #ea580c); }
    .rank-default::before { background: linear-gradient(180deg, #334155, #1e293b); }

    /* ── Rank Badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px; height: 42px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1rem;
        color: #fff;
        flex-shrink: 0;
    }
    .rank-badge-gold   { background: linear-gradient(135deg, #f59e0b, #d97706); box-shadow: 0 2px 12px rgba(245,158,11,0.35); }
    .rank-badge-silver { background: linear-gradient(135deg, #94a3b8, #475569); box-shadow: 0 2px 12px rgba(148,163,184,0.25); }
    .rank-badge-bronze { background: linear-gradient(135deg, #fb923c, #c2410c); box-shadow: 0 2px 12px rgba(251,146,60,0.25); }
    .rank-badge-default { background: linear-gradient(135deg, #334155, #1e293b); }

    /* ── Score Bar ── */
    .score-bar-container {
        background: rgba(30,41,59,0.5);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .score-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    .score-bar-gold   { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .score-bar-silver { background: linear-gradient(90deg, #94a3b8, #cbd5e1); }
    .score-bar-bronze { background: linear-gradient(90deg, #fb923c, #fdba74); }
    .score-bar-default { background: linear-gradient(90deg, #475569, #64748b); }

    /* ── Skill Pills ── */
    .skill-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 3px;
        letter-spacing: 0.01em;
    }
    .skill-core  { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }
    .skill-sota  { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }
    .skill-other { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.2); }

    /* ── Signal Tags ── */
    .signal-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px 0;
    }
    .signal-yes { background: rgba(34,197,94,0.12); color: #4ade80; }
    .signal-no  { background: rgba(239,68,68,0.1);  color: #f87171; }

    /* ── Score Badge ── */
    .score-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .score-badge-gold   { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(217,119,6,0.1)); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .score-badge-silver { background: linear-gradient(135deg, rgba(148,163,184,0.15), rgba(71,85,105,0.1)); color: #e2e8f0; border: 1px solid rgba(148,163,184,0.25); }
    .score-badge-bronze { background: linear-gradient(135deg, rgba(251,146,60,0.15), rgba(194,65,12,0.1)); color: #fdba74; border: 1px solid rgba(251,146,60,0.25); }
    .score-badge-default { background: linear-gradient(135deg, rgba(71,85,105,0.2), rgba(30,41,59,0.2)); color: #94a3b8; border: 1px solid rgba(71,85,105,0.3); }

    /* ── Hidden Gem Badge ── */
    .gem-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(217,119,6,0.1));
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(245,158,11,0.3);
        letter-spacing: 0.02em;
    }

    /* ── Pipeline Mode Tag ── */
    .mode-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .mode-ml     { background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.1)); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.25); }
    .mode-heur   { background: linear-gradient(135deg, rgba(251,146,60,0.12), rgba(194,65,12,0.08)); color: #fdba74; border: 1px solid rgba(251,146,60,0.2); }

    /* ── Section Headings ── */
    .section-heading {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(56,189,248,0.2);
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* ── Career History ── */
    .career-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.08);
    }
    .career-item:last-child { border-bottom: none; }

    /* ── Tab Styling ── */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        color: #0f172a;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 20px rgba(56,189,248,0.35);
        transform: translateY(-1px);
    }

    /* ── Feedback Buttons ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.82rem;
        transition: all 0.2s;
    }

    /* ── Expander ── */
    details[data-testid="stExpander"] {
        border: 1px solid rgba(148,163,184,0.1) !important;
        border-radius: 10px !important;
        background: rgba(30,41,59,0.3);
    }
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(PROJECT_DIR, "feedback_logs.jsonl")


def retrain_with_feedback(feature_matrix, candidates):
    """Retrain ranker artifacts using current feedback logs and uploaded dataset."""
    # Save current features and candidates to temporary files so the trainer uses the uploaded data
    temp_features = os.path.join(PROJECT_DIR, "temp_features.npz")
    feature_names = np.array(FEATURE_NAMES, dtype=object)
    candidate_id_array = np.array([c.get("candidate_id", "") for c in candidates], dtype=object)
    np.savez_compressed(temp_features, features=feature_matrix, feature_names=feature_names, candidate_ids=candidate_id_array)
    
    temp_cands = os.path.join(PROJECT_DIR, "temp_candidates.jsonl")
    with open(temp_cands, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    # Use the venv python (streamlit's own executable = the venv python we launched with)
    python_exe = sys.executable
    # If streamlit is installed in a venv, sys.executable already points there
    cmd = [
        python_exe,
        "train_ranker.py",
        "--features",
        temp_features,
        "--candidates",
        temp_cands,
        "--model-out",
        "artifacts/ranker.xgb",
        "--lgb-model-out",
        "artifacts/ranker.lgb",
        "--feedback",
        "feedback_logs.jsonl",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        if os.path.exists(temp_features):
            os.remove(temp_features)
        if os.path.exists(temp_cands):
            os.remove(temp_cands)
            
    return result


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           SIDEBAR                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("### ⚡ MatchWise")

    def handle_upload():
        st.session_state.pop("fused", None)

    # ── Dataset ──
    dataset_source = st.selectbox(
        "Dataset",
        ["Default Sample (50 profiles)", "Full Dataset (100k profiles)", "Upload Custom File"],
        index=0,
        on_change=handle_upload,
        help="Choose between the sample subset, the pre-installed full set, or upload your own JSONL."
    )
    uploaded = None
    if dataset_source == "Upload Custom File":
        uploaded = st.file_uploader(
            "Upload candidate profiles (.json, .jsonl, .gz)",
            type=["json", "jsonl", "gz"],
            on_change=handle_upload
        )

    st.markdown("---")

    # ── Pipeline ──
    with st.expander("**Pipeline**", expanded=True):
        use_crossencoder = st.toggle("FlashRank Reranking", value=True,
            help="Semantic reranking of top candidates. Depth adjustable below.")
        top_k_xgb = st.number_input("XGBoost Top-K", min_value=10, max_value=500, value=200, step=10,
            help="How many candidates the ML model scores before reranking.")

    # ── Preset Quick-Select ──
    PRESET_MAP = {
        "Default (Balanced)":       (0.40, 0.20, 0.40),
        "ML Heavy":                 (0.60, 0.25, 0.15),
        "Heuristic Heavy":          (0.20, 0.10, 0.70),
        "Semantic Focus":           (0.35, 0.50, 0.15),
        "Balanced ML+":             (0.45, 0.30, 0.25),
    }
    preset_names = list(PRESET_MAP.keys())

    def _apply_preset():
        name = st.session_state.preset_selector
        if name in PRESET_MAP:
            w_xgb, w_ce, w_heur = PRESET_MAP[name]
            st.session_state.sl_xgb = w_xgb
            st.session_state.sl_ce = w_ce
            st.session_state.sl_heur = w_heur

    # ── Score Weights ──
    with st.expander("**Score Weights**", expanded=True):
        st.selectbox(
            "Quick Preset",
            preset_names,
            index=0,
            key="preset_selector",
            on_change=_apply_preset,
            help="Pre-tuned weight configurations from A/B experiment. Select, then fine-tune below."
        )
        st.caption("Or drag sliders individually — weights auto-normalize.")
        w1 = st.slider("ML Model", 0.0, 1.0, rank_v2.W_XGB, 0.05, key="sl_xgb",
            help="XGBoost LambdaMART influence. Higher = trusts learned patterns more.")
        w2 = st.slider("Semantic Match", 0.0, 1.0, rank_v2.W_CE, 0.05, key="sl_ce",
            help="Cross-encoder influence. Higher = better skill-meaning matching.")
        w3 = st.slider("Heuristic Rules", 0.0, 1.0, rank_v2.W_HEURISTIC, 0.05, key="sl_heur",
            help="Hand-crafted rules influence. Higher = stricter ATD-based ranking.")
        total = w1 + w2 + w3
        if total > 0:
            rank_v2.W_XGB = w1 / total
            rank_v2.W_CE = w2 / total
            rank_v2.W_HEURISTIC = w3 / total
        st.caption(
            f"**{rank_v2.W_XGB:.0%}** ML · "
            f"**{rank_v2.W_CE:.0%}** Semantic · "
            f"**{rank_v2.W_HEURISTIC:.0%}** Heuristic"
        )

    # ── Advanced ──
    with st.expander("**Advanced**"):
        filt_val = st.slider("Filter Aggressiveness", 0.1, 0.9, rank_v2.FILTER_THRESHOLD, 0.05, key="sl_filt",
            help="Skill-inflation cutoff. 0.5 = 5+ inflated skills filtered.")
        rank_v2.FILTER_THRESHOLD = filt_val
        rerank_val = st.slider("Rerank Count", 10, 200, rank_v2.RERANK_DEPTH, 10, key="sl_rerank",
            help="Candidates getting semantic reranking. More = better quality, slower.")
        rank_v2.RERANK_DEPTH = rerank_val
        if st.button("↺ Reset All", use_container_width=True, key="btn_reset"):
            rank_v2.W_XGB = 0.40
            rank_v2.W_CE = 0.20
            rank_v2.W_HEURISTIC = 0.40
            rank_v2.FILTER_THRESHOLD = 0.5
            rank_v2.RERANK_DEPTH = 50
            st.session_state.preset_selector = "Default (Balanced)"
            st.session_state.sl_xgb = 0.40
            st.session_state.sl_ce = 0.20
            st.session_state.sl_heur = 0.40
            st.rerun()

    st.markdown("---")

    # ── Evaluation ──
    with st.expander("**Evaluation**"):
        blind_ab_mode = st.toggle("Blind A/B Mode", value=False,
            help="Interleave Heuristic vs ML rankings blindly for recruiter feedback.")
        st.caption("Vote 👍👎 on candidates in the Search tab to build benchmarks.")

    st.markdown("---")

    # ── Actions ──
    run_btn = st.button("▶ Rerun Pipeline", type="primary", use_container_width=True, key="btn_run")
    if run_btn:
        st.session_state["run_pipeline"] = True

    retrain_btn = st.button("🔄 Retrain with Feedback", use_container_width=True, key="btn_retrain")
    if retrain_btn:
        if uploaded is None:
            st.error("Please upload a dataset before retraining.")
        else:
            with st.spinner("Retraining rankers with recruiter feedback..."):
                try:
                    raw_bytes = uploaded.getvalue()
                    candidates, candidate_ids, feature_matrix = parse_and_extract(raw_bytes)
                    result = retrain_with_feedback(feature_matrix, candidates)
                except subprocess.TimeoutExpired:
                    st.error("Retraining timed out after 15 minutes.")
                else:
                    if result.returncode == 0:
                        st.success("Retraining complete. Re-ranking with the new model.")
                        st.session_state.pop("fused", None)
                        st.session_state["run_pipeline"] = True
                        st.rerun()
                    else:
                        st.error("Retraining failed. Check the training output below.")
                        st.code((result.stderr or result.stdout)[-4000:])

    st.markdown("---")
    st.caption("Built for **Redrob Intelligent Candidate Discovery**")

# Initialize Feedback Log
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

def log_feedback(cid, rank, score, model_source, feedback_val):
    if cid in st.session_state.feedback_submitted:
        return
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "candidate_id": cid,
            "rank_position": rank,
            "score": score,
            "model_source": model_source,
            "feedback": feedback_val
        }) + "\n")
    st.session_state.feedback_submitted.add(cid)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          MAIN APP                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

st.markdown("## MatchWise: Omni-Context Ranking")
st.markdown("##### Senior AI Engineer (Founding Team)")

@st.cache_resource(show_spinner=False)
def load_flashrank():
    try:
        from flashrank import Ranker
        return Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./flashrank_cache")
    except Exception as e:
        st.warning(f"Failed to load FlashRank: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_sample_dataset(file_path):
    import json
    candidates = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
    # Extract features
    feature_matrix = np.zeros((len(candidates), NUM_FEATURES), dtype=np.float32)
    candidate_ids = []
    for i, cand in enumerate(candidates):
        cid = cand.get("candidate_id", "")
        candidate_ids.append(cid)
        feature_matrix[i] = extract_features(cand)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    return candidates, candidate_ids, feature_matrix

@st.cache_resource(show_spinner=False)
def load_precomputed_dataset(features_path, candidates_path):
    import gzip
    import json
    data = np.load(features_path, allow_pickle=True)
    feature_matrix = data["features"]
    candidate_ids = list(data["candidate_ids"])
    
    candidates = []
    opener = gzip.open if candidates_path.endswith(".gz") else open
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
    return candidates, candidate_ids, feature_matrix

@st.cache_resource(show_spinner=False, max_entries=1)
def parse_and_extract(raw_bytes):
    """Caches the parsing and feature extraction so it only runs once per file."""
    if raw_bytes.startswith(b"\x1f\x8b"):
        import gzip
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception as e:
            st.error(f"Failed to decompress gzipped upload: {e}")
            return [], [], np.array([])
            
    candidates = []
    try:
        parsed = json.loads(raw_bytes)
        if isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = [parsed]
    except Exception:
        import io
        stream = io.BytesIO(raw_bytes)
        for line in stream:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line.decode("utf-8")))
                except Exception:
                    pass  # Skip the malformed line to prevent crashing the entire stream
                
    if not candidates:
        return [], [], np.array([])
        
    feature_matrix = np.zeros((len(candidates), NUM_FEATURES), dtype=np.float32)
    candidate_ids = [cand.get("candidate_id", "") for cand in candidates]
    
    # Run feature extraction in parallel for large datasets
    if len(candidates) > 500:
        import os
        from concurrent.futures import ProcessPoolExecutor
        num_workers = min(os.cpu_count() or 2, 8)
        try:
            chunk = max(100, len(candidates) // (num_workers * 4))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(extract_features, candidates, chunksize=chunk))
            for i, feats in enumerate(results):
                feature_matrix[i] = feats
        except Exception:
            # Fallback to sequential
            for i, cand in enumerate(candidates):
                feature_matrix[i] = extract_features(cand)
    else:
        for i, cand in enumerate(candidates):
            feature_matrix[i] = extract_features(cand)
            
    # Batch encode semantic similarity (feature index 44) if SentenceTransformer is available
    emb, jd = _ensure_embedder()
    if emb is not None and jd is not None:
        semantic_texts = [cand.get("_semantic_text", "") for cand in candidates]
        try:
            all_embs = emb.encode(semantic_texts, batch_size=256, show_progress_bar=False)
            norms = np.linalg.norm(all_embs, axis=1)
            norms[norms == 0] = 1e-9
            sims = np.dot(all_embs, jd) / (norms * np.linalg.norm(jd))
            feature_matrix[:, 44] = sims
        except Exception:
            pass

    # Replace NaN/Inf hazards
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    return candidates, candidate_ids, feature_matrix

# ── STAGE 0: Load Candidates and Features ──
candidates = []
candidate_ids = []
feature_matrix = np.array([])

if dataset_source == "Default Sample (50 profiles)":
    with st.spinner("Stage 0: Loading default sample dataset (50 profiles)..."):
        sample_path = os.path.join(PROJECT_DIR, "data", "candidates.jsonl")
        candidates, candidate_ids, feature_matrix = load_sample_dataset(sample_path)

elif dataset_source == "Full Dataset (100k profiles)":
    with st.spinner("Stage 0: Loading precomputed full dataset (100k profiles - ~5 seconds)..."):
        features_path = os.path.join(PROJECT_DIR, "artifacts", "precomputed_features.npz")
        candidates_path = os.path.join(PROJECT_DIR, "data", "candidates_backup.jsonl.gz")
        candidates, candidate_ids, feature_matrix = load_precomputed_dataset(features_path, candidates_path)

elif dataset_source == "Upload Custom File" and uploaded is not None:
    try:
        with st.spinner("Stage 0: Extracting offline features from uploaded file..."):
            raw_bytes = uploaded.getvalue()
            candidates, candidate_ids, feature_matrix = parse_and_extract(raw_bytes)
    except Exception as e:
        import traceback
        st.error(f"❌ Error processing uploaded file: {e}")
        st.code(traceback.format_exc(), language="python")
        st.stop()

if not candidates:
    # Empty State Premium Layout
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(textwrap.dedent("""\
        <div style="background-color: rgba(30, 41, 59, 0.5); padding: 2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); text-align: center;">
            <h2 style="margin-bottom: 0;">Welcome to MatchWise v2</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 10px;">
                The Omni-Context Ranking Engine for technical talent.
            </p>
            <br>
            <div style="text-align: left; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                <h4>🚀 Getting Started</h4>
                <p>Select a dataset or upload a candidate dataset in the sidebar to benchmark the engine.</p>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
    st.stop()

if "fused" not in st.session_state or st.session_state.get("run_pipeline", False):
    t_start = time.time()

    # Load Models (graceful fallback if missing)
    with st.spinner("Stage 0: Loading ML Models..."):
        honeypot_path = os.path.join(PROJECT_DIR, "artifacts", "honeypots.json")
        honeypot_ids = load_honeypots(honeypot_path)
        
        model_path = os.path.join(PROJECT_DIR, "artifacts", "ranker.xgb")
        lgb_path = os.path.join(PROJECT_DIR, "artifacts", "ranker.lgb")
        xgb_model = None
        lgb_model = None
        
        if os.path.exists(model_path) and os.path.exists(lgb_path):
            import xgboost as xgb
            xgb_model = xgb.XGBRanker()
            xgb_model.load_model(model_path)
            lgb_model = load_lightgbm_model(lgb_path)
            st.info("✅ ML models loaded")
        else:
            st.warning("⚠️ ML models (ranker.xgb/lgb) not found — using heuristic-only scoring")
            
    # ── STAGE 1: Hard Filters ──
    with st.spinner("Stage 1: Applying Hard Filters..."):
        viable_mask = apply_hard_filters(feature_matrix, candidate_ids, honeypot_ids)
        viable_indices = np.where(viable_mask)[0]
        viable_features = feature_matrix[viable_mask]
        num_viable = len(viable_indices)

    # ── STAGE 2: Score candidates ──
    with st.spinner("Stage 2: Scoring Candidates..."):
        if num_viable > 0:
            if xgb_model is not None:
                xgb_raw_scores = xgb_model.predict(viable_features)
                lgb_raw_scores = lgb_model.predict(viable_features) if lgb_model is not None else None
                ranker_scores = ensemble_model_scores(xgb_raw_scores, lgb_raw_scores)
            else:
                # Heuristic-only fallback: use singularity score (ATD^1.5 × HEA)
                ranker_scores = viable_features[:, IDX_SINGULARITY]
            top_k_positions = np.argsort(ranker_scores)[::-1][:top_k_xgb]
            top_indices = viable_indices[top_k_positions]
            top_xgb_scores = ranker_scores[top_k_positions]
        else:
            top_indices = np.array([])
            top_xgb_scores = np.array([])

    # ── STAGE 3: FlashRank Reranking ──
    ce_scores = {}
    if use_crossencoder and len(top_indices) > 0:
        with st.spinner(f"Stage 3: FlashRank Semantic Reranking (Top {rank_v2.RERANK_DEPTH})..."):
            try:
                from flashrank import RerankRequest
                ranker = load_flashrank()
                if ranker is not None:
                    # Build passages
                    passages = []
                    rerank_k = min(len(top_indices), rank_v2.RERANK_DEPTH)
                    for idx in top_indices[:rerank_k]:
                        cid = candidate_ids[idx]
                        cand = candidates[idx] # Assuming same order
                        prof = cand.get("profile", {})
                        skills = cand.get("skills", [])
                        career = cand.get("career_history", [])
                        
                        text_parts = [
                            prof.get("headline", ""), prof.get("summary", ""),
                            f"Current: {prof.get('current_title', '')} at {prof.get('current_company', '')}",
                            f"Experience: {prof.get('years_of_experience', 0)} years",
                            f"Skills: {', '.join(s.get('name', '') for s in skills[:15])}",
                        ]
                        if career:
                            text_parts.append(f"Recent work: {career[0].get('description', '')[:200]}")
                        
                        text = " ".join(text_parts)[:512]
                        passages.append({"id": str(idx), "text": text, "meta": {"cid": cid}})
                    
                    JD_QUERY = (
                        "Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning, "
                        "sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS, "
                        "evaluation NDCG MRR MAP A/B testing, Python, production systems, "
                        "startup product company, Pune Noida India"
                    )
                    
                    req = RerankRequest(query=JD_QUERY, passages=passages)
                    results = ranker.rerank(req)
                    
                    for res in results:
                        ce_scores[int(res["id"])] = res["score"]
            except Exception as e:
                st.warning(f"FlashRank failed: {e}")

    # ── STAGE 4: Score Fusion ──
    with st.spinner("Stage 4: Fusing Scores..."):
        fused = []
        if len(top_indices) > 0:
            xgb_norm = normalize_scores(top_xgb_scores)
            
            for i, idx in enumerate(top_indices):
                xgb_s = xgb_norm[i]
                ce_s = ce_scores.get(idx, xgb_s * 0.8)
                
                feats = feature_matrix[idx]
                heuristic_s = feats[IDX_SINGULARITY]
                
                final_score = rank_v2.W_XGB * xgb_s + rank_v2.W_CE * ce_s + rank_v2.W_HEURISTIC * heuristic_s
                fused.append({
                    "idx": idx,
                    "cid": candidate_ids[idx],
                    "score": final_score,
                    "xgb": xgb_s,
                    "ce": ce_s,
                    "heuristic": heuristic_s
                })
            
            # ── A/B Interleave Logic ──
            if blind_ab_mode:
                # Model A = Heuristic, Model B = ML final_score
                fused_a = sorted(fused, key=lambda x: (-x["heuristic"], x["cid"]))
                fused_b = sorted(fused, key=lambda x: (-x["score"], x["cid"]))
                
                interleaved = []
                seen_cids = set()
                # Interleave top 100 from each
                for i in range(100):
                    if i < len(fused_a):
                        ca = fused_a[i]
                        if ca["cid"] not in seen_cids:
                            ca["model_source"] = "Model_A"
                            interleaved.append(ca)
                            seen_cids.add(ca["cid"])
                    if i < len(fused_b):
                        cb = fused_b[i]
                        if cb["cid"] not in seen_cids:
                            cb["model_source"] = "Model_B"
                            interleaved.append(cb)
                            seen_cids.add(cb["cid"])
                fused = interleaved
            else:
                fused.sort(key=lambda x: (-x["score"], x["cid"]))
                source = "Heuristic" if xgb_model is None else "Model_B"
                for item in fused:
                    item["model_source"] = source
    t_end = time.time()
    st.session_state['fused'] = fused
    st.session_state['runtime'] = t_end - t_start
    st.session_state['run_pipeline'] = False
    st.session_state['model_mode'] = "Heuristic" if xgb_model is None else "ML"

fused = st.session_state['fused']
runtime = st.session_state['runtime']
model_mode = st.session_state.get('model_mode', 'ML')


# ── Tabs ──
tab_search, tab_eval = st.tabs(["🔍 Candidate Search", "📈 Evaluation Dashboard"])

with tab_search:
    # ── Mode Tag ──
    mode_cls = "mode-ml" if model_mode == "ML" else "mode-heur"
    mode_label = "🤖 ML Pipeline" if model_mode == "ML" else "📐 Heuristic Only"
    st.markdown(f'<div class="mode-tag {mode_cls}">{mode_label}</div>', unsafe_allow_html=True)

    # ── KPI Ribbon ──
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("📋 Candidates Scanned", f"{len(candidates):,}")
    with kpi2:
        st.metric("⚡ Latency", f"{runtime:.2f}s")
    with kpi3:
        st.metric("✅ Viable Matches", f"{len(fused):,}")
    with kpi4:
        best_score = fused[0]["score"] if fused else 0
        st.metric("🏆 Top Score", f"{best_score:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    dl_placeholder = st.empty()
    st.markdown('<div class="section-heading">🏆 Top Recommendations</div>', unsafe_allow_html=True)
        
    export_rows = []
    
    for rank_idx, item in enumerate(fused[:100], start=1):
        idx = item["idx"]
        cid = item["cid"]
        score = item["score"]
        cand = candidates[idx]
        prof = cand.get("profile", {})
        skills_list = [s.get("name", "") for s in cand.get("skills", [])]
        
        core_matched = [s for s in skills_list if s.lower() in CORE_IR_SKILLS]
        sota_matched = [s for s in skills_list if ATD_TAXONOMY.get(s.lower(), 0) >= 3]
        
        # Generate dynamic reasoning based on features
        feats = feature_matrix[idx]
        atd = float(feats[41])  # exact_atd
        hea = float(feats[42])  # exact_hea
        reasoning = generate_reasoning(cand, rank_idx, atd, hea)

        pipeline_note = (
            f" [Pipeline: Score={score:.3f}, "
            f"IR_match={feats[5]:.2f}, "
            f"depth={feats[9]:.2f}]"
        )
        reasoning += pipeline_note

        # ── Rank tier styling ──
        if rank_idx <= 3:
            tier, badge_cls, bar_cls, score_cls = "gold", "rank-badge-gold", "score-bar-gold", "score-badge-gold"
        elif rank_idx <= 10:
            tier, badge_cls, bar_cls, score_cls = "silver", "rank-badge-silver", "score-bar-silver", "score-badge-silver"
        elif rank_idx <= 25:
            tier, badge_cls, bar_cls, score_cls = "bronze", "rank-badge-bronze", "score-bar-bronze", "score-badge-bronze"
        else:
            tier, badge_cls, bar_cls, score_cls = "default", "rank-badge-default", "score-bar-default", "score-badge-default"

        score_pct = score * 100

        # ── Skills ──
        core_skills = [s["name"] for s in cand.get("skills", []) if s.get("proficiency") == "advanced"][:5]
        other_skills = [s["name"] for s in cand.get("skills", []) if s.get("proficiency") != "advanced"][:5]

        # ── Hidden Gem ──
        gem_atd = compute_atd(cand.get("skills", []), cand.get("career_history", []))
        gem_hea = compute_hea(cand)
        is_gem = gem_hea >= 1.0 and gem_atd < 1.0
        gem_html = '<span class="gem-badge">💎 Hidden Gem</span>' if is_gem else ""

        # ── Skill pills HTML ──
        core_pills = "".join(f'<span class="skill-pill skill-core">{s}</span>' for s in core_skills)
        other_pills = "".join(f'<span class="skill-pill skill-other">{s}</span>' for s in other_skills)
        skills_html = f'<div style="margin-top:6px">{core_pills}{other_pills}</div>' if (core_skills or other_skills) else ""

        # ── Signals HTML ──
        sigs = cand.get("redrob_signals", {})
        otw_cls = "signal-yes" if sigs.get("open_to_work_flag") else "signal-no"
        otw_txt = "✅ Open to Work" if sigs.get("open_to_work_flag") else "❌ Not Looking"
        reloc_cls = "signal-yes" if sigs.get("willing_to_relocate") else "signal-no"
        reloc_txt = "✅ Will Relocate" if sigs.get("willing_to_relocate") else "❌ No Relocation"
        notice = sigs.get("notice_period_days", "N/A")
        resp = sigs.get("recruiter_response_rate", 0) * 100

        # ── Career (top 3) ──
        career_items = ""
        for job in cand.get("career_history", [])[:3]:
            desc = (job.get("description") or "")[:120]
            desc = desc + "..." if len(desc) == 120 else desc
            career_items += f'<div class="career-item"><strong>{job.get("title", "")}</strong> @ {job.get("company", "")} <span style="color:#64748b;font-size:0.75rem">({job.get("duration_months", 0)} mo)</span>'
            if desc:
                career_items += f'<br><span style="color:#94a3b8;font-size:0.78rem">{desc}</span>'
            career_items += '</div>'

        # ── Candidate Name ──
        name = prof.get("anonymized_name", cid)
        headline = prof.get("headline", "")
        edu = ""
        if cand.get("education"):
            edu = cand["education"][0].get("tier", "")

        # ── Card HTML ──
        card_html = textwrap.dedent(f"""\
        <div class="candidate-card rank-{tier}">
          <div style="display:flex;gap:16px;align-items:flex-start">
            <div class="rank-badge {badge_cls}">#{rank_idx}</div>
            <div style="flex:1;min-width:0">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap">
                <div>
                  <div style="font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-bottom:2px">{name}</div>
                  <div style="font-size:0.88rem;color:#94a3b8;margin-bottom:2px">{prof.get("current_title", "")} @ {prof.get("current_company", "")}</div>
                  <div style="font-size:0.78rem;color:#64748b">📍 {prof.get("location", "")} · ⏱️ {prof.get("years_of_experience", 0)} yrs {"· 🎓 " + edu if edu else ""}</div>
                  {gem_html}
                </div>
                <div style="text-align:right;flex-shrink:0">
                  <div class="score-badge {score_cls}">{score:.4f}</div>
                  <div style="margin-top:6px">
                    <div class="score-bar-container" style="width:120px">
                      <div class="score-bar {bar_cls}" style="width:{score_pct}%"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div style="margin-top:8px;font-size:0.8rem;color:#64748b;font-style:italic">{headline}</div>
              {skills_html}
              <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
                <span class="signal-tag {otw_cls}">{otw_txt}</span>
                <span class="signal-tag {reloc_cls}">{reloc_txt}</span>
                <span class="signal-tag signal-no">📅 Notice: {notice}d</span>
                <span class="signal-tag signal-yes">📞 Response: {resp:.0f}%</span>
              </div>
            </div>
          </div>
        </div>
        """).strip()

        # Strip all leading/trailing whitespace per line to avoid Markdown treating indented lines as code blocks
        card_html = "".join(line.strip() for line in card_html.split("\n"))

        # ── Render card HTML + Streamlit interactive elements ──
        st.markdown(card_html, unsafe_allow_html=True)

        # Feedback + career expander (Streamlit interactive parts)
        fcol1, fcol2, fcol3 = st.columns([1, 1, 4])
        with fcol1:
            if st.button("👍 Good", key=f"up_{cid}_{rank_idx}"):
                log_feedback(cid, rank_idx, score, item["model_source"], 1)
                st.success("Logged!")
        with fcol2:
            if st.button("👎 Reject", key=f"down_{cid}_{rank_idx}"):
                log_feedback(cid, rank_idx, score, item["model_source"], -1)
                st.error("Logged!")

        with st.expander("Career History & AI Reasoning"):
            if not blind_ab_mode:
                st.info(reasoning)
            st.markdown(career_items, unsafe_allow_html=True)

        # Save to export list
        export_rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning.replace("\n", " ")
        })
        
    # Download Button logic via placeholder
    df_export = pd.DataFrame(export_rows)
    
    # Download Button logic via popover using the placeholder
    with dl_placeholder.container():
        with st.popover("📥 Export Top 100"):
            st.markdown("Select format to download:")
            # CSV Export
            csv_data = df_export.to_csv(index=False, quoting=csv.QUOTE_ALL)
            st.download_button(
                label="📄 CSV (.csv)",
                data=csv_data,
                file_name="streamlit_submission.csv",
                mime="text/csv",
                use_container_width=True
            )
            # Excel Export
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_export.to_excel(writer, sheet_name="Top 100", index=False)
            excel_data = buf.getvalue()
            st.download_button(
                label="📊 Excel (.xlsx)",
                data=excel_data,
                file_name="streamlit_submission.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

with tab_eval:
    st.markdown('<div class="section-heading">📈 Evaluation & Offline Benchmarks</div>', unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_FILE):
        try:
            logs = []
            with open(FEEDBACK_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))

            if logs:
                df_logs = pd.DataFrame(logs)

                # ── A/B Stats ──
                if "model_source" in df_logs.columns:
                    ab_stats = df_logs.groupby("model_source")["feedback"].agg(["mean", "count"]).reset_index()
                    ab_stats.rename(columns={"mean": "Avg Sentiment", "count": "Total Votes"}, inplace=True)
                    ab_stats["Avg Sentiment"] = ab_stats["Avg Sentiment"].round(3)
                    st.markdown("**Real-world A/B Test Results**")
                    st.dataframe(ab_stats, use_container_width=True, hide_index=True)

                # ── Trend Chart ──
                df_logs["date"] = pd.to_datetime(df_logs["timestamp"], unit="s")
                daily = df_logs.groupby([df_logs["date"].dt.date, "model_source"])["feedback"].mean().reset_index()
                fig = px.line(daily, x="date", y="feedback", color="model_source",
                              title="Recruiter Engagement Over Time",
                              labels={"feedback": "Avg Sentiment", "date": ""})
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_traces(line=dict(width=2.5))
                st.plotly_chart(fig, use_container_width=True)

                # ── Recent Feedback ──
                st.markdown("**Recent Feedback**")
                st.dataframe(
                    df_logs[["date", "candidate_id", "rank_position", "model_source", "feedback"]].tail(10),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No feedback data yet.")
        except Exception as e:
            st.error(f"Error loading logs: {e}")
    else:
        st.info("No recruiter feedback logged yet. Start rating candidates in the Search tab to build the evaluation benchmarks.")

