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
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import offline feature engineering & taxonomy
from build_features import extract_features, FEATURE_NAMES, NUM_FEATURES
from src.rank import (
    ATD_TAXONOMY,
    CORE_IR_SKILLS,
    compute_atd,
    compute_hea,
    generate_reasoning,
    load_honeypots,
)

# Import pipeline constants
from rank_v2 import (
    IDX_TIMELINE, IDX_INFLATION, IDX_SERVICE, IDX_GHOST, IDX_MAX_ATD,
    IDX_SINGULARITY, IDX_CORE_IR, IDX_DEPTH,
    W_XGB, W_CE, W_HEURISTIC,
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
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        "ranker.xgb",
        "--lgb-model-out",
        "ranker.lgb",
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
    st.markdown("### ⚡ Omni-Context Engine")
    st.markdown("---")
    
    def handle_upload():
        st.session_state.pop("fused", None)
        
    uploaded = st.file_uploader(
        "Upload candidate profiles (.json or .jsonl)",
        type=["json", "jsonl"],
        on_change=handle_upload
    )
    
    st.markdown("---")
    st.markdown("""
    **Architecture:**
    1. **Hard Filtering** (Heuristics)
    2. **XGBoost LambdaMART** (LTR)
    3. **FlashRank** (Cross-Encoder)
    4. **Score Fusion**
    """)
    st.markdown("---")
    
    st.markdown("### ⚙️ Pipeline Controls")
    use_crossencoder = st.toggle("Enable FlashRank (Stage 3)", value=True, 
                                 help="Uses ms-marco-TinyBERT to semantically rerank top 50 candidates.")
    top_k_xgb = st.number_input("XGBoost Top-K", min_value=10, max_value=500, value=200, step=10)
    
    st.markdown("### 🧪 Evaluation Infrastructure")
    blind_ab_mode = st.toggle("Blind A/B Test Mode", value=False, help="Interleave Model A (Heuristics) vs Model B (ML) blindly for recruiter feedback.")
    
    st.markdown("---")
    run_btn = st.button("▶️ Rerun Pipeline", type="primary", use_container_width=True)
    if run_btn:
        st.session_state["run_pipeline"] = True

    retrain_btn = st.button("🔄 Retrain with Feedback", use_container_width=True)
    if retrain_btn:
        if uploaded is None:
            st.error("Please upload a dataset before retraining.")
        else:
            with st.spinner("Retraining rankers with recruiter feedback..."):
                try:
                    # Use the instantly cached parser to get the current uploaded features
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
    st.markdown("Built for the **Redrob Intelligent Candidate Discovery**")

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

@st.cache_resource(show_spinner=False, max_entries=1)
def parse_and_extract(raw_bytes):
    """Caches the parsing and feature extraction so it only runs once per file."""
    candidates = []
    try:
        parsed = json.loads(raw_bytes)
        if isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = [parsed]
    except Exception:
        raw_text = raw_bytes.decode("utf-8")
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip the malformed line to prevent crashing the entire stream
                
    if not candidates:
        return [], [], np.array([])
        
    feature_matrix = np.zeros((len(candidates), NUM_FEATURES), dtype=np.float32)
    candidate_ids = []
    for i, cand in enumerate(candidates):
        cid = cand.get("candidate_id", "")
        candidate_ids.append(cid)
        feature_matrix[i] = extract_features(cand)
        
    # Replace NaN/Inf hazards
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    return candidates, candidate_ids, feature_matrix

if uploaded is not None:
    # Parse and extract from uploaded (Cached!)
    with st.spinner("Stage 0: Extracting offline features (Cached for speed - may take 1-3 mins for 100k candidates)..."):
        raw_bytes = uploaded.getvalue()
        candidates, candidate_ids, feature_matrix = parse_and_extract(raw_bytes)

    if not candidates:
        st.error("No valid candidate profiles found.")
        st.stop()

    if "fused" not in st.session_state or st.session_state.get("run_pipeline", False):
        t_start = time.time()

        # Load Models (graceful fallback if missing)
        with st.spinner("Stage 0: Loading ML Models..."):
            honeypot_path = os.path.join(PROJECT_DIR, "honeypots.json")
            honeypot_ids = load_honeypots(honeypot_path)
            
            model_path = os.path.join(PROJECT_DIR, "ranker.xgb")
            lgb_path = os.path.join(PROJECT_DIR, "ranker.lgb")
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
            with st.spinner("Stage 3: FlashRank Semantic Reranking (Top 50)..."):
                try:
                    from flashrank import RerankRequest
                    ranker = load_flashrank()
                    if ranker is not None:
                        # Build passages
                        passages = []
                        rerank_k = min(len(top_indices), 50)
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
                    
                    final_score = W_XGB * xgb_s + W_CE * ce_s + W_HEURISTIC * heuristic_s
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
        # 1. Global KPI Ribbon
        # Show which mode is active
        mode_tag = "🧪 Heuristic-Only Mode" if model_mode == "Heuristic" else "🤖 Full ML Pipeline"
        st.markdown(f"<span style='background:#334155;padding:4px 14px;border-radius:20px;font-size:0.85rem;'>{mode_tag}</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Candidates Scanned", f"{len(candidates):,}")
        with kpi2:
            st.metric("End-to-End Latency", f"{runtime:.2f}s")
        with kpi3:
            st.metric("Viable Matches", f"{len(fused):,}")
        with kpi4:
            best_score = fused[0]["score"] if fused else 0
            st.metric("Top Match Score", f"{best_score:.4f}")
    
        st.markdown("<br>", unsafe_allow_html=True)
            
        dl_placeholder = st.empty()
        st.markdown("### 🏆 Top Recommendations")
            
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
            
            # Better structured candidate representation
            with st.container(border=True):
                col_main, col_metrics, col_score = st.columns([2.5, 1.5, 1])
                
                with col_main:
                    st.subheader(f"#{rank_idx} {prof.get('anonymized_name', cid)}")
                    st.markdown(f"**{prof.get('current_title', 'Unknown Title')}** @ {prof.get('current_company', 'Unknown Company')}")
                    
                    # Use tags for quick demographic info
                    demo_tags = f"📍 {prof.get('location', 'Remote')} | ⏱️ {prof.get('years_of_experience', 0)} Yrs Exp | "
                    demo_tags += f"🎓 {cand.get('education', [{}])[0].get('tier', 'Unknown Tier')}" if cand.get('education') else "🎓 No Edu Data"
                    st.caption(demo_tags)
                    
                    # Skills categorized
                    adv_skills = [s['name'] for s in cand.get('skills', []) if s.get('proficiency') == 'advanced'][:5]
                    other_skills = [s['name'] for s in cand.get('skills', []) if s.get('proficiency') != 'advanced'][:5]
                    
                    if adv_skills:
                        st.markdown(f"**🔥 Core Strengths:** `{'` `'.join(adv_skills)}`")
                    if other_skills:
                        st.markdown(f"**Other Skills:** {', '.join(other_skills)}")
                    
                    st.markdown(f"_{prof.get('headline', '')}_")
                    
                    # Hidden Gem badge (matches generate_reasoning logic in rank.py)
                    gem_atd = compute_atd(cand.get("skills", []), cand.get("career_history", []))
                    gem_hea = compute_hea(cand)
                    if gem_hea >= 1.0 and gem_atd < 1.0:
                        st.markdown(
                            '<span style="background-color:#f59e0b;color:#1e293b;'
                            'padding:2px 10px;border-radius:12px;font-size:0.75rem;'
                            'font-weight:600;">💎 Hidden Gem</span>',
                            unsafe_allow_html=True,
                        )

                with col_metrics:
                    st.markdown("**Redrob Signals**")
                    sigs = cand.get("redrob_signals", {})
                    
                    # Signal badges
                    otw = "✅ Open to Work" if sigs.get('open_to_work_flag') else "❌ Not Actively Looking"
                    reloc = "✅ Willing to Relocate" if sigs.get('willing_to_relocate') else "❌ No Relocation"
                    st.caption(f"{otw}")
                    st.caption(f"{reloc}")
                    st.caption(f"**Notice Period:** {sigs.get('notice_period_days', 'N/A')} Days")
                    st.caption(f"**Response Rate:** {sigs.get('recruiter_response_rate', 0)*100:.0f}%")
                    
                
                with col_score:
                    if blind_ab_mode:
                        st.metric("Final Score", "Blind Mode")
                    else:
                        st.metric("Final Score", f"{score:.4f}")
                        st.caption(f"Ranker: {item['xgb']:.4f}")
                        if use_crossencoder:
                            st.caption(f"FlashRank: {item['ce']:.4f}")
                
                # Feedback Buttons
                f_col1, f_col2, f_col3 = st.columns([1,1,2])
                with f_col1:
                    if st.button("👍 Good", key=f"up_{cid}_{rank_idx}"):
                        log_feedback(cid, rank_idx, score, item["model_source"], 1)
                        st.success("Logged!")
                with f_col2:
                    if st.button("👎 Reject", key=f"down_{cid}_{rank_idx}"):
                        log_feedback(cid, rank_idx, score, item["model_source"], -1)
                        st.error("Logged!")
                
                with st.expander("Career History & AI Reasoning"):
                    if not blind_ab_mode:
                        st.info(reasoning)
                    for job in cand.get("career_history", [])[:3]:
                        st.markdown(f"- **{job.get('title', '')}** @ {job.get('company', '')} ({job.get('duration_months', 0)} mos)")
                        if job.get('description'):
                            st.caption(job.get('description')[:150] + "...")

            # Save to export list
            export_rows.append({
                "candidate_id": cid,
                "rank": rank_idx,
                "score": round(score, 6),
                "reasoning": reasoning.replace("\n", " ")
            })
            
        # Download Button logic via placeholder
        df_export = pd.DataFrame(export_rows)
        csv_data = df_export.to_csv(index=False, quoting=csv.QUOTE_ALL)
        
        dl_placeholder.download_button(
            label="📥 Export Top 100 as CSV",
            data=csv_data,
            file_name="streamlit_submission.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with tab_eval:
        st.markdown("### 📈 Evaluation & Offline Benchmarks")
        
        if os.path.exists(FEEDBACK_FILE):
            try:
                logs = []
                with open(FEEDBACK_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
                            
                if logs:
                    df_logs = pd.DataFrame(logs)
                    
                    st.markdown("#### Real-world A/B Test Results")
                    # Calculate win rates
                    if "model_source" in df_logs.columns:
                        ab_stats = df_logs.groupby("model_source")["feedback"].agg(['mean', 'count']).reset_index()
                        ab_stats.rename(columns={'mean': 'Avg Score (-1 to 1)', 'count': 'Votes'}, inplace=True)
                        st.dataframe(ab_stats, use_container_width=True)
                    
                    # Trend chart
                    df_logs['date'] = pd.to_datetime(df_logs['timestamp'], unit='s')
                    daily = df_logs.groupby([df_logs['date'].dt.date, 'model_source'])['feedback'].mean().reset_index()
                    fig = px.line(daily, x='date', y='feedback', color='model_source', title="Recruiter Engagement (Moving Average)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### Raw Feedback Log")
                    st.dataframe(df_logs[['date', 'candidate_id', 'rank_position', 'model_source', 'feedback']].tail(10))
                else:
                    st.info("No feedback data yet.")
            except Exception as e:
                st.error(f"Error loading logs: {e}")
        else:
            st.info("No recruiter feedback logged yet. Start rating candidates in the Search tab to build the evaluation benchmarks.")

else:
    # Empty State Premium Layout
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="background-color: rgba(30, 41, 59, 0.5); padding: 2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); text-align: center;">
                <h2 style="margin-bottom: 0;">Welcome to MatchWise v2</h2>
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 10px;">
                    The Omni-Context Ranking Engine for technical talent.
                </p>
                <br>
                <div style="text-align: left; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                    <h4>🚀 Getting Started</h4>
                    <p>Upload a candidate dataset in the sidebar to benchmark the engine.</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
