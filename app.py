#!/usr/bin/env python3
"""
Redrob MatchWise — Streamlit Sandbox Demo
==========================================
A lightweight web application that allows judges to upload a small
candidate sample (≤ 100 candidates) and verify the ranking system
end-to-end on CPU within 5 minutes.

Run locally:
    streamlit run app.py

Deploy to HuggingFace Spaces or Streamlit Cloud as the sandbox link.
"""

import csv
import io
import json
import os
import time

import streamlit as st

# ── Import ranking logic from rank.py ────────────────────────────────
# We import the scoring and reasoning functions directly so the sandbox
# uses the exact same logic as the CLI ranker.
from rank import (
    score_candidate,
    generate_reasoning,
    load_honeypots,
    SERVICE_COMPANIES,
    UNRELATED_TITLES,
    CORE_IR_SKILLS,
    ADVANCED_ML_SKILLS,
    load_precomputed_embeddings,
    get_cosine_scores,
    get_bm25_scores,
)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          PAGE CONFIG                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@st.cache_resource
def get_cached_embeddings():
    return load_precomputed_embeddings()

st.set_page_config(
    page_title="Redrob MatchWise",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d2b 100%);
        color: #e0e0f0;
    }

    .main-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.5);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .candidate-card {
        background: rgba(30, 30, 60, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
    }

    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-right: 0.5rem;
    }

    .score-badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
    }

    .skill-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #c4b5fd;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        margin: 0.1rem 0.15rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .reasoning-text {
        color: #cbd5e1;
        font-style: italic;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 0.5rem;
        padding-left: 0.5rem;
        border-left: 3px solid rgba(139, 92, 246, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           SIDEBAR                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("### 🎯 Redrob MatchWise")
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Upload a `.json` or `.jsonl` file containing candidate profiles
    2. The ranker scores and ranks candidates in real-time
    3. Download the ranked CSV output

    **Constraints enforced:**
    - CPU only, no GPU
    - No external API calls
    - Top 100 candidates max
    """)
    st.markdown("---")
    st.markdown("Built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          MAIN APP                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

st.markdown('<p class="main-header">Redrob MatchWise</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Intelligent Discovery & Ranking — Senior AI Engineer (Founding Team)</p>',
    unsafe_allow_html=True,
)

# File upload
uploaded = st.file_uploader(
    "Upload candidate profiles (.json or .jsonl)",
    type=["json", "jsonl"],
    help="Upload a JSON array or JSONL file of candidate profiles (≤ 100 candidates).",
)

if uploaded is not None:
    t0 = time.time()

    # Parse candidates
    raw = uploaded.read().decode("utf-8")
    candidates = []
    try:
        # Try JSON array first
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = [parsed]
    except json.JSONDecodeError:
        # Try JSONL
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if not candidates:
        st.error("No valid candidate profiles found in the uploaded file.")
        st.stop()

    st.success(f"Loaded **{len(candidates)}** candidate profiles.")

    # Load honeypots
    honeypot_path = os.path.join(os.path.dirname(__file__), "honeypots.json")
    honeypot_ids = load_honeypots(honeypot_path)

    # Score candidates
    with st.spinner("Scoring and ranking candidates..."):
        # Load precomputed embeddings (cached resource)
        cand_embs, jd_emb, id_to_idx = get_cached_embeddings()

        # Apply hard filters first (BM25 needs to run on viable candidate corpus)
        valid_candidates = []
        for cand in candidates:
            cid = cand.get("candidate_id", "")
            if cid in honeypot_ids:
                continue
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            
            # Service-only career -> disqualify
            companies = {j.get("company") for j in career if j.get("company")}
            if companies and all(c in SERVICE_COMPANIES for c in companies):
                continue
                
            # Current title is completely unrelated -> disqualify
            curr_title = profile.get("current_title", "").lower().strip()
            if curr_title in UNRELATED_TITLES:
                continue
                
            valid_candidates.append(cand)

        if valid_candidates:
            # Compute BM25 scores
            bm25_scores = get_bm25_scores(valid_candidates)
            # Compute cosine similarities (loads ST model on-the-fly if missing from cache)
            cos_scores = get_cosine_scores(valid_candidates, cand_embs, jd_emb, id_to_idx)
            
            scored = []
            for cand in valid_candidates:
                cid = cand.get("candidate_id", "")
                cos_val = cos_scores.get(cid, 0.0)
                bm25_val = bm25_scores.get(cid, 0.0)
                sc, debug = score_candidate(cand, honeypot_ids, cos_score=cos_val, bm25_score=bm25_val)
                scored.append((cid, sc, cand, debug))
        else:
            scored = []

        scored.sort(key=lambda x: (-x[1], x[0]))
        top = scored[:100]

    t_end = time.time()
    runtime = t_end - t0

    # ── Metrics row ───────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(candidates)}</div>
            <div class="metric-label">Candidates Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(scored)}</div>
            <div class="metric-label">Passed Filters</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(top)}</div>
            <div class="metric-label">Ranked Output</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{runtime:.1f}s</div>
            <div class="metric-label">Runtime</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Ranked candidates display ─────────────────────────────────────
    st.markdown("### 🏆 Ranked Candidates")

    csv_rows = []
    for rank_idx, (cid, score, cand, debug) in enumerate(top, start=1):
        prof = cand.get("profile", {})
        skills_list = [s.get("name", "") for s in cand.get("skills", [])]
        reasoning = generate_reasoning(cand, rank_idx)

        csv_rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

        # Highlight core skills
        core_matched = [s for s in skills_list if s.lower() in CORE_IR_SKILLS]
        adv_matched  = [s for s in skills_list if s.lower() in ADVANCED_ML_SKILLS]

        skill_chips = ""
        for s in core_matched[:5]:
            skill_chips += f'<span class="skill-chip" style="border-color: rgba(34,197,94,0.4); color: #4ade80;">{s}</span>'
        for s in adv_matched[:3]:
            skill_chips += f'<span class="skill-chip">{s}</span>'

        st.markdown(f"""
        <div class="candidate-card">
            <div>
                <span class="rank-badge">#{rank_idx}</span>
                <strong style="font-size: 1.05rem;">{prof.get('anonymized_name', cid)}</strong>
                &nbsp;
                <span class="score-badge">Score: {score:.4f}</span>
            </div>
            <div style="margin-top: 0.4rem; color: #94a3b8; font-size: 0.88rem;">
                {prof.get('current_title', '')} at {prof.get('current_company', '')}
                &nbsp;·&nbsp; {prof.get('years_of_experience', 0):.1f} yrs
                &nbsp;·&nbsp; {prof.get('location', '')}
            </div>
            <div style="margin-top: 0.4rem;">{skill_chips}</div>
            <div class="reasoning-text">{reasoning}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download CSV ──────────────────────────────────────────────────
    st.markdown("---")
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    writer.writerows(csv_rows)

    st.download_button(
        label="📥 Download Ranked CSV",
        data=output.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )

else:
    # Default state with instructions
    st.markdown("---")
    st.info(
        "👆 Upload a candidate profile file to get started. "
        "The system will score, rank, and explain each candidate's fit "
        "for the Senior AI Engineer (Founding Team) role."
    )

    with st.expander("📋 Expected candidate JSON schema"):
        st.json({
            "candidate_id": "CAND_XXXXXXX",
            "profile": {
                "anonymized_name": "...",
                "headline": "...",
                "summary": "...",
                "location": "City, State",
                "country": "India",
                "years_of_experience": 6.5,
                "current_title": "ML Engineer",
                "current_company": "...",
                "current_company_size": "51-200",
                "current_industry": "Software",
            },
            "career_history": [{"company": "...", "title": "...", "start_date": "2020-01-01", "end_date": None, "duration_months": 48, "is_current": True, "industry": "...", "company_size": "...", "description": "..."}],
            "education": [{"institution": "...", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2014, "end_year": 2018, "grade": "8.5 CGPA", "tier": "tier_1"}],
            "skills": [{"name": "PyTorch", "proficiency": "advanced", "endorsements": 25, "duration_months": 36}],
            "redrob_signals": {
                "profile_completeness_score": 92.0,
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.85,
                "notice_period_days": 30,
                "willing_to_relocate": True,
                "interview_completion_rate": 0.9,
                "...": "..."
            },
        })
