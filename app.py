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
import pandas as pd
import time

import streamlit as st

# ── Import ranking logic from rank.py ────────────────────────────────
# We import the scoring and reasoning functions directly so the sandbox
# uses the exact same logic as the CLI ranker.
from rank import (
    compute_atd,
    compute_hea,
    compute_singularity_score,
    generate_reasoning,
    load_honeypots,
    load_reasoning_cache,
    SERVICE_COMPANIES,
    UNRELATED_TITLES,
    CORE_IR_SKILLS,
    CV_SPEECH_SKILLS,
    ATD_TAXONOMY,
)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          PAGE CONFIG                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# Removed cache_resource to force live re-generation of reasoning text
def get_cached_reasoning():
    return {}

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

    .hidden-gem-tag {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        margin-right: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .caveat-tag {
        color: #ef4444;
        font-weight: 600;
        font-size: 0.85rem;
        background: rgba(239, 68, 68, 0.1);
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
    }

    .highlight-tag {
        color: #10b981;
        font-weight: 600;
        font-size: 0.85rem;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
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
# ║                           SIDEBAR                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("### 🎯 Redrob MatchWise")
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Adjust scoring weights in the settings below.
    2. Upload a `.json` or `.jsonl` file of candidates.
    3. The ranker scores and ranks candidates in real-time.
    4. Download the ranked CSV output.
    """)
    st.markdown("---")
    
    st.markdown("### ⚙️ Scoring Calibration")
    
    w_title = st.slider("Core AI Title Bonus", 1.0, 2.0, 1.20, 0.05, 
                        help="Multiplier for candidates holding Senior AI/ML/NLP/Search titles currently.")
    w_exp = st.slider("Experience Sweet Spot Weight", 1.0, 2.0, 1.10, 0.05, 
                      help="Peak score bonus for candidates with ~7 years of experience.")
    w_github = st.slider("GitHub Activity Weight", 0.1, 1.0, 0.35, 0.05,
                         help="Weight for GitHub activity continuous bonus.")
    w_fullstack = st.slider("DevOps / Full-stack Bonus", 1.0, 1.5, 1.15, 0.05,
                            help="Multiplier for founding engineers with DevOps/Backend skills.")
    w_startup = st.slider("Startup Survival Bonus", 1.0, 1.5, 1.15, 0.05,
                          help="Multiplier for candidates with experience at small companies.")
    w_product = st.slider("Product DNA Weight", 1.0, 1.5, 1.15, 0.05,
                          help="Multiplier for candidates with product company stints.")
    w_recency = st.slider("Active Recency Weight", 0.0, 2.0, 1.0, 0.1,
                          help="Importance of recent platform activity vs inactive ghost profiles.")
    w_notice = st.slider("Notice Period Weight", 0.0, 2.0, 1.0, 0.1,
                         help="Importance of short notice periods (<=30 days).")
    w_domain = st.slider("Domain Match Bonus", 1.0, 1.5, 1.15, 0.05,
                         help="Multiplier for experience in HR-Tech, Recruitment, or Marketplaces.")
    
    custom_weights = {
        "title": w_title,
        "experience": w_exp,
        "github": w_github,
        "fullstack": w_fullstack,
        "startup": w_startup,
        "product": w_product,
        "recency": w_recency,
        "notice": w_notice,
        "domain": w_domain
    }
    
    st.markdown("---")
    st.markdown("Built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          MAIN APP                                     ║
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

    # ── Load resources ───────────────────────────────────────────────────
    honeypot_path = os.path.join(os.path.dirname(__file__), "honeypots.json")
    honeypot_ids = load_honeypots(honeypot_path)
    reasoning_cache = get_cached_reasoning()

    # ── Pre-score and filter all candidates ──────────────────────────────
    scored = []
    all_candidates_info = []

    with st.spinner("Analyzing, scoring and ranking profiles..."):
        for idx, cand in enumerate(candidates):
            if not cand:
                continue
            cid = cand.get("candidate_id", "")
            profile = cand.get("profile", {}) or {}
            career  = cand.get("career_history", []) or []
            skills  = cand.get("skills", []) or []

            # ── Check filters ──
            filters_tripped = []
            
            # 1. Honeypot check
            if cid in honeypot_ids:
                filters_tripped.append("Honeypot")

            # 2. Service-only check
            companies = {j.get("company") for j in career if j.get("company")}
            if companies and all(c in SERVICE_COMPANIES for c in companies):
                filters_tripped.append("Service-only")

            # 3. Unrelated title check
            curr_title = profile.get("current_title", "").lower().strip()
            if curr_title in UNRELATED_TITLES:
                career_titles_text = " ".join(j.get("title", "") for j in career).lower()
                has_tech_history = any(
                    kw in career_titles_text
                    for kw in ("engineer", "developer", "scientist", "ml",
                               "ai", "data", "research")
                )
                if not has_tech_history:
                    filters_tripped.append("Unrelated Title")

            # Compute raw scores
            atd = compute_atd(skills, career)
            
            # 4. Low ATD check
            if atd < 0.10:
                filters_tripped.append("Low ATD")

            hea = compute_hea(cand, custom_weights)
            
            # 5. Low HEA check
            if hea <= 0.0:
                filters_tripped.append("Low HEA")

            # 6. LangChain tourist check
            max_lvl = max([ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) for s in skills] + [0])
            max_ml_months = max([s.get("duration_months", 0) for s in skills if ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) >= 1] + [0])
            if max_lvl == 1 and max_ml_months < 12:
                filters_tripped.append("LangChain Tourist")

            # Compute final score
            score = compute_singularity_score(atd, hea)

            # Generate reasoning text
            reasoning = reasoning_cache.get(cid, "")
            if not reasoning:
                # We can approximate a rank index (using 1-based index)
                reasoning = generate_reasoning(cand, idx + 1, atd, hea)

            info = {
                "cand": cand,
                "cid": cid,
                "profile": profile,
                "skills": skills,
                "score": score,
                "atd": atd,
                "hea": hea,
                "reasoning": reasoning,
                "filters_tripped": filters_tripped,
                "original_index": idx
            }
            all_candidates_info.append(info)

            # If passed all filters, append to scored
            if not filters_tripped:
                scored.append({'cid': cid, 'score': score, 'cand': cand, 'atd': atd, 'hea': hea})

        # Find 100th score threshold for standard candidates
        scored.sort(key=lambda x: (-x['score'], x['cid']))
        threshold_score = scored[99]['score'] if len(scored) > 100 else 0.0

        standard_candidates = []
        hidden_gems = []
        
        for idx, cand_obj in enumerate(scored):
            # Check hidden gem criteria
            if cand_obj['hea'] >= 1.0 and cand_obj['atd'] < 1.0:
                hidden_gems.append(cand_obj)
            else:
                standard_candidates.append(cand_obj)
                
        # Sort hidden gems by HEA descending to find the absolute best athletes
        hidden_gems.sort(key=lambda x: -x['hea'])
                
        final_top = []
        quota = 100 - min(len(hidden_gems), 10)
        final_top.extend(standard_candidates[:quota])
        final_top.extend(hidden_gems[:10])
        
        # Sort combined top 100 by final score
        final_top.sort(key=lambda x: (-x['score'], x['cid']))
        
        # Convert back to tuple expected by Streamlit rendering logic
        top = [(c['cid'], c['score'], c['cand'], c['atd'], c['hea']) for c in final_top[:100]]

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

    # ── Scatter plot (ATD vs HEA) ─────────────────────────────────────
    st.markdown("### 📊 Candidate Distribution (ATD vs HEA)")
    
    chart_data = []
    for rank_idx, (cid, score, cand, atd, hea) in enumerate(top, start=1):
        prof = cand.get("profile", {})
        chart_data.append({
            "Rank": f"#{rank_idx}",
            "Candidate": prof.get("anonymized_name", cid),
            "Technical Floor (ATD)": float(atd),
            "Execution Agency (HEA)": float(hea),
            "Final Score": float(score)
        })
    
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.scatter_chart(
            df,
            x="Technical Floor (ATD)",
            y="Execution Agency (HEA)",
            size="Final Score",
            color="Rank",
            use_container_width=True
        )

    st.markdown("---")

    # ── Ranked candidates display ─────────────────────────────────────
    st.markdown("### 🏆 Ranked Candidates")

    csv_rows = []
    for rank_idx, (cid, score, cand, atd, hea) in enumerate(top, start=1):
        prof = cand.get("profile", {})
        skills_list = [s.get("name", "") for s in cand.get("skills", [])]
        
        # Look up reasoning from cache or generate on-the-fly
        reasoning = reasoning_cache.get(cid, "")
        if not reasoning:
            reasoning = generate_reasoning(cand, rank_idx, atd, hea)

        csv_rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

        # Highlight core skills
        core_matched = [s for s in skills_list if s.lower() in CORE_IR_SKILLS]
        # SOTA skills (level 3 or 4 in taxonomy)
        sota_matched = [s for s in skills_list if ATD_TAXONOMY.get(s.lower(), 0) >= 3]

        skill_chips = ""
        for s in core_matched[:5]:
            skill_chips += f'<span class="skill-chip" style="border-color: rgba(34,197,94,0.4); color: #4ade80;">{s}</span>'
        for s in sota_matched[:5]:
            skill_chips += f'<span class="skill-chip">{s}</span>'

        # Apply rich HTML formatting to the reasoning text
        display_reasoning = reasoning
        if "[Hidden Gem 💎]" in display_reasoning:
            display_reasoning = display_reasoning.replace(
                "[Hidden Gem 💎]", 
                "<span class='hidden-gem-tag'>💎 Hidden Gem</span>"
            )
        
        for kw in ["Note:", "Caveat:", "Consideration:", "Flag:", "Potential concern:"]:
            if kw in display_reasoning:
                display_reasoning = display_reasoning.replace(
                    kw, f"<span class='caveat-tag'>{kw}</span>"
                )
                
        for kw in ["Highlight:", "Key Strength:", "Bonus:"]:
            if kw in display_reasoning:
                display_reasoning = display_reasoning.replace(
                    kw, f"<span class='highlight-tag'>{kw}</span>"
                )

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
            <div class="reasoning-text">{display_reasoning}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Actions Row (Download CSV & Show Candidates Toggle) ───────────
    st.markdown("---")
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    writer.writerows(csv_rows)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Ranked CSV",
            data=output.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if "show_uploaded" not in st.session_state:
            st.session_state.show_uploaded = False

        def toggle_show_uploaded():
            st.session_state.show_uploaded = not st.session_state.show_uploaded

        st.button(
            "📋 Show All Uploaded Candidates" if not st.session_state.show_uploaded else "🙈 Hide Uploaded Candidates",
            on_click=toggle_show_uploaded,
            key="toggle_uploaded_btn",
            use_container_width=True,
        )

    if st.session_state.show_uploaded:
        st.markdown("---")
        st.markdown("### 📋 All Uploaded Candidates")
        
        # Add search and sorting controls
        ctrl_col1, ctrl_col2 = st.columns([3, 1])
        with ctrl_col1:
            search_query = st.text_input(
                "🔍 Search loaded candidates",
                "",
                key="uploaded_search_all",
                help="Type to search by name, title, company, skills, or location."
            )
        with ctrl_col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Upload Order", "Score (High to Low)"],
                index=0,
                key="uploaded_sort_by"
            )

        # Apply search filter
        filtered_display = []
        for item in all_candidates_info:
            prof = item["profile"] or {}
            skills_list = item["skills"] or []
            skills_str = " ".join(s.get("name", "") for s in skills_list if s).lower()
            text_to_search = f"{item['cid']} {prof.get('anonymized_name', '')} {prof.get('current_title', '')} {prof.get('current_company', '')} {prof.get('location', '')} {prof.get('country', '')} {skills_str}".lower()
            if not search_query or search_query.lower() in text_to_search:
                filtered_display.append(item)

        # Apply sort
        if sort_by == "Score (High to Low)":
            filtered_display.sort(key=lambda x: (-x["score"], x["cid"]))
        else:
            filtered_display.sort(key=lambda x: x["original_index"])

        if not filtered_display:
            st.warning("No candidates matched your search query.")
        else:
            # Reset page to 1 if search query changed
            if "last_search_query" not in st.session_state:
                st.session_state.last_search_query = ""
            if search_query != st.session_state.last_search_query:
                st.session_state.uploaded_page = 1
                st.session_state.last_search_query = search_query

            # ── Pagination logic ──
            items_per_page = 10
            total_items = len(filtered_display)
            total_pages = (total_items - 1) // items_per_page + 1

            if "uploaded_page" not in st.session_state:
                st.session_state.uploaded_page = 1

            if st.session_state.uploaded_page > total_pages:
                st.session_state.uploaded_page = total_pages
            if st.session_state.uploaded_page < 1:
                st.session_state.uploaded_page = 1

            current_page = st.session_state.uploaded_page

            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            page_items = filtered_display[start_idx:end_idx]

            st.info(f"Showing **{start_idx + 1}-{end_idx}** of **{total_items}** candidates (Page {current_page} of {total_pages})")

            # Render cards
            for item in page_items:
                cand = item["cand"]
                cid = item["cid"]
                prof = item["profile"] or {}
                score = item["score"]
                reasoning = item["reasoning"]
                filters = item["filters_tripped"]
                skills_list = [s.get("name", "") for s in item["skills"] if s]
                
                # Highlight core/SOTA skills
                core_matched = [s for s in skills_list if s.lower() in CORE_IR_SKILLS]
                sota_matched = [s for s in skills_list if ATD_TAXONOMY.get(s.lower(), 0) >= 3]
                
                # Render skills list
                all_matched = core_matched + [s for s in sota_matched if s not in core_matched]
                other_skills = [s for s in skills_list if s not in all_matched]
                display_skills = all_matched + other_skills
                
                skill_chips = ""
                for s in display_skills[:12]:
                    if s.lower() in CORE_IR_SKILLS:
                        skill_chips += f'<span class="skill-chip" style="border-color: rgba(34,197,94,0.4); color: #4ade80;">{s}</span>'
                    elif ATD_TAXONOMY.get(s.lower(), 0) >= 3:
                        skill_chips += f'<span class="skill-chip" style="border-color: rgba(139,92,246,0.4); color: #c4b5fd;">{s}</span>'
                    else:
                        skill_chips += f'<span class="skill-chip">{s}</span>'

                # Format reasoning
                display_reasoning = reasoning
                if "[Hidden Gem 💎]" in display_reasoning:
                    display_reasoning = display_reasoning.replace(
                        "[Hidden Gem 💎]", 
                        "<span class='hidden-gem-tag'>💎 Hidden Gem</span>"
                    )
                
                for kw in ["Note:", "Caveat:", "Consideration:", "Flag:", "Potential concern:"]:
                    if kw in display_reasoning:
                        display_reasoning = display_reasoning.replace(
                            kw, f"<span class='caveat-tag'>{kw}</span>"
                        )
                        
                for kw in ["Highlight:", "Key Strength:", "Bonus:"]:
                    if kw in display_reasoning:
                        display_reasoning = display_reasoning.replace(
                            kw, f"<span class='highlight-tag'>{kw}</span>"
                        )

                # Filter status badge
                status_badge = ""
                if filters:
                    reasons_str = ", ".join(filters)
                    status_badge = f'<span class="caveat-tag" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; margin-left: 0.5rem; padding: 0.15rem 0.4rem; border-radius: 6px;">Filtered: {reasons_str}</span>'
                else:
                    status_badge = f'<span class="highlight-tag" style="background: rgba(16, 185, 129, 0.15); color: #10b981; margin-left: 0.5rem; padding: 0.15rem 0.4rem; border-radius: 6px;">Passed Filters</span>'

                st.markdown(f"""
                <div class="candidate-card">
                    <div>
                        <strong style="font-size: 1.05rem;">{prof.get('anonymized_name', cid)}</strong>
                        &nbsp;&nbsp;
                        <span class="score-badge">Score: {score:.4f}</span>
                        {status_badge}
                    </div>
                    <div style="margin-top: 0.4rem; color: #94a3b8; font-size: 0.88rem;">
                        {prof.get('current_title', '')} at {prof.get('current_company', '')}
                        &nbsp;·&nbsp; {prof.get('years_of_experience', 0.0) or 0.0:.1f} yrs
                        &nbsp;·&nbsp; {f"{prof.get('location', '') or ''}, {prof.get('country', '') or ''}".strip(", ") or 'N/A'}
                    </div>
                    <div style="margin-top: 0.4rem;">{skill_chips}</div>
                    <div class="reasoning-text">{display_reasoning}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Pagination buttons ──
            if total_pages > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                pag_col1, pag_col2, pag_col3 = st.columns([1, 2, 1])
                with pag_col1:
                    if st.button("⬅️ Previous Page", disabled=(current_page == 1), key="prev_page_btn", use_container_width=True):
                        st.session_state.uploaded_page -= 1
                        st.rerun()
                with pag_col2:
                    st.markdown(f"<p style='text-align: center; font-size: 1rem; margin-top: 0.3rem;'>Page <strong>{current_page}</strong> of <strong>{total_pages}</strong></p>", unsafe_allow_html=True)
                with pag_col3:
                    if st.button("Next Page ➡️", disabled=(current_page == total_pages), key="next_page_btn", use_container_width=True):
                        st.session_state.uploaded_page += 1
                        st.rerun()

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
