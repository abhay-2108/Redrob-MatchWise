#!/usr/bin/env python3
"""
build_features.py — Offline Feature Engineering
=================================================
Reads all candidates from a JSONL file and extracts ~40 numerical
features per candidate into a compressed numpy matrix.

This runs OFFLINE before submission (no time limit).
The output `precomputed_features.npz` is loaded at runtime by rank_v2.py.

Usage:
    python build_features.py --candidates <path/to/candidates.jsonl>
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

# Lazy-loaded SentenceTransformer to avoid paying 80MB model load cost at import time.
# It is loaded only when batch semantic encoding actually runs (in main()).
_embedder = None
_JD_EMB = None
_JD_TEXT = (
    "Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning, "
    "sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS, "
    "evaluation NDCG MRR MAP A/B testing, Python, production systems, "
    "startup product company, Pune Noida India"
)

def _ensure_embedder():
    """Lazy-load the SentenceTransformer model and JD embedding on first use."""
    global _embedder, _JD_EMB
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading SentenceTransformer model (lazy)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            _JD_EMB = _embedder.encode(_JD_TEXT)
        except Exception as e:
            print(f"WARNING: sentence-transformers unavailable ({e}). Semantic feature will be 0.")
            _embedder = None
            _JD_EMB = None
    return _embedder, _JD_EMB


# ─── Import the existing taxonomy from rank.py ────────────────────────
# We reuse the hand-crafted ATD taxonomy because it's genuinely good.
from src.rank import (
    ATD_TAXONOMY,
    ATD_DESC_KEYWORDS,
    CORE_AI_TITLES,
    SWE_TITLES,
    UNRELATED_TITLES,
    SERVICE_COMPANIES,
    DEVOPS_SKILLS,
    BACKEND_SKILLS,
    PRODUCT_INDUSTRIES,
    HR_TECH_INDUSTRIES,
    CV_SPEECH_KEYWORDS,
    EVAL_KEYWORDS,
    CORE_IR_SKILLS,
    PROF_MULTIPLIER,
    canonicalize_skill,
    REF_DATE,
    compute_atd,
    compute_hea
)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     FEATURE DEFINITIONS                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# Feature names in exact order — this defines the column layout of the matrix.
FEATURE_NAMES = [
    # ── A. Technical Depth (11 features) ──
    "max_atd_level",             # 0-4: highest taxonomy level reached
    "l4_skill_count",            # count of Level 4 skills
    "l3_skill_count",            # count of Level 3 skills
    "l2_skill_count",            # count of Level 2 skills
    "l1_skill_count",            # count of Level 1 skills
    "core_ir_match_ratio",       # fraction of CORE_IR_SKILLS the candidate has
    "advanced_skill_months",     # total duration_months across L3+L4 skills
    "python_months",             # duration_months for Python specifically
    "eval_framework_signal",     # binary: has NDCG/MRR/MAP/AB-test skills
    "skill_depth_score",         # weighted sum: L4*4+L3*3+L2*2+L1*1, normalized
    "assessment_avg",            # mean of skill_assessment_scores

    # ── B. Career Trajectory (10 features) ──
    "years_experience",          # from profile.years_of_experience
    "exp_sweet_spot",            # Gaussian centered at 7 years
    "product_company_ratio",     # fraction of career at product companies
    "career_momentum",           # upward trajectory score
    "avg_tenure_months",         # mean tenure per job
    "current_role_relevance",    # how well current title matches target
    "has_shipped_system",        # binary: production deployment keywords
    "startup_experience_count",  # count of jobs at companies sized 1-200
    "career_consistency",        # all roles in tech vs random pivots
    "total_career_months",       # sum of all job durations

    # ── C. Behavioral Signals (12 features) ──
    "github_activity",           # 0-100 score
    "profile_completeness",      # 0-100 score
    "recency_days",              # days since last_active_date
    "is_active_recent",          # binary: active within last 90 days
    "recruiter_response_rate",   # 0-1
    "notice_period_days",        # raw days
    "willing_to_relocate",       # binary
    "open_to_work",              # binary
    "interview_completion",      # 0-1
    "saved_by_recruiters",       # count
    "verification_score",        # sum of verified_email+phone+linkedin
    "connection_count",          # raw count

    # ── D. Red Flag / Honeypot Detection (6 features) ──
    "timeline_impossible",       # binary: overlapping stints > 90 days
    "skill_inflation",           # count of "expert" with 0 duration_months
    "title_skill_mismatch",      # binary: non-tech title + many AI skills
    "service_company_only",      # binary: entire career at TCS/Wipro/etc
    "cv_speech_trap",            # binary: heavy CV/Speech, zero NLP/IR
    "ghost_candidate",           # binary: inactive >180 days AND resp <5%
    "is_india",                  # binary: location country is India
    "is_target_city",            # binary: Pune, Noida, Delhi, NCR

    # ── E. Singularity Matrix Integration (3 features) ──
    "exact_atd",                 # Precise ATD score from rank.py
    "exact_hea",                 # Precise HEA multiplier from rank.py
    "singularity_score",         # Raw (ATD^1.5) * HEA score
    
    # ── F. Semantic Features (1 feature) ──
    "semantic_jd_similarity",    # Cosine similarity to JD

    # ── G. Deeper NLP Regex Features (6 features) ──
    "impact_metrics",            # Count of quantified impact statements
    "scale_signals",             # Count of scale / throughput statements
    "ownership_signals",         # Count of ownership / build signals
    "leadership_signals",        # Count of leadership / mentoring signals
    "achievement_count",         # Sum of impact + scale + ownership signals
    "description_verbosity",     # Total character length of career descriptions
]

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 51, f"Expected 51 features, got {NUM_FEATURES}"

# Wider regex: captures quantified impact statements like "drove a 22% productivity gain"
IMPACT_METRIC_RE = re.compile(r"""(?ix)
    (improved|increased|reduced|decreased|boosted|drove|
     cut|saved|grew|optimized|delivered|scaled|accelerated)
    .{0,40} \d+[%x]
""")
# Wider regex: captures data volume and scale statements like "processing ~500GB daily"
SCALE_SIGNAL_RE = re.compile(r"""(?ix)
    (millions?|billions?|thousands?|petabytes?|terabytes?|gigabytes?)
    \s+ (of|per|in|across) |
    (high.throughput|large.scale|at.scale|enterprise.scale) |
    processing \s+ [~\d,]+ [KMGT]?B
""")
OWNERSHIP_SIGNAL_RE = re.compile(r"(?i)(built from scratch|architected|created|designed)")
LEADERSHIP_SIGNAL_RE = re.compile(r"(?i)(led( a)? team|managed|head of|mentored)")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     FEATURE EXTRACTION                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def extract_features(cand: dict) -> np.ndarray:
    """Extract all features for a single candidate.
    
    Returns a 1D numpy array of shape (NUM_FEATURES,).
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    feats = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    # ── A. Technical Depth Features ──────────────────────────────────
    max_level = 0
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_l3l4_months = 0.0
    python_months = 0.0
    has_eval = False
    weighted_depth = 0.0
    skill_names_lower = set()
    
    for s in skills:
        raw_name = s.get("name", "")
        name = canonicalize_skill(raw_name)
        name_lower = raw_name.lower().strip()
        skill_names_lower.add(name_lower)
        level = ATD_TAXONOMY.get(name, 0)
        dur = s.get("duration_months", 0)
        prof = s.get("proficiency", "beginner")
        pm = PROF_MULTIPLIER.get(prof, 0.5)
        
        if level > 0 and dur > 0:
            level_counts[level] += 1
            if level > max_level:
                max_level = level
            weighted_depth += level * pm * min(2.0, dur / 24.0)
            if level >= 3:
                total_l3l4_months += dur
        
        if name_lower in ("python",) or name == "python":
            python_months = max(python_months, dur)
        
        if name_lower in EVAL_KEYWORDS or name in EVAL_KEYWORDS:
            has_eval = True
    
    # Also scan career descriptions
    career_descriptions = [j.get("description", "") or "" for j in career]
    career_text_raw = " ".join(career_descriptions)
    career_text = career_text_raw.lower()
    for keyword, level in ATD_DESC_KEYWORDS.items():
        if keyword in career_text:
            if level > max_level:
                max_level = level
            level_counts[min(level, 4)] += 1
    
    if not has_eval:
        if any(kw in career_text for kw in EVAL_KEYWORDS):
            has_eval = True
    
    # Core IR match ratio
    ir_match_count = len(skill_names_lower & CORE_IR_SKILLS)
    core_ir_ratio = ir_match_count / max(len(CORE_IR_SKILLS), 1)
    
    # Assessment average
    assessments = signals.get("skill_assessment_scores", {})
    if assessments and isinstance(assessments, dict):
        assess_vals = [v for v in assessments.values() if isinstance(v, (int, float))]
        assess_avg = sum(assess_vals) / len(assess_vals) if assess_vals else 0.0
    else:
        assess_avg = 0.0
    
    total_skills = sum(level_counts.values())
    norm_depth = min(1.0, weighted_depth / 10.0) if total_skills > 0 else 0.0
    
    feats[0] = max_level
    feats[1] = level_counts[4]
    feats[2] = level_counts[3]
    feats[3] = level_counts[2]
    feats[4] = level_counts[1]
    feats[5] = core_ir_ratio
    feats[6] = min(total_l3l4_months, 240.0)  # cap at 20 years
    feats[7] = min(python_months, 240.0)
    feats[8] = 1.0 if has_eval else 0.0
    feats[9] = norm_depth
    feats[10] = assess_avg / 100.0  # normalize to 0-1
    
    # ── B. Career Trajectory Features ────────────────────────────────
    years = profile.get("years_of_experience", 0)
    exp_sweet = math.exp(-((years - 7.0) ** 2) / 18.0)  # Gaussian peak at 7
    
    product_count = 0
    total_jobs = max(len(career), 1)
    small_co_count = 0
    has_shipped = False
    tech_role_count = 0
    total_duration = 0
    
    for j in career:
        industry = j.get("industry", "").lower()
        size = j.get("company_size", "")
        title = j.get("title", "").lower()
        desc = j.get("description", "").lower()
        dur = j.get("duration_months", 0)
        total_duration += dur
        
        if industry in PRODUCT_INDUSTRIES:
            product_count += 1
        if size in ("1-10", "11-50", "51-200"):
            small_co_count += 1
        if any(kw in desc for kw in ("shipped", "deployed", "production", "scale", "millions", "serving")):
            has_shipped = True
        if any(kw in title for kw in ("engineer", "developer", "scientist", "ml", "ai", "data", "research")):
            tech_role_count += 1
    
    avg_tenure = total_duration / total_jobs if total_jobs > 0 else 0
    product_ratio = product_count / total_jobs
    career_consistency = tech_role_count / total_jobs if total_jobs > 0 else 0
    
    # Career momentum: did company sizes grow? (simple heuristic)
    size_order = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
                  "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8}
    sizes = [size_order.get(j.get("company_size", ""), 4) for j in career if j.get("company_size")]
    if len(sizes) >= 2:
        # Positive momentum = moving to larger companies (or staying)
        momentum = sum(1 for i in range(len(sizes)-1) if sizes[i+1] >= sizes[i]) / (len(sizes)-1)
    else:
        momentum = 0.5  # neutral
    
    # Current role relevance
    curr_title = profile.get("current_title", "").lower().strip()
    if curr_title in CORE_AI_TITLES:
        role_relevance = 1.0
    elif curr_title in SWE_TITLES:
        role_relevance = 0.6
    elif curr_title in UNRELATED_TITLES:
        role_relevance = 0.0
    else:
        role_relevance = 0.3
    
    feats[11] = min(years, 30.0) / 30.0  # normalize
    feats[12] = exp_sweet
    feats[13] = product_ratio
    feats[14] = momentum
    feats[15] = min(avg_tenure, 120.0) / 120.0  # normalize
    feats[16] = role_relevance
    feats[17] = 1.0 if has_shipped else 0.0
    feats[18] = min(small_co_count, 5) / 5.0  # normalize
    feats[19] = career_consistency
    feats[20] = min(total_duration, 360.0) / 360.0  # normalize
    
    # ── C. Behavioral Signal Features ────────────────────────────────
    gh = signals.get("github_activity_score", -1)
    feats[21] = max(gh, 0.0) / 100.0  # normalize, -1 -> 0
    feats[22] = signals.get("profile_completeness_score", 0.0) / 100.0
    
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (REF_DATE - last_dt).days
            feats[23] = min(max(days_inactive, 0), 365) / 365.0  # normalize
            feats[24] = 1.0 if days_inactive <= 90 else 0.0
        except (ValueError, TypeError):
            feats[23] = 1.0  # unknown = assume inactive
            feats[24] = 0.0
    else:
        feats[23] = 1.0
        feats[24] = 0.0
    
    feats[25] = signals.get("recruiter_response_rate", 0.0)
    feats[26] = min(signals.get("notice_period_days", 180), 180) / 180.0
    feats[27] = 1.0 if signals.get("willing_to_relocate", False) else 0.0
    feats[28] = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    feats[29] = signals.get("interview_completion_rate", 0.0)
    feats[30] = min(signals.get("saved_by_recruiters_30d", 0), 50) / 50.0
    
    verified = (
        (1 if signals.get("verified_email", False) else 0) +
        (1 if signals.get("verified_phone", False) else 0) +
        (1 if signals.get("linkedin_connected", False) else 0)
    )
    feats[31] = verified / 3.0
    feats[32] = min(signals.get("connection_count", 0), 500) / 500.0
    
    # ── D. Red Flag / Honeypot Detection Features ────────────────────
    # Timeline overlap detection
    stints = []
    for j in career:
        start_str = j.get("start_date")
        end_str = j.get("end_date")
        if start_str:
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                end_dt = datetime.strptime(end_str, "%Y-%m-%d") if end_str else REF_DATE
                stints.append((start_dt, end_dt))
            except (ValueError, TypeError):
                pass
    stints.sort()
    timeline_clash = 0.0
    for i in range(len(stints) - 1):
        if stints[i+1][0] < stints[i][1]:
            overlap_days = (stints[i][1] - stints[i+1][0]).days
            if overlap_days > 90:
                timeline_clash = 1.0
                break
    feats[33] = timeline_clash
    
    # Skill inflation: "expert" with 0 duration
    inflation_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    feats[34] = min(inflation_count, 10) / 10.0
    
    # Title-skill mismatch: non-tech title but many AI skills
    ai_skill_count = sum(1 for s in skills if ATD_TAXONOMY.get(canonicalize_skill(s.get("name", "")), 0) >= 2)
    is_non_tech_title = curr_title in UNRELATED_TITLES
    feats[35] = 1.0 if (is_non_tech_title and ai_skill_count > 5) else 0.0
    
    # Service company only
    companies = {j.get("company") for j in career if j.get("company")}
    feats[36] = 1.0 if (companies and all(c in SERVICE_COMPANIES for c in companies)) else 0.0
    
    # CV/Speech trap
    cv_speech_count = len(skill_names_lower & CV_SPEECH_KEYWORDS)
    ir_count = sum(1 for s in skills if ATD_TAXONOMY.get(canonicalize_skill(s.get("name", "")), 0) >= 3)
    feats[37] = 1.0 if (cv_speech_count >= 2 and ir_count == 0) else 0.0
    
    # Ghost candidate
    days_inactive_raw = feats[23] * 365  # un-normalize
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    feats[38] = 1.0 if (days_inactive_raw > 180 and resp_rate <= 0.05) else 0.0
    
    # ── E. Location Features ─────────────────────────────────────────
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    feats[39] = 1.0 if country == "india" else 0.0
    target_cities = ("pune", "noida", "bangalore", "hyderabad", "mumbai",
                     "delhi", "gurgaon", "ncr", "ghaziabad")
    feats[40] = 1.0 if any(c in location for c in target_cities) else 0.0
    # ── E. Singularity Matrix Integration ──────────────────────────────
    atd = compute_atd(skills, career)
    hea = compute_hea(cand)
    singularity = (atd ** 1.5) * hea
    
    feats[41] = atd
    feats[42] = hea
    feats[43] = singularity
    
    # ── F. Semantic Features ───────────────────────────────────────────
    text_parts = [profile.get("headline", ""), profile.get("summary", "")]
    text_parts.append(f"Skills: {', '.join(s.get('name', '') for s in skills[:15])}")
    if career:
        text_parts.append(f"Recent work: {career[0].get('description', '')[:200]}")
    cand_text = " ".join(filter(None, text_parts))[:512]
    cand["_semantic_text"] = cand_text
    
    feats[44] = 0.0  # Will be overwritten in main() via batching

    # ── G. Deeper NLP Regex Features ───────────────────────────────────────
    impact_count = len(IMPACT_METRIC_RE.findall(career_text_raw))
    scale_count = len(SCALE_SIGNAL_RE.findall(career_text_raw))
    ownership_count = len(OWNERSHIP_SIGNAL_RE.findall(career_text_raw))
    leadership_count = len(LEADERSHIP_SIGNAL_RE.findall(career_text_raw))

    feats[45] = impact_count
    feats[46] = scale_count
    feats[47] = ownership_count
    feats[48] = leadership_count
    feats[49] = impact_count + scale_count + ownership_count
    feats[50] = sum(len(desc) for desc in career_descriptions)

    return feats


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           MAIN                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Build feature matrix from candidate data")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates JSONL file (.jsonl or .jsonl.gz)")
    parser.add_argument("--out", default="./artifacts/precomputed_features.npz",
                        help="Output path for the feature matrix")
    args = parser.parse_args()

    t0 = time.time()

    # Resolve path
    cpath = args.candidates
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
        else:
            print(f"ERROR: Candidate file not found at {cpath} or {alt}")
            sys.exit(1)

    print("=" * 60)
    print("  FEATURE ENGINEERING — Redrob MatchWise v2")
    print(f"  Extracting {NUM_FEATURES} features per candidate")
    print("=" * 60)

    # Stream and extract features
    candidate_ids = []
    feature_rows = []
    semantic_texts = []
    total = 0

    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total += 1
            
            cid = cand.get("candidate_id", "")
            candidate_ids.append(cid)
            
            feats = extract_features(cand)
            feature_rows.append(feats)
            semantic_texts.append(cand.get("_semantic_text", ""))
            
            if total % 20000 == 0:
                print(f"  Processed {total} candidates...")

    # Convert to numpy
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    candidate_id_array = np.array(candidate_ids, dtype=object)

    # ── Batch encode semantic features ─────────────────────────────────
    emb, jd = _ensure_embedder()
    if emb is not None and jd is not None:
        print(f"  Batch encoding semantic features for {total} candidates...")
        all_embs = emb.encode(semantic_texts, batch_size=256, show_progress_bar=True)
        norms = np.linalg.norm(all_embs, axis=1)
        norms[norms == 0] = 1e-9
        sims = np.dot(all_embs, jd) / (norms * np.linalg.norm(jd))
        feature_matrix[:, 44] = sims

    # Sanity checks
    assert feature_matrix.shape == (total, NUM_FEATURES), \
        f"Shape mismatch: {feature_matrix.shape} vs ({total}, {NUM_FEATURES})"

    # Check for NaN/Inf
    nan_count = np.isnan(feature_matrix).sum()
    inf_count = np.isinf(feature_matrix).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: {nan_count} NaN and {inf_count} Inf values detected. Replacing with 0.")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # Save
    np.savez_compressed(
        args.out,
        features=feature_matrix,
        candidate_ids=candidate_id_array,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )

    t_end = time.time()
    file_size_mb = os.path.getsize(args.out) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"  Processed {total} candidates in {t_end - t0:.1f}s")
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Saved to: {args.out} ({file_size_mb:.1f} MB)")
    print(f"{'=' * 60}")

    # Print feature statistics
    print(f"\n  Feature Statistics:")
    print(f"  {'Feature':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*66}")
    for i, name in enumerate(FEATURE_NAMES):
        col = feature_matrix[:, i]
        print(f"  {name:<30} {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f}")


if __name__ == "__main__":
    main()
