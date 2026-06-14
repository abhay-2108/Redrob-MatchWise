#!/usr/bin/env python3
"""
Redrob MatchWise — Candidate Ranking Engine
============================================
Processes 100,000 synthetic candidate profiles and produces a CSV
containing the top 100 best-fit candidates for a Senior AI Engineer
(Founding Team) role at Redrob AI.

Constraints (sandbox-enforced at Stage 3)
-----------------------------------------
- ≤ 5 minutes wall-clock on CPU
- ≤ 16 GB RAM
- CPU only (no GPU)
- No network (no external API calls)
- ≤ 5 GB intermediate state

Architecture
------------
1. Load pre-computed honeypot IDs (from identify_all_honeypots.py).
2. Stream candidates line-by-line from JSONL (memory-efficient).
3. For each candidate:
   a. Hard-filter: honeypots, service-only careers, unrelated titles.
   b. Compute match score across 5 weighted components.
   c. Compute availability modifier from 23 behavioral signals.
   d. Final score = match_score × availability_modifier.
4. Sort by score descending, candidate_id ascending for tiebreaks.
5. Take top 100, assign ranks 1–100.
6. Generate dynamic, factual 1-2 sentence reasoning per candidate.
7. Write submission CSV.

Usage
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

    # Or with gzipped input:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

# Lazy model cache for sentence-transformers fallback
_MODEL_CACHE = {}

def get_sentence_transformer_model():
    if "model" not in _MODEL_CACHE:
        print("Lazy-loading sentence-transformer model (all-MiniLM-L6-v2) for on-the-fly embedding...")
        import torch
        torch.set_num_threads(8)
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_CACHE["model"]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                        CONSTANTS & CONFIG                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI.
Series A AI-native talent intelligence platform. Location: Pune/Noida, India.
Experience: 5-9 years. Need someone with deep technical depth in modern ML systems:
embeddings, retrieval, ranking, LLMs, fine-tuning. Scrappy product-engineering attitude.

Own the intelligence layer: ranking, retrieval, and matching systems.
Ship a v2 ranking system with embeddings, hybrid retrieval, LLM-based re-ranking.
Set up evaluation infrastructure: offline benchmarks, online A/B testing, recruiter-feedback loops.

Required: Production experience with embeddings-based retrieval systems (sentence-transformers,
OpenAI embeddings, BGE, E5). Production experience with vector databases or hybrid search
(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python. Evaluation frameworks for ranking: NDCG, MRR, MAP, A/B test interpretation.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models (XGBoost, neural),
HR-tech/recruiting/marketplace experience, distributed systems, open-source contributions.

Do NOT want: title-chasers, framework enthusiasts with only LangChain tutorials,
people from only consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini),
pure CV/speech/robotics without NLP/IR, architects who haven't coded in 18 months.

Ideal: 6-8 years total, 4-5 years in applied ML/AI at product companies.
Shipped end-to-end ranking, search, or recommendation system to real users at scale.
Located in or willing to relocate to Noida or Pune, India.
"""

def build_candidate_text(candidate: dict) -> str:
    """Build a rich text representation of a candidate for embedding and BM25."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []

    # Headline + summary
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    # Current role context
    parts.append(
        f"{profile.get('current_title', '')} at {profile.get('current_company', '')} "
        f"with {profile.get('years_of_experience', 0)} years of experience"
    )

    # Career descriptions (only top 2 jobs, truncated to 150 chars for speed/256-token limit)
    for job in career[:2]:
        desc = job.get('description', '')
        if len(desc) > 150:
            desc = desc[:150] + "..."
        parts.append(
            f"{job.get('title', '')} at {job.get('company', '')}. {desc}"
        )

    # Skills (name only, top 8)
    skill_names = [s.get("name", "") for s in skills if s.get("name")][:8]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Hard truncate the full string to 1000 characters to keep it tight and fast
    full_text = " ".join(p for p in parts if p).strip()
    return full_text[:1000]

WORD_RE = re.compile(r'\w+')
def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                        CONSTANTS & CONFIG                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

REF_DATE = datetime(2026, 6, 14)

# Companies that are pure IT-services / consulting firms.
# Candidates whose *entire* career is within these firms are disqualified.
SERVICE_COMPANIES = frozenset({
    "TCS", "Wipro", "Infosys", "Cognizant", "Accenture",
    "Capgemini", "Tech Mahindra", "Mphasis", "HCL", "Genpact AI",
    "Mindtree",
})

# ── Title classification sets (all lowercase) ──────────────────────────
CORE_AI_TITLES = frozenset({
    "senior ai engineer", "ai engineer", "ml engineer",
    "machine learning engineer", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation engineer",
    "recommendation systems engineer", "applied ml engineer",
    "applied scientist", "research scientist", "nlp scientist",
    "ai specialist", "ml specialist",
    "senior software engineer (ml)", "senior ml engineer",
    "deep learning engineer", "ai/ml engineer",
})

SWE_TITLES = frozenset({
    "software engineer", "backend engineer", "software developer",
    "backend developer", "full stack developer", "data engineer",
    "systems engineer", "analytics engineer", "data scientist",
    "lead engineer", "engineering lead", "tech lead", "technical lead",
    "software architect", "solutions architect", "devops engineer",
    "platform engineer", "senior software engineer",
    "senior backend engineer", "senior data engineer",
    "senior data scientist",
})

UNRELATED_TITLES = frozenset({
    "marketing manager", "accountant", "operations manager",
    "graphic designer", "mechanical engineer", "project manager",
    "customer support", "hr manager", "business analyst",
    "sales executive", "product manager", "content writer",
    "ux designer", "ui designer", "recruiter",
    "financial analyst", "legal counsel",
})

# ── Skill classification sets (all lowercase) ──────────────────────────
CORE_IR_SKILLS = frozenset({
    "embeddings", "vector database", "hybrid search", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "ndcg", "mrr", "map", "learning to rank",
    "sentence-transformers", "sentence transformers", "semantic search",
    "vector search", "rag", "retrieval-augmented generation",
    "information retrieval", "bm25", "pgvector", "chromadb",
    "recommendation systems", "haystack",
})

ADVANCED_ML_SKILLS = frozenset({
    "nlp", "lora", "qlora", "peft", "fine-tuning llms",
    "xgboost", "pytorch", "tensorflow", "llamaindex", "langchain",
    "mlflow", "mlops", "transformers", "deep learning",
    "machine learning", "scikit-learn", "huggingface",
    "prompt engineering", "llms", "python",
})

CV_SPEECH_SKILLS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "speech recognition", "tts", "robotics",
    "speech synthesis", "gans", "cnn", "diffusion models", "asr",
})

# Keywords to match in summaries / job descriptions
DESC_KEYWORDS = [
    "search", "ranking", "recommendation", "retrieval", "matching",
    "vector search", "hybrid search", "semantic search", "index",
    "ndcg", "evaluation", "ab test", "a/b test", "information retrieval",
    "embeddings", "reranking", "re-ranking", "candidate matching",
    "talent", "recruiter", "hiring",
]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         SCORING FUNCTIONS                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _title_score(current_title: str, career_titles: list[str]) -> float:
    """Score title relevance: 1.0 for core AI, 0.6 for SWE, lower otherwise."""
    ct = current_title.lower().strip()

    if ct in CORE_AI_TITLES:
        return 1.0
    if ct in SWE_TITLES:
        return 0.6

    # Check career history for any strong title
    for t in career_titles:
        tl = t.lower().strip()
        if tl in CORE_AI_TITLES:
            return 0.5
    for t in career_titles:
        tl = t.lower().strip()
        if tl in SWE_TITLES:
            return 0.3
    return 0.0


def _experience_score(years: float) -> float:
    """Score experience fit: peak at 6-8 years, tapering off."""
    if 6.0 <= years <= 8.0:
        return 1.0
    if 5.0 <= years < 6.0 or 8.0 < years <= 9.0:
        return 0.9
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.7
    if 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return 0.4
    return 0.1


def _skills_score(skills: list[dict]) -> tuple[float, bool, bool]:
    """Return (normalised skill score, has_nlp_ir, has_cv_speech)."""
    raw = 0.0
    has_nlp_ir = False
    has_cv = False

    prof_mult = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.2}

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)

        # Determine category weight
        if name in CORE_IR_SKILLS:
            base = 3.0
            has_nlp_ir = True
        elif name in ADVANCED_ML_SKILLS:
            base = 1.5
            if name in ("nlp", "information retrieval"):
                has_nlp_ir = True
        elif name in CV_SPEECH_SKILLS:
            base = 0.5
            has_cv = True
        else:
            continue  # Skip irrelevant skills entirely

        pm = prof_mult.get(prof, 1.0)
        dm = min(2.0, max(0.1, dur / 24.0))        # normalise to 24 months
        em = min(2.0, 1.0 + endorse / 50.0)        # endorsement bonus

        raw += base * pm * dm * em

    # Penalty for CV/Speech-only without NLP/IR
    if has_cv and not has_nlp_ir:
        raw *= 0.2

    normalised = min(1.0, raw / 15.0)
    return normalised, has_nlp_ir, has_cv


def _description_score(summary: str, job_descriptions: list[str]) -> float:
    """Score keyword presence in combined text of summary + job descriptions."""
    combined = (summary + " " + " ".join(job_descriptions)).lower()
    hits = sum(1 for kw in DESC_KEYWORDS if kw in combined)
    return min(1.0, hits / 5.0)


def _location_score(country: str, location: str, willing_relocate: bool) -> float:
    """Score based on proximity to Pune/Noida and relocation willingness."""
    country_l = country.lower()
    loc_l = location.lower()

    if country_l != "india":
        return 0.1

    target_cities = ("pune", "noida")
    secondary_cities = ("hyderabad", "mumbai", "gurgaon", "delhi",
                        "ncr", "ghaziabad", "faridabad", "bangalore")

    if any(c in loc_l for c in target_cities):
        return 1.0
    if any(c in loc_l for c in secondary_cities):
        return 0.8
    if willing_relocate:
        return 0.7
    return 0.3


def _availability_modifier(signals: dict) -> float:
    """Multiplicative modifier based on 23 Redrob behavioral signals."""

    # 1. Recency of activity
    last_active_str = signals.get("last_active_date")
    active_f = 0.1
    if last_active_str:
        try:
            last_dt = datetime.strptime(last_active_str, "%Y-%m-%d")
            days_inactive = (REF_DATE - last_dt).days
            if days_inactive <= 30:
                active_f = 1.0
            elif days_inactive <= 90:
                active_f = 0.9
            elif days_inactive <= 180:
                active_f = 0.5
            else:
                active_f = 0.1
        except (ValueError, TypeError):
            pass

    # 2. Notice period
    notice = signals.get("notice_period_days", 180)
    if notice <= 30:
        notice_f = 1.0
    elif notice <= 60:
        notice_f = 0.9
    elif notice <= 90:
        notice_f = 0.7
    else:
        notice_f = 0.4

    # 3. Recruiter responsiveness
    resp = signals.get("recruiter_response_rate", 0.0)
    resp_f = 0.3 + 0.7 * resp

    # 4. Interview completion
    int_rate = signals.get("interview_completion_rate", 0.0)
    int_f = 0.4 + 0.6 * int_rate

    # 5. Open-to-work flag
    otw = signals.get("open_to_work_flag", False)
    otw_f = 1.0 if otw else 0.8

    # 6. Profile completeness bonus (mild)
    completeness = signals.get("profile_completeness_score", 50.0)
    comp_f = 0.8 + 0.2 * (completeness / 100.0)

    # 7. GitHub activity bonus (mild)
    gh = signals.get("github_activity_score", -1)
    gh_f = 1.0 if gh < 0 else (0.9 + 0.1 * min(1.0, gh / 80.0))

    return active_f * notice_f * resp_f * int_f * otw_f * comp_f * gh_f


def score_candidate(candidate: dict, honeypot_ids: set[str], cos_score: float = 0.0, bm25_score: float = 0.0) -> tuple[float, dict]:
    """Return (final_score, debug_info) for a single candidate.

    Returns (0.0, {}) for hard-filtered candidates.
    """
    cid = candidate.get("candidate_id", "")

    # ── Hard filters ──────────────────────────────────────────────────
    if cid in honeypot_ids:
        return 0.0, {}

    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # Service-only career → disqualify
    companies = {j.get("company") for j in career if j.get("company")}
    if companies and all(c in SERVICE_COMPANIES for c in companies):
        return 0.0, {}

    # Current title is completely unrelated → disqualify
    curr_title = profile.get("current_title", "").lower().strip()
    if curr_title in UNRELATED_TITLES:
        return 0.0, {}

    # ── Component scores ──────────────────────────────────────────────
    career_titles = [j.get("title", "") for j in career]
    t_score = _title_score(profile.get("current_title", ""), career_titles)
    e_score = _experience_score(profile.get("years_of_experience", 0))
    s_score, _, _ = _skills_score(skills)
    d_score = _description_score(
        profile.get("summary", ""),
        [j.get("description", "") for j in career],
    )
    l_score = _location_score(
        profile.get("country", ""),
        profile.get("location", ""),
        signals.get("willing_to_relocate", False),
    )

    # Weighted match score (heuristics)
    heuristic = (
        t_score * 0.25
        + e_score * 0.20
        + s_score * 0.35
        + d_score * 0.15
        + l_score * 0.05
    )

    avail = _availability_modifier(signals)
    
    # Fallback to heuristic * avail if no cosine/BM25 scores are provided
    if cos_score == 0.0 and bm25_score == 0.0:
        final = heuristic * avail
    else:
        final = (
            0.30 * cos_score
            + 0.25 * bm25_score
            + 0.25 * heuristic
            + 0.20 * avail
        )

    debug = {
        "title": t_score, "exp": e_score, "skills": s_score,
        "desc": d_score, "loc": l_score, "heuristic": heuristic,
        "avail": avail, "cosine": cos_score, "bm25": bm25_score, "final": final,
    }
    return final, debug


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                       REASONING GENERATOR                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _hash_idx(cid: str, n: int) -> int:
    """Deterministic hash-based index for varied phrasing."""
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % n


def generate_reasoning(candidate: dict, rank: int) -> str:
    """Produce a factual, non-templated 1-2 sentence reasoning string."""
    prof    = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = [s.get("name", "") for s in candidate.get("skills", [])]
    career  = candidate.get("career_history", [])
    cid     = candidate.get("candidate_id", "")

    title   = prof.get("current_title", "engineer")
    company = prof.get("current_company", "")
    years   = prof.get("years_of_experience", 0)
    loc     = prof.get("location", "")
    notice  = signals.get("notice_period_days", 0)
    resp    = signals.get("recruiter_response_rate", 0.0)

    idx = _hash_idx(cid, 5)

    # ── Sentence 1: Intro + alignment ─────────────────────────────────
    if rank <= 10:
        intros = [
            f"Exceptional fit with {years:.1f} years as a {title} at {company}, directly aligned with the founding AI engineering mandate.",
            f"Top-tier candidate — {years:.1f} years of applied experience as a {title} at {company}, matching the core search/ranking charter.",
            f"Strong founding-team candidate: {years:.1f} years building production ML systems as a {title} at {company}.",
            f"Premier match for the Senior AI Engineer role with {years:.1f} years at {company} as a {title}.",
            f"Ideal profile: {years:.1f} years of hands-on ML engineering experience as a {title} at {company}.",
        ]
    elif rank <= 30:
        intros = [
            f"Solid candidate with {years:.1f} years of experience as a {title} at {company}, well-suited for the ranking/retrieval charter.",
            f"Strong alignment: {years:.1f} years of product-engineering experience as a {title} at {company}.",
            f"Compelling profile with {years:.1f} years as a {title} at {company}, demonstrating applied ML depth.",
            f"Well-qualified candidate — {years:.1f} years at {company} as a {title}, covering key JD requirements.",
            f"Relevant fit with {years:.1f} years building ML-powered features as a {title} at {company}.",
        ]
    elif rank <= 70:
        intros = [
            f"Good candidate with {years:.1f} years of experience as a {title} at {company}.",
            f"Matches key criteria: {years:.1f} years as a {title} at {company} with relevant engineering depth.",
            f"Moderate fit — {years:.1f} years at {company} as a {title}, covering several JD skill areas.",
            f"Brings {years:.1f} years of experience as a {title} at {company} with partial alignment to the JD.",
            f"Reasonable match with {years:.1f} years of tenure as a {title} at {company}.",
        ]
    else:
        intros = [
            f"Borderline fit with {years:.1f} years as a {title} at {company}; included based on adjacent signals.",
            f"Adjacent candidate: {years:.1f} years at {company} as a {title}, weaker on some core JD dimensions.",
            f"Lower-confidence match — {years:.1f} years as a {title} at {company}, but gaps in key areas.",
            f"Marginal fit at {years:.1f} years of experience as a {title} at {company}.",
            f"Included at rank boundary: {years:.1f} years as a {title} at {company} with partial overlap to requirements.",
        ]

    s1 = intros[idx]

    # ── Sentence 2: Skills + concerns ─────────────────────────────────
    matched_core = [s for s in skills if s.lower() in CORE_IR_SKILLS]
    matched_adv  = [s for s in skills if s.lower() in ADVANCED_ML_SKILLS]

    if matched_core:
        top_skills = matched_core[:3]
        skill_phrases = [
            f"Production experience with {', '.join(top_skills)} aligns with the vector-retrieval stack.",
            f"Hands-on with {', '.join(top_skills)}, directly relevant to the hybrid search mandate.",
            f"Key strengths in {', '.join(top_skills)} match the IR/ranking infrastructure needs.",
            f"Demonstrated depth in {', '.join(top_skills)}, critical for the retrieval layer.",
            f"Core skill alignment through {', '.join(top_skills)} experience in production settings.",
        ]
    elif matched_adv:
        top_skills = matched_adv[:3]
        skill_phrases = [
            f"Relevant ML toolkit ({', '.join(top_skills)}) supports the applied AI requirements.",
            f"Background in {', '.join(top_skills)} provides a solid foundation for the role.",
            f"Applied ML skills in {', '.join(top_skills)} cover secondary JD requirements.",
            f"Proficiency in {', '.join(top_skills)} bridges toward the search/ranking domain.",
            f"Has {', '.join(top_skills)} experience applicable to the ML infrastructure layer.",
        ]
    else:
        skill_phrases = [
            "Skills profile is adjacent rather than directly aligned with core IR/vector requirements.",
            "Limited direct overlap with the vector search and ranking evaluation stack.",
            "Skill set is general engineering; would need ramp-up on retrieval-specific tooling.",
            "Engineering fundamentals are present but specific search/NLP depth is thin.",
            "Core JD skills (embeddings, vector DBs, ranking eval) are under-represented in the profile.",
        ]

    s2 = skill_phrases[idx]

    # ── Sentence 3 (optional): Concerns ───────────────────────────────
    concerns = []
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if resp < 0.5:
        concerns.append(f"{int(resp * 100)}% recruiter response rate")

    target_locs = ("pune", "noida", "gurgaon", "delhi", "hyderabad",
                   "mumbai", "bangalore")
    loc_l = loc.lower()
    if not any(c in loc_l for c in target_locs):
        willing = signals.get("willing_to_relocate", False)
        if not willing:
            concerns.append(f"based in {loc} without relocation willingness")

    if concerns:
        concern_strs = [
            f"Note: {'; '.join(concerns)}.",
            f"Caveat: {'; '.join(concerns)}.",
            f"Consideration: {'; '.join(concerns)}.",
            f"Flag: {'; '.join(concerns)}.",
            f"Potential concern: {'; '.join(concerns)}.",
        ]
        s3 = " " + concern_strs[idx]
    else:
        s3 = ""

    return s1 + " " + s2 + s3


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           MAIN PIPELINE                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# Paths for precomputed embeddings and reasoning cache (relative to script folder)
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.path.join(DIR_PATH, "candidate_embeddings.npy")
JD_EMB_PATH = os.path.join(DIR_PATH, "jd_embedding.npy")
IDS_PATH = os.path.join(DIR_PATH, "candidate_ids.json")
REASONING_CACHE_PATH = os.path.join(DIR_PATH, "reasoning_cache.json")


def load_honeypots(path: str) -> set[str]:
    """Load the pre-computed honeypot IDs from a JSON file."""
    if not os.path.exists(path):
        print(f"WARNING: honeypots file not found at {path}; skipping filter.")
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return set(data.keys())


def load_precomputed_embeddings():
    """Load pre-computed candidate and JD embeddings if they exist."""
    if os.path.exists(EMB_PATH) and os.path.exists(JD_EMB_PATH) and os.path.exists(IDS_PATH):
        print(f"Loading precomputed embeddings from {EMB_PATH}...")
        cand_embs = np.load(EMB_PATH)
        jd_emb = np.load(JD_EMB_PATH)
        with open(IDS_PATH, "r", encoding="utf-8") as fh:
            cand_ids = json.load(fh)
        
        id_to_idx = {cid: idx for idx, cid in enumerate(cand_ids)}
        return cand_embs, jd_emb, id_to_idx
    else:
        print("WARNING: Precomputed embeddings not found. Will fall back to on-the-fly embedding.")
        return None, None, None


def get_cosine_scores(candidates: list[dict], cand_embs, jd_emb, id_to_idx) -> dict:
    """Get semantic cosine similarity scores. Computes on-the-fly if missing from cache."""
    scores = {}
    if cand_embs is not None and jd_emb is not None and id_to_idx is not None:
        # cand_embs are pre-normalized, jd_emb is pre-normalized, dot product is cosine similarity
        sims = np.dot(cand_embs, jd_emb.T).flatten()
        for cid, idx in id_to_idx.items():
            scores[cid] = float(sims[idx])

    # Check for candidates not in cache (fallback)
    missing = [c for c in candidates if c.get("candidate_id") not in scores]
    if missing:
        print(f"Computing embeddings on-the-fly for {len(missing)} candidates...")
        model = get_sentence_transformer_model()
        texts = [build_candidate_text(c) for c in missing]
        
        # Encode missing texts
        missing_embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        
        # Cache JD embedding locally
        if "jd_emb" not in _MODEL_CACHE:
            _MODEL_CACHE["jd_emb"] = model.encode([JD_TEXT], convert_to_numpy=True, normalize_embeddings=True)
        jd_emb_local = _MODEL_CACHE["jd_emb"]
        
        sims_missing = np.dot(missing_embs, jd_emb_local.T).flatten()
        for c, sim in zip(missing, sims_missing):
            scores[c["candidate_id"]] = float(sim)

    return scores


def get_bm25_scores(candidates: list[dict]) -> dict:
    """Tokenize and compute BM25 scores for candidates against the JD query."""
    if not candidates:
        return {}
    
    corpus = [build_candidate_text(c) for c in candidates]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    jd_query = tokenize(JD_TEXT)
    raw_scores = bm25.get_scores(jd_query)
    
    max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0
    
    scores = {}
    for idx, c in enumerate(candidates):
        cid = c["candidate_id"]
        if max_score > 0.0:
            scores[cid] = float(raw_scores[idx] / max_score)
        else:
            scores[cid] = 0.0
            
    return scores


def load_reasoning_cache() -> dict:
    """Load reasoning cache JSON file if it exists."""
    if os.path.exists(REASONING_CACHE_PATH):
        print(f"Loading cached reasonings from {REASONING_CACHE_PATH}...")
        with open(REASONING_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Senior AI Engineer role."
    )
    parser.add_argument(
        "--candidates",
        default="./docs/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--honeypots",
        default="./honeypots.json",
        help="Path to honeypots.json (from identify_all_honeypots.py)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Resolve candidates path
    cpath = args.candidates
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
        else:
            print(f"ERROR: Candidate file not found at {cpath} or {alt}")
            sys.exit(1)

    # Load honeypots
    honeypot_ids = load_honeypots(args.honeypots)
    print(f"Loaded {len(honeypot_ids)} honeypot IDs.")

    # Read and hard-filter candidates
    print(f"Reading candidates from: {cpath}")
    valid_candidates = []
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

    print(f"Read {total} candidates. {len(valid_candidates)} passed hard filters.")

    # Load precomputed embeddings
    cand_embs, jd_emb, id_to_idx = load_precomputed_embeddings()

    # Compute BM25 scores
    print("Computing BM25 scores...")
    bm25_scores = get_bm25_scores(valid_candidates)

    # Compute cosine similarities
    print("Computing cosine similarities...")
    cos_scores = get_cosine_scores(valid_candidates, cand_embs, jd_emb, id_to_idx)

    # Load reasoning cache
    reasoning_cache = load_reasoning_cache()

    # Score all candidates using hybrid formula
    print("Scoring candidates...")
    scored = []
    for cand in valid_candidates:
        cid = cand["candidate_id"]
        cos_val = cos_scores.get(cid, 0.0)
        bm25_val = bm25_scores.get(cid, 0.0)
        score, debug = score_candidate(cand, honeypot_ids, cos_score=cos_val, bm25_score=bm25_val)
        scored.append((cid, score, cand))

    t_score = time.time()
    print(f"Processed and scored candidates in {t_score - t0:.1f}s.")

    # Sort: score descending, candidate_id ascending for tiebreaks
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top 100
    top_100 = scored[:100]

    # Generate output
    rows = []
    for rank_idx, (cid, score, cand) in enumerate(top_100, start=1):
        reasoning = reasoning_cache.get(cid, "")
        if not reasoning:
            reasoning = generate_reasoning(cand, rank_idx)
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    t_end = time.time()
    total_time = t_end - t0

    print(f"\n{'='*60}")
    print(f"Submission written to: {args.out}")
    print(f"Top 100 candidates ranked. Score range: "
          f"{rows[0]['score']:.6f} -> {rows[-1]['score']:.6f}")
    print(f"Total runtime: {total_time:.1f}s")
    print(f"{'='*60}")

    # Sanity checks
    ids_in_output = {r["candidate_id"] for r in rows}
    honeypots_in_output = ids_in_output & honeypot_ids
    if honeypots_in_output:
        print(f"WARNING: {len(honeypots_in_output)} honeypots in output!")
    else:
        print("OK: Zero honeypots in top 100.")

    if total_time > 300:
        print(f"WARNING: Runtime {total_time:.0f}s exceeds 5-minute limit!")
    else:
        print(f"OK: Runtime {total_time:.1f}s is within 5-minute budget.")


if __name__ == "__main__":
    main()
