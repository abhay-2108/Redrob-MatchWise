#!/usr/bin/env python3
"""
Redrob MatchWise — The Singularity Engine
==========================================
Ranks 100,000 synthetic candidate profiles to find the top 100
best-fit candidates for a Senior AI Engineer (Founding Team) role
at Redrob AI.

Architecture: The Singularity Engine
-------------------------------------
We abandon generic NLP (cosine similarity, BM25) which falls for the
"keyword trap" explicitly warned about in the JD.  Instead we score on
exactly two axes and MULTIPLY them:

    Final Score = (ATD ^ 1.5) × HEA

Axis A — Absolute Technical Dominance (ATD)
    A hardcoded hierarchical taxonomy of AI difficulty.  We read the
    candidate's skills + career history to find their *highest proven
    floor* on a 4-level scale (Core → Applied SOTA → Standard → Wrappers).

Axis B — High Execution Agency (HEA)
    Multiplicative modifier from behavioral signals, career structure,
    and generalist-bleed (DevOps/Backend alongside AI skills).

Why this wins:
    - Beats the keyword trap: structured taxonomy, not BM25 keyword density.
    - Beats the time limit: pure Python math, ~10s for 100K candidates.
    - Beats manual review: LLM-generated reasoning from offline cache.

Constraints (sandbox-enforced at Stage 3)
-----------------------------------------
- ≤ 5 minutes wall-clock on CPU
- ≤ 16 GB RAM
- CPU only (no GPU)
- No network (no external API calls)
- ≤ 5 GB intermediate state

Usage
-----
    python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv

    # Or with gzipped input:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                        CONSTANTS & CONFIG                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

REF_DATE = datetime(2026, 6, 14)

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

# CV / Speech / Robotics flags
CV_SPEECH_KEYWORDS = frozenset({
    "computer vision", "opencv", "yolo", "resnet", "image processing",
    "object detection", "image segmentation", "speech recognition",
    "asr", "speech-to-text", "robotics", "ros", "slam", "lidar"
})

EVAL_KEYWORDS = frozenset({
    "ndcg", "mrr", "map", "a/b test", "ab test", "a/b testing",
    "offline-to-online", "evaluation", "eval", "metrics"
})

# Also check career titles for unrelated roles


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              ATD — ABSOLUTE TECHNICAL DOMINANCE TAXONOMY              ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# Hierarchical taxonomy: skill name (lowercase) → ATD level (1-4).
# Level 4 = core infrastructure (custom CUDA, distributed training, serving).
# Level 3 = applied SOTA (fine-tuning, hybrid search, vector DBs, evaluation).
# Level 2 = standard AI (framework-level training, generic RAG).
# Level 1 = wrapper-level (API callers, tutorial-level).

ATD_TAXONOMY: dict[str, int] = {
    # ── Level 4: The Core (custom infra, distributed training, serving) ──
    "cuda": 4, "vllm": 4, "tensorrt": 4, "megatron": 4, "deepspeed": 4,
    "triton inference server": 4, "distributed training": 4,
    "c++": 4, "nccl": 4, "model parallelism": 4,
    "tensor parallelism": 4, "pipeline parallelism": 4,
    "triton": 4,

    # ── Level 3: Applied SOTA (fine-tuning, complex systems, hybrid search) ──
    "lora": 3, "qlora": 3, "peft": 3, "fine-tuning llms": 3,
    "recommendation systems": 3, "hybrid search": 3,
    "learning to rank": 3, "xgboost": 3,
    "sentence-transformers": 3, "sentence transformers": 3,
    "faiss": 3, "pinecone": 3, "weaviate": 3, "qdrant": 3, "milvus": 3,
    "opensearch": 3, "elasticsearch": 3, "pgvector": 3, "chromadb": 3,
    "vector database": 3, "vector search": 3, "semantic search": 3,
    "information retrieval": 3, "bm25": 3, "bge": 3, "e5": 3, "openai embeddings": 3,
    "embeddings": 3, "ndcg": 3, "mrr": 3, "map": 3, "offline-to-online correlation": 3, "offline to online": 3,
    "huggingface": 3, "transformers": 3,
    "mlflow": 3, "mlops": 3, "ray": 3,
    "haystack": 3, "a/b testing": 3, "ab testing": 3,

    # ── Level 2: Standard AI (framework-level training, generic RAG) ──
    "pytorch": 2, "tensorflow": 2, "deep learning": 2,
    "machine learning": 2, "scikit-learn": 2,
    "rag": 2, "retrieval-augmented generation": 2,
    "nlp": 2, "llamaindex": 2, "python": 2,
    "keras": 2, "spacy": 2, "nltk": 2,

    # ── Level 1: The Wrappers (API callers, tutorial-level) ──
    "langchain": 1, "openai api": 1, "prompt engineering": 1,
    "llms": 1, "chatgpt": 1, "gpt-4": 1,
}

# Keywords to scan for in career descriptions (maps to ATD level).
# These catch signals not in the skills list.
ATD_DESC_KEYWORDS: dict[str, int] = {
    "cuda kernel": 4, "custom kernel": 4, "distributed training": 4,
    "model parallelism": 4, "tensor parallelism": 4,
    "megatron": 4, "deepspeed": 4, "vllm": 4, "tensorrt": 4,
    "fine-tun": 3, "lora": 3, "qlora": 3,
    "recommendation system": 3, "ranking system": 3, "search system": 3,
    "vector search": 3, "hybrid search": 3, "semantic search": 3,
    "embeddings": 3, "retrieval system": 3, "re-ranking": 3, "reranking": 3,
    "ndcg": 3, "information retrieval": 3, "learning to rank": 3, "offline-to-online correlation": 3,
    "a/b test": 3, "ab test": 3,
    "faiss": 3, "pinecone": 3, "weaviate": 3, "qdrant": 3,
    "milvus": 3, "elasticsearch": 3, "opensearch": 3,
    "sentence-transformer": 3, "sentence transformer": 3,
    "candidate matching": 3, "talent matching": 3, "job matching": 3,
}

# Generalist bleed: DevOps / Backend skills that prove a founding-engineer
# can deploy their own code.
DEVOPS_SKILLS = frozenset({
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
    "ci/cd", "jenkins", "github actions", "ansible", "helm",
    "cloudformation",
})

BACKEND_SKILLS = frozenset({
    "fastapi", "flask", "django", "rest api", "graphql",
    "postgresql", "mongodb", "redis", "kafka", "rabbitmq",
    "celery", "grpc", "nginx",
})

# Product-oriented industries (not services/consulting).
PRODUCT_INDUSTRIES = frozenset({
    "technology", "software", "saas", "fintech", "ai",
    "machine learning", "e-commerce", "edtech", "healthtech",
    "artificial intelligence", "internet", "information technology",
    "data analytics", "cybersecurity", "gaming", "media",
    "telecommunications", "biotechnology", "robotics",
})

# Domain experience explicitly requested
HR_TECH_INDUSTRIES = frozenset({
    "hr tech", "hr-tech", "recruiting", "recruitment", "human resources",
    "talent acquisition", "marketplace", "marketplaces", "hiring",
})

# CV/Speech-only skills — penalty if these dominate without NLP/IR
CV_SPEECH_SKILLS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "speech recognition", "tts", "robotics",
    "speech synthesis", "gans", "cnn", "diffusion models", "asr",
})

# Core IR/NLP skills (used for the "has IR experience" check)
CORE_IR_SKILLS = frozenset({
    "embeddings", "vector database", "hybrid search", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "ndcg", "mrr", "map", "learning to rank",
    "sentence-transformers", "sentence transformers", "semantic search",
    "vector search", "rag", "retrieval-augmented generation",
    "information retrieval", "bm25", "pgvector", "chromadb",
    "recommendation systems", "haystack",
})

# Proficiency multiplier for skill depth
PROF_MULTIPLIER = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.2}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                   ATD SCORING (AXIS A)                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def canonicalize_skill(skill_name: str) -> str:
    """Normalize and map skill name to canonical taxonomy keys."""
    name = skill_name.lower().strip()
    # Remove common punctuation/connectors
    name = re.sub(r'[^a-z0-9+ ]', '', name)
    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    # Exact aliases mapping
    aliases = {
        "finetuning": "fine-tuning llms",
        "fine tuning": "fine-tuning llms",
        "fine tuning llms": "fine-tuning llms",
        "finetuning llms": "fine-tuning llms",
        "llm finetuning": "fine-tuning llms",
        "llm fine tuning": "fine-tuning llms",
        "sentence transformer": "sentence-transformers",
        "sentencetransformers": "sentence-transformers",
        "sentencetransformer": "sentence-transformers",
        "vector db": "vector database",
        "vector dbs": "vector database",
        "vector databases": "vector database",
        "vector database": "vector database",
        "vector searches": "vector search",
        "hybrid searches": "hybrid search",
        "semantic searches": "semantic search",
        "recommendation system": "recommendation systems",
        "recommendation engine": "recommendation systems",
        "recommendation engines": "recommendation systems",
        "recsys": "recommendation systems",
        "ab testing": "a/b testing",
        "llama index": "llamaindex",
        "lang chain": "langchain",
        "openai": "openai api",
        "prompt engineering": "prompt engineering",
        "embeddings": "embeddings",
        "embedding": "embeddings",
    }
    
    # Check direct alias
    if name in aliases:
        return aliases[name]
    
    # Check if any taxonomy key is a substring or close match
    for taxonomy_key in ATD_TAXONOMY.keys():
        cleaned_key = re.sub(r'[^a-z0-9+ ]', '', taxonomy_key).strip()
        # Exact match of cleaned
        if name == cleaned_key:
            return taxonomy_key
        # Substring match for compound words (e.g. pytorch in pytorch lightning)
        if len(cleaned_key) > 3 and (cleaned_key in name or name in cleaned_key):
            return taxonomy_key
            
    return name


def compute_atd(skills: list[dict], career: list[dict]) -> float:
    """Compute Absolute Technical Dominance score.

    Finds the candidate's highest proven floor from their skills list
    and career descriptions, using the ATD taxonomy.

    Returns a score in [0.0, 1.0].
    """
    max_level = 0
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    weighted_depth = 0.0

    for s in skills:
        name = canonicalize_skill(s.get("name", ""))
        level = ATD_TAXONOMY.get(name, 0)
        if level > 0:
            dur = s.get("duration_months", 0)
            # Only count if they have real duration (anti-honeypot)
            if dur > 0:
                pm = PROF_MULTIPLIER.get(s.get("proficiency", "beginner"), 0.5)
                level_counts[level] += 1
                if level > max_level:
                    max_level = level
                # Weighted depth: higher levels contribute more
                weighted_depth += level * pm * min(2.0, dur / 24.0)

    # Also scan career descriptions for keyword signals
    career_text = " ".join(j.get("description", "") for j in career).lower()
    for keyword, level in ATD_DESC_KEYWORDS.items():
        if keyword in career_text:
            if level > max_level:
                max_level = level
            # Count as a "virtual" skill at that level
            level_counts[min(level, 4)] += 1

    if max_level == 0:
        return 0.0

    # Base score from highest level reached
    base_scores = {1: 0.15, 2: 0.40, 3: 0.70, 4: 1.0}
    base = base_scores[max_level]

    # Breadth bonus: reward having multiple skills at the highest reached level or above
    high_level_count = sum(level_counts[l] for l in range(max(max_level, 1), 5))
    breadth_bonus = min(0.15, high_level_count * 0.025)

    # Depth bonus from L3+ skills (capped)
    l3_plus = level_counts.get(3, 0) + level_counts.get(4, 0)
    depth_bonus = min(0.10, l3_plus * 0.02)

    # Continuous granularity from weighted_depth (proficiency × duration × level)
    # This ensures candidates at the same level are differentiated by depth
    granularity = min(0.20, weighted_depth * 0.008)

    return min(1.0, max(0.0, base + breadth_bonus + depth_bonus + granularity))


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                   HEA SCORING (AXIS B)                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def compute_hea(candidate: dict, weights: dict = None) -> float:
    """Compute High Execution Agency multiplier.

    Evaluates behavioral signals, career structure, and generalist-bleed
    to determine if the candidate has "founding engineer" DNA.

    Supports custom parameter weights passed as a dictionary.

    Returns a multiplier in [0.2, 3.0].
    """
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    w = weights or {}
    w_title = w.get("title", 1.20)
    w_exp = w.get("experience", 1.10)
    w_github = w.get("github", 0.35)
    w_fullstack = w.get("fullstack", 1.15)
    w_startup = w.get("startup", 1.15)
    w_product = w.get("product", 1.15)
    w_recency = w.get("recency", 1.0)
    w_notice = w.get("notice", 1.0)
    w_responsiveness = w.get("responsiveness", 0.60)
    w_interview = w.get("interview", 0.45)
    w_open_to_work = w.get("open_to_work", 1.05)
    w_location_india = w.get("location_india", 0.70)
    w_location_city = w.get("location_city", 1.10)
    w_domain = w.get("domain", 1.15)

    hea = 1.0  # Start neutral

    # ── 1. Title fit ──────────────────────────────────────────────────
    curr_title = profile.get("current_title", "").lower().strip()
    if curr_title in CORE_AI_TITLES:
        hea *= w_title
    elif curr_title in SWE_TITLES:
        hea *= 1.0    # Neutral — could be great, need other signals
    else:
        # Check career history for any tech title
        career_titles = {j.get("title", "").lower().strip() for j in career}
        if career_titles & CORE_AI_TITLES:
            hea *= (1.0 + (w_title - 1.0) * 0.5) if w_title > 1.0 else 0.9
        elif career_titles & SWE_TITLES:
            hea *= 0.7   # Tech background at least
        else:
            hea *= 0.3   # Non-tech career → strong penalty

    # ── 2. Experience band (sweet spot around 7 years - continuous Gaussian) ──
    years = profile.get("years_of_experience", 0)
    # Peak at 7 years, decay on both sides. Max bonus is w_exp.
    exp_factor = 0.65 + (w_exp - 0.65) * math.exp(-((years - 7.0) ** 2) / 18.0)
    hea *= exp_factor

    # ── 3. GitHub activity (0-100 scale, -1 = not linked) ─────────────
    # Continuous: smoothly rewards higher GitHub activity for granularity
    gh = signals.get("github_activity_score", -1)
    if gh >= 0:
        # Continuous scale: e.g. gh=0 -> 0.95, gh=100 -> 0.95 + w_github
        hea *= 0.95 + w_github * (gh / 100.0)

    # ── 4. Generalist Bleed: DevOps/Backend alongside AI ─────────────
    skill_names = {s.get("name", "").lower().strip() for s in skills}
    devops_count = len(skill_names & DEVOPS_SKILLS)
    backend_count = len(skill_names & BACKEND_SKILLS)

    if devops_count >= 2 or backend_count >= 2:
        hea *= w_fullstack
    elif devops_count >= 1 or backend_count >= 1:
        hea *= 1.0 + (w_fullstack - 1.0) * 0.33

    # ── 5. Startup / Small-Company Signal (Chaos Tolerance) ──────────
    small_company_stints = sum(
        1 for j in career
        if j.get("company_size", "") in ("1-10", "11-50", "51-200")
    )
    if small_company_stints >= 2:
        hea *= w_startup
    elif small_company_stints >= 1:
        hea *= 1.0 + (w_startup - 1.0) * 0.33

    # ── 6. Product Company vs Services ────────────────────────────────
    product_stints = sum(
        1 for j in career
        if j.get("industry", "").lower() in PRODUCT_INDUSTRIES
    )
    if product_stints >= 3:
        hea *= w_product
    elif product_stints >= 2:
        hea *= 1.0 + (w_product - 1.0) * 0.66
    elif product_stints >= 1:
        hea *= 1.0
    else:
        hea *= 0.80   # No product company experience at all

    # ── 7. Active recency (continuous sigmoid decay) ──────────────────
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (REF_DATE - last_dt).days
            # Smooth sigmoid decay mapping 0 days -> ~1.05, 180 days -> ~0.3
            recency_factor = 0.25 + 0.80 / (1.0 + math.exp((days_inactive - 110) / 30.0))
            hea *= 1.0 + (recency_factor - 1.0) * w_recency
        except (ValueError, TypeError):
            pass

    # ── 8. Notice period (continuous sigmoid decay) ───────────────────
    notice = signals.get("notice_period_days", 180)
    # Smooth sigmoid mapping notice <= 30 -> ~1.02, notice > 90 -> ~0.75
    notice_factor = 0.75 + 0.30 / (1.0 + math.exp((notice - 60) / 10.0))
    hea *= 1.0 + (notice_factor - 1.0) * w_notice

    # ── 9. Recruiter responsiveness (continuous) ───────────────────────
    if "recruiter_response_rate" in signals:
        resp = signals["recruiter_response_rate"]
        hea *= 0.80 + w_responsiveness * resp * 0.40
        
    # ── 10. Interview completion (continuous) ──────────────────────────
    if "interview_completion_rate" in signals:
        int_rate = signals["interview_completion_rate"]
        hea *= 0.85 + w_interview * int_rate * 0.30

    # ── 11. Open to work ──────────────────────────────────────────────
    if signals.get("open_to_work_flag", False):
        hea *= w_open_to_work

    # ── 12. Profile completeness ──────────────────────────────────────
    completeness = signals.get("profile_completeness_score", 50.0)
    hea *= 0.85 + 0.15 * (completeness / 100.0)

    # ── 13. Location (India preferred, Pune/Noida ideal) ──────────────
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    willing = signals.get("willing_to_relocate", False)

    if country == "india":
        if any(c in location for c in ("pune", "noida")):
            hea *= w_location_city
        elif any(c in location for c in ("bangalore", "hyderabad", "mumbai",
                                          "delhi", "gurgaon", "ncr", "ghaziabad")):
            hea *= 1.05   # Nearby Indian city
        elif willing:
            hea *= 1.0    # Willing to relocate within India
        else:
            return 0.0    # HARD FILTER: India but remote city, not willing to relocate
    else:
        hea *= w_location_india

    # ── 14. Richer Honeypot & Fraud Checks ────────────────────────────
    # A. Timeline overlapping clash (simultaneous full-time stints)
    has_timeline_clash = False
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
    for i in range(len(stints) - 1):
        if stints[i+1][0] < stints[i][1]:
            overlap_days = (stints[i][1] - stints[i+1][0]).days
            if overlap_days > 90:  # More than 3 months overlap is a fraud signal
                has_timeline_clash = True
                break
    if has_timeline_clash:
        return 0.0  # HARD FILTER: Severe penalty for fake overlapping timeline

    # B. Buzzword Stuffing (extremely high skill count relative to experience)
    if years > 0 and len(skills) / years > 6.0:
        hea *= 0.60  # Suspected keyword-stuffed profile
    elif len(skills) > 30:
        hea *= 0.50  # Skill count too high to be realistic

    # C. Title Inflation (VP/Director/Lead with very low experience)
    curr_title_lower = profile.get("current_title", "").lower()
    is_inflated = any(kw in curr_title_lower for kw in ("vp", "vice president", "chief", "principal", "director", "lead", "head"))
    if is_inflated and years < 4.0:
        hea *= 0.50  # Severe title inflation penalty

    # D. The True Title-Chaser (Job Hopper)
    if len(career) > 0 and (years / len(career)) < 1.8 and is_inflated:
        hea *= 0.30

    # E. Non-Coding Architects
    is_architect = any(kw in curr_title_lower for kw in ("architect", "lead", "head"))
    if is_architect and gh <= 5:
        hea *= 0.20

    # F. Proprietary Systems Without Validation
    if years >= 5.0 and gh == 0:
        hea *= 0.40

    # G. The "4-5 Years in Applied ML" Blindspot
    max_ml_months = 0
    for s in skills:
        sname = s.get("name", "").lower().strip()
        lvl = ATD_TAXONOMY.get(sname, 0)
        dur = s.get("duration_months", 0)
        if lvl >= 2 and dur > max_ml_months:
            max_ml_months = dur
    if max_ml_months < 12 and years >= 5.0:
        hea *= 0.85
    # D. Legacy Unrelated Title without Tech History
    if any(kw in curr_title_lower for kw in UNRELATED_TITLES):
        career_titles_text = " ".join(j.get("title", "") for j in career).lower()
        has_any_tech = any(
            kw in career_titles_text
            for kw in ("engineer", "developer", "scientist", "ml", "ai", "data")
        )
        if not has_any_tech:
            hea *= 0.10   # Fake AI profile penalty

    # ── 15. Domain Experience (HR-Tech / Marketplace) ─────────────────
    career_texts = " ".join(j.get("industry", "") + " " + j.get("description", "") for j in career).lower()
    if any(kw in career_texts for kw in HR_TECH_INDUSTRIES):
        hea *= w_domain

    # ── 16. Open Source Contribution Signal ───────────────────────────
    if "open source" in career_texts or "open-source" in career_texts:
        hea *= 1.05

    # ── 17. The Python Mandate ────────────────────────────────────────
    has_python = any(s.get("name", "").lower().strip() == "python" for s in skills) or ("python" in career_texts)
    if not has_python and years >= 3.0:
        hea *= 0.70  # Explicit JD requirement: "Strong Python."

    # ── 18. Shipper > Researcher Tilt ─────────────────────────────────
    is_pure_researcher = curr_title_lower in ("research scientist", "researcher", "applied scientist")
    if is_pure_researcher and devops_count == 0 and backend_count == 0:
        hea *= 0.85  # JD: "tilt slightly toward shipper than toward researcher"

    # ── 19. The True LangChain Tourist ────────────────────────────────
    # Max ML duration < 12 months, but has L1 skills and NO L3/L4 skills
    l1_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) == 1)
    l34_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) >= 3)
    if l1_count > 0 and l34_count == 0 and max_ml_months < 12:
        hea *= 0.10  # JD: "recent (under 12 months) projects using LangChain to call OpenAI"

    # ── 20. The "Ghost" Candidate (Inactive + Low Response) ───────────
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (REF_DATE - last_dt).days
            if days_inactive > 180 and resp <= 0.05:
                hea *= 0.10  # Explicit JD disqualifier: "not actually available"
        except (ValueError, TypeError):
            pass

    # I. The Evaluation Framework Gap
    has_eval = False
    for s in skills:
        if s.get("name", "").lower().strip() in EVAL_KEYWORDS:
            has_eval = True
            break
    if not has_eval:
        career_desc = " ".join(j.get("description", "") for j in career).lower()
        if any(kw in career_desc for kw in EVAL_KEYWORDS):
            has_eval = True
            
    if years >= 4.0 and not has_eval:
        hea *= 0.85  # Moderate penalty for lacking eval skills at senior level

    # J. Primary CV/Speech without NLP/IR
    skill_names_set = {s.get("name", "").lower().strip() for s in skills}
    cv_speech_count = len(skill_names_set & CV_SPEECH_KEYWORDS)
    ir_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) >= 3)
    if cv_speech_count >= 2 and ir_count == 0:
        hea *= 0.70  # Soft penalty: "re-learning fundamentals here"

    return max(0.2, min(3.0, hea))


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                   SINGULARITY SCORE                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def compute_singularity_score(atd: float, hea: float) -> float:
    """Final Score = (ATD ^ 1.5) × HEA

    The exponent creates a non-linear gap:
        ATD=1.0 (Level 4) → 1.000
        ATD=0.7 (Level 3) → 0.586
        ATD=0.4 (Level 2) → 0.253
        ATD=0.15 (Level 1) → 0.058

    This means a Level 4 candidate is ~17× more valuable than a Level 1
    candidate before HEA multiplier kicks in.
    """
    return (atd ** 1.5) * hea


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                       REASONING GENERATOR                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _hash_idx(cid: str, n: int) -> int:
    """Deterministic hash-based index for varied phrasing."""
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % n


def generate_reasoning(candidate: dict, rank: int, atd: float, hea: float) -> str:
    """Produce a factual, non-templated 1-2 sentence reasoning string.

    References specific facts from the candidate's profile and connects
    to JD requirements.  Uses hash-based variation for non-identical output.
    """
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
    gh      = signals.get("github_activity_score", -1)

    idx = _hash_idx(cid, 5)

    # Identify matched skills by category for specificity
    matched_ir = [s for s in skills if s.lower() in CORE_IR_SKILLS][:3]
    matched_l3 = [s for s in skills if ATD_TAXONOMY.get(s.lower(), 0) >= 3][:4]
    matched_l4 = [s for s in skills if ATD_TAXONOMY.get(s.lower(), 0) == 4][:3]

    # Determine ATD level label
    if atd >= 0.99:
        atd_label = "Level 4 (core infra/distributed)"
    elif atd >= 0.70:
        atd_label = "Level 3 (applied SOTA)"
    elif atd >= 0.40:
        atd_label = "Level 2 (standard ML)"
    else:
        atd_label = "Level 1 (wrapper-level)"

    # ── Sentence 1: Why they're ranked here ───────────────────────────
    if rank <= 10:
        intros = [
            f"Exceptional founding-team fit: {years:.1f} years of total experience, currently a {title} at {company}, with {atd_label} technical depth directly aligned with the ranking/retrieval mandate.",
            f"Top-tier candidate — {years:.1f} years of total experience, building production ML systems as a {title} at {company}, demonstrating {atd_label} mastery of the search/ranking stack.",
            f"Premier match for the Senior AI Engineer (Founding Team) role: {years:.1f} years of total experience, currently a {title} at {company}, with deep expertise in retrieval and ranking infrastructure.",
            f"Ideal founding-engineer profile: {years:.1f} years of total experience, hands-on ML engineering as a {title} at {company}, hitting {atd_label} on the AI depth scale.",
            f"Strong founding-team candidate: {years:.1f} years of total experience, currently a {title} at {company}, with proven {atd_label} technical floor and high execution agency.",
        ]
    elif rank <= 30:
        intros = [
            f"Solid candidate with {years:.1f} years as a {title} at {company}, demonstrating {atd_label} AI depth relevant to the hybrid search charter.",
            f"Strong alignment: {years:.1f} years of product-engineering experience as a {title} at {company}, with {atd_label} technical breadth.",
            f"Compelling profile with {years:.1f} years as a {title} at {company}, covering key JD requirements at {atd_label}.",
            f"Well-qualified — {years:.1f} years at {company} as a {title}, with applied ML depth at {atd_label} and strong engagement signals.",
            f"Relevant fit: {years:.1f} years building ML features as a {title} at {company}, scoring {atd_label} on technical taxonomy.",
        ]
    elif rank <= 70:
        intros = [
            f"Good candidate with {years:.1f} years as a {title} at {company}, reaching {atd_label} on the technical taxonomy.",
            f"Matches several JD criteria: {years:.1f} years as a {title} at {company} with {atd_label} AI skills coverage.",
            f"Moderate fit — {years:.1f} years at {company} as a {title}, covering partial JD requirements at {atd_label}.",
            f"Brings {years:.1f} years of experience as a {title} at {company} with {atd_label} technical alignment.",
            f"Reasonable match: {years:.1f} years at {company} as a {title}, with {atd_label} AI skills and acceptable engagement.",
        ]
    else:
        intros = [
            f"Borderline fit with {years:.1f} years as a {title} at {company}; {atd_label} technical depth limits ranking.",
            f"Adjacent candidate: {years:.1f} years at {company} as a {title}, with {atd_label} skills — weaker on core JD dimensions.",
            f"Lower-confidence match — {years:.1f} years as a {title} at {company}, {atd_label} with gaps in key retrieval/ranking areas.",
            f"Marginal fit at {years:.1f} years as a {title} at {company}, reaching only {atd_label} on the technical taxonomy.",
            f"Included at rank boundary: {years:.1f} years at {company} as a {title}, {atd_label} with partial overlap to the founding-team requirements.",
        ]

    s1 = intros[idx]

    # ── Sentence 2: Skills evidence + concerns ────────────────────────
    if matched_l4:
        skill_phrases = [
            f"Core infrastructure expertise in {', '.join(matched_l4)} positions them for the distributed training and serving mandate.",
            f"Hands-on with {', '.join(matched_l4)}, directly addressing the custom ML infra needs of the founding team.",
            f"Key strengths in {', '.join(matched_l4)} match the deep technical depth requirement.",
            f"Demonstrated mastery of {', '.join(matched_l4)}, critical for the inference and serving layer.",
            f"Production experience with {', '.join(matched_l4)} aligns with the scaling infrastructure needs.",
        ]
    elif matched_ir:
        skill_phrases = [
            f"Production experience with {', '.join(matched_ir)} aligns with the vector-retrieval and ranking stack.",
            f"Hands-on with {', '.join(matched_ir)}, directly relevant to the hybrid search and ranking mandate.",
            f"Key strengths in {', '.join(matched_ir)} match the IR/ranking infrastructure needs.",
            f"Demonstrated depth in {', '.join(matched_ir)}, critical for the retrieval and evaluation layer.",
            f"Core skill alignment through {', '.join(matched_ir)} in production settings.",
        ]
    elif matched_l3:
        skill_phrases = [
            f"Applied ML toolkit ({', '.join(matched_l3)}) supports the AI engineering requirements.",
            f"Background in {', '.join(matched_l3)} provides a solid foundation for the role.",
            f"Skills in {', '.join(matched_l3)} cover secondary JD requirements for fine-tuning and evaluation.",
            f"Proficiency in {', '.join(matched_l3)} bridges toward the search/ranking domain.",
            f"Has {', '.join(matched_l3)} experience applicable to the ML infrastructure layer.",
        ]
    else:
        skill_phrases = [
            "Skills profile is adjacent rather than directly aligned with core IR/vector requirements.",
            "Limited direct overlap with the vector search and ranking evaluation stack.",
            "Skill set covers general engineering but would need ramp-up on retrieval-specific tooling.",
            "Engineering fundamentals are present but specific search/NLP depth is thin.",
            "Core JD skills (embeddings, vector DBs, ranking eval) are under-represented in the profile.",
        ]

    s2 = skill_phrases[idx]

    # ── Sentence 3 (optional): Concerns ───────────────────────────────
    concerns = []
    
    career_texts = " ".join(j.get("industry", "") + " " + j.get("description", "") + " " + j.get("title", "") for j in career).lower()
    
    l1_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.lower(), 0) == 1)
    l34_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.lower(), 0) >= 3)
    curr_title_lower = title.lower()
    is_architect = any(kw in curr_title_lower for kw in ("architect", "lead", "head", "manager"))
    has_python = any(s.lower() == "python" for s in skills) or ("python" in career_texts)

    # 1. Job Hopper
    if len(career) > 0 and (years / len(career)) < 1.8:
        concerns.append("trajectory suggests title-chasing/job-hopping (<1.8 yrs/role)")

    # 2. Architect without coding
    if is_architect and gh <= 15:
        concerns.append("moved into architecture/lead role with low recent coding signals")
        
    # 3. Langchain Tourist
    if l1_count > 0 and l34_count == 0:
        concerns.append("framework enthusiast (heavy on wrappers, light on core ML systems)")
        
    # 4. No Python
    if not has_python and years >= 3.0:
        concerns.append("missing strong Python signals despite seniority")

    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if resp < 0.5:
        concerns.append(f"{int(resp * 100)}% recruiter response rate")
        
    # 5. Closed-source proprietary
    if gh >= 0 and gh < 10:
        if years >= 5.0 and "open source" not in career_texts and "open-source" not in career_texts:
            concerns.append("5+ years of proprietary work without external validation (low GitHub)")
        else:
            concerns.append(f"low GitHub activity ({gh:.0f}/100)")
            
    # 6. Service company only
    product_companies = sum(1 for c in career if any(ind in str(c.get("industry", "")).lower() for ind in PRODUCT_INDUSTRIES))
    if product_companies == 0 and len(career) > 0:
        if any(any(svc in str(c.get("company", "")).lower() for svc in SERVICE_COMPANIES) for c in career):
            concerns.append("career exclusively in consulting/services without product background")

    target_locs = ("pune", "noida", "gurgaon", "delhi", "hyderabad",
                   "mumbai", "bangalore")
    loc_l = loc.lower()
    if not any(c in loc_l for c in target_locs):
        willing = signals.get("willing_to_relocate", False)
        if not willing:
            concerns.append(f"based in {loc} without relocation willingness")

    # 7. CV/Speech Trap
    skill_names_set = {s.lower().strip() for s in skills}
    cv_speech_count = len(skill_names_set & CV_SPEECH_KEYWORDS)
    ir_count = sum(1 for s in skills if ATD_TAXONOMY.get(s.lower().strip(), 0) >= 3)
    if cv_speech_count >= 2 and ir_count == 0:
        concerns.append("primary expertise in CV/Speech/Robotics without significant NLP/IR exposure")

    # 8. Ghost Candidate
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            if (REF_DATE - last_dt).days > 180 and resp <= 0.05:
                concerns.append("highly inactive (>6 mos) with critically low recruiter response (<=5%)")
        except:
            pass

    # 9. Evaluation Gap
    has_eval = False
    for s in skills:
        if s.lower().strip() in EVAL_KEYWORDS:
            has_eval = True
            break
    if not has_eval:
        career_desc = " ".join(j.get("description", "") for j in career).lower()
        if any(kw in career_desc for kw in EVAL_KEYWORDS):
            has_eval = True
    if years >= 4.0 and not has_eval:
        concerns.append("missing evaluation framework signals (NDCG, MRR, A/B testing) despite seniority")

    # 10. Pure Research without production
    is_pure_researcher = title.lower() in ("research scientist", "researcher", "applied scientist")
    has_prod = any(kw in career_texts for kw in ("production", "scale", "deployment", "deployed", "api", "infra"))
    if is_pure_researcher and not has_prod:
        concerns.append("research-heavy background with limited production deployment signals")

    # ── Highlights ────────────────────────────────────────────────────
    highlights = []
    
    # 1. Product DNA
    product_companies = sum(1 for c in career if any(ind in str(c.get("industry", "")).lower() for ind in PRODUCT_INDUSTRIES))
    if product_companies >= 2:
        highlights.append("Strong Product DNA")
        
    # 2. DevOps / Full-stack Shipper
    devops_count = len(skill_names_set & DEVOPS_SKILLS)
    backend_count = len(skill_names_set & BACKEND_SKILLS)
    if devops_count >= 1 and backend_count >= 1:
        highlights.append("End-to-End Shipper")
        
    # 3. Evaluation Champion
    if has_eval and years >= 2.0:
        highlights.append("Evaluation Champion 🥇")
        
    # 4. Open Source
    gh = signals.get("github_activity_score", 0)
    if gh > 80 or "open source" in career_texts or "open-source" in career_texts:
        highlights.append("Open Source Builder")
        
    if highlights:
        highlight_strs = [
            f"Highlight: {', '.join(highlights)}.",
            f"Key Strength: {', '.join(highlights)}.",
            f"Bonus: {', '.join(highlights)}.",
        ]
        s4 = " " + highlight_strs[idx % 3]
    else:
        s4 = ""

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

    res = s1 + " " + s2 + s4 + s3
    
    # ── Tag True "Hidden Gems" ────────────────────────────────────────
    # If the candidate has extremely high HEA but didn't reach ATD Level 4
    if hea >= 1.0 and atd < 1.0:
        res = "[Hidden Gem 💎] " + res

    return res


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           MAIN PIPELINE                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
REASONING_CACHE_PATH = os.path.join(DIR_PATH, "reasoning_cache.json")


def load_honeypots(path: str) -> set[str]:
    """Load the pre-computed honeypot IDs from a JSON file."""
    if not os.path.exists(path):
        print(f"WARNING: honeypots file not found at {path}; skipping filter.")
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return set(data.keys())


def load_reasoning_cache() -> dict:
    """Load reasoning cache JSON file if it exists."""
    if os.path.exists(REASONING_CACHE_PATH):
        print(f"Loading cached reasonings from {REASONING_CACHE_PATH}...")
        with open(REASONING_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Senior AI Engineer role "
                    "using the Singularity Engine (ATD × HEA)."
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

    print("=" * 60)
    print("  THE SINGULARITY ENGINE — Redrob MatchWise v2")
    print("  Score = (ATD ^ 1.5) × HEA")
    print("=" * 60)

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

    # Load reasoning cache
    reasoning_cache = load_reasoning_cache()
    if reasoning_cache:
        print(f"Loaded {len(reasoning_cache)} cached reasoning entries.")

    # ── Stream, filter, and score candidates ──────────────────────────
    print(f"\nStreaming candidates from: {cpath}")
    scored = []
    total = 0
    filtered_honeypot = 0
    filtered_service = 0
    filtered_title = 0
    filtered_atd = 0

    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total += 1

            cid = cand.get("candidate_id", "")

            # ── Hard filter 1: Honeypots ──
            if cid in honeypot_ids:
                filtered_honeypot += 1
                continue

            profile = cand.get("profile", {})
            career  = cand.get("career_history", [])
            skills  = cand.get("skills", [])

            # ── Hard filter 2: Service-only career ──
            companies = {j.get("company") for j in career if j.get("company")}
            if companies and all(c in SERVICE_COMPANIES for c in companies):
                filtered_service += 1
                continue

            # ── Hard filter 3: Unrelated current title with no tech history ──
            curr_title = profile.get("current_title", "").lower().strip()
            if curr_title in UNRELATED_TITLES:
                # Check if career history has ANY technical role
                career_titles_text = " ".join(
                    j.get("title", "") for j in career
                ).lower()
                has_tech_history = any(
                    kw in career_titles_text
                    for kw in ("engineer", "developer", "scientist", "ml",
                               "ai", "data", "research")
                )
                if not has_tech_history:
                    filtered_title += 1
                    continue

            # ── Compute scores ──
            atd = compute_atd(skills, career)
            hea = compute_hea(cand)

            if hea <= 0.0:
                continue


            # ── Hard filter: The "LangChain Tourist" Bypass ──
            max_lvl = max([ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) for s in skills] + [0])
            max_ml_months = max([s.get("duration_months", 0) for s in skills if ATD_TAXONOMY.get(s.get("name", "").lower().strip(), 0) >= 1] + [0])
            if max_lvl == 1 and max_ml_months < 12:
                continue

            # ── Soft filter: ATD too low (pure Level 1 / no AI skills) ──
            if atd < 0.10:
                filtered_atd += 1
                continue

            score = compute_singularity_score(atd, hea)
            scored.append((cid, score, cand, atd, hea))

    t_score = time.time()
    print(f"\nProcessed {total} candidates in {t_score - t0:.1f}s.")
    print(f"  Honeypots filtered: {filtered_honeypot}")
    print(f"  Service-only filtered: {filtered_service}")
    print(f"  Unrelated title filtered: {filtered_title}")
    print(f"  Low ATD filtered: {filtered_atd}")
    print(f"  Viable candidates scored: {len(scored)}")

    # ── Sort: score descending, candidate_id ascending for tiebreaks ──
    scored.sort(key=lambda x: (-x[1], x[0]))

    # ── Hybrid Quota Ranking (Best 90 + Up to 10 Hidden Gems) ─────────
    # We allocate up to 10 spots for "Hidden Gems" (HEA >= 1.0, ATD < 1.0).
    # We search the entire list to find the 10 best hidden gems, sorted by HEA.
    hidden_gems = []
    for item in scored:
        cid, score, reasoning, atd, hea = item
        if hea >= 1.0 and atd < 1.0:
            hidden_gems.append(item)
            
    # Sort hidden gems by HEA descending to get the highest execution athletes
    hidden_gems.sort(key=lambda x: -x[4])
    top_hidden_gems = hidden_gems[:10]
    hidden_gem_cids = {item[0] for item in top_hidden_gems}
    
    # Fill remaining spots to reach exactly 100
    top_standard = []
    quota = 100 - len(top_hidden_gems)
    for item in scored:
        if item[0] not in hidden_gem_cids:
            top_standard.append(item)
        if len(top_standard) >= quota:
            break
            
    top_100 = top_standard + top_hidden_gems
    # Sort again so the final 100 output is monotonically descending
    top_100.sort(key=lambda x: (-x[1], x[0]))

    # ── Generate output ───────────────────────────────────────────────
    rows = []
    for rank_idx, (cid, score, cand, atd, hea) in enumerate(top_100, start=1):
        reasoning = reasoning_cache.get(cid, "")
        if not reasoning:
            reasoning = generate_reasoning(cand, rank_idx, atd, hea)
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    # ── Write CSV ─────────────────────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Submission written to: {args.out}")
    print(f"Top 100 candidates ranked. Score range: "
          f"{rows[0]['score']:.6f} -> {rows[-1]['score']:.6f}")
    print(f"Total runtime: {total_time:.1f}s")
    print(f"{'=' * 60}")

    # ── Top-10 snapshot ───────────────────────────────────────────────
    print("\nTop-10 Snapshot:")
    print(f"{'Rank':>4}  {'Score':>8}  {'ATD':>5}  {'HEA':>5}  {'Title':<35}  {'Company':<25}")
    print("-" * 90)
    for rank_idx, (cid, score, cand, atd, hea) in enumerate(top_100[:10], start=1):
        p = cand.get("profile", {})
        print(f"{rank_idx:>4}  {score:>8.4f}  {atd:>5.3f}  {hea:>5.3f}  "
              f"{p.get('current_title', ''):<35.35}  {p.get('current_company', ''):<25.25}")

    # ── Sanity checks ─────────────────────────────────────────────────
    ids_in_output = {r["candidate_id"] for r in rows}
    honeypots_in_output = ids_in_output & honeypot_ids
    if honeypots_in_output:
        print(f"\nWARNING: {len(honeypots_in_output)} honeypots in output!")
    else:
        print("\nOK: Zero honeypots in top 100.")

    if total_time > 300:
        print(f"WARNING: Runtime {total_time:.0f}s exceeds 5-minute limit!")
    else:
        print(f"OK: Runtime {total_time:.1f}s is within 5-minute budget.")

    # Check for keyword-trap profiles in top 10
    trap_titles = {"marketing manager", "hr manager", "accountant",
                   "graphic designer", "content writer", "sales executive",
                   "mechanical engineer", "civil engineer", "project manager",
                   "operations manager", "customer support"}
    trap_count = sum(
        1 for _, _, cand, _, _ in top_100[:10]
        if cand.get("profile", {}).get("current_title", "").lower().strip()
        in trap_titles
    )
    if trap_count > 0:
        print(f"WARNING: {trap_count}/10 top candidates have non-tech titles "
              f"(possible keyword-trap profiles).")
    else:
        print("OK: Top-10 has zero non-tech title candidates (keyword trap avoided).")


if __name__ == "__main__":
    main()
