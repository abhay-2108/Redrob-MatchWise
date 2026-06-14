#!/usr/bin/env python3
"""
Redrob Hackathon — LLM Reasoning Pre-computation (Singularity Engine)
======================================================================
Offline step: generates high-quality, factual reasoning strings for the
top ~500 candidates using the Google Gemini API.

Produces:
  - reasoning_cache.json  (dict: candidate_id -> reasoning string)

The ranking script loads this cache at runtime and looks up reasoning
for each top-100 candidate. Candidates not in the cache fall back to
a template-based reasoning generator.

Usage:
    # Using Gemini API (recommended for quality):
    set GEMINI_API_KEY=your_key
    python precompute_reasoning.py --mode gemini --top-n 500

    # Using local model:
    python precompute_reasoning.py --mode local --model Qwen/Qwen2.5-1.5B-Instruct --top-n 300
"""

import argparse
import gzip
import json
import os
import sys
import time


def build_candidate_summary(cand: dict) -> str:
    """Build a concise profile summary for the LLM prompt."""
    p = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    career_lines = []
    for j in career[:4]:  # Top 4 jobs for founding-engineer context
        career_lines.append(
            f"- {j.get('title', '')} at {j.get('company', '')} "
            f"({j.get('duration_months', 0)} months, {j.get('industry', '')}, "
            f"company size: {j.get('company_size', 'unknown')})"
        )

    skill_names = [s.get("name", "") for s in skills if s.get("name")][:15]

    return f"""Candidate: {p.get('anonymized_name', '')}
Current Role: {p.get('current_title', '')} at {p.get('current_company', '')}
Experience: {p.get('years_of_experience', 0)} years
Location: {p.get('location', '')}, {p.get('country', '')}
Industry: {p.get('current_industry', '')}
Company Size: {p.get('current_company_size', '')}

Career History:
{chr(10).join(career_lines)}

Skills: {', '.join(skill_names)}

Behavioral Signals:
- Notice Period: {signals.get('notice_period_days', 'N/A')} days
- Recruiter Response Rate: {signals.get('recruiter_response_rate', 0):.0%}
- Interview Completion: {signals.get('interview_completion_rate', 0):.0%}
- Open to Work: {signals.get('open_to_work_flag', False)}
- Last Active: {signals.get('last_active_date', 'N/A')}
- GitHub Activity: {signals.get('github_activity_score', -1)}
- Willing to Relocate: {signals.get('willing_to_relocate', False)}"""


def build_prompt(candidate_summary: str, rank: int, atd: float, hea: float) -> str:
    """Build the LLM prompt for generating reasoning."""
    return f"""You are evaluating candidates for a Senior AI Engineer (Founding Team) role at Redrob AI, a Series A talent intelligence platform in Pune/Noida, India.

The ideal candidate has:
- 5-9 years experience, 4-5 in applied ML/AI at product companies (NOT consulting firms)
- Production experience with embeddings, vector databases, hybrid search, ranking systems
- Strong Python, evaluation frameworks (NDCG, MAP), and shipped search/recommendation systems
- Located in or willing to relocate to Pune/Noida
- "Founding engineer" DNA: works at startups, deploys own code, builds in public

We use the Singularity Engine scoring:
- ATD (Absolute Technical Dominance): {atd:.3f} — measures depth on a 4-level AI difficulty taxonomy
  (Level 4: CUDA/distributed training, Level 3: fine-tuning/vector DBs, Level 2: PyTorch/RAG, Level 1: API wrappers)
- HEA (High Execution Agency): {hea:.3f} — measures startup DNA, GitHub activity, generalist skills, availability

This candidate is ranked #{rank} out of 100.

{candidate_summary}

Write exactly 1-2 sentences explaining why this candidate is ranked at position #{rank}. Be specific: reference their actual title, company, years of experience, and relevant skills from the list above. If there are concerns (notice period, location, response rate, gaps), mention them honestly. Do NOT hallucinate skills or facts not listed above. Do NOT use generic praise — be precise about what makes them a fit or a concern."""


def generate_with_gemini(candidates_with_ranks: list, api_key: str) -> dict:
    """Generate reasoning using Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai not installed. Run: uv pip install google-generativeai")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    cache = {}
    total = len(candidates_with_ranks)

    for i, (cand, rank, atd, hea) in enumerate(candidates_with_ranks):
        cid = cand.get("candidate_id", "")
        summary = build_candidate_summary(cand)
        prompt = build_prompt(summary, rank, atd, hea)

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.3,
                ),
            )
            reasoning = response.text.strip().replace("\n", " ")
            # Truncate to 2 sentences max
            sentences = reasoning.split(". ")
            if len(sentences) > 2:
                reasoning = ". ".join(sentences[:2]) + "."
            cache[cid] = reasoning
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] Generated reasoning for {cid}")
        except Exception as e:
            print(f"  Error for {cid}: {e}")
            continue

        # Brief pause to respect rate limits
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    return cache


def generate_with_local_llm(candidates_with_ranks: list, model_name: str) -> dict:
    """Generate reasoning using a local HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        print("ERROR: transformers not installed. Run: uv pip install transformers torch")
        sys.exit(1)

    print(f"Loading local model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")

    cache = {}
    total = len(candidates_with_ranks)

    for i, (cand, rank, atd, hea) in enumerate(candidates_with_ranks):
        cid = cand.get("candidate_id", "")
        summary = build_candidate_summary(cand)
        prompt = build_prompt(summary, rank, atd, hea)

        try:
            messages = [{"role": "user", "content": prompt}]
            result = pipe(messages)
            reasoning = result[0]["generated_text"][-1]["content"].strip().replace("\n", " ")
            sentences = reasoning.split(". ")
            if len(sentences) > 2:
                reasoning = ". ".join(sentences[:2]) + "."
            cache[cid] = reasoning
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}] Generated reasoning for {cid}")
        except Exception as e:
            print(f"  Error for {cid}: {e}")
            continue

    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute LLM-generated reasoning for top candidates."
    )
    parser.add_argument(
        "--candidates", default="./docs/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--honeypots", default="./honeypots.json",
        help="Path to honeypots.json",
    )
    parser.add_argument(
        "--mode", choices=["gemini", "local"], default="gemini",
        help="LLM mode: gemini (API) or local (HuggingFace)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Local model name (for --mode local)",
    )
    parser.add_argument(
        "--top-n", type=int, default=500,
        help="Number of top candidates to generate reasoning for",
    )
    parser.add_argument(
        "--out", default="./reasoning_cache.json",
        help="Output reasoning cache JSON file",
    )
    args = parser.parse_args()

    # ── Step 1: Run the Singularity Engine to identify top N candidates ──
    print("Running Singularity Engine scoring to identify top candidates...")

    # Import scoring functions from rank.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rank import (
        compute_atd, compute_hea, compute_singularity_score,
        load_honeypots, SERVICE_COMPANIES, UNRELATED_TITLES,
    )

    honeypot_ids = load_honeypots(args.honeypots)

    # Score all candidates
    cpath = args.candidates
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt

    scored = []
    all_candidates = {}

    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            all_candidates[cid] = cand

            # Apply hard filters
            if cid in honeypot_ids:
                continue

            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])

            companies = {j.get("company") for j in career if j.get("company")}
            if companies and all(c in SERVICE_COMPANIES for c in companies):
                continue

            curr_title = profile.get("current_title", "").lower().strip()
            if curr_title in UNRELATED_TITLES:
                career_titles_text = " ".join(
                    j.get("title", "") for j in career
                ).lower()
                has_tech = any(
                    kw in career_titles_text
                    for kw in ("engineer", "developer", "scientist", "ml",
                               "ai", "data", "research")
                )
                if not has_tech:
                    continue

            atd = compute_atd(skills, career)
            hea = compute_hea(cand)

            if atd < 0.10:
                continue

            score = compute_singularity_score(atd, hea)
            scored.append((cid, score, atd, hea))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top_n = scored[:args.top_n]

    print(f"Identified top {len(top_n)} candidates for reasoning generation.")

    # ── Step 2: Generate reasoning ────────────────────────────────────
    candidates_with_ranks = [
        (all_candidates[cid], rank_idx, atd, hea)
        for rank_idx, (cid, _, atd, hea) in enumerate(top_n, start=1)
    ]

    if args.mode == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.")
            print("Set it with: set GEMINI_API_KEY=your_api_key")
            sys.exit(1)
        print(f"Generating reasoning with Gemini API for {len(candidates_with_ranks)} candidates...")
        cache = generate_with_gemini(candidates_with_ranks, api_key)
    else:
        print(f"Generating reasoning with local model: {args.model}")
        cache = generate_with_local_llm(candidates_with_ranks, args.model)

    # ── Step 3: Save cache ────────────────────────────────────────────
    # Load existing cache if present and merge
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Merging with existing cache ({len(existing)} entries)...")
        existing.update(cache)
        cache = existing

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(cache)} reasoning entries -> {args.out}")


if __name__ == "__main__":
    main()
