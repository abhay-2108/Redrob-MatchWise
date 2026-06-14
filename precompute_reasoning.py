#!/usr/bin/env python3
"""
Redrob Hackathon — LLM Reasoning Pre-computation
==================================================
Offline step: generates high-quality, factual reasoning strings for the
top ~500 candidates using a lightweight local LLM or an API.

Produces:
  - reasoning_cache.json  (dict: candidate_id -> reasoning string)

The ranking script loads this cache at runtime and looks up reasoning
for each top-100 candidate. Candidates not in the cache fall back to
a template-based reasoning generator.

Supports two modes:
  1. Local LLM via transformers (e.g. Qwen2.5-1.5B-Instruct)
  2. Google Gemini API (set GEMINI_API_KEY env var)

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

# We'll import rank.py's scoring logic to identify top candidates
# before generating reasoning for them.


def build_candidate_summary(cand: dict) -> str:
    """Build a concise profile summary for the LLM prompt."""
    p = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    career_lines = []
    for j in career[:3]:  # Top 3 jobs
        career_lines.append(
            f"- {j.get('title', '')} at {j.get('company', '')} "
            f"({j.get('duration_months', 0)} months, {j.get('industry', '')})"
        )

    skill_names = [s.get("name", "") for s in skills if s.get("name")][:15]

    return f"""Candidate: {p.get('anonymized_name', '')}
Current Role: {p.get('current_title', '')} at {p.get('current_company', '')}
Experience: {p.get('years_of_experience', 0)} years
Location: {p.get('location', '')}, {p.get('country', '')}
Industry: {p.get('current_industry', '')}

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


def build_prompt(candidate_summary: str, rank: int) -> str:
    """Build the LLM prompt for generating reasoning."""
    return f"""You are evaluating candidates for a Senior AI Engineer (Founding Team) role at Redrob AI, a Series A talent intelligence platform in Pune/Noida, India.

The ideal candidate has:
- 5-9 years experience, 4-5 in applied ML/AI at product companies
- Production experience with embeddings, vector databases, hybrid search, ranking systems
- Strong Python, evaluation frameworks (NDCG, MAP), and shipped search/recommendation systems
- Located in or willing to relocate to Pune/Noida

This candidate is ranked #{rank} out of 100.

{candidate_summary}

Write exactly 1-2 sentences explaining why this candidate is ranked at position #{rank}. Be specific: reference their actual title, company, years of experience, and relevant skills. If there are concerns (notice period, location, response rate), mention them honestly. Do not hallucinate skills or facts not listed above."""


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

    for i, (cand, rank) in enumerate(candidates_with_ranks):
        cid = cand.get("candidate_id", "")
        summary = build_candidate_summary(cand)
        prompt = build_prompt(summary, rank)

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

    for i, (cand, rank) in enumerate(candidates_with_ranks):
        cid = cand.get("candidate_id", "")
        summary = build_candidate_summary(cand)
        prompt = build_prompt(summary, rank)

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
        "--embeddings", default="./candidate_embeddings.npy",
        help="Path to pre-computed embeddings .npy file",
    )
    parser.add_argument(
        "--jd-embedding", default="./jd_embedding.npy",
        help="Path to JD embedding .npy file",
    )
    parser.add_argument(
        "--candidate-ids", default="./candidate_ids.json",
        help="Path to candidate_ids.json",
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

    # ── Step 1: Run the ranking pipeline to identify top N candidates ──
    print("Running ranking pipeline to identify top candidates...")

    # Import rank.py scoring
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rank import score_candidate, load_honeypots

    import numpy as np

    honeypot_ids = load_honeypots(args.honeypots)

    # Load embeddings for cosine similarity
    has_embeddings = os.path.exists(args.embeddings) and os.path.exists(args.jd_embedding)
    cosine_scores = {}
    if has_embeddings:
        print("Loading pre-computed embeddings...")
        cand_embs = np.load(args.embeddings)
        jd_emb = np.load(args.jd_embedding)
        with open(args.candidate_ids, "r") as f:
            emb_ids = json.load(f)
        # Cosine similarity (embeddings are L2-normalised, so dot product = cosine)
        scores_vec = cand_embs @ jd_emb.T
        scores_vec = scores_vec.flatten()
        for idx, cid in enumerate(emb_ids):
            cosine_scores[cid] = float(scores_vec[idx])
        print(f"Loaded cosine scores for {len(cosine_scores)} candidates.")

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

            heuristic_score, _ = score_candidate(cand, honeypot_ids)
            if heuristic_score <= 0:
                continue

            # Combine with cosine if available
            cos = cosine_scores.get(cid, 0.0)
            combined = 0.5 * heuristic_score + 0.5 * cos
            scored.append((cid, combined))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top_n = scored[:args.top_n]

    print(f"Identified top {len(top_n)} candidates for reasoning generation.")

    # ── Step 2: Generate reasoning ────────────────────────────────────
    candidates_with_ranks = [
        (all_candidates[cid], rank_idx)
        for rank_idx, (cid, _) in enumerate(top_n, start=1)
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
