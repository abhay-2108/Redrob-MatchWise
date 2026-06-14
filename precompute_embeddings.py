#!/usr/bin/env python3
"""
Redrob Hackathon — Embedding Pre-computation
=============================================
Offline step: generates dense vector embeddings for all 100K candidate
profiles and the Job Description using sentence-transformers.

Produces:
  - candidate_embeddings.npy   (100000 x 384 float32, ~150 MB)
  - candidate_ids.json         (ordered list of candidate_id strings)
  - jd_embedding.npy           (1 x 384 float32)

These files are loaded by rank.py at runtime to compute cosine similarity
in < 2 seconds on CPU (numpy dot product).

Usage:
    python precompute_embeddings.py [--candidates PATH] [--model MODEL]
"""

import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Limit CPU threads to avoid contention and heat throttling on CPU
torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# Job Description text (condensed from job_description.docx)
# ---------------------------------------------------------------------------
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
    """Build a rich text representation of a candidate for embedding."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute sentence-transformer embeddings for the candidate pool."
    )
    parser.add_argument(
        "--candidates", default="./docs/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Output directory for .npy and .json files",
    )
    parser.add_argument(
        "--honeypots", default="./honeypots.json",
        help="Path to honeypots.json",
    )
    args = parser.parse_args()

    # Import filter constants and functions from rank
    from rank import SERVICE_COMPANIES, UNRELATED_TITLES, load_honeypots

    # Resolve candidates path
    cpath = args.candidates
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
        else:
            print(f"ERROR: Candidate file not found at {cpath} or {alt}")
            raise SystemExit(1)

    # Load honeypots
    honeypot_ids = load_honeypots(args.honeypots)
    print(f"Loaded {len(honeypot_ids)} honeypot IDs.")

    # Load model
    print(f"Loading sentence-transformer model: {args.model}")
    t0 = time.time()
    model = SentenceTransformer(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Read all candidates and apply hard filters before embedding
    print(f"Reading and filtering candidates from: {cpath}")
    candidate_ids = []
    candidate_texts = []
    total_read = 0

    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            total_read += 1
            
            cid = cand.get("candidate_id", "")
            
            # Apply hard filters from rank.py
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
                
            candidate_ids.append(cid)
            candidate_texts.append(build_candidate_text(cand))

    print(f"Read {total_read} candidates. {len(candidate_ids)} passed hard filters and will be encoded.")

    # Encode candidates
    print(f"Encoding candidate texts (batch_size={args.batch_size})...")
    t1 = time.time()
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalised for fast cosine via dot product
    )
    t2 = time.time()
    print(f"Encoded {len(candidate_ids)} candidates in {t2 - t1:.1f}s")
    print(f"Embedding shape: {candidate_embeddings.shape}")

    # Encode JD
    print("Encoding Job Description...")
    jd_embedding = model.encode(
        [JD_TEXT],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    emb_path = os.path.join(args.out_dir, "candidate_embeddings.npy")
    np.save(emb_path, candidate_embeddings.astype(np.float32))
    print(f"Saved candidate embeddings -> {emb_path} "
          f"({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    jd_path = os.path.join(args.out_dir, "jd_embedding.npy")
    np.save(jd_path, jd_embedding.astype(np.float32))
    print(f"Saved JD embedding -> {jd_path}")

    ids_path = os.path.join(args.out_dir, "candidate_ids.json")
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump(candidate_ids, fh)
    print(f"Saved candidate IDs -> {ids_path}")

    total = time.time() - t0
    print(f"\nTotal pre-computation time: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
