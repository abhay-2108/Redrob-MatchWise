# Redrob MatchWise — Candidate Discovery & Ranking Engine

A production-grade candidate ranking system for the **Senior AI Engineer (Founding Team)** role at Redrob AI.

Processes 100,000 synthetic candidate profiles and outputs a CSV containing the **top 100 best-fit candidates** ranked in descending order of suitability.

### Quick Start

### 1. Pre-computation (offline, one-time)

Generate the honeypot filter list:

```bash
python identify_all_honeypots.py --candidates ./docs/candidates.jsonl --out ./honeypots.json
```

Generate the candidate semantic dense embeddings:

```bash
python precompute_embeddings.py --candidates ./docs/candidates.jsonl
```

### 2. Run the ranker (sandbox-compliant)

```bash
python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv
```

This single command:
- Loads the candidate pool (100K profiles)
- Filters out honeypots, service-only careers, and irrelevant titles
- Loads precomputed embeddings and evaluates cosine similarity
- Builds an on-the-fly BM25 index on valid candidate profiles to evaluate query relevance
- Scores candidates across 5 weighted components (heuristics)
- Integrates a behavioral availability modifier
- Generates factual reasoning (lookup from cache or dynamic generator)
- Writes the final `submission.csv`

**Runtime:** ~23 seconds on CPU | **Memory:** < 200 MB | **Dependencies:** Python 3.10+ (`numpy`, `rank_bm25`, `sentence-transformers`)

### 3. Validate the submission

```bash
python docs/validate_submission.py ./submission.csv
```

### 4. Launch the sandbox demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    HYBRID RANKING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   candidates.jsonl ──► Hard Filters ──► 4-Layer Scoring Combined ──► Sort ──► CSV       │
│         (100K)          │                      │                                        │
│                         ├─ Honeypots           ├─ Layer 1: Dense Vectors (30%)          │
│                         ├─ Service-only        ├─ Layer 2: BM25 Keywords (25%)          │
│                         └─ Unrelated titles    ├─ Layer 3: Heuristics (25%)             │
│                                                └─ Layer 4: Availability (20%)           │
│                                                                                         │
│   Final Score = 0.30 * Cosine + 0.25 * BM25 + 0.25 * Heuristics + 0.20 * Availability   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Scoring Components & Layer Weights

1. **Layer 1: Semantic Dense Vectors (30%)**: Cosine similarity against the Job Description using the `all-MiniLM-L6-v2` sentence-transformer. Embeddings are pre-computed offline (for 3x speed and small Git footprint) with a lazy-loading fallback on CPU for verification.
2. **Layer 2: BM25 Keyword Matching (25%)**: Built on-the-fly using `BM25Okapi` over filtered candidates. Evaluates JD query term frequency-inverse document frequency.
3. **Layer 3: Structural Heuristics (25%)**: Evaluates title fit (AI vs SWE vs career fallbacks), experience fit (peak at 6-8 years), skills depth (NLP/IR/vector systems weighted by proficiency, duration, and endorsements), and Noida/Pune proximity.
4. **Layer 4: Behavioral availability (20%)**: Multiplicative modification based on 23 Redrob signals (recency of activity, notice period, responsiveness, open-to-work flag, GitHub activity, completeness).

## Compute Constraints

| Constraint | Limit | Actual |
|------------|-------|--------|
| Runtime | ≤ 5 min | ~23 sec |
| Memory | ≤ 16 GB | < 200 MB |
| Compute | CPU only | ✓ |
| Network | Offline | ✓ |
| Disk | ≤ 5 GB | ~50 MB |

## AI Tools Used

- **Claude / Gemini**: Architecture design, speed benchmarking, code review, scoring heuristic refinement
- No candidate data was processed by any external LLM during the ranking phase (offline or online)
