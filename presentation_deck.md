# Redrob MatchWise — Presentation Deck
> Copy-paste content for each slide into your PPT.

---

## Slide 1 — Title Slide

**Team Name:** Agent Mania

**Problem Statement:** Intelligent Candidate Discovery & Ranking for Senior AI Engineer (Founding Team) at Redrob AI

**Team Leader:** Abhay Tiwari

---

## Slide 2 — Solution Overview

**What is your proposed solution?**

A 4-stage CPU-only ranking pipeline that processes 100,000 candidate profiles in ~8–10 seconds and outputs the top 100 best-fit candidates. It combines gradient-boosted trees (XGBoost + LightGBM), a FlashRank cross-encoder, and a proven heuristic formula to score, rank, and explain each candidate.

**What differentiates your approach from traditional candidate matching systems?**

- **Multi-stage architecture** — hard filters remove noise, ML models score, cross-encoder reranks semantically, then weighted fusion produces final ranking
- **Explainable** — every candidate gets a plain-English reasoning string (not a black-box score)
- **Fraud-resistant** — 7 hard filters catch honeypots, ghost profiles, skill inflation, service-only careers, and location/relocation issues
- **CPU-only, 8–10 seconds** — no GPU, no network, stays within hackathon constraints
- **Hidden Gem detection** — flags high-execution candidates with moderate technical depth who might be overlooked

---

## Slide 3 — JD Understanding & Candidate Evaluation

**Key requirements extracted from the JD:**

| Requirement | Priority |
|-------------|----------|
| NLP / IR / Embeddings / Retrieval | Critical |
| Python, production ML systems | Critical |
| Startup / founding team experience | High |
| Search / ranking infrastructure | High |
| Evaluation methodology (NDCG, MRR) | High |
| Pune / Noida / India location | Medium |
| 3+ year tenure commitment | Medium |

**How does your solution evaluate candidate fit beyond keyword matching?**

- **4-level ATD taxonomy** — classifies skills from L1 (API callers) to L4 (distributed training, GPU kernels)
- **14-signal HEA score** — product company ratio, tenure stability, career momentum, GitHub activity
- **Semantic similarity** — Sentence-BERT + FlashRank cross-encoder compares profile text against JD at meaning level
- **Fraud detection** — timeline overlap, skill inflation, ghost detection prevent gaming

---

## Slide 4 — Ranking Methodology

**How does your system retrieve, score, and rank candidates?**

```
Stage 1 (Hard Filter) → ~59K removed (honeypots, zero-skill, service-only, ghosts)
Stage 2 (GBM Score)   → XGBoost + LightGBM score all ~40K viable, keep top 200
Stage 3 (Rerank)      → FlashRank TinyBERT cross-encoder reranks top 50
Stage 4 (Fusion)      → 40% XGBoost + 20% FlashRank + 40% Heuristic → top 100
```

**Models, algorithms, and heuristics used:**

- **XGBoost LambdaMART** — listwise ranking model trained on heuristic-labeled data
- **LightGBM Ranker** — gradient-boosted ranking, ensemble with XGBoost (60/40 split)
- **FlashRank ms-marco-TinyBERT-L-2-v2** — lightweight cross-encoder for semantic relevance
- **Singularity Score (Heuristic)** — `ATD¹·⁵ × HEA` proven formula from rank.py

**How multiple signals are combined:**

```python
final_score = 0.40 × XGBoost + 0.20 × FlashRank + 0.40 × Heuristic
```

---

## Slide 5 — Explainability & Data Validation

**How are ranking decisions explained?**

Each candidate gets a plain-English reasoning string like:

> *"Strong founding-team candidate: 6.6 years experience, Applied ML Engineer at Sarvam AI, Level 4 technical depth, high execution agency. Core skill alignment: Recommendation Systems, Sentence Transformers, Weaviate. Key strength: Strong Product DNA, Evaluation Champion."*

**How do you prevent hallucinations or unsupported justifications?**

- Reasoning is template-based on actual computed signals (ATD, HEA, core IR match)
- No generative LLM involved — reasoning is derived from verifiable features
- Pipeline metadata appended: `[Pipeline: Ranker=0.787, IR_match=0.15, depth=1.00]`

**How does your solution handle inconsistent or suspicious profiles?**

| Problem | Detection |
|---------|-----------|
| Honeypot candidates | Pre-identified 293 trap IDs |
| Service-only companies | TCS/Infosys/Wipro all-career filter |
| Ghost profiles | Zero connections, no activity |
| Skill inflation | 5+ inflated keyword claims |
| Timeline fraud | Overlapping employment dates |

---

## Slide 6 — End-to-End Workflow

```
                         ┌─────────────────────┐
                         │  candidates.jsonl   │
                         │  (100,000 profiles) │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                  ┌──────│  Stage 0: Load      │
                  │      │  Features + Models  │
                  │      └──────────┬──────────┘
                  │                 ▼
                  │      ┌─────────────────────┐
                  │      │  Stage 1: Hard      │
                  │      │  Filters            │
                  │      │  ~59K removed       │
                  │      └──────────┬──────────┘
                  │                 ▼
                  │      ┌─────────────────────┐
                  │      │  Stage 2: GBM       │
                  │      │  Ensemble Score     │
                  │      │  ~40K → top 200     │
                  │      └──────────┬──────────┘
                  │                 ▼
                  │      ┌─────────────────────┐
                  │      │  Stage 3: FlashRank │
                  │      │  Rerank top 50      │
                  │      └──────────┬──────────┘
                  │                 ▼
                  │      ┌─────────────────────┐
                  │      │  Stage 4: Fusion    │
                  │      │  40/20/40 + Output  │
                  │      └──────────┬──────────┘
                  │                 ▼
                  │      ┌─────────────────────┐
                  └──────│  submission.csv     │
                         │  (top 100 ranked)   │
                         └─────────────────────┘
```

**Offline pre-computation** (one-time, ~35 min):
```
build_features.py    →  precomputed_features.npz (100K × 51 matrix)
train_ranker.py      →  ranker.xgb + ranker.lgb
```

---

## Slide 7 — System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker / Streamlit Host                     │
│                                                                 │
│  ┌──────────┐    ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Stage 1 │──▶│  Stage 2 │──▶│  Stage 3 │──▶│  Stage 4    │ │
│  │  Filter  │    │  XGB+LGB │   │ FlashRank│   │ Fusion+CSV   │ │
│  └──────────┘    └──────────┘   └──────────┘   └──────────────┘ │
│       │              │              │               │           │
│       ▼              ▼              ▼               ▼           │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │
│  │Honeypots│   │rank_v2.py│   │ TinyBERT │   │ Plotly/Stream│   │
│  │  .json  │   │  .xgb/.lgb│  │  model   │   │   lit UI     │   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────────┘   │
│                                                                 │
│  Offline: build_features.py → precomputed_features.npz          │
│           train_ranker.py   → ranker.xgb, ranker.lgb            │
└─────────────────────────────────────────────────────────────────┘
```

**Tech Stack:**
- **Language:** Python 3.11
- **ML:** XGBoost LambdaMART, LightGBM Ranker
- **NLP:** FlashRank TinyBERT, Sentence-BERT (offline only)
- **UI:** Streamlit + Plotly
- **Infra:** Docker, HuggingFace Spaces

---

## Slide 8 — Results & Performance

**What results or insights demonstrate ranking quality?**

| Metric | Value |
|--------|-------|
| Candidates processed | 100,000 |
| Viable candidates | ~20,600 (after 7 hard filters) |
| Top-1 score | 0.854 (CAND_0069905 — Sarvam AI) |
| Top-10 candidate example | Recommendation Systems, FAISS, Weaviate specialists |
| Hidden gems detected | 12 high-execution candidates |
| Zero honeypots in top 100 | ✅ |

**How does your solution meet runtime and compute constraints?**

| Constraint | Limit | Actual |
|------------|-------|--------|
| Runtime | ≤ 300s | **~8–10s** |
| Memory | ≤ 16 GB | **~2 GB** |
| CPU only | Required | **Yes** |
| No network | Required | **Yes** |
| Output rows | 100 | **Yes** |

---

## Slide 9 — Technologies Used

| Technology | Purpose | Why This |
|------------|---------|----------|
| **XGBoost LambdaMART** | Listwise ranking | Industry standard for LTR tasks with proven performance |
| **LightGBM Ranker** | Gradient-boosted ensemble | Faster training than XGBoost, good ensemble diversity |
| **FlashRank TinyBERT** | Semantic reranking | CPU-optimized cross-encoder (3.5MB model) |
| **Sentence-BERT** | Offline feature extraction | 80MB all-MiniLM-L6-v2 for profile-JD similarity |
| **Streamlit + Plotly** | Dashboard | Fast prototyping, interactive weight tuning |
| **NumPy / Pandas** | Data processing | Vectorized operations on 100K × 51 matrix |
| **Docker** | Deployment | Consistent environment across platforms |
| **HuggingFace Spaces** | Hosting | Free tier with Streamlit SDK support |

---

## Slide 10 — Submission Assets

**GitHub Repository:**
https://github.com/abhay-2108/Redrob-MatchWise

**Demo Video:**
[Link to video]

**Streamlit Sandbox:**
https://redrob-matchwise.streamlit.app/

---

## Slide 11 — Thank You

**Team Agent Mania**

- Abhay Tiwari — abhaytiwari0821@gmail.com
- Kshitij Pal — kshitijpalsinghtomar@gmail.com
- P Yes Kumar — yeskumar10507@gmail.com
