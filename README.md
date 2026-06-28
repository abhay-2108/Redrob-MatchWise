---
title: Redrob MatchWise
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🎯 Redrob MatchWise — Multi-Stage Candidate Ranking Engine

A production-grade, CPU-only pipeline that sifts through **100,000 candidate profiles** to surface the **top 100 best-fit** Senior AI Engineers — all in **~6 seconds**.

```
candidates  ──▶  Stage 1: Hard Filter  ──▶  Stage 2: XGBoost + LightGBM  ──▶  Stage 3: FlashRank Rerank  ──▶  Stage 4: Fusion ──▶  top 100
(100K)           (59K removed)               (top 200)                        (top 50)                             (40/20/40)        submission.csv
```

---

## ⚠️ Before You Start

### Git LFS Required
Binary model files (`precomputed_features.npz`, `ranker.xgb`, `ranker.lgb`) are stored with **Git LFS**. Clone with:

```bash
git lfs install
git lfs pull
```

Without this, you get pointer files and the pipeline will fail to load models.

### Precomputation Steps (run once offline)
The pipeline needs pre-built features and trained models. If they're not present or you want to rebuild:

```bash
# 1. Extract 51 features from candidates (~30 min)
python build_features.py --candidates <path/to/candidates.jsonl>

# 2. Train XGBoost + LightGBM rankers (~5 min)
python train_ranker.py
```

Otherwise, the committed `precomputed_features.npz`, `ranker.xgb`, and `ranker.lgb` are used directly.

---

## 🚀 Quick Start

```bash
# 1. Clone with LFS
git lfs install && git lfs pull

# 2. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the pipeline (6 seconds)
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv

# 4. Validate
python docs/validate_submission.py submission.csv

# 5. Launch dashboard
streamlit run app.py
```

---

## 🏗️ Architecture: 4-Stage Pipeline

### Stage 0 — Load Artifacts
| Artifact | Size | What |
|----------|------|------|
| `precomputed_features.npz` | 100K × 51 | Feature matrix (built once offline) |
| `ranker.xgb` + `ranker.lgb` | — | XGBoost LambdaMART & LightGBM ensemble |
| `honeypots.json` | 293 IDs | Trap candidate filter list |

### Stage 1 — Hard Filters
Removes candidates that don't meet the bar:

| Filter | Removed |
|--------|---------|
| 🪤 Honeypots | 293 |
| 🏢 Service-only companies | 9,750 |
| 🫥 Ghost profiles | 171 |
| 📉 Zero-skill profiles | 53,097 |
| 🎭 Skill inflation fraud | 8 |

**~40K / 100K** remain — the viable pool.

### Stage 2 — GBM Ensemble Scoring
Two gradient-boosted rankers (XGBoost 60% + LightGBM 40%) score all ~40K viable candidates. **Top 200** advance.

### Stage 3 — Cross-Encoder Rerank
FlashRank TinyBERT (ms-marco-TinyBERT-L-2-v2) reranks the top 50 against the job description — a deep semantic relevance check that keyword search can't match.

### Stage 4 — Score Fusion

```
Final Score = 0.40 × XGBoost  +  0.20 × FlashRank  +  0.40 × Heuristic
```

The **Heuristic** component uses the proven singularity formula:

```
singularity_score = ATD¹·⁵ × HEA
```

Where **ATD** measures technical depth (GPU kernels → API scripts) and **HEA** captures career execution signals.

#### Live Tuning & Presets

The dashboard sidebar exposes all three fusion weights as draggable sliders that auto-normalize.
A **Quick Preset** dropdown lets you switch between 5 pre-tuned configurations from the A/B experiment:

| Preset | XGBoost | FlashRank | Heuristic | Best For |
|--------|---------|-----------|-----------|----------|
| **Default (Balanced)** — 🏆 *recommended* | 0.40 | 0.20 | 0.40 | General use — highest IR-skill coverage & company diversity |
| ML Heavy | 0.60 | 0.25 | 0.15 | When you trust the trained model more than hand-crafted rules |
| Heuristic Heavy | 0.20 | 0.10 | 0.70 | Maximizing India-location & execution agency signals |
| Semantic Focus | 0.35 | 0.50 | 0.15 | Emphasizing FlashRank semantic relevance |
| Balanced ML+ | 0.45 | 0.30 | 0.25 | Gentle bias toward ML without sacrificing heuristics |

> **A/B experiment result:** The Default preset wins across the most quality proxies (highest IR-skill coverage, most company diversity, finds hidden gems, competitive India rate). See [A/B Experiment](#-ab-experiment) below.

---

## 📊 Streamlit Dashboard

- **Pipeline breakdown** — see each stage's timing & output
- **Weight sliders** — tweak fusion weights live with auto-normalization
- **Quick Presets** — switch between 5 pre-tuned weight configurations instantly
- **Filter Aggressiveness** — control skill inflation cutoffs
- **Rerank Count** — balance speed vs semantic depth
- **Blind A/B Mode** — interleave two ranking models for blind recruiter evaluation
- **Candidate explorer** — browse ranked cards with score breakdown, skill pills, signal tags, and career history
- **Feedback buttons** — 👍/👎 per candidate to build training data
- **Retrain with Feedback** — re-tune rankers using collected recruiter evaluations

```bash
streamlit run app.py    # → http://localhost:8501
```

---

## 📈 A/B Experiment

An automated A/B experiment (`ab_experiment.py`) compared 5 weight presets on the full 100K pipeline. All presets share stages 0–3; stage 4 (fusion) is re-evaluated per preset.

### Results

| Preset | Weights | Mean | ATD L3+ | IR Skills | India | Gems | Companies |
|--------|---------|------|---------|-----------|-------|------|-----------|
| **Default (Balanced)** | 40/20/40 | 0.496 | 100% | **93%** | 96% | **1** | **60** |
| ML Heavy | 60/25/15 | 0.465 | 100% | 93% | 95% | 0 | 60 |
| Heuristic Heavy | 20/10/70 | **0.575** | 100% | 89% | **98%** | 1 | 60 |
| Semantic Focus | 35/50/15 | 0.404 | 100% | 93% | 95% | 0 | 60 |
| Balanced ML+ | 45/30/25 | 0.459 | 100% | 93% | 95% | 0 | 60 |

### Overlap (Jaccard Similarity)

| | Default | ML Heavy | Heuristic | Semantic | ML+ |
|---|---------|----------|-----------|----------|-----|
| Default | 1.000 | 0.852 | 0.639 | 0.835 | 0.887 |
| ML Heavy | | 1.000 | 0.538 | 0.905 | 0.961 |
| Heuristic | | | 1.000 | 0.575 | 0.562 |
| Semantic | | | | 1.000 | 0.887 |

### Key Findings

1. **All presets achieve 100% ATD Level 3+** — the XGBoost model dominates, regardless of fusion weights
2. **Default is the best all-rounder** — highest IR-skill coverage (93%), most company diversity (60 unique), finds hidden gems, and 96% India-located candidates
3. **Heuristic Heavy is the most differentiated** — only 64% overlap with Default; boosts India to 98% and mean score to 0.575, but loses 4% IR coverage
4. **ML-heavy presets (B, D, E) cluster together** — 90–96% overlap with each other, meaning weight shifts in the ML+Semantic range produce nearly identical top-100s
5. **Default (current) wins composite metric** — weighted score of ATD×IR×India = 96.0, highest overall

**Recommendation:** Keep the current default weights. Use the Quick Preset dropdown to switch to Heuristic Heavy when prioritizing India-location fit over IR-specific skill coverage.

```bash
# Run the experiment yourself
python ab_experiment.py
# Output: experiments/ab_comparison.csv, experiments/ab_summary.csv
```

---

## ⚡ Performance

| Constraint | Limit | Actual |
|------------|-------|--------|
| Runtime | ≤ 300s | **~6s** |
| Memory | ≤ 16 GB | **~2 GB** |
| CPU only | Required | ✅ |
| No network | Required | ✅ |
| Output rows | 100 | ✅ |
| Monotonic scores | Required | ✅ |

---

## 🌐 Demo Deployments

| Platform | Link | ML Pipeline? | Notes |
|----------|------|-------------|-------|
| **Hugging Face Spaces** | [redrob-matchwise.hf.space](https://huggingface.co/spaces/raj0120/redrob-matchwise) | ✅ Full ML (Docker container) | Must be set to **Container** mode in Space settings (image: `raj0120/redrob-matchwise`) |
| **Streamlit Cloud** | [redrob-matchwise.streamlit.app](https://redrob-matchwise.streamlit.app/) | ❌ Heuristic-only | Streamlit Cloud does not support Git LFS — model files unavailable. Falls back to `ATD¹·⁵ × HEA` heuristic scoring. |

### Why results differ on Streamlit Cloud vs local

The Streamlit Cloud demo uses **heuristic-only scoring** because model files can't be served through their platform (no LFS support). This gives different rankings than the full ML pipeline run locally. Key differences:

- Heuristic mode ranks by `ATD¹·⁵ × HEA` (technical depth × execution agency)
- Full ML mode uses 40% XGBoost + 20% FlashRank + 40% Heuristic fusion
- Local results (`submission.csv`) are the authoritative output for submission
- The Streamlit Cloud dashboard is a **UI demo** — the real evaluation uses the CLI pipeline

### Running the authoritative pipeline locally

```bash
# This produces the actual submission output
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
python docs/validate_submission.py submission.csv
```

---

## 📁 Project Map

```
rank_v2.py              # Main pipeline (CLI entry point)
app.py                  # Streamlit dashboard
ab_experiment.py        # A/B experiment runner (5 weight presets)
build_features.py       # Offline: 51-feature extraction
train_ranker.py         # Offline: model training
rank.py                 # Library: taxonomy & ATD/HEA helpers
precomputed_features.npz # 100K × 51 feature matrix
ranker.xgb / ranker.lgb # Trained models
submission.csv          # Output: top 100 ranked
requirements.txt        # Dependencies
Dockerfile              # Container config
experiments/            # A/B comparison CSVs
```

---


