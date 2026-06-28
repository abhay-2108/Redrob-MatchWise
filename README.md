---
title: Redrob MatchWise
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🎯 Redrob MatchWise — Multi-Stage Candidate Ranking Engine

A production-grade, CPU-only pipeline that sifts through **100,000 candidate profiles** to surface the **top 100 best-fit** Senior AI Engineers — all in **~8–10 seconds**.

```
candidates  ──▶  Stage 1: Hard Filter  ──▶  Stage 2: GBM Ensemble    ──▶  Stage 3: FlashRank Rerank  ──▶  Stage 4: Fusion ──▶  top 100
(100K)           (~79K removed)              (top 200)                        (top 50)                             (40/20/40)        submission.csv
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

# 3. Run the pipeline (~8–10 seconds)
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
| 🌍 Location/Relocation | 50,498 |

**~20.6K / 100K** remain — the viable pool. The location filter (added in the latest pipeline update) is the second-largest filter and ensures international candidates without relocation intent are excluded.

### Stage 2 — GBM Ensemble Scoring
Two gradient-boosted rankers (XGBoost 60% + LightGBM 40%) score all ~20.6K viable candidates. **Top 200** advance.

### Stage 3 — Cross-Encoder Rerank
FlashRank TinyBERT (ms-marco-TinyBERT-L-2-v2) reranks the top 50 against the job description — a deep semantic relevance check that keyword search can't match.

### Stage 4 — Score Fusion

```
Final Score = 0.40 × GBM Ensemble  +  0.20 × FlashRank  +  0.40 × Heuristic
```

Where **GBM Ensemble** = XGBoost LambdaMART (60%) + LightGBM (40%).

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
- **Multi-format export** — download results as CSV or Excel (.xlsx) with a single click

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
| Runtime | ≤ 300s | **~8–10s** (measured: 7.4–12.3s across 3 runs) |
| Memory | ≤ 16 GB | **~2 GB** |
| CPU only | Required | ✅ |
| No network | Required | ✅ |
| Output rows | 100 | ✅ |
| Monotonic scores | Required | ✅ |
| Deterministic | Required | ✅ (3 runs, byte-identical) |

---

## 🌐 Demo Deployments

| Platform | Link | ML Pipeline? | Status |
|----------|------|-------------|--------|
| **Hugging Face Spaces** | [redrob-matchwise.hf.space](https://huggingface.co/spaces/raj0120/redrob-matchwise) | Partial — running older artifacts | ❌ **Out of date** — missing latest model retraining, location filter, and pipeline fixes. Only ~83% candidate overlap with the authoritative pipeline. Needs redeployment. |
| **Streamlit Cloud** | [redrob-matchwise.streamlit.app](https://redrob-matchwise.streamlit.app/) | ✅ Full ML | ⚠️ **Slightly outdated** — same 100 candidates but fusion scores differ by ~1.5% max. Needs model artifact refresh. |

### Why results differ across deployments

Our latest pipeline includes changes made *after* these deployments:

| Change | Local | Streamlit Cloud | HuggingFace |
|--------|:-----:|:---------------:|:-----------:|
| XGBoost + LightGBM retraining | ✅ | ❌ (older models) | ❌ |
| Location hard filter | ✅ | ✅ | ❌ |
| Fusion weight tuning | ✅ | ✅ | ❌ |
| Candidate overlap vs local | 100% | 100% | **83.5%** |
| Score correlation vs local | 1.0000 | 0.9996 | 0.9432 |

### Running the authoritative pipeline locally

```bash
# This produces the actual submission output
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
python docs/validate_submission.py submission.csv
```

The pipeline is **fully deterministic** — running it 3 times on the same input produces byte-identical output. See `submission.csv` for the latest authoritative result.

---

## 📁 Project Map

```
rank_v2.py              # Main pipeline (CLI entry point) — multi-stage ranking
app.py                  # Streamlit dashboard — tuning, export (CSV + Excel), feedback
ab_experiment.py        # A/B experiment runner (5 weight presets)
build_features.py       # Offline: 51-feature extraction
train_ranker.py         # Offline: XGBoost + LightGBM training
src/rank.py             # Library: taxonomy, ATD/HEA helpers, heuristic engine
precomputed_features.npz # 100K × 51 feature matrix
ranker.xgb / ranker.lgb # Trained models (XGBoost + LightGBM)
honeypots.json          # 293 known-fake candidate IDs
submission.csv          # Output: top 100 ranked (tracked in git)
requirements.txt        # Dependencies (openpyxl for Excel export)
Dockerfile              # Container config
experiments/            # A/B comparison CSVs & summary
```

---


