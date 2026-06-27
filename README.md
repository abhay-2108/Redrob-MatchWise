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

---

## 📊 Streamlit Dashboard

- **Pipeline breakdown** — see each stage's timing & output
- **Weight sliders** — tweak fusion weights live
- **Candidate explorer** — search by ID, inspect full score breakdown
- **Honeypot detector** — list filtered candidates with reasons

```bash
streamlit run app.py    # → http://localhost:8501
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
build_features.py       # Offline: 51-feature extraction
train_ranker.py         # Offline: model training
rank.py                 # Library: taxonomy & ATD/HEA helpers
precomputed_features.npz # 100K × 51 feature matrix
ranker.xgb / ranker.lgb # Trained models
submission.csv          # Output: top 100 ranked
requirements.txt        # Dependencies
Dockerfile              # Container config
```

---


