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

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline (6 seconds)
python rank_v2.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. Validate
python docs/validate_submission.py submission.csv

# 4. Launch dashboard
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


