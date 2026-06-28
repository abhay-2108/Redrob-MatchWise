---
title: Redrob MatchWise
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎯 Redrob MatchWise — Multi-Stage Candidate Ranking Engine

Filters **100,000 candidate profiles** down to the **top 100 best-fit** Senior AI Engineers (Founding Team) in **~8–10 seconds** on CPU.

```
                       Stage 1                      Stage 2                Stage 3                  Stage 4
candidates ──▶  Hard Filter (7 rules) ──▶  GBM Ensemble (top 200) ──▶  FlashRank (top 50) ──▶  Score Fusion ──▶  submission.csv
(100K)            79K removed                 XGBoost 60% + LGBM 40%      TinyBERT CE                40/20/40          100 rows
```

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture-4-stage-pipeline)
- [Interactive Dashboard](#-interactive-dashboard)
- [A/B Experiment](#-ab-experiment)
- [Performance](#-performance)
- [Deployment](#-deployment)
- [Project Map](#-project-map)
- [Dependencies](#-dependencies)

---

## 🚀 Quick Start

```bash
# 1. Setup virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the ranking pipeline
python rank_v2.py --candidates ./data/candidates_backup.jsonl.gz --out ./submission.csv

# 3. Validate output format
python docs/validate_submission.py submission.csv

# 4. Launch the local interactive dashboard
streamlit run app.py
```

**Supported input formats:** `.jsonl`, `.jsonl.gz` (auto-detected and parsed efficiently in memory).

---

## 🏗️ Architecture: 4-Stage Pipeline

### Stage 0 — Load Artifacts
* Loads precomputed candidate embeddings and features (`precomputed_features.npz`).
* Loads the trained gradient boosting rankers (`ranker.xgb` and `ranker.lgb`).
* Loads the trap list of mock/fraud candidates (`honeypots.json`).

### Stage 1 — Hard Filters
We apply 7 strict filtering rules to narrow the candidate pool down to viable talent in India (excluding remote candidates unwilling to relocate, zero-skill listings, and service-firm traps):
1. **🪤 Honeypots:** Removes matches on the known trap profile list.
2. **⏳ Timeline Fraud:** Filters profiles with overlapping job periods > 90 days.
3. **🎭 Skill Inflation:** Filters profiles listing $\ge$ 5 expert skills with 0 months duration.
4. **🏢 Service Company Trap:** Filters candidates whose entire careers are at service-only companies (TCS, Infosys, Wipro, Accenture, etc.).
5. **🫥 Ghost Candidates:** Removes profiles inactive > 180 days with < 5% response rate.
6. **📉 Zero Relevant Skills:** Removes profiles with 0 skills matching the ATD taxonomy.
7. **🌍 Location / Relocation:** Filters candidates outside India who are unwilling to relocate.

*Filters reduce the active pool to **~20,600 / 100,000 viable candidates**.*

### Stage 2 — GBM Ensemble Scoring
The remaining viable candidates are scored by a multi-model ensemble:
* **XGBoost LambdaMART** (60% weight) optimized directly for NDCG.
* **LightGBM Ranker** (40% weight).

The **top 200** candidates by ensemble score advance to the semantic reranking stage.

### Stage 3 — FlashRank Cross-Encoder Reranking
The top candidates are semantically evaluated against the job description using **ms-marco-TinyBERT-L-2-v2** to gauge exact role relevance:
```
Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning,
sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS,
evaluation NDCG MRR MAP A/B testing, Python, production systems,
startup product company, Pune Noida India
```

### Stage 4 — Score Fusion
The final candidate rank is determined by combining our models and heuristics:
$$Score = 0.40 \times \text{GBM Ensemble} + 0.20 \times \text{FlashRank Reranker} + 0.40 \times \text{Heuristic}$$

* **GBM Ensemble:** Min-max normalized XGBoost + LightGBM prediction score.
* **FlashRank Reranker:** Semantic relevance match score.
* **Heuristic:** The *Singularity Engine* formula ($ATD^{1.5} \times HEA$).

---

## 📊 Interactive Dashboard

Launch the Streamlit app to explore candidates and adjust scoring settings in real-time:
```bash
streamlit run app.py
```

### Features:
* **Dataset Selection:** Switch between the 50-profile sample, the 100k full dataset, or upload custom JSONL files.
* **Real-time Tuning:** Use sidebar sliders to adjust fusion weights, filter thresholds, and rerank depths.
* **Feedback Collection:** Recruiter thumbs up/down feedback is saved to `feedback_logs.jsonl` for offline retraining.
* **A/B Testing:** Evaluate heuristic-only versus ML-fused ranking lists.
* **Interactive Export:** Download search results as CSV or Excel sheets.

---

## 📈 A/B Experiment

We compared 5 fusion weight configurations over the candidate database to evaluate Jaccard overlaps and candidate quality profiles:

| Preset | Weights (ML / Sem / Heur) | Mean Score | ATD L3+ | IR Skills | India | Hidden Gems | Companies |
|--------|:-------------------------:|:----------:|:-------:|:---------:|:-----:|:-----------:|:---------:|
| **Default (Balanced)** 🏆 | 40 / 20 / 40 | 0.496 | **100%** | **93%** | 96% | **1** | **60** |
| ML Heavy | 60 / 25 / 15 | 0.465 | 100% | 93% | 95% | 0 | 60 |
| Heuristic Heavy | 20 / 10 / 70 | **0.575** | 100% | 89% | **98%** | 1 | 60 |
| Semantic Focus | 35 / 50 / 15 | 0.404 | 100% | 93% | 95% | 0 | 60 |
| Balanced ML+ | 45 / 30 / 25 | 0.459 | 100% | 93% | 95% | 0 | 60 |

---

## ⚡ Performance

| Parameter | Limit | MatchWise Metrics |
|-----------|-------|-------------------|
| Wall-clock runtime | $\le$ 300s | **~8–10s** |
| Memory | $\le$ 16 GB | **~2 GB** |
| Hardware constraint | CPU Only | ✅ Supported |
| Network dependencies | No internet | ✅ Offline execution |
| Output integrity | Exactly 100 rows | ✅ Output validated |
| Score ordering | Monotonic | ✅ Strictly decreasing |

---

## 🌐 Deployment

The MatchWise ranking engine is deployed as a Docker container on Hugging Face Spaces:

* **Production Link:** [Redrob MatchWise on Hugging Face](https://huggingface.co/spaces/raj0120/redrob-matchwise)

### Local Container Build
To build and run the container locally:
```bash
docker build -t redrob-matchwise .
docker run -p 7860:7860 redrob-matchwise
```

---

## 📁 Project Map

```
root/
├── rank_v2.py                  # Main pipeline CLI
├── app.py                      # Streamlit dashboard
├── ab_experiment.py            # A/B weight comparison script
├── build_features.py           # Offline feature extractor
├── train_ranker.py             # Offline model training
│
├── src/
│   └── rank.py                 # Core business logic and heuristics
│
├── submission.csv              # Authoritative top-100 submission output
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # HF Spaces container config
│
├── data/
│   ├── candidates.jsonl         # Sample dataset (50 rows)
│   └── candidates_backup.jsonl.gz  # Full candidate profiles (100k rows)
│
├── artifacts/
│   ├── precomputed_features.npz  # Feature matrix
│   ├── ranker.xgb               # XGBoost model
│   ├── ranker.lgb               # LightGBM model
│   └── honeypots.json           # Mock/trap candidate list
```

---

## 📦 Dependencies

The package versions are pinned to guarantee identical execution and scoring metrics across Local and Hugging Face deployments:

| Library | Version | Role |
|---------|---------|------|
| `streamlit` | `1.58.0` | Dashboard interface |
| `numpy` | `2.4.6` | Numerical arrays and matrix mathematics |
| `xgboost` | `3.2.0` | Gradient-boosted trees ranker |
| `lightgbm` | `4.6.0` | Auxiliary LightGBM ranker |
| `sentence-transformers` | `5.5.1` | Text embedding vectorizer (offline) |
| `flashrank` | `0.2.10` | Cross-encoder semantic ranker |
| `pandas` | `3.0.3` | Data formatting |
| `openpyxl` | `3.1.5` | Excel worksheet downloads |

---

## 📄 License

Proprietary. Built for the Redrob Intelligent Candidate Discovery Hackathon.
