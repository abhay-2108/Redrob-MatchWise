# 📋 Redrob MatchWise — Step-by-Step Execution Guide

How to set up and run the ranking pipeline from scratch, end to end.

---

## 🧰 Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.10+ |
| RAM | 16 GB (recommended) |
| Disk | ~500 MB for dataset + models |
| Network | Only during `pip install` & first model download |
| GPU | Not needed — runs on CPU |

---

## ⚙️ Setup

```bash
# Clone & enter the repo
git clone https://github.com/abhay-2108/Redrob-MatchWise.git
cd Redrob-MatchWise

# Create virtual environment
python -m venv .venv

# Activate it
#   Windows (PowerShell): .venv\Scripts\Activate.ps1
#   macOS / Linux:        source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

There are **two modes** depending on what stage you're at.

### A) Quick Reproduce (5–10 seconds)

If you already have the pre-computed files (`precomputed_features.npz`, `ranker.xgb`, `ranker.lgb`), just run:

```bash
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
```

The pipeline loads pre-built artifacts, filters, scores, reranks, and writes the top 100 to `submission.csv`.

### B) Full Rebuild from Scratch (~35 minutes)

If you need to rebuild the feature matrix and models:

```bash
# Step 1 — Extract 51 features for every candidate
python build_features.py --candidates <path/to/candidates.jsonl>

# Step 2 — Train XGBoost LambdaMART + LightGBM rankers
python train_ranker.py

# Step 3 — Run the ranking pipeline
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
```

---

## ✅ Validating the Output

```bash
python docs/validate_submission.py submission.csv
```

Expected result: `Submission is valid.`

The check ensures:
- Exactly 100 rows
- Ranks 1–100 (unique)
- Scores monotonically non-increasing
- Correct column names & format

---

## 🖥️ Launching the Dashboard

```bash
streamlit run app.py
```

Opens at **[http://localhost:8501](http://localhost:8501)** — explore candidates, adjust weights, inspect scores.

---

## 📦 What Each File Does

| File | Role |
|------|------|
| `rank_v2.py` | 🏁 Main pipeline — CLI entry point |
| `app.py` | 🎨 Streamlit web dashboard |
| `build_features.py` | 🔨 Offline feature engineering |
| `train_ranker.py` | 🧠 Offline model training |
| `src/rank.py` | 📚 Library: taxonomy constants, ATD/HEA, singularity engine |
| `precomputed_features.npz` | 📐 100K×51 feature matrix (pre-built) |
| `ranker.xgb` / `ranker.lgb` | 🤖 Trained ranking models |
| `submission.csv` | 📄 Output: top 100 candidates |
| `requirements.txt` | 📃 Python dependencies |
| `docs/validate_submission.py` | ✅ Hackathon format validator |

---

## 🔁 Reproduce Command

One-liner for the hackathon submission portal:

```bash
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
```
