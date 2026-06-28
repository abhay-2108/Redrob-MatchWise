# ⚡ SETUP & RUN GUIDE — Redrob MatchWise

Fast-track instructions from zero to a ranked candidate list.

---

## 1. Environment

```bash
python -m venv .venv
# Activate:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Reproduce Submission

```bash
python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
```

⏱️ ~8–10 seconds on CPU.

## 3. Validate

```bash
python docs/validate_submission.py submission.csv
```

## 4. Launch UI

```bash
streamlit run app.py
```

---

## Offline Rebuild (only if needed)

```bash
python build_features.py --candidates <path/to/candidates.jsonl>   # ~30 min
python train_ranker.py                                                # ~5 min
```

---

## Resource Footprint

| Metric | Requirement | Actual |
|--------|-------------|--------|
| Runtime | ≤ 5 min | **~8–10 sec** |
| Memory | ≤ 16 GB | **~2 GB** |
| CPU only | Required | **Yes** |
| No network | Required | **Yes** |
| Output | 100 rows | **Yes** |
| Deterministic | Required | **Yes** |
