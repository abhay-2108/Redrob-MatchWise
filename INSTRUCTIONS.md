# Redrob MatchWise — Step-by-Step Execution Guide

This guide explains how to set up and run the **Redrob MatchWise** candidate ranking system from scratch on a clean machine.

---

## 1. Prerequisites
Ensure you have Python 3.10+ installed on your system. Using `uv` (a fast Python package installer) is highly recommended, but standard `pip` works perfectly.

---

## 2. Setup & Installation

### Step 2.1: Clone the Repository
Clone your project repository and navigate into the root directory:
```bash
git clone https://github.com/abhay-2108/Redrob-MatchWise.git
cd Redrob-MatchWise
```

### Step 2.2: Create a Virtual Environment
Create a clean virtual environment in `.venv/`:
```bash
# Using uv (recommended for speed)
uv venv .venv

# Or using standard python
python -m venv .venv
```

### Step 2.3: Activate the Virtual Environment
Activate the environment based on your OS:
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 2.4: Install Dependencies
Install the required packages from `requirements.txt`:
```bash
# Using uv (fast)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

*(Optional CPU Optimization: If PyTorch installs with GPU-CUDA support on a non-GPU machine, you can force install the lighter CPU-only package)*:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Data Preparation

Place the candidate pool file inside the `docs/` directory:
* Ensure your candidate dataset is saved at: `docs/candidates.jsonl` (or `docs/candidates.jsonl.gz`).

---

## 4. Pre-computation (Offline Steps)

Before running the ranker, you must generate the honeypot list and the dense semantic vectors. These steps run offline (once) and do not count toward the 5-minute sandbox budget.

### Step 4.1: Identify Honeypots & Anomalies
Scans profiles for tenure contradictions or impossible credentials:
```bash
python identify_all_honeypots.py --candidates ./docs/candidates.jsonl --out ./honeypots.json
```
*Output: Generates `honeypots.json` (identifies 293 anomalous candidate IDs to be hard-filtered).*

### Step 4.2: Generate Candidate & JD Embeddings
Pre-computes sentence embeddings for all viable candidates using the `all-MiniLM-L6-v2` transformer model:
```bash
python precompute_embeddings.py --candidates ./docs/candidates.jsonl
```
*Output: Generates `candidate_embeddings.npy`, `jd_embedding.npy`, and `candidate_ids.json` in the root folder (takes ~19 minutes on CPU).*

---

## 5. Running the Ranking Pipeline (Stage 3 Sandbox Entry Point)

Run the main ranker script. This command performs hard filtering, loads pre-computed embeddings, tokenizes candidate profiles to build a BM25 Okapi index, scores candidates, and generates the final CSV submission.

```bash
python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv
```
* **Expected Runtime:** ~23 seconds on CPU.
* **Output:** Writes the top 100 ranked candidates to `submission.csv`.

---

## 6. Validate the Submission Format

Run the hackathon validator script against the output file to ensure full compliance with the submission specifications:
```bash
python docs/validate_submission.py submission.csv
```
* **Expected Output:** `Submission is valid.` (Confirms correct row length, column order, score ordering, tiebreaking, and 0% honeypot presence).

---

## 7. Launch the Local Streamlit Sandbox Demo

Run the interactive dashboard app to test the ranker using sample uploads (or a custom candidates subset):
```bash
streamlit run app.py
```
* **Access URL:** Opens a local window at `http://localhost:8501`.
* **Testing:** Drag and drop `docs/sample_candidates.json` to verify live scoring, ranking, and reasoning displays in a premium UI.
