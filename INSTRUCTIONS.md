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

---

## 3. Data Preparation

Place the candidate pool file inside the `docs/` directory:
* Ensure your candidate dataset is saved at: `docs/candidates.jsonl` (or `docs/candidates.jsonl.gz`).

---

## 4. Pre-computation (Offline, Local Development Only)

**Note for Hackathon Deployment (Hugging Face Spaces):** You do *not* need to run these steps in your sandbox, nor do you need to upload the massive 100K `candidates.jsonl` file to your codebase! We have already executed these steps locally and included the resulting files (like `honeypots.json`) in the repository. The judges will use the Streamlit UI's File Uploader to test their own candidate files.

Before running the ranker locally for the first time, you must generate the honeypot list and can optionally pre-compute the high-quality LLM reasoning cache for the top candidates. These steps run offline (once) and do not count toward the 5-minute sandbox budget.

### Step 4.1: Identify Honeypots & Anomalies
Scans profiles for tenure contradictions or impossible credentials:
```bash
python identify_all_honeypots.py --candidates ./docs/candidates.jsonl --out ./honeypots.json
```
*Output: Generates `honeypots.json` (identifies 293 anomalous candidate IDs to be hard-filtered).*

### Step 4.2: Pre-compute LLM Reasoning Cache
Generates high-quality, factual reasoning strings for the top candidates using Google Gemini API (recommended) or a local model. This cache is saved as `reasoning_cache.json` and read by the ranker during runtime:
```bash
# Using Gemini API (requires GEMINI_API_KEY env variable set)
python precompute_reasoning.py --mode gemini --top-n 500

# Or using a local model (requires installing transformers and torch)
python precompute_reasoning.py --mode local --top-n 300
```
*Output: Generates `reasoning_cache.json` containing detailed, non-templated reasoning explanations.*

---

## 5. Running the Ranking Pipeline (Stage 3 Sandbox Entry Point)

Run the main ranker script. This command performs hard filtering (honeypots, service-only careers, unrelated titles), evaluates candidates using the hierarchical AI difficulty taxonomy (ATD) and the 14-signal execution agency multiplier (HEA), resolves tiebreaks continuously, loads the pre-computed reasoning cache (falling back to taxonomy-based generator if missing), and writes the final CSV submission.

```bash
python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv
```
* **Expected Runtime:** ~20 seconds on CPU.
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
