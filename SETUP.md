# SETUP & RUN GUIDE — Redrob MatchWise

This guide provides step-by-step instructions to set up and run the **Redrob MatchWise** candidate ranking engine from scratch.

---

## 1. Prerequisites
Ensure you have Python 3.10+ installed on your system. Using `uv` is recommended for fast package installation, but standard `pip` works perfectly.

---

## 2. Installation & Environment Setup

### Step 2.1: Clone the Repository
Clone the project repository and enter the project folder:
```bash
git clone https://github.com/abhay-2108/Redrob-MatchWise.git
cd Redrob-MatchWise
```

### Step 2.2: Create a Virtual Environment
Initialize a Python virtual environment:
```bash
# Using uv (fast)
uv venv .venv

# Or using standard python
python -m venv .venv
```

### Step 2.3: Activate the Environment
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (cmd):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 2.4: Install Dependencies
```bash
# Using uv (fast)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

---

## 3. Running the Ranking Pipeline

### Step 3.1: Run Honeypot Scanner (Offline Step)
Scans candidate profiles for calendar inconsistencies and invalid tenure claims:
```bash
python identify_all_honeypots.py --candidates ./docs/candidates.jsonl --out ./honeypots.json
```
*Generates `honeypots.json` containing pre-filtered anomalous candidate IDs.*

### Step 3.2: Run the Main Ranker (Sandbox Entry Point)
Executes candidate evaluation based on the AI Technical Depth (ATD) and Hiring & Execution Availability (HEA) metrics:
```bash
python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv
```
* **Execution Time:** ~20 seconds on CPU.
* **Output:** Generates `submission.csv` containing the top 100 best-fit candidates.

---

## 4. Verification & Validation

Verify that the output file meets the format, sorting, and constraint specifications of the hackathon:
```bash
python docs/validate_submission.py submission.csv
```
* **Expected Output:** `Submission is valid.`

---

## 5. Running the Web UI Demo (Streamlit)

Start the interactive dashboard app to upload custom candidate samples (e.g. `docs/sample_candidates.json`) and visually inspect the scored, ranked list with reasoning:
```bash
streamlit run app.py
```
* **Access URL:** Opens in your browser at `http://localhost:8501`.
