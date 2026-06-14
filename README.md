# Redrob MatchWise — Candidate Discovery & Ranking Engine (v2.1)

A production-grade, ultra-efficient candidate ranking system for the **Senior AI Engineer (Founding Team)** role at Redrob AI.

Processes 100,000 synthetic candidate profiles and outputs a CSV containing the **top 100 best-fit candidates** ranked in descending order of suitability under 81 seconds on CPU, completely avoiding the "keyword trap" of traditional recruiters.

---

## Quick Start

### 1. Setup & Installation
Create a virtual environment and install the lightweight requirements:
```bash
python -m venv .venv
# Activate virtual environment:
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pre-computation (Offline, One-Time)
These offline preparation steps generate metadata and LLM-written reasoning caches before the main sandbox ranking runs:

- **Generate the honeypot filter list:**
  ```bash
  python identify_all_honeypots.py --candidates ./docs/candidates.jsonl --out ./honeypots.json
  ```
- **Precompute the high-quality LLM reasoning cache:**
  ```bash
  # Using Google Gemini API (requires GEMINI_API_KEY environment variable set)
  python precompute_reasoning.py --mode gemini --top-n 500
  ```

### 3. Run the Sandbox Ranker
Executes the main ranking script (pure Python standard library math at runtime):
```bash
python rank.py --candidates ./docs/candidates.jsonl --out ./submission.csv
```
**Runtime:** ~80 seconds on CPU | **Memory:** ~200 MB | **Dependencies:** Pure Python standard library.

### 4. Validate the Submission
Verify that the output is formatted correctly and complies with the challenge specifications:
```bash
python docs/validate_submission.py ./submission.csv
```

### 5. Launch the Sandbox Demo Dashboard
Explore candidate distributions and adjust weights dynamically:
```bash
streamlit run app.py
```
- Open `http://localhost:8501` to access the interactive user interface.

---

## Architecture: The Singularity Engine

Rather than using linear weighted addition (which rewards keyword stuffing), the engine operates on two orthogonal axes and multiplies them:

$$\text{Final Score} = (\text{ATD}^{1.5}) \times \text{HEA}$$

```
   candidates.jsonl ──► [ Hard Filters ] ──► [ AXIS A: ATD ]
         (100K)          ├─ Honeypots             └─ 4-level taxonomy
                         ├─ Service-only          └─ Synonym stemming
                         └─ Unrelated titles              │
                                                          ▼
                                                    [ AXIS B: HEA ]  ──► Sort & output
                                                      (14 continuous     top 100 CSV
                                                      signals & fraud)
```

### 1. Axis A — Absolute Technical Dominance (ATD)
Evaluates candidates using a strict 4-level difficulty hierarchy of AI engineering depth:
* **Level 4 (1.00):** Custom GPU kernels (`cuda`, `triton`), distributed training architectures (`deepspeed`, `megatron`), and serving optimization (`vllm`).
* **Level 3 (0.70):** Applied SOTA techniques (`fine-tuning llms`, `lora`, `peft`), evaluation frameworks (`ndcg`, `mrr`), vector DBs (`faiss`, `pinecone`, `qdrant`), and hybrid search.
* **Level 2 (0.40):** Standard frameworks (`pytorch`, `tensorflow`), NLP (`nltk`, `spacy`), and basic RAG.
* **Level 1 (0.15):** API callers (`openai api`, `chatgpt`) and wrapper tools (`langchain`).

The $ATD^{1.5}$ exponent ensures a Level-4 candidate scores **~17x higher** than a Level-1 wrapper engineer before HEA modifiers, rendering keyword-stuffed wrapper profiles non-viable for the founding engineer role.

* **Advanced Synonym Stemming:** Uses a zero-dependency canonicalization function (`canonicalize_skill`) to parse spelling variants and abbreviations (e.g. `recsys` -> `recommendation systems`, `vector db` -> `vector database`, `fine tuning` -> `fine-tuning llms`) to maximize recall.

### 2. Axis B — High Execution Agency (HEA)
A continuous multiplicative modifier mapping 14 career structure and behavioral signals:
* **Experience Fit:** Continuous Gaussian curve peaking at `7` years.
* **Active Recency & Notice Period:** Smooth continuous sigmoid decay curves to penalize inactive ghost profiles and long notice periods.
* **Chaos Tolerance & Startup DNA:** Multipliers for experience in small-scale startup environments.
* **Product DNA:** Penalizes pure outsourcing/consulting company histories and rewards product experience.
* **GitHub & Engagement Signals:** Continuous functions for GitHub activity, interview completion rates, and recruiter response rates.

### 3. Integrated Fraud Scanners
- **Timeline Overlapping Check:** Detects and penalizes resumes with overlapping full-time stint dates (simultaneous stints > 90 days).
- **Buzzword Stuffing Penalty:** Penalizes profiles with an unnaturally high skill count relative to their total experience.
- **Title Inflation Detector:** Flags profiles claiming senior leadership titles (VP, Lead, Principal, Chief) with under 4 years of total experience.

---

## Streamlit Dashboard Features
- **Weight Calibration:** recasting sliders to adjust priorities (e.g. increase notice period penalty, dial up GitHub activity importance, or modify startup experience weights).
- **Interactive Scatter Chart:** Plots candidate positioning in real-time, showing Technical Floor (ATD) vs. Execution Agency (HEA) with bubble sizing representing their final rank.

---

## Compute & Compliance Constraints

| Constraint | Limit | Actual | Status |
|------------|-------|--------|--------|
| **Runtime** | ≤ 300 seconds | **~80 seconds** | Passed |
| **Memory** | ≤ 16 GB | **~200 MB** | Passed |
| **Compute** | CPU Only | **CPU Only** | Passed |
| **Network** | Offline during ranking | **Offline** | Passed |
| **Disk Space**| ≤ 5 GB | **~50 MB** | Passed |

---

## AI Tools Used
- **Claude / Gemini**: Used for architectural optimization, continuous mathematical curve design, and dashboard layout planning.
- No candidate data was sent to any external model during the main sandboxed ranking runs.
