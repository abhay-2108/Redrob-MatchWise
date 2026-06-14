#!/usr/bin/env python3
"""
Redrob Hackathon — Honeypot & Anomaly Detection Utility
========================================================
Pre-computation step (runs offline, outside the 5-minute sandbox window).

Scans the full candidate pool and flags profiles with structurally
impossible claims.  Output is a JSON file mapping candidate_id → list
of anomaly reasons.  The ranking script loads this file at runtime to
hard-filter honeypots from the top-100 output.

Detected anomaly types
----------------------
1. Job duration (months) exceeds the actual calendar span between
   start_date and end_date (or the reference date for current roles).
2. Candidate claims employment at a startup *before* its real-world
   founding year (e.g. working at Krutrim in 2019, founded 2023).
3. Skills listed as "expert" proficiency with 0 months of duration —
   an impossible combination that signals a fabricated profile.

Usage
-----
    python identify_all_honeypots.py [--candidates PATH] [--out PATH]
"""

import argparse
import gzip
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Reference date — used to compute calendar span for current (open-ended) jobs
# ---------------------------------------------------------------------------
REF_DATE = datetime(2026, 6, 14)

# ---------------------------------------------------------------------------
# Real-world founding years for startups present in the dataset
# ---------------------------------------------------------------------------
REAL_FOUNDING_YEARS: dict[str, int] = {
    "Krutrim":      2023,
    "Sarvam AI":    2023,
    "CRED":         2018,
    "Rephrase.ai":  2019,
    "Glance":       2019,
}


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse an ISO-8601 date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def detect_anomalies(candidate: dict) -> list[str]:
    """Return a list of anomaly reason strings for a single candidate.

    An empty list means the candidate appears structurally sound.
    """
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    years_exp = profile.get("years_of_experience", 0)

    # ------------------------------------------------------------------
    # Anomaly 1 — Job duration exceeds actual calendar span
    # ------------------------------------------------------------------
    for job in career:
        start_dt = _parse_date(job.get("start_date"))
        end_dt   = _parse_date(job.get("end_date")) if job.get("end_date") else REF_DATE
        reported = job.get("duration_months", 0)
        if start_dt and end_dt:
            actual = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if reported > actual + 1:               # 1-month rounding grace
                reasons.append(
                    f"Job at {job.get('company')}: reported {reported} months "
                    f"but calendar span is only {actual} months "
                    f"({job.get('start_date')} → {job.get('end_date') or 'present'})"
                )

    # ------------------------------------------------------------------
    # Anomaly 2 — Single job duration exceeds total stated experience
    # ------------------------------------------------------------------
    for job in career:
        dur_years = job.get("duration_months", 0) / 12.0
        if dur_years > years_exp + 0.5:             # 6-month grace
            reasons.append(
                f"Job at {job.get('company')} lasted {dur_years:.1f} years, "
                f"exceeding total stated experience of {years_exp:.1f} years"
            )

    # ------------------------------------------------------------------
    # Anomaly 3 — Employment predates company founding
    # ------------------------------------------------------------------
    for job in career:
        company = job.get("company", "")
        if company in REAL_FOUNDING_YEARS:
            start_str = job.get("start_date", "")
            try:
                start_year = int(start_str.split("-")[0])
            except (ValueError, IndexError, AttributeError):
                continue
            founding = REAL_FOUNDING_YEARS[company]
            if start_year < founding:
                reasons.append(
                    f"Claims employment at {company} in {start_year}, "
                    f"but the company was founded in {founding}"
                )

    # ------------------------------------------------------------------
    # Anomaly 4 — "Expert" proficiency with zero duration
    # ------------------------------------------------------------------
    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if expert_zero:
        reasons.append(
            f"Expert proficiency with 0 months duration on: "
            f"{', '.join(expert_zero)}"
        )

    return reasons


def scan_candidates(candidates_path: str) -> dict[str, list[str]]:
    """Scan the full candidate file and return a dict of anomalous IDs."""
    anomalies: dict[str, list[str]] = {}

    opener = gzip.open if candidates_path.endswith(".gz") else open
    with opener(candidates_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            cid     = candidate.get("candidate_id", "")
            reasons = detect_anomalies(candidate)
            if reasons:
                anomalies[cid] = reasons

    return anomalies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect honeypot / anomalous candidates in the Redrob dataset."
    )
    parser.add_argument(
        "--candidates",
        default="./docs/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./honeypots.json",
        help="Output JSON file mapping candidate_id → anomaly reasons",
    )
    args = parser.parse_args()

    # Resolve candidates path — try .jsonl first, then .jsonl.gz
    cpath = args.candidates
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
        else:
            print(f"ERROR: Candidate file not found at {cpath} or {alt}")
            raise SystemExit(1)

    print(f"Scanning candidates from: {cpath}")
    anomalies = scan_candidates(cpath)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(anomalies, fh, indent=2, ensure_ascii=False)

    print(f"Detected {len(anomalies)} anomalous candidates -> {args.out}")

    # Summary by anomaly type
    type_counts: dict[str, int] = {}
    for reasons in anomalies.values():
        for r in reasons:
            if "calendar span" in r:
                key = "duration > calendar span"
            elif "total stated experience" in r:
                key = "duration > total experience"
            elif "founded in" in r:
                key = "predates founding year"
            elif "Expert proficiency" in r:
                key = "expert skill with 0 duration"
            else:
                key = "other"
            type_counts[key] = type_counts.get(key, 0) + 1

    print("\nAnomaly breakdown:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
