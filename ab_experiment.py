#!/usr/bin/env python3
"""
A/B Experiment -- Score Fusion Weight Presets
=============================================
Compares 5 weight configurations on the full 100K pipeline.
Stages 0-3 run once; Stage 4 (fusion) is re-evaluated per preset.

Usage:
    .\.venv\Scripts\python.exe ab_experiment.py

Output:
    - experiments/ab_comparison.csv   -- per-preset top-100 rows
    - experiments/ab_summary.csv      -- aggregate comparison table
    - Console summary with metrics
"""

import csv
import gzip
import json
import os
import sys
import time

import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from build_features import FEATURE_NAMES, NUM_FEATURES
from src.rank import (
    compute_atd,
    compute_hea,
    generate_reasoning,
    ATD_TAXONOMY,
    CORE_IR_SKILLS,
    PRODUCT_INDUSTRIES,
)
import rank_v2
from rank_v2 import (
    apply_hard_filters,
    normalize_scores,
    load_lightgbm_model,
    ensemble_model_scores,
    IDX_SINGULARITY, IDX_CORE_IR, IDX_DEPTH,
)

# -- Presets --------------------------------------------------------------
PRESETS = {
    "A_Default":       {"W_XGB": 0.40, "W_CE": 0.20, "W_HEURISTIC": 0.40},
    "B_ML_Heavy":      {"W_XGB": 0.60, "W_CE": 0.25, "W_HEURISTIC": 0.15},
    "C_Heuristic_Heavy": {"W_XGB": 0.20, "W_CE": 0.10, "W_HEURISTIC": 0.70},
    "D_Semantic":      {"W_XGB": 0.35, "W_CE": 0.50, "W_HEURISTIC": 0.15},
    "E_Balanced_ML":   {"W_XGB": 0.45, "W_CE": 0.30, "W_HEURISTIC": 0.25},
}

ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "precomputed_features.npz")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ranker.xgb")
LGB_PATH = os.path.join(ARTIFACTS_DIR, "ranker.lgb")
HONEYPOTS_PATH = os.path.join(ARTIFACTS_DIR, "honeypots.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates_backup.jsonl.gz")

# JD query (must match rank_v2.py exactly)
JD_QUERY = (
    "Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning, "
    "sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS, "
    "evaluation NDCG MRR MAP A/B testing, Python, production systems, "
    "startup product company, Pune Noida India"
)


def load_candidates_map(candidates_path: str, cid_set: set) -> dict:
    """Load candidate data for given candidate IDs (for reasoning)."""
    result = {}
    cpath = candidates_path
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id", "")
            if cid in cid_set:
                result[cid] = cand
    return result


def compute_quality_metrics(candidate_data: dict, feature_matrix: np.ndarray,
                            indices: list, scores: list) -> dict:
    """Compute quality metrics for a set of top candidates."""
    atd_l3plus = 0
    has_ir = 0
    has_product = 0
    in_india = 0
    hidden_gems = 0
    total_hea = 0.0
    total_atd = 0.0
    companies = set()
    locations = set()
    
    for idx in indices:
        feats = feature_matrix[idx]
        atd = float(feats[41])  # exact_atd
        hea = float(feats[42])  # exact_hea
        
        total_atd += atd
        total_hea += hea
        
        if atd >= 0.7:  # Level 3+
            atd_l3plus += 1
        if feats[IDX_CORE_IR] > 0:  # Has IR skills
            has_ir += 1
        if feats[39] > 0.5:  # In India
            in_india += 1
        if hea >= 1.0 and atd < 1.0:
            hidden_gems += 1
        
        # Try to get candidate data for richer metrics
        cid = candidate_ids[idx]
        cand = candidate_data.get(cid, {})
        career = cand.get("career_history", [])
        prof = cand.get("profile", {})
        
        for job in career:
            c_name = job.get("company", "")
            if c_name:
                companies.add(c_name)
            industry = str(job.get("industry", "")).lower()
            if any(ind in industry for ind in PRODUCT_INDUSTRIES):
                has_product += 1
        
        loc = prof.get("location", "")
        if loc:
            locations.add(loc)
    
    n = len(indices)
    return {
        "count": n,
        "atd_l3plus": atd_l3plus,
        "atd_l3plus_pct": atd_l3plus / n * 100,
        "has_ir": has_ir,
        "has_ir_pct": has_ir / n * 100,
        "has_product": has_product,
        "in_india": in_india,
        "in_india_pct": in_india / n * 100,
        "hidden_gems": hidden_gems,
        "mean_atd": total_atd / n,
        "mean_hea": total_hea / n,
        "unique_companies": len(companies),
        "unique_locations": len(locations),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


# ========================================================================
# MAIN
# ========================================================================

t_start = time.time()
print("=" * 70)
print("  A/B EXPERIMENT -- Score Fusion Weight Presets")
print("=" * 70)

# -- Stage 0: Load artifacts --
print("\n[Stage 0] Loading artifacts...")
t0 = time.time()

data = np.load(FEATURES_PATH, allow_pickle=True)
feature_matrix = data["features"]
candidate_ids = list(data["candidate_ids"])
print(f"  Features: {feature_matrix.shape[0]} candidates x {feature_matrix.shape[1]}")

import xgboost as xgb
xgb_model = xgb.XGBRanker()
xgb_model.load_model(MODEL_PATH)
print(f"  XGBoost: loaded")
lgb_model = load_lightgbm_model(LGB_PATH)

honeypot_ids = set()
if os.path.exists(HONEYPOTS_PATH):
    with open(HONEYPOTS_PATH, "r") as fh:
        hp_data = json.load(fh)
    honeypot_ids = set(hp_data.keys())
    print(f"  Honeypots: {len(honeypot_ids)} IDs")

print(f"  Stage 0: {time.time() - t0:.2f}s")

# -- Stage 1: Hard filters --
print("\n[Stage 1] Hard filters...")
t0 = time.time()
viable_mask = apply_hard_filters(feature_matrix, candidate_ids, honeypot_ids)
viable_indices = np.where(viable_mask)[0]
viable_features = feature_matrix[viable_mask]
print(f"  Stage 1: {time.time() - t0:.2f}s > {viable_indices.shape[0]} viable")

# -- Stage 2: XGBoost scoring --
print("\n[Stage 2] XGBoost ensemble scoring...")
t0 = time.time()
top_k_xgb = 200
xgb_raw_scores = xgb_model.predict(viable_features)
lgb_raw_scores = lgb_model.predict(viable_features) if lgb_model is not None else None
ranker_scores = ensemble_model_scores(xgb_raw_scores, lgb_raw_scores)
top_k_positions = np.argsort(ranker_scores)[::-1][:top_k_xgb]
top_indices = viable_indices[top_k_positions]
top_xgb_scores = ranker_scores[top_k_positions]
print(f"  Stage 2: {time.time() - t0:.2f}s")

# -- Stage 3: FlashRank reranking --
print("\n[Stage 3] FlashRank cross-encoder reranking...")
t0 = time.time()
ce_scores = {}
try:
    from flashrank import Ranker, RerankRequest
    flash_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./flashrank_cache")
    
    rerank_k = min(len(top_indices), rank_v2.RERANK_DEPTH)
    passages = []
    candidate_texts = {}
    
    # Load top candidates' text
    rerank_cids = {candidate_ids[idx] for idx in top_indices[:rerank_k]}
    cpath = CANDIDATES_PATH
    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id", "")
            if cid in rerank_cids:
                profile = cand.get("profile", {})
                skills = cand.get("skills", [])
                career = cand.get("career_history", [])
                text_parts = [
                    profile.get("headline", ""), profile.get("summary", ""),
                    f"Current: {profile.get('current_title', '')} at {profile.get('current_company', '')}",
                    f"Experience: {profile.get('years_of_experience', 0)} years",
                    f"Skills: {', '.join(s.get('name', '') for s in skills[:15])}",
                ]
                if career:
                    text_parts.append(f"Recent work: {career[0].get('description', '')[:200]}")
                candidate_texts[cid] = " ".join(text_parts)[:512]
    
    for idx in top_indices[:rerank_k]:
        cid = candidate_ids[idx]
        text = candidate_texts.get(cid, "")
        if text:
            passages.append({"id": str(idx), "text": text, "meta": {"cid": cid}})
    
    if passages:
        req = RerankRequest(query=JD_QUERY, passages=passages)
        results = flash_ranker.rerank(req)
        for res in results:
            ce_scores[int(res["id"])] = res["score"]
    
    print(f"  Reranked {len(ce_scores)} candidates, FlashRank OK")
except Exception as e:
    print(f"  FlashRank skipped: {e}")

print(f"  Stage 3: {time.time() - t0:.2f}s")

# -- Stage 4: Fusion for each preset --
print("\n[Stage 4] Running fusion for each preset...")

all_results = {}  # preset_name -> list of result dicts
all_metrics = {}  # preset_name -> quality metrics

# Preload candidate data for all top indices
all_top_cids = {candidate_ids[idx] for idx in top_indices}
candidate_data = load_candidates_map(CANDIDATES_PATH, all_top_cids)
print(f"  Loaded {len(candidate_data)} candidate profiles for reasoning")

xgb_norm = normalize_scores(top_xgb_scores)

for preset_name, weights in PRESETS.items():
    print(f"\n  -- {preset_name} ({weights}) --")
    
    W_XGB = weights["W_XGB"]
    W_CE = weights["W_CE"]
    W_HEURISTIC = weights["W_HEURISTIC"]
    total_w = W_XGB + W_CE + W_HEURISTIC
    if abs(total_w - 1.0) > 0.001:
        W_XGB /= total_w
        W_CE /= total_w
        W_HEURISTIC /= total_w
        print(f"    Normalized weights: {W_XGB:.2f}, {W_CE:.2f}, {W_HEURISTIC:.2f}")
    
    fused = []
    for i, idx in enumerate(top_indices):
        xgb_s = xgb_norm[i]
        ce_s = ce_scores.get(idx, xgb_s * 0.8)
        feats = feature_matrix[idx]
        heuristic_s = feats[IDX_SINGULARITY]
        
        final_score = W_XGB * xgb_s + W_CE * ce_s + W_HEURISTIC * heuristic_s
        fused.append({
            "idx": int(idx),
            "cid": candidate_ids[idx],
            "score": float(final_score),
            "xgb": float(xgb_s),
            "ce": float(ce_s),
            "heuristic": float(heuristic_s),
        })
    
    fused.sort(key=lambda x: (-x["score"], x["cid"]))
    top_100 = fused[:100]
    
    # Generate reasoning and output rows
    output_rows = []
    indices_100 = [item["idx"] for item in top_100]
    scores_100 = [item["score"] for item in top_100]
    
    for rank_idx, item in enumerate(top_100, start=1):
        idx = item["idx"]
        cid = item["cid"]
        cand = candidate_data.get(cid, {})
        
        skills = cand.get("skills", [])
        career = cand.get("career_history", [])
        atd = float(feature_matrix[idx, 41])
        hea = float(feature_matrix[idx, 42])
        reasoning = generate_reasoning(cand, rank_idx, atd, hea)
        
        feats = feature_matrix[idx]
        pipeline_note = (
            f" [Score={item['score']:.3f}, "
            f"XGB={item['xgb']:.3f}, "
            f"CE={item['ce']:.3f}, "
            f"Heur={item['heuristic']:.3f}, "
            f"W=({W_XGB:.2f},{W_CE:.2f},{W_HEURISTIC:.2f})]"
        )
        reasoning += pipeline_note
        
        output_rows.append({
            "preset": preset_name,
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(item["score"], 6),
            "xgb_score": round(item["xgb"], 4),
            "ce_score": round(item["ce"], 4),
            "heuristic_score": round(item["heuristic"], 4),
            "reasoning": reasoning,
            "cid_idx": idx,
        })
    
    all_results[preset_name] = output_rows
    
    # Compute quality metrics
    metrics = compute_quality_metrics(candidate_data, feature_matrix, indices_100, scores_100)
    all_metrics[preset_name] = metrics
    
    print(f"    Top score: {scores_100[0]:.4f}  Bottom: {scores_100[-1]:.4f}")
    print(f"    ATD L3+: {metrics['atd_l3plus']}/{metrics['count']}  IR: {metrics['has_ir']}  India: {metrics['in_india']}")
    print(f"    Hidden gems: {metrics['hidden_gems']}  Companies: {metrics['unique_companies']}")

# -- Overlap analysis --
print(f"\n{'=' * 70}")
print("  OVERLAP ANALYSIS (Jaccard Similarity)")
print(f"{'=' * 70}")

preset_names = list(PRESETS.keys())
overlap_matrix = {}
for p1 in preset_names:
    cids1 = {r["candidate_id"] for r in all_results[p1]}
    for p2 in preset_names:
        if p1 <= p2:
            cids2 = {r["candidate_id"] for r in all_results[p2]}
            intersection = len(cids1 & cids2)
            union = len(cids1 | cids2)
            jaccard = intersection / union if union > 0 else 0
            overlap_matrix[(p1, p2)] = jaccard

# Print overlap table
print(f"\n  {'Preset':<20}", end="")
for p in preset_names:
    print(f"  {p:<18}", end="")
print()
for p1 in preset_names:
    print(f"  {p1:<20}", end="")
    for p2 in preset_names:
        j = overlap_matrix.get((p1, p2), overlap_matrix.get((p2, p1), 0))
        print(f"  {j:<18.3f}", end="")
    print()

# -- Output comparison as CSV --
print(f"\n{'=' * 70}")
print("  WRITING RESULTS")
print(f"{'=' * 70}")

# Combined top-100 CSV
all_rows = []
for preset_name in PRESETS:
    all_rows.extend(all_results[preset_name])

combined_path = os.path.join(OUT_DIR, "ab_comparison.csv")
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "preset", "rank", "candidate_id", "score",
        "xgb_score", "ce_score", "heuristic_score", "reasoning"
    ])
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row[k] for k in writer.fieldnames})
print(f"  Combined top-100: {combined_path}")

# Summary CSV
summary_path = os.path.join(OUT_DIR, "ab_summary.csv")
summary_fields = [
    "preset", "weights",
    "mean_score", "median_score", "min_score", "max_score",
    "atd_l3plus_pct", "has_ir_pct", "in_india_pct",
    "hidden_gems", "mean_atd", "mean_hea",
    "unique_companies", "unique_locations",
]
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for preset_name in preset_names:
        m = all_metrics[preset_name]
        w = PRESETS[preset_name]
        writer.writerow({
            "preset": preset_name,
            "weights": f"XGB={w['W_XGB']}, CE={w['W_CE']}, Heur={w['W_HEURISTIC']}",
            "mean_score": f"{m['mean_score']:.4f}",
            "median_score": f"{m['median_score']:.4f}",
            "min_score": f"{m['min_score']:.4f}",
            "max_score": f"{m['max_score']:.4f}",
            "atd_l3plus_pct": f"{m['atd_l3plus_pct']:.1f}",
            "has_ir_pct": f"{m['has_ir_pct']:.1f}",
            "in_india_pct": f"{m['in_india_pct']:.1f}",
            "hidden_gems": m["hidden_gems"],
            "mean_atd": f"{m['mean_atd']:.3f}",
            "mean_hea": f"{m['mean_hea']:.3f}",
            "unique_companies": m["unique_companies"],
            "unique_locations": m["unique_locations"],
        })
print(f"  Summary: {summary_path}")

# -- Summary table --
print(f"\n{'=' * 100}")
print(f"  {'Preset':<20} {'Weights':<36} {'MeanSc':>6} {'MedSc':>6} {'L3+%':>6} {'IR%':>6} {'IN%':>6} {'Gem':>4} {'Co':>3} {'ATD':>5} {'HEA':>5}")
print(f"  {'-'*100}")
for preset_name in preset_names:
    m = all_metrics[preset_name]
    w = PRESETS[preset_name]
    w_str = "X=%.2f C=%.2f H=%.2f" % (w["W_XGB"], w["W_CE"], w["W_HEURISTIC"])
    print(("  %-20s %-36s %6.3f %6.3f %5.1f%% %4.1f%% %4.1f%% %4d %3d %5.3f %5.3f") % (
        preset_name, w_str,
        m["mean_score"], m["median_score"],
        m["atd_l3plus_pct"], m["has_ir_pct"], m["in_india_pct"],
        m["hidden_gems"], m["unique_companies"],
        m["mean_atd"], m["mean_hea"]))

# -- Highlight best per metric --
print(f"\n{'=' * 70}")
print("  BEST PER METRIC")
print(f"{'=' * 70}")

metric_names = {
    "mean_score": "Highest Mean Score",
    "atd_l3plus_pct": "Highest % ATD L3+",
    "has_ir_pct": "Highest % with IR Skills",
    "in_india_pct": "Highest % in India",
    "hidden_gems": "Most Hidden Gems",
    "unique_companies": "Most Company Diversity",
}
for metric_key, label in metric_names.items():
    best_preset = max(preset_names, key=lambda p: all_metrics[p][metric_key])
    best_val = all_metrics[best_preset][metric_key]
    print(f"  {label:<30} > {best_preset:<20} ({best_val})")

# -- Final recommendation --
print(f"\n{'=' * 70}")
print("  RECOMMENDATION")
print(f"{'=' * 70}")

# Compare A (default) vs B (ML heavy) on key quality metrics
a = all_metrics["A_Default"]
b = all_metrics["B_ML_Heavy"]
default_slug = "A_Default"

# Determine if B is better on most quality proxies
improvements = []
for metric_key in ["atd_l3plus_pct", "has_ir_pct", "in_india_pct", "mean_atd"]:
    if b[metric_key] > a[metric_key]:
        improvements.append((metric_key, b[metric_key] - a[metric_key]))

if len(improvements) >= 3:
    print(f"  => Preset B (ML Heavy) shows consistent improvement over default on {len(improvements)}/4 quality proxies.")
    print(f"     Recommend updating defaults to: W_XGB=0.60, W_CE=0.25, W_HEURISTIC=0.15")
    default_slug = "B_ML_Heavy"
elif len(improvements) >= 1:
    print(f"  => Preset B (ML Heavy) improves on {len(improvements)}/4 quality proxies but regresses on others.")
    print(f"     Recommend keeping A_Default as baseline and investigating per-candidate overlap.")
else:
    print(f"  => Default preset (A_Default) is competitive. No clear winner from quality metrics.")
    print(f"     Consider running with B_ML_Heavy for specific searches requiring higher technical depth.")

# Best combined metric: a balanced score
print(f"\n  Best by weighted score (ATD*IR*India):")
best_overall = max(preset_names, key=lambda p: (
    all_metrics[p]["atd_l3plus_pct"] * 0.3 +
    all_metrics[p]["has_ir_pct"] * 0.4 +
    all_metrics[p]["in_india_pct"] * 0.3
))
print(f"    -> {best_overall} ({PRESETS[best_overall]})")
print(f"      Composite: {all_metrics[best_overall]['atd_l3plus_pct']*0.3 + all_metrics[best_overall]['has_ir_pct']*0.4 + all_metrics[best_overall]['in_india_pct']*0.3:.1f}")

print(f"\n  Total runtime: {time.time() - t_start:.1f}s")
print(f"  Results saved to {OUT_DIR}/")
