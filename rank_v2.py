
#!/usr/bin/env python3
"""
rank_v2.py — Multi-Stage Ranking Pipeline (Submission Runner)
==============================================================
The runtime ranking script that produces the final submission CSV.

Architecture (runs in < 60 seconds on CPU):
  Stage 0: Load precomputed features + XGBoost model
  Stage 1: Hard-filter honeypots, service-only, zero-relevance
  Stage 2: XGBoost LambdaMART scoring (rank:ndcg optimized)
  Stage 3: FlashRank cross-encoder reranking on top 50
  Stage 4: Score fusion + reasoning generation + CSV output

Prerequisites (offline, no time limit):
  1. python build_features.py --candidates <path/to/candidates.jsonl>
  2. python train_ranker.py

Usage:
  python rank_v2.py --candidates <path/to/candidates.jsonl> --out ./submission.csv
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

import numpy as np

# Import existing taxonomy + helpers from rank.py (kept as library)
from src.rank import (
    compute_atd,
    compute_hea,
    generate_reasoning,
)

from build_features import FEATURE_NAMES, NUM_FEATURES


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     CONFIGURATION                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURES = os.path.join(DIR_PATH, "artifacts", "precomputed_features.npz")
DEFAULT_MODEL = os.path.join(DIR_PATH, "artifacts", "ranker.xgb")
DEFAULT_LGBM = os.path.join(DIR_PATH, "artifacts", "ranker.lgb")
DEFAULT_HONEYPOTS = os.path.join(DIR_PATH, "artifacts", "honeypots.json")

# Feature indices for hard filters (must match FEATURE_NAMES order)
IDX_MAX_ATD = FEATURE_NAMES.index("max_atd_level")
IDX_TIMELINE = FEATURE_NAMES.index("timeline_impossible")
IDX_INFLATION = FEATURE_NAMES.index("skill_inflation")
IDX_SERVICE = FEATURE_NAMES.index("service_company_only")
IDX_GHOST = FEATURE_NAMES.index("ghost_candidate")
IDX_SINGULARITY = FEATURE_NAMES.index("singularity_score")
IDX_CORE_IR = FEATURE_NAMES.index("core_ir_match_ratio")
IDX_DEPTH = FEATURE_NAMES.index("skill_depth_score")

# Score fusion weights (tuned offline)
W_XGB = 0.40    # XGBoost LambdaMART score
W_CE = 0.20     # Cross-encoder score (FlashRank) — kept low; TinyBERT is noisy
W_HEURISTIC = 0.40  # True singularity score (ATD^1.5)*HEA
W_STAGE2_XGB = 0.60
W_STAGE2_LGBM = 0.40


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     STAGE 0: LOAD ARTIFACTS                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def load_lightgbm_model(model_path: str):
    """Load optional LightGBM model; return None when unavailable."""
    if not model_path or not os.path.exists(model_path):
        print(f"    LightGBM model not found at {model_path}; using XGBoost only")
        return None
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        print(f"    LightGBM: {model_path} loaded")
        return model
    except Exception as e:
        print(f"    WARNING: LightGBM load failed ({e}); using XGBoost only")
        return None


def load_artifacts(features_path: str, model_path: str, honeypots_path: str, lgb_model_path: str = DEFAULT_LGBM):
    """Load all precomputed artifacts into memory."""
    print("  Stage 0: Loading artifacts...")
    t0 = time.time()
    
    # Feature matrix
    data = np.load(features_path, allow_pickle=True)
    feature_matrix = data["features"]
    candidate_ids = list(data["candidate_ids"])
    feature_names = list(data["feature_names"])
    
    print(f"    Features: {feature_matrix.shape[0]} candidates × {feature_matrix.shape[1]} features")
    if feature_matrix.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Feature matrix has {feature_matrix.shape[1]} columns, but current extractor expects "
            f"{NUM_FEATURES}. Run build_features.py and train_ranker.py again."
        )
    if feature_names != FEATURE_NAMES:
        print("    WARNING: Feature names differ from current FEATURE_NAMES order.")
    
    # XGBoost model
    import xgboost as xgb
    model = xgb.XGBRanker()
    model.load_model(model_path)
    print(f"    Model: {model_path} loaded")
    lgb_model = load_lightgbm_model(lgb_model_path)
    
    # Honeypots
    honeypot_ids = set()
    if os.path.exists(honeypots_path):
        with open(honeypots_path, "r", encoding="utf-8") as fh:
            hp_data = json.load(fh)
        honeypot_ids = set(hp_data.keys())
        print(f"    Honeypots: {len(honeypot_ids)} IDs loaded")
    
    t1 = time.time()
    print(f"    -> Stage 0 complete in {t1 - t0:.2f}s")
    
    return feature_matrix, candidate_ids, feature_names, model, lgb_model, honeypot_ids


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     STAGE 1: HARD FILTERS                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def apply_hard_filters(feature_matrix, candidate_ids, honeypot_ids):
    """Apply hard filters to eliminate non-viable candidates.
    
    Returns a boolean mask of viable candidates.
    """
    print("  Stage 1: Applying hard filters...")
    t0 = time.time()
    
    N = len(candidate_ids)
    viable = np.ones(N, dtype=bool)
    
    # Filter 1: Known honeypots
    honeypot_count = 0
    for i, cid in enumerate(candidate_ids):
        if cid in honeypot_ids:
            viable[i] = False
            honeypot_count += 1
    
    # Filter 2: Timeline impossibilities (detected by feature engineering)
    timeline_filter = feature_matrix[:, IDX_TIMELINE] > 0.5
    viable &= ~timeline_filter
    
    # Filter 3: Extreme skill inflation (>= 5 expert skills with 0 duration)
    inflation_filter = feature_matrix[:, IDX_INFLATION] >= 0.5  # 5+ inflated skills
    viable &= ~inflation_filter
    
    # Filter 4: Service-company-only career
    service_filter = feature_matrix[:, IDX_SERVICE] > 0.5
    viable &= ~service_filter
    
    # Filter 5: Ghost candidates (inactive + unresponsive)
    ghost_filter = feature_matrix[:, IDX_GHOST] > 0.5
    viable &= ~ghost_filter
    
    # Filter 6: Zero relevant skills (max_atd_level == 0)
    no_skills = feature_matrix[:, IDX_MAX_ATD] < 0.5  # level 0
    viable &= ~no_skills
    
    viable_count = viable.sum()
    t1 = time.time()
    
    print(f"    Honeypots removed: {honeypot_count}")
    print(f"    Timeline fraud: {timeline_filter.sum()}")
    print(f"    Skill inflation: {inflation_filter.sum()}")
    print(f"    Service-only: {service_filter.sum()}")
    print(f"    Ghost candidates: {ghost_filter.sum()}")
    print(f"    Zero skills: {no_skills.sum()}")
    print(f"    -> {viable_count}/{N} candidates remain ({t1 - t0:.2f}s)")
    
    return viable


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     STAGE 2: XGBOOST RANKING                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def ensemble_model_scores(xgb_scores: np.ndarray, lgb_scores: np.ndarray | None = None) -> np.ndarray:
    """Normalize and ensemble model scores for ranking selection/fusion."""
    xgb_norm = normalize_scores(np.asarray(xgb_scores, dtype=np.float32))
    if lgb_scores is None:
        return xgb_norm

    lgb_norm = normalize_scores(np.asarray(lgb_scores, dtype=np.float32))
    return (W_STAGE2_XGB * xgb_norm) + (W_STAGE2_LGBM * lgb_norm)


def xgboost_rank(feature_matrix, viable_mask, model, lgb_model=None, top_k=200):
    """Score viable candidates with XGBoost LambdaMART and return top-K.
    
    Returns:
        top_indices: indices into the original feature_matrix of the top-K
        xgb_scores: scores for those candidates
    """
    print("  Stage 2: XGBoost/LightGBM ensemble scoring...")
    t0 = time.time()
    
    viable_features = feature_matrix[viable_mask]
    viable_indices = np.where(viable_mask)[0]
    
    # Predict relevance scores
    xgb_scores = model.predict(viable_features)
    lgb_scores = lgb_model.predict(viable_features) if lgb_model is not None else None
    model_scores = ensemble_model_scores(xgb_scores, lgb_scores)
    
    # Get top-K by ensemble score
    top_k_positions = np.argsort(model_scores)[::-1][:top_k]
    
    top_indices = viable_indices[top_k_positions]
    top_scores = model_scores[top_k_positions]
    
    t1 = time.time()
    print(f"    Scored {viable_mask.sum()} candidates, selected top {top_k}")
    if lgb_model is None:
        print("    LightGBM unavailable; Stage 2 used normalized XGBoost scores only")
    print(f"    Score range: {top_scores[0]:.4f} -> {top_scores[-1]:.4f}")
    print(f"    -> Stage 2 complete in {t1 - t0:.2f}s")
    
    return top_indices, top_scores


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     STAGE 3: CROSS-ENCODER RERANKING                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def cross_encoder_rerank(candidates_path: str, candidate_ids: list,
                          top_indices: np.ndarray, top_k_rerank: int = 50):
    """Use FlashRank cross-encoder to rerank the top candidates.
    
    Loads the actual candidate text (headline + summary + skills) and
    scores them against the JD using a lightweight cross-encoder.
    
    Returns:
        ce_scores: dict mapping candidate index -> cross-encoder score
    """
    print(f"  Stage 3: FlashRank cross-encoder reranking (top {top_k_rerank})...")
    t0 = time.time()
    
    # Try to import FlashRank
    try:
        from flashrank import Ranker, RerankRequest
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./flashrank_cache")
        has_flashrank = True
        print("    FlashRank model loaded (ms-marco-TinyBERT-L-2-v2)")
    except ImportError:
        print("    WARNING: FlashRank not installed. Skipping cross-encoder stage.")
        print("    Install with: pip install flashrank")
        has_flashrank = False
    except Exception as e:
        print(f"    WARNING: FlashRank error: {e}. Skipping cross-encoder stage.")
        has_flashrank = False
    
    if not has_flashrank:
        return {}
    
    # We need the actual text of the top candidates
    # Load only the ones we need from the JSONL
    rerank_indices = set(top_indices[:top_k_rerank].tolist())
    rerank_cids = {candidate_ids[idx] for idx in rerank_indices}
    
    # Build candidate text map
    candidate_texts = {}
    cpath = candidates_path
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
    
    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id", "")
            if cid in rerank_cids:
                # Build a concise text representation
                profile = cand.get("profile", {})
                skills = cand.get("skills", [])
                career = cand.get("career_history", [])
                
                text_parts = [
                    profile.get("headline", ""),
                    profile.get("summary", ""),
                    f"Current: {profile.get('current_title', '')} at {profile.get('current_company', '')}",
                    f"Experience: {profile.get('years_of_experience', 0)} years",
                    f"Skills: {', '.join(s.get('name', '') for s in skills[:15])}",
                ]
                # Add most recent job description
                if career:
                    text_parts.append(f"Recent work: {career[0].get('description', '')[:200]}")
                
                candidate_texts[cid] = " ".join(text_parts)[:512]  # Truncate for speed
    
    # Build passages for FlashRank
    JD_QUERY = (
        "Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning, "
        "sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS, "
        "evaluation NDCG MRR MAP A/B testing, Python, production systems, "
        "startup product company, Pune Noida India"
    )
    
    passages = []
    passage_cids = []
    for idx in top_indices[:top_k_rerank]:
        cid = candidate_ids[idx]
        text = candidate_texts.get(cid, "")
        if text:
            passages.append({"id": str(idx), "text": text, "meta": {"cid": cid}})
            passage_cids.append(idx)
    
    if not passages:
        return {}
    
    # Rerank
    rerank_request = RerankRequest(query=JD_QUERY, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Map back to indices -> scores
    ce_scores = {}
    for result in results:
        idx = int(result["id"])
        ce_scores[idx] = result["score"]
    
    t1 = time.time()
    if ce_scores:
        scores_list = list(ce_scores.values())
        print(f"    Reranked {len(ce_scores)} candidates")
        print(f"    CE score range: {max(scores_list):.4f} -> {min(scores_list):.4f}")
    print(f"    -> Stage 3 complete in {t1 - t0:.2f}s")
    
    return ce_scores


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     STAGE 4: SCORE FUSION + OUTPUT                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-9:
        return np.ones_like(scores) * 0.5
    return (scores - min_s) / (max_s - min_s)


def fuse_and_output(candidates_path: str, candidate_ids: list,
                     feature_matrix: np.ndarray,
                     top_indices: np.ndarray, xgb_scores: np.ndarray,
                     ce_scores: dict, honeypot_ids: set,
                     out_path: str) -> list:
    """Fuse scores, generate reasoning, and write CSV."""
    print("  Stage 4: Score fusion + reasoning + CSV output...")
    t0 = time.time()
    
    # ── Normalize ranker ensemble scores ──
    xgb_norm = normalize_scores(xgb_scores)
    
    # ── Build fused scores ──
    fused = []
    for i, idx in enumerate(top_indices):
        xgb_s = xgb_norm[i]
        
        # Cross-encoder score (if available)
        if idx in ce_scores:
            ce_s = ce_scores[idx]
        else:
            ce_s = xgb_s * 0.8  # fallback: slightly discounted XGB score
        
        # Legacy heuristic score — true singularity score (ATD^1.5)*HEA
        # This is the proven formula from rank.py that produced top-tier candidates
        feats = feature_matrix[idx]
        heuristic_s = feats[IDX_SINGULARITY]
        
        # Weighted fusion
        final_score = (
            W_XGB * xgb_s +
            W_CE * ce_s +
            W_HEURISTIC * heuristic_s
        )
        
        cid = candidate_ids[idx]
        fused.append((cid, final_score, idx, xgb_s, ce_s, heuristic_s))
    
    # Sort by fused score descending, then candidate_id for tiebreaks
    fused.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 100
    top_100 = fused[:100]
    
    # ── Load candidate data for reasoning generation ──
    top_100_cids = {item[0] for item in top_100}
    candidate_data = {}
    
    cpath = candidates_path
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
    
    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id", "")
            if cid in top_100_cids:
                candidate_data[cid] = cand
    
    # ── Generate output rows ──
    rows = []
    for rank_idx, (cid, final_score, matrix_idx, xgb_s, ce_s, h_s) in enumerate(top_100, start=1):
        cand = candidate_data.get(cid, {})
        
        # Compute ATD/HEA for reasoning (these are fast, single-candidate calls)
        skills = cand.get("skills", [])
        career = cand.get("career_history", [])
        atd = compute_atd(skills, career)
        hea = compute_hea(cand)
        
        # Generate reasoning using the existing (good) reasoning generator
        # (Note: generate_reasoning in rank.py auto-tags Hidden Gems with HEA>=1.0 and ATD<1.0)
        reasoning = generate_reasoning(cand, rank_idx, atd, hea)
        
        # Append pipeline metadata to reasoning
        feats = feature_matrix[matrix_idx]
        pipeline_note = (
            f" [Pipeline: Ranker={xgb_s:.3f}, "
            f"IR_match={feats[IDX_CORE_IR]:.2f}, "
            f"depth={feats[IDX_DEPTH]:.2f}]"
        )
        reasoning += pipeline_note
        
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(final_score, 6),
            "reasoning": reasoning,
        })
    
    # ── Write CSV ──
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
    
    t1 = time.time()
    print(f"    Top 100 written to {out_path}")
    print(f"    Score range: {rows[0]['score']:.6f} -> {rows[-1]['score']:.6f}")
    print(f"    -> Stage 4 complete in {t1 - t0:.2f}s")
    
    return rows


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           MAIN PIPELINE                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates using the Multi-Stage Pipeline "
                    "(XGBoost LambdaMART + FlashRank Cross-Encoder)"
    )
    parser.add_argument(
        "--candidates", default="./candidates.jsonl",
        help="Path to candidates JSONL file (.jsonl or .jsonl.gz)",
    )
    parser.add_argument(
        "--out", default="./submission.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--features", default=DEFAULT_FEATURES,
        help="Path to precomputed_features.npz",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to ranker.xgb",
    )
    parser.add_argument(
        "--lgb-model", default=DEFAULT_LGBM,
        help="Path to optional ranker.lgb",
    )
    parser.add_argument(
        "--honeypots", default=DEFAULT_HONEYPOTS,
        help="Path to honeypots.json",
    )
    parser.add_argument(
        "--no-crossencoder", action="store_true",
        help="Skip the FlashRank cross-encoder stage (faster, slightly less precise)",
    )
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("=" * 70)
    print("  REDROB MATCHWISE — Multi-Stage Ranking Pipeline v2")
    print("  Stage 0: Load | Stage 1: Filter | Stage 2: XGBoost LambdaMART")
    print("  Stage 3: FlashRank Cross-Encoder | Stage 4: Fusion + Output")
    print("=" * 70)
    
    # Verify prerequisites
    if not os.path.exists(args.features):
        print(f"\nERROR: Feature file not found at {args.features}")
        print("Run this first:  python build_features.py --candidates <path/to/candidates.jsonl>")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"\nERROR: Model file not found at {args.model}")
        print("Run this first:  python train_ranker.py")
        sys.exit(1)
    
    # ── Stage 0: Load ──
    feature_matrix, candidate_ids, feature_names, model, lgb_model, honeypot_ids = \
        load_artifacts(args.features, args.model, args.honeypots, args.lgb_model)
    
    # ── Stage 1: Filter ──
    viable_mask = apply_hard_filters(feature_matrix, candidate_ids, honeypot_ids)
    
    # ── Stage 2: XGBoost Ranking ──
    top_indices, xgb_scores = xgboost_rank(feature_matrix, viable_mask, model, lgb_model, top_k=200)
    
    # ── Stage 3: Cross-Encoder Reranking ──
    if args.no_crossencoder:
        ce_scores = {}
        print("  Stage 3: SKIPPED (--no-crossencoder flag)")
    else:
        ce_scores = cross_encoder_rerank(
            args.candidates, candidate_ids, top_indices, top_k_rerank=50
        )
    
    # ── Stage 4: Fusion + Output ──
    rows = fuse_and_output(
        args.candidates, candidate_ids, feature_matrix,
        top_indices, xgb_scores, ce_scores, honeypot_ids, args.out
    )
    
    # ── Summary ──
    t_end = time.time()
    total_time = t_end - t_start
    
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Output: {args.out}")
    print(f"{'=' * 70}")
    
    # Top-10 snapshot
    print("\n  Top-10 Snapshot:")
    print(f"  {'Rank':>4}  {'Score':>8}  {'CID':<15}")
    print(f"  {'-'*30}")
    for row in rows[:10]:
        print(f"  {row['rank']:>4}  {row['score']:>8.4f}  {row['candidate_id']:<15}")
    
    # Sanity checks
    output_cids = {r["candidate_id"] for r in rows}
    hp_in_output = output_cids & honeypot_ids
    
    if hp_in_output:
        print(f"\n  WARNING: {len(hp_in_output)} honeypots in output!")
    else:
        print(f"\n  OK: Zero honeypots in top 100")
    
    if total_time > 300:
        print(f"  WARNING: Runtime {total_time:.0f}s exceeds 5-minute limit!")
    else:
        print(f"  OK: Runtime {total_time:.1f}s within 5-minute budget")
    
    if len(rows) != 100:
        print(f"  WARNING: Output has {len(rows)} rows, expected 100!")
    else:
        print(f"  OK: Exactly 100 rows in output")
    
    # Check score monotonicity
    scores = [r["score"] for r in rows]
    is_mono = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    if is_mono:
        print(f"  OK: Scores are monotonically non-increasing")
    else:
        print(f"  WARNING: Scores are NOT monotonically non-increasing!")


if __name__ == "__main__":
    main()
