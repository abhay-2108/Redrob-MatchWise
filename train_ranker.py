#!/usr/bin/env python3
"""
train_ranker.py — Train XGBoost LambdaMART Ranking Model
=========================================================
Uses the precomputed feature matrix from build_features.py and
generates proxy relevance grades using the existing ATD/HEA logic.

Trains an XGBRanker with objective='rank:ndcg' — this directly
optimizes for the exact metric the hackathon scores on.

Usage:
    python train_ranker.py [--features ./precomputed_features.npz]
"""

import argparse
import gzip
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import xgboost as xgb
    print(f"  XGBoost version: {xgb.__version__}")
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

try:
    import lightgbm as lgb
    print(f"  LightGBM version: {lgb.__version__}")
    HAS_LIGHTGBM = True
except ImportError:
    print("WARNING: lightgbm not installed. LightGBM ensemble model will be skipped.")
    HAS_LIGHTGBM = False

# Import the existing scoring functions to generate proxy labels
from rank import compute_atd, compute_hea, REF_DATE, SERVICE_COMPANIES
from build_features import FEATURE_NAMES, NUM_FEATURES


DEFAULT_FEEDBACK_PATH = "./feedback_logs.jsonl"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1] with a stable constant fallback."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))
    if max_s - min_s < 1e-9:
        return np.ones_like(scores, dtype=np.float32) * 0.5
    return (scores - min_s) / (max_s - min_s)


def load_feedback_overrides(feedback_path: str) -> dict:
    """Return the latest thumbs-up/down feedback per candidate ID."""
    if not feedback_path or not os.path.exists(feedback_path):
        return {}

    latest = {}
    with open(feedback_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"    WARNING: Skipping malformed feedback line {line_no}")
                continue

            cid = row.get("candidate_id")
            feedback = row.get("feedback")
            if cid is None or feedback not in (-1, 1):
                continue

            timestamp = row.get("timestamp", 0)
            prev = latest.get(cid)
            if prev is None or timestamp >= prev[0]:
                latest[cid] = (timestamp, int(feedback))

    return {cid: feedback for cid, (_, feedback) in latest.items()}


def generate_proxy_labels(
    candidates_path: str,
    semantic_scores: np.ndarray | None = None,
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
) -> tuple:
    """
    Use FlashRank (Cross-Encoder) on a heuristic-filtered sample of candidates
    to generate high-quality semantic proxy labels for training LambdaMART.
    
    Returns:
        labels: np.array of shape (N,) with relevance grades 0-4
        candidate_ids: list of candidate IDs in order
        raw_scores: heuristic scores
    """
    print("  Generating proxy labels using FlashRank (Semantic Distillation)...")
    
    candidate_ids = []
    raw_scores = []
    candidate_texts = {}
    yoe_list = []
    
    cpath = candidates_path
    if not os.path.exists(cpath):
        alt = cpath + ".gz"
        if os.path.exists(alt):
            cpath = alt
            
    opener = gzip.open if cpath.endswith(".gz") else open
    with opener(cpath, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            
            cid = cand.get("candidate_id", "")
            candidate_ids.append(cid)
            
            skills = cand.get("skills", [])
            career = cand.get("career_history", [])
            profile = cand.get("profile", {})
            yoe = profile.get("years_of_experience", 0)
            yoe_list.append(yoe)
            
            atd = compute_atd(skills, career)
            hea = compute_hea(cand)
            raw_scores.append((atd ** 1.5) * hea)
            
            # Store text for semantic scoring
            text_parts = [profile.get("headline", ""), profile.get("summary", "")]
            text_parts.append(f"Skills: {', '.join(s.get('name', '') for s in skills[:15])}")
            if career:
                text_parts.append(f"Recent work: {career[0].get('description', '')[:200]}")
            candidate_texts[cid] = " ".join(filter(None, text_parts))[:512]
            
            if (i + 1) % 20000 == 0:
                print(f"    Loaded {i + 1} candidates...")
                
    raw_scores = np.array(raw_scores, dtype=np.float32)
    if semantic_scores is not None:
        semantic_scores = np.asarray(semantic_scores, dtype=np.float32)
        if len(semantic_scores) != len(raw_scores):
            raise ValueError(
                f"semantic_scores length {len(semantic_scores)} != candidates {len(raw_scores)}"
            )
    
    # 2. Select a stratified union for FlashRank labels:
    #    high heuristic candidates plus high semantic-similarity candidates.
    top_k = min(5000, len(raw_scores))
    heuristic_k = min(3000, len(raw_scores))
    semantic_k = min(3000, len(raw_scores))
    heuristic_indices = np.argsort(raw_scores)[::-1][:heuristic_k]

    if semantic_scores is not None and np.any(semantic_scores):
        semantic_indices = np.argsort(semantic_scores)[::-1][:semantic_k]
        union_indices = np.unique(np.concatenate([heuristic_indices, semantic_indices]))
        selection_score = np.maximum(
            normalize_scores(raw_scores[union_indices]),
            normalize_scores(semantic_scores[union_indices]),
        )
        top_indices = union_indices[np.argsort(selection_score)[::-1][:top_k]]
        print(
            f"    Stratified label pool: {len(heuristic_indices)} heuristic + "
            f"{len(semantic_indices)} semantic -> {len(union_indices)} unique -> "
            f"{len(top_indices)} labeled"
        )
    else:
        top_indices = np.argsort(raw_scores)[::-1][:top_k]
        print("    Semantic scores unavailable; using heuristic-only label pool.")
    
    print(f"    Scoring top {len(top_indices)} candidates with FlashRank (batched)...")
    try:
        from flashrank import Ranker, RerankRequest
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./flashrank_cache")
        JD_QUERY = (
            "Senior AI Engineer founding team, embeddings retrieval ranking LLM fine-tuning, "
            "sentence-transformers BGE E5 vector database Pinecone Weaviate FAISS, "
            "evaluation NDCG MRR MAP A/B testing, Python, production systems, "
            "startup product company, Pune Noida India"
        )
        ce_scores = {}
        batch_size = 100
        # Process in smaller batches to avoid OOM
        for batch_start in range(0, len(top_indices), batch_size):
            batch = top_indices[batch_start:batch_start + batch_size]
            passages = []
            for idx in batch:
                cid = candidate_ids[idx]
                passages.append({"id": str(idx), "text": candidate_texts[cid]})
            rerank_request = RerankRequest(query=JD_QUERY, passages=passages)
            results = ranker.rerank(rerank_request)
            for r in results:
                ce_scores[int(r["id"])] = r["score"]
            if (batch_start // batch_size + 1) % 5 == 0:
                print(f"      FlashRank batch {batch_start // batch_size + 1}/{ (len(top_indices) + batch_size - 1) // batch_size } complete ({len(ce_scores)}/{len(top_indices)})")
    except Exception as e:
        print(f"    WARNING: FlashRank failed ({e}). Falling back to heuristics.")
        ce_scores = {idx: raw_scores[idx] for idx in top_indices}
        
    # 3. Bin CE scores into grades
    grades = np.zeros(len(raw_scores), dtype=np.int32)
    ce_score_values = list(ce_scores.values())
    
    if len(ce_score_values) > 0:
        p33 = np.percentile(ce_score_values, 33)
        p66 = np.percentile(ce_score_values, 66)
        p90 = np.percentile(ce_score_values, 90)
    else:
        p33, p66, p90 = 0.1, 0.3, 0.6
        
    for idx in top_indices:
        score = ce_scores.get(idx, raw_scores[idx])
        yoe = yoe_list[idx]
        if score <= p33:
            g = 1
        elif score <= p66:
            g = 2
        elif score <= p90:
            g = 3
        else:
            g = 4
            
        if yoe < 4.5:
            g = min(g, 3)
            
        grades[idx] = g

    feedback = load_feedback_overrides(feedback_path)
    applied_feedback = 0
    if feedback:
        for idx, cid in enumerate(candidate_ids):
            fb = feedback.get(cid)
            if fb == 1:
                grades[idx] = min(4, grades[idx] + 1)
                applied_feedback += 1
            elif fb == -1:
                grades[idx] = max(0, grades[idx] - 1)
                applied_feedback += 1
        print(f"  Applied {applied_feedback} latest feedback overrides from {feedback_path}")
        
    print(f"  Grade distribution: " + ", ".join(
        f"Grade {g}: {(grades == g).sum()}" for g in range(5)
    ))
    
    return grades, candidate_ids, raw_scores


def train_ranker(
    features_path: str,
    candidates_path: str,
    model_out: str,
    lgb_model_out: str | None = None,
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
):
    """Train the XGBoost LambdaMART ranking model."""
    
    t0 = time.time()
    
    print("=" * 60)
    print("  TRAIN RANKER — XGBoost LambdaMART (rank:ndcg)")
    print("=" * 60)
    
    # Load features
    print(f"\n  Loading features from {features_path}...")
    data = np.load(features_path, allow_pickle=True)
    feature_matrix = data["features"]
    feature_names = data["feature_names"]
    feature_cids = data["candidate_ids"]
    
    N, D = feature_matrix.shape
    print(f"  Feature matrix: {N} candidates × {D} features")

    if D != NUM_FEATURES:
        raise ValueError(
            f"Feature matrix has {D} columns, but current extractor expects {NUM_FEATURES}. "
            "Run build_features.py before training."
        )
    if list(feature_names) != FEATURE_NAMES:
        print("  WARNING: Feature names differ from current FEATURE_NAMES order.")
    
    # Generate proxy labels
    semantic_scores = None
    if "semantic_jd_similarity" in feature_names:
        semantic_idx = list(feature_names).index("semantic_jd_similarity")
        semantic_scores = feature_matrix[:, semantic_idx]

    grades, label_cids, raw_scores = generate_proxy_labels(
        candidates_path,
        semantic_scores=semantic_scores,
        feedback_path=feedback_path,
    )
    
    # Verify alignment
    assert len(grades) == N, f"Label count {len(grades)} != feature count {N}"
    for i in range(min(10, N)):
        assert feature_cids[i] == label_cids[i], \
            f"ID mismatch at {i}: {feature_cids[i]} vs {label_cids[i]}"
    print("  OK: Feature-label alignment verified")
    
    # ── Split into train/validation ──────────────────────────────────
    # Stratified split: 80% train, 20% validation
    np.random.seed(42)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    split = int(N * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    X_train = feature_matrix[train_idx]
    y_train = grades[train_idx]
    X_val = feature_matrix[val_idx]
    y_val = grades[val_idx]
    
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    # ── For LambdaMART, massive query groups (80k) cause vanishing gradients.
    # We break the candidates into artificial query groups of size 100.
    # Since grades are absolute, ranking locally within groups of 100
    # successfully trains the model to rank globally without underflow.
    GROUP_SIZE = 100
    qid_train = (np.arange(len(train_idx)) // GROUP_SIZE).astype(np.int32)
    qid_val = (np.arange(len(val_idx)) // GROUP_SIZE).astype(np.int32)
    
    # ── Train the XGBRanker ──────────────────────────────────────────
    print(f"\n  Training XGBRanker with rank:ndcg objective...")
    
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.1,
        n_estimators=300,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # Fast histogram-based training
    )
    
    ranker.fit(
        X_train, y_train,
        qid=qid_train,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=10
    )

    lgb_ranker = None
    if HAS_LIGHTGBM:
        print(f"\n  Training LGBMRanker with lambdarank objective...")
        lgb_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            n_estimators=300,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        group_train = np.bincount(qid_train)
        group_val = np.bincount(qid_val)
        lgb_ranker.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=[10, 50],
        )
    else:
        print("\n  Skipping LGBMRanker training because lightgbm is unavailable.")
    
    # ── Save model ───────────────────────────────────────────────────
    ranker.save_model(model_out)
    model_size = os.path.getsize(model_out) / 1024
    print(f"\n  Model saved to {model_out} ({model_size:.0f} KB)")

    if lgb_ranker is not None:
        if lgb_model_out is None:
            lgb_model_out = model_out.replace(".xgb", ".lgb")
        lgb_ranker.booster_.save_model(lgb_model_out)
        lgb_size = os.path.getsize(lgb_model_out) / 1024
        print(f"  LightGBM model saved to {lgb_model_out} ({lgb_size:.0f} KB)")
    
    # ── Evaluate on validation set ───────────────────────────────────
    val_scores = ranker.predict(X_val)
    
    # Compute NDCG@10 manually
    ndcg10 = compute_ndcg(y_val, val_scores, k=10)
    ndcg50 = compute_ndcg(y_val, val_scores, k=50)
    map_score = compute_map(y_val, val_scores)
    
    print(f"\n  {'='*50}")
    print(f"  VALIDATION METRICS:")
    print(f"  NDCG@10:  {ndcg10:.4f}")
    print(f"  NDCG@50:  {ndcg50:.4f}")
    print(f"  MAP:      {map_score:.4f}")
    print(f"  {'='*50}")
    
    # ── Feature importances ──────────────────────────────────────────
    importances = ranker.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"\n  Top 15 Feature Importances:")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print(f"  {'-'*42}")
    for i in range(min(15, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]:<30} {importances[idx]:>10.4f}")
    
    # Save feature importances separately for the dashboard
    np.savez_compressed(
        model_out.replace(".xgb", "_importances.npz"),
        importances=importances,
        feature_names=feature_names,
    )

    if lgb_ranker is not None:
        np.savez_compressed(
            model_out.replace(".xgb", "_lgb_importances.npz"),
            importances=lgb_ranker.feature_importances_,
            feature_names=feature_names,
        )
    
    t_end = time.time()
    print(f"\n  Total training time: {t_end - t0:.1f}s")
    
    return ranker


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    EVALUATION METRICS                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at position k."""
    # Sort by predicted scores (descending)
    sorted_idx = np.argsort(y_pred)[::-1]
    top_k = sorted_idx[:k]
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(top_k):
        rel = y_true[idx]
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    
    # Ideal DCG (sort by true relevance)
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_sorted):
        idcg += (2 ** rel - 1) / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_map(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 3) -> float:
    """Compute Mean Average Precision (treating grade >= threshold as relevant)."""
    sorted_idx = np.argsort(y_pred)[::-1]
    
    relevant = 0
    sum_precision = 0.0
    
    for i, idx in enumerate(sorted_idx):
        if y_true[idx] >= threshold:
            relevant += 1
            sum_precision += relevant / (i + 1)
    
    total_relevant = (y_true >= threshold).sum()
    if total_relevant == 0:
        return 0.0
    return sum_precision / total_relevant


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           MAIN                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost LambdaMART ranker")
    parser.add_argument("--features", default="./precomputed_features.npz",
                        help="Path to precomputed feature matrix")
    parser.add_argument("--candidates", default="./docs/candidates.jsonl",
                        help="Path to candidates.jsonl (for label generation)")
    parser.add_argument("--model-out", default="./ranker.xgb",
                        help="Output path for the trained model")
    parser.add_argument("--lgb-model-out", default="./ranker.lgb",
                        help="Output path for the trained LightGBM model")
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK_PATH,
                        help="Path to feedback_logs.jsonl for active-learning overrides")
    args = parser.parse_args()
    
    if not os.path.exists(args.features):
        print(f"ERROR: Feature file not found at {args.features}")
        print(f"Run build_features.py first!")
        sys.exit(1)
    
    train_ranker(args.features, args.candidates, args.model_out, args.lgb_model_out, args.feedback)


if __name__ == "__main__":
    main()
