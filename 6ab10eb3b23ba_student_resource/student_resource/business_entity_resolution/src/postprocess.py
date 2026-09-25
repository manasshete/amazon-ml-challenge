"""Turn calibrated pair probabilities into a final {s1_id: [matched ids]} map.

Two selectable strategies (Section 8 of the plan):
  - two-threshold rule (train.two_threshold_predictions) -- simple baseline.
  - expected-F0.5 maximization (best_prefix below) -- usually stronger,
    since it directly optimizes the competition metric per S1 entity
    instead of a single global threshold pair.

Both assume `feats` has been restricted to one-to-one assignment first, if
EDA confirmed each S2/S3 record belongs to at most one S1 (see eda notes).
"""

import numpy as np
import pandas as pd


def one_to_one(feats: pd.DataFrame) -> pd.DataFrame:
    """Keep each candidate only for the S1 with the highest probability for it."""
    best = feats.loc[feats.groupby("cand_id")["prob"].idxmax(), ["cand_id", "s1_id"]]
    return feats.merge(best, on=["cand_id", "s1_id"], how="inner")


def best_prefix(probs: np.ndarray, p_miss: float = 0.0, n_sim: int = 4000,
                 rng: np.random.Generator = None) -> int:
    """Expected-F0.5-maximizing number of top candidates to output.

    probs: calibrated probabilities, sorted descending, for one S1's
    candidates (already capped to the top ~10 for speed).
    p_miss: probability a true match exists outside the candidate set at all
    (estimated from blocking recall on train); adds one always-missed "ghost"
    true match with this probability, so predicting non-empty can never reach
    a perfect score once misses are possible.
    Returns k (0 = predict empty).
    """
    rng = rng or np.random.default_rng(0)
    m = len(probs)
    if m == 0:
        return 0
    T = rng.random((n_sim, m)) < probs
    n_true = T.sum(1) + (rng.random(n_sim) < p_miss)
    best_k, best_val = 0, float((n_true == 0).mean())
    tp_cum = np.cumsum(T, axis=1)
    for k in range(1, m + 1):
        tp = tp_cum[:, k - 1]
        p = tp / k
        r = np.divide(tp, n_true, out=np.zeros_like(p, dtype=float), where=n_true > 0)
        den = 0.25 * p + r
        f = np.divide(1.25 * p * r, den, out=np.zeros_like(p, dtype=float), where=den > 0)
        val = float(f.mean())
        if val > best_val:
            best_k, best_val = k, val
    return best_k


def expected_f05_predictions(feats: pd.DataFrame, p_miss: float = 0.0, top_n: int = 10) -> dict:
    """Apply best_prefix per S1 entity. feats must have s1_id, cand_id, prob."""
    rng = np.random.default_rng(0)
    preds = {}
    for s1_id, group in feats.groupby("s1_id"):
        group = group.sort_values("prob", ascending=False).head(top_n)
        probs = group["prob"].to_numpy()
        k = best_prefix(probs, p_miss=p_miss, rng=rng)
        preds[s1_id] = group["cand_id"].tolist()[:k]
    return preds
