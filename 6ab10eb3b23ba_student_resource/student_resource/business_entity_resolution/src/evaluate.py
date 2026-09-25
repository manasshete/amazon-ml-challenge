"""Exact local implementation of the challenge's scoring metric.

Why this exists: the leaderboard only gives you 15 total submissions across
3 days. Every modeling decision (blocking K, feature set, threshold, model
choice) must be judged on this local scorer against a held-out slice of
train data, never on the leaderboard directly.
"""

from collections import defaultdict


def f05_entity(pred, true, beta: float = 0.5) -> float:
    """Score one Source-1 entity.

    pred / true: iterables of matched entity ids (order doesn't matter).

    Special cases mirror the challenge's rules exactly:
    - true empty, pred empty  -> 1.0 (correctly identified a singleton)
    - true empty, pred non-empty -> 0.0 (false merge on a singleton: harshly
      punished, since F0.5 favors precision)
    - true non-empty, pred empty -> 0.0 (missed every real match)
    - otherwise: standard F-beta from precision/recall, with beta=0.5
      weighting precision twice as heavily as recall.
    """
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred_map: dict, true_map: dict, beta: float = 0.5) -> float:
    """Macro-average F0.5 across every S1 entity in true_map.

    Every entity in true_map counts once, equally -- a singleton counts as
    much as a chain with ten matches. If an S1 id is missing from pred_map
    entirely, it's treated as an empty prediction (matches the "missing
    row = rejected/empty" behavior you should never actually trigger in a
    real submission, but is the safe default for local scoring).
    """
    if not true_map:
        return 0.0
    total = sum(f05_entity(pred_map.get(s1, []), matches, beta) for s1, matches in true_map.items())
    return total / len(true_map)


def score_breakdown(pred_map: dict, true_map: dict, meta: dict | None = None) -> dict:
    """Compute macro F0.5 plus the diagnostic slices the plan asks for.

    meta: optional {s1_id: {"country": ..., "n_true": ...}} to enable the
    per-country / per-match-count breakdowns. Without it you still get the
    headline score and the singleton-baseline comparison.

    Returns a dict with:
    - "macro_f05": the real, official metric.
    - "n_entities": how many S1 entities were scored.
    - "singleton_fraction": fraction of entities with an empty true set --
      this IS the score of the trivial "always predict empty" baseline.
      Any real model must beat this number by a meaningful margin.
    - "by_country" / "by_n_true": macro F0.5 restricted to each slice, only
      present when meta is supplied.
    """
    n = len(true_map)
    singleton_fraction = sum(1 for v in true_map.values() if not v) / n if n else 0.0

    result = {
        "macro_f05": macro_f05(pred_map, true_map),
        "n_entities": n,
        "singleton_fraction": singleton_fraction,
    }

    if meta:
        by_country = defaultdict(list)
        by_n_true = defaultdict(list)
        for s1, true in true_map.items():
            score = f05_entity(pred_map.get(s1, []), true)
            info = meta.get(s1, {})
            if "country" in info:
                by_country[info["country"]].append(score)
            by_n_true[len(true)].append(score)

        result["by_country"] = {k: sum(v) / len(v) for k, v in by_country.items()}
        result["by_n_true"] = {k: sum(v) / len(v) for k, v in sorted(by_n_true.items())}

    return result


if __name__ == "__main__":
    # Sanity check from the problem statement: expect 0.714.
    example_score = f05_entity(
        ["S2-00047", "S2-00193", "S3-00812"],
        ["S2-00047", "S3-00812"],
    )
    assert abs(example_score - 0.714) < 1e-3, example_score
    print(f"Scorer self-check passed: {example_score:.3f} (expected 0.714)")
