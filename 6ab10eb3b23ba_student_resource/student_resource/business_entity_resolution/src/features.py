"""Pairwise feature engineering for (S1, candidate) pairs.

Consumes the candidate dict produced by blocking.py -- {s1_id: [cand_ids]} --
plus the cached normalized Parquet views (build_normalized.py) and returns one
row per (s1_id, cand_id) pair with the similarity/rank/context features the
LightGBM model in train.py is trained on.

Feature groups (Section 7.2 of the plan):
  - name similarity  (name_clean/name_core/tokens_sorted views)
  - address similarity (addr_clean/addr_no_landmark views)
  - postal / numeric tri-state agreement
  - rank / reverse-rank / gap context features -- computed from the union of
    ALL candidates for ALL S1s in this split, since reverse-rank needs to
    know every S1 competing for a given pool record.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from blocking import load_normalized  # noqa: E402
from io_utils import parse_ids, read_tsv  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rapidfuzz import fuzz, distance  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "dataset")

_FEATURE_COLS = [
    "entity_id", "country", "name_clean", "name_core", "legal_form",
    "name_tokens_sorted", "acronym", "addr_clean", "addr_no_landmark",
    "postal", "numbers",
]


def _char_trigram_set(text: str) -> set:
    text = text.replace(" ", "_")
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tri_state_equal(a: str, b: str) -> float:
    """1.0 = equal & non-empty, 0.0 = both non-empty & different, -1.0 = either missing."""
    if not a or not b:
        return -1.0
    return 1.0 if a == b else 0.0


def load_pool(split: str) -> pd.DataFrame:
    """S2 + S3 records for a split, indexed by entity_id, feature columns only."""
    s2 = load_normalized(split, "source2", columns=_FEATURE_COLS)
    s3 = load_normalized(split, "source3", columns=_FEATURE_COLS)
    pool = pd.concat([s2, s3], ignore_index=True)
    return pool.set_index("entity_id", drop=False)


def _rank_features(candidates: dict, name_sim: dict) -> pd.DataFrame:
    """Rank / reverse-rank / gap features for every (s1, cand) pair.

    name_sim: {(s1_id, cand_id): score} used as the ranking criterion.
    Rank: position of `cand` among `s1`'s own candidates (0 = best).
    Reverse rank: position of `s1` among every S1 that also lists `cand`.
    Gap: this pair's score minus the next-best score in the same ranking.
    """
    by_s1 = defaultdict(list)
    by_cand = defaultdict(list)
    for (s1, cand), score in name_sim.items():
        by_s1[s1].append((cand, score))
        by_cand[cand].append((s1, score))

    rank, rank_gap, n_above_half = {}, {}, {}
    for s1, items in by_s1.items():
        items.sort(key=lambda x: -x[1])
        scores = [s for _, s in items]
        for i, (cand, score) in enumerate(items):
            rank[(s1, cand)] = i
            rank_gap[(s1, cand)] = score - scores[i + 1] if i + 1 < len(scores) else score
            n_above_half[(s1, cand)] = sum(1 for s in scores if s >= 0.5)

    rev_rank, rev_gap = {}, {}
    for cand, items in by_cand.items():
        items.sort(key=lambda x: -x[1])
        scores = [s for _, s in items]
        for i, (s1, score) in enumerate(items):
            rev_rank[(s1, cand)] = i
            rev_gap[(s1, cand)] = score - scores[i + 1] if i + 1 < len(scores) else score

    rows = []
    for s1, cands in candidates.items():
        for cand in cands:
            key = (s1, cand)
            rows.append({
                "s1_id": s1, "cand_id": cand,
                "rank": rank.get(key, -1),
                "rank_gap": rank_gap.get(key, 0.0),
                "n_candidates_above_half": n_above_half.get(key, 0),
                "reverse_rank": rev_rank.get(key, -1),
                "reverse_rank_gap": rev_gap.get(key, 0.0),
                "n_own_candidates": len(cands),
            })
    return pd.DataFrame(rows)


def build_pair_features(split: str, candidates: dict) -> pd.DataFrame:
    """One row per (s1_id, cand_id) pair with every similarity/context feature.

    split: "train" or "test" -- selects which cached normalized Parquet to
    join against. candidates: output of blocking.run_split(split).
    """
    s1_df = load_normalized(split, "source1", columns=_FEATURE_COLS).set_index("entity_id", drop=False)
    pool_df = load_pool(split)

    name_sim = {}
    rows = []
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_df.index:
            continue
        a = s1_df.loc[s1_id]
        for cand_id in cand_ids:
            if cand_id not in pool_df.index:
                continue
            b = pool_df.loc[cand_id]

            name_ratio = fuzz.ratio(a.name_clean, b.name_clean) / 100.0
            token_set_ratio = fuzz.token_set_ratio(a.name_clean, b.name_clean) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(a.name_clean, b.name_clean) / 100.0
            jw_name = 1 - distance.JaroWinkler.normalized_distance(a.name_core, b.name_core)
            name_sim[(s1_id, cand_id)] = (name_ratio + token_set_ratio + jw_name) / 3.0

            a_nums = set(a.numbers.split(",")) if a.numbers else set()
            b_nums = set(b.numbers.split(",")) if b.numbers else set()

            rows.append({
                "s1_id": s1_id, "cand_id": cand_id, "is_s3": cand_id.startswith("S3-"),
                "same_country": a.country == b.country,
                # name similarity
                "name_ratio": name_ratio,
                "name_token_set_ratio": token_set_ratio,
                "name_token_sort_ratio": token_sort_ratio,
                "name_jaro_winkler": jw_name,
                "name_core_ratio": fuzz.ratio(a.name_core, b.name_core) / 100.0,
                "name_tokens_sorted_ratio": fuzz.ratio(a.name_tokens_sorted, b.name_tokens_sorted) / 100.0,
                "name_char_trigram_jaccard": _jaccard(_char_trigram_set(a.name_core), _char_trigram_set(b.name_core)),
                "name_token_jaccard": _jaccard(set(a.name_core.split()), set(b.name_core.split())),
                "acronym_match": _tri_state_equal(a.acronym, b.acronym),
                "legal_form_match": _tri_state_equal(a.legal_form, b.legal_form),
                "name_len_ratio": (min(len(a.name_core), len(b.name_core)) + 1)
                                  / (max(len(a.name_core), len(b.name_core)) + 1),
                # address similarity
                "addr_ratio": fuzz.ratio(a.addr_clean, b.addr_clean) / 100.0,
                "addr_no_landmark_ratio": fuzz.ratio(a.addr_no_landmark, b.addr_no_landmark) / 100.0,
                "addr_token_set_ratio": fuzz.token_set_ratio(a.addr_clean, b.addr_clean) / 100.0,
                "addr_char_trigram_jaccard": _jaccard(_char_trigram_set(a.addr_no_landmark), _char_trigram_set(b.addr_no_landmark)),
                "postal_match": _tri_state_equal(a.postal, b.postal),
                "postal_prefix_match": _tri_state_equal(a.postal[:3], b.postal[:3]),
                "numbers_jaccard": _jaccard(a_nums, b_nums),
                "numbers_match": _tri_state_equal(",".join(sorted(a_nums)), ",".join(sorted(b_nums))),
            })

    feats = pd.DataFrame(rows)
    if feats.empty:
        return feats
    ranks = _rank_features(candidates, name_sim)
    feats = feats.merge(ranks, on=["s1_id", "cand_id"], how="left")
    return feats


def attach_labels(feats: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
    """Add a 0/1 `label` column: 1 if (s1_id, cand_id) is a true match pair."""
    gt = read_tsv(ground_truth_path)
    true_pairs = set()
    for row in gt.itertuples():
        for cand in parse_ids(row.matched_entity_ids):
            true_pairs.add((row.source1_entity_id, cand))
    feats = feats.copy()
    feats["label"] = [
        1 if (s1, c) in true_pairs else 0
        for s1, c in zip(feats["s1_id"], feats["cand_id"])
    ]
    return feats


FEATURE_COLUMNS = [
    "same_country", "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "name_jaro_winkler", "name_core_ratio", "name_tokens_sorted_ratio",
    "name_char_trigram_jaccard", "name_token_jaccard", "acronym_match",
    "legal_form_match", "name_len_ratio", "addr_ratio", "addr_no_landmark_ratio",
    "addr_token_set_ratio", "addr_char_trigram_jaccard", "postal_match",
    "postal_prefix_match", "numbers_jaccard", "numbers_match", "is_s3",
    "rank", "rank_gap", "n_candidates_above_half", "reverse_rank",
    "reverse_rank_gap", "n_own_candidates",
]


if __name__ == "__main__":
    from blocking import run_split

    print("Blocking train split...")
    cands = run_split("train")
    print("Building pair features...")
    feats = build_pair_features("train", cands)
    feats = attach_labels(feats, os.path.join(DATA, "train", "train_ground_truth.tsv"))
    print(feats.shape, "positive rate:", feats["label"].mean())

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_features.parquet")
    feats.to_parquet(out_path, index=False)
    print("Wrote", out_path)
