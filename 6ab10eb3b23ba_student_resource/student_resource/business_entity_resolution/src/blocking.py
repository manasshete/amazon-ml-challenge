"""Candidate generation (blocking), v1: rare-token inverted index + postal
code exact match, both restricted to same-country pools (EDA confirmed
100% of true matches share country -- Section 6 of the plan).

Why not brute-force TF-IDF cosine similarity over the full corpus: that
requires an (n_s1 x n_pool) similarity matrix. Even chunked, e.g. US alone
has ~1.3M S1 records against a ~6.2M S2+S3 pool -- comparing one chunk of
1,000 S1 rows against the full pool as a dense array is already
1,000 x 6,200,000 x 8 bytes ~= 49 GB. That does not fit in memory.

Instead this module builds an inverted index: token -> list of pool row
indices, but ONLY for tokens that are "rare" (below a document-frequency
cutoff). A common word like "store" might appear in 40,000 records and is
useless for narrowing anything down; a distinctive word like "lakshmi" or
"orelee" might appear in 12 records and is a strong, cheap signal. This
retrieval is O(matching records) per S1, not O(pool size), so it scales.

A second exact-match key (postal code) is unioned in, since two records
sharing a full postal code is a strong, free signal regardless of name
similarity.

This is "blocking v1" -- the plan's Day-2 step of adding char n-gram TF-IDF
retrieval as a second retriever comes next, once recall@K here is measured.
"""

import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import parse_ids, read_tsv  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NORM_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "normalized")
DATA = os.path.join(ROOT, "dataset")

RARE_TOKEN_DF_CUTOFF = 50  # a name_core token is "rare" if <= this many pool rows contain it
K_FINAL = 30  # candidate cap per S1 entity


# Only the columns blocking actually needs — avoids loading the full parquet
# (name_norm, addr_norm, numbers, etc.) into RAM, which caused MemoryError
# on large sources like train_source2/3 (~650 MB each on disk, several GB
# once decompressed and held as Python objects).
_BLOCKING_COLS = ["entity_id", "country", "name_core", "postal"]


def load_normalized(split: str, source: str,
                    columns: list = None) -> pd.DataFrame:
    path = os.path.join(NORM_DIR, f"{split}_{source}.parquet")
    return pd.read_parquet(path, columns=columns or _BLOCKING_COLS)


def build_rare_token_index(pool: pd.DataFrame) -> dict:
    """token -> list of positional indices into `pool`, rare tokens only.

    Tokens are the whitespace-split words of name_core. A token's document
    frequency is how many pool rows contain it at least once; tokens above
    RARE_TOKEN_DF_CUTOFF are dropped from the index entirely (they would
    only add noise and cost, never precision).
    """
    df_counts = defaultdict(int)
    row_tokens = []
    for name_core in pool["name_core"]:
        toks = set(name_core.split())
        row_tokens.append(toks)
        for t in toks:
            df_counts[t] += 1

    index = defaultdict(list)
    for i, toks in enumerate(row_tokens):
        for t in toks:
            if df_counts[t] <= RARE_TOKEN_DF_CUTOFF:
                index[t].append(i)
    return index


def build_postal_index(pool: pd.DataFrame) -> dict:
    """postal code -> list of positional indices into `pool` (non-empty only)."""
    index = defaultdict(list)
    for i, postal in enumerate(pool["postal"]):
        if postal:
            index[postal].append(i)
    return index


def candidates_for_s1(name_core: str, postal: str, token_index: dict, postal_index: dict,
                       pool_ids) -> list:
    """Union rare-token and postal-code candidates for one S1 record."""
    hit_positions = set()
    for t in set(name_core.split()):
        hit_positions.update(token_index.get(t, ()))
    if postal:
        hit_positions.update(postal_index.get(postal, ()))
    return [pool_ids[i] for i in hit_positions]


def block_country(s1_norm: pd.DataFrame, pool_norm: pd.DataFrame) -> dict:
    """Return {s1_entity_id: [candidate ids]} for one (split, country) slice.

    pool_norm must already be restricted to the same country and combine
    S2 + S3 rows (with their entity_id column intact).
    """
    token_index = build_rare_token_index(pool_norm)
    postal_index = build_postal_index(pool_norm)
    pool_ids = pool_norm["entity_id"].tolist()

    out = {}
    for row in s1_norm.itertuples():
        cands = candidates_for_s1(row.name_core, row.postal, token_index, postal_index, pool_ids)
        if len(cands) > K_FINAL:
            cands = cands[:K_FINAL]  # v1: arbitrary cap; v2 will rank by similarity first
        out[row.entity_id] = cands
    return out


def run_split(split: str) -> dict:
    """Block every S1 entity in `split` ("train" or "test"), grouped by country.

    Memory optimisation: instead of loading the full S2+S3 pool into RAM and
    then filtering by country, we load only the 4 blocking columns (see
    _BLOCKING_COLS) and process one country at a time so peak RAM stays
    manageable even when source files are hundreds of MB on disk.
    """
    s1 = load_normalized(split, "source1")
    # Load pool sources with minimal columns only
    s2 = load_normalized(split, "source2")
    s3 = load_normalized(split, "source3")
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3  # free RAM immediately after concat

    all_candidates = {}
    for country, s1_group in s1.groupby("country"):
        pool_group = pool[pool["country"] == country].reset_index(drop=True)
        t0 = time.time()
        cands = block_country(s1_group, pool_group)
        all_candidates.update(cands)
        n_cand = sum(len(v) for v in cands.values())
        print(f"  {split}/{country}: {len(s1_group):,} S1 rows, pool {len(pool_group):,}, "
              f"{n_cand:,} candidate pairs, {time.time() - t0:.1f}s")
    return all_candidates


def measure_recall(candidates: dict, ground_truth_path: str) -> None:
    """Pair recall and entity full-recall against train_ground_truth.tsv."""
    gt = read_tsv(ground_truth_path)
    total_pairs = 0
    found_pairs = 0
    full_recall_entities = 0
    non_singleton = 0
    for row in gt.itertuples():
        true = set(parse_ids(row.matched_entity_ids))
        if not true:
            continue
        non_singleton += 1
        cand = set(candidates.get(row.source1_entity_id, []))
        hit = true & cand
        total_pairs += len(true)
        found_pairs += len(hit)
        if hit == true:
            full_recall_entities += 1

    print(f"\nPair recall: {found_pairs:,}/{total_pairs:,} = {found_pairs / total_pairs:.4%}")
    print(f"Entity full-recall (non-singletons): {full_recall_entities:,}/{non_singleton:,} "
          f"= {full_recall_entities / non_singleton:.4%}")


if __name__ == "__main__":
    print("Blocking train split...")
    train_candidates = run_split("train")
    measure_recall(train_candidates, os.path.join(DATA, "train", "train_ground_truth.tsv"))
