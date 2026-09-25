"""Candidate generation (blocking): union of a rare-token inverted index,
postal-code exact match, and char n-gram TF-IDF retrieval on name and
address (Section 6 of the plan), all restricted to same-country pools
(EDA confirmed 100% of true matches share country).

Why TF-IDF needs care at this scale: a brute-force (n_s1 x n_pool) dense
similarity matrix is impossible -- US alone has ~1.3M S1 records against a
~6.2M S2+S3 pool, so even a 1,000-row chunk densified against the full pool
is 1,000 x 6,200,000 x 8 bytes ~= 49 GB. `_sparse_topk_per_row` below never
densifies: scipy keeps a sparse @ sparse product sparse (a pair of rows
only gets a nonzero entry if they share at least one n-gram), and we then
take each row's top-K from only its nonzero entries. Row nnz stays small in
practice because `min_df`/`max_df` on the vectorizer drop n-grams that are
either too rare to matter or so common they'd blow up row density (the
"store"-token problem, generalized to n-grams).

The rare-token inverted index and postal exact-match keys are still unioned
in: they catch some exact-token and exact-postal cases outside whatever K
the TF-IDF retrieval keeps, for near-zero extra cost.
"""

import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import parse_ids, read_tsv  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NORM_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "normalized")
DATA = os.path.join(ROOT, "dataset")

RARE_TOKEN_DF_CUTOFF = 50  # a name_core token is "rare" if <= this many pool rows contain it
K_FINAL = 30  # candidate cap per S1 entity

TFIDF_TOP_K = 20      # candidates kept per S1, per TF-IDF retriever
TFIDF_MIN_DF = 2       # drop n-grams that appear in fewer than this many pool rows (typos/noise)
# max_df as a *fraction* is useless at multi-million-row scale (a char n-gram
# vocabulary is small -- 27^3 possible trigrams -- so even a "rare" n-gram
# clears 5% of 4M+ rows easily). Use an absolute row-count cap instead, so
# the result stays sparse regardless of pool size.
TFIDF_MAX_DF = 500
TFIDF_CHUNK = 2000    # S1 rows per chunk of the sparse similarity matmul


# Only the columns blocking actually needs — avoids loading the full parquet
# (name_norm, addr_norm, numbers, etc.) into RAM, which caused MemoryError
# on large sources like train_source2/3 (~650 MB each on disk, several GB
# once decompressed and held as Python objects).
_BLOCKING_COLS = ["entity_id", "country", "name_core", "postal", "addr_no_landmark"]


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


def _fit_transform_tfidf(s1_text, pool_text):
    """Fit a char n-gram TF-IDF vectorizer on the pool, transform both sides.

    char_wb (word-boundary-aware character n-grams) is robust to typos,
    abbreviations, and suffix variants without needing word tokenization --
    important since France's accented/French text uses the same code path.
    Fitting on the pool (not S1) means the resulting vocabulary/IDF weights
    reflect what's actually being searched, matching how blocking.py is
    used at inference (pool = test S2/S3, never seen at fit time otherwise).
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
        min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, dtype=np.float32,
    )
    pool_matrix = vectorizer.fit_transform(pool_text)
    s1_matrix = vectorizer.transform(s1_text)
    return s1_matrix, pool_matrix


def _sparse_topk_per_row(sim: sp.csr_matrix, k: int):
    """Top-k (col_index, score) pairs per row of a sparse matrix, without
    ever densifying a row. Only examines each row's existing nonzero
    entries, so cost is proportional to actual overlap, not pool size.
    """
    sim = sim.tocsr()
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    results = []
    for i in range(sim.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        row_idx = indices[start:end]
        row_val = data[start:end]
        if len(row_val) > k:
            top = np.argpartition(-row_val, k - 1)[:k]
        else:
            top = np.arange(len(row_val))
        order = np.argsort(-row_val[top])
        top = top[order]
        results.append(list(zip(row_idx[top].tolist(), row_val[top].tolist())))
    return results


def tfidf_candidates(s1_norm: pd.DataFrame, pool_norm: pd.DataFrame, text_col: str, k: int) -> dict:
    """{s1_entity_id: [(cand_id, score), ...]} via chunked sparse TF-IDF cosine similarity.

    text_col: "name_core" or "addr_no_landmark" -- the view to compare on.
    Chunking S1 (not the pool) keeps a single sparse @ sparse product's
    output manageable; the pool side is never split since scipy only
    materializes nonzero (chunk_rows x pool_rows) entries anyway.
    """
    if len(pool_norm) == 0 or len(s1_norm) == 0:
        return {}
    s1_matrix, pool_matrix = _fit_transform_tfidf(s1_norm[text_col], pool_norm[text_col])
    pool_ids = pool_norm["entity_id"].to_numpy()
    pool_t = pool_matrix.T.tocsr()

    out = {}
    s1_ids = s1_norm["entity_id"].tolist()
    for start in range(0, s1_matrix.shape[0], TFIDF_CHUNK):
        chunk = s1_matrix[start:start + TFIDF_CHUNK]
        sim_chunk = (chunk @ pool_t).tocsr()
        for offset, hits in enumerate(_sparse_topk_per_row(sim_chunk, k)):
            s1_id = s1_ids[start + offset]
            out[s1_id] = [(pool_ids[idx], score) for idx, score in hits]
    return out


def block_country(s1_norm: pd.DataFrame, pool_norm: pd.DataFrame) -> dict:
    """Return {s1_entity_id: [candidate ids]} for one (split, country) slice.

    pool_norm must already be restricted to the same country and combine
    S2 + S3 rows (with their entity_id column intact). Unions four cheap
    retrievers: rare-token index, postal exact match, name char n-gram
    TF-IDF, address char n-gram TF-IDF. Final cap keeps the K_FINAL
    candidates with the highest best-retriever score, not an arbitrary
    slice.
    """
    token_index = build_rare_token_index(pool_norm)
    postal_index = build_postal_index(pool_norm)
    pool_ids = pool_norm["entity_id"].tolist()

    name_tfidf = tfidf_candidates(s1_norm, pool_norm, "name_core", TFIDF_TOP_K)
    addr_tfidf = tfidf_candidates(s1_norm, pool_norm, "addr_no_landmark", TFIDF_TOP_K)

    out = {}
    for row in s1_norm.itertuples():
        scores = defaultdict(float)
        for t in set(row.name_core.split()):
            for i in token_index.get(t, ()):
                scores[pool_ids[i]] = max(scores[pool_ids[i]], 1.0)
        if row.postal:
            for i in postal_index.get(row.postal, ()):
                scores[pool_ids[i]] = max(scores[pool_ids[i]], 1.0)
        for cand_id, score in name_tfidf.get(row.entity_id, ()):
            scores[cand_id] = max(scores[cand_id], float(score))
        for cand_id, score in addr_tfidf.get(row.entity_id, ()):
            scores[cand_id] = max(scores[cand_id], float(score))

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:K_FINAL]
        out[row.entity_id] = [cand_id for cand_id, _ in ranked]
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
