"""Safe TSV I/O for the entity resolution pipeline.

Why this file exists: pandas' default CSV reader is dangerous on this data
because business names/addresses contain quote characters, and IDs/list
strings like "NA" would otherwise get silently converted to NaN. Every
loader in the pipeline should go through here instead of calling
pd.read_csv directly.
"""

import csv
import os

import pandas as pd

DELIM = "\t"


def read_tsv(path: str) -> pd.DataFrame:
    """Read a tab-separated source/ground-truth file safely.

    - sep="\t": files are tab-separated; commas appear inside addresses and
      ID lists, so a comma-based reader would split columns incorrectly.
    - dtype=str: every column (including entity IDs) is read as text. If we
      let pandas guess types, an ID column could be mis-parsed and business
      names that look numeric could be mangled.
    - keep_default_na=False, na_filter=False: pandas normally treats
      strings like "NA", "NULL", "None" as missing values (NaN). Business
      names such as "NA Traders" would be silently destroyed. We keep every
      cell as the literal string it is; a genuinely empty field becomes "".
    - quoting=csv.QUOTE_NONE: business names occasionally contain a literal
      double-quote character. The default CSV quoting rules would treat
      that as the start/end of a quoted field and swallow the rest of the
      line. QUOTE_NONE disables that interpretation entirely.
    """
    return pd.read_csv(
        path,
        sep=DELIM,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def parse_ids(cell: str) -> list:
    """Turn a comma-separated ID-list cell into a Python list.

    Empty string -> []. Filters out any accidental empty tokens from
    trailing/double commas so we never store "" as if it were a real ID.
    """
    if not cell:
        return []
    return [x for x in cell.split(",") if x]


def load_ground_truth(path: str) -> dict:
    """Return {source1_entity_id: [matched ids...]} from train_ground_truth.tsv."""
    df = read_tsv(path)
    return {row.source1_entity_id: parse_ids(row.matched_entity_ids) for row in df.itertuples()}


def write_output(path: str, s1_ids, mapping: dict, col: str) -> None:
    """Write a matching_results.tsv / candidate_pairs.tsv style file.

    s1_ids: the full, required list of S1 ids for this split (every one of
    these gets exactly one row, per the challenge's format rules).
    mapping: {s1_id: iterable of matched/candidate ids}.
    col: output column name ("matched_entity_ids" or "candidate_entity_ids").

    dict.fromkeys(...) dedupes an id list while preserving first-seen order
    (a plain set() would work too but silently reorders ids, which makes
    diffs between submission versions harder to read).
    """
    rows = []
    for s1 in s1_ids:
        ids = mapping.get(s1, [])
        deduped = list(dict.fromkeys(ids))
        rows.append((s1, ",".join(deduped)))

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", col])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out_df.to_csv(path, sep=DELIM, index=False, quoting=csv.QUOTE_NONE)


def assert_valid_source_file(path: str, prefix: str) -> None:
    """Sanity-check a loaded source file: correct ID prefix, no duplicate IDs.

    Cheap to run once per file load; catches corrupted downloads or a wrong
    --data path early instead of producing confusing errors three stages
    later in the pipeline.
    """
    df = read_tsv(path)
    ids = df["entity_id"]
    assert ids.str.startswith(prefix).all(), f"{path}: found IDs without prefix {prefix}"
    assert ids.is_unique, f"{path}: duplicate entity_id values found"
