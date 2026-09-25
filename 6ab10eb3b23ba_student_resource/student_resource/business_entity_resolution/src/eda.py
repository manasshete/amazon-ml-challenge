"""Day-1 EDA checklist from the implementation plan (Section 4).

Answers the structural questions that change pipeline design *before* any
normalization/blocking/model code is written:
  - Q3: does any S2/S3 id ever appear in more than one S1's match list?
        (=> is one-to-one assignment a safe, free precision lever?)
  - Q5: do matched pairs always share the same country?
        (=> can blocking safely restrict candidates to same-country?)
  - Q4: what fraction of S2/S3 records match nothing at all? (orphan rate)
  - Q10: postal-code presence rate (a strong exact-match feature later)
  - Q12: any non-Latin script / accented text? (France + India considerations)

Run: .venv\\Scripts\\python.exe src\\eda.py
Writes findings to notes/eda.md as it goes.
"""

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import parse_ids, read_tsv  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "dataset")
NOTES = os.path.join(os.path.dirname(__file__), "..", "notes", "eda.md")

lines = ["# EDA Findings (Day 1)\n"]


def log(msg: str) -> None:
    print(msg)
    lines.append(msg)


def main() -> None:
    s1 = read_tsv(os.path.join(DATA, "train", "train_source1.tsv"))
    s2 = read_tsv(os.path.join(DATA, "train", "train_source2.tsv"))
    s3 = read_tsv(os.path.join(DATA, "train", "train_source3.tsv"))
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))

    log("## Record counts\n")
    log(f"- S1 train: {len(s1):,}")
    log(f"- S2 train: {len(s2):,}")
    log(f"- S3 train: {len(s3):,}")
    log(f"- ground truth rows: {len(gt):,}\n")

    # ---- Q3: one-to-one assignment check ----------------------------------
    # Build id -> list of S1 ids it was matched to, across ALL ground-truth
    # rows. If every S2/S3 id appears in at most one list, matches never
    # "share" a record between two different S1 entities.
    id_to_s1s = {}
    for row in gt.itertuples():
        s1_id = row.source1_entity_id
        for mid in parse_ids(row.matched_entity_ids):
            id_to_s1s.setdefault(mid, set()).add(s1_id)

    shared = {mid: s1s for mid, s1s in id_to_s1s.items() if len(s1s) > 1}
    log("## Q3: one-to-one assignment\n")
    log(f"- distinct matched S2/S3 ids: {len(id_to_s1s):,}")
    log(f"- ids matched to MORE than one S1: {len(shared):,} "
        f"({len(shared) / max(len(id_to_s1s), 1):.4%})")
    if shared:
        example = next(iter(shared.items()))
        log(f"  example: {example}")
    log("")

    # ---- Q5: does country ever differ within a matched pair? --------------
    s1_country = dict(zip(s1.entity_id, s1.country))
    s2_country = dict(zip(s2.entity_id, s2.country))
    s3_country = dict(zip(s3.entity_id, s3.country))

    def country_of(entity_id: str) -> str:
        if entity_id.startswith("S2-"):
            return s2_country.get(entity_id, "?")
        return s3_country.get(entity_id, "?")

    cross_country = 0
    total_pairs = 0
    for row in gt.itertuples():
        c1 = s1_country.get(row.source1_entity_id, "?")
        for mid in parse_ids(row.matched_entity_ids):
            total_pairs += 1
            if country_of(mid) != c1:
                cross_country += 1

    log("## Q5: country consistency within matched pairs\n")
    log(f"- total ground-truth pairs: {total_pairs:,}")
    log(f"- pairs where S1 country != matched-record country: {cross_country:,} "
        f"({cross_country / max(total_pairs, 1):.4%})\n")

    # ---- Q4: orphan rate in S2/S3 (match nothing) --------------------------
    matched_s2s3_ids = set(id_to_s1s.keys())
    s2_ids = set(s2.entity_id)
    s3_ids = set(s3.entity_id)
    orphan_s2 = len(s2_ids - matched_s2s3_ids)
    orphan_s3 = len(s3_ids - matched_s2s3_ids)

    log("## Q4: orphan rate (records matching no S1 at all)\n")
    log(f"- S2 orphans: {orphan_s2:,} / {len(s2_ids):,} ({orphan_s2 / len(s2_ids):.4%})")
    log(f"- S3 orphans: {orphan_s3:,} / {len(s3_ids):,} ({orphan_s3 / len(s3_ids):.4%})\n")

    # ---- Q10: postal code presence rate ------------------------------------
    postal_re = re.compile(r"\b\d{5,6}(-\d{4})?\b")

    def postal_presence(df) -> float:
        hits = df["business_address"].str.contains(postal_re, regex=True, na=False)
        return hits.mean()

    log("## Q10: postal-code-like token presence (rough regex, per source)\n")
    for name, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        log(f"- {name}: {postal_presence(df):.4%}")
    log("")

    # ---- Q12: script / accent check ----------------------------------------
    def has_non_ascii(s: str) -> bool:
        return any(ord(ch) > 127 for ch in s)

    log("## Q12: non-ASCII character presence in business_name (by source)\n")
    for name, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        frac = df["business_name"].map(has_non_ascii).mean()
        log(f"- {name}: {frac:.4%} of names contain non-ASCII characters")
    log("")

    # ---- country mix ---------------------------------------------------------
    log("## Country mix (train)\n")
    for name, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        counts = Counter(df["country"])
        log(f"- {name}: {dict(counts)}")

    os.makedirs(os.path.dirname(NOTES), exist_ok=True)
    with open(NOTES, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {NOTES}")


if __name__ == "__main__":
    main()
