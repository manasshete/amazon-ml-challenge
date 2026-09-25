"""One-off smoke test: load every train/test file, print shape + a sample row,
and run the assert_valid_source_file sanity checks. Not part of the pipeline;
delete or ignore once you trust io_utils.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from io_utils import assert_valid_source_file, load_ground_truth, read_tsv

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")

for split in ("train", "test"):
    for src in ("source1", "source2", "source3"):
        path = os.path.join(DATA, split, f"{split}_{src}.tsv")
        if not os.path.isfile(path):
            print(f"MISSING: {path}")
            continue
        t0 = time.time()
        df = read_tsv(path)
        dt = time.time() - t0
        prefix = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}[src]
        assert_valid_source_file(path, prefix)
        print(f"{split}/{src}: {len(df):>9} rows, {dt:5.1f}s  | sample: {df.iloc[0].to_dict()}")

gt_path = os.path.join(DATA, "train", "train_ground_truth.tsv")
t0 = time.time()
gt = load_ground_truth(gt_path)
dt = time.time() - t0
n_singleton = sum(1 for v in gt.values() if not v)
print(f"\nground_truth: {len(gt)} S1 entities, {dt:.1f}s")
print(f"singleton fraction (trivial-baseline score): {n_singleton / len(gt):.4f}")
match_counts = {}
for v in gt.values():
    match_counts[len(v)] = match_counts.get(len(v), 0) + 1
print("match-count distribution (n_matches: count), top 10:")
for k in sorted(match_counts, key=lambda k: -match_counts[k])[:10]:
    print(f"  {k}: {match_counts[k]}")
