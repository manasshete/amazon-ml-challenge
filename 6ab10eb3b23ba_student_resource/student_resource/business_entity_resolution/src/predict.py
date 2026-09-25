"""End-to-end test inference: block -> features -> score -> post-process ->
write matching_results.tsv + candidate_pairs.tsv.

Run: .venv\\Scripts\\python.exe src\\predict.py --out ../../output
Requires artifacts/model_fold*.txt and artifacts/calibrator.pkl (train.py)
and artifacts/normalized/test_*.parquet (build_normalized.py) to exist.
"""

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blocking import run_split  # noqa: E402
from features import FEATURE_COLUMNS, build_pair_features  # noqa: E402
from io_utils import read_tsv, write_output  # noqa: E402
from postprocess import expected_f05_predictions, one_to_one  # noqa: E402
from train import LGB_PARAMS  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "dataset")


def load_ensemble():
    boosters = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(ARTIFACTS, "model_fold*.txt")))]
    with open(os.path.join(ARTIFACTS, "calibrator.pkl"), "rb") as f:
        iso = pickle.load(f)
    return boosters, iso


def score_features(feats: pd.DataFrame, boosters, iso) -> pd.DataFrame:
    X = feats[FEATURE_COLUMNS]
    raw = np.mean([b.predict(X, num_iteration=b.best_iteration) for b in boosters], axis=0)
    feats = feats.copy()
    feats["prob"] = iso.predict(raw)
    return feats


def run(out_dir: str, p_miss: float = 0.02, apply_one_to_one: bool = True) -> None:
    print("Blocking test split...")
    candidates = run_split("test")

    print("Building test pair features...")
    feats = build_pair_features("test", candidates)
    print(feats.shape, "candidate pairs")

    boosters, iso = load_ensemble()
    feats = score_features(feats, boosters, iso)

    if apply_one_to_one:
        feats = one_to_one(feats)

    matched = expected_f05_predictions(feats, p_miss=p_miss)

    s1_test = read_tsv(os.path.join(DATA, "test", "test_source1.tsv"))
    s1_ids = s1_test["entity_id"].tolist()

    os.makedirs(out_dir, exist_ok=True)
    write_output(os.path.join(out_dir, "matching_results.tsv"), s1_ids, matched, "matched_entity_ids")

    candidate_map = feats.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_output(os.path.join(out_dir, "candidate_pairs.tsv"), s1_ids, candidate_map, "candidate_entity_ids")
    print("Wrote", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "output"))
    parser.add_argument("--p-miss", type=float, default=0.02,
                        help="estimated probability a true match falls outside the candidate set")
    args = parser.parse_args()
    run(args.out, p_miss=args.p_miss)
