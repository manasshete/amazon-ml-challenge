"""GroupKFold LightGBM training on pairwise features, with OOF predictions
and isotonic calibration (Section 7.3 of the plan).

Run: .venv\\Scripts\\python.exe src\\train.py
Expects artifacts/train_features.parquet (produced by features.py) to exist.
Writes:
  artifacts/model_fold{i}.txt      -- one LightGBM booster per fold
  artifacts/oof_predictions.parquet -- s1_id, cand_id, label, prob (calibrated)
  artifacts/calibrator.pkl         -- isotonic regressor fit on OOF
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS  # noqa: E402
from evaluate import macro_f05  # noqa: E402
from io_utils import read_tsv  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")

SEED = 0
N_FOLDS = 5

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbosity": -1,
}


def train_cv(feats: pd.DataFrame) -> tuple:
    """GroupKFold (grouped by s1_id) LightGBM training.

    Returns (boosters, oof_probs). Grouping by s1_id (not a random row split)
    keeps every candidate of a given S1 entity in the same fold -- otherwise
    rank/reverse-rank features would leak information about a held-out S1's
    own candidate set into the training fold.
    """
    X = feats[FEATURE_COLUMNS]
    y = feats["label"].values
    groups = feats["s1_id"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(feats))
    boosters = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        train_set = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        val_set = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=train_set)

        booster = lgb.train(
            LGB_PARAMS, train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)
        boosters.append(booster)
        print(f"fold {fold}: best_iteration={booster.best_iteration} "
              f"best_score={booster.best_score['valid_0']['auc']:.4f}")

    return boosters, oof


def calibrate(oof_probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Isotonic regression fit on OOF probabilities -> calibrated probabilities.

    Needed because raw LightGBM scores aren't true probabilities, and the
    expected-F0.5 post-processing (postprocess.py) requires calibrated ones
    to simulate outcomes correctly.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_probs, labels)
    return iso


def two_threshold_predictions(feats: pd.DataFrame, t_first: float, t_add: float) -> dict:
    """Baseline post-processing (Section 8.2): sort each S1's candidates by
    calibrated probability; include the top one if >= t_first, include each
    additional one if >= t_add."""
    preds = {}
    for s1_id, group in feats.groupby("s1_id"):
        group = group.sort_values("prob", ascending=False)
        chosen = []
        for i, (_, row) in enumerate(group.iterrows()):
            thresh = t_first if i == 0 else t_add
            if row["prob"] >= thresh:
                chosen.append(row["cand_id"])
            else:
                break
        preds[s1_id] = chosen
    return preds


def grid_search_thresholds(feats: pd.DataFrame, true_map: dict) -> tuple:
    """Grid-search (t_first, t_add) on OOF predictions against the exact metric."""
    best = (0.5, 0.5, -1.0)
    for t_first in np.arange(0.3, 0.71, 0.05):
        for t_add in np.arange(t_first, 0.96, 0.05):
            preds = two_threshold_predictions(feats, t_first, t_add)
            score = macro_f05(preds, true_map)
            if score > best[2]:
                best = (t_first, t_add, score)
    return best


if __name__ == "__main__":
    feats_path = os.path.join(ARTIFACTS, "train_features.parquet")
    feats = pd.read_parquet(feats_path)
    print("Loaded", feats.shape, "positive rate:", feats["label"].mean())

    boosters, oof = train_cv(feats)
    feats["oof_raw"] = oof

    iso = calibrate(oof, feats["label"].values)
    feats["prob"] = iso.predict(oof)

    os.makedirs(ARTIFACTS, exist_ok=True)
    for i, booster in enumerate(boosters):
        booster.save_model(os.path.join(ARTIFACTS, f"model_fold{i}.txt"))
    with open(os.path.join(ARTIFACTS, "calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)
    feats[["s1_id", "cand_id", "label", "oof_raw", "prob"]].to_parquet(
        os.path.join(ARTIFACTS, "oof_predictions.parquet"), index=False
    )

    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    true_map = {}
    from io_utils import parse_ids
    for row in gt.itertuples():
        true_map[row.source1_entity_id] = parse_ids(row.matched_entity_ids)

    t_first, t_add, score = grid_search_thresholds(feats, true_map)
    print(f"\nBest thresholds: t_first={t_first:.2f} t_add={t_add:.2f} -> OOF macro F0.5={score:.4f}")
    print("Singleton baseline:", sum(1 for v in true_map.values() if not v) / len(true_map))
