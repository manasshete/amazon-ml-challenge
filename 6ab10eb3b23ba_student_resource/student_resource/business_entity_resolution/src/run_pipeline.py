"""Single CLI entry point for the whole pipeline.

Usage:
  python -m src.run_pipeline --stage all --data ../dataset --out ../output

Stages (run in order for --stage all):
  normalize -> block -> features -> train -> predict
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def stage_normalize():
    import build_normalized
    for split, source in build_normalized.FILES:
        build_normalized.normalize_file(split, source)


def stage_block():
    import blocking
    for split in ("train", "test"):
        print(f"Blocking {split}...")
        cands = blocking.run_split(split)
        if split == "train":
            gt_path = os.path.join(blocking.DATA, "train", "train_ground_truth.tsv")
            blocking.measure_recall(cands, gt_path)


def stage_features():
    import blocking
    import features

    for split in ("train", "test"):
        print(f"Building {split} features...")
        cands = blocking.run_split(split)
        feats = features.build_pair_features(split, cands)
        if split == "train":
            gt_path = os.path.join(features.DATA, "train", "train_ground_truth.tsv")
            feats = features.attach_labels(feats, gt_path)
        out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        os.makedirs(out_dir, exist_ok=True)
        feats.to_parquet(os.path.join(out_dir, f"{split}_features.parquet"), index=False)


def stage_train():
    import train as train_mod  # avoid clashing with the --stage "train" string

    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    feats = train_mod.pd.read_parquet(os.path.join(artifacts, "train_features.parquet"))
    boosters, oof = train_mod.train_cv(feats)
    feats["oof_raw"] = oof
    iso = train_mod.calibrate(oof, feats["label"].values)
    feats["prob"] = iso.predict(oof)

    import pickle
    for i, booster in enumerate(boosters):
        booster.save_model(os.path.join(artifacts, f"model_fold{i}.txt"))
    with open(os.path.join(artifacts, "calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)
    feats[["s1_id", "cand_id", "label", "oof_raw", "prob"]].to_parquet(
        os.path.join(artifacts, "oof_predictions.parquet"), index=False
    )

    from io_utils import read_tsv, parse_ids
    gt = read_tsv(os.path.join(train_mod.DATA, "train", "train_ground_truth.tsv"))
    true_map = {row.source1_entity_id: parse_ids(row.matched_entity_ids) for row in gt.itertuples()}
    t_first, t_add, score = train_mod.grid_search_thresholds(feats, true_map)
    print(f"Best thresholds: t_first={t_first:.2f} t_add={t_add:.2f} -> OOF macro F0.5={score:.4f}")


def stage_predict(out_dir: str):
    import predict
    predict.run(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["normalize", "block", "features", "train", "predict", "all"],
                        default="all")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    args = parser.parse_args()

    stages = ["normalize", "block", "features", "train", "predict"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== stage: {stage} ===")
        if stage == "normalize":
            stage_normalize()
        elif stage == "block":
            stage_block()
        elif stage == "features":
            stage_features()
        elif stage == "train":
            stage_train()
        elif stage == "predict":
            stage_predict(args.out)
