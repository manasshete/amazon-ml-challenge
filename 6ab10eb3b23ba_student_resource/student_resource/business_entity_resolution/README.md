# Business Entity Resolution

Pipeline: normalize -> block (candidate generation) -> pairwise features -> LightGBM training (GroupKFold + isotonic calibration) -> post-processing (one-to-one + expected-F0.5) -> test prediction.

## Environment

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
.venv/bin/pip install -r requirements.txt           # Linux/Colab
```

Dataset is expected at `../dataset/{train,test}/*.tsv` (sibling of this directory), matching the original challenge zip layout. It is not committed to git (see `.gitignore`).

## Commands

Run one stage at a time or the whole pipeline:

```
python -m src.run_pipeline --stage normalize   # cache normalized views to artifacts/normalized/*.parquet
python -m src.run_pipeline --stage block       # candidate generation, prints pair recall on train
python -m src.run_pipeline --stage features    # pairwise similarity/rank features -> artifacts/{train,test}_features.parquet
python -m src.run_pipeline --stage train       # GroupKFold LightGBM + calibration -> artifacts/model_fold*.txt
python -m src.run_pipeline --stage predict --out ../output   # writes matching_results.tsv + candidate_pairs.tsv
python -m src.run_pipeline --stage all
```

Then validate before any leaderboard upload:

```
python ../utils/validate_submission.py --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv --test-dir ../dataset/test
```

## Training on Google Colab

`notebooks/colab_train.ipynb` runs the same pipeline on Colab. Upload your `dataset/` folder to Google Drive first (e.g. `MyDrive/amazon_ml_challenge/dataset/`), open the notebook in Colab, update `DRIVE_DATASET_DIR` if needed, and run all cells. A standard CPU runtime is sufficient. Final models and outputs are copied back to Drive so they survive a runtime disconnect.

## Seeds / reproducibility

All randomness (`numpy`, LightGBM, GroupKFold) is fixed via `SEED = 0` in `src/train.py`.
