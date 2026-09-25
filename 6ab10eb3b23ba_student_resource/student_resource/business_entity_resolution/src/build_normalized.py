"""Apply normalize.py to every train/test source file at full scale and
cache the result as Parquet under artifacts/normalized/.

Why cache to Parquet instead of recomputing every time: normalization
involves several regex passes per row across ~24M rows total, so it is not
free. Every later stage (blocking, features) reads the cached Parquet
instead of re-running normalize.py, and Parquet preserves dtypes (unlike
re-reading a TSV) so lists like `numbers` round-trip correctly.

Run: .venv\\Scripts\\python.exe src\\build_normalized.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import read_tsv  # noqa: E402
from normalize import normalize_address, normalize_name  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "normalized")

FILES = [
    ("train", "source1"), ("train", "source2"), ("train", "source3"),
    ("test", "source1"), ("test", "source2"), ("test", "source3"),
]


def normalize_file(split: str, source: str) -> None:
    path = os.path.join(DATA, split, f"{split}_{source}.tsv")
    df = read_tsv(path)

    t0 = time.time()
    name_views = pd.DataFrame(df["business_name"].map(normalize_name).tolist(), index=df.index)
    addr_views = pd.DataFrame(df["business_address"].map(normalize_address).tolist(), index=df.index)
    dt = time.time() - t0

    # "numbers" is a Python list per row; Parquet needs a fixed scalar or a
    # supported nested type, so store it as a comma-joined string instead
    # (parse back with .split(",") when needed).
    addr_views["numbers"] = addr_views["numbers"].map(lambda xs: ",".join(xs))

    out = pd.concat([df[["entity_id", "country"]], name_views, addr_views], axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{split}_{source}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"{split}/{source}: {len(out):>9} rows normalized in {dt:5.1f}s -> {out_path}")


if __name__ == "__main__":
    overall_t0 = time.time()
    for split, source in FILES:
        normalize_file(split, source)
    print(f"\nTotal wall time: {time.time() - overall_t0:.1f}s")
