"""Verify that the train/val/test split contains no case-level leakage.

VIA screening datasets contain several photographs per examined case. If the
split is drawn at the image level, near-duplicate frames of the same cervix end
up on both sides of the split and the reported test metrics are optimistic. This
script asserts that every case_id maps to exactly one split, and prints the
per-split case and class counts.

    python scripts/check_split_leakage.py data/labels.csv
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def main(csv_path):
    df = pd.read_csv(csv_path)

    splits_by_case = defaultdict(set)
    for case_id, split in zip(df["case_id"], df["split"]):
        splits_by_case[case_id].add(split)

    spanning = {c: sorted(s) for c, s in splits_by_case.items() if len(s) > 1}

    print(f"file            : {csv_path}")
    print(f"images          : {len(df)}")
    print(f"cases           : {len(splits_by_case)}")
    print(f"images per case : "
          f"{df.groupby('case_id').size().min()}-"
          f"{df.groupby('case_id').size().max()}")
    print()

    print("split    images  cases   " + "  ".join(f"{n:>12}" for n in
                                                  sorted(df["class_name"].unique())))
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        counts = Counter(sub["class_name"])
        cells = "  ".join(
            f"{counts.get(n, 0):>12}" for n in sorted(df["class_name"].unique())
        )
        print(f"{split:<8} {len(sub):>6}  {sub['case_id'].nunique():>5}   {cells}")
    print()

    if spanning:
        print(f"FAIL: {len(spanning)} case(s) appear in more than one split:")
        for case_id, splits in list(spanning.items())[:20]:
            print(f"  {case_id}: {splits}")
        return 1

    print("PASS: every case belongs to exactly one split (no case-level leakage).")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/labels.csv"
    sys.exit(main(Path(path)))
