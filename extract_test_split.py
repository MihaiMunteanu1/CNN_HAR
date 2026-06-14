"""
Extract only the samples from the "test" split from a large .npz file (train+val+test,
with augmentations) into a small .npz file, suitable for committing to git (Streamlit demo).

Usage:
    python extract_test_split.py --input ../hog/hog_aug_tvt_19_f10_g2_runfix.npz --output data/hog_test_only.npz
"""

import argparse
from pathlib import Path

import numpy as np


TEST_SUBJECTS = [2,3,5,6,7,8,9,10,22]


def split_from_subject(subject_id: int) -> str:
    return "test" if subject_id in TEST_SUBJECTS else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",required=True, help="npz mare (train+val+test)")
    parser.add_argument("--output", default="data/hog_test_only.npz", help="npz mic, doar test")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    features = data["features"]
    bboxes = data["bboxes"]
    labels = data["labels"]
    metadata = data["metadata"]

    keep = []
    for i, meta in enumerate(metadata):
        split = meta.get("split")
        if split is None:
            subject = meta.get("subject")
            split = split_from_subject(subject) if subject is not None else None
        if split == "test":
            keep.append(i)

    print(f"Keeping {len(keep)}/{len(metadata)} test samples")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features[keep],
        bboxes=bboxes[keep],
        labels=labels[keep],
        metadata=np.array([metadata[i] for i in keep], dtype=object),
    )
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()