"""
Evaluate an ensemble of trained Conv3D HAR checkpoints on the KTH test split.
Loads every checkpoint , runs them over the test set, averages predictions across the
ensemble and prints per-model accuracy, the ensemble accuracy, and the confusion matrix.
"""

import argparse
import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HOGDataset, KTH_CLASSES
from model import build_model


def expand_globs(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if matched:
            out.extend(matched)
        elif os.path.exists(p):
            out.append(p)
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _temporal_shift_tensor(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    if shift > 0:
        pad = torch.zeros((x.size(0), shift) + x.shape[2:], device=x.device, dtype=x.dtype)
        return torch.cat([pad, x[:, :-shift]], dim=1)
    shift = -shift
    pad = torch.zeros((x.size(0), shift) + x.shape[2:], device=x.device, dtype=x.dtype)
    return torch.cat([x[:, shift:], pad], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../hog/hog_aug_7.npz")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--include_diff", action="store_true", default=True)
    parser.add_argument("--no_include_diff", dest="include_diff", action="store_false")
    parser.add_argument("--include_bbox", action="store_true", default=True)
    parser.add_argument("--no_include_bbox", dest="include_bbox", action="store_false")
    parser.add_argument("--include_bbox_vel", action="store_true", default=True)
    parser.add_argument("--no_include_bbox_vel", dest="include_bbox_vel", action="store_false")
    parser.add_argument("--ensemble_mode", type=str, default="logits",
                        choices=["logits", "softmax"])
    parser.add_argument("--tta_reverse", action="store_true", default=False)
    parser.add_argument("--tta_shift", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpts = expand_globs(args.checkpoints)
    if not ckpts:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints}")

    test_dataset = HOGDataset(
        args.data_path, split="test",
        as_image=True, augment=False,
        include_diff=args.include_diff,
        include_bbox=args.include_bbox,
        include_bbox_vel=args.include_bbox_vel,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    sample_x, _ = test_dataset[0]
    sample_shape = tuple(sample_x.shape)
    print(f"Test samples: {len(test_dataset)} | sample shape: {sample_shape}")

    print(f"\nLoading {len(ckpts)} checkpoint(s):")
    models = []
    for ckpt in ckpts:
        m = build_model(hog_shape=sample_shape)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
        print(f"  loaded {ckpt}")

    n_classes = len(KTH_CLASSES)
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    individual_correct = [0 for _ in models]
    ensemble_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits_sum = None
            probs_sum = None
            for i, m in enumerate(models):
                logits = m(inputs)
                if args.ensemble_mode == "softmax":
                    probs = torch.softmax(logits, dim=1)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                else:
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                individual_correct[i] += logits.argmax(1).eq(labels).sum().item()

            tta_logits_sum = logits_sum
            tta_probs_sum = probs_sum
            if args.tta_reverse or args.tta_shift > 0:
                if args.tta_reverse:
                    flipped = torch.flip(inputs, dims=[1])
                    for m in models:
                        logits = m(flipped)
                        if args.ensemble_mode == "softmax":
                            probs = torch.softmax(logits, dim=1)
                            tta_probs_sum = probs if tta_probs_sum is None else tta_probs_sum + probs
                        else:
                            tta_logits_sum = logits if tta_logits_sum is None else tta_logits_sum + logits

                if args.tta_shift > 0:
                    for shift in (args.tta_shift, -args.tta_shift):
                        shifted = _temporal_shift_tensor(inputs, shift)
                        for m in models:
                            logits = m(shifted)
                            if args.ensemble_mode == "softmax":
                                probs = torch.softmax(logits, dim=1)
                                tta_probs_sum = probs if tta_probs_sum is None else tta_probs_sum + probs
                            else:
                                tta_logits_sum = logits if tta_logits_sum is None else tta_logits_sum + logits

            if args.ensemble_mode == "softmax":
                denom = len(models)
                if args.tta_reverse or args.tta_shift > 0:
                    denom += (len(models) if args.tta_reverse else 0) + (2 * len(models) if args.tta_shift > 0 else 0)
                avg_probs = tta_probs_sum / max(denom, 1)
                preds = avg_probs.argmax(1)
            else:
                denom = len(models)
                if args.tta_reverse or args.tta_shift > 0:
                    denom += (len(models) if args.tta_reverse else 0) + (2 * len(models) if args.tta_shift > 0 else 0)
                avg_logits = tta_logits_sum / max(denom, 1)
                preds = avg_logits.argmax(1)

            ensemble_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf[t, p] += 1

    print("\nIndividual model accuracies:")
    for i, ckpt in enumerate(ckpts):
        acc = 100.0 * individual_correct[i] / max(total, 1)
        print(f"  [{i}] {os.path.basename(ckpt):40s}  {acc:.2f}%")

    ens_acc = 100.0 * ensemble_correct / max(total, 1)
    delta = ens_acc - max(100.0 * c / max(total, 1) for c in individual_correct)
    print(
        f"\nEnsemble (mean {args.ensemble_mode}): {ens_acc:.2f}%   "
        f"(Δ over best single: {delta:+.2f}%)"
    )

    print("\nEnsemble confusion matrix (rows=true, cols=pred):")
    header = "          " + " ".join(f"{c[:5]:>6s}" for c in KTH_CLASSES)
    print(header)
    for i, cls in enumerate(KTH_CLASSES):
        row = " ".join(f"{conf[i, j]:>6d}" for j in range(n_classes))
        recall = 100.0 * conf[i, i] / max(conf[i].sum(), 1)
        print(f"{cls[:9]:>9s} {row}   recall={recall:5.1f}%")


if __name__ == "__main__":
    main()