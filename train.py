import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import HOGDataset, KTH_CLASSES
from model import build_model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}
        self.num_updates = 0

    def update(self, model: nn.Module):
        self.num_updates += 1
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def mixup_data(x, y, alpha, device):
    """Returns (mixed_x, y_a, y_b, lam) for mixup training.
    When alpha <= 0 this is the identity transform (lam=1, y_a==y_b==y)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    use_image = args.model_type in ("cnn", "temporal", "conv3d")

    train_dataset = HOGDataset(
        args.data_path, split="train",
        as_image=use_image, augment=args.augment,
        noise_std=args.noise_std, feat_dropout_p=args.feat_dropout,
        include_diff=args.include_diff and use_image,
        include_bbox=args.include_bbox and use_image,
        include_bbox_vel=args.include_bbox_vel and use_image,
        temporal_reverse_p=args.temporal_reverse_p,
        temporal_shift_max=args.temporal_shift_max,
    )
    val_dataset = HOGDataset(
        args.data_path, split="val",
        as_image=use_image, augment=False,
        include_diff=args.include_diff and use_image,
        include_bbox=args.include_bbox and use_image,
        include_bbox_vel=args.include_bbox_vel and use_image,
    )
    test_dataset = HOGDataset(
        args.data_path, split="test",
        as_image=use_image, augment=False,
        include_diff=args.include_diff and use_image,
        include_bbox=args.include_bbox and use_image,
        include_bbox_vel=args.include_bbox_vel and use_image,
    )

    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check your JSON path or data format.")
        return

    # Distribuția pe clase în train. Pe KTH cu split tvt + T=19/g=10, clasa
    # `running` e masiv sub-reprezentată (~1.6% din train) fiindcă alergătorul
    # iese din cadru → puține grupuri valide. Fără corecție, modelul prezice
    # mereu jogging pentru running (recall ~15%).
    train_labels = np.array([int(s[2]) for s in train_dataset.samples], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=len(KTH_CLASSES))
    print("Train class counts: "
          + ", ".join(f"{c}={n}" for c, n in zip(KTH_CLASSES, class_counts.tolist())))

    # WeightedRandomSampler: oversamplează clasele rare ca fiecare epoch să vadă
    # clasele ~echilibrat. 'inv' = balans complet (w ∝ 1/count), 'sqrt_inv' = mai blând.
    sampler = None
    if args.balanced_sampler != "none":
        safe = np.maximum(class_counts, 1)
        class_w = (1.0 / safe) if args.balanced_sampler == "inv" else (1.0 / np.sqrt(safe))
        sample_w = class_w[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )
        print(f"WeightedRandomSampler activ (mode={args.balanced_sampler}).")

    # Legacy NPZ files generated before the 8/8/9 reorganization have no val
    # split. Use test as a stand-in so the loop still runs end-to-end, but warn
    # loudly so the user knows the early-stop signal is not held-out anymore.
    use_val_for_stop = len(val_dataset) > 0
    if not use_val_for_stop:
        print("WARN: val split empty in this dataset. Falling back to test for early stopping "
              "(no held-out signal — regenerate the data with the new pipeline to fix).")
        eval_dataset = test_dataset
    else:
        eval_dataset = val_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    sample_feature, _ = train_dataset[0]
    print(f"Sample shape: {tuple(sample_feature.shape)} | model_type={args.model_type}")

    model = build_model(
        hog_shape=tuple(sample_feature.shape),
        model_type=args.model_type,
        dropout=args.dropout,
    )
    model = model.to(device)

    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    # Pondere pe clase în loss (alternativă/complement la sampler). A NU stiva
    # 'inv' peste un sampler 'inv' — dublă corecție => over-predicție pe running.
    loss_weight = None
    if args.class_weight_loss != "none":
        safe = np.maximum(class_counts, 1).astype(np.float64)
        w = class_counts.sum() / (len(KTH_CLASSES) * safe)
        if args.class_weight_loss == "sqrt_inv":
            w = np.sqrt(w)
        loss_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("Class-weighted loss (mode=" + args.class_weight_loss + "): "
              + ", ".join(f"{c}={v:.2f}" for c, v in zip(KTH_CLASSES, w.tolist())))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=loss_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    eval_label = "Val" if use_val_for_stop else "Test"
    best_acc = 0.0
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"har_{args.model_type}{args.save_suffix}.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=args.mixup, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None and epoch >= args.ema_start:
                ema.update(model)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        use_ema = ema is not None and ema.num_updates > 0 and epoch >= args.ema_start
        if use_ema:
            ema.apply_shadow(model)
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        if use_ema:
            ema.restore(model)

        eval_acc = 100.0 * correct / max(total, 1)
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"Train Loss: {running_loss/max(len(train_loader),1):.4f} Acc: {train_acc:.2f}% | "
            f"{eval_label} Loss: {eval_loss/max(len(eval_loader),1):.4f} Acc: {eval_acc:.2f}%"
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            if use_ema:
                ema.apply_shadow(model)
            torch.save(model.state_dict(), save_path)
            if use_ema:
                ema.restore(model)
            print(f" => Best model saved with {eval_label.lower()} accuracy {best_acc:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement.")
                break

    print(f"\nFinished. Best {eval_label.lower()} accuracy: {best_acc:.2f}%  (saved to {save_path})")

    # Final unseen-data evaluation on the test set, regardless of whether val
    # was available. When val was used for early stopping this gives the true
    # generalization number; when test was the stand-in this reprints it.
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    n = len(KTH_CLASSES)
    conf = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            preds = model(inputs).argmax(1).cpu().numpy()
            test_total += labels.size(0)
            test_correct += int((preds == labels.numpy()).sum())
            for t, p in zip(labels.numpy(), preds):
                conf[t, p] += 1
    test_acc_final = 100.0 * test_correct / max(test_total, 1)
    print(f"Final test accuracy: {test_acc_final:.2f}%  ({test_correct}/{test_total})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "          " + " ".join(f"{c[:5]:>6s}" for c in KTH_CLASSES)
    print(header)
    for i, cls in enumerate(KTH_CLASSES):
        row = " ".join(f"{conf[i, j]:>6d}" for j in range(n))
        recall = 100.0 * conf[i, i] / max(conf[i].sum(), 1)
        print(f"{cls[:9]:>9s} {row}   recall={recall:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../hog/hog_aug_pose_g4.npz")
    parser.add_argument("--model_type", type=str, default="conv3d",
                        choices=["mlp", "cnn", "temporal", "conv3d"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Classifier-head dropout (lower = less regularization).")
    parser.add_argument("--mixup", type=float, default=0.02,
                        help="Mixup alpha; 0 disables.")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--noise_std", type=float, default=0.003)
    parser.add_argument("--feat_dropout", type=float, default=0.015)
    parser.add_argument("--temporal_reverse_p", type=float, default=0.2,
                        help="Randomly reverse frame order during training.")
    parser.add_argument("--temporal_shift_max", type=int, default=1,
                        help="Max temporal shift (in frames) with zero padding.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay; 0 disables EMA.")
    parser.add_argument("--ema_start", type=int, default=5,
                        help="Start EMA updates after this epoch.")
    parser.add_argument("--patience", type=int, default=60,
                        help="Early-stop patience on test accuracy; 0 disables.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--include_diff", action="store_true", default=True,
                        help="Concatenate pseudo-motion stream (HOG_t - HOG_{t-1}).")
    parser.add_argument("--no_include_diff", dest="include_diff", action="store_false")
    parser.add_argument("--include_bbox", action="store_true", default=True,
                        help="Concatenate bbox metadata (cx, cy, w, h) as 4 extra channels.")
    parser.add_argument("--no_include_bbox", dest="include_bbox", action="store_false")
    parser.add_argument("--include_bbox_vel", action="store_true", default=True,
                        help="Concatenate bbox deltas (dcx, dcy, dw, dh) as 4 extra channels.")
    parser.add_argument("--no_include_bbox_vel", dest="include_bbox_vel", action="store_false")
    parser.add_argument("--balanced_sampler", type=str, default="inv",
                        choices=["none", "inv", "sqrt_inv"],
                        help="WeightedRandomSampler pe clase: 'inv'=balans complet (recomandat "
                             "pentru running), 'sqrt_inv'=moderat, 'none'=dezactivat.")
    parser.add_argument("--class_weight_loss", type=str, default="none",
                        choices=["none", "inv", "sqrt_inv"],
                        help="Pondere pe clase în CrossEntropyLoss. NU folosi 'inv' simultan cu "
                             "--balanced_sampler inv (dublă corecție).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility / ensemble runs.")
    parser.add_argument("--save_suffix", type=str, default="",
                        help="Suffix appended to checkpoint filename (e.g. _s0).")
    args = parser.parse_args()

    train(args)