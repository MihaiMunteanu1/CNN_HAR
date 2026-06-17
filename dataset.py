"""
PyTorch Dataset for KTH HOG features.

HOGDataset loads either a pre-computed .npz (output of extract_hog_augmented.py)
or a bbox-only .json (HOG recomputed at load time), filters samples by split, and
yields per-sample tensors. With as_image=True it reshapes the flat HOG vector into
a (T, C, H, W) tensor and can append extra channel streams (motion via HOG diff,
bbox, bbox-velocity), plus online train-time augmentation (Gaussian noise, feature
dropout, temporal reverse/shift).
"""

import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import re

KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(KTH_CLASSES)}

TRAIN_SUBJECTS = {11, 12, 13, 14, 15, 16, 17, 18}
VAL_SUBJECTS   = {19, 20, 21, 23, 24, 25,  1,  4}
TEST_SUBJECTS  = { 2,  3,  5,  6,  7,  8,  9, 10, 22}
VALID_SPLITS = ("train", "val", "test")


def split_from_video_key(video_key):
    head = str(video_key).replace("\\", "/").split("/", 1)[0]
    return head if head in VALID_SPLITS else None


def split_from_subject(subject):
    if subject in TRAIN_SUBJECTS: return "train"
    if subject in VAL_SUBJECTS:   return "val"
    if subject in TEST_SUBJECTS:  return "test"
    return None

HOG_WIN_W = 64
HOG_WIN_H = 128


HOG_NBLOCKS_X = 7
HOG_NBLOCKS_Y = 15
HOG_FEAT_PER_BLOCK = 36
HOG_FEAT_PER_FRAME = HOG_NBLOCKS_X * HOG_NBLOCKS_Y * HOG_FEAT_PER_BLOCK  # 3780


def parse_kth_filename(filename):
    name = Path(filename).name
    match = re.match(r'person(\d+)_(\w+)', name, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        remaining = match.group(2).lower()
        for action in KTH_CLASSES:
            if remaining.startswith(action):
                return subject, action
    return None, None


def _extract_groups_from_video(video_path, groups, frame_w, frame_h):
    """Read a video once and compute the HOG vector + normalized bbox for every
    (group, frame) that has a bbox. Returns dicts keyed by (group_idx, frame_idx)."""
    needed = {}
    for gi, group in enumerate(groups):
        for fi, frame_data in enumerate(group):
            fidx = frame_data["frame_idx"]
            bbox = frame_data.get("selected_bbox") or (frame_data.get("bboxes") or [None])[0]
            if bbox is None:
                continue
            if fidx not in needed:
                needed[fidx] = []
            needed[fidx].append((gi, fi, bbox))

    if not needed:
        return {}, {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, {}

    hog_desc = cv2.HOGDescriptor(
        _winSize=(HOG_WIN_W, HOG_WIN_H),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    frame_feats = {}
    frame_bbox = {}
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            resized = cv2.resize(frame, (frame_w, frame_h))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            for gi, fi, bbox in needed[idx]:
                x = max(0, min(int(bbox["x"]), frame_w - 1))
                y = max(0, min(int(bbox["y"]), frame_h - 1))
                w = max(1, min(int(bbox["w"]), frame_w - x))
                h = max(1, min(int(bbox["h"]), frame_h - y))
                crop = gray[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (HOG_WIN_W, HOG_WIN_H))
                feat = hog_desc.compute(crop_resized)
                if feat is not None:
                    frame_feats[(gi, fi)] = feat.flatten()
                    cx = (x + w / 2.0) / frame_w
                    cy = (y + h / 2.0) / frame_h
                    frame_bbox[(gi, fi)] = np.array(
                        [cx, cy, w / frame_w, h / frame_h], dtype=np.float32
                    )
        idx += 1

    cap.release()
    return frame_feats, frame_bbox


class HOGDataset(Dataset):
    """
    HOG features dataset for KTH.
    Args:
        json_path: path to .json (bbox-only) or .npz (pre-computed) file.
        split: "train", "val", or "test"
        video_root: path containing the KTH video files (only used for .json).
        transform: optional callable applied to feature tensor
        as_image: if True, returns tensor of shape (T, C=36, H=15, W=7) for the
                  Conv3D model. If False, returns the flat vector (T*3780,).
        augment: if True (only on train split), applies feature-level
                 augmentation: Gaussian noise + random feature dropout.
        noise_std: std of additive Gaussian noise (input is in [0,1]).
        feat_dropout_p: probability per feature of being zero-masked.
    """

    def __init__(self, json_path, split="train",
                 video_root="/home/mmuntean/kth_organized", transform=None,
                 as_image: bool = False, augment: bool = False,
                 noise_std: float = 0.02, feat_dropout_p: float = 0.05,
                 include_diff: bool = False, include_bbox: bool = False,
                 include_bbox_vel: bool = False,
                 temporal_reverse_p: float = 0.0, temporal_shift_max: int = 0):
        self.split = split
        self.transform = transform
        self.as_image = as_image
        self.augment = augment and split == "train"
        self.noise_std = noise_std
        self.feat_dropout_p = feat_dropout_p
        self.include_diff = include_diff
        self.include_bbox = include_bbox
        self.include_bbox_vel = include_bbox_vel
        self.temporal_reverse_p = temporal_reverse_p
        self.temporal_shift_max = temporal_shift_max
        self.samples = []
        self.metadata = []

        print(f"Loading HOG data from {json_path} for {split} split...")
        if str(json_path).endswith(".npz"):
            self._load_npz(json_path)
        else:
            self._load_json(json_path, video_root)

        print(f"Loaded {len(self.samples)} {split} samples.")

    def _load_json(self, json_path, video_root):
        with open(json_path, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        frame_w = config.get("frame_width", 160)
        frame_h = config.get("frame_height", 120)

        for video_key, video_data in data.get("videos", {}).items():
            subject, action = parse_kth_filename(video_key)
            if subject is None:
                continue

            sample_split = split_from_video_key(video_key) or split_from_subject(subject)
            if sample_split != self.split:
                continue

            label_idx = CLASS_TO_IDX[action]
            video_path = Path(video_root) / video_key
            groups = video_data.get("groups", [])
            if not groups:
                continue

            frame_feats, frame_bbox = _extract_groups_from_video(
                video_path, groups, frame_w, frame_h
            )

            for gi, group in enumerate(groups):
                T = len(group)
                vecs = [frame_feats.get((gi, fi)) for fi in range(T)]
                if any(v is None for v in vecs):
                    continue
                combined = np.concatenate(vecs).astype(np.float32)
                bbox_seq = np.stack(
                    [frame_bbox.get((gi, fi), np.zeros(4, dtype=np.float32))
                     for fi in range(T)], axis=0
                )
                self.samples.append((
                    torch.from_numpy(combined),
                    torch.from_numpy(bbox_seq),
                    label_idx,
                ))
                self.metadata.append({
                    "video_key": video_key,
                    "subject": subject,
                    "action": action,
                    "label_idx": label_idx,
                    "group_idx": gi,
                    "frame_indices": [int(group[fi]["frame_idx"]) for fi in range(T)],
                })

    def _load_npz(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        features = data["features"]      # (N, T*3780) float32
        bboxes = data["bboxes"]          # (N, T, 4) float32
        labels = data["labels"]          # (N,) int64
        all_meta = data["metadata"].tolist()

        for i in range(features.shape[0]):
            meta = all_meta[i] if i < len(all_meta) else {}
            sample_split = meta.get("split")
            if sample_split is None:
                subject = meta.get("subject")
                if subject is None:
                    continue
                sample_split = split_from_subject(subject)
                if sample_split is None:
                    continue
            if sample_split != self.split:
                continue

            self.samples.append((
                torch.from_numpy(features[i].astype(np.float32, copy=False)),
                torch.from_numpy(bboxes[i].astype(np.float32, copy=False)),
                int(labels[i]),
            ))
            self.metadata.append(meta)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _shift_with_pad(seq: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return seq
        if shift > 0:
            pad = torch.zeros((shift,) + seq.shape[1:], device=seq.device, dtype=seq.dtype)
            return torch.cat([pad, seq[:-shift]], dim=0)
        shift = -shift
        pad = torch.zeros((shift,) + seq.shape[1:], device=seq.device, dtype=seq.dtype)
        return torch.cat([seq[shift:], pad], dim=0)

    def _temporal_augment(self, x: torch.Tensor, bbox: torch.Tensor):
        """Apply random temporal reverse and/or shift to the frame sequence"""
        T = x.numel() // HOG_FEAT_PER_FRAME
        if T <= 1:
            return x, bbox
        x_seq = x.view(T, HOG_FEAT_PER_FRAME)
        bbox_seq = bbox

        if self.temporal_reverse_p > 0 and torch.rand(1).item() < self.temporal_reverse_p:
            x_seq = torch.flip(x_seq, dims=[0])
            bbox_seq = torch.flip(bbox_seq, dims=[0])

        if self.temporal_shift_max > 0:
            shift = int(torch.randint(-self.temporal_shift_max, self.temporal_shift_max + 1, (1,)).item())
            if shift != 0:
                x_seq = self._shift_with_pad(x_seq, shift)
                bbox_seq = self._shift_with_pad(bbox_seq, shift)

        return x_seq.reshape(-1), bbox_seq

    def _to_image(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        T = x.numel() // HOG_FEAT_PER_FRAME
        x = x.view(T, HOG_NBLOCKS_Y, HOG_NBLOCKS_X, HOG_FEAT_PER_BLOCK)
        x = x.permute(0, 3, 1, 2).contiguous()  # (T, 36, 15, 7)

        if self.include_diff:
            diff = torch.zeros_like(x)
            diff[1:] = x[1:] - x[:-1]
            x = torch.cat([x, diff], dim=1)  # (T, 72, 15, 7)

        if self.include_bbox:
            _, _, H, W = x.shape
            bbox_map = bbox.view(T, 4, 1, 1).expand(T, 4, H, W)
            x = torch.cat([x, bbox_map], dim=1)

        if self.include_bbox_vel:
            _, _, H, W = x.shape
            bbox_vel = torch.zeros_like(bbox)
            bbox_vel[1:] = bbox[1:] - bbox[:-1]
            bbox_vel_map = bbox_vel.view(T, 4, 1, 1).expand(T, 4, H, W)
            x = torch.cat([x, bbox_vel_map], dim=1)
        return x

    def __getitem__(self, idx):
        x, bbox, y = self.samples[idx]

        if self.augment:
            x = x.clone()
            bbox = bbox.clone()
            x, bbox = self._temporal_augment(x, bbox)
            if self.noise_std > 0:
                x.add_(torch.randn_like(x) * self.noise_std)
            if self.feat_dropout_p > 0:
                mask = torch.empty_like(x).bernoulli_(1.0 - self.feat_dropout_p)
                x.mul_(mask)

        if self.as_image:
            x = self._to_image(x, bbox)

        if self.transform:
            x = self.transform(x)
        return x, y