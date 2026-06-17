"""
Pre-compute augmented HOG features for KTH and save them to a .npz.
Reads the bbox metadata JSON, decodes each video once, and for every group of T
frames computes a HOG descriptor per frame (crop bbox -> resize 64x128 -> HOG).
Train videos also get num_aug augmented variants (flip, bbox jitter, brightness /
gamma / blur / noise); val and test get the original variant only. The output
.npz holds the features, bboxes, labels, metadata and config arrays consumed by
HOGDataset.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(KTH_CLASSES)}
VALID_SPLITS = ("train", "val", "test")


def split_from_video_key(video_key):
    head = video_key.replace("\\", "/").split("/", 1)[0]
    return head if head in VALID_SPLITS else None

HOG_WIN_W = 64
HOG_WIN_H = 128
HOG_FEAT_PER_FRAME = 3780  # cv2 default HOG on 64x128


def parse_kth_filename(filename):
    name = Path(filename).name
    match = re.match(r'person(\d+)_(\w+)', name, re.IGNORECASE)
    if match:
        subject = int(match.group(1))
        rest = match.group(2).lower()
        for action in KTH_CLASSES:
            if rest.startswith(action):
                return subject, action
    return None, None


def jitter_bbox(bbox, frame_w, frame_h, dx, dy, scale):
    """Augment a bbox: scale it about its center, then translate by (dx, dy).
    The result is clamped so it stays fully inside the frame."""
    cx = bbox["x"] + bbox["w"] / 2.0          # current center
    cy = bbox["y"] + bbox["h"] / 2.0
    new_w = max(1, int(bbox["w"] * scale))    # scaled size
    new_h = max(1, int(bbox["h"] * scale))
    new_x = int(cx - new_w / 2.0 + dx)        # re-center scaled box, then shift
    new_y = int(cy - new_h / 2.0 + dy)

    new_x = max(0, min(new_x, frame_w - 1))
    new_y = max(0, min(new_y, frame_h - 1))
    new_w = max(1, min(new_w, frame_w - new_x))
    new_h = max(1, min(new_h, frame_h - new_y))
    return {"x": new_x, "y": new_y, "w": new_w, "h": new_h}


def apply_gamma_u8(img_u8, gamma):
    """gamma correction to a uint8 image via a 256-entry lookup table"""
    if gamma == 1.0:
        return img_u8
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_u8, table)


def compute_hog_with_aug(frame_bgr, bbox, aug, hog_desc, frame_w, frame_h, rng):
    """Build the HOG vector for one frame under one augmentation spec.

    Pipeline: jitter the bbox -> crop -> resize to 64x128 -> optional flip /
    brightness-contrast / gamma / blur / Gaussian noise -> grayscale -> HOG.
    Returns (hog_vector, post_box) where post_box is the bbox expressed in the
    augmented image's coordinates, or (None, None) if the crop/HOG is empty."""
    box = jitter_bbox(bbox, frame_w, frame_h, aug["dx"], aug["dy"], aug["scale"])

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (HOG_WIN_W, HOG_WIN_H))   # fixed HOG window
    if aug["flip"]:
        crop = cv2.flip(crop, 1)                       # horizontal mirror
    if aug["alpha"] != 1.0 or aug["beta"] != 0:
        crop = cv2.convertScaleAbs(crop, alpha=aug["alpha"], beta=aug["beta"])  # contrast/brightness
    if aug["gamma"] != 1.0:
        crop = apply_gamma_u8(crop, aug["gamma"])
    if aug["blur_ksize"] > 0:
        k = aug["blur_ksize"]
        crop = cv2.GaussianBlur(crop, (k, k), 0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if aug["noise_std"] > 0:
        noise = rng.normal(0.0, aug["noise_std"], size=gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    feat = hog_desc.compute(gray)
    if feat is None:
        return None, None


    if aug["flip"]:
        post_box = dict(box)
        post_box["x"] = frame_w - box["x"] - box["w"]
    else:
        post_box = box

    return feat.flatten().astype(np.float32), post_box


def load_video_frames(video_path, frame_indices, frame_w, frame_h):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    needed = set(frame_indices)
    found = {}
    idx = 0
    max_needed = max(needed) if needed else -1
    while True:
        ret, frame = cap.read()
        if not ret or idx > max_needed:
            break
        if idx in needed:
            found[idx] = cv2.resize(frame, (frame_w, frame_h))
        idx += 1
    cap.release()
    return found


def build_aug_cfg(args):
    """Return the augmentation ranges for the chosen profil"""
    if args.aug_profile == "strong":
        cfg = {
            "scale_min": 0.88,
            "scale_max": 1.15,
            "dx_max": 8,
            "dy_max": 6,
            "alpha_min": 0.75,
            "alpha_max": 1.25,
            "beta_min": -25,
            "beta_max": 25,
            "gamma_min": 0.85,
            "gamma_max": 1.15,
            "noise_std": 4.0,
            "noise_p": 0.7,
            "blur_p": 0.25,
            "blur_ksize": 3,
        }
    else:
        cfg = {
            "scale_min": 0.92,
            "scale_max": 1.08,
            "dx_max": 5,
            "dy_max": 3,
            "alpha_min": 0.85,
            "alpha_max": 1.15,
            "beta_min": -15,
            "beta_max": 15,
            "gamma_min": 0.95,
            "gamma_max": 1.05,
            "noise_std": 2.0,
            "noise_p": 0.5,
            "blur_p": 0.15,
            "blur_ksize": 3,
        }
    cfg["bbox_padding"] = args.bbox_padding
    return cfg


def make_aug_specs(num_aug, rng, aug_cfg):
    """Spec 0 is original, spec 1: horiz flip, the rest are random jit"""
    pad = aug_cfg.get("bbox_padding", 1.25)

    #v0
    specs = [{
        "name": "orig", "flip": False, "scale": 1.0 * pad,
        "dx": 0, "dy": 0, "alpha": 1.0, "beta": 0.0,
        "gamma": 1.0, "noise_std": 0.0, "blur_ksize": 0,
    }]
    if num_aug <= 1:
        return specs

    #v1
    specs.append({
        "name": "flip", "flip": True, "scale": 1.0 * pad,
        "dx": 0, "dy": 0, "alpha": 1.0, "beta": 0.0,
        "gamma": 1.0, "noise_std": 0.0, "blur_ksize": 0,
    })

    #rest
    for j in range(num_aug - 2):
        use_noise = rng.random() < aug_cfg["noise_p"]
        use_blur = rng.random() < aug_cfg["blur_p"]
        specs.append({
            "name": f"jit{j}",
            "flip": bool(rng.random() < 0.5),
            "scale": float(rng.uniform(aug_cfg["scale_min"], aug_cfg["scale_max"])) * pad,
            "dx": int(rng.integers(-aug_cfg["dx_max"], aug_cfg["dx_max"] + 1)),
            "dy": int(rng.integers(-aug_cfg["dy_max"], aug_cfg["dy_max"] + 1)),
            "alpha": float(rng.uniform(aug_cfg["alpha_min"], aug_cfg["alpha_max"])),
            "beta": float(rng.uniform(aug_cfg["beta_min"], aug_cfg["beta_max"])),
            "gamma": float(rng.uniform(aug_cfg["gamma_min"], aug_cfg["gamma_max"])),
            "noise_std": float(aug_cfg["noise_std"] if use_noise else 0.0),
            "blur_ksize": int(aug_cfg["blur_ksize"] if use_blur else 0),
        })
    return specs


def process_video(video_key, video_data, video_root, frame_w, frame_h,
                  num_aug, hog_desc, rng, aug_cfg):
    subject, action = parse_kth_filename(video_key)
    if subject is None or action is None:
        return
    label_idx = CLASS_TO_IDX[action]
    split = split_from_video_key(video_key)
    if split is None:
        return

    video_path = Path(video_root) / video_key
    if not video_path.exists():
        return

    groups = video_data.get("groups", [])
    if not groups:
        return

    needed_frames = set()
    for group in groups:
        for fd in group:
            needed_frames.add(int(fd["frame_idx"]))

    frames = load_video_frames(video_path, sorted(needed_frames), frame_w, frame_h)
    if not frames:
        return

    #augment only train videos
    #val/test -> original
    aug_specs = make_aug_specs(num_aug if split == "train" else 1, rng, aug_cfg)

    for ai, aug in enumerate(aug_specs):
        for gi, group in enumerate(groups):
            T = len(group)
            feats = np.zeros((T, HOG_FEAT_PER_FRAME), dtype=np.float32)
            boxes = np.zeros((T, 4), dtype=np.float32)
            ok = True

            for fi, frame_data in enumerate(group):
                idx = int(frame_data["frame_idx"])
                bbox = frame_data.get("selected_bbox") or (frame_data.get("bboxes") or [None])[0]
                if bbox is None or idx not in frames:
                    ok = False
                    break

                hog_vec, post_box = compute_hog_with_aug(
                    frames[idx], bbox, aug, hog_desc, frame_w, frame_h, rng
                )
                if hog_vec is None:
                    ok = False
                    break

                feats[fi] = hog_vec

                cx = (post_box["x"] + post_box["w"] / 2.0) / frame_w
                cy = (post_box["y"] + post_box["h"] / 2.0) / frame_h
                boxes[fi] = [cx, cy, post_box["w"] / frame_w, post_box["h"] / frame_h]

            if not ok:
                continue

            meta = {
                "video_key": video_key,
                "subject": subject,
                "action": action,
                "label_idx": label_idx,
                "group_idx": gi,
                "aug_idx": ai,
                "aug_name": aug["name"],
                "frame_indices": [int(fd["frame_idx"]) for fd in group],
                "split": split,
            }
            yield feats.reshape(-1), boxes, label_idx, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--video_root", type=str, default="/home/mmuntean/kth_organized")
    parser.add_argument("--num_aug", type=int, default=4)
    parser.add_argument("--aug_profile", type=str, default="mild",choices=["mild", "strong"])
    parser.add_argument("--bbox_padding", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_every", type=int, default=25)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    hog_desc = cv2.HOGDescriptor(
        _winSize=(HOG_WIN_W, HOG_WIN_H),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    aug_cfg = build_aug_cfg(args)


    print(f"Loading bbox JSON: {args.bbox_json}")
    with open(args.bbox_json, "r") as f:
        in_data = json.load(f)

    config = in_data.get("config", {})
    frame_w = config.get("frame_width", 160)
    frame_h = config.get("frame_height", 120)
    T = config.get("temporal_kernel", 7)

    print(f"Frame size: {frame_w}x{frame_h} | T={T} | num_aug per train: {args.num_aug}")
    print(f"Video root: {args.video_root}")
    print(f"Aug profile: {args.aug_profile}")

    videos = in_data.get("videos", {})
    n_videos = len(videos)

    feats_list = []
    boxes_list = []
    labels_list = []
    meta_list = []

    video_counts = {s: 0 for s in VALID_SPLITS}

    for vi, (video_key, video_data) in enumerate(videos.items()):
        n_made = 0
        for feats, boxes, label, meta in process_video(
            video_key, video_data, args.video_root,
            frame_w, frame_h, args.num_aug, hog_desc, rng, aug_cfg,
        ):
            feats_list.append(feats)
            boxes_list.append(boxes)
            labels_list.append(label)
            meta_list.append(meta)
            n_made += 1

        if n_made > 0:
            video_counts[meta_list[-1]["split"]] += 1

        if (vi + 1) % args.report_every == 0 or vi == n_videos - 1:
            print(f"  [{vi+1:4d}/{n_videos}] last={video_key} -> {n_made} variant(s) "
                  f"| total samples so far: {len(feats_list)}")

    if not feats_list:
        raise SystemExit("No samples produced.")

    features = np.stack(feats_list, axis=0)
    bboxes = np.stack(boxes_list, axis=0)
    labels = np.asarray(labels_list, dtype=np.int64)
    metadata = np.array(meta_list, dtype=object)

    split_counts = {s: sum(1 for m in meta_list if m["split"] == s) for s in VALID_SPLITS}

    print()
    print("Stats:")
    print(f"  total samples : {len(meta_list)}")
    for s in VALID_SPLITS:
        print(f"  {s:5s} samples : {split_counts[s]}  (from {video_counts[s]} videos)")
    print(f"  features shape: {features.shape}  ({features.nbytes / 1e6:.1f} MB)")
    print(f"  bboxes shape  : {bboxes.shape}")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"\nSaving to {out_path} ...")

    np.savez(
        out_path,
        features=features,
        bboxes=bboxes,
        labels=labels,
        metadata=metadata,
        config=np.array(
            {
                **config,
                "hog_precomputed": True,
                "augmented": True,
                "num_aug_per_train": args.num_aug,
                "aug_profile": args.aug_profile,
                "aug_cfg": aug_cfg,
                "source_json": os.path.basename(args.bbox_json),
                "seed": args.seed,
            },
            dtype=object,
        ),
    )
    print("Done")


if __name__ == "__main__":
    main()

