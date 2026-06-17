# 3D Convolutional Neural Networks for Human Action Recognition (HAR) on the KTH Dataset

https://cnn-har-munteanu-mihai.streamlit.app/

> This document completely describes the classic CNN pipeline implemented in the `cnn_har_app/` folder. Its purpose is to serve as the counterpart (CNN baseline) to the Convolutional Spiking Neural Network (CSNN) described in the previous chapters of the bachelor's thesis, thus allowing a direct comparison under equivalent conditions (same dataset, same temporal segmentation, same HOG descriptors, **same standard subject split, identical to the one used for the CSNN**).
>
> The pipeline is organized such that most of the computationally expensive components (person detection, HOG extraction, spatial and photometric augmentations) are precomputed *only once* and stored in an `.npz` file. The actual training of the CNN reads this tensor and focuses solely on learning spatiotemporal representations.
>
> **Current configuration of the paper** (aligned with the CSNN dataset): **T = 19 frames** per sample, **frame_gap = 2**, **num_groups = 10** groups/video, **train/val/test subject split (8/8/9)**, HOG descriptor window **64 × 128** (3780 features/frame). *(The file name `..._f10_g2` is just a label; the real value is frame_gap=2.)*

---

## 1. Table of Contents

1. [Pipeline Overview](#2-pipeline-overview)
2. [KTH Dataset and Standard Split](#3-kth-dataset-and-standard-split)
3. [Stage 1 — Bounding Boxes (`hog_person_data*.json`)](#4-stage-1--person-detection)
4. [Stage 2 — Precomputing Augmented HOG Features (`extract_hog_augmented.py`)](#5-stage-2--extract_hog_augmentedpy)
5. [Stage 3 — PyTorch Dataset and Additional Streams (`dataset.py`)](#6-stage-3--datasetpy)
6. [Stage 4 — Model Architecture (`model.py`)](#7-stage-4--modelpy)
7. [Stage 5 — Training (`train.py`)](#8-stage-5--trainpy)
8. [Stage 6 — Ensemble Evaluation (`eval_ensemble.py`)](#9-stage-6--eval_ensemblepy)
9. [Stage 7 — Streamlit Demonstrator (`app.py`)](#10-stage-7--apppy-streamlit)
10. [Experimental Results](#11-experimental-results)
11. [CNN ↔ CSNN Comparison](#12-cnn--csnn-comparison)
12. [Future Directions](#13-future-directions)
13. [Suggested Figures for the Paper](#14-suggested-figures-for-the-paper)

---

## 2. Pipeline Overview

The CNN pipeline strictly follows the same pre-processing logic as the CSNN variant, ensuring that the only source of difference between the two systems is the *classifier*. The essential difference appears only at the final step: CSNN uses spiking neurons with local plasticity (STDP), while the CNN uses dense 3D convolutions trained via backpropagation.

```text
       ┌──────────────────────┐
       │  KTH Video (.avi)    │
       └─────────┬────────────┘
                 │ (1) extract_bboxes_kth.py
                 ▼
       ┌──────────────────────┐
       │  bbox JSON           │   ← groups of T=19 frames
       │  (frame_idx + bbox)  │     with frame_gap = 2, 10 groups/video
       │  key split:          │     train/ , val/ , test/
       └─────────┬────────────┘
                 │ (2) extract_hog_augmented.py
                 ▼
       ┌──────────────────────┐
       │  hog_aug_*.npz       │   ← augmented HOG features
       │  (N, T·3780)=        │     + bboxes + labels + metadata
       │  (14715, 71820)      │
       └─────────┬────────────┘
                 │ (3) HOGDataset (dataset.py)
                 ▼
       ┌──────────────────────┐
       │  Tensor (T,C,H,W)    │   ← C = 36 HOG + 36 diff
       │  = (19, 80, 15, 7)   │       + bbox(4) + bbox_vel(4) = 80
       └─────────┬────────────┘
                 │ (4) HARConv3DNet (model.py)
                 ▼
       ┌──────────────────────┐
       │  Logits (6 classes)  │
       └─────────┬────────────┘
                 │ (5) AdamW + CosineLR + EMA + Mixup
                 │     best model selection on VAL (early stopping)
                 ▼
       ┌──────────────────────┐
       │  Ensemble of 5 seeds │
       │  + TTA (reverse, shf)│
       └──────────────────────┘
```

> **Suggested Figure #1**: The above schema drawn as a horizontal pipeline with icons (camera, JSON, tensor, network, evaluation).

---

## 3. KTH Dataset and Standard Split

The KTH dataset contains 6 actions × 25 subjects × 4 scenarios = **600 clips** (160×120 px resolution, ~25 fps, static background, grayscale).

| Index | Action         | Dominant Characteristic      |
|------:|----------------|------------------------------|
| 0     | boxing         | local motion in chest area   |
| 1     | handclapping   | rhythmic hands approaching   |
| 2     | handwaving     | up-down hands, high amplitude|
| 3     | jogging        | medium horizontal translation|
| 4     | running        | fast horizontal translation  |
| 5     | walking        | slow horizontal translation  |

**Train/val/test subject split (8/8/9).** Unlike the classic train/test protocol 1–16 / 17–25, a **three-way** split is used here, identical to the CSNN pipeline, so that:

- **train** is used to train the model,
- **val** is used to select the best model (early stopping) — a *held-out* set, never used for training,
- **test** provides the final generalization score, evaluated only once at the end.

| Split | Subjects                          | # Subjects | Videos with valid groups* |
|-------|-----------------------------------|:----------:|:-------------------------:|
| Train | 11, 12, 13, 14, 15, 16, 17, 18    | 8          | 154                       |
| Val   | 19, 20, 21, 23, 24, 25, 1, 4      | 8          | 161                       |
| Test  | 2, 3, 5, 6, 7, 8, 9, 10, 22       | 9          | 190                       |

> *The effective number of videos that yield at least one valid group of T=19 frames (person detection succeeds), from the current extraction (with `running` re-extracted, see 11.4). Total: 505 out of 600 clips.

> In the code, these sets are the ground truth `TRAIN_SUBJECTS` / `VAL_SUBJECTS` / `TEST_SUBJECTS` in `dataset.py`. However, the split is normally read **directly from the key prefix** in the JSON (`train/`, `val/`, `test/`) — `split_from_video_key()` — with subject classification serving only as a fallback for older JSONs without prefixes.

---

## 4. Stage 1 — Person Detection

The script `src/tool/extract_bboxes_kth.py` (also used by the CSNN pipeline, therefore kept unchanged here) processes each video and outputs a `hog_person_data_*.json` file. Person detection combines **HOG + MOG2** (background/motion) with temporal fusion and smoothing of the bounding box. JSON structure:

```json
{
  "config": {
    "temporal_kernel": 19,
    "num_groups": 10,
    "frame_gap": 2,
    "frame_width": 160,
    "frame_height": 120
  },
  "videos": {
    "train/boxing/person11_boxing_d1_uncomp.avi": {
      "groups": [
        [
          { "frame_idx": 50,
            "selected_bbox": {"x": 68, "y": 54, "w": 30, "h": 61, ...}
          },
          ...  // T = 19 frames
        ],
        ...  // up to num_groups groups per video
      ]
    }
  }
}
```

The split (`train`/`val`/`test`) is encoded as **the first path component** in the video key, written by the extractor based on the folder structure `kth_organized_tvt/{train,val,test}/<action>/<video>.avi`. This allows downstream steps to route samples without having to perform lookups on the subject ID.

The two key hyperparameters reused throughout the CNN pipeline are:

- **`temporal_kernel` = T** — the number of frames comprising a sample (in the current configuration **T = 19**);
- **`frame_gap` = g** — the distance between consecutive frames within the same group. The actual frames of a group are $\{f_0, f_0 + g, f_0 + 2g, \dots, f_0 + (T-1)g\}$.

The frame gap controls how much *temporal context* a sample captures. With **T = 19, g = 2**, a sample spans $(T-1)\cdot g = 36$ frames ≈ **1.4 s** at 25 fps — roughly one sub-cycle or one short motion cycle. From each video, up to **num_groups = 10** such windows are extracted (centered around high-confidence detections).

| T  | `frame_gap` | Span (frames) | Duration ≈ | Covers |
|---:|------------:|--------------:|----------:|--------|
| **19** | **2**   | **36**        | **1.4 s** | **Current config (aligned with CSNN), one motion sub-cycle** |
| 7  | 4           | 24            | 0.96 s    | (Old CNN config) almost one cycle |

> **Suggested Figure #2**: A sequence of T = 19 frames (frame_gap = 2) with the bounding box drawn in green.

---

## 5. Stage 2 — `extract_hog_augmented.py`

**This is the core pre-processing component and deserves detailed explanation**, as it produces the `.npz` file that dominates the total training time and decouples the "heavy" part (video decoding + spatial augmentations + HOG) from the "light" part (forward/backward through the CNN).

### 5.1. Motivation for Precomputation

In naive implementations, HOG features are recomputed at every epoch from the source videos. Over hundreds of epochs, this means hundreds of passes through the codec decoder + resizing + executions of `cv2.HOGDescriptor.compute()`, which dominates the total time. The adopted solution:

1. **A single pass** through each video to obtain all HOG features.
2. **Spatial and photometric augmentations** (flip, bbox jitter, brightness, gamma, blur, noise) are applied *before* HOG, so they are already incorporated into the feature vectors.
3. **Feature-level augmentations** (Gaussian noise, feature dropout, temporal shift/reverse) are applied *online* in `Dataset.__getitem__` — they are sufficiently cheap.

This design transforms a training experiment taking ~hours per seed into one taking ~minutes (once the `.npz` exists).

### 5.2. HOG (Histogram of Oriented Gradients) Descriptor

For each person crop, the HOG descriptor operates on a **64 × 128 px** window with Dalal–Triggs parameters:

| Parameter        | Value      | Comment                                 |
|------------------|-----------:|-----------------------------------------|
| `winSize`        | 64 × 128   | Fixed window, crop is resized           |
| `blockSize`      | 16 × 16    | Block composed of 2 × 2 cells           |
| `blockStride`    | 8 × 8      | 50% overlap                             |
| `cellSize`       | 8 × 8      | Elementary cell                         |
| `nbins`          | 9          | Histograms with 9 orientations [0°, 180°)|
| `nblocks_x`      | 7          | $(64-16)/8 + 1$                         |
| `nblocks_y`      | 15         | $(128-16)/8 + 1$                        |
| Features / block | 36         | $(16/8)^2 \times 9$                     |
| **Total / frame**| **3780**   | $7 \times 15 \times 36$                 |

Thus, a sample of **T = 19 frames produces a vector of 19 × 3780 = 71,820 HOG values**.

> **Design Note (Important).** The HOG window must remain **64 × 128**. An experimental variant on 32 × 64 (756 features/frame, 3 × 7 block grid) was tested but is **incompatible with `HARConv3DNet`**: the two `MaxPool3d(2,2,2)` layers collapse the grid width $W = 3 \to 1 \to 0$ (empty tensor → error). At 64 × 128, the grid is 15 × 7, and pooling works: $W = 7 \to 3 \to 1$. The constant `HOG_FEAT_PER_FRAME = 3780` must be identical in `extract_hog_augmented.py`, `dataset.py`, and implicitly in the geometry expected by `model.py`.

**Intuitive Mathematical Calculation for a Pixel.** The gradient at pixel $(x, y)$ has the components:

$$
G_x = I(x+1, y) - I(x-1, y), \qquad G_y = I(x, y+1) - I(x, y-1)
$$

with magnitude and orientation:

$$
\|G\| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \operatorname{atan2}(G_y, G_x) \bmod \pi
$$

Each pixel contributes with $\|G\|$ to the corresponding bin $\theta$ in its cell's histogram. The blocks (2 × 2 cells) are then L2 normalized (with clamping at 0.2 — *L2-Hys*) for robustness to illumination changes.

> **Suggested Figure #3**: A person crop (boxing) + gradient map + overlay of 8 × 8 cells + HOG vector visualized as a "hedgehog" over the image.

### 5.3. Augmentation Policy

Augmentations are applied **only to the `train` split**. On `val` and `test`, each sample produces *a single* original variant (no jitter, no flip, no modified brightness), exactly as in the CSNN pipeline.

For each group of T frames (sample), `num_aug` variants are generated:

1. **`orig` variant** (always present): no perturbations, just crop + resize 64 × 128 + HOG.
2. **`flip` variant** (always present for `num_aug ≥ 2`): horizontal flip.
3. **`num_aug − 2` random `jit{i}` variants**: a random combination of geometric jitter + photometric perturbations.

For a sample, the random augmentation applied to a bbox $(x, y, w, h)$ generates a bbox $(x', y', w', h')$ as follows:

$$
\begin{aligned}
c_x &= x + w/2, & c_y &= y + h/2 \\
w' &= \max(1,\ \lfloor w \cdot s \cdot p \rfloor), & h' &= \max(1,\ \lfloor h \cdot s \cdot p \rfloor) \\
x' &= \lfloor c_x - w'/2 + \delta_x \rfloor, & y' &= \lfloor c_y - h'/2 + \delta_y \rfloor
\end{aligned}
$$

with $s$ — random scaling, $p$ — `bbox_padding` (universal padding factor, default 1.25), $\delta_x, \delta_y$ — horizontal/vertical translations. Then, on the cropped and resized image, the following are applied:

- **Horizontal flip** with probability 0.5;
- **Brightness/contrast**: $I' = \operatorname{clip}(\alpha \cdot I + \beta,\ 0,\ 255)$;
- **Gamma** (LUT on [0, 255]): $I' = 255 \cdot (I/255)^{1/\gamma}$;
- **Gaussian blur** (kernel 3, with probability `blur_p`);
- **Gaussian noise** after grayscale conversion: $I' = \operatorname{clip}(I + \mathcal{N}(0, \sigma^2),\ 0,\ 255)$.

#### `mild` vs `strong` Profiles

| Parameter     | `mild`             | `strong`            |
|---------------|--------------------|---------------------|
| scale         | [0.92, 1.08]       | [0.88, 1.15]        |
| $\delta_x$    | [−5, +5] px        | [−8, +8] px         |
| $\delta_y$    | [−3, +3] px        | [−6, +6] px         |
| $\alpha$ (contrast) | [0.85, 1.15] | [0.75, 1.25]        |
| $\beta$ (brightness) | [−15, +15]  | [−25, +25]          |
| $\gamma$      | [0.95, 1.05]       | [0.85, 1.15]        |
| noise $\sigma$ | 2.0 (p = 0.5)     | 4.0 (p = 0.7)       |
| blur ksize    | 3 (p = 0.15)       | 3 (p = 0.25)        |

In the final experiments, `--aug_profile strong --num_aug 8` was used, which means **8 variants per training video** (1 original + 1 flip + 6 randomly perturbed). On val/test, `num_aug` is forced to 1.

#### Coherence Between the HOG Stream and the BBox Stream

A subtle but important detail: when the variant includes a `flip`, the bounding box stored in the output **is also mirrored** (see `extract_hog_augmented.py`, `compute_hog_with_aug`):

```python
if aug["flip"]:
    post_box["x"] = frame_w - box["x"] - box["w"]
```

Thus, when the model receives in parallel the HOG features (from the mirrored image) and the bboxes (geometric description), they remain **conceptually consistent** — the bounding box indicates where the person is located in the image that HOG actually processed.

### 5.4. Resulting `.npz` File Structure

```
features  : (N, T*3780) float32       — HOG vectors concatenated over T
bboxes    : (N, T, 4)   float32       — (cx, cy, w, h) normalized ∈ [0, 1]
labels    : (N,)        int64         — index in [0, 5]
metadata  : (N,)        object        — list of Python dicts
config    : (1,)        object        — dict with config + augmentation parameters
```

For the current dataset (T = 19, g = 2, num_groups = 10, strong profile, num_aug = 8):

```
total samples : 14715
  train : 11376  (from 154 videos, augmented ×8 → 1422 raw groups × 8)
  val   :  1529  (from 161 videos, no augmentation)
  test  :  1810  (from 190 videos, no augmentation)
features shape : (14715, 71820)   ≈ 4.23 GB float32
bboxes shape   : (14715, 19, 4)
```

The `metadata[i]` field contains:

```python
{
  "video_key": "train/boxing/person11_boxing_d1_uncomp.avi",
  "subject": 11, "action": "boxing", "label_idx": 0,
  "group_idx": 3, "aug_idx": 5, "aug_name": "jit2",
  "frame_indices": [50, 52, 54, ..., 86],   # 19 indices, step g=2
  "split": "train"
}
```

The `split` field is crucial: it is precomputed (from the path prefix) and allows `HOGDataset` to quickly filter samples without recomputing anything.

#### Binary Layout of the `.npz` File and Byte Order

To understand why loading is so fast compared to video re-decoding, it is useful to know exactly what `np.savez` produces on disk.

An `.npz` is actually **a ZIP archive** containing multiple `.npy` files, one for each saved array:

```
hog_aug_tvt_19_f10_g2_runfix.npz  =  ZIP container
   ├── features.npy   (N × T × 3780 × 4 bytes = float32)
   ├── bboxes.npy     (N × T × 4 × 4 bytes = float32)
   ├── labels.npy     (N × 8 bytes = int64)
   ├── metadata.npy   (object array — Python pickle)
   └── config.npy     (object array — Python pickle)
```

Numeric files (features, bboxes, labels) are saved in the native NumPy binary `.npy` format:

```
\x93NUMPY\x01\x00<hl><header_dict><raw_bytes>
   ^magic    ^ver  ^len  ^dict     ^values sequentially
```

The header is a serialized Python dict as a string:

```python
{'descr': '<f4', 'fortran_order': False, 'shape': (14715, 71820)}
#         ^ float32 little-endian      ^ N=14715 samples × 71820 features
```

This is followed by `N × (T · F) × sizeof(float32)` consecutive bytes. For `features`:

$$
N_{\text{bytes}} = 14\,715 \times 19 \times 3780 \times 4 = 4\,227\,310\,800 \approx 4.23\ \text{GB}
$$

NumPy uses **C-order (row-major)** layout. For a conceptual array of shape `(N, T, F)`, the linear index of element `[i, t, f]` is:

$$
\text{offset}(i, t, f) = i \cdot (T \cdot F) + t \cdot F + f
$$

and the concrete byte address (4 bytes per float32):

$$
\text{addr}_{\text{byte}} = \text{header\_size} + 4 \cdot \text{offset}(i, t, f)
$$

This means that **the complete sample `i` (all T = 19 frames × all 3780 features) occupies a continuous block** of bytes. Reading a sample = one `memcpy` from the flat vector to the PyTorch tensor.

`np.load(path, mmap_mode='r')` maps the file into the address space without copying it to RAM. Upon accessing `features[42]`, the operating system fetches the necessary pages on-demand from the disk. For our workload (data that fits in RAM), the real advantage is *avoiding full preloading* at process start — the most accessed pages end up in the kernel's page cache.

### 5.5. Usage Examples

```bash
cd cnn_har_app

# Augmented HOG Precomputation on the FINAL JSON (running fixed), T=19, g=2, 10 groups, strong profile, 8 augmentations:
python3 extract_hog_augmented.py \
    --bbox_json ../hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --output   ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --num_aug 8 --aug_profile strong \
    --video_root /home/mmuntean/kth_organized_tvt
```

At the end, the script displays statistics by split; verify `features shape: (14715, 71820)` — if you see 71820 (= 19 × 3780), the geometry is correct.

> **Suggested Figure #4**: For the same sample (same video, same group), 4–6 augmented variants placed side-by-side as thumbnails (orig, flip, jit1 with low brightness, jit2 with zoom in + negative dy, etc.).

---

## 6. Stage 3 — `dataset.py`

`HOGDataset` is a `torch.utils.data.Dataset` subclass that encapsulates loading, reorganization, and online augmentation. It automatically determines the input format based on the file extension:

- **`.json`** — bbox-only; HOG is recomputed at *runtime* (used only for debugging);
- **`.npz`** — the file produced by `extract_hog_augmented.py` (the standard path for production).

Filtering by split is done via the `meta["split"]` field of each sample (fallback to subject only for old NPZs without tags).

### 6.1. HOG Reorganization → 4D Tensor

The flat vector of 3780 features per frame is reorganized as a 3D tensor before being fed into the 3D convolutions. OpenCV's convention for block ordering is row-major over $(n_y, n_x, 36)$, so the reshape + permute operation is:

$$
\underbrace{(T \cdot 3780)}_{\text{flat}}
\quad \xrightarrow{\text{view}} \quad
(T, n_y{=}15, n_x{=}7, 36)
\quad \xrightarrow{\text{permute}} \quad
\underbrace{(T, C{=}36, H{=}15, W{=}7)}_{\text{4D Tensor}}
$$

C corresponds to the 36 features per block (thus 36 "semantic" HOG channels), and H × W = 15 × 7 represents the spatial grid of blocks. With T = 19, the resulting tensor (before auxiliary streams) is $(19, 36, 15, 7)$.

#### Concrete Sample Access in `HOGDataset.__getitem__`

Here is what the DataLoader does when it requests sample `i` (T = 19):

```python
def __getitem__(self, i):
    # 1. View into data (no copy):
    feat_flat = self.features[i]          # shape (T*3780,)
    bb        = self.bboxes[i]            # shape (T, 4)
    label     = self.labels[i]            # int64

    # 2. Reshape + permute to (T, C=36, H=15, W=7):
    feat = feat_flat.reshape(T, 15, 7, 36).transpose(0, 3, 1, 2)

    # 3. Add diff (motion):
    diff = np.diff(feat, axis=0, prepend=feat[:1])     # (T, 36, 15, 7)

    # 4. Broadcast bbox over the spatial grid:
    bbox_grid = np.broadcast_to(bb[:, :, None, None], (T, 4, 15, 7))
    bbox_vel  = np.diff(bb, axis=0, prepend=bb[:1])
    bbox_vel  = np.broadcast_to(bbox_vel[:, :, None, None], (T, 4, 15, 7))

    # 5. Concatenate channels:
    x = np.concatenate([feat, diff, bbox_grid, bbox_vel], axis=1)  # (T, 80, 15, 7)

    # 6. Online augmentations (only train) ...
    # 7. Convert to tensor.
    return torch.from_numpy(x).float(), label
```

Cost of each step for a sample (T = 19):

| Step | Touched Memory | Approx. Cost |
|---|---|---|
| view in data | 0 (lazy) | ~µs |
| reshape + transpose | 19 × 3780 × 4 ≈ 287 KB | ~25 µs |
| temporal diff | 287 KB | ~25 µs |
| broadcast bbox | 0 (broadcast, no allocation) | ~µs |
| concatenate 80 channels | 19 × 80 × 15 × 7 × 4 ≈ 638 KB | ~50 µs |
| augmentations | 638 KB | ~60 µs |
| → torch | 638 KB | ~40 µs |

Overall: ~200 µs/sample. For a batch of 64, ~13 ms — comparable to or less than the cost of a forward+backward pass on the GPU.

### 6.2. Additional Input Streams

Besides the raw HOG stream, the dataset can concatenate 3 auxiliary streams along the channel axis. The final model uses all 3, resulting in **80 channels per frame**:

| Stream         | Channels | Definition                                                 | Physical Capture                        |
|----------------|---------:|------------------------------------------------------------|-----------------------------------------|
| HOG            | 36       | $H_t$                                                      | spatial appearance of the person at $t$ |
| Diff (motion)  | 36       | $\Delta H_t = H_t - H_{t-1}$, with $\Delta H_0 = 0$         | temporal derivative of appearance       |
| BBox metadata  | 4        | $(c_x, c_y, w, h)$ broadcast over the $H \times W$ grid     | absolute position in frame (translation)|
| BBox velocity  | 4        | $(\dot c_x, \dot c_y, \dot w, \dot h)$ broadcast over grid | velocity of translation                 |

**Why bbox streams are important.** After crop + resize to 64 × 128, *all* subjects look as if they have the same size and position inside the HOG window. Thus, the model loses exactly the information that differentiates walking from jogging and running: the global translation speed of the person on the screen. By inserting the bbox as 4 broadcasted channels over the entire $H \times W$ grid, the model regains this signal **without needing to indirectly recover translation from small cropping artifacts**.

> **Suggested Figure #5**: Schematic of the channels vertically — 36 HOG + 36 diff + 4 bbox + 4 bbox_vel = 80 (with differently colored blocks).

### 6.3. Online Augmentations on Train (in `__getitem__`)

Applied only if `split == "train"` and `augment=True`:

1. **Temporal reverse** with probability $p_{\text{rev}} = 0.3$: reverses the order of the T frames (both HOG and bbox).
2. **Temporal shift** uniformly in $[-\Delta, +\Delta]$ with $\Delta = 2$ frames, zero-padded.
3. **Gaussian noise** on the HOG vector with $\sigma = 0.003$.
4. **Feature dropout** Bernoulli with $p = 0.015$ on each element of the HOG vector.

Combined with the `.npz` augmentations (offline), the model practically never sees the exact same sample twice.

---

## 7. Stage 4 — `model.py`

The `model.py` file defines a single architecture, built via `build_model(...)`:

| Class             | Input                  | Comment                                     |
|-------------------|------------------------|---------------------------------------------|
| **`HARConv3DNet`**| $(B,\ T,\ C,\ H,\ W)$  | **Main model**; dense 3D convolutions       |

**The production model is `HARConv3DNet`** (all results reported in Section 11 are obtained with this architecture).

### 7.1. `HARConv3DNet` — Spatiotemporal Convolutions

Conceptually, this network is analogous to **HOG3D** (Kläser et al., 2008), the HOG descriptor extended in space-time, which represents the strongest HOG-based baseline on KTH (~91%). Unlike HOG3D, however, the 3D kernels *are learned* from the data along with the classifier.

The input tensor $(B, T, C, H, W)$ is permuted internally to $(B, C, T, H, W)$ to respect the PyTorch `Conv3d` convention (channel-first).

#### Full Architecture (Exactly as in Code)

```
Block 1 — local spatiotemporal learning:
  Conv3d(80 → 96,  k=3×3×3, pad=1) + BN + ReLU
  Conv3d(96 → 128, k=3×3×3, pad=1) + BN + ReLU
  MaxPool3d(2×2×2)                              # (T,H,W): 19→9, 15→7, 7→3
  Dropout3d(0.25)

Block 2 — deepening + spatiotemporal pooling:
  Conv3d(128 → 192, k=3×3×3, pad=1) + BN + ReLU
  Conv3d(192 → 256, k=3×3×3, pad=1) + BN + ReLU
  MaxPool3d(2×2×2)                              # (T,H,W): 9→4, 7→3, 3→1
  Dropout3d(0.25)

Block 3 — collapse to global descriptor:
  Conv3d(256 → 256, k=3×3×3, pad=1) + BN + ReLU
  AdaptiveAvgPool3d(1)                          # (1,1,1)

Classifier (head):
  Flatten → Dropout(p=dropout)
  Linear(256 → 64) + ReLU
  Dropout(p=dropout/2)
  Linear(64 → 6)                                # 6 KTH classes
```

#### Design Decisions and Why They Matter

- **3 × 3 × 3 Kernel on all blocks** — a uniform cubic kernel, symmetric across all 3 dimensions (time, height, width). This allows the network to learn local spatiotemporal patterns (e.g., a horizontal edge moving vertically over 3 frames = hand rising in handwaving).
- **Two `MaxPool3d(2,2,2)`** — each halves all 3 dimensions. On the T = 19, 15 × 7 grid configuration, the path is $19{\times}15{\times}7 \to 9{\times}7{\times}3 \to 4{\times}3{\times}1$. Important: the grid width (W = 7) is exactly sufficient not to vanish ($7 \to 3 \to 1$); hence the constraint of the HOG window to 64 × 128 (see note in 5.2).
- **`BatchNorm3d` + ReLU** after each convolution — stabilizes batch training (B = 64), important for reproducibility across multiple seeds.
- **`Dropout3d` (channel-wise)** between blocks instead of standard Dropout — randomizes entire feature maps directly, a more aggressive regularizer for convolutions.
- **`AdaptiveAvgPool3d(1)`** — removes any dependence on input size and produces a global 256-d descriptor.
- **Thin classifier head** (256 → 64 → 6) — the remaining capacity lies in the extractor; the small head + dropout prevents overfitting on only 8 training subjects.

#### Model Capacity

| Component              | Parameters |
|------------------------|-----------:|
| Block 1 (Conv3d × 2)   | ~340 K     |
| Block 2 (Conv3d × 2)   | ~1.99 M    |
| Block 3 (Conv3d × 1)   | ~1.77 M    |
| Classifier             | ~17 K      |
| **Total** *(approx., with C = 80)* | **~4.1 M** |

> The exact number of parameters is printed at the beginning of training (`Trainable params: X.XXM`).

> **Suggested Figure #6**: Architecture diagram with tensor volumes annotated on the edges (input 80×19×15×7 → … → 6 logits).

### 7.2. Detailed Forward Pass with Shape Tracking

Let's trace a batch of 64 through the network, dimension by dimension. Input: `(B=64, T=19, C=80, H=15, W=7)`.

**Step 0 — Permute for PyTorch `Conv3d` convention**

`Conv3d` expects `(B, C, D, H, W)` (channel-first, then depth/time). We convert:

$$
(B, T, C, H, W) \xrightarrow{\text{permute}(0, 2, 1, 3, 4)} (B, C, T, H, W) = (64, 80, 19, 15, 7)
$$

**Block 1 — local spatiotemporal learning**

```text
(64, 80, 19, 15, 7)
   │ Conv3d(in=80, out=96, k=3×3×3, padding=1)
   ▼
(64, 96, 19, 15, 7)           ← padding=1 preserves T, H, W
   │ BatchNorm3d + ReLU
   │ Conv3d(96 → 128, k=3×3×3, padding=1) + BN + ReLU
   ▼
(64, 128, 19, 15, 7)
   │ MaxPool3d(kernel=(2, 2, 2))   ← T, H, W all ÷ 2 (with floor)
   ▼
(64, 128, 9, 7, 3)
   │ Dropout3d(0.25)
   ▼
(64, 128, 9, 7, 3)
```

**Block 2 — deepening + spatiotemporal pooling**

```text
(64, 128, 9, 7, 3)
   │ Conv3d(128 → 192) + BN + ReLU
   │ Conv3d(192 → 256) + BN + ReLU
   ▼
(64, 256, 9, 7, 3)
   │ MaxPool3d(kernel=(2, 2, 2))    ← T: 9→4, H: 7→3, W: 3→1
   ▼
(64, 256, 4, 3, 1)
   │ Dropout3d(0.25)
   ▼
(64, 256, 4, 3, 1)
```

**Block 3 — global collapse**

```text
(64, 256, 4, 3, 1)
   │ Conv3d(256 → 256) + BN + ReLU
   │ AdaptiveAvgPool3d(output=(1,1,1))
   ▼
(64, 256, 1, 1, 1)
   │ Flatten
   ▼
(64, 256)
```

**Classifier Head**

```text
(64, 256) → Dropout(0.2) → Linear(256→64) + ReLU → Dropout(0.1) → Linear(64→6) → (64, 6)
```

Pooling rule used: $D_{\text{out}} = \lfloor (D_{\text{in}} - k)/s + 1 \rfloor$ with $k = s = 2$. Verification: $19 \to \lfloor 9.5 \rfloor = 9 \to \lfloor 4.5 \rfloor = 4$; $15 \to 7 \to 3$; $7 \to 3 \to 1$.

#### Math of the Conv3d Operation in a Point

For a single output `(b, c', t, h, w)` of the first convolution (k=3, pad=1):

$$
y[b, c', t, h, w] = \beta_{c'} + \sum_{c=0}^{C_{\text{in}}-1} \sum_{dt=-1}^{+1} \sum_{dh=-1}^{+1} \sum_{dw=-1}^{+1} W[c', c, dt, dh, dw] \cdot x[b, c, t+dt, h+dh, w+dw]
$$

with zero-padding on the edges. The sum runs over $C_{\text{in}} \times 27$ products for each output → for Block 1 conv1 with $C_{\text{in}}=80$: $80 \times 27 = 2160$ products × $96 \times 19 \times 15 \times 7 = 191,520$ outputs ≈ **414 M MACs** per batch element.

#### BatchNorm 3D

For each channel $c'$ separately, normalizes over `(B, T, H, W)`:

$$
\mu_{c'} = \frac{1}{B T H W} \sum_{b,t,h,w} y[b, c', t, h, w], \qquad
\sigma_{c'}^2 = \frac{1}{B T H W} \sum_{b,t,h,w} (y - \mu_{c'})^2
$$

$$
\hat y[b, c', t, h, w] = \gamma_{c'} \cdot \frac{y[b, c', t, h, w] - \mu_{c'}}{\sqrt{\sigma_{c'}^2 + \epsilon}} + \beta_{c'}
$$

with $\gamma, \beta$ being learnable per-channel weights.

### 7.3. Loss, Gradients, and Weight Updates

#### Cross-Entropy with Label Smoothing

For a sample with label $y \in \{0, \dots, 5\}$ and logits $z \in \mathbb{R}^6$:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad q_k = \begin{cases} 1 - \varepsilon & k = y \\ \varepsilon / (K-1) & k \neq y \end{cases}
$$

$$
\mathcal{L} = -\sum_{k=0}^{K-1} q_k \log p_k
$$

With $\varepsilon = 0.02, K = 6$: $q_y = 0.98$, the rest being $0.004$ each.

#### Gradient at the Logits Layer

For cross-entropy + softmax, the gradient has a very simple closed form:

$$
\frac{\partial \mathcal{L}}{\partial z_k} = p_k - q_k
$$

This gradient is backpropagated via the chain rule. For a 3D convolution with weights $W$:

$$
\frac{\partial \mathcal{L}}{\partial W[c', c, dt, dh, dw]} = \sum_{b, t, h, w} \frac{\partial \mathcal{L}}{\partial y[b, c', t, h, w]} \cdot x[b, c, t+dt, h+dh, w+dw]
$$

— a correlation between the output's gradient and the input.

#### AdamW Update

For each parameter $\theta$, at step $t$ (with $g_t = \partial \mathcal{L} / \partial \theta$):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
$$

with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$, $\lambda=3 \times 10^{-4}$. The difference compared to classic Adam: weight decay $\lambda \theta_{t-1}$ is applied directly to the parameters, not through the gradient (Loshchilov & Hutter, 2019).

#### Cosine Annealing

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\tfrac{t}{T_{\max}} \pi\right)\right)
$$

with $\eta_{\max}=10^{-3}$, $\eta_{\min}\approx 0$, $T_{\max}=200$ epochs.

---

## 8. Stage 5 — `train.py`

### 8.1. Training Loop

```text
For each epoch:
  For each batch:
    1. Input Mixup (α = 0.02 → almost identity, weak regularizer)
    2. Forward + loss (cross-entropy with label_smoothing = 0.02)
    3. Backward + grad-clipping (max-norm = 1.0)
    4. Optimizer step (AdamW)
    5. Update EMA shadow weights (if epoch ≥ ema_start)
  Schedule LR (CosineAnnealingLR)
  Evaluate on VAL (using EMA weights, if active)
  Save checkpoint if VAL accuracy increases   ← model selection
At the end:
  Reload best checkpoint and report accuracy on TEST (only once)
```

> **Model selection is performed on `val`, not on `test`.** Val is a held-out set, meaning early stopping never "sees" the test set. If the dataset lacks a val split (old NPZs), `train.py` falls back to test as a surrogate and warns loudly — but the current dataset has val (1529 samples), so the protocol is clean. The reported generalization figure is the **test accuracy** achieved by the model chosen on val.

### 8.2. Regularization Techniques Used

| Technique                | Value            | Role                                                         |
|--------------------------|------------------|--------------------------------------------------------------|
| **AdamW**                | lr = 1e-3, wd = 3e-4 | Standard optimizer with decoupled weight decay               |
| **CosineAnnealingLR**    | T_max = epochs   | LR smoothly decreases to ~0 by the end of training           |
| **Mixup**                | α = 0.02         | Very weak — linear combination of two samples                |
| **Label smoothing**      | ε = 0.02         | Targets become $1 - \varepsilon$ for the correct class, $\varepsilon/5$ for the rest |
| **Gradient clipping**    | max_norm = 1.0   | Stability against rare, large-magnitude gradients            |
| **EMA (Exp. Moving Avg.)**| decay = 0.999    | Evaluation weights = exponential average of training weights |
| **Early stopping**       | patience = 60    | Stops if **val** accuracy doesn't improve for 60 epochs      |
| **Dropout (head)**       | 0.2              | Classic regularizer in the classifier                        |

#### Mixup Detail

For two samples $(x_i, y_i)$ and $(x_j, y_j)$ and $\lambda \sim \operatorname{Beta}(\alpha, \alpha)$:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \qquad
\mathcal{L} = \lambda \cdot \mathrm{CE}(\hat y, y_i) + (1 - \lambda) \cdot \mathrm{CE}(\hat y, y_j)
$$

With $\alpha = 0.02$, the Beta distribution is strongly concentrated at 0 and 1, so in practice, mixup gently perturbs only a small fraction of the batches.

#### EMA — Why It Matters

The weights actually used during evaluation are:

$$
\theta_{\text{EMA}}^{(t)} = \rho \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \rho) \cdot \theta^{(t)}, \quad \rho = 0.999
$$

EMA yields a "smoother" model in parameter space, avoiding mini-oscillations at the end of training. In experiments, it typically contributes ~0.3–0.6% absolute accuracy over un-averaged final weights.

### 8.3. Configuration Used for Reported Results

Hardware: **larochette** node (Grid'5000), **AMD Instinct MI210** GPU (gfx90a), PyTorch on **ROCm 6.3** stack (`torch 2.8.0+rocm6.3`). The node has 4 GPUs, so the 5 seeds can run 4-in-parallel (one per GPU via `HIP_VISIBLE_DEVICES`).

```bash
cd cnn_har_app
for s in 42 123 7 13 99; do
  python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --balanced_sampler none \
    --seed $s \
    --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 \
    --temporal_shift_max 2 \
    --ema_decay 0.999 \
    --ema_start 5 \
    2>&1 | tee ../data/log_cnn_tvt19fix_s$s.txt
done
```

4-GPU parallel variant (4 seeds simultaneously, 5th afterwards):

```bash
for i in 0 1 2 3; do
  seeds=(42 123 7 13); s=${seeds[$i]}
  HIP_VISIBLE_DEVICES=$i python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --balanced_sampler none --seed $s --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 --temporal_shift_max 2 \
    > ../data/log_cnn_tvt19fix_s$s.txt 2>&1 &
done
wait
```

> The first line of each run should be `Using device: cuda` (on ROCm, the AMD GPU is also exposed through the "cuda" API). If it says `cpu`, the torch build is incorrect (NVIDIA `+cu128` instead of `+rocm6.3`).

> **Suggested Figure #7**: Train/val loss and train/val accuracy curves over the course of training for a representative seed, marking the chosen epoch (best val).

---

## 9. Stage 6 — `eval_ensemble.py`

### 9.1. Ensemble Over Seeds

We train 5 identical models with seeds $\{42, 123, 7, 13, 99\}$ and combine their predictions at inference on **test**. Two aggregation modes:

| Mode       | Formula                                              | Comment                            |
|------------|------------------------------------------------------|------------------------------------|
| `logits`   | $\bar z = \frac{1}{M} \sum_m z_m$, then $\operatorname{argmax} \bar z$ | **Recommended**; preserves raw scale |
| `softmax`  | $\bar p = \frac{1}{M} \sum_m \operatorname{softmax}(z_m)$, then $\operatorname{argmax} \bar p$ | Compresses very confident scores   |

Mean-logits is the one used for the results because in practice it's more stable when one of the models is overly confident but wrong — softmax would artificially amplify its voice in the vote.

> **Pay attention to the CSNN comparison.** The 5-model ensemble provides **a single** score (usually higher than any individual seed) and DOES NOT compare 1:1 with a single-model CSNN. For a fair comparison vs. CSNN, use the **mean ± standard deviation of the 5 individual test accuracies** (which measures typical performance + variability). The ensemble + TTA is reported separately, as an upper limit for the method.

### 9.2. Test-Time Augmentation (TTA)

In addition to averaging over seeds, we also aggregate predictions on transformed versions of the same sample:

- **`--tta_reverse`**: inference on the clip with temporally reversed frames. Useful because periodic actions (handwaving, handclapping, walking) are roughly invariant to reversal.
- **`--tta_shift 1`**: inference on the clip shifted by ±1 frame (zero-padded). Increases robustness to the exact synchronization of the action start.

The total number of evaluations per sample becomes:

$$
N_{\text{eval}} = M \cdot \big(1 + \mathbb{1}[\text{tta\_reverse}] + 2 \cdot \mathbb{1}[\text{tta\_shift}>0]\big)
$$

For $M = 5$, `--tta_reverse --tta_shift 1`, this means **5 × (1 + 1 + 2) = 20 evaluations per test clip**, and the final logit is the average of these 20.

### 9.3. Used Command

```bash
python3 eval_ensemble.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --checkpoints "models/har_conv3d_tvt19fix_s*.pth" \
    --ensemble_mode logits \
    --tta_reverse \
    --tta_shift 1
```

---

## 10. Stage 7 — `app.py` (Streamlit)

### 10.1. Purpose

`app.py` is an interactive demonstrator intended to be run during the thesis presentation. It allows:

1. Selecting a subset of checkpoints (configurable ensemble).
2. Choosing a clip from the test split, filtered by subject/action.
3. Running real inference (not mocked) aggregating the selected models.
4. Displaying the probability distribution across all 6 classes.
5. Visualizing the T = 19 source frames with the bounding box drawn on top, to visually demonstrate that the model "looks" exactly at the person.

> **Demo configured on the final dataset/models**: in `app.py`, `DATA_PATH = hog_aug_tvt_19_f10_g2_runfix.npz`, `ckpt_pattern = har_conv3d_tvt19fix_s*.pth`, and `VIDEO_ROOT` points to the KTH root reorganized as tvt (with `test/<action>/`). On Windows, paths are absolute (see the top of the file).

### 10.2. User Flow

```text
Sidebar:
  - Checkpoint selection (multiselect) → loads models
  - Ensemble mode selection (logits / softmax)
  - Info panel: architecture, # parameters, device

Main:
  Step 1: Choose subject → action → clip
  Step 2: "Classify" button → runs inference
  Step 3: Displays:
    - Prediction vs ground truth (✅ / ❌)
    - Bar chart with per-class probabilities
    - Source frames with bounding boxes in green
```

### 10.3. Caching for Responsiveness

Streamlit re-runs the script at every interaction. To avoid reloading the model and the dataset on every click, we use:

- `@st.cache_resource` for `HOGDataset` and PyTorch models ("heavy" resources);
- `@st.cache_data` for fetching frames from the video (cache key is the tuple `(video_path, frame_indices)`).

### 10.4. Launching

```bash
# Local:
streamlit run app.py

# On a remote node, exposed externally (with port 8501 open):
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

> **Suggested Figure #8**: Screenshot of the Streamlit UI (sidebar + thumbnails with green bounding boxes + probability bar chart).

---

## 11. Experimental Results

> **Final results (2026-06-09)** — training on larochette (MI210, ROCm). The bottleneck with the `running` class (see history in 11.4) was **fixed at the source** by re-extracting the running class with permissive detection thresholds + higher temporal carry tolerance (`--max_carry 4`). The dataset used is `hog_aug_tvt_19_f10_g2_runfix.npz` (JSON `hog_person_data_tvt_19_f10_g2_runfix.json`).

Configuration for all results below:

- **Architecture**: `HARConv3DNet` (4.32M parameters)
- **Dataset**: T = 19, frame_gap = 2, num_groups = 10, HOG window 64 × 128, tvt split (8/8/9 subjects), **running re-extracted** (max_carry 4)
- **80 channels per frame**: HOG (36) + diff (36) + bbox (4) + bbox_vel (4)
- **5 seeds**: 42, 123, 7, 13, 99
- **Model selection**: best on **val**; score reported on **test**
- **Offline augmentation**: `strong` profile, `num_aug = 8` (train only)
- **Online augmentation**: `temporal_reverse_p = 0.3`, `temporal_shift_max = 2`, Gaussian noise, feature dropout
- **No sampler** (`--balanced_sampler none`) — see 11.4 why oversampling was counterproductive
- **Ensemble**: mean logits + TTA reverse + TTA shift ±1

### 11.1. Per-Seed Accuracy (Test)

| Seed | Best val | **Test** Accuracy | running recall |
|-----:|:--------:|:-----------------:|:--------------:|
| 42   | 92.74 %  | **90.28 %**       | 77.6 %         |
| 123  | 92.15 %  | 89.78 %           | 78.1 %         |
| 7    | 93.00 %  | 88.73 %           | 73.8 %         |
| 13   | 92.41 %  | 88.90 %           | 79.5 %         |
| 99   | 94.11 %  | 91.33 %           | 69.5 %         |
| **Mean ± std (test)** | | **89.80 % ± 1.06** | **~75.7 %** |

### 11.2. Ensemble + TTA

| Configuration | Test Accuracy |
|---|:---:|
| Ensemble 5 seeds + TTA (reverse + shift ±1), mean logits | **90.55 %** |
| Δ ensemble vs. best individual seed (91.33 %) | **−0.78 %** |

### 11.3. Confusion Matrix (Test, Ensemble)

|              | boxing | handclap | handwav | jogging | running | walking | Recall |
|--------------|-------:|---------:|--------:|--------:|--------:|--------:|-------:|
| **boxing**       | 300 | 0   | 0   | 0   | 0   | 8   | 97.4 % |
| **handclapping** | 4   | 328 | 18  | 0   | 0   | 0   | 93.7 % |
| **handwaving**   | 0   | 53  | 277 | 0   | 0   | 0   | 83.9 % |
| **jogging**      | 6   | 0   | 0   | 244 | 4   | 0   | 96.1 % |
| **running**      | 4   | 0   | 0   | 50  | 153 | 3   | **72.9 %** |
| **walking**      | 7   | 0   | 0   | 14  | 0   | 337 | 94.1 % |

### 11.4. Error Interpretation

Two sources of errors, very different in nature:

1. **`running` → `jogging` (50 out of 210) — recall 72.9 %, raised from 12.9 % via a data fix.** Initially, out of 32 running videos in `train/running/`, only **4** produced valid groups during extraction (`extract_bboxes_kth.py` keeps a group only if all 19 frames have a good bbox; with fast motion, the person leaves the frame and detection fails) → 21 unique running groups vs 1648–2480 for the other classes, so the model was barely learning the class (recall 12.9 %).
   - **Fix applied (at source, not via sampler):** re-extraction *only* of the running class with (a) relaxed detection thresholds (`hit_threshold -1.2`, `min_bbox_area_ratio 0.004`, aspect 0.15–2.0, `mog2_min_area 300`) and (b) temporal carry tolerance `--max_carry 4` (carries over previous bbox for up to 4 consecutive frames when the person briefly exits the frame). Then merging `*/running/*` keys into the JSON (other classes remain identical) and regenerating the npz. Result: **train running 4 → 16 videos**, recall **12.9 % → 72.9 %**, ensemble **89.22 % → 90.55 %**.
   - **Why NOT a sampler:** a previous variant with `WeightedRandomSampler(inv)` oversampled the 21 running groups — this didn't create new information, only instability (Val Acc collapsed in early epochs), and actually *lowered* the ensemble score. Oversampling does not substitute real data; the correct fix is at extraction. Final training is with `--balanced_sampler none`.
   - The 50 residual running→jogging errors are now an **intrinsic ceiling** (the two classes differ only in translation speed), not a lack of data.
2. **`handwaving` ↔ `handclapping`** (53 handwaving → handclapping). Classic confusable pair on KTH — both are local hand movements, without global translation; they differ by amplitude/synchronization.

> **Suggested Figure #9**: Bar chart of per-seed individual accuracy vs. ensemble.
> **Suggested Figure #10**: Heatmap of the test confusion matrix (highlighting the running→jogging cell).

---

## 12. CNN ↔ CSNN Comparison

The entire pre-processing chain (person detection → bbox → crop → HOG) and the **same subject split (tvt 8/8/9)** are identical between the two pipelines, making the metric-to-metric comparison fair. The differences come down to *how* the classification on temporal HOG vectors is done.

| Aspect                  | CSNN (Spiking Network)                     | CNN (this pipeline)                        |
|-------------------------|--------------------------------------------|--------------------------------------------|
| Coding                  | Latency / rate coding on HOG               | Dense float, no coding                     |
| Learning                | Local STDP + WTA, no backprop              | End-to-end backprop (AdamW + CosineLR)     |
| Unit                    | LIF (Leaky Integrate-and-Fire) Neuron      | Conv3D + BatchNorm + ReLU                  |
| Temporal Representation | Continuous-time accumulation               | 3D convolutions over T = 19 frames         |
| Segmentation            | T = 19, frame_gap = 2                      | T = 19, frame_gap = 2 (identical)          |
| Augmentation            | Same HOG source                            | Same HOG source + CNN augmentations        |
| Selection / Reporting   | val + test per seeds                       | best on val, report mean ± std on test     |
| Energy & Sparsity       | Sparse activity, neuromorphic hardware fit | Dense, energy proportional to MACs         |
| Test Accuracy (mean ± std) | _____ % ± _____ *(from CSNN results)*   | **89.80 % ± 1.06** |
| Test Accuracy (ensemble + TTA)| —                                    | **90.55 %** |

> CNN figures are **final** (running fixed at the source, see 11.4). The table is to be completed when CSNN figures are available. Interpretation: The CNN provides the **upper ceiling** on this HOG representation with this split; the CSNN is evaluated relative to that ceiling as an energy-efficient alternative for neuromorphic deployment.

---

## 13. Future Directions

1. **Exploring other (T, frame_gap) combinations.** The current configuration (T = 19, g = 2) covers ~1.4 s of context. It would be useful to compare against a larger g (longer temporal context) vs bounding box stability.
2. **Secondary classifier specialized for confusable pairs.** If most residual errors come from `handclapping ↔ handwaving` and `jogging ↔ running`, a two-stage strategy might help: stage 1 = 6-class classifier; stage 2 = dedicated CNN just for the suspected pair (e.g. only on bbox-velocity for `jogging ↔ running`).
3. **Robustness over more seeds.** Current: 5 seeds. Useful to test 10 seeds for testing (as in the CSNN protocol) to strengthen the statistical claim.
4. **Confidence calibration (temperature scaling)** on the ensemble — so that "X% confidence" in Streamlit is numerically meaningful.
5. **Cross-scenario transfer**: train on d1+d2+d3 and test on d4 — robustness to background/clothing changes.

---

## 14. Suggested Figures for the Paper

| # | Content | Place in Paper |
|--:|---------|----------------|
| 1 | Full pipeline schema (video → JSON → NPZ → tensor → CNN → ensemble) | Beginning of chapter |
| 2 | A sequence of T = 19 frames (step g = 2) with the bbox drawn | Bounding boxes subsection |
| 3 | HOG visualization (person crop + gradient map + 8 × 8 cells overlay) | HOG subsection |
| 4 | Mosaic of augmented variants (orig + flip + 4 jit) for the same sample | Augmentation subsection |
| 5 | Diagram of the 80 channels (HOG + diff + bbox + bbox_vel) | Streams subsection |
| 6 | Architecture diagram `HARConv3DNet` with tensor volumes (80×19×15×7 → … → 6) | Architecture subsection |
| 7 | Train/val loss + accuracy curves for a representative seed | Training subsection |
| 8 | Screenshot of Streamlit UI with a classified example | Demonstrator subsection |
| 9 | Bar chart of individual seed vs ensemble accuracy | Results subsection |
| 10 | Confusion matrix on test as heatmap | Results subsection |

---

## Appendix A — Reproducing the Results

```bash
# 1) Bboxes (CSNN-side; only if JSON does not already exist). T=19, frame_gap=2:
python3 src/tool/extract_bboxes_kth.py \
    --input_path /home/mmuntean/kth_organized_tvt/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --output hog/hog_person_data_tvt_19_f10_g2.json

# 1b) Running fix: re-extract only running with permissive thresholds + larger carry,
#     merge into JSON from step 1 → canonical "runfix" JSON. See 11.4.
python3 src/tool/extract_bboxes_kth.py \
    --input_path /home/mmuntean/kth_organized_tvt/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --frame_width 160 --frame_height 120 \
    --only_action running \
    --hit_threshold -1.2 --min_bbox_area_ratio 0.004 \
    --min_bbox_aspect 0.15 --max_bbox_aspect 2.0 \
    --mog2_min_area 300 --max_carry 4 \
    --merge_into hog/hog_person_data_tvt_19_f10_g2.json \
    --output    hog/hog_person_data_tvt_19_f10_g2_runfix.json
# (steps 1b–4 are automated in cnn_har_app/reextract_running.sh)

# 2) Augmented HOG (run from cnn_har_app/):
cd cnn_har_app
python3 extract_hog_augmented.py \
    --bbox_json ../hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --output   ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --num_aug 8 --aug_profile strong \
    --video_root /home/mmuntean/kth_organized_tvt
# verify: features shape: (14715, 71820)

# 3) Train 5 seeds (best on val, report on test; NO sampler):
for s in 42 123 7 13 99; do
  python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --balanced_sampler none \
    --seed $s \
    --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 \
    --temporal_shift_max 2 \
    --ema_decay 0.999 \
    --ema_start 5 \
    2>&1 | tee ../data/log_cnn_tvt19fix_s$s.txt
done

# 4) Evaluate as ensemble with TTA:
python3 eval_ensemble.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --checkpoints "models/har_conv3d_tvt19fix_s*.pth" \
    --ensemble_mode logits \
    --tta_reverse \
    --tta_shift 1

# 5) Mean ± std on test from logs (CSNN comparison):
python3 - <<'PY'
import re, glob, statistics
accs=[]
for f in sorted(glob.glob("../data/log_cnn_tvt19fix_s*.txt")):
    m=re.findall(r"Final test accuracy:\s*([\d.]+)%", open(f).read())
    if m: accs.append(float(m[-1])); print(f.split('/')[-1], m[-1]+'%')
if len(accs)>=2:
    print(f"TEST: {statistics.mean(accs):.2f}% +- {statistics.stdev(accs):.2f} (n={len(accs)})")
PY

# 6) Interactive demonstrator:
streamlit run app.py
```

---

## Appendix B — Quick Glossary

| Term | Meaning |
|------|---------|
| **HAR** | Human Action Recognition |
| **HOG** | Histogram of Oriented Gradients |
| **HOG3D** | Spatio-temporal extension of HOG (Kläser et al., 2008) |
| **bbox** | Bounding box, the rectangle enclosing the person |
| **frame_gap (g)** | Frame distance between consecutive samples in a clip (current: 2) |
| **num_groups** | How many T-frame windows are extracted per video (current: 10) |
| **temporal_kernel (T)** | Number of frames used for one sample (current: 19) |
| **tvt** | Train / val / test subject split (8 / 8 / 9) |
| **EMA** | Exponential Moving Average (over model weights) |
| **TTA** | Test-Time Augmentation |
| **Mixup** | Technique that linearly combines pairs of samples and labels |
| **Label smoothing** | Replaces one-hot $(1, 0, \dots)$ with $(1-\varepsilon,\ \varepsilon/(K-1), \dots)$ |
| **ROCm** | AMD stack for GPU compute (CUDA equivalent); MI210 = gfx90a |
| **CSNN** | Convolutional Spiking Neural Network |
| **STDP** | Spike-Timing-Dependent Plasticity |
| **LIF** | Leaky Integrate-and-Fire (neuron model) |