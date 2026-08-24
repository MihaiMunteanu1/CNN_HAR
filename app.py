"""
Streamlit demo for KTH Human Action Recognition.
Loads one or more trained Conv3D checkpoints and runs real inference on test-split clips.
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from dataset import HOGDataset, KTH_CLASSES
from model import build_model

# streamlit run app.py

DATA_PATH = "data/hog_test_only.npz"

MODEL_DIR = "models"
VIDEO_ROOT = "kth_organized_tvt"

KTH_DISPLAY = {
    "boxing":       "🥊 Boxing",
    "handclapping": "👏 Handclapping",
    "handwaving":   "👋 Handwaving",
    "jogging":      "🏃 Jogging",
    "running":      "🏃‍♂️ Running",
    "walking":      "🚶 Walking",
}



@st.cache_resource(show_spinner=False)
def load_test_dataset():
    return HOGDataset(
        DATA_PATH, split="test", video_root=VIDEO_ROOT,
        as_image=True, augment=False,
        include_diff=True, include_bbox=True,
        include_bbox_vel=True,
    )


@st.cache_resource(show_spinner=False)
def load_models(checkpoint_paths_tuple, sample_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in checkpoint_paths_tuple:
        m = build_model(hog_shape=sample_shape)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        models.append(m)
    return models, device


@st.cache_data(show_spinner=False)
def fetch_video_frames(video_path: str, frame_indices: tuple):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    needed = set(frame_indices)
    found = {}
    idx = 0
    max_needed = max(needed) if needed else -1
    while True:
        ret, frame = cap.read()
        if not ret or idx > max_needed:
            break
        if idx in needed:
            found[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return [found[i] for i in frame_indices if i in found]


def select_preview_positions(bboxes, n=6):
    T = len(bboxes)
    valid = [i for i in range(T) if bboxes[i][2] > 0 and bboxes[i][3] > 0]
    pool = valid if len(valid) >= n else list(range(T))
    if len(pool) <= n:
        return pool
    picks = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return sorted({pool[p] for p in picks})



def predict(models, x, device):
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits_sum = None
        for m in models:
            logits = m(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
    avg_logits = logits_sum / len(models)
    return torch.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy()



st.set_page_config(page_title="CNN for HAR — KTH", layout="wide")
st.title("Human Action Recognition — KTH")
st.caption(
    "Inferență pe clipurile din test split. "
    "Model: 3D-CNN pe HOG features cu bbox metadata. "
    "Baseline-ul CNN a rețelei spiking CSNN."
)

st.sidebar.header("Model")

ckpt_pattern = os.path.join(MODEL_DIR, "har_conv3d_tvt19fix_s*.pth")
ckpts = sorted(glob.glob(ckpt_pattern))
if not ckpts:
    st.error(
        f"Nu exista checkpoint-uri la `{ckpt_pattern}`.\n "
    )
    st.stop()

def format_ckpt_name(path):
    base = os.path.basename(path)
    if "_s" in base and base.endswith(".pth"):
        seed = base.split("_s")[-1].replace(".pth", "")
        return f"Model (Seed {seed})"
    return base

selected_ckpts = st.sidebar.multiselect(
    "Checkpoint-uri Conv3D pentru ensemble",
    options=ckpts,
    default=ckpts,
    format_func=format_ckpt_name,
)
if not selected_ckpts:
    st.warning("Selecteaza cel putin un checkpoint.")
    st.stop()

mode = "Ensemble" if len(selected_ckpts) > 1 else "Single model"
st.sidebar.markdown(f"**Mod**: {mode} ({len(selected_ckpts)} model)")

with st.spinner("Se incarca setul de test..."):
    dataset = load_test_dataset()

if not dataset.metadata:
    st.error("Setul de test e gol")
    st.stop()

sample_x, _ = dataset[0]
sample_shape = tuple(sample_x.shape)
# st.sidebar.text(f"Input shape: {sample_shape}")

with st.spinner(f"Se incarca {len(selected_ckpts)} checkpoint(uri)..."):
    models, device = load_models(tuple(selected_ckpts), sample_shape)

st.sidebar.text(f"Test samples: {len(dataset)}")

with st.sidebar.expander("Despre arhitectura"):
    n_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    st.markdown(
        f"""
- **HARConv3DNet** — 3 blocuri Conv3D (kernel 3×3×3) peste tensor `(C={sample_shape[1]}, T={sample_shape[0]}, H={sample_shape[2]}, W={sample_shape[3]})`
- Trainable params: **{n_params/1e6:.2f}M**
- Split tvt (8/8/9 subiecti): train 11-18, val {{19,20,21,23,24,25,1,4}}, test {{2,3,5,6,7,8,9,10,22}}
- Augmentari: 8 variante/video, doar pe train (flip orizontal, jitter bbox, brightness/gamma, blur+noise)
"""
    )

st.sidebar.divider()
st.subheader("1. Alege un clip de test")

col1, col2, col3 = st.columns(3)
with col1:
    subjects = sorted({m["subject"] for m in dataset.metadata})
    subject_sel = st.selectbox("Subiect", subjects)
with col2:
    actions_for_subject = sorted({
        m["action"] for m in dataset.metadata if m["subject"] == subject_sel
    })
    action_sel = st.selectbox(
        "Acțiune (ground truth)", actions_for_subject,
        format_func=lambda a: KTH_DISPLAY.get(a, a),
    )
with col3:
    matching = [
        i for i, m in enumerate(dataset.metadata)
        if m["subject"] == subject_sel and m["action"] == action_sel
    ]
    group_pos = st.selectbox(
        f"Clip ({len(matching)} disponibile)",
        list(range(len(matching))),
        format_func=lambda i: f"clip {i+1}",
    )

selected_idx = matching[group_pos]
selected_meta = dataset.metadata[selected_idx]
st.markdown(
    f"📁 Selectat: `{selected_meta['video_key']}`"
)

run = st.button("Clasifica", type="primary")

if run:
    sample_x, gt_label = dataset[selected_idx]
    probs = predict(models, sample_x, device)
    pred_idx = int(np.argmax(probs))
    pred_class = KTH_CLASSES[pred_idx]
    gt_class = KTH_CLASSES[gt_label]
    confidence = probs[pred_idx] * 100
    is_correct = pred_idx == gt_label

    st.divider()
    st.subheader("2. Rezultat")

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        marker = "✅" if is_correct else "❌"
        if is_correct:
            st.success(
                f"{marker} **Predictie corecta**: {KTH_DISPLAY[pred_class]} "
                f"({confidence:.1f}% confidence)"
            )
        else:
            st.error(
                f"{marker} **Predictie gresita**: {KTH_DISPLAY[pred_class]} "
                f"({confidence:.1f}% confidence) — corect era {KTH_DISPLAY[gt_class]}"
            )
        st.markdown(f"**Ground truth**: {KTH_DISPLAY[gt_class]}")
        st.markdown(
            f"**Mod inferenta**: {len(models)} model"
            + ("e (mean logits)" if len(models) > 1 else "")
        )

    with res_col2:
        st.markdown("**Distributia probabilitatilor**")
        order = np.argsort(probs)[::-1]
        for i in order:
            cls = KTH_CLASSES[i]
            disp = KTH_DISPLAY[cls]
            tag = ""
            if i == pred_idx:
                tag = " ← predictie"
            if i == gt_label and i != pred_idx:
                tag = " ← ground truth"
            st.progress(int(probs[i] * 100), text=f"{disp}: {probs[i]*100:.1f}%{tag}")

    st.subheader("3. Frame-uri sursa (cu Bounding Box)")
    video_path = Path(VIDEO_ROOT) / selected_meta["video_key"]
    if video_path.exists():
        frame_indices = selected_meta["frame_indices"]
        frames = fetch_video_frames(str(video_path), tuple(frame_indices))
        if frames and len(frames) == len(frame_indices):
            _, bboxes, _ = dataset.samples[selected_idx]

            positions = select_preview_positions(bboxes, n=6)
            positions = [p for p in positions if p < len(frames)]

            cols = st.columns(len(positions))
            for col, pos in zip(cols, positions):
                img = frames[pos].copy()
                fi = frame_indices[pos]
                h_img, w_img, _ = img.shape
                cx, cy, w, h = bboxes[pos].tolist()

                if w > 0 and h > 0:
                    x1, y1 = int((cx - w / 2) * w_img), int((cy - h / 2) * h_img)
                    x2, y2 = int((cx + w / 2) * w_img), int((cy + h / 2) * h_img)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                col.image(img, caption=f"frame {fi}", width='stretch')
        elif frames:
            st.info(
                f"Extras: {len(frames)}/{len(frame_indices)} cadre (unele frame-uri "
                "lipsesc)."
            )
        else:
            st.info("Video gasit, nu se pot extrage frame-urile.")
    else:
        st.info(f"Video sursa lipseste la `{video_path}`")

st.divider()
st.caption(
    "Aplicatie construita ca demonstrator practic pentru lucrarea de licenta."
)