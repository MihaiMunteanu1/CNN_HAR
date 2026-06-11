"""
Streamlit demo for KTH Human Action Recognition.

Loads one or more trained Conv3D checkpoints and runs real inference on
test-split clips. Optionally averages softmax outputs across checkpoints
(ensemble).

Run:
    cd cnn_har_app
    pip install streamlit
    streamlit run app.py
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


# --- caches -----------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_test_dataset():
    return HOGDataset(
        DATA_PATH, split="test", video_root=VIDEO_ROOT,
        as_image=True, augment=False,
        include_diff=True, include_bbox=True,
        include_bbox_vel=True,
    )


@st.cache_resource(show_spinner=False)
def load_models(checkpoint_paths_tuple, model_type, sample_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in checkpoint_paths_tuple:
        m = build_model(hog_shape=sample_shape, model_type=model_type)
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
    """Alege ~n poziții (în T) pentru preview: cadre cu bbox valid, distribuite
    uniform pe tot clipul (nu primele n). bboxes: array (T, 4) = (cx, cy, w, h)."""
    T = len(bboxes)
    valid = [i for i in range(T) if bboxes[i][2] > 0 and bboxes[i][3] > 0]
    pool = valid if len(valid) >= n else list(range(T))
    if len(pool) <= n:
        return pool
    picks = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return sorted({pool[p] for p in picks})


# --- inference --------------------------------------------------------------


def predict(models, x, device, ensemble_mode):
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits_sum = None
        probs_sum = None
        for m in models:
            logits = m(x)
            if ensemble_mode == "softmax":
                p = torch.softmax(logits, dim=1)
                probs_sum = p if probs_sum is None else probs_sum + p
            else:
                logits_sum = logits if logits_sum is None else logits_sum + logits
    if ensemble_mode == "softmax":
        return (probs_sum / len(models)).squeeze(0).cpu().numpy()
    avg_logits = logits_sum / len(models)
    return torch.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy()


# --- UI ---------------------------------------------------------------------

st.set_page_config(page_title="CNN for HAR — KTH", page_icon="🎥", layout="wide")
st.title("Human Action Recognition — KTH")
st.caption(
    "Inferență reală pe clipuri din test split. "
    "Model: 3D-CNN pe HOG features cu bbox metadata. "
    "Contrapartea clasică (CNN) a rețelei spiking (CSNN) descrisă în lucrare."
)

# Sidebar — model selection
st.sidebar.header("Model")

ckpt_pattern = os.path.join(MODEL_DIR, "har_conv3d_tvt19fix_s*.pth")
ckpts = sorted(glob.glob(ckpt_pattern))
if not ckpts:
    st.error(
        f"Nu există checkpoint-uri la `{ckpt_pattern}`. "
        "Antrenează modelul întâi:\n\n"
        "```\npython3 train.py --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz "
        "--balanced_sampler none --save_suffix _tvt19fix_s42\n```"
    )
    st.stop()

def format_ckpt_name(path):
    base = os.path.basename(path)
    if "_s" in base and base.endswith(".pth"):
        seed = base.split("_s")[-1].replace(".pth", "")
        return f"Model (Seed {seed})"
    return base

selected_ckpts = st.sidebar.multiselect(
    "Checkpoint(uri) Conv3D pentru ensemble",
    options=ckpts,
    default=ckpts,
    format_func=format_ckpt_name,
)
if not selected_ckpts:
    st.warning("Selectează cel puțin un checkpoint.")
    st.stop()

mode = "Ensemble" if len(selected_ckpts) > 1 else "Single model"
st.sidebar.markdown(f"**Mod**: {mode} ({len(selected_ckpts)} model)")

ensemble_mode = st.sidebar.selectbox(
    "Ensemble mode",
    options=["logits", "softmax"],
    index=0,
    format_func=lambda m: "Mean logits (recommended)" if m == "logits" else "Mean softmax",
)

# Load
with st.spinner("Se încarcă setul de test..."):
    dataset = load_test_dataset()

if not dataset.metadata:
    st.error("Setul de test e gol sau metadata nu e populată. Verifică json-ul.")
    st.stop()

sample_x, _ = dataset[0]
sample_shape = tuple(sample_x.shape)
st.sidebar.text(f"Input shape: {sample_shape}")

with st.spinner(f"Se încarcă {len(selected_ckpts)} checkpoint(uri)..."):
    models, device = load_models(tuple(selected_ckpts), "conv3d", sample_shape)

# st.sidebar.text(f"Device: {device}")
st.sidebar.text(f"Test samples: {len(dataset)}")

with st.sidebar.expander("Despre arhitectură"):
    n_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    st.markdown(
        f"""
- **HARConv3DNet** — 3 blocuri Conv3D (kernel 3×3×3) peste tensor `(C={sample_shape[1]}, T={sample_shape[0]}, H={sample_shape[2]}, W={sample_shape[3]})`
- Input: HOG (36) + diff (36) + bbox (4) + bbox_vel (4) = 80 canale per frame
- Trainable params: **{n_params/1e6:.2f}M**
- Split tvt (8/8/9 subiecți): train 11-18, val {{19,20,21,23,24,25,1,4}}, test {{2,3,5,6,7,8,9,10,22}}
- Augmentări: mixup α=0.02, Gaussian noise, label smoothing 0.02
"""
    )

st.sidebar.divider()
# Filter UI
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
    f"📁 Selectat: `{selected_meta['video_key']}`" #— group #{selected_meta['group_idx']} "
    # f"— frame-uri {selected_meta['frame_indices']}"
)

run = st.button("🎯 Clasifică", type="primary")

if run:
    sample_x, gt_label = dataset[selected_idx]
    probs = predict(models, sample_x, device, ensemble_mode)
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
                f"{marker} **Predicție corectă**: {KTH_DISPLAY[pred_class]} "
                f"({confidence:.1f}% confidență)"
            )
        else:
            st.error(
                f"{marker} **Predicție greșită**: {KTH_DISPLAY[pred_class]} "
                f"({confidence:.1f}% confidență) — corect era {KTH_DISPLAY[gt_class]}"
            )
        st.markdown(f"**Ground truth**: {KTH_DISPLAY[gt_class]}")
        st.markdown(
            f"**Mod inferență**: {len(models)} model"
            + (f"e (mean {ensemble_mode})" if len(models) > 1 else "")
        )

    with res_col2:
        st.markdown("**Distribuția probabilităților**")
        order = np.argsort(probs)[::-1]
        for i in order:
            cls = KTH_CLASSES[i]
            disp = KTH_DISPLAY[cls]
            tag = ""
            if i == pred_idx:
                tag = " ← predicție"
            if i == gt_label and i != pred_idx:
                tag = " ← ground truth"
            st.progress(int(probs[i] * 100), text=f"{disp}: {probs[i]*100:.1f}%{tag}")

    # Frame preview — 5-6 cadre bune (cu bbox valid), distribuite pe tot clipul
    st.subheader("3. Frame-uri sursă (cu Bounding Box)")
    video_path = Path(VIDEO_ROOT) / selected_meta["video_key"]
    if video_path.exists():
        frame_indices = selected_meta["frame_indices"]
        frames = fetch_video_frames(str(video_path), tuple(frame_indices))
        if frames and len(frames) == len(frame_indices):
            # bboxes (T, 4) din lista de samples a dataset-ului
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
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # verde

                col.image(img, caption=f"frame {fi}", use_container_width=True)
        elif frames:
            st.info(
                f"Am extras {len(frames)}/{len(frame_indices)} cadre (unele frame-uri "
                "lipsesc la decodare) — preview-ul cu bbox necesită aliniere completă."
            )
        else:
            st.info("Video găsit dar nu pot extrage frame-urile (codec?).")
    else:
        st.info(f"Video sursă lipsește la `{video_path}` — nu pot afișa frame-urile.")

st.divider()
st.caption(
    "Aplicație construită ca demonstrator practic pentru lucrarea de licență."
)