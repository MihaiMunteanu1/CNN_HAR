# 3D Convolutional Neural Networks for Human Action Recognition (HAR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cnn-har-munteanu-mihai.streamlit.app/)

This project implements a 3D Convolutional Neural Network (CNN) for Human Action Recognition on the KTH dataset. It serves as a high-performance baseline to be compared against a neuromorphic (CSNN) approach under identical conditions (dataset, temporal segmentation, and features).

The pipeline is optimized by pre-computing HOG (Histogram of Oriented Gradients) features, including spatial and photometric augmentations. This allows the CNN training to focus solely on learning spatiotemporal patterns from these features.

## Pipeline Overview

The workflow is divided into several distinct stages:

1.  **Person Detection**: Bounding boxes for subjects are detected in each video and saved to a `.json` file. This step uses a combination of HOG and MOG2 background subtraction.
2.  **Feature Pre-computation**: From the videos and bounding boxes, augmented HOG features are extracted and stored in a compressed `.npz` file. This is the most computationally intensive step and is only performed once.
3.  **Dataset & Dataloading**: A PyTorch `Dataset` class loads the `.npz` file, reshapes the HOG vectors, and creates additional input streams like temporal differences and bounding box kinematics.
4.  **Model Training**: A 3D CNN (`HARConv3DNet`) is trained on the prepared data. The model uses spatiotemporal convolutions to learn action patterns. Training is performed over 5 different random seeds.
5.  **Evaluation**: The final performance is evaluated by creating an ensemble of the 5 models and using Test-Time Augmentation (TTA) for robustness.
6.  **Interactive Demo**: A Streamlit application (`app.py`) allows for interactive classification of test clips and visualization of results.

## KTH Dataset

The KTH dataset contains 6 actions performed by 25 subjects in 4 different scenarios. For this project, a standard subject-based split is used to ensure generalization:
- **Train**: 8 subjects
- **Validation**: 8 subjects
- **Test**: 9 subjects

## Core Components

-   **`extract_hog_augmented.py`**: The script for pre-computing augmented HOG features. It applies flips, jitter, and photometric transformations to the training data.
-   **`dataset.py`**: Defines the `HOGDataset` which prepares multi-stream tensors for the model:
    -   **HOG features** (36 channels)
    -   **Temporal difference of HOG** (36 channels)
    -   **Bounding box position** (4 channels)
    -   **Bounding box velocity** (4 channels)
-   **`model.py`**: Defines the `HARConv3DNet` architecture, which consists of stacked 3D convolutional blocks followed by a classifier head.
-   **`train.py`**: The main training script. It uses AdamW, Cosine Annealing, EMA weights, and early stopping based on validation accuracy.
-   **`eval_ensemble.py`**: Script to evaluate the ensemble of trained models with optional TTA.
-   **`app.py`**: The Streamlit web application for interactive demonstration.

## How to Run

### 1. Install Dependencies
```bash
pip install torch numpy opencv-python streamlit
```

### 2. Pre-compute HOG Features
*(This assumes you have the KTH videos and the `hog_person_data_*.json` file from the person detection stage.)*
```bash
cd cnn_har_app
python extract_hog_augmented.py \
    --bbox_json ../hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --output ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --num_aug 8 --aug_profile strong \
    --video_root /path/to/kth_videos
```

### 3. Train the Models
Train 5 models with different seeds.
```bash
cd cnn_har_app
for s in 42 123 7 13 99; do
  python train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --seed $s \
    --save_suffix _tvt19fix_s$s
done
```

### 4. Evaluate the Ensemble
```bash
cd cnn_har_app
python eval_ensemble.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --checkpoints "models/har_conv3d_tvt19fix_s*.pth" \
    --tta_reverse --tta_shift 1
```

### 5. Launch the Interactive Demo
```bash
streamlit run app.py
```

## Results

The model achieves the following performance on the test set:

-   **Individual Models (Mean ± Std)**: **89.80 % ± 1.06 %**
-   **5-Model Ensemble + TTA**: **90.55 %**

The main source of confusion is between the `running` and `jogging` classes, which is a known challenge for this dataset.
