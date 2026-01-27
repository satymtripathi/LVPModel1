# ============================================================
# KeratitisAI – Batch + Single Image Diagnostic UI (V5)
# SAFE | DEVICE-ROBUST | DOCTOR-LIKE
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from torchvision import models
from torch.cuda.amp import autocast
from dataclasses import dataclass
from typing import List, Dict, Tuple

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SEG_CKPT: str = "Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth"
    TRAIN_OUT_DIR: str = "training_results_v5"
    CLS_CKPT: str = "training_results_v5/checkpoints/best.pth"

    FEATURE_KEYS_JSON: str = "training_results_v5/feature_keys.json"
    FEAT_MU_NPY: str = "training_results_v5/feat_mu.npy"
    FEAT_SIGMA_NPY: str = "training_results_v5/feat_sigma.npy"

    CANONICAL_SIZE: int = 512
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224
    MAX_TILES: int = 24
    TOP_TILES_FOR_FEATURES: int = 6

    CLASSES_4: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")

cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# IMPORT PROJECT UTILS
# =========================
from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks

# =========================
# NORMALIZATION
# =========================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def to_tensor_norm(rgb):
    t = torch.from_numpy(rgb).permute(2,0,1).float()/255.
    return (t - MEAN) / STD

def resize_rgb(img, size):
    return cv2.resize(img, (size[1], size[0]), cv2.INTER_AREA)

# =========================
# LOAD EVERYTHING (CACHED)
# =========================
@st.cache_resource
def load_everything():
    with open(cfg.FEATURE_KEYS_JSON) as f:
        feat_keys = json.load(f)

    feat_mu = np.load(cfg.FEAT_MU_NPY).astype(np.float32)
    feat_sigma = np.maximum(np.load(cfg.FEAT_SIGMA_NPY).astype(np.float32), 1e-6)

    seg_model, idx_crop, idx_limbus, img_size = load_model(cfg.SEG_CKPT, DEVICE)
    seg_model.eval()

    ckpt = torch.load(cfg.CLS_CKPT, map_location=DEVICE)
    state = ckpt["model"] if "model" in ckpt else ckpt

    class MIL(nn.Module):
        def __init__(self, feat_dim):
            super().__init__()
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.tile_feat = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1280 + feat_dim, 4)

        def forward(self, img, feat):
            x = self.pool(self.tile_feat(img)).flatten(1)
            x = torch.cat([x, feat], 1)
            return self.fc(x)

    model = MIL(len(feat_keys)).to(DEVICE).eval()
    model.load_state_dict(state, strict=False)

    return seg_model, idx_crop, idx_limbus, img_size, model, feat_keys, feat_mu, feat_sigma

# =========================
# SINGLE IMAGE PIPELINE
# =========================
def run_single_image(uploaded_file, assets):
    seg_model, idx_crop, idx_limbus, img_size, model, FEAT_KEYS, FEAT_MU, FEAT_SIGMA = assets

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Invalid image")

    masks = predict_masks(seg_model, bgr, img_size, DEVICE)
    crop = masks[idx_crop]
    bgr = cv2.resize(bgr, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # global branch only (safe)
    g = resize_rgb(rgb, (cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE))
    g_t = to_tensor_norm(g).unsqueeze(0).to(DEVICE)

    # dummy features (replace with your extractor if needed)
    feat_vec = np.zeros(len(FEAT_KEYS), np.float32)
    feat_vec = (feat_vec - FEAT_MU) / FEAT_SIGMA
    f_t = torch.from_numpy(feat_vec).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), autocast(enabled=(DEVICE=="cuda")):
        logits = model(g_t, f_t)
        probs = torch.softmax(logits,1).cpu().numpy()[0]

    pred = cfg.CLASSES_4[int(np.argmax(probs))]
    return {
        "filename": uploaded_file.name,
        "prediction": pred,
        "confidence": float(np.max(probs)),
        "probs": probs
    }

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("KeratitisAI – Batch Diagnostic", layout="wide", page_icon="👁️")
st.title("🛡️ KeratitisAI – Batch Diagnostic Dashboard")

assets = load_everything()

uploaded_files = st.sidebar.file_uploader(
    "Upload Slit Lamp Images",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more images to start.")
    st.stop()

batch_mode = len(uploaded_files) > 1
st.sidebar.success(f"Mode: {'BATCH' if batch_mode else 'SINGLE'}")

# =========================
# RUN INFERENCE
# =========================
results = []
progress = st.progress(0)

for i, uf in enumerate(uploaded_files):
    try:
        out = run_single_image(uf, assets)
        results.append(out)
    except Exception as e:
        results.append({
            "filename": uf.name,
            "prediction": "ERROR",
            "confidence": 0.0,
            "error": str(e)
        })
    progress.progress((i+1)/len(uploaded_files))

st.success("Inference Completed")

# =========================
# RESULTS TABLE
# =========================
df = pd.DataFrame([
    {
        "Image": r["filename"],
        "Prediction": r["prediction"],
        "Confidence (%)": round(r["confidence"]*100,1)
    } for r in results
])

st.subheader("📊 Batch Results")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    csv,
    "keratitis_batch_results.csv",
    "text/csv"
)

# =========================
# PER IMAGE DETAILS
# =========================
st.subheader("🔍 Per-Image Details")
for r in results:
    with st.expander(r["filename"]):
        st.metric("Prediction", r["prediction"])
        st.metric("Confidence", f"{r['confidence']*100:.1f}%")

