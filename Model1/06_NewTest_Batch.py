"""
============================================================
RAW IMAGE INFERENCE (Single / Folder Batch) - MATCHES V5 TRAINING
------------------------------------------------------------
This script runs FULL pipeline from ONE raw slit-lamp image:

1) Read raw image
2) Run LIMBUS SEGMENTATION (multi-class)
3) GLOBAL image = apply crop mask (same idea as cache _global.jpg)
4) Generate POLAR tiles (rings x sectors) inside limbus
5) Clinical tile scoring (V5) + qnorm bias
6) Pad to MAX_TILES with black tiles + valid_mask
7) Compute TABULAR FEATURES:
      - global features (g_)
      - aggregate tile mean/std from top-K quality tiles
8) Standardize features using TRAIN feat_mu/feat_sigma (from ckpt or files)
9) Run V5 classifier (soft attention pooling) and output:
      - pred + probs
      - attention + top tiles
      - debug images optional

NO PRECOMPUTED CACHE REQUIRED.

USAGE:
Single image:
  python infer_raw_v5.py --seg_ckpt path/to/seg.pth --cls_ckpt path/to/best.pth --input path/to/img.jpg --train_out_dir train_precomputed_run_SAFE_V5_doctorlike --out_dir out

Folder batch:
  python infer_raw_v5.py --seg_ckpt path/to/seg.pth --cls_ckpt path/to/best.pth --input path/to/folder --train_out_dir train_precomputed_run_SAFE_V5_doctorlike --out_dir out

Notes:
- This matches your V5 training logic:
  * clinical tile scoring (glare + border-edge penalty + opacity reward)
  * soft attention pooling
  * feature standardization using train mu/sigma
- Loads feat_keys + mu/sigma from checkpoint if present, else from train_out_dir files.
============================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import glob
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision import models
from torch.cuda.amp import autocast

# -------------------------
# WINDOWS STABILITY
# -------------------------
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# CONFIG
# ============================================================
@dataclass
class InferCFG:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True  # only used on cuda

    # segmentation threshold
    SEG_THRESH: float = 0.5

    # sizes (must match training)
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224
    MAX_TILES: int = 24

    # polar tiling (match your precompute style)
    POLAR_THETA: int = 8
    POLAR_RINGS: int = 3
    RING_EDGES_FRAC: Tuple[float, float, float, float] = (0.0, 0.35, 0.70, 1.0)
    POLAR_MIN_PIXELS: int = 250
    POLAR_PAD: int = 2

    # features (match training)
    TOP_TILES_FOR_FEATURES: int = 6

    # MIL (must match training cfg)
    TOPK_POOL: int = 4          # for reporting only
    QUALITY_BETA: float = 0.7

    # debug
    SAVE_DEBUG: bool = True
    DEBUG_TOP_TILES: int = 6

cfg = InferCFG()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================
# IMAGE IO / PREPROCESS
# ============================================================
def read_bgr(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path)

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_rgb(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)

def to_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    return t

def apply_mask_rgb(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    out[mask01 == 0] = 0
    return out


# ============================================================
# SEGMENTATION LOADER
# ============================================================
def load_seg_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Seg checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_s = ckpt.get("config", {})
    state = ckpt["state_dict"]

    encoder_name = cfg_s.get("encoder_name", "timm-efficientnet-b0")
    target_list = cfg_s.get("target_list", [{"label": "crop"}, {"label": "limbus"}])

    labels = [t["label"].strip().lower() for t in target_list]
    idx_crop = labels.index("crop") if "crop" in labels else 0
    idx_limbus = labels.index("limbus") if "limbus" in labels else min(1, len(labels) - 1)

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=len(target_list),
        activation=None
    )
    model.load_state_dict(state)
    model.to(device).eval()

    img_size = cfg_s.get("img_size", (512, 512))  # (H,W)
    return model, idx_crop, idx_limbus, img_size

def build_seg_transform(img_size_hw: Tuple[int, int]):
    return A.Compose([
        A.Resize(img_size_hw[0], img_size_hw[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_masks_fast(model, image_bgr: np.ndarray, img_size_hw: Tuple[int, int], device: str, transform, thresh: float) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = transform(image=rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()  # (C,h,w)

    masks = []
    for c in range(probs.shape[0]):
        m = cv2.resize(probs[c], (W, H), interpolation=cv2.INTER_LINEAR)
        masks.append((m > thresh).astype(np.uint8))
    return np.stack(masks, axis=0)  # (C,H,W)


# ============================================================
# V5 Clinical Tile Scoring (MATCH TRAINING)
# ============================================================
def _gray_entropy_u8(gray_u8: np.ndarray) -> float:
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())

def _local_entropy_map(gray_u8: np.ndarray, k: int = 15) -> float:
    g = gray_u8.astype(np.float32)
    m = cv2.blur(g, (k, k))
    m_u8 = np.clip(m, 0, 255).astype(np.uint8)
    return _gray_entropy_u8(m_u8)

def _lbp_riu2_entropy(gray_u8: np.ndarray) -> Tuple[float, float]:
    g = gray_u8
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0, 0.0
    c = g[1:-1, 1:-1]
    n0 = g[0:-2, 0:-2]; n1 = g[0:-2, 1:-1]; n2 = g[0:-2, 2:]
    n3 = g[1:-1, 2:];   n4 = g[2:, 2:];      n5 = g[2:, 1:-1]
    n6 = g[2:, 0:-2];   n7 = g[1:-1, 0:-2]
    bits = np.stack([(n0 >= c), (n1 >= c), (n2 >= c), (n3 >= c),
                     (n4 >= c), (n5 >= c), (n6 >= c), (n7 >= c)], axis=-1).astype(np.uint8)
    trans = np.sum(bits[..., :-1] != bits[..., 1:], axis=-1) + (bits[..., 0] != bits[..., -1]).astype(np.uint8)
    ones = bits.sum(axis=-1)
    lbp = np.where(trans <= 2, ones, 9).astype(np.int32)
    hist = np.bincount(lbp.ravel(), minlength=10).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    ent = float(-(p[p > 1e-12] * np.log2(p[p > 1e-12])).sum())
    uniform_ratio = float(p[:9].sum())
    return ent, uniform_ratio

def _fft_high_freq_ratio(gray_u8: np.ndarray) -> float:
    f = np.fft.fft2(gray_u8.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-8
    mag = np.log(mag)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max() + 1e-8
    high = mag[rr > 0.6 * rmax].sum()
    total = mag.sum() + 1e-8
    return float(high / total)

def _border_edge_ratio(edges_u8: np.ndarray, border: int = 16) -> float:
    h, w = edges_u8.shape
    if h <= 2 * border or w <= 2 * border:
        return 0.0
    total = float(edges_u8.mean() / 255.0) + 1e-8
    ring = edges_u8.copy()
    ring[border:h-border, border:w-border] = 0
    ring_d = float(ring.mean() / 255.0)
    return float(ring_d / total)

def _glare_penalty(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    L = lab[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)
    glare = (L > 210) & (S < 40)
    frac = float(glare.mean())
    return float(max(0.0, (frac - 0.02) / 0.15))

def _opacity_score(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    cand = (L > 150) & (L < 225)
    frac = float(cand.mean())
    return float(min(1.0, frac / 0.35))

def tile_quality_score_v5(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0

    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    valid_ratio = float(valid.mean())
    if valid_ratio < 0.10:
        return 0.0

    g_valid = gray[valid]
    contrast = float(g_valid.std()) if g_valid.size > 0 else 0.0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lapv = float(lap[valid].var()) if valid.any() else 0.0

    loc_ent = _local_entropy_map(gray, k=15)
    lbp_ent, uniform_ratio = _lbp_riu2_entropy(gray)
    hf_ratio = _fft_high_freq_ratio(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges[valid].mean() / 255.0) if valid.any() else 0.0
    border_ratio = _border_edge_ratio(edges, border=max(8, gray.shape[0] // 14))

    glare_pen = _glare_penalty(tile_rgb)
    opacity = _opacity_score(tile_rgb)

    uniform_pen = 1.0 - 0.5 * uniform_ratio

    score = (
        0.22 * math.log1p(lapv) +
        0.10 * math.log1p(contrast) +
        0.22 * loc_ent +
        0.18 * lbp_ent +
        0.10 * hf_ratio +
        0.05 * edge_density +
        0.13 * opacity
    )
    score = score * (0.5 + 0.5 * valid_ratio) * uniform_pen
    score = score * (1.0 - 0.8 * min(1.0, border_ratio)) * (1.0 - 0.9 * min(1.0, glare_pen))
    return float(max(0.0, score))


# ============================================================
# POLAR TILING (vectorized)
# ============================================================
def polar_tiles_v5(rgb_full: np.ndarray, crop_mask: np.ndarray, limbus_mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      global_rgb          : RGB, crop applied
      tiles_sorted        : list of RGB tiles sorted by quality desc (unpadded, <= MAX_TILES)
      qnorm_padded (T,)   : normalized 0..1, padded to MAX_TILES
      valid_mask (T,)     : 1 real tile, 0 padded
    """
    H, W = limbus_mask.shape[:2]
    global_rgb = apply_mask_rgb(rgb_full, crop_mask.astype(np.uint8))

    ys, xs = np.where(limbus_mask > 0)
    if xs.size == 0:
        return global_rgb, [], np.zeros((cfg.MAX_TILES,), np.float32), np.zeros((cfg.MAX_TILES,), np.float32)

    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5:
        return global_rgb, [], np.zeros((cfg.MAX_TILES,), np.float32), np.zeros((cfg.MAX_TILES,), np.float32)

    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]

    tiles: List[np.ndarray] = []
    qualities: List[float] = []

    for r in range(cfg.POLAR_RINGS):
        r0, r1 = ring_edges[r], ring_edges[r + 1]
        ring_mask = (limbus_mask > 0) & (rr >= r0) & (rr < r1)
        if ring_mask.sum() < cfg.POLAR_MIN_PIXELS:
            continue

        for s in range(cfg.POLAR_THETA):
            t0, t1 = 2 * np.pi * s / cfg.POLAR_THETA, 2 * np.pi * (s + 1) / cfg.POLAR_THETA
            wedge = ring_mask & (tt >= t0) & (tt < t1)
            if wedge.sum() < cfg.POLAR_MIN_PIXELS:
                continue

            ys_w, xs_w = np.where(wedge)
            x0, y0, x1, y1 = xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()
            x0, y0 = max(0, x0 - cfg.POLAR_PAD), max(0, y0 - cfg.POLAR_PAD)
            x1, y1 = min(W - 1, x1 + cfg.POLAR_PAD), min(H - 1, y1 + cfg.POLAR_PAD)

            tile = global_rgb[y0:y1 + 1, x0:x1 + 1].copy()
            tile_w = wedge[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
            tile[tile_w == 0] = 0

            tile_res = cv2.resize(tile, (cfg.TILE_SIZE, cfg.TILE_SIZE), interpolation=cv2.INTER_AREA)
            q = tile_quality_score_v5(tile_res)

            tiles.append(tile_res)
            qualities.append(float(q))

    if not tiles:
        return global_rgb, [], np.zeros((cfg.MAX_TILES,), np.float32), np.zeros((cfg.MAX_TILES,), np.float32)

    q_np = np.array(qualities, dtype=np.float32)
    order = np.argsort(-q_np)
    tiles = [tiles[i] for i in order.tolist()]
    q_np = q_np[order]

    tiles = tiles[:cfg.MAX_TILES]
    q_np = q_np[:cfg.MAX_TILES]

    # normalize q -> 0..1 (on real tiles only)
    mn, mx = float(q_np.min()), float(q_np.max())
    q01 = (q_np - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(q_np)

    # pad q01 + valid_mask to MAX_TILES
    T = cfg.MAX_TILES
    valid_mask = np.zeros((T,), dtype=np.float32)
    qnorm = np.zeros((T,), dtype=np.float32)

    n = len(tiles)
    if n > 0:
        valid_mask[:n] = 1.0
        qnorm[:n] = q01.astype(np.float32)

    return global_rgb, tiles, qnorm, valid_mask


# ============================================================
# FEATURE EXTRACTION (same as training)
# ============================================================
def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())

def glcm_contrast_approx(gray: np.ndarray, levels: int = 32) -> float:
    g = (gray.astype(np.int32) * (levels - 1) // 255).astype(np.int32)
    a = g[:, :-1]
    b = g[:, 1:]
    return float(np.mean((a - b) ** 2))

def colourfulness_index(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

def fft_features(gray: np.ndarray) -> Tuple[float, float]:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-8
    mag_log = np.log(mag)

    h, w = mag_log.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    rr_flat = rr.ravel()
    mag_flat = mag_log.ravel()
    mask = rr_flat > 3
    if mask.sum() < 10:
        return 0.0, 0.0

    peak_idx = np.argmax(mag_flat[mask])
    peak_r = float(rr_flat[mask][peak_idx])

    rmax = rr.max() + 1e-8
    high = mag_log[rr > 0.6 * rmax].sum()
    total = mag_log.sum() + 1e-8
    return peak_r, float(high / total)

def lbp_riu2_features(gray: np.ndarray) -> Tuple[float, float, float, float]:
    g = gray.astype(np.uint8)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0, 0.0, 0.0, 0.0

    c = g[1:-1, 1:-1]
    n0 = g[0:-2, 0:-2]
    n1 = g[0:-2, 1:-1]
    n2 = g[0:-2, 2:]
    n3 = g[1:-1, 2:]
    n4 = g[2:, 2:]
    n5 = g[2:, 1:-1]
    n6 = g[2:, 0:-2]
    n7 = g[1:-1, 0:-2]

    bits = np.stack([(n0 >= c), (n1 >= c), (n2 >= c), (n3 >= c),
                     (n4 >= c), (n5 >= c), (n6 >= c), (n7 >= c)], axis=-1).astype(np.uint8)
    trans = np.sum(bits[..., :-1] != bits[..., 1:], axis=-1) + (bits[..., 0] != bits[..., -1]).astype(np.uint8)
    ones = bits.sum(axis=-1)
    lbp = np.where(trans <= 2, ones, 9).astype(np.int32)

    hist = np.bincount(lbp.ravel(), minlength=10).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    codes = np.arange(10, dtype=np.float64)
    mean = (p * codes).sum()
    var = (p * (codes - mean) ** 2).sum()
    ent = float(-(p[p > 1e-12] * np.log2(p[p > 1e-12])).sum())
    uniform_ratio = float(p[:9].sum())
    energy = float((p ** 2).sum())
    return float(var), ent, uniform_ratio, energy

def local_entropy_stats(gray: np.ndarray, k: int = 15) -> Tuple[float, float]:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sqr_mean = cv2.blur(g * g, (k, k))
    std = np.sqrt(np.maximum(0.0, sqr_mean - mean * mean))
    mean_u8 = np.clip(mean, 0, 255).astype(np.uint8)
    std_u8 = np.clip(std, 0, 255).astype(np.uint8)
    return grayscale_entropy(mean_u8), grayscale_entropy(std_u8)

def edge_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)
    laplacian_density = float((lap_abs > 20).mean())

    cnts, _ = cv2.findContours((edges > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = 0.0
    for c in cnts[:50]:
        perim += cv2.arcLength(c, True)
    return edge_density, laplacian_density, float(perim)

def wavelet_haar_energies(gray: np.ndarray, levels: int = 3) -> Dict[str, float]:
    g = gray.astype(np.float32)
    out: Dict[str, float] = {}
    cur = g.copy()
    for lv in range(1, levels + 1):
        h, w = cur.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        cur = cur[:h2, :w2]

        a = cur[0::2, 0::2]
        b = cur[0::2, 1::2]
        c = cur[1::2, 0::2]
        d = cur[1::2, 1::2]

        ll = (a + b + c + d) * 0.25
        lh = (a - b + c - d) * 0.25
        hl = (a + b - c - d) * 0.25
        hh = (a - b - c + d) * 0.25

        out[f"l{lv}_lh_band"] = float(np.mean(lh * lh))
        out[f"l{lv}_hl_band"] = float(np.mean(hl * hl))
        out[f"l{lv}_hh_band"] = float(np.mean(hh * hh))
        cur = ll
    return out

def compute_colour_stats(rgb: np.ndarray) -> Dict[str, float]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    lab_l = float(lab[..., 0].mean())
    lab_a = float(lab[..., 1].mean())
    lab_b = float(lab[..., 2].mean())

    hsv_h_mean = float(hsv[..., 0].mean())
    hsv_s_mean = float(hsv[..., 1].mean())

    colour = colourfulness_index(rgb)

    a_ch = lab[..., 1].ravel()
    b_ch = lab[..., 2].ravel()
    a_b_cov = float(np.cov(a_ch, b_ch)[0, 1]) if a_ch.size > 10 else 0.0

    return {
        "lab_l": lab_l,
        "lab_a": lab_a,
        "lab_b": lab_b,
        "hsv_h_mean": hsv_h_mean,
        "hsv_s_mean": hsv_s_mean,
        "colour": colour,
        "a_b_cov": a_b_cov,
    }

def compute_gray_stats(gray: np.ndarray) -> Dict[str, float]:
    return {
        "hist_mean": float(gray.mean()),
        "hist_std": float(gray.std()),
        "hist_ent": grayscale_entropy(gray),
        "glcm_cont": glcm_contrast_approx(gray),
    }

def compute_texture_features(gray: np.ndarray) -> Dict[str, float]:
    fft_peak_freq, fft_high_freq_ratio = fft_features(gray)
    lbp_var, lbp_ent, lbp_uniform_ratio, lbp_energy = lbp_riu2_features(gray)
    edge_density, laplacian_density, perimeter_pixels = edge_metrics(gray)
    local_ent_mean, local_ent_std = local_entropy_stats(gray, k=15)

    g_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    glcm_loc_contrast = glcm_contrast_approx(g_blur)
    glcm_loc_hom = float(1.0 / (1.0 + glcm_loc_contrast))
    glcm_loc_corr = float(np.corrcoef(gray.ravel(), g_blur.ravel())[0, 1]) if gray.size > 10 else 0.0

    return {
        "fft_peak_freq": float(fft_peak_freq),
        "fft_high_freq_ratio": float(fft_high_freq_ratio),
        "lbp_var": float(lbp_var),
        "lbp_ent": float(lbp_ent),
        "lbp_uniform_ratio": float(lbp_uniform_ratio),
        "lbp_energy": float(lbp_energy),
        "edge_density": float(edge_density),
        "laplacian_density": float(laplacian_density),
        "perimeter_pixels": float(perimeter_pixels),
        "local_entropy_mean": float(local_ent_mean),
        "local_entropy_std": float(local_ent_std),
        "glcm_loc_contrast": float(glcm_loc_contrast),
        "glcm_loc_hom": float(glcm_loc_hom),
        "glcm_loc_corr": float(glcm_loc_corr),
    }

def compute_pos_shape_proxy(gray: np.ndarray) -> Dict[str, float]:
    mask = (gray > 5).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size < 20:
        return {"pos_central": 0.0, "round": 0.0, "el": 0.0, "solidity": 0.0}

    h, w = gray.shape
    cx, cy = xs.mean(), ys.mean()
    dx = (cx - (w / 2.0)) / (w / 2.0 + 1e-8)
    dy = (cy - (h / 2.0)) / (h / 2.0 + 1e-8)
    dist = math.sqrt(dx * dx + dy * dy)
    pos_central = float(max(0.0, 1.0 - dist))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"pos_central": pos_central, "round": 0.0, "el": 0.0, "solidity": 0.0}

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c) + 1e-8)
    perim = float(cv2.arcLength(c, True) + 1e-8)
    roundness = float(4.0 * math.pi * area / (perim * perim))

    x, y, bw, bh = cv2.boundingRect(c)
    el = float(max(bw, bh) / (min(bw, bh) + 1e-8))

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull) + 1e-8)
    solidity = float(area / hull_area)

    return {"pos_central": pos_central, "round": roundness, "el": el, "solidity": solidity}

def extract_feature_dict_from_rgb(rgb: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats: Dict[str, float] = {}
    feats.update(compute_gray_stats(gray))
    feats.update(compute_colour_stats(rgb))
    feats.update(compute_texture_features(gray))
    feats.update(compute_pos_shape_proxy(gray))
    feats.update(wavelet_haar_energies(gray, levels=3))
    return feats

def dict_to_feature_vec(d: Dict[str, float], keys: List[str]) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float32)

def aggregate_tile_features(tile_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not tile_dicts:
        return {}
    keys = sorted(tile_dicts[0].keys())
    mat = np.stack([dict_to_feature_vec(td, keys) for td in tile_dicts], axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    out: Dict[str, float] = {}
    for i, k in enumerate(keys):
        out[f"tile_mean_{k}"] = float(mean[i])
        out[f"tile_std_{k}"] = float(std[i])
    return out

def compute_features_for_inference(global_rgb: np.ndarray, tiles_sorted_by_quality: List[np.ndarray], feat_keys: List[str]) -> np.ndarray:
    feats_global = extract_feature_dict_from_rgb(global_rgb)
    feats: Dict[str, float] = {f"g_{k}": v for k, v in feats_global.items()}

    top_tiles = tiles_sorted_by_quality[:max(1, min(cfg.TOP_TILES_FOR_FEATURES, len(tiles_sorted_by_quality)))]
    tile_dicts = [extract_feature_dict_from_rgb(t) for t in top_tiles] if top_tiles else []
    feats.update(aggregate_tile_features(tile_dicts))

    return dict_to_feature_vec(feats, feat_keys)


# ============================================================
# CLASSIFIER MODEL (V5: soft attention pooling + valid_mask)
# ============================================================
class GatedAttention(nn.Module):
    def __init__(self, L, D, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x):
        return self.attention_weights(self.attention_V(x) * self.attention_U(x))

class DualBranchMIL_WithFeatures_V5(nn.Module):
    def __init__(self, num_classes: int, topk_pool: int, quality_beta: float, feat_dim: int):
        super().__init__()
        self.topk_pool = topk_pool
        self.quality_beta = quality_beta

        tile_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.tile_feat = tile_base.features
        self.tile_pool = nn.AdaptiveAvgPool2d(1)

        self.tile_latent = 256
        self.tile_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, self.tile_latent),
            nn.LayerNorm(self.tile_latent),
            nn.ReLU(),
        )
        self.attention = GatedAttention(self.tile_latent, 128, 1)

        global_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.global_feat = global_base.features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.global_latent = 256
        self.global_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, self.global_latent),
            nn.LayerNorm(self.global_latent),
            nn.ReLU(),
        )

        self.feat_latent = 128
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, self.feat_latent),
            nn.LayerNorm(self.feat_latent),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.tile_latent + self.global_latent + self.feat_latent, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, global_img, tiles, feats, qnorm=None, valid_mask=None):
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B * T, C, H, W)

        t = self.tile_pool(self.tile_feat(tiles_))
        z = self.tile_projector(t).view(B, T, -1)  # (B,T,256)

        a_logits = self.attention(z).squeeze(-1)   # (B,T)

        if qnorm is not None:
            a_logits = a_logits + self.quality_beta * qnorm.to(a_logits.device)

        if valid_mask is not None:
            a_logits = a_logits.masked_fill(valid_mask.to(a_logits.device) == 0, -1e9)

        a = torch.softmax(a_logits, dim=1)         # (B,T)

        # V5: soft pooling (doctor-like "consider all")
        bag = (a.unsqueeze(-1) * z).sum(dim=1)     # (B,256)

        # top tiles only for reporting
        k = min(self.topk_pool, T)
        top_idx = torch.topk(a, k=k, dim=1).indices

        g = self.global_pool(self.global_feat(global_img))
        g = self.global_projector(g)

        f = self.feat_mlp(feats)

        fused = torch.cat([bag, g, f], dim=1)
        logits = self.classifier(fused)
        return logits, a, top_idx


# ============================================================
# CLASSIFIER CHECKPOINT LOADER (loads feat_keys + mu/sigma)
# ============================================================
ORIG_4 = ["Edema", "Scar", "Infection", "Normal"]

def load_feature_meta_from_files(train_out_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys_path = os.path.join(train_out_dir, "feature_keys.json")
    mu_path = os.path.join(train_out_dir, "feat_mu.npy")
    sig_path = os.path.join(train_out_dir, "feat_sigma.npy")

    if not os.path.exists(keys_path):
        raise FileNotFoundError(keys_path)
    if not os.path.exists(mu_path):
        raise FileNotFoundError(mu_path)
    if not os.path.exists(sig_path):
        raise FileNotFoundError(sig_path)

    with open(keys_path, "r", encoding="utf-8") as f:
        feat_keys = json.load(f)

    feat_mu = np.load(mu_path).astype(np.float32)
    feat_sigma = np.load(sig_path).astype(np.float32)
    feat_sigma = np.maximum(feat_sigma, 1e-6)

    if len(feat_keys) != int(feat_mu.shape[0]) or len(feat_keys) != int(feat_sigma.shape[0]):
        raise RuntimeError("feature_keys.json and mu/sigma length mismatch")

    return feat_keys, feat_mu, feat_sigma

def load_classifier_v5(ckpt_path: str, train_out_dir: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_ck = ckpt.get("cfg", {})

    use_3class = bool(cfg_ck.get("USE_3CLASS_MERGE", False))
    classes = ["NonInfect_Other", "Normal", "Infection"] if use_3class else ORIG_4

    topk_pool = int(cfg_ck.get("TOPK_POOL", cfg.TOPK_POOL))
    quality_beta = float(cfg_ck.get("QUALITY_BETA", cfg.QUALITY_BETA))

    # prefer checkpoint meta (best: no extra files)
    feat_keys = ckpt.get("feat_keys", [])
    feat_mu = ckpt.get("feat_mu", None)
    feat_sigma = ckpt.get("feat_sigma", None)

    if not feat_keys or feat_mu is None or feat_sigma is None:
        # fallback to files
        feat_keys, feat_mu, feat_sigma = load_feature_meta_from_files(train_out_dir)

    feat_mu = np.array(feat_mu, dtype=np.float32)
    feat_sigma = np.maximum(np.array(feat_sigma, dtype=np.float32), 1e-6)

    model = DualBranchMIL_WithFeatures_V5(
        num_classes=len(classes),
        topk_pool=topk_pool,
        quality_beta=quality_beta,
        feat_dim=len(feat_keys),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    return model, classes, feat_keys, feat_mu, feat_sigma


# ============================================================
# DEBUG SAVE
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_debug(out_dir: str, stem: str, bgr: np.ndarray, crop_mask: np.ndarray, limbus_mask: np.ndarray,
               global_rgb: np.ndarray, tiles: List[np.ndarray], probs: np.ndarray, pred_name: str):
    ensure_dir(out_dir)
    base = os.path.join(out_dir, stem)
    ensure_dir(base)

    cv2.imwrite(os.path.join(base, "input.jpg"), bgr)
    cv2.imwrite(os.path.join(base, "mask_crop.png"), (crop_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(base, "mask_limbus.png"), (limbus_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(base, "global.jpg"), cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR))

    for i, t in enumerate(tiles[:cfg.DEBUG_TOP_TILES]):
        cv2.imwrite(os.path.join(base, f"tile_{i:02d}.jpg"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))

    with open(os.path.join(base, "pred.txt"), "w", encoding="utf-8") as f:
        f.write(f"pred={pred_name}\n")
        for i, p in enumerate(probs.tolist()):
            f.write(f"class_{i}={p:.6f}\n")


# ============================================================
# INPUT DISCOVERY
# ============================================================
def collect_images(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    paths: List[str] = []
    for ext in VALID_EXTS:
        paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
    return sorted(list(set(paths)))


# ============================================================
# FULL PIPELINE FOR ONE IMAGE (MATCHES V5)
# ============================================================
@torch.no_grad()
def infer_one_image_v5(
    img_path: str,
    seg_model,
    idx_crop: int,
    idx_limbus: int,
    seg_img_size_hw: Tuple[int, int],
    seg_transform,
    cls_model: nn.Module,
    classes: List[str],
    feat_keys: List[str],
    feat_mu: np.ndarray,
    feat_sigma: np.ndarray,
    device: str
) -> Dict[str, Any]:
    bgr = read_bgr(img_path)
    if bgr is None:
        return {"path": img_path, "error": "cv2.imread failed"}

    # ---- segmentation ----
    masks = predict_masks_fast(seg_model, bgr, seg_img_size_hw, device, seg_transform, cfg.SEG_THRESH)
    crop_mask = masks[idx_crop].astype(np.uint8)
    limbus_mask = masks[idx_limbus].astype(np.uint8)

    # fallback: if crop empty, use limbus
    if crop_mask.sum() < 50 and limbus_mask.sum() > 50:
        crop_mask = limbus_mask.copy()

    rgb_full = bgr_to_rgb(bgr)

    # ---- global + tiles + qnorm + valid_mask ----
    global_rgb, tiles_sorted, qnorm, valid_mask = polar_tiles_v5(rgb_full, crop_mask, limbus_mask)

    # pad tiles to MAX_TILES with black
    black_tile = np.zeros((cfg.TILE_SIZE, cfg.TILE_SIZE, 3), dtype=np.uint8)
    if len(tiles_sorted) < cfg.MAX_TILES:
        tiles_sorted = tiles_sorted + [black_tile] * (cfg.MAX_TILES - len(tiles_sorted))
    else:
        tiles_sorted = tiles_sorted[:cfg.MAX_TILES]

    # ---- tensors ----
    global_rs = resize_rgb(global_rgb, (cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE))
    g_t = to_tensor_norm(global_rs).unsqueeze(0).to(device)

    tiles_t = torch.stack([to_tensor_norm(t) for t in tiles_sorted]).unsqueeze(0).to(device)
    q_t = torch.from_numpy(qnorm.astype(np.float32)).unsqueeze(0).to(device)
    vm_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).to(device)

    # ---- features (compute + STANDARDIZE like training) ----
    fvec = compute_features_for_inference(global_rgb, tiles_sorted, feat_keys).astype(np.float32)
    fvec = (fvec - feat_mu.astype(np.float32)) / np.maximum(feat_sigma.astype(np.float32), 1e-6)
    f_t = torch.from_numpy(fvec).unsqueeze(0).to(device)

    # ---- classifier ----
    use_amp = (device == "cuda" and cfg.USE_AMP)
    with autocast(enabled=use_amp):
        logits, att, top_idx = cls_model(g_t, tiles_t, f_t, qnorm=q_t, valid_mask=vm_t)

    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    pred_name = classes[pred_id]

    att_np = att.detach().cpu().numpy()[0] if att is not None else None
    top_np = top_idx.detach().cpu().numpy()[0].tolist() if top_idx is not None else []

    # count real tiles
    n_real = int(valid_mask.sum())

    return {
        "path": img_path,
        "pred_id": pred_id,
        "pred_name": pred_name,
        "probs": probs,
        "top_idx": top_np,
        "att": att_np,
        "crop_mask_sum": int(crop_mask.sum()),
        "limbus_mask_sum": int(limbus_mask.sum()),
        "tiles_real": n_real,
        "global_rgb": global_rgb,
        "tiles": tiles_sorted,
        "bgr": bgr,
        "crop_mask": crop_mask,
        "limbus_mask": limbus_mask,
        "qnorm": qnorm,
        "valid_mask": valid_mask,
    }


# ============================================================
# CSV WRITE
# ============================================================
def write_csv(path: str, rows: List[Dict[str, Any]], classes: List[str]):
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    header = ["path", "pred_id", "pred_name"] + [f"prob_{c}" for c in classes]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            probs = r.get("probs", None)
            if probs is None:
                line = [r.get("path", ""), "", ""] + [""] * len(classes)
            else:
                line = [r.get("path", ""), str(r.get("pred_id", "")), str(r.get("pred_name", ""))]
                line += [f"{float(probs[i]):.6f}" for i in range(len(classes))]
            f.write(",".join(line) + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_ckpt", required=True, type=str, help="Segmentation checkpoint (.pth)")
    parser.add_argument("--cls_ckpt", required=True, type=str, help="Classifier checkpoint best.pth/last.pth")
    parser.add_argument("--train_out_dir", required=True, type=str, help="Training OUT_DIR (for feature_keys/mu/sigma fallback)")
    parser.add_argument("--input", required=True, type=str, help="Image path or folder")
    parser.add_argument("--out_dir", default="infer_raw_out_v5", type=str)
    parser.add_argument("--device", default=cfg.DEVICE, type=str)
    parser.add_argument("--seg_thresh", default=cfg.SEG_THRESH, type=float)
    parser.add_argument("--no_debug", action="store_true")
    args = parser.parse_args()

    cfg.DEVICE = args.device
    cfg.SEG_THRESH = float(args.seg_thresh)
    cfg.SAVE_DEBUG = (not args.no_debug)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "debug"))

    # ---- load models ----
    seg_model, idx_crop, idx_limbus, seg_img_size_hw = load_seg_model(args.seg_ckpt, cfg.DEVICE)
    seg_transform = build_seg_transform(seg_img_size_hw)

    cls_model, classes, feat_keys, feat_mu, feat_sigma = load_classifier_v5(args.cls_ckpt, args.train_out_dir, cfg.DEVICE)

    print("DEVICE:", cfg.DEVICE)
    print("Seg ckpt:", args.seg_ckpt)
    print("Cls ckpt:", args.cls_ckpt)
    print("Train out:", args.train_out_dir)
    print("Seg idx_crop:", idx_crop, "idx_limbus:", idx_limbus, "img_size:", seg_img_size_hw)
    print("Classes:", classes)
    print("Feat dim:", len(feat_keys))

    # ---- run ----
    img_paths = collect_images(args.input)
    if not img_paths:
        raise RuntimeError(f"No images found in: {args.input}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for p in tqdm(img_paths, desc="Infer raw V5"):
        stem = os.path.splitext(os.path.basename(p))[0]
        out = infer_one_image_v5(
            img_path=p,
            seg_model=seg_model,
            idx_crop=idx_crop,
            idx_limbus=idx_limbus,
            seg_img_size_hw=seg_img_size_hw,
            seg_transform=seg_transform,
            cls_model=cls_model,
            classes=classes,
            feat_keys=feat_keys,
            feat_mu=feat_mu,
            feat_sigma=feat_sigma,
            device=cfg.DEVICE
        )

        if "error" in out:
            rows.append(out)
            continue

        # record row
        row = {"path": out["path"], "pred_id": out["pred_id"], "pred_name": out["pred_name"], "probs": out["probs"]}
        rows.append(row)

        # debug save
        if cfg.SAVE_DEBUG:
            save_debug(
                out_dir=os.path.join(args.out_dir, "debug"),
                stem=stem,
                bgr=out["bgr"],
                crop_mask=out["crop_mask"],
                limbus_mask=out["limbus_mask"],
                global_rgb=out["global_rgb"],
                tiles=out["tiles"],
                probs=out["probs"],
                pred_name=out["pred_name"]
            )

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    write_csv(csv_path, rows, classes)

    dt = time.time() - t0
    print(f"\n✅ Saved: {csv_path}")
    print(f"Total images: {len(img_paths)} | Time: {dt:.2f}s | Avg: {dt/max(1,len(img_paths)):.3f}s/img")


if __name__ == "__main__":
    main()
