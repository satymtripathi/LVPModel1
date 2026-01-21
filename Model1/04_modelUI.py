import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
# Add project root to sys.path to resolve imports when running from subfolders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.cuda.amp import autocast
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import streamlit as st

from Limbus_Crop_Segmentation_System.inference_utils import load_model, predict_masks

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    SEG_CKPT: str = "Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth"
    CLS_CKPT: str = "training_results_v5/checkpoints/best.pth"
    FEATURE_KEYS_JSON: str = "training_results_v5/feature_keys.json"

    CLASSES_4: Tuple[str, ...] = ("Edema", "Scar", "Infection", "Normal")
    CLASS_COLORS = {
        "Edema": "#3498db",
        "Scar": "#9b59b6",
        "Infection": "#e74c3c",
        "Normal": "#2ecc71",
        "NonInfect_Other": "#9b59b6",
    }

    CANONICAL_SIZE: int = 512
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224
    MAX_TILES: int = 24

    POLAR_THETA: int = 8
    POLAR_RINGS: int = 3
    RING_EDGES_FRAC: Tuple[float, float, float, float] = (0.0, 0.35, 0.70, 1.0)
    POLAR_MIN_PIXELS: int = 250
    POLAR_PAD: int = 2

    TOP_TILES_FOR_FEATURES: int = 6

    TOPK_POOL_DEFAULT: int = 4
    QUALITY_BETA_DEFAULT: float = 0.7

    # ROI rectangle crop improvements
    CROP_RECT_PAD: int = 20
    CROP_MIN_AREA_FRAC: float = 0.002

cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# FAST tensor preprocess
# =========================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def resize_rgb(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)

def to_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    return t

def normalize01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(arr)

def apply_mask_rgb(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    out[mask01 == 0] = 0
    return out

# =========================
# ROI HELPERS (CROP OUT, NO OVERLAY)
# =========================
def postprocess_binary(mask01: np.ndarray, k_close: int = 21, k_open: int = 9) -> np.ndarray:
    m = (mask01.astype(np.uint8) * 255)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k2, iterations=1)
    return (m > 127).astype(np.uint8)

def largest_contour(mask01: np.ndarray):
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def rect_mask_from_crop(crop_mask01: np.ndarray, pad: int = 20, min_area_frac: float = 0.002) -> np.ndarray:
    """
    Make a clean filled rectangle mask from crop mask's largest contour bbox.
    """
    H, W = crop_mask01.shape[:2]
    c = largest_contour(crop_mask01)
    if c is None:
        return crop_mask01.astype(np.uint8)

    area = float(cv2.contourArea(c))
    if area < (min_area_frac * H * W):
        return crop_mask01.astype(np.uint8)

    x, y, w, h = cv2.boundingRect(c)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W - 1, x + w + pad)
    y2 = min(H - 1, y + h + pad)

    rect = np.zeros((H, W), dtype=np.uint8)
    rect[y1:y2+1, x1:x2+1] = 1
    return rect

def crop_roi_by_largest_rect(rgb_512: np.ndarray, crop_mask01_512: np.ndarray, pad: int = 20, min_area_frac: float = 0.002):
    """
    Crop the RGB image by bbox of largest contour in crop mask.
    Returns (roi_rgb, bbox or None), bbox=(x1,y1,x2,y2) in 512 space.
    """
    H, W = crop_mask01_512.shape[:2]
    c = largest_contour(crop_mask01_512)
    if c is None:
        return None, None

    area = float(cv2.contourArea(c))
    if area < (min_area_frac * H * W):
        return None, None

    x, y, w, h = cv2.boundingRect(c)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W - 1, x + w + pad)
    y2 = min(H - 1, y + h + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = rgb_512[y1:y2+1, x1:x2+1].copy()
    return roi, (x1, y1, x2, y2)

# =========================
# MIL model with features
# =========================
class GatedAttention(nn.Module):
    def __init__(self, L, D, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)
    def forward(self, x):
        return self.attention_weights(self.attention_V(x) * self.attention_U(x))

class DualBranchMIL_WithFeatures(nn.Module):
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

    def forward(self, global_img, tiles, feats, qnorm=None):
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B*T, C, H, W)

        t = self.tile_pool(self.tile_feat(tiles_))
        z = self.tile_projector(t).view(B, T, -1)

        a_logits = self.attention(z).squeeze(-1)
        if qnorm is not None:
            a_logits = a_logits + self.quality_beta * qnorm.to(a_logits.device)

        a = torch.softmax(a_logits, dim=1)
        k = min(self.topk_pool, T)
        top_idx = torch.topk(a, k=k, dim=1).indices

        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
        z_top = torch.gather(z, dim=1, index=idx_exp)
        bag = z_top.mean(dim=1)

        g = self.global_pool(self.global_feat(global_img))
        g = self.global_projector(g)

        f = self.feat_mlp(feats)

        fused = torch.cat([bag, g, f], dim=1)
        logits = self.classifier(fused)
        return logits, a, top_idx

# =========================
# Feature extraction (must match feature_keys.json)
# =========================
def _gray_entropy_u8(gray_u8: np.ndarray) -> float:
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())

def _local_entropy(gray_u8: np.ndarray, k: int = 15) -> float:
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

def _wavelet_haar_energies(gray: np.ndarray, levels: int = 3) -> Dict[str, float]:
    g = gray.astype(np.float32)
    out = {}
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

def extract_feature_dict(rgb: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if float(valid.mean()) < 0.05:
        valid = np.ones_like(gray, dtype=bool)

    g = gray[valid].astype(np.float32)

    feats = {}
    feats["hist_mean"] = float(g.mean())
    feats["hist_std"]  = float(g.std())
    feats["hist_ent"]  = float(_gray_entropy_u8(gray))
    feats["glcm_cont"] = float(np.mean((gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32)) ** 2))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    feats["lab_l"] = float(lab[..., 0][valid].mean())
    feats["lab_a"] = float(lab[..., 1][valid].mean())
    feats["lab_b"] = float(lab[..., 2][valid].mean())
    feats["hsv_h_mean"] = float(hsv[..., 0][valid].mean())
    feats["hsv_s_mean"] = float(hsv[..., 1][valid].mean())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    feats["laplacian_density"] = float((np.abs(lap) > 20).mean())
    feats["edge_density"] = float(cv2.Canny(gray, 50, 150).mean() / 255.0)
    feats["fft_high_freq_ratio"] = float(_fft_high_freq_ratio(gray))
    lbp_ent, lbp_uniform_ratio = _lbp_riu2_entropy(gray)
    feats["lbp_ent"] = float(lbp_ent)
    feats["lbp_uniform_ratio"] = float(lbp_uniform_ratio)
    feats["local_entropy_mean"] = float(_local_entropy(gray, k=15))
    feats["local_entropy_std"]  = float(_local_entropy(gray, k=25))

    feats.update(_wavelet_haar_energies(gray, levels=3))
    return feats

def dict_to_vec(d: Dict[str, float], keys: List[str]) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float32)

def aggregate_tile_features(tile_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not tile_dicts:
        return {}
    base_keys = sorted(tile_dicts[0].keys())
    mat = np.stack([dict_to_vec(td, base_keys) for td in tile_dicts], axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    out = {}
    for i, k in enumerate(base_keys):
        out[f"tile_mean_{k}"] = float(mean[i])
        out[f"tile_std_{k}"]  = float(std[i])
    return out

def compute_feature_vector(global_rgb: np.ndarray, tiles_sorted: List[np.ndarray], feat_keys: List[str]) -> np.ndarray:
    g_feats = extract_feature_dict(global_rgb)
    feats = {f"g_{k}": v for k, v in g_feats.items()}

    if cfg.TOP_TILES_FOR_FEATURES > 0 and len(tiles_sorted) > 0:
        top_tiles = tiles_sorted[:max(1, min(cfg.TOP_TILES_FOR_FEATURES, len(tiles_sorted)))]
        t_dicts = [extract_feature_dict(t) for t in top_tiles]
        feats.update(aggregate_tile_features(t_dicts))

    return dict_to_vec(feats, feat_keys)

# =========================
# Tile quality
# =========================
import math
def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    valid_ratio = float(valid.mean())
    if valid_ratio < 0.10:
        return 0.0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lapv = float(lap[valid].var()) if valid.any() else 0.0
    contrast = float(gray[valid].std()) if valid.any() else 0.0
    loc_ent = float(_local_entropy(gray, k=15))
    lbp_ent, uniform_ratio = _lbp_riu2_entropy(gray)
    hf_ratio = float(_fft_high_freq_ratio(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges[valid].mean() / 255.0) if valid.any() else 0.0

    uniform_pen = 1.0 - 0.5 * float(uniform_ratio)
    score = (
        0.20 * math.log1p(lapv) +
        0.10 * math.log1p(contrast) +
        0.25 * loc_ent +
        0.20 * float(lbp_ent) +
        0.15 * hf_ratio +
        0.10 * edge_density
    )
    score = score * (0.5 + 0.5 * valid_ratio) * uniform_pen
    return float(max(0.0, score))

# =========================
# Polar tiles: tiles from limbus wedge (coords in full 512)
# =========================
def polar_tiles_from_limbus(global_masked_512: np.ndarray, limbus_mask_512: np.ndarray):
    H, W = limbus_mask_512.shape[:2]

    ys, xs = np.where(limbus_mask_512 > 0)
    if xs.size == 0:
        return [], np.zeros((0,), dtype=np.float32), []

    cx, cy = float(xs.mean()), float(ys.mean())
    rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if rmax < 5:
        return [], np.zeros((0,), dtype=np.float32), []

    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    ring_edges = [f * rmax for f in cfg.RING_EDGES_FRAC]

    tiles, q, coords = [], [], []
    for r in range(cfg.POLAR_RINGS):
        for s in range(cfg.POLAR_THETA):
            t0, t1 = 2 * np.pi * s / cfg.POLAR_THETA, 2 * np.pi * (s + 1) / cfg.POLAR_THETA
            wedge = (limbus_mask_512 > 0) & (rr >= ring_edges[r]) & (rr < ring_edges[r + 1]) & (tt >= t0) & (tt < t1)
            if wedge.sum() < cfg.POLAR_MIN_PIXELS:
                continue

            ys_w, xs_w = np.where(wedge)
            x0, y0, x1, y1 = xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()
            x0, y0 = max(0, x0 - cfg.POLAR_PAD), max(0, y0 - cfg.POLAR_PAD)
            x1, y1 = min(W - 1, x1 + cfg.POLAR_PAD), min(H - 1, y1 + cfg.POLAR_PAD)

            tile = global_masked_512[y0:y1+1, x0:x1+1].copy()
            w = wedge[y0:y1+1, x0:x1+1].astype(np.uint8)
            tile[w == 0] = 0

            tile_res = cv2.resize(tile, (cfg.TILE_SIZE, cfg.TILE_SIZE), interpolation=cv2.INTER_AREA)
            tiles.append(tile_res)
            q.append(tile_quality_score(tile_res))
            coords.append((x0, y0, x1, y1))

    if not tiles:
        return [], np.zeros((0,), dtype=np.float32), []

    q = np.array(q, dtype=np.float32)
    order = np.argsort(-q)
    tiles = [tiles[i] for i in order.tolist()]
    coords = [coords[i] for i in order.tolist()]
    q = q[order]
    return tiles, q, coords

# =========================
# Loading models + feature keys
# =========================
@st.cache_resource
def load_everything():
    if not os.path.exists(cfg.FEATURE_KEYS_JSON):
        raise FileNotFoundError(f"feature_keys.json not found: {cfg.FEATURE_KEYS_JSON}")

    with open(cfg.FEATURE_KEYS_JSON, "r", encoding="utf-8") as f:
        feat_keys = json.load(f)
    if not isinstance(feat_keys, list) or len(feat_keys) < 10:
        raise ValueError("feature_keys.json is invalid or empty")

    seg_model, idx_crop, idx_limbus, img_size = load_model(cfg.SEG_CKPT, DEVICE)
    seg_model.eval()

    ckpt = torch.load(cfg.CLS_CKPT, map_location=DEVICE)
    cfg_ck = ckpt.get("cfg", {})
    state = ckpt["model"] if "model" in ckpt else ckpt

    topk_pool = int(cfg_ck.get("TOPK_POOL", cfg.TOPK_POOL_DEFAULT))
    quality_beta = float(cfg_ck.get("QUALITY_BETA", cfg.QUALITY_BETA_DEFAULT))
    use_3class = bool(cfg_ck.get("USE_3CLASS_MERGE", False))
    classes = ("NonInfect_Other", "Normal", "Infection") if use_3class else cfg.CLASSES_4

    model = DualBranchMIL_WithFeatures(num_classes=len(classes), topk_pool=topk_pool, quality_beta=quality_beta, feat_dim=len(feat_keys))
    model.load_state_dict(state, strict=True)
    model.float().to(DEVICE).eval()

    return seg_model, idx_crop, idx_limbus, img_size, model, classes, feat_keys, topk_pool, quality_beta

# =========================
# UI
# =========================
st.set_page_config(page_title="KeratitisAI - Expert System", layout="wide", page_icon="👁️")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .status-card { padding: 6px 12px; border-radius: 10px; text-align: center; color: white; font-weight: 700;
                   font-size: 18px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                   width: fit-content; margin-left: auto; margin-right: auto; min-width: 140px; }
</style>
""", unsafe_allow_html=True)

try:
    with st.spinner("Initializing Deep Learning Engine..."):
        seg_model, idx_crop, idx_limbus, img_size, clf_model, CLASSES, FEAT_KEYS, CK_TOPK, CK_QB = load_everything()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

st.title("🛡️ KeratitisAI Diagnostic Dashboard")
st.markdown("**State-of-the-Art Keratitis Diagnostics: Deep Fusion of Global ROI and Localized Patch Evidence**")

st.sidebar.title("🩺 Clinical Control")
uploaded_file = st.sidebar.file_uploader("Upload Slit Lamp Photography", type=["jpg", "jpeg", "png"])
st.sidebar.divider()
st.sidebar.subheader("Parameters")
topk_val = st.sidebar.slider("Attention Top-K", 1, 12, int(CK_TOPK))
quality_beta = st.sidebar.slider("Quality Bias (β)", 0.0, 3.0, float(CK_QB), step=0.1)
seg_thresh = st.sidebar.slider("Segmentation Threshold", 0.1, 0.9, 0.5, step=0.05)
fast_mode = st.sidebar.toggle("Fast Mode (no tile features)", value=False)
if fast_mode:
    cfg.TOP_TILES_FOR_FEATURES = 0

st.sidebar.caption(f"Device: {DEVICE} | feat_dim: {len(FEAT_KEYS)}")

if not uploaded_file:
    st.info("Upload an image from the sidebar to run inference.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
bgr_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr_orig is None:
    st.error("Could not decode image.")
    st.stop()

# segmentation
with st.spinner("Segmenting crop + limbus..."):
    masks = predict_masks(seg_model, bgr_orig, img_size, DEVICE, thresh=float(seg_thresh))
    m_crop = masks[idx_crop].astype(np.uint8)
    m_limb = masks[idx_limbus].astype(np.uint8)

# canonical space
bgr_512 = cv2.resize(bgr_orig, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_AREA)
rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)

crop_512 = cv2.resize(m_crop, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
limb_512 = cv2.resize(m_limb, (cfg.CANONICAL_SIZE, cfg.CANONICAL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

# smooth crop -> rectangle mask
crop_512 = postprocess_binary(crop_512, k_close=21, k_open=9)
crop_rect_512 = rect_mask_from_crop(crop_512, pad=cfg.CROP_RECT_PAD, min_area_frac=cfg.CROP_MIN_AREA_FRAC)

# fallback if crop missing but limbus exists
if crop_rect_512.sum() < 50 and limb_512.sum() > 50:
    crop_rect_512 = limb_512.copy()

# ---- GLOBAL BRANCH INPUT: REAL CROPPED ROI ----
roi_rgb_512, roi_box = crop_roi_by_largest_rect(
    rgb_512,
    crop_rect_512,
    pad=0,  # crop_rect_512 already padded
    min_area_frac=cfg.CROP_MIN_AREA_FRAC
)

if roi_rgb_512 is None:
    global_rgb_for_model = apply_mask_rgb(rgb_512, crop_rect_512)
else:
    global_rgb_for_model = roi_rgb_512

# ---- TILE EXTRACTION: still on full 512 masked canvas so coords remain correct ----
global_masked_full_512 = apply_mask_rgb(rgb_512, crop_rect_512)
tiles_sorted, q_sorted, coords_sorted = polar_tiles_from_limbus(global_masked_full_512, limb_512)

# pad to 24
tiles_pad = tiles_sorted[:cfg.MAX_TILES]
q_pad = q_sorted[:cfg.MAX_TILES]
coords_pad = coords_sorted[:cfg.MAX_TILES]
if len(tiles_pad) < cfg.MAX_TILES:
    pad_n = cfg.MAX_TILES - len(tiles_pad)
    tiles_pad += [np.zeros((cfg.TILE_SIZE, cfg.TILE_SIZE, 3), dtype=np.uint8)] * pad_n
    q_pad = np.concatenate([q_pad, np.zeros((pad_n,), dtype=np.float32)], axis=0)
    coords_pad += [None] * pad_n

q01 = normalize01(q_pad)

# tensors
g_rs = resize_rgb(global_rgb_for_model, (cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE))
g_t = to_tensor_norm(g_rs).unsqueeze(0).to(DEVICE)
t_t = torch.stack([to_tensor_norm(t) for t in tiles_pad]).unsqueeze(0).to(DEVICE)
q_t = torch.from_numpy(q01.astype(np.float32)).unsqueeze(0).to(DEVICE)

# features: use GLOBAL ROI (cropped) for global features, and tiles_sorted for tile features
feat_vec = compute_feature_vector(global_rgb_for_model if roi_rgb_512 is not None else global_masked_full_512, tiles_sorted, FEAT_KEYS)
f_t = torch.from_numpy(feat_vec.astype(np.float32)).unsqueeze(0).to(DEVICE)

# runtime params
clf_model.topk_pool = int(topk_val)
clf_model.quality_beta = float(quality_beta)

# inference
with st.spinner("Running MIL diagnosis..."):
    with torch.no_grad():
        with autocast(enabled=(DEVICE == "cuda")):
            logits, att, top_idx = clf_model(g_t, t_t, f_t, qnorm=q_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

pred_id = int(np.argmax(probs))
pred_label = CLASSES[pred_id]
conf = float(probs[pred_id])

# UI output
col_main, col_res = st.columns([1.25, 1])

with col_main:
    st.subheader("Anatomical Visualization")
    tabs = st.tabs(["Raw Input", "Global ROI", "Slice + Attention", "Attention Boxes"])

    tabs[0].image(cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Raw Input")
    tabs[1].image(global_rgb_for_model, use_container_width=True, caption="Global ROI (cropped from crop contour bbox)")

    top_indices = top_idx[0].detach().cpu().numpy().tolist()

    with tabs[2]:
        st.markdown(f"**Slices generated:** {len(tiles_sorted)} (padded to {cfg.MAX_TILES})")
        cols = st.columns(6)
        for i in range(cfg.MAX_TILES):
            with cols[i % 6]:
                star = " ⭐" if i in top_indices else ""
                st.image(tiles_pad[i], caption=f"T{i+1}{star}", use_container_width=True)

    with tabs[3]:
        att_vis = global_masked_full_512.copy()
        for ti in top_indices:
            box = coords_pad[ti] if (ti < len(coords_pad)) else None
            if box is None:
                continue
            x0, y0, x1, y1 = box
            cv2.rectangle(att_vis, (x0, y0), (x1, y1), (255, 0, 0), 3)
        tabs[3].image(att_vis, use_container_width=True, caption="Top-K attention boxes (red)")

with col_res:
    st.subheader("Diagnostic Status")
    bg_color = cfg.CLASS_COLORS.get(pred_label, "#34495e")
    st.markdown(f'<div class="status-card" style="background-color: {bg_color};">{pred_label.upper()}</div>', unsafe_allow_html=True)

    st.markdown("### Confidence Analysis")
    tbl_html = """<table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 6px; border: 1px solid #ddd; text-align: left;">Condition</th>
            <th style="padding: 6px; border: 1px solid #ddd; text-align: right;">Probability</th>
        </tr>"""
    for i, cls in enumerate(CLASSES):
        p_str = f"{(probs[i]*100):.1f}%"
        is_pred = (i == pred_id)
        bg = "#d4edda" if is_pred else "#ffffff"
        weight = "700" if is_pred else "400"
        color = cfg.CLASS_COLORS.get(cls, "#333") if is_pred else "#333"
        tbl_html += f"""
        <tr style="background-color: {bg}; font-weight: {weight};">
            <td style="padding: 6px; border: 1px solid #ddd; color: {color};">{cls}{" (Pred)" if is_pred else ""}</td>
            <td style="padding: 6px; border: 1px solid #ddd; text-align: right;">{p_str}</td>
        </tr>"""
    tbl_html += "</table>"
    st.write(tbl_html, unsafe_allow_html=True)

    st.metric("Predicted Confidence", f"{conf*100:.1f}%")
    st.metric("Slice Used", cfg.MAX_TILES)
    st.metric("Attention Top-K", int(topk_val))

st.divider()
st.subheader("👁️ AI Hotspots (Top Attention Slices)")
top_indices = top_idx[0].detach().cpu().numpy().tolist()
cols_att = st.columns(4)
for i, ti in enumerate(top_indices[:4]):
    with cols_att[i]:
        score = float(att[0, ti].detach().cpu().item())
        st.image(tiles_pad[ti], caption=f"Rank {i+1} | att={score:.3f}", use_container_width=True)
