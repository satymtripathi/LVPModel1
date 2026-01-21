import os, glob, json, random, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ---- CRITICAL WINDOWS STABILITY ----
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    CACHE_ROOT: str = r"DatasetSatyam_Precomputed1"
    OUT_DIR: str = r"training_results_v4"

    USE_3CLASS_MERGE: bool = False

    # sizes
    GLOBAL_SIZE: int = 384
    TILE_SIZE: int = 224
    MAX_TILES: int = 24

    # training
    EPOCHS: int = 20
    BATCH_SIZE: int = 2
    GRAD_ACCUM: int = 4
    BASE_LR: float = 2e-5
    WEIGHT_DECAY: float = 1e-4
    SEED: int = 42

    # MIL
    TOPK_POOL: int = 4
    QUALITY_BETA: float = 0.7
    TILE_DROPOUT: float = 0.10

    # loss
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTH: float = 0.05

    # entropy warmup
    LAMBDA_ATT_ENT_START: float = 0.03
    LAMBDA_ATT_ENT_AFTER: float = 0.01
    ENT_WARM_EPOCHS: int = 5

    # weights
    W4: Tuple[float, float, float, float] = (1.0, 1.0, 3.0, 1.5)
    W3: Tuple[float, float, float] = (1.0, 1.4, 2.2)

    # DataLoader
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = False

    # speed
    CUDNN_BENCHMARK: bool = True
    LIMIT_TILES_READ: int = 24

    # features
    FEATURE_FROM: str = "global+top_tiles"  # "global_only" | "global+top_tiles"
    TOP_TILES_FOR_FEATURES: int = 6         # aggregate features from top-K quality tiles
    SAVE_FEATURES_JSON: bool = True         # writes per-sample features to disk (for debug)

    # resume
    RESUME: bool = True

cfg = CFG()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda" and cfg.CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True


# =========================
# REPRO
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# LABEL MAP
# =========================
ORIG_4 = ["Edema", "Scar", "Infection", "Normal"]
if cfg.USE_3CLASS_MERGE:
    CLASSES = ["NonInfect_Other", "Normal", "Infection"]
    def map_label(lbl: str) -> str:
        return "NonInfect_Other" if lbl in ["Edema", "Scar"] else lbl
else:
    CLASSES = ORIG_4
    def map_label(lbl: str) -> str:
        return lbl

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}


# =========================
# FAST tensor preprocess (NO PIL)
# =========================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_rgb(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)

def to_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    return t

def maybe_aug(rgb: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        rgb = cv2.flip(rgb, 1)
    return rgb


# =========================
# CLINICAL TILE SCORING (NEW)
# =========================
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

def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    valid_ratio = float(valid.mean())
    if valid_ratio < 0.10:
        return 0.0
    g_valid = gray[valid]
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lapv = float(lap[valid].var()) if valid.any() else 0.0
    contrast = float(g_valid.std()) if g_valid.size > 0 else 0.0
    loc_ent = _local_entropy_map(gray, k=15)
    lbp_ent, uniform_ratio = _lbp_riu2_entropy(gray)
    hf_ratio = _fft_high_freq_ratio(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges[valid].mean() / 255.0) if valid.any() else 0.0
    uniform_pen = 1.0 - 0.5 * uniform_ratio
    score = (
        0.20 * math.log1p(lapv) +
        0.10 * math.log1p(contrast) +
        0.25 * loc_ent +
        0.20 * lbp_ent +
        0.15 * hf_ratio +
        0.10 * edge_density
    )
    score = score * (0.5 + 0.5 * valid_ratio) * uniform_pen
    return float(max(0.0, score))


# =========================
# FEATURE ENGINE (NO skimage)
# =========================
def _safe_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())

def glcm_contrast_approx(gray: np.ndarray, levels: int = 32) -> float:
    # simple 0-degree, distance=1 contrast approximation (fast)
    g = (gray.astype(np.int32) * (levels - 1) // 255).astype(np.int32)
    a = g[:, :-1]
    b = g[:, 1:]
    # contrast = E[(i-j)^2]
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

    # peak freq radius (excluding center)
    rr_flat = rr.ravel()
    mag_flat = mag_log.ravel()
    # ignore DC neighborhood
    mask = rr_flat > 3
    if mask.sum() < 10:
        return 0.0, 0.0

    peak_idx = np.argmax(mag_flat[mask])
    peak_r = float(rr_flat[mask][peak_idx])

    # high frequency ratio
    rmax = rr.max() + 1e-8
    high = mag_log[rr > 0.6 * rmax].sum()
    total = mag_log.sum() + 1e-8
    return peak_r, float(high / total)

def lbp_riu2_features(gray: np.ndarray) -> Tuple[float, float, float, float]:
    # simple LBP(8,1) without skimage
    g = gray.astype(np.uint8)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0, 0.0, 0.0, 0.0

    c = g[1:-1, 1:-1]
    # neighbors clockwise starting at top-left
    n0 = g[0:-2, 0:-2]
    n1 = g[0:-2, 1:-1]
    n2 = g[0:-2, 2:  ]
    n3 = g[1:-1, 2:  ]
    n4 = g[2:  , 2:  ]
    n5 = g[2:  , 1:-1]
    n6 = g[2:  , 0:-2]
    n7 = g[1:-1, 0:-2]

    bits = np.stack([(n0 >= c), (n1 >= c), (n2 >= c), (n3 >= c),
                     (n4 >= c), (n5 >= c), (n6 >= c), (n7 >= c)], axis=-1).astype(np.uint8)

    # count transitions for uniformity
    trans = np.sum(bits[..., :-1] != bits[..., 1:], axis=-1) + (bits[..., 0] != bits[..., -1]).astype(np.uint8)

    # uniform patterns: trans <= 2 => map to number of 1s (0..8), else -> 9
    ones = bits.sum(axis=-1)
    lbp = np.where(trans <= 2, ones, 9).astype(np.int32)

    hist = np.bincount(lbp.ravel(), minlength=10).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    # stats
    # variance of codes
    codes = np.arange(10, dtype=np.float64)
    mean = (p * codes).sum()
    var = (p * (codes - mean) ** 2).sum()
    ent = float(-(p[p > 1e-12] * np.log2(p[p > 1e-12])).sum())
    uniform_ratio = float(p[:9].sum())
    energy = float((p ** 2).sum())
    return float(var), ent, uniform_ratio, energy

def local_entropy_stats(gray: np.ndarray, k: int = 15) -> Tuple[float, float]:
    # approximate: entropy of local mean & local std maps (fast)
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sqr_mean = cv2.blur(g * g, (k, k))
    std = np.sqrt(np.maximum(0.0, sqr_mean - mean * mean))

    # treat mean map as "image" and entropy it
    mean_u8 = np.clip(mean, 0, 255).astype(np.uint8)
    std_u8 = np.clip(std, 0, 255).astype(np.uint8)
    return grayscale_entropy(mean_u8), grayscale_entropy(std_u8)

def edge_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)
    laplacian_density = float((lap_abs > 20).mean())

    # simple contour perimeter pixels
    cnts, _ = cv2.findContours((edges > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = 0.0
    for c in cnts[:50]:
        perim += cv2.arcLength(c, True)
    return edge_density, laplacian_density, float(perim)

def wavelet_haar_energies(gray: np.ndarray, levels: int = 3) -> Dict[str, float]:
    # Haar wavelet energy without pywt. We do simple 2x2 pooling operations.
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
    if a_ch.size > 10:
        a_b_cov = float(np.cov(a_ch, b_ch)[0, 1])
    else:
        a_b_cov = 0.0

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
    hist_mean = float(gray.mean())
    hist_std = float(gray.std())
    hist_ent = grayscale_entropy(gray)
    glcm_cont = glcm_contrast_approx(gray)
    return {
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "hist_ent": hist_ent,
        "glcm_cont": glcm_cont,
    }

def compute_texture_features(gray: np.ndarray) -> Dict[str, float]:
    fft_peak_freq, fft_high_freq_ratio = fft_features(gray)
    lbp_var, lbp_ent, lbp_uniform_ratio, lbp_energy = lbp_riu2_features(gray)
    edge_density, laplacian_density, perimeter_pixels = edge_metrics(gray)
    local_ent_mean, local_ent_std = local_entropy_stats(gray, k=15)

    # "localized glcm" proxy: compute contrast on blurred & original as different scales
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
    # Since you have only masked/cropped ROI in _global.jpg, we estimate shape from non-black area
    mask = (gray > 5).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size < 20:
        return {"pos_central": 0.0, "round": 0.0, "el": 0.0, "solidity": 0.0}

    h, w = gray.shape
    cx, cy = xs.mean(), ys.mean()
    # centrality: distance of centroid from image center (normalized, inverted)
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

def extract_feature_vector_from_rgb(rgb: np.ndarray) -> Dict[str, float]:
    gray = _safe_gray(rgb)

    feats = {}
    feats.update(compute_gray_stats(gray))
    feats.update(compute_colour_stats(rgb))
    feats.update(compute_texture_features(gray))
    feats.update(compute_pos_shape_proxy(gray))
    feats.update(wavelet_haar_energies(gray, levels=3))

    
    return feats

def dict_to_feature_vec(d: Dict[str, float], keys: List[str]) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float32)

def aggregate_tile_features(tile_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    # mean + std aggregation over tiles
    if not tile_dicts:
        return {}
    keys = sorted(tile_dicts[0].keys())
    mat = np.stack([dict_to_feature_vec(td, keys) for td in tile_dicts], axis=0)  # (T,F)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    out = {}
    for i, k in enumerate(keys):
        out[f"tile_mean_{k}"] = float(mean[i])
        out[f"tile_std_{k}"] = float(std[i])
    return out


# =========================
# INDEX BUILD
# =========================
def find_global_files(cache_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(cache_root, "**", "*_global.jpg"), recursive=True))

def parse_sample_from_global(global_path: str) -> Optional[dict]:
    try:
        cls = os.path.basename(os.path.dirname(global_path))
        base = global_path[:-len("_global.jpg")]
        meta_path = base + "_meta.json"

        tiles, quality = [], []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for m in meta:
                tile_file = m.get("file", "")
                if not tile_file:
                    continue
                tile_path = os.path.join(os.path.dirname(global_path), tile_file)
                if os.path.exists(tile_path):
                    tiles.append(tile_path)
                    quality.append(float(m.get("quality", 0.0)))
        else:
            tiles = sorted(glob.glob(base + "_tile_*.jpg"))
            quality = [0.0] * len(tiles)

        return {
            "label": cls,
            "global_path": global_path,
            "tiles": tiles[:cfg.LIMIT_TILES_READ],
            "quality": np.array(quality[:cfg.LIMIT_TILES_READ], dtype=np.float32),
            "sample_id": os.path.basename(base),
            "base": base,
        }
    except Exception:
        return None

def build_index(cache_root: str) -> List[dict]:
    globals_ = find_global_files(cache_root)
    out = []
    for gp in tqdm(globals_, desc="Indexing cache"):
        s = parse_sample_from_global(gp)
        if s is not None:
            out.append(s)
    return out


# =========================
# FEATURE CACHE (computed once per sample)
# =========================
def compute_sample_features(sample: dict) -> Dict[str, float]:
    g_rgb = _read_rgb(sample["global_path"])
    if g_rgb is None:
        return {}

    feats_global = extract_feature_vector_from_rgb(g_rgb)

    if cfg.FEATURE_FROM == "global_only":
        return feats_global

    # select top tiles by quality for features
    tiles = list(sample["tiles"])
    
    # Re-score tiles using new clinical criteria (overriding JSON quality)
    q_vals = []
    tile_rgbs_cached = []
    for tp in tiles:
        trgb = _read_rgb(tp)
        tile_rgbs_cached.append(trgb)
        q_vals.append(tile_quality_score(trgb) if trgb is not None else 0.0)
    
    q = np.array(q_vals, dtype=np.float32)
    if len(tiles) == 0:
        return feats_global

    order = np.argsort(-q)
    topk = order[:min(cfg.TOP_TILES_FOR_FEATURES, len(tiles))]

    tile_dicts = []
    for i in topk:
        trgb = tile_rgbs_cached[int(i)]
        if trgb is None:
            continue
        tile_dicts.append(extract_feature_vector_from_rgb(trgb))

    feats_tiles = aggregate_tile_features(tile_dicts)
    feats = {}
    feats.update({f"g_{k}": v for k, v in feats_global.items()})
    feats.update(feats_tiles)
    return feats


# =========================
# DATASET
# =========================
class PrecomputedFolderDataset(Dataset):
    def __init__(self, samples: List[dict], features_by_id: Dict[str, np.ndarray], train: bool):
        self.samples = samples
        self.train = train
        self.features_by_id = features_by_id

        self.black_tile = np.zeros((cfg.TILE_SIZE, cfg.TILE_SIZE, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            label_str = map_label(str(s["label"]))
            y = CLASS_TO_ID[label_str]

            g_rgb = _read_rgb(s["global_path"])
            if g_rgb is None:
                return None

            if self.train:
                g_rgb = maybe_aug(g_rgb)

            g_rgb = resize_rgb(g_rgb, (cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE))
            g_t = to_tensor_norm(g_rgb)

            tile_paths = list(s["tiles"])
            q = np.array(s["quality"], dtype=np.float32)

            # normalize quality 0..1
            if q.size > 0:
                mn, mx = float(q.min()), float(q.max())
                q01 = (q - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(q)
            else:
                q01 = np.zeros((len(tile_paths),), dtype=np.float32)

            # tile dropout
            idxs = np.arange(len(tile_paths))
            if self.train and cfg.TILE_DROPOUT > 0 and len(idxs) > 6:
                keep_n = max(6, int(len(idxs) * (1 - cfg.TILE_DROPOUT)))
                idxs = np.random.choice(idxs, size=keep_n, replace=False)
                tile_paths = [tile_paths[i] for i in idxs]
                q01 = q01[idxs]

            # load tiles (up to MAX_TILES)
            tile_rgbs, loaded_paths = [], []
            for tp in tile_paths[:cfg.MAX_TILES]:
                trgb = _read_rgb(tp)
                if trgb is None:
                    continue
                trgb = resize_rgb(trgb, (cfg.TILE_SIZE, cfg.TILE_SIZE))
                tile_rgbs.append(trgb)
                loaded_paths.append(tp)

            q01 = q01[:len(loaded_paths)] if len(loaded_paths) > 0 else np.zeros((0,), dtype=np.float32)

            # pad to MAX_TILES
            T = cfg.MAX_TILES
            if len(tile_rgbs) < T:
                pad_n = T - len(tile_rgbs)
                tile_rgbs += [self.black_tile] * pad_n
                q01 = np.concatenate([q01, np.zeros((pad_n,), dtype=np.float32)], axis=0)
                loaded_paths += [None] * pad_n
            else:
                tile_rgbs = tile_rgbs[:T]
                q01 = q01[:T]
                loaded_paths = loaded_paths[:T]

            tiles_t = torch.stack([to_tensor_norm(t) for t in tile_rgbs])  # (T,3,224,224)
            q_t = torch.from_numpy(q01.astype(np.float32))                 # (T,)

            # features
            sid = s["sample_id"]
            fvec = self.features_by_id.get(sid, None)
            if fvec is None:
                # fallback zeros
                fvec = np.zeros((1,), dtype=np.float32)
            f_t = torch.from_numpy(fvec.astype(np.float32))

            return g_t, tiles_t, q_t, f_t, y, sid, s["global_path"], loaded_paths

        except Exception:
            return None


def collate_precomputed(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    g, tiles, q, f, y, sid, gpath, tpaths = zip(*batch)
    # pad features to same dim (safety)
    maxF = max(int(x.numel()) for x in f)
    f_pad = []
    for x in f:
        if x.numel() == maxF:
            f_pad.append(x)
        else:
            z = torch.zeros(maxF, dtype=torch.float32)
            z[:x.numel()] = x.view(-1)
            f_pad.append(z)
    return (
        torch.stack(g),
        torch.stack(tiles),
        torch.stack(q),
        torch.stack([x.view(-1) for x in f_pad]),
        torch.tensor(y, dtype=torch.long),
        sid, gpath, tpaths
    )


# =========================
# MODEL
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

        # tabular feature branch
        self.feat_dim = feat_dim
        self.feat_latent = 128
        self.feat_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
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
        """
        global_img: (B,3,G,G)
        tiles: (B,T,3,224,224)
        feats: (B,F)
        qnorm: (B,T)
        """
        B, T, C, H, W = tiles.shape
        tiles_ = tiles.view(B*T, C, H, W)

        t = self.tile_pool(self.tile_feat(tiles_))   # (B*T,1280,1,1)
        z = self.tile_projector(t).view(B, T, -1)    # (B,T,256)

        a_logits = self.attention(z).squeeze(-1)     # (B,T)
        if qnorm is not None:
            a_logits = a_logits + self.quality_beta * qnorm.to(a_logits.device)

        a = torch.softmax(a_logits, dim=1)
        k = min(self.topk_pool, T)
        top_idx = torch.topk(a, k=k, dim=1).indices

        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
        z_top = torch.gather(z, dim=1, index=idx_exp)  # (B,k,256)
        bag = z_top.mean(dim=1)                        # (B,256)

        g = self.global_pool(self.global_feat(global_img))
        g = self.global_projector(g)                   # (B,256)

        f = self.feat_mlp(feats)                       # (B,128)

        fused = torch.cat([bag, g, f], dim=1)
        logits = self.classifier(fused)
        return logits, a, top_idx


# =========================
# LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smooth=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smooth = label_smooth

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smooth
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

def attention_entropy_batch(a, eps=1e-8):
    p = a.clamp(min=eps)
    return (-(p * torch.log(p)).sum(dim=1)).mean()


# =========================
# CHECKPOINTING
# =========================
def save_ckpt(path: str, epoch: int, model, optimizer, scaler, best_acc: float, feat_keys: List[str]):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
        "cfg": cfg.__dict__,
        "feat_keys": feat_keys
    }, path)

def load_ckpt(path: str, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    feat_keys = ckpt.get("feat_keys", [])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_acc", 0.0)), feat_keys


# =========================
# MAIN
# =========================
def main():
    seed_everything(cfg.SEED)

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUT_DIR, "checkpoints"), exist_ok=True)

    # ---- index ----
    samples = build_index(cfg.CACHE_ROOT)
    if len(samples) == 0:
        raise RuntimeError(f"No globals in {cfg.CACHE_ROOT}")

    # ---- compute features ONCE ----
    print("🧠 Computing features (once)...")
    feats_raw_by_id: Dict[str, Dict[str, float]] = {}
    for s in tqdm(samples, desc="Feature extract"):
        sid = s["sample_id"]
        feats = compute_sample_features(s)
        feats_raw_by_id[sid] = feats

    # build stable feature key order
    all_keys = set()
    for d in feats_raw_by_id.values():
        for k in d.keys():
            all_keys.add(k)
    feat_keys = sorted(list(all_keys))
    feat_dim = len(feat_keys)
    print(f"✅ Feature dim = {feat_dim}")

    # vectorize
    features_by_id: Dict[str, np.ndarray] = {}
    for sid, d in feats_raw_by_id.items():
        features_by_id[sid] = dict_to_feature_vec(d, feat_keys)

    if cfg.SAVE_FEATURES_JSON:
        with open(os.path.join(cfg.OUT_DIR, "feature_keys.json"), "w") as f:
            json.dump(feat_keys, f, indent=2)

    # ---- stratified split ----
    labels = np.array([CLASS_TO_ID[map_label(str(s["label"]))] for s in samples], dtype=np.int64)
    tr_samples, va_samples = train_test_split(
        samples, test_size=0.2, random_state=cfg.SEED, stratify=labels
    )
    epochs = cfg.EPOCHS
    print(f"train={len(tr_samples)} val={len(va_samples)} epochs={epochs}")

    # ---- class-balanced sampling ----
    tr_y = np.array([CLASS_TO_ID[map_label(str(s["label"]))] for s in tr_samples], dtype=np.int64)
    counts = {i: max(1, int((tr_y == i).sum())) for i in range(len(CLASSES))}
    sample_w = np.array([1.0 / counts[int(y)] for y in tr_y], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    # ---- datasets ----
    train_ds = PrecomputedFolderDataset(tr_samples, features_by_id, train=True)
    val_ds = PrecomputedFolderDataset(va_samples, features_by_id, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
        collate_fn=collate_precomputed
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
        collate_fn=collate_precomputed
    )

    model = DualBranchMIL_WithFeatures(len(CLASSES), cfg.TOPK_POOL, cfg.QUALITY_BETA, feat_dim=feat_dim).to(DEVICE)

    alpha = torch.tensor(cfg.W3 if cfg.USE_3CLASS_MERGE else cfg.W4, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(cfg.FOCAL_GAMMA, alpha=alpha, label_smooth=cfg.LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    ckpt_dir = os.path.join(cfg.OUT_DIR, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "last.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    start_epoch = 1
    best_acc = 0.0
    if cfg.RESUME and os.path.exists(last_ckpt):
        try:
            start_epoch, best_acc, old_keys = load_ckpt(last_ckpt, model, optimizer, scaler)
            # if feature keys differ, resume is unsafe (dims mismatch)
            if old_keys and old_keys != feat_keys:
                print("[RESUME] feature keys changed -> restarting from epoch 1")
                start_epoch, best_acc = 1, 0.0
        except Exception as e:
            print("[RESUME] failed:", e)

    with open(os.path.join(cfg.OUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # ---- train loop ----
    for epoch in range(start_epoch, epochs + 1):
        lambda_ent = cfg.LAMBDA_ATT_ENT_START if epoch <= cfg.ENT_WARM_EPOCHS else cfg.LAMBDA_ATT_ENT_AFTER

        model.train()
        running, steps, none_batches = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for bi, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{epochs}")):
            if batch is None:
                none_batches += 1
                continue

            gimgs, tiles_b, q_b, feats_b, y_b, *_ = batch
            gimgs = gimgs.to(DEVICE, non_blocking=True)
            tiles_b = tiles_b.to(DEVICE, non_blocking=True)
            q_b = q_b.to(DEVICE, non_blocking=True)
            feats_b = feats_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits, att, _ = model(gimgs, tiles_b, feats_b, qnorm=q_b)
                cls_loss = criterion(logits, y_b)
                ent = attention_entropy_batch(att)
                loss = (cls_loss - lambda_ent * ent) / cfg.GRAD_ACCUM

            scaler.scale(loss).backward()

            if ((bi + 1) % cfg.GRAD_ACCUM == 0):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * cfg.GRAD_ACCUM
            steps += 1

        if (steps % cfg.GRAD_ACCUM) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / max(1, steps)
        print(f"Train Loss: {train_loss:.4f} | None: {none_batches}")

        # ---- val ----
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{epochs}"):
                if batch is None:
                    continue
                gimgs, tiles_b, q_b, feats_b, y_b, *_ = batch
                gimgs = gimgs.to(DEVICE, non_blocking=True)
                tiles_b = tiles_b.to(DEVICE, non_blocking=True)
                q_b = q_b.to(DEVICE, non_blocking=True)
                feats_b = feats_b.to(DEVICE, non_blocking=True)

                with autocast(enabled=(DEVICE == "cuda")):
                    logits, _, _ = model(gimgs, tiles_b, feats_b, qnorm=q_b)

                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                ytrue = y_b.cpu().numpy().tolist()
                all_true.extend(ytrue)
                all_pred.extend(preds)

        all_true = np.array(all_true, dtype=np.int64)
        all_pred = np.array(all_pred, dtype=np.int64)

        val_acc = float((all_true == all_pred).mean())
        cm = confusion_matrix(all_true, all_pred)

        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(all_true, all_pred, target_names=CLASSES, digits=4, zero_division=0))

        with open(os.path.join(cfg.OUT_DIR, "metrics_log.txt"), "a") as f:
            f.write(f"epoch={epoch} train_loss={train_loss:.6f} val_acc={val_acc:.6f}\n")

        save_ckpt(last_ckpt, epoch, model, optimizer, scaler, best_acc, feat_keys)
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(best_ckpt, epoch, model, optimizer, scaler, best_acc, feat_keys)
            print(f"[BEST] saved best.pth | best_acc={best_acc:.4f}")

    print("\nDone. Best Acc:", best_acc)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
