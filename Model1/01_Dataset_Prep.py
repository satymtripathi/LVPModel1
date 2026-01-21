import sys
import os
import glob
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to sys.path to resolve imports when running from subfolders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Limbus_Crop_Segmentation_System.inference_utils import load_model

# =========================
# CONFIG
# =========================
DATA_ROOT = r"DatasetSatyam"
CHECKPOINT_PATH = r"Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth"
OUT_CACHE = "DatasetSatyam_Precomputed1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Edema", "Scar", "Infection", "Normal"]
BATCH_SIZE = 16
NUM_WORKERS = 4  # if Windows crashes, set to 0

# Tiling Config
POLAR_THETA = 8
POLAR_RINGS = 3
RING_EDGES_FRAC = [0.0, 0.35, 0.70, 1.0]
POLAR_MIN_PIXELS = 150
POLAR_PAD = 2
TILE_SAVE_SIZE = 224
CANONICAL_SIZE = 512

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# =========================
# UTILS
# =========================
def tile_quality_score(tile_rgb: np.ndarray) -> float:
    if tile_rgb is None or tile_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.mean() < 0.05:
        return 0.0
    g = gray[valid]
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lapv = float(lap[valid].var())
    contrast = float(g.std())
    glare_ratio = float((g > 240).mean())
    glare_pen = 1.0 - min(1.0, glare_ratio * 4.0)
    return float(max(0.0, (0.6 * math.log1p(lapv) + 0.4 * math.log1p(contrast)) * glare_pen))


def apply_mask_rgb(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    out[mask01 == 0] = 0
    return out


def _safe_to_numpy_rgb(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] == 3:
            # CHW -> HWC (uint8 likely already, but keep safe)
            x = x.permute(1, 2, 0)
        return x.numpy()
    return np.array(x)


# =========================
# DATASET
# =========================
class PrecomputeDataset(Dataset):
    def __init__(self, paths, img_size_hw):
        self.paths = paths
        self.img_size = img_size_hw  # (H,W)
        self.transform = A.Compose([
            A.Resize(img_size_hw[0], img_size_hw[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        bgr = cv2.imread(p)
        if bgr is None:
            rgb_512 = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            return tensor, p, rgb_512

        bgr_512 = cv2.resize(bgr, (self.img_size[1], self.img_size[0]))
        rgb_512 = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb_512)["image"]
        return tensor, p, rgb_512


# =========================
# WORKER: SAVE GLOBAL + TILES
# =========================
def process_and_save(p: str, rgb_512: np.ndarray, prob_crop: np.ndarray, prob_limbus: np.ndarray, save_root: str):
    """
    - global image uses idx_crop (rect around limbus)
    - tiles/meta use idx_limbus (true limbus segmentation) with same polar slicing
    """
    try:
        # 1) GLOBAL: RECT CROP MASK (idx_crop / class 0)
        mask_crop_raw = (prob_crop > 0.5).astype(np.uint8)
        
        # Make a proper rectangle mask (just like in inference_utils)
        mask_crop = np.zeros_like(mask_crop_raw)
        contours, _ = cv2.findContours(mask_crop_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            mask_crop[y:y+h, x:x+w] = 1
            
        masked_global = apply_mask_rgb(rgb_512, mask_crop)

        rel_path = os.path.relpath(p, DATA_ROOT)
        save_base = os.path.join(save_root, os.path.splitext(rel_path)[0])
        os.makedirs(os.path.dirname(save_base), exist_ok=True)

        cv2.imwrite(save_base + "_global.jpg", cv2.cvtColor(masked_global, cv2.COLOR_RGB2BGR))

        # 2) TILES: LIMBUS MASK (idx_limbus)
        mask_limbus = (prob_limbus > 0.5).astype(np.uint8)

        H, W = mask_limbus.shape
        ys, xs = np.where(mask_limbus > 0)
        if len(xs) == 0:
            with open(save_base + "_meta.json", "w") as f:
                json.dump([], f)
            return

        cx, cy = float(xs.mean()), float(ys.mean())
        rmax = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
        if rmax < 5:
            with open(save_base + "_meta.json", "w") as f:
                json.dump([], f)
            return

        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        tt = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
        ring_edges = [f * rmax for f in RING_EDGES_FRAC]

        meta_list = []
        for r in range(POLAR_RINGS):
            for s in range(POLAR_THETA):
                t0 = 2 * np.pi * s / POLAR_THETA
                t1 = 2 * np.pi * (s + 1) / POLAR_THETA

                wedge = (
                    (mask_limbus > 0) &
                    (rr >= ring_edges[r]) & (rr < ring_edges[r + 1]) &
                    (tt >= t0) & (tt < t1)
                )

                if wedge.sum() < POLAR_MIN_PIXELS:
                    continue

                ys_w, xs_w = np.where(wedge)
                x0, y0, x1, y1 = xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()
                x0, y0 = max(0, x0 - POLAR_PAD), max(0, y0 - POLAR_PAD)
                x1, y1 = min(W - 1, x1 + POLAR_PAD), min(H - 1, y1 + POLAR_PAD)

                # keep wedge from LIMBUS, pixel content from GLOBAL-CROP image
                tile = masked_global[y0:y1 + 1, x0:x1 + 1].copy()
                tile_w = wedge[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
                tile[tile_w == 0] = 0

                tile_res = cv2.resize(tile, (TILE_SAVE_SIZE, TILE_SAVE_SIZE), interpolation=cv2.INTER_AREA)
                tile_name = f"tile_r{r}_s{s}.jpg"
                cv2.imwrite(save_base + "_" + tile_name, cv2.cvtColor(tile_res, cv2.COLOR_RGB2BGR))

                meta_list.append({
                    "ring": r,
                    "sector": s,
                    "quality": tile_quality_score(tile_res),
                    "file": os.path.basename(save_base) + "_" + tile_name
                })

        with open(save_base + "_meta.json", "w") as f:
            json.dump(meta_list, f)

    except Exception as e:
        err_path = os.path.join(save_root, "errors.txt")
        with open(err_path, "a") as ef:
            ef.write(f"{p} -> {repr(e)}\n")


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_CACHE, exist_ok=True)

    # load segmentation model (multi-class)
    model, idx_crop, idx_limbus, img_size = load_model(CHECKPOINT_PATH, DEVICE)
    model.eval()

    # collect image paths
    all_paths = []
    for c in CLASSES:
        folder = os.path.join(DATA_ROOT, c)
        paths = glob.glob(os.path.join(folder, "*.*"))
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in VALID_EXTS:
                all_paths.append(p)

    print(f"🚀 Optimized Precomputing {len(all_paths)} images to {OUT_CACHE}...")
    dataset = PrecomputeDataset(all_paths, (CANONICAL_SIZE, CANONICAL_SIZE))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    futures = []
    executor = ThreadPoolExecutor(max_workers=8)

    with torch.no_grad():
        for batch_tensors, batch_paths, batch_rgbs in tqdm(loader):
            out = model(batch_tensors.to(DEVICE))

            # handle tuple outputs
            logits = out[0] if isinstance(out, (tuple, list)) else out

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            prob_crop_batch = probs[:, idx_crop]       # class 0: rectangle crop
            prob_limbus_batch = probs[:, idx_limbus]   # limbus segmentation

            for i in range(len(batch_paths)):
                rgb_np = _safe_to_numpy_rgb(batch_rgbs[i])
                futures.append(
                    executor.submit(
                        process_and_save,
                        batch_paths[i],
                        rgb_np,
                        prob_crop_batch[i],
                        prob_limbus_batch[i],
                        OUT_CACHE
                    )
                )

    # surface thread exceptions
    for f in as_completed(futures):
        _ = f.result()

    executor.shutdown(wait=True)
    print("\n✅ All pre-computation finished!")


if __name__ == "__main__":
    main()
