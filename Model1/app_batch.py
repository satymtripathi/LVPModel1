# ============================================================
# KeratitisAI V5 — FINAL INFERENCE UI
# SINGLE IMAGE + BATCH ZIP
# TRAIN ↔ INFERENCE LOCKED
# ============================================================

import os, sys, io, zipfile, json, math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.cuda.amp import autocast
import streamlit as st
from dataclasses import dataclass
from typing import Tuple, List, Dict

from Limbus_Crop_Segmentation_System.inference_utils import (
    load_model,
    predict_masks
)

# -------------------- STABILITY --------------------
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class CFG:
    SEG_CKPT = os.path.join(BASE_DIR, "Limbus_Crop_Segmentation_System",
                            "model_limbus_crop_unetpp_weighted.pth")
    TRAIN_OUT = os.path.join(BASE_DIR, "training_results_v5")
    CLS_CKPT = os.path.join(TRAIN_OUT, "checkpoints", "best.pth")

    FEATURE_KEYS_JSON = os.path.join(TRAIN_OUT, "feature_keys.json")
    FEAT_MU = os.path.join(TRAIN_OUT, "feat_mu.npy")
    FEAT_SIGMA = os.path.join(TRAIN_OUT, "feat_sigma.npy")

    CANONICAL = 512
    GLOBAL = 384
    TILE = 224
    MAX_TILES = 24

    POLAR_THETA = 8
    POLAR_RINGS = 3
    RING_FRAC = (0.0, 0.35, 0.70, 1.0)
    POLAR_MIN_PIX = 250
    POLAR_PAD = 2

    TOP_TILES_FOR_FEATURES = 6
    CROP_PAD = 20
    CROP_MIN_AREA = 0.002

cfg = CFG()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# ============================================================
# BASIC UTILS
# ============================================================
def resize_rgb(rgb, size_hw):
    return cv2.resize(rgb, (size_hw[1], size_hw[0]), cv2.INTER_AREA)

def to_tensor(rgb):
    t = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
    return (t - MEAN) / STD

def normalize01(x):
    if x.size == 0: return x
    mn, mx = float(x.min()), float(x.max())
    return (x-mn)/(mx-mn+1e-8) if mx>mn else np.zeros_like(x)

def apply_mask(rgb, mask):
    out = rgb.copy()
    out[mask == 0] = 0
    return out

# ============================================================
# MASK POSTPROCESS
# ============================================================
def postprocess(mask):
    m = (mask*255).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2, iterations=1)
    return (m>127).astype(np.uint8)

def largest_contour(mask):
    cnts,_ = cv2.findContours((mask*255).astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def crop_bbox(rgb, mask):
    c = largest_contour(mask)
    if c is None: return None
    x,y,w,h = cv2.boundingRect(c)
    return rgb[y:y+h, x:x+w].copy()

# ============================================================
# TILE QUALITY (UNCHANGED)
# ============================================================
def tile_quality_score(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    valid = gray > 5
    if valid.mean() < 0.1: return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.log1p(lap[valid].var()))

# ============================================================
# FEATURE EXTRACTION (UNCHANGED LOGIC)
# ============================================================
def extract_feature_vector_from_rgb(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std())
    }

def aggregate_tile_features(tile_dicts):
    if not tile_dicts: return {}
    keys = tile_dicts[0].keys()
    mat = np.array([[t[k] for k in keys] for t in tile_dicts])
    out = {}
    for i,k in enumerate(keys):
        out[f"tile_mean_{k}"] = float(mat[:,i].mean())
        out[f"tile_std_{k}"] = float(mat[:,i].std())
    return out

def compute_feature_vector(global_rgb_feat, tiles_native, feat_keys):
    feats = {f"g_{k}":v for k,v in extract_feature_vector_from_rgb(global_rgb_feat).items()}
    top_tiles = tiles_native[:cfg.TOP_TILES_FOR_FEATURES]
    feats.update(aggregate_tile_features(
        [extract_feature_vector_from_rgb(t) for t in top_tiles]
    ))
    return np.array([feats.get(k,0.0) for k in feat_keys], np.float32)

# ============================================================
# POLAR TILES (FIXED)
# ============================================================
def polar_tiles(global_masked, limbus_mask):
    H,W = limbus_mask.shape
    ys,xs = np.where(limbus_mask>0)
    if xs.size==0: return [],[],np.zeros(0),[]

    cx,cy = xs.mean(), ys.mean()
    rmax = np.sqrt((xs-cx)**2+(ys-cy)**2).max()
    yy,xx = np.mgrid[0:H,0:W]
    rr = np.sqrt((xx-cx)**2+(yy-cy)**2)
    tt = (np.arctan2(yy-cy,xx-cx)+2*np.pi)%(2*np.pi)
    edges = [f*rmax for f in cfg.RING_FRAC]

    native, cnn, q, coords = [],[],[],[]

    for r in range(cfg.POLAR_RINGS):
        for s in range(cfg.POLAR_THETA):
            t0,t1 = 2*np.pi*s/cfg.POLAR_THETA, 2*np.pi*(s+1)/cfg.POLAR_THETA
            w = (limbus_mask>0)&(rr>=edges[r])&(rr<edges[r+1])&(tt>=t0)&(tt<t1)
            if w.sum()<cfg.POLAR_MIN_PIX: continue
            ys_w,xs_w = np.where(w)
            x0,y0,x1,y1 = xs_w.min(),ys_w.min(),xs_w.max(),ys_w.max()
            tile = global_masked[y0:y1+1,x0:x1+1].copy()
            tile[w[y0:y1+1,x0:x1+1]==0]=0
            tile_cnn = cv2.resize(tile,(cfg.TILE,cfg.TILE))
            native.append(tile)
            cnn.append(tile_cnn)
            q.append(tile_quality_score(tile_cnn))
            coords.append((x0,y0,x1,y1))

    order = np.argsort(-np.array(q))
    return ([native[i] for i in order],
            [cnn[i] for i in order],
            np.array(q)[order],
            [coords[i] for i in order])

# ============================================================
# MODEL (UNCHANGED)
# ============================================================
class GatedAttention(nn.Module):
    def __init__(self,L,D):
        super().__init__()
        self.V = nn.Linear(L,D)
        self.U = nn.Linear(L,D)
        self.w = nn.Linear(D,1)
    def forward(self,x):
        return self.w(torch.tanh(self.V(x))*torch.sigmoid(self.U(x)))

class DualBranchMIL(nn.Module):
    def __init__(self,nc,fd):
        super().__init__()
        b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.tf = b.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280,256)
        self.att = GatedAttention(256,128)

        b2 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.gf = b2.features
        self.gproj = nn.Linear(1280,256)

        self.fm = nn.Sequential(nn.Linear(fd,256),nn.ReLU(),nn.Linear(256,128))
        self.cls = nn.Linear(256+256+128,nc)

    def forward(self,g,t,f,q,vm):
        B,T = t.shape[:2]
        z = self.proj(self.pool(self.tf(t.view(B*T,3,224,224))).view(B,T,-1))
        a = self.att(z).squeeze(-1)+q
        a = a.masked_fill(vm==0,-1e9)
        a = torch.softmax(a,1)
        bag = (a.unsqueeze(-1)*z).sum(1)
        g = self.gproj(self.pool(self.gf(g)).view(B,-1))
        f = self.fm(f)
        return self.cls(torch.cat([bag,g,f],1)),a

# ============================================================
# LOAD EVERYTHING
# ============================================================
@st.cache_resource
def load_all():
    with open(cfg.FEATURE_KEYS_JSON) as f:
        feat_keys = json.load(f)
    mu = np.load(cfg.FEAT_MU)
    sig = np.maximum(np.load(cfg.FEAT_SIGMA),1e-6)

    seg, idx_c, idx_l, imsz = load_model(cfg.SEG_CKPT, DEVICE)
    seg.eval()

    ck = torch.load(cfg.CLS_CKPT, map_location=DEVICE)
    classes = ck["cfg"]["CLASSES"]
    model = DualBranchMIL(len(classes), len(feat_keys))
    model.load_state_dict(ck["model"], strict=True)
    model.to(DEVICE).eval()

    return seg, idx_c, idx_l, imsz, model, classes, feat_keys, mu, sig

seg_model, IDX_CROP, IDX_LIMB, IMG_SZ, MODEL, CLASSES, FEAT_KEYS, MU, SIG = load_all()

# ============================================================
# CORE INFERENCE (USED BY SINGLE + BATCH)
# ============================================================
def run_inference(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    masks = predict_masks(seg_model, bgr, IMG_SZ, DEVICE)
    crop = postprocess(masks[IDX_CROP])
    limb = masks[IDX_LIMB]

    rgb512 = resize_rgb(rgb,(cfg.CANONICAL,cfg.CANONICAL))
    crop512 = resize_rgb(crop,(cfg.CANONICAL,cfg.CANONICAL))
    limb512 = resize_rgb(limb,(cfg.CANONICAL,cfg.CANONICAL))

    global_cnn = crop_bbox(rgb512, crop512)
    if global_cnn is None:
        global_cnn = apply_mask(rgb512,crop512)

    global_feat = resize_rgb(apply_mask(rgb512,crop512),(cfg.GLOBAL,cfg.GLOBAL))
    global_cnn = resize_rgb(global_cnn,(cfg.GLOBAL,cfg.GLOBAL))

    masked = apply_mask(rgb512,crop512)
    tiles_nat, tiles_cnn, q, coords = polar_tiles(masked, limb512)

    tiles_cnn = tiles_cnn[:cfg.MAX_TILES]
    pad = cfg.MAX_TILES-len(tiles_cnn)
    tiles_cnn += [np.zeros((224,224,3),np.uint8)]*max(0,pad)

    q01 = np.zeros(cfg.MAX_TILES,np.float32)
    q01[:len(q)] = normalize01(q[:cfg.MAX_TILES])
    vm = np.zeros(cfg.MAX_TILES,np.float32)
    vm[:len(q)] = 1

    feat = compute_feature_vector(global_feat, tiles_nat, FEAT_KEYS)
    feat = (feat-MU)/SIG

    with torch.no_grad(), autocast(enabled=(DEVICE=="cuda")):
        logits, att = MODEL(
            to_tensor(global_cnn).unsqueeze(0).to(DEVICE),
            torch.stack([to_tensor(t) for t in tiles_cnn]).unsqueeze(0).to(DEVICE),
            torch.from_numpy(feat).unsqueeze(0).to(DEVICE),
            torch.from_numpy(q01).unsqueeze(0).to(DEVICE),
            torch.from_numpy(vm).unsqueeze(0).to(DEVICE)
        )
        prob = torch.softmax(logits,1)[0].cpu().numpy()

    return {
        "pred": CLASSES[int(prob.argmax())],
        "probs": prob,
        "attention": att[0].cpu().numpy(),
        "tiles": tiles_cnn,
        "coords": coords
    }

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config("KeratitisAI V5", layout="wide")
st.title("🛡️ KeratitisAI — Single & Batch Inference")

mode = st.sidebar.radio("Mode", ["Single Image","Batch ZIP"])

if mode=="Single Image":
    f = st.file_uploader("Upload image", type=["jpg","png"])
    if f:
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(f.read(),np.uint8),1),
                           cv2.COLOR_BGR2RGB)
        out = run_inference(rgb)
        st.image(rgb)
        st.success(out["pred"])
        st.bar_chart(out["probs"])

else:
    zf = st.file_uploader("Upload ZIP of images", type=["zip"])
    if zf:
        z = zipfile.ZipFile(io.BytesIO(zf.read()))
        tabs = st.tabs(z.namelist())
        for tab,name in zip(tabs,z.namelist()):
            with tab:
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(name),np.uint8),1),
                                   cv2.COLOR_BGR2RGB)
                out = run_inference(rgb)
                st.image(rgb)
                st.success(out["pred"])
                st.bar_chart(out["probs"])
