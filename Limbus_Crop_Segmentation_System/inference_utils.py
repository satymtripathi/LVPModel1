import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =======================
# UTILS & SHARED LOGIC
# =======================

def load_model(ckpt_path, device):
    """
    Loads the trained model checkpoint.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    state = ckpt["state_dict"]

    encoder_name = cfg.get("encoder_name", "timm-efficientnet-b0")
    target_list = cfg.get("target_list", [{"label": "crop"}, {"label": "limbus"}])

    labels = [t["label"].strip().lower() for t in target_list]
    idx_crop = labels.index("crop") if "crop" in labels else 0
    idx_limbus = labels.index("limbus") if "limbus" in labels else 1

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=len(target_list),
        activation=None
    )
    model.load_state_dict(state)
    model.to(device).eval()

    img_size = cfg.get("img_size", (512, 512))
    return model, idx_crop, idx_limbus, img_size

def predict_masks(model, image_bgr, img_size, device, thresh=0.5):
    """
    Runs model inference and returns binary masks resized to original shape.
    """
    H, W = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    x = transform(image=rgb)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Resize back to original
    masks = []
    for c in range(probs.shape[0]):
        m = cv2.resize(probs[c], (W, H), interpolation=cv2.INTER_LINEAR)
        masks.append((m > thresh).astype(np.uint8))
        
    return np.stack(masks)

def largest_contour(mask01):
    m = (mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def postprocess_mask(mask01, kernel=7):
    m = (mask01 * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    
    h, w = m.shape[:2]
    flood = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    m = cv2.bitwise_or(m, holes)
    return (m > 127).astype(np.uint8)

# =======================
# DRAWING STYLES
# =======================

def draw_standard_contours(bgr, crop_mask, limbus_mask):
    """
    The style from limbus_crop_roi.py:
    Yellow/Cyan simple contours.
    """
    vis = bgr.copy()
    c1 = largest_contour(crop_mask)
    c2 = largest_contour(limbus_mask)
    
    if c1 is not None: cv2.drawContours(vis, [c1], -1, (0, 255, 255), 2) # Yellow, thickness 2
    if c2 is not None: cv2.drawContours(vis, [c2], -1, (255, 255, 0), 2) # Cyan, thickness 2
    return vis

def draw_smooth_contours(bgr, crop_mask, limbus_mask):
    """
    The style from smooth_limbus_and_crop_generation.py:
    Perfect rectangle and Ellipse-fit with glow.
    """
    vis = bgr.copy()
    
    # Post-process for smoothness
    crop_mask = postprocess_mask(crop_mask, kernel=9)
    limbus_mask = postprocess_mask(limbus_mask, kernel=7)
    
    c_crop = largest_contour(crop_mask)
    c_limb = largest_contour(limbus_mask)
    
    # CROP: Rectangle
    if c_crop is not None:
        x, y, w, h = cv2.boundingRect(c_crop)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 10) # Yellow, thickness 10
        
    # LIMBUS: Ellipse with Glow
    if c_limb is not None:
        if len(c_limb) >= 50: # ELLIPSE_MIN_POINTS = 50
            ellipse = cv2.fitEllipse(c_limb)
            # Glow effect
            dark = tuple(int(c * 0.35) for c in (255, 255, 0))
            cv2.ellipse(vis, ellipse, dark, 14) # OUTER_THICK = 14
            cv2.ellipse(vis, ellipse, (255, 255, 0), 8) # INNER_THICK = 8
        else:
            # Fallback to smooth contour if ellipse fails
            hull = cv2.convexHull(c_limb)
            peri = cv2.arcLength(hull, True)
            smooth = cv2.approxPolyDP(hull, 0.01 * peri, True)
            dark = tuple(int(c * 0.35) for c in (255, 255, 0))
            cv2.drawContours(vis, [smooth], -1, dark, 14)
            cv2.drawContours(vis, [smooth], -1, (255, 255, 0), 8)
            
    return vis

def contour_bbox(contour, H, W, pad=10):
    if contour is None: return None
    x, y, w, h = cv2.boundingRect(contour)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    if x2 <= x1 or y2 <= y1: return None
    return (x1, y1, x2, y2)

def get_cropped_roi(bgr, crop_mask, pad=10):
    """
    Extracts the ROI from the original image based on the crop mask.
    """
    H, W = bgr.shape[:2]
    contour = largest_contour(crop_mask)
    bbox = contour_bbox(contour, H, W, pad=pad)
    if bbox:
        x1, y1, x2, y2 = bbox
        return bgr[y1:y2, x1:x2].copy()
    return None
