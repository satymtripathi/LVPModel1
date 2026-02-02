import os, glob, cv2
import numpy as np
from tqdm import tqdm
import torch
from inference_utils import load_model, predict_masks, draw_standard_contours

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "model_limbus_crop_unetpp_weighted.pth"
INPUT_DIR = r"C:\Users\satyam.tripathi\Downloads\Suture_Model\test"
OUT_DIR = r"C:\Users\satyam.tripathi\Downloads\Suture_Model\test\output_standard"
# ----------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model using utils
    model, idx_crop, idx_limbus, img_size = load_model(CHECKPOINT_PATH, device)
    
    # Get images
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    paths = sorted(list(set(paths)))
    
    print(f"Standard Inference | Device: {device} | Images: {len(paths)}")
    
    for p in tqdm(paths):
        bgr = cv2.imread(p)
        if bgr is None: continue
        
        # Predict
        masks = predict_masks(model, bgr, img_size, device)
        
        # Draw Standard
        vis = draw_standard_contours(bgr, masks[idx_crop], masks[idx_limbus])
        
        # Save
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_standard.jpg"), vis)

if __name__ == "__main__":
    main()
