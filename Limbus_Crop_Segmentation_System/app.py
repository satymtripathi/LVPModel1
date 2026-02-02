import gradio as gr
import cv2
import numpy as np
import torch
import os
from inference_utils import (
    load_model, predict_masks, draw_standard_contours, 
    draw_smooth_contours, get_cropped_roi
)

# =======================
# APP CONFIG
# =======================
MODEL_PATH = "model_limbus_crop_unetpp_weighted.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model globally
model, idx_crop, idx_limbus, img_size = load_model(MODEL_PATH, DEVICE)

def process_image(input_img):
    """
    Gradio processing function.
    input_img is a numpy array (RGB) from Gradio.
    """
    if input_img is None:
        return None, None, None
        
    # Convert RGB to BGR for OpenCV processing
    bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    # Run Inference
    masks = predict_masks(model, bgr, img_size, DEVICE)
    crop_mask = masks[idx_crop]
    limbus_mask = masks[idx_limbus]
    
    # Generate Standard Contours
    standard_vis_bgr = draw_standard_contours(bgr, crop_mask, limbus_mask)
    standard_vis_rgb = cv2.cvtColor(standard_vis_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate Smooth Pretty Contours
    smooth_vis_bgr = draw_smooth_contours(bgr, crop_mask, limbus_mask)
    smooth_vis_rgb = cv2.cvtColor(smooth_vis_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate Cropped ROI
    roi_bgr = get_cropped_roi(bgr, crop_mask)
    if roi_bgr is not None:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    else:
        roi_rgb = None
        
    return standard_vis_rgb, smooth_vis_rgb, roi_rgb

# =======================
# GRADIO INTERFACE
# =======================

with gr.Blocks(title="Limbus & Crop Segmentation System") as demo:
    gr.Markdown("# 👁️ Limbus & Crop Segmentation System")
    
    with gr.Row():
        # LEFT COLUMN: INPUT
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Panel")
            input_image = gr.Image(label="Upload Eye Image", type="numpy")
            btn = gr.Button("Analyze Image", variant="primary")
            
        # RIGHT COLUMN: OUTPUTS
        with gr.Column(scale=2):
            gr.Markdown("### 📤 Output Panels")
            with gr.Tabs():
                with gr.TabItem("Standard Contours"):
                    out_standard = gr.Image(label="Standard Contours", height=400)
                with gr.TabItem("Smooth (Glow/Ellipse)"):
                    out_smooth = gr.Image(label="Smooth Contours", height=400)
                with gr.TabItem("Cropped ROI"):
                    out_roi = gr.Image(label="Extracted Crop ROI", height=400)

    btn.click(
        fn=process_image, 
        inputs=input_image, 
        outputs=[out_standard, out_smooth, out_roi]
    )
    
    gr.Markdown("### Features:")
    gr.Markdown("- **Standard Contours**: Direct segmentation mask contours.")
    gr.Markdown("- **Smooth Contours**: Post-processed contours using ellipse fitting and rectangle bounding boxes for a clean, professional look.")

if __name__ == "__main__":
    demo.launch(share=False)
