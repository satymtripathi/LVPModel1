# Limbus & Crop Segmentation System

This system performs high-precision 2-class segmentation on eye images, identifying the **Crop ROI** (Region of Interest) and the **Limbus Boundary**.

## 📂 Directory Structure

```text
Limbus_Crop_Segmentation_System/
├── app.py                      # Interactive Gradio UI
├── train_pipeline.py           # Model training script
├── inference_contour.py        # Standard inference (CLI)
├── inference_smooth.py         # Advanced smooth inference (CLI)
├── inference_utils.py          # Shared segmentation & drawing logic
├── README.md                   # Project documentation
└── model_limbus_crop_unetpp_weighted.pth  # Trained model weights
```

---

## 🚀 Features

### 1. 🔬 Deep Learning Model
- **Architecture**: U-Net++ with EfficientNet-B0 backbone.
- **Classes**: 
  - `crop`: Rectangular ROI for extraction.
  - `limbus`: Polygon/Circular boundary of the limbus.
- **Training**: Optimized with weighted multilabel Loss (Dice + BCE), prioritizing Limbus accuracy.

### 2. 🎨 Visualization Styles
The system provides two distinct ways to view result boundaries:

| Style | Description | Logic |
| :--- | :--- | :--- |
| **Standard** | Thin, raw contours directly from the mask. | `inference_contour.py` |
| **Smooth** | clean boxes, fitted ellipses, and glow effects. | `inference_smooth.py` |

---

## 🛠 Usage

### Prerequisites
Ensure you have the following installed:
`pip install torch torchvision segmentation-models-pytorch albumentations opencv-python gradio tqdm pandas`

### 1. Training (`train_pipeline.py`)
To train a new model on your data:
- Edit the `config` dictionary at the bottom of the script with your data paths.
- Run: `python train_pipeline.py`

### 2. Interactive Web UI (`app.py`)
Run the Gradio interface to analyze images one-by-one:
- Run: `python app.py`
- Open the provided local URL in your browser.
- Upload an image and see three panels: Original, Standard, and Smooth.

### 3. Batch Inference (CLI)
For processing entire folders of images:
- **Standard Contours**: `python inference_contour.py`
- **Smooth/Professional Contours**: `python inference_smooth.py`
- Edit the `INPUT_DIR` and `OUT_DIR` inside the scripts before running.

---

## 📝 Key Components

- **`inference_utils.py`**: Centralizes mask prediction and post-processing.
- **`model_limbus_crop_unetpp_weighted.pth`**: The default model used by the UI and inference scripts.

---
*Designed for high-accuracy ophthalmic image analysis.*
