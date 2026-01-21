# LVPModel1: Keratitis Detection System (V5 Expert System)

A state-of-the-art diagnostic system for Keratitis detection using a Dual-Branch MIL architecture with clinical-informed tile scoring and soft attention pooling.

## 🚀 Key Features
- **Deep Fusion**: Combines global ocular features with local patch-based evidence.
- **Clinical Hybrid**: Handcrafted tabular features merged with deep learning embeddings.
- **Doctor-like Scoring**: Tile attention is biased by clinical quality metrics (glare, visibility, opacity).
- **Device Robust**: Standardized feature normalization for stability across different slit-lamps.

## 📁 Project Structure
- `Model1/`: Core training and inference scripts (01-08).
- `Limbus_Crop_Segmentation_System/`: Pre-trained U-Net++ segmentation models.
- `training_results_v5/`: Best classifier weights and standardization stats.
- `requirements.txt`: Unified Python dependencies.
- `packages.txt`: System-level dependencies for cloud deployment.

## 🛠️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Expert UI (Streamlit)
```bash
streamlit run Model1/08_batchUI.py
```

### 3. Batch Inference
```bash
python Model1/06_NewTest_Batch.py --seg_ckpt Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth --cls_ckpt training_results_v5/checkpoints/best.pth --train_out_dir training_results_v5 --input path/to/images --out_dir outputs
```

## 📊 Data Flow
1. **Raw Image** -> **Segmentation** -> **Crop ROI & Limbus Mask**
2. **Crop ROI** -> **Polar Tiling** -> **Clinical Scoring**
3. **Global ROI + Tiles + Tabular Features** -> **Dual-Branch MIL** -> **Diagnosis**

---
*Developed for LVPEI*
