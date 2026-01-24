# Theory & Pipeline: 05_NewTraining.py

This document provides a detailed technical breakdown of the advanced Dual-Branch Multiple Instance Learning (MIL) pipeline used for keratitis classification.

## 1. System Architecture
The system uses a **Triple-Input Multi-Branch** architecture to capture information at different scales and modalities:

| Branch | Input Type | Backbone | Purpose |
| :--- | :--- | :--- | :--- |
| **Global Branch** | Full Image (384x384) | EfficientNet-B0 | Captures overall eye context and large-scale structures. |
| **Tile Branch (MIL)** | 24 Local Tiles (224x224) | EfficientNet-B0 | Focuses on micro-textures and local lesions (infiltrates). |
| **Tabular Branch** | 36+ Hand-crafted Features | Multi-Layer Perceptron | Captures mathematical descriptors (LBP, GLCM, FFT). |

---

## 2. Tile Processing & MIL Flow
This is the core of the "No Code Segmentation" logic. Instead of needing pixel-level masks, the model learns to "attend" to relevant areas.

### A. Tile Extraction & Sampling
- Each image is treated as a **"Bag"** of instances (tiles).
- Up to **24 tiles** are extracted per image.
- During training, **Tile Dropout (10%)** is applied to ensure the model doesn't over-rely on a single "perfect" tile.

### B. Feature Extraction (Backbone)
- Every tile is passed through an **EfficientNet-B0** backbone.
- The output is a high-dimensional feature vector (1280 dims), which is then projected into a **256-dimensional latent space**.

### C. Foreground/Background Isolation (Masking)
A critical feature of this pipeline is **Sector-Specific Masking**:
- **Tile Masking**: Within each 224x224 tile, only the pixels belonging to the specific "Polar Sector" are preserved. All other pixels (background) are zeroed to black.
- **Why?**: This forces the AI to ignore structural noise (eyelids, eyelashes, scleral reflections) and focus strictly on the corneal tissue in the foreground.
- **Valid Masking**: The model uses a `valid_mask` to ensure that empty/padded tiles (where no tissue was detected) are completely ignored during the attention calculation.

#### How it works (Technical Mechanism):
1.  **Limbus Centroid**: The system calculates the geographic center $(cx, cy)$ of the cornea using the `limbus_mask`.
2.  **Coordinate Mapping**: Every pixel in the image is mapped to a Radius ($r$) and Angle ($\theta$) relative to that center.
3.  **Wedge Definition**: 24 sectors are defined by mathematical bounds:
    -   **Radius Rings**: 3 rings (Inner, Middle, Outer).
    -   **Angular Slices**: 8 sectors (45° each).
4.  **Bitwise Intersection**: A binary mask is created for each sector (the "Wedge"). The system performs a bitwise operation: `Result = Image AND Wedge`.
5.  **Result**: Any pixel *not* inside the radius/angle bounds is forced to RGB (0,0,0), creating the blacked-out background effect.

### D. Gated Attention Mechanism
The model uses a learnable attention mechanism to score each tile:
- **Attention V & U**: These two linear layers compute a "gating" score using Tanh and Sigmoid activations.
- **Clinical Bias**: The precomputed **Clinical Quality Score** is added to the attention logits. This forces the model to prioritize tiles that are sharp, centered, and show "opacity-like" patterns rather than glare or border edges.
- **Softmax Activation**: The scores are normalized so they sum to 1. This represents the probability that a specific tile is the most diagnostic.

### D. Soft Attention Pooling
Unlike "Top-K" pooling which only looks at the best tiles, **Soft Attention Pooling** calculates a weighted average of all 24 tiles based on their attention scores.
> [!NOTE]
> This is a "Doctor-like" approach: The model considers the entire eye but gives much more weight to the most suspicious-looking lesions.

---

## 3. Clinical Tile Scoring Logic
Before training, each tile is ranked based on a heuristic "Clinical Quality" formula:
1.  **Sharpness**: Uses the Laplacian variance to penalize blur.
2.  **Specular Glare Penalty**: Penalizes regions with very high brightness (L > 210) and low saturation, which are usually light reflections.
3.  **Border-Edge Penalty**: Penalizes tiles where most edges are at the very boundary (often artifacts).
4.  **Opacity Reward**: Rewards pixels with "infiltrate-like" brightness profiles (L between 150-225).
5.  **Texture Diversity**: Uses LBP Entropy and FFT high-frequency ratios to ensure the tile isn't just a flat, uniform surface.

---

## 4. Advanced Hand-crafted Features (Tabular Branch)
The model extracts **36 features** per global image and for the top clinical tiles:
- **LBP (Local Binary Patterns)**: Captures micro-texture regularity.
- **GLCM (Gray-Level Co-occurrence Matrix)**: Approximates contrast and homogeneity.
- **FFT (Fast Fourier Transform)**: Measures frequency energy (detects fine versus coarse structures).
- **Wavelet Haar Energies**: Multi-scale frequency analysis.
- **Color Stats**: LAB and HSV mean/std to capture redness (Infection) or dullness (Scar).

---

## 5. Training Dynamics
- **Focal Loss**: Used to handle class imbalance, focusing more on hard-to-classify samples (e.g., distinguishing Scar from Edema).
- **Entropy Regularization**: During the first 5 epochs (Warmup), the model is penalized for having "too sharp" attention. This forces it to "look around" the whole image before settling on specific lesions.
- **Device-Robust Augmentation**: To handle images from different camera hardware, we apply mild Gamma, Brightness, and JPEG compression augmentations.
