# Model 1: Hybrid MIL-Tabular Training Pipeline

This directory contains the pipeline for training a universal model using a combination of deep learning features, Multi-Instance Learning (MIL) for tiles, and handcrafted tabular features.

## Data Flow Diagram

```mermaid
graph TD
    subgraph "Stage 1: Dataset Preparation (01_Dataset_Prep.py)"
        A[Raw Slit-Lamp Images] --> B[Segmentation Model]
        B --> C{Predictions}
        C -->|Class 0| D[Rectangular Global Crop]
        C -->|Class 1| E[Polar Limbus Tiles]
        D --> F[Save to Cache]
        E --> F
        F --> G[Generate meta.json]
    end

    subgraph "Stage 2: Feature Engineering (02_Training_pipeline.py)"
        G --> H[Index Cache Files]
        H --> I[Re-score Tiles]
        I -->|Clinical Logic| J[Sharpness, Entropy, HF-ratio]
        J --> K[Select Top-K Tiles]
        K --> L[Extract Handcrafted Features]
        L -->|LBP, FFT, GLCM, Haar| M[Vectorize Tabular Data]
    end

    subgraph "Stage 3: Training (02_Training_pipeline.py)"
        M --> N[Hybrid Data Loader]
        D --> N
        K --> N
        N --> O{Model: DualBranchMIL}
        
        subgraph "Neural Network Architecture"
            O --> P[Global Branch: EfficientNet-B0]
            O --> Q[Tile Branch: EfficientNet-B0]
            Q --> R[Gated Attention MIL]
            R --> S[Bag Feature Aggregation]
            O --> T[Tabular Branch: MLP]
        end
        
        P --> U[Feature Fusion Layer]
        S --> U
        T --> U
        U --> V[Final Classifier]
        V --> W[Focal Loss + Entropy Reg]
        W --> X[Model Weights]
    end
```

## Detailed Process Breakdown

### 1. Dataset Preparation (`01_Dataset_Prep.py`)
*   **Input**: Raw medical images from `DatasetSatyam`.
*   **Segmentation**: Uses a U-Net++ model to identify the `Crop` (general ROI) and `Limbus` (eye structure).
*   **Global Crop**: Extracts a binary rectangular area around the limbus.
*   **Tiling**: Performs polar slicing to create circular wedges (tiles) centered on the eye.
*   **Caching**: Saves everything into a precomputed folder to avoid redundant computation during training.

### 2. Feature Engineering & Quality Assessment
*   **Clinical Scoring**: Tiles are evaluated for:
    *   **Sharpness**: Laplacian variance.
    *   **Texture**: Local entropy and LBP (Local Binary Patterns).
    *   **Frequency**: FFT high-frequency ratio to detect fine details.
*   **Handcrafted Features**: Extracts high-level statistical features from both the global image and the best-performing tiles, including:
    *   **Texture**: GLCM Contrast, Haar Wavelet energies.
    *   **Color**: LAB and HSV color space statistics.
    *   **Shape**: Roundness, Elongation, and Solidity of the cropped region.

### 3. Model Architecture (`DualBranchMIL_WithFeatures`)
The model is a **tri-modal fusion architecture**:
1.  **Global Branch**: Processes the entire cropped eye to capture context.
2.  **Tile Branch (MIL)**: Processes multiple small tiles to find localized pathologies. It uses an **Attention mechanism** to learn which tiles are most important for the diagnosis.
3.  **Tabular Branch**: Processes the handcrafted feature vector directly through an MLP (Multi-Layer Perceptron), providing traditional computer vision insights to the deep learning model.

### 4. Training Objectives
*   **Focal Loss**: Used to handle class imbalance (e.g., differentiating rare infections from normal cases).
*   **Attention Entropy**: Penalizes the model if it focuses on too many tiles at once, encouraging it to find clear, diagnostic "evidence."
