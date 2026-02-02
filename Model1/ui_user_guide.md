# Clinical User Guide: KeratitisAI Dashboard

This guide explains how to interpret the diagnostic outputs of the KeratitisAI dashboard ([07_modelUI_updated.py](file:///c:/Users/satyam.tripathi/Desktop/NoCodeSegmentation/LVPModel1/Model1/07_modelUI_updated.py)).

## 1. Diagnostic Status (The Verdict)
The primary output is the **Diagnostic Status Card**, color-coded by clinical severity:
- 🔴 **INFECTION**: High-priority alert; requires immediate clinical correlation.
- 🔵 **EDEMA**: Fluid accumulation detected (typically Blue).
- 🟣 **SCAR / Non-Infectious**: Chronic or healed tissue change (typically Purple).
- 🟢 **NORMAL**: Clear cornea detected.

> [!TIP]
> **Confidence Score**: The percentage indicates the AI's mathematical certainty. A "Normal" prediction with 98% confidence is much stronger than one with 55%.

---

## 2. Anatomical Visualization (Explainable AI)
The dashboard allows you to "see what the AI sees" through four tabs:

### A. Global ROI (Region of Interest)
This shows the AI's segmented cornea area. If the image is poorly centered, the AI might crop it differently. Ensure the global ROI looks centered on the cornea.

### B. Slice + Attention
The AI breaks the cornea into **24 polar slices**. 
- **Star (⭐)**: Slices marked with a star have the highest "Diagnostic Weight" (Attention). 
- **AI Focus**: These are the specific tissue regions that drove the final diagnosis.

### C. Active Masking (Why are parts black?)
You will notice that each slice has significant black areas.
- **Intentional Filtering**: The AI intentionally blacks out the "Background" within each slice (eyelids, sclera, reflections).
- **Foreground Isolation**: This ensures the AI's "brain" is strictly analyzing **Corneal Tissue** in the foreground. If a slice is entirely black, it means no tissue was detected in that sector.

#### How this is happening:
Consider this a **"Digital Flashlight"**. The system uses the mathematical center of the eye to shine a light only on one specific tissue pocket at a time.
1.  **Tissue Targeting**: The AI calculates the exact coordinates of the cornea.
2.  **Background Blackout**: Any pixel that is outside the current "tissue sector" is mathematically set to black.
3.  **Noise Reduction**: By removing the "background noise" (like lashes or bright light reflections on the sclera), the AI can't get distracted by anything except the corneal lesion itself.

---

## 3. AI Hotspots (Top Attention Slices)
Located at the bottom of the screen, these 4 slices are the **"Hotspots"**.
- These are the most suspicious lesions or texture changes found by the AI.
- **Clinical Verification**: Doctors should compare these hotspots to the raw image. If the AI flags an "Infection" and the hotspot shows a clear infiltrate, the AI has correctly identified the diagnostic evidence.

---

## 4. Parameter Control (Sidebar)
- **Quality Bias (β)**: Increasing this forces the AI to trust "sharper" images more. If you have a very blurry image, lowering this might allow the AI to look at the global context more.
- **Attention Top-K**: Use this to see more (or fewer) "starred" boxes on the cornea map.
