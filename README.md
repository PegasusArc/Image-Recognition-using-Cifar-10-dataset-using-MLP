# 🧠 CIFAR-10 Image Recognition using Pure MLP and MLP‑Mixer (PyTorch)

This repository contains a full, end-to-end workflow for classifying CIFAR‑10 images using an **all-MLP architecture**, including an **MLP‑Mixer** variant that achieves strong accuracy without convolutions or attention.

The project is implemented in a single, reproducible **Jupyter Notebook** and can be run locally or on **Google Colab**.

---

## 👨‍💻 Author
**Praneel Sahu**

---

## 📂 Repository Contents
- `2025A_IMG_MLP_MIXER_Praneel_Final_v2.ipynb` – Main notebook with data loading, models (Deep MLP and MLP‑Mixer), training, evaluation, and visualizations.
- `README.md` – Project overview and execution instructions.
- `LICENSE` – MIT License.

Artifacts created during runs (not stored in repo):
- `data/` – CIFAR‑10 dataset (auto-downloaded by `torchvision`).
- `Models/ImprovedMLP_model_final.pth` – Saved trained model (serialized via `pickle`).

---

## ⚙️ Prerequisites
- **Python**: 3.9+
- **PyTorch** and **torchvision**
- **torchinfo** (for model summary)
- **scikit-learn**, **matplotlib**, **pandas**, **numpy**

Install locally:

```bash
pip install torch torchvision torchaudio
pip install torchinfo scikit-learn matplotlib pandas numpy
```

Or run directly on **Google Colab** (recommended). The notebook includes a Colab badge and a one-liner to install `torchinfo`.

---

## ▶️ How to Execute

### Option A: Run on Google Colab
1. Open the notebook in Colab using the badge at the top of the notebook.
2. Set runtime to GPU (Runtime → Change runtime type → Hardware accelerator: GPU).
3. Run all cells (Runtime → Run all).

### Option B: Run Locally (Jupyter)
1. Create and activate a virtual environment (optional but recommended).
2. Install prerequisites (see above).
3. Launch Jupyter and open `2025A_IMG_MLP_MIXER_Praneel_Final_v2.ipynb`:
   ```bash
   jupyter notebook
   ```
4. Run all cells in order. The dataset will download automatically on first run.

---

## 🧩 Project Overview

- **Dataset**: CIFAR‑10 (60k images, 32×32 RGB, 10 classes)
- **Baselines**: Deep MLP with GELU, BatchNorm, Dropout, and residual (skip) connections
- **Final Model**: **MLP‑Mixer** (
  - Patch size: 4×4 (flattened patches as tokens)
  - Depth: 8 Mixer layers
  - Hidden dims: token mixing 256, channel dim 128 with channel MLP expansion ×4
  - LayerNorm + residual connections throughout
)
- **Training**:
  - Loss: Cross-entropy with label smoothing (0.1)
  - Optimizer: AdamW (lr=5e-4, weight_decay=2e-4)
  - Scheduler: CosineAnnealingWarmRestarts
  - Epochs: up to 100 with early stopping on validation accuracy
  - Augmentations: random flip/crop, color jitter, rotation, affine, random erasing

---

## 📈 Results
- **Best Validation Accuracy**: ~82.7%
- **Test Accuracy**: **82.4%** on the 5k test split

The notebook also includes:
- Confusion matrix and `classification_report`
- Training/validation loss and accuracy curves
- Visualizations of predicted samples (correct/incorrect)

---

## 🔧 How the Code Is Organized (inside the notebook)
- Data transforms and `DataLoader`s for train/val/test
- Two models:
  - `DeepMLP`: 8-layer halving-width MLP with BatchNorm, GELU, Dropout, and skip connections
  - `MLPMixer`: patch embedding → repeated Mixer layers (token/channel MLPs) → pooling → classifier
- Mixed-precision training on GPU (`torch.amp`), early stopping, LR scheduling
- Metrics, plots, confusion matrix, class-wise report
- Model checkpointing to `Models/ImprovedMLP_model_final.pth`

---

## ▶️ Quick Start (Colab)
Once opened in Colab:
1. Ensure GPU runtime is enabled.
2. Run all cells. Training will start and logs will print per-epoch statistics.
3. At the end, evaluation metrics and visualizations are shown; the trained model is saved under `Models/`.

---

## 📝 Notes
- Training time varies by GPU: ~15s/epoch on high-end GPUs; ~50s/epoch on T4.
- For faster experimentation, reduce epochs or batch size (default `bs=256`).
- You can switch between `DeepMLP` and `MLPMixer` by modifying the `model = ...` cell.

---

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📬 Contact
For questions or contributions, please contact: **Praneel Sahu** via GitHub.

© 2025 **PegasusArc**
