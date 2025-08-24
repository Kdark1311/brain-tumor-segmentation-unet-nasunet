# 🧠 Brain Tumor Segmentation with U-Net and NASU-Net  

🚀 A deep learning project for **brain tumor segmentation** on MRI scans using **U-Net** and **NASU-Net** architectures.  

---

## 📂 Project Structure
Segmentation_project/
│── archive/ # Dataset (images, annotations, masks, processed npy)
│── processed/ # Augmented .npy data for training
│── model_unet.py # Standard U-Net model
│── model_nasunet.py # NASU-Net model (searched architecture)
│── train.py # Training pipeline
│── eval_predict.py # Evaluation & prediction script
│── make_binary_masks.py # Convert COCO annotations to binary masks
│── augmentation.py # Data augmentation & preprocessing
│── requirements.txt # Dependencies

yaml
Copy
Edit

---

## ⚙️ Setup
```bash
# Create environment
python -m venv venv
# Activate venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
🧠 Training
Train with U-Net
bash
Copy
Edit
python train.py --model unet --data "archive/processed" --epochs 50 --batch 8 --lr 3e-4 --base 32 --ckpt ./checkpoints
Train with NASU-Net
bash
Copy
Edit
python train.py --model nasunet --data "archive/processed" --epochs 50 --batch 8 --lr 3e-4 --base 32 --groups 8 --ckpt ./checkpoints
📊 Evaluation & Prediction
bash
Copy
Edit
python eval_predict.py --model nasunet --data "archive/processed" \
  --weights ./checkpoints/nasunet_best.pt --batch 8 --thr 0.5 --post --out ./pred_masks_npy
🔍 Features
✅ Convert COCO annotations → binary masks

✅ Data augmentation with Albumentations

✅ Two model options: U-Net and NASU-Net

✅ Dice Loss + IoU evaluation

✅ Training & evaluation scripts ready-to-use

📌 Requirements
Python 3.9+

PyTorch

Albumentations

OpenCV

pycocotools

Install via:

bash
Copy
Edit
pip install -r requirements.txt
📈 Results
Dice Score, IoU metrics

Predicted tumor masks (post-processed with morphological ops)

Visualization: Ground truth vs Prediction overlay

🎨 Visualization Example
Bạn có thể hiển thị ảnh gốc, mask thật và mask dự đoán bằng matplotlib:

python
Copy
Edit
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_results(image, true_mask, pred_mask):
    plt.figure(figsize=(12,4))

    # Ảnh gốc
    plt.subplot(1,3,1)
    plt.imshow(image, cmap="gray")
    plt.title("Original MRI")
    plt.axis("off")

    # Mask ground truth
    plt.subplot(1,3,2)
    plt.imshow(true_mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Overlay prediction
    plt.subplot(1,3,3)
    overlay = cv2.addWeighted(image, 0.7, (pred_mask*255).astype(np.uint8), 0.3, 0)
    plt.imshow(overlay, cmap="gray")
    plt.title("Predicted Mask Overlay")
    plt.axis("off")

    plt.show()
Ví dụ output trực quan sẽ hiển thị như sau:

🖼 Original MRI

🎭 Ground Truth Mask

🔴 Predicted Mask Overlay (khối u highlight màu đỏ trên ảnh MRI)

🙌 Credits
U-Net Paper (Ronneberger et al., 2015)

NASU-Net Paper

Dataset: Brain MRI (COCO-annotated format)
