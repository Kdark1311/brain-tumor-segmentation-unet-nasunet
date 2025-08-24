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
python train.py --model nasunet --data "archive/processed" --epochs 50 --batch 8 --lr 3e-4 --base 32 --groups 8 --ckpt ./checkpoints
📊 Evaluation & Prediction
bash
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
🙌 Credits
U-Net Paper (Ronneberger et al., 2015)

NASU-Net Paper

Dataset: Brain MRI (COCO-annotated format)
