import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import model
from model_unet import UNet
from model_nasunet import NASUNet


# =========================
# Dataset Loader
# =========================
class BrainMRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.img_files[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img).unsqueeze(0)   # (1,H,W)
        mask = torch.tensor(mask).unsqueeze(0) # (1,H,W)
        return img, mask


# =========================
# Dice Loss
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice


# =========================
# IoU Metric
# =========================
def iou_score(y_pred, y_true, thr=0.5):
    y_pred = (torch.sigmoid(y_pred) > thr).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    return (2. * intersection / (union + 1e-6)).item()


# =========================
# Training function
# =========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on {device}")

    # Dataset
    train_dataset = BrainMRIDataset(
        os.path.join(args.data, "train", "images"),
        os.path.join(args.data, "train", "masks")
    )
    val_dataset = BrainMRIDataset(
        os.path.join(args.data, "valid", "images"),
        os.path.join(args.data, "valid", "masks")
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # Model
    if args.model == "unet":
        model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    elif args.model == "nasunet":
        model = NASUNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base,
            groups=args.groups
        ).to(device)
    else:
        raise ValueError("❌ Model must be either 'unet' or 'nasunet'")

    # Loss & Optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                val_iou += iou_score(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"📊 Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save checkpoints
        os.makedirs(args.ckpt, exist_ok=True)
        last_path = os.path.join(args.ckpt, f"{args.model}_last.pt")
        best_path = os.path.join(args.ckpt, f"{args.model}_best.pt")

        torch.save(model.state_dict(), last_path)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved new best model with IoU {best_iou:.4f}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", help="Model: unet | nasunet")
    parser.add_argument("--data", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=32, help="Base channels for UNet/NASUNet")
    parser.add_argument("--groups", type=int, default=8, help="Group convs for NASUNet")
    parser.add_argument("--ckpt", type=str, default="./checkpoints", help="Save checkpoint dir")
    args = parser.parse_args()

    train(args)
