# datasets.py
import os, numpy as np, torch
from torch.utils.data import Dataset

class BrainMRINumpyDataset(Dataset):
    """
    Đọc .npy lazy từ processed/<split>/(images|masks)
    Trả về tensor (B,1,H,W), ảnh [0,1], mask {0,1}
    """
    def __init__(self, img_dir, mask_dir):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.img_files  = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])
        assert len(self.img_files)==len(self.mask_files)>0, "No npy pairs found"

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir,  self.img_files[idx])) # uint8 HxW
        msk = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))# uint8/0-1 HxW
        img = (img.astype(np.float32) / 255.0)[None, ...]              # 1xHxW
        msk = (msk>0).astype(np.float32)[None, ...]
        return torch.from_numpy(img), torch.from_numpy(msk)
