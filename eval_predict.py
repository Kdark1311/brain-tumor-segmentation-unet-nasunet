# eval_predict.py
import os, argparse, numpy as np, torch, cv2
from torch.utils.data import DataLoader
from datasets import BrainMRINumpyDataset
from model_unet import UNet
from model_nasunet import NASUNet
from losses import BCEDiceLoss, iou_score, dice_coeff
from tqdm import tqdm
from scipy.ndimage import binary_opening, binary_closing

@torch.no_grad()
def predict_and_save(model, loader, device, out_dir, thr=0.5, post=False):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idx=0
    for x,_ in tqdm(loader, desc="Predict"):
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        for p in probs:
            m = (p[0]>thr).astype(np.uint8)
            if post:
                m = binary_opening(m, structure=np.ones((3,3))).astype(np.uint8)
                m = binary_closing(m, structure=np.ones((3,3))).astype(np.uint8)
            np.save(os.path.join(out_dir, f"pred_{idx}.npy"), m)
            idx+=1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model=="unet":
        model = UNet(in_ch=1, out_ch=1, base=args.base)
    else:
        model = NASUNet(in_ch=1, out_ch=1, base=args.base, groups=args.groups)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # loaders
    test_img = os.path.join(args.data, "test", "images")
    test_msk = os.path.join(args.data, "test", "masks")  # nếu có ground-truth
    ds = BrainMRINumpyDataset(test_img, test_msk if os.path.exists(test_msk) else test_img)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # đánh giá nếu có GT
    if os.path.exists(test_msk):
        loss_fn = BCEDiceLoss()
        loss = 0; iou=0; dice=0; n=0
        model.eval()
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss += loss_fn(logits,y).item()*x.size(0)
            iou  += iou_score(logits,y).item()*x.size(0)
            dice += dice_coeff(logits,y).item()*x.size(0)
            n += x.size(0)
        print(f"TEST loss {loss/n:.4f} dice {dice/n:.4f} iou {iou/n:.4f}")

    # predict & lưu
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    predict_and_save(model, loader, device, args.out, thr=args.thr, post=args.post)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r"D:\Progamming\Progamming_courses\Python\Segmentation_project\archive\processed")
    ap.add_argument("--model", type=str, choices=["unet","nasunet"], default="nasunet")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base",  type=int, default=32)
    ap.add_argument("--groups",type=int, default=8)
    ap.add_argument("--thr",   type=float, default=0.5)
    ap.add_argument("--post",  action="store_true", help="morphology opening/closing nhỏ")
    ap.add_argument("--out",   type=str, default="./pred_masks_npy")
    args = ap.parse_args()
    main(args)
