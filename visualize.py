# visualize.py
import os, argparse, numpy as np, cv2, matplotlib.pyplot as plt

def overlay(gray, mask, alpha=0.4):
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(gray3); color[:,:,2] = 255  # đỏ
    mask3 = np.stack([mask]*3, axis=-1).astype(np.uint8)
    over = cv2.addWeighted(gray3, 1.0, (color*mask3), alpha, 0)
    return over

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_npy", required=True) # path ảnh gốc npy
    ap.add_argument("--gt_npy",  required=False) # path mask GT npy
    ap.add_argument("--pred_npy",required=True)  # path mask dự đoán npy
    args = ap.parse_args()

    img  = np.load(args.img_npy).astype(np.uint8)
    pred = (np.load(args.pred_npy)>0).astype(np.uint8)
    over = overlay(img, pred, 0.4)

    plt.figure(); plt.title("Image"); plt.imshow(img, cmap="gray"); plt.axis("off")
    if args.gt_npy:
        gt = (np.load(args.gt_npy)>0).astype(np.uint8)
        plt.figure(); plt.title("GT"); plt.imshow(gt, cmap="gray"); plt.axis("off")
    plt.figure(); plt.title("Pred"); plt.imshow(pred, cmap="gray"); plt.axis("off")
    plt.figure(); plt.title("Overlay"); plt.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.show()
