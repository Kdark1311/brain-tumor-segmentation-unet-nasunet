    # losses.py
import torch
import torch.nn as nn

def dice_coeff(y_pred, y_true, eps=1e-6):
    # y_pred,y_true: (B,1,H,W), y_pred sigmoid logits/ probs
    y_pred = torch.sigmoid(y_pred)
    y_true = (y_true>0.5).float()
    inter = (y_pred*y_true).sum(dim=(1,2,3))
    union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean()

class DiceLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, logits, targets):
        return 1.0 - dice_coeff(logits, targets)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w = bce_weight
    def forward(self, logits, targets):
        return self.w*self.bce(logits, targets) + (1-self.w)*self.dice(logits, targets)

def iou_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs>thresh).float()
    targets = (targets>0.5).float()
    inter = (preds*targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - inter
    return ((inter+eps)/(union+eps)).mean()
