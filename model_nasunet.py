# model_nasunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_gn_groups(num_channels: int, groups: int) -> int:
    """
    GroupNorm yêu cầu num_channels % num_groups == 0.
    Hàm này tự điều chỉnh num_groups cho hợp lệ.
    """
    g = min(groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(1, g)


class DWConvGN(nn.Module):
    """
    Khối Depthwise(3x3) -> Pointwise(1x1) + GroupNorm + ReLU
    - depthwise giúp giảm tham số
    - group norm ổn định trên batch nhỏ
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        gn_groups = _safe_gn_groups(out_ch, groups)
        self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class cSE(nn.Module):
    """
    Channel Squeeze-and-Excitation: học trọng số theo kênh (weighted skip)
    """
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(ch // r, 1)
        self.fc1 = nn.Conv2d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, ch, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DownCell(nn.Module):
    """
    Cell down-sampling: 2 nhánh
      - path1: DWConvGN(stride=2) + cSE
      - path2: MaxPool2d(2) -> DWConvGN
    Sau đó concat -> conv 1x1 để trộn kênh.
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.path1 = nn.Sequential(
            DWConvGN(in_ch, out_ch, stride=2, groups=groups),
            cSE(out_ch)
        )
        self.path2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DWConvGN(in_ch, out_ch, stride=1, groups=groups)
        )
        self.merge = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        x = torch.cat([p1, p2], dim=1)
        x = self.merge(x)
        return x


class UpCell(nn.Module):
    """
    Cell up-sampling:
      - ConvTranspose2d để upsample
      - cSE trên skip (weighted skip)
      - concat rồi qua 2 khối DWConvGN
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_gate = cSE(skip_ch)
        self.conv = nn.Sequential(
            DWConvGN(out_ch + skip_ch, out_ch, stride=1, groups=groups),
            DWConvGN(out_ch, out_ch, stride=1, groups=groups),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # căn chỉnh kích thước (phòng trường hợp lệch 1px do chia lẻ)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.skip_gate(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class NASUNet(nn.Module):
    """
    NAS-UNet tối giản theo tinh thần NAS (dùng primitive gọn nhẹ),
    U-like encoder-decoder với weighted skip (cSE) và depthwise separable conv.

    Tham số khởi tạo linh hoạt:
      - NASUNet(in_ch=1, out_ch=1, base=32, groups=8)
      - NASUNet(in_channels=1, out_channels=1, base_channels=32, groups=8)
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, groups: int = 8, **kwargs):
        super().__init__()
        # Hỗ trợ alias tên tham số
        if "in_channels" in kwargs:
            in_ch = kwargs["in_channels"]
        if "out_channels" in kwargs:
            out_ch = kwargs["out_channels"]
        if "base_channels" in kwargs:
            base = kwargs["base_channels"]

        # Stem
        self.stem = nn.Sequential(
            DWConvGN(in_ch, base, stride=1, groups=groups),
            DWConvGN(base, base, stride=1, groups=groups)
        )

        # Encoder (Down)
        self.d1 = DownCell(base,     base * 2,  groups)
        self.d2 = DownCell(base * 2, base * 4,  groups)
        self.d3 = DownCell(base * 4, base * 8,  groups)
        self.d4 = DownCell(base * 8, base * 16, groups)

        # Decoder (Up)
        self.u4 = UpCell(base * 16, base * 8,  base * 8,  groups)
        self.u3 = UpCell(base * 8,  base * 4,  base * 4,  groups)
        self.u2 = UpCell(base * 4,  base * 2,  base * 2,  groups)
        self.u1 = UpCell(base * 2,  base,      base,      groups)

        # Head
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Khởi tạo Kaiming phù hợp cho ReLU
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)   # b
        x1 = self.d1(x0)    # 2b, 1/2
        x2 = self.d2(x1)    # 4b, 1/4
        x3 = self.d3(x2)    # 8b, 1/8
        x4 = self.d4(x3)    # 16b,1/16

        # Decoder
        y3 = self.u4(x4, x3)   # 8b
        y2 = self.u3(y3, x2)   # 4b
        y1 = self.u2(y2, x1)   # 2b
        y0 = self.u1(y1, x0)   # b

        logits = self.out_conv(y0)  # (B, out_ch, H, W)
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = NASUNet(in_ch=1, out_ch=1, base=32, groups=8)
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
