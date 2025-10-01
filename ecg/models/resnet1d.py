import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k//2
        self.conv = nn.Conv1d(c_in, c_out, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResidualUnit1D(nn.Module):
    def __init__(self, c_in, c_out, k=16, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=k//2, bias=False)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.act1  = nn.ReLU(inplace=True)
        self.do1   = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.conv2 = nn.Conv1d(c_out, c_out, k, stride=stride, padding=k//2, bias=False)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.act2  = nn.ReLU(inplace=True)
        self.do2   = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.pool_skip = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True) if stride>1 else None
        self.proj = nn.Conv1d(c_in, c_out, 1, bias=False) if c_in!=c_out else None

    def forward(self, x):
        y = x
        if self.pool_skip is not None:
            y = self.pool_skip(y)
        if self.proj is not None:
            y = self.proj(y)
        out = self.conv1(x); out = self.bn1(out); out = self.act1(out); out = self.do1(out)
        out = self.conv2(out); out = self.bn2(out)
        out = out + y
        out = self.act2(out); out = self.do2(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, n_classes=6, in_ch=12, stem=64, blocks=None, k=16, dropout=0.2, use_gap=True):
        super().__init__()
        if blocks is None:
            blocks=[{"out_channels":128,"stride":4},{"out_channels":196,"stride":4},
                    {"out_channels":256,"stride":4},{"out_channels":320,"stride":4}]
        self.stem = ConvBNAct(in_ch, stem, k, s=1, p=k//2)
        layers = []
        c_in = stem
        for b in blocks:
            layers.append(ResidualUnit1D(c_in, b["out_channels"], k=k, stride=b["stride"], dropout=dropout))
            c_in = b["out_channels"]
        self.backbone = nn.Sequential(*layers)
        self.use_gap = use_gap
        if use_gap:
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c_in, n_classes))
        else:
            # flatten over time; assumes final length 16 if using the provided strides
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in*16, n_classes))

    def forward(self, x):  # x: [B, 12, 4096]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)   # logits
        return x
