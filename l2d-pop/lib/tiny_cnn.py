import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """A deliberately small conv net (≈ 70 k params).

    Default input: 3 × 16 × 16 or 3 × 32 × 32
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16→8  (or 32→16)

            # Block 2
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8→4   (or 16→8)

            # # Block 3 (optional extra depth)
            # nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)              # [B, 128, 4, 4] (for 16×16 input)
        x = F.adaptive_avg_pool2d(x, 1)   # [B, 128, 1, 1]
        x = torch.flatten(x, 1)           # [B, 128]
        return x     # logits


# ---------- quick sanity check ----------
if __name__ == "__main__":
    net = TinyCNN(num_classes=10, in_channels=3)
    dummy = torch.randn(4, 3, 16, 16)     # batch of 4
    out = net(dummy)
    print(out.shape)                      # -> torch.Size([4, 10])
