import torch.nn as nn

class TaskNetwork(nn.Module):
    def __init__(self, in_channels=128, num_classes=80):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)
