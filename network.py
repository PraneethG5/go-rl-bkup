import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(5, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(6)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, 1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.Linear(2*9*9, 82)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(9*9, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)
