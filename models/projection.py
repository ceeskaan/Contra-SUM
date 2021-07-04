import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1024, 128, 11, stride=2, bias=False, groups=128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 3, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )
        
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.mean(-1)
        x = self.fc(x)
        return x

