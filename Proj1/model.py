import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNetwork(nn.Module):
    def __init__(self):
        super(BaselineNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 16 * 2, 2, 1)  # 2 input channels (one for each digit image), each is 14 x 14
        self.conv2 = nn.Conv2d(16 * 2, 32 * 2, 2, 1)  # 32 input channels, each is 13 x 13
        self.fc1 = nn.Linear(32 * 2 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


