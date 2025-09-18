import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import math
import datetime
import  config




# 3D CNN模型
class SimpleCNN(nn.Module):
    """
    简单的CNN baseline模型
    只用几层基础卷积，适合作为baseline
    """

    def __init__(self, configs):
        super(SimpleCNN, self).__init__()

        self.input_channels = configs.input_dim  # 6
        self.output_channels = configs.output_dim  # 6

        # 简单的2D CNN (逐帧处理)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, self.output_channels, kernel_size=3, padding=1)

        # 简单的批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        """
        简单的前向传播
        输入: (batch_size, seq_len, channels, height, width)
        输出: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape

        # 逐帧处理：将时间维度合并到batch维度
        x = x.view(batch_size * seq_len, channels, height, width)

        # 简单的卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv4(x)

        # 恢复时间维度
        x = x.view(batch_size, seq_len, self.output_channels, height, width)

        return x