import torch
import numpy as np
import random
from scipy.interpolate import interp1d


class BaselineWander(torch.nn.Module):
    def __init__(self, fs=500, p=0.5, max_amplitude=0.5):
        super().__init__()
        self.fs = fs
        self.p = p
        self.max_amplitude = max_amplitude

    def forward(self, sample):
        # sample shape: [Channels, Length]
        if random.random() > self.p:
            return sample

        # 简单的基线漂移实现
        c, l = sample.shape
        # 生成低频噪声
        x = np.linspace(0, 1, 10)  # 锚点
        y = np.random.uniform(-self.max_amplitude, self.max_amplitude, (c, 10))
        f = interp1d(x, y, kind='cubic', axis=1)
        x_new = np.linspace(0, 1, l)
        drift = torch.tensor(f(x_new), dtype=sample.dtype, device=sample.device)

        return sample + drift


class RandomMasking(torch.nn.Module):
    def __init__(self, fs=500, p=0.5, mask_ratio=0.1):
        super().__init__()
        self.fs = fs
        self.p = p
        self.mask_ratio = mask_ratio

    def forward(self, sample):
        if random.random() > self.p:
            return sample

        c, l = sample.shape
        mask_len = int(l * self.mask_ratio)
        start = random.randint(0, l - mask_len)

        # 将一段信号置零
        mask = torch.ones_like(sample)
        mask[:, start:start + mask_len] = 0
        return sample * mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x