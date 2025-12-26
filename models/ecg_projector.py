import torch
import torch.nn as nn
import re

from transformers import MistralForQuestionAnswering
from configuration_deepseek_v2 import DeepseekV2Config

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_ecg_projector(ecg_embed,n_embed):
    ecg_embed = 768
    n_embed = 1280
    modules = [nn.Linear(ecg_embed, n_embed)]
    for _ in range(1, 2):
        modules.append(nn.GELU())
        modules.append(nn.Linear(n_embed, n_embed))
    return nn.Sequential(*modules)

if __name__ == '__main__':
    class DeepseekOCRConfig(DeepseekV2Config):
        model_type = "DeepseekOCR"
    config = DeepseekOCRConfig()
    proj = build_ecg_projector(config)
    print(proj)
