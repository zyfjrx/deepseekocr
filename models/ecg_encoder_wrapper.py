import os

import torch
import torch.nn as nn
import sys

from open_clip.open_clip import create_model_and_transforms


class ECGEncoderWrapper(nn.Module):
    def __init__(self, model_name="coca_ViT-B-32", pretrained="path/to/ecg_chat.pt"):
        super().__init__()
        print(f"Loading ECG-Chat model: {model_name} from {pretrained}")

        # 1. 加载模型
        self.model, _, _ = create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # 2. 冻结参数 (Backbone 不参与训练)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. 获取输出维度
        if hasattr(self.model, 'ecg'):
            self.hidden_size = self.model.ecg.output_dim
        else:
            self.hidden_size = 512  # Fallback

    def forward(self, ecg_signal):
        """
        输入: ecg_signal [Batch, Channels, Length]
        输出: features [Batch, Seq_Len, Dim]
        """
        with torch.no_grad():
            ecg_latent, token_embs = self.model.ecg(ecg_signal,output_last_transformer_layer=True)

        return token_embs
if __name__ == '__main__':
    model = ECGEncoderWrapper(pretrained="/Users/zhangyf/Documents/cfel/epoch_20.pt")
    result = model(torch.randn(1, 12, 5000))
    print(result.shape)

