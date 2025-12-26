
import torch
import torch.nn as nn
from open_clip.open_clip import create_model_and_transforms
from open_clip.training import get_ecg_encoder






class ECGEncoderWrapper(nn.Module):
    def __init__(self, model_name="coca_ViT-B-32", pretrained="epoch_20.pt"):
        super().__init__()
        print(f"Loading ECG-Chat model: {model_name} from {pretrained}")

        # 1. 加载模型
        self.model,_,self.config = get_ecg_encoder(model_name, pretrained,device='cpu')
        ecg_config = self.config.get('ecg_cfg', {})
        self.hidden_size = ecg_config.get('width', 768)
        self.seq_length = ecg_config.get('seq_length', 5000)
        self.patch_size = ecg_config.get('patch_size', 50)

        # 2. 冻结参数 (Backbone 不参与训练)
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, ecg_signal):
        """
        输入: ecg_signal [Batch, Channels, Length]
        输出: features [Batch, Seq_Len, Dim]
        """
        with torch.no_grad():
            ecg_features = self.model(ecg_signal,output_last_transformer_layer=True)

        return ecg_features
if __name__ == '__main__':
    model = ECGEncoderWrapper(pretrained="/Users/zhangyf/Documents/cfel/epoch_20.pt")
    result = model(torch.randn(1, 12, 5000))
    print(result.shape)

