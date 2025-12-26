from modeling_deepseekocr import DeepseekOCRConfig


class DeepseekECGOCRConfig(DeepseekOCRConfig):
    model_type = "deepseek_ecg_ocr"  # 定义一个新的模型类型名称

    def __init__(
            self,
            # === 新增 ECG 相关参数 ===
            ecg_seq_length= 5000,
            ecg_lead_num= 12,
            ecg_layers= 12,
            ecg_width= 768,
            ecg_patch_size= 50,
            ecg_output_tokens= True,
            ecg_model_name="coca_ViT-B-32",
            ecg_projector_type="mlp2x_gelu",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ecg_model_name = ecg_model_name
        self.ecg_projector_type = ecg_projector_type
        self.ecg_seq_length = ecg_seq_length
        self.ecg_lead_num = ecg_lead_num
        self.ecg_layers = ecg_layers
        self.ecg_width = ecg_width
        self.ecg_patch_size = ecg_patch_size
        self.ecg_output_tokens = ecg_output_tokens

