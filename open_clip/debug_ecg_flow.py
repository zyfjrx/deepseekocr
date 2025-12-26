import sys
import os
import torch
import numpy as np
import torch.nn as nn

# 将当前脚本所在的目录添加到 python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import open_clip
    from open_clip.ecg_transform import Normalize, Resize, Compose
except ImportError as e:
    print(f"Error: Failed to import open_clip. {e}")
    sys.exit(1)

def get_dummy_ecg_data(channels=12, length=5000):
    """模拟一个随机的 ECG 信号 (channels, length)"""
    print(f"Generating dummy ECG data: shape=({channels}, {length})")
    return np.random.randn(channels, length).astype(np.float32)

def register_hooks(model):
    """注册 hook 以打印每层的输出形状"""
    hooks = []
    
    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        # 处理不同类型的输出
        if isinstance(output, tuple):
            out_shape = [tuple(o.shape) for o in output if isinstance(o, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            out_shape = tuple(output.shape)
        else:
            out_shape = "Non-Tensor"
            
        print(f"  --> [Layer: {class_name}] Output Shape: {out_shape}")

    # Hook 关键层
    # 1. Conv1d (Patch Embedding)
    hooks.append(model.conv1.register_forward_hook(hook_fn))
    
    # 2. Transformer Blocks (打印第一层和最后一层)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
        hooks.append(model.transformer.resblocks[0].register_forward_hook(hook_fn))
        hooks.append(model.transformer.resblocks[-1].register_forward_hook(hook_fn))
    
    # 3. LayerNorm (Post-Norm)
    hooks.append(model.ln_post.register_forward_hook(hook_fn))
    
    return hooks

def main():
    # 1. 准备数据
    raw_ecg = get_dummy_ecg_data()
    
    # 2. 预处理 (模拟 inference 时的 transform)
    # 注意：这里的 mean/std 是示例值，实际应使用 constants.py 中的 MIMIC_IV_MEAN 等
    transform = Compose([
        Normalize(mean=[0]*12, std=[1]*12),
        Resize(seq_length=5000)
    ])
    
    # 转换为 Tensor 并增加 Batch 维度: (12, 5000) -> (1, 12, 5000)
    ecg_tensor = torch.from_numpy(raw_ecg).unsqueeze(0)
    ecg_tensor = transform(ecg_tensor)
    
    print(f"\nPreprocessed Input Tensor Shape: {ecg_tensor.shape}")
    
    # 3. 加载模型
    model_name = 'coca_ViT-B-32'
    print(f"\nLoading model '{model_name}'...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=None,
        force_custom_text=True
    )
    ecg_encoder = model.ecg
    ecg_encoder.eval()
    
    # 4. 注册 Hook 监控数据流
    print("\n" + "="*20 + " Forward Pass Trace " + "="*20)
    hooks = register_hooks(ecg_encoder)
    
    # 5. 前向传播
    with torch.no_grad():
        output = ecg_encoder(ecg_tensor)
        
    print("="*60)
    print(f"Final Embedding Shape: {output.shape}")

if __name__ == "__main__":
    main()