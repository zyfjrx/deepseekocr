import sys
import os
import torch

# 将当前脚本所在的目录添加到 python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import open_clip
except ImportError as e:
    print(f"Error: Failed to import open_clip.")
    print(f"Details: {e}")
    # Print traceback for deeper debugging
    import traceback

    traceback.print_exc()

    print("\nCheck if you are running this script from the correct environment and directory.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def main():
    model_name = 'coca_ViT-B-32'
    print(f"Creating model '{model_name}' to inspect architecture...")

    try:
        # 创建模型 (不需要加载预训练权重，只需要架构)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
            force_custom_text=True  # Ensure we use the custom config logic
        )
        print(model)

        print("\n" + "=" * 30 + " ECG Encoder Architecture " + "=" * 30 + "\n")

        # 根据之前的分析，ECG-Chat 的 CoCa 模型将 ECG 编码器存储在 'ecg' 属性中
        if hasattr(model, 'ecg'):
            print(model.ecg)

            # 打印一些关键参数统计
            print("\n" + "=" * 30 + " Summary " + "=" * 30)
            total_params = sum(p.numel() for p in model.ecg.parameters())
            print(f"Total Parameters in ECG Encoder: {total_params:,}")
            print(f"Input Sequence Length: {model.ecg.seq_length}")
            print(f"Patch Size: {model.ecg.patch_size}")
            print(f"Embedding Width: {model.ecg.transformer.width}")
            print(f"Transformer Layers: {model.ecg.transformer.layers}")

        else:
            print("Error: Could not find 'ecg' attribute in the model.")
            print("Model keys:", model.__dict__.keys())

    except Exception as e:
        print(f"Error creating model: {e}")
        print("\nAvailable models:", open_clip.list_models())


if __name__ == "__main__":
    main()