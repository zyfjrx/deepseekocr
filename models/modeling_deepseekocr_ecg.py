from .ecg_projector import build_ecg_projector
from .modeling_deepseekv2 import DeepseekV2Model, DeepseekV2ForCausalLM
from .configuration_deepseek_v2 import DeepseekV2Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import requests
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict
from transformers import TextStreamer
from .conversation import get_conv_template
from abc import ABC
import math
import re
from tqdm import tqdm
import numpy as np
import time
from models.ecg_encoder import ECGEncoderWrapper
import scipy.signal
try:
    import wfdb
except ImportError:
    wfdb = None


def load_image(image_path):
    try:
        image = Image.open(image_path)

        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image

    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):
    image_width, image_height = image.size

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    # try:
    # except IOError:
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 20) 
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                       fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):
    result_image = draw_bounding_boxes(image, ref_texts, output_path)

    return result_image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


def format_messages(
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t


def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()

            # pil_img = Image.open(image_path)
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            normalize: bool = True
    ):
        self.mean = mean
        self.std = std

        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x):
        x = self.transform(x)
        return x


class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class DeepseekOCRConfig(DeepseekV2Config):
    model_type = "DeepseekOCR"
    def __init__(self,
        ecg_checkpoint = "/Users/zhangyf/Documents/cfel/epoch_20.pt",
        mm_projector_type = "mlp2x_gelu",
        ecg_hidden_size = 768,
        ** kwargs
    ):
        super(DeepseekOCRConfig, self).__init__(**kwargs)
        # 4. 将新参数赋值给 self (实例属性)
        self.ecg_checkpoint = ecg_checkpoint
        self.mm_projector_type = mm_projector_type
        self.ecg_hidden_size = ecg_hidden_size



class DeepseekOCRModel(DeepseekV2Model):
    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        # self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2)
        n_embed = 1280
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        # ================= [新增整合部分] =================
        # ECG 信号编码器
        self.ecg_encoder = ECGEncoderWrapper(pretrained=config.ecg_checkpoint)
        # ECG Projector (对齐层)
        # 将 ECG 维度 (768) 映射到 DeepSeek 维度 (config.hidden_size=1280)
        self.ecg_projector = build_ecg_projector(config)

        # 3. 初始化可学习的 Start/End 向量 (Soft Embeddings)
        # 这里的 std 参考了 DeepSeek 的初始化方式
        ecg_embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.ecg_start_embed = nn.Parameter(torch.randn(n_embed) * ecg_embed_std)
        self.ecg_end_embed = nn.Parameter(torch.randn(n_embed) * ecg_embed_std)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.FloatTensor] = None,
            images_spatial_crop: Optional[torch.FloatTensor] = None,
            # [新增] ECG 相关参数
            ecg_values: Optional[torch.FloatTensor] = None,  # [Batch, 12, 5000]
            ecg_seq_mask: Optional[torch.BoolTensor] = None,  # [Batch, Seq_Len]
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sam_model = getattr(self, 'sam_model', None)
        # sam_model = self.sam_model
        vision_model = getattr(self, 'vision_model', None)

        if sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            idx = 0

            # sam_model = torch.jit.script(sam_model)

            # start_time = time.time()
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]

                # with torch.no_grad():
                # with torch.inference_mode(): 

                if torch.sum(patches).item() != 0:
                    # P, C, H, W = patches.shape
                    crop_flag = 1
                    local_features_1 = sam_model(patches)
                    local_features_2 = vision_model(patches, local_features_1)
                    # vit_time = time.time()
                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)),
                                               dim=-1)
                    local_features = self.projector(local_features)
                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)
                    print('=====================')
                    print('BASE: ', global_features.shape)
                    print('PATCHES: ', local_features.shape)
                    print('=====================')
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)
                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )
                    global_features = global_features.view(-1, n_dim)
                    local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2,
                                                                                                                  1, 3,
                                                                                                                  4).reshape(
                        height_crop_num * h2, width_crop_num * w2, n_dim2)
                    local_features = torch.cat(
                        [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)],
                        dim=1
                    )
                    local_features = local_features.view(-1, n_dim2)
                    global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]],
                                                      dim=0)
                    # end_time = time.time()
                    # print('sam: ', sam_time - start_time)
                    # print('vit: ', vit_time - sam_time)
                    # print('all: ', end_time - start_time)
                    # exit()

                else:
                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)
                    print('=====================')
                    print('BASE: ', global_features.shape)
                    print('NO PATCHES')
                    print('=====================')
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                images_in_this_batch.append(global_local_features)

                # print(inputs_embeds.shape)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    # exit()

                    inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).to(self.device),
                                                       images_in_this_batch)

                idx += 1

        if ecg_values is not None and self.ecg_encoder is not None and (input_ids.shape[1] != 1 or self.training):
            # A. 提取特征 (冻结参数)
            # 输出: [Batch, Seq_Len_ECG, 768]
            ecg_features = self.ecg_encoder(ecg_values.to(inputs_embeds.device))

            # B. 投影对齐 (可训练参数)
            # 输出: [Batch, Seq_Len_ECG, Hidden_Size]
            ecg_features = self.ecg_projector(ecg_features.to(inputs_embeds.dtype))

            # 测试
            # ecg_features = ecg_features * 0.0

            # C. 拼接 Start/End 向量
            batch_size = ecg_features.shape[0]

            # 扩展可学习向量到 [Batch, 1, Hidden]
            start_token = self.ecg_start_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            end_token = self.ecg_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)

            # 最终特征序列: [Start, F1, F2, ..., Fn, End]
            # 长度 = 1 + ecg_len + 1
            ecg_final_embeds = torch.cat([
                start_token.to(ecg_features.dtype),
                ecg_features,
                end_token.to(ecg_features.dtype)
            ], dim=1)

            print('ECG: ', ecg_final_embeds.shape)

            # D. 填入 inputs_embeds
            for idx in range(len(inputs_embeds)):
                # 检查当前样本是否有 ECG 占位符
                if ecg_seq_mask[idx].any():
                    # 目标填入区域数量
                    num_slots = ecg_seq_mask[idx].sum()
                    # 准备好的特征数量
                    num_feats = ecg_final_embeds.shape[1]

                    if num_slots == num_feats:
                        inputs_embeds[idx].masked_scatter_(
                            ecg_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device),
                            ecg_final_embeds[idx]
                        )
                    else:
                        # 简单的容错: 如果长度对不上 (比如 dataset 里的 token_len 设错了)，打印警告或截断
                        print(f"ECG Size Mismatch: Mask {num_slots} vs Feat {num_feats}")


            # # C. 填入 inputs_embeds
            # # 逻辑：在 ecg_seq_mask 为 True 的位置，用 ecg_features 替换掉原来的 text embedding
            # if ecg_seq_mask is not None:
            #     for idx in range(len(inputs_embeds)):
            #         # 确保当前样本有 ECG 数据
            #         if torch.sum(ecg_seq_mask[idx]) > 0:
            #             # 展平特征以匹配 mask
            #             # 注意：mask 的 True 数量必须等于 ecg_features 的序列长度
            #             # 如果不一致，这里需要根据实际情况做截断或 Padding
            #             valid_features = ecg_features[idx]
            #
            #             # 执行替换
            #             inputs_embeds[idx].masked_scatter_(
            #                 ecg_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device),
            #                 valid_features
            #             )
            print('=====================')
            print(inputs_embeds.shape)
            print('=====================')

        return super(DeepseekOCRModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids=position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):
    config_class = DeepseekOCRConfig

    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCRModel(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.FloatTensor] = None,
            images_spatial_crop: Optional[torch.FloatTensor] = None,
            # [新增] ECG Args
            ecg_values: Optional[torch.FloatTensor] = None,
            ecg_seq_mask: Optional[torch.BoolTensor] = None,
            return_dict: Optional[bool] = None,

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            # ECG 参数
            ecg_values=ecg_values,
            ecg_seq_mask=ecg_seq_mask,
            return_dict=return_dict

        )

        # print(transformer_outputs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            print(f"loss:==========={loss}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if self.generation_config.cache_implementation == "static":
        #     # generation with static cache
        #     cache_position = kwargs.get("cache_position", None)
        #     if cache_position is None:
        #         past_length = 0
        #     else:
        #         past_length = cache_position[-1] + 1
        #     input_ids = input_ids[:, past_length:]
        #     position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
                # [新增] ECG
                "ecg_values": kwargs.get("ecg_values", None),
                "ecg_seq_mask": kwargs.get("ecg_seq_mask", None),
            }
        )
        return model_inputs

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    # def infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True,
    #           test_compress=False, save_results=False, eval_mode=False):
    #     self.disable_torch_init()
    #     device = self.model.device
    #
    #     os.makedirs(output_path, exist_ok=True)
    #     os.makedirs(f'{output_path}/images', exist_ok=True)
    #
    #     if prompt and image_file:
    #         conversation = [
    #             {
    #                 "role": "<|User|>",
    #                 # "content": "<image>\n<|grounding|>Given the layout of the image. ",
    #                 "content": f'{prompt}',
    #                 # "content": "君不见黄河之水天上来的下一句是什么？",
    #                 # "content": "<image>\nFree OCR. ",
    #                 # "content": "<image>\nParse the figure. ",
    #                 # "content": "<image>\nExtract the text in the image. ",
    #                 "images": [f'{image_file}'],
    #             },
    #             {"role": "<|Assistant|>", "content": ""},
    #         ]
    #
    #     elif prompt:
    #         conversation = [
    #             {
    #                 "role": "<|User|>",
    #                 # "content": "<image>\n<|grounding|>Given the layout of the image. ",
    #                 "content": f'{prompt}',
    #                 # "content": "君不见黄河之水天上来的下一句是什么？",
    #                 # "content": "<image>\nFree OCR. ",
    #                 # "content": "<image>\nParse the figure. ",
    #                 # "content": "<image>\nExtract the text in the image. ",
    #                 # "images": [f'{image_file}'],
    #             },
    #             {"role": "<|Assistant|>", "content": ""},
    #         ]
    #     else:
    #         assert False, f'prompt is none!'
    #
    #     prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')
    #
    #     patch_size = 16
    #     downsample_ratio = 4
    #     images = load_pil_images(conversation)
    #
    #     valid_img_tokens = 0
    #     ratio = 1
    #
    #     image_draw = images[0].copy()
    #
    #     w, h = image_draw.size
    #     # print(w, h)
    #     ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    #
    #     image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    #     images_seq_mask = []
    #
    #     image_token = '<image>'
    #     image_token_id = 128815
    #     text_splits = prompt.split(image_token)
    #
    #     images_list, images_crop_list, images_seq_mask = [], [], []
    #     tokenized_str = []
    #     images_spatial_crop = []
    #     for text_sep, image in zip(text_splits, images):
    #
    #         tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
    #         tokenized_str += tokenized_sep
    #         images_seq_mask += [False] * len(tokenized_sep)
    #
    #         if crop_mode:
    #
    #             if image.size[0] <= 640 and image.size[1] <= 640:
    #                 crop_ratio = [1, 1]
    #
    #             else:
    #                 if crop_mode:
    #                     # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
    #                     images_crop_raw, crop_ratio = dynamic_preprocess(image)
    #                 else:
    #                     # best_width, best_height = self.image_size, self.image_size
    #                     crop_ratio = [1, 1]
    #
    #             """process the global view"""
    #             # image = image.resize((base_size, base_size))
    #             global_view = ImageOps.pad(image, (base_size, base_size),
    #                                        color=tuple(int(x * 255) for x in image_transform.mean))
    #
    #             if base_size == 1024:
    #                 valid_img_tokens += int(256 * ratio)
    #             elif base_size == 1280:
    #                 valid_img_tokens += int(400 * ratio)
    #             # elif base_size == 640:
    #             #     valid_img_tokens += int(100 * ratio)
    #
    #             images_list.append(image_transform(global_view).to(torch.bfloat16))
    #
    #             # global_view_tensor = image_transform(global_view).to(torch.bfloat16)
    #
    #             width_crop_num, height_crop_num = crop_ratio
    #
    #             images_spatial_crop.append([width_crop_num, height_crop_num])
    #
    #             if width_crop_num > 1 or height_crop_num > 1:
    #                 """process the local views"""
    #
    #                 for i in range(len(images_crop_raw)):
    #                     images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
    #
    #             if image_size == 640:
    #                 valid_img_tokens += len(images_crop_list) * 100
    #
    #             num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    #             num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
    #
    #             """add image tokens"""
    #
    #             tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
    #             tokenized_image += [image_token_id]
    #             if width_crop_num > 1 or height_crop_num > 1:
    #                 tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
    #                         num_queries * height_crop_num)
    #             tokenized_str += tokenized_image
    #             images_seq_mask += [True] * len(tokenized_image)
    #             # num_image_tokens.append(len(tokenized_image))
    #
    #         else:
    #             # best_width, best_height = self.image_size, self.image_size
    #             # print(image.size, (best_width, best_height)) # check the select_best_resolutions func
    #
    #             """process the global view"""
    #             if image_size <= 640:
    #                 print('directly resize')
    #                 image = image.resize((image_size, image_size))
    #             # else:
    #             global_view = ImageOps.pad(image, (image_size, image_size),
    #                                        color=tuple(int(x * 255) for x in image_transform.mean))
    #             images_list.append(image_transform(global_view).to(torch.bfloat16))
    #
    #             if base_size == 1024:
    #                 valid_img_tokens += int(256 * ratio)
    #             elif base_size == 1280:
    #                 valid_img_tokens += int(400 * ratio)
    #             elif base_size == 640:
    #                 valid_img_tokens += int(100 * 1)
    #             elif base_size == 512:
    #                 valid_img_tokens += int(64 * 1)
    #
    #             width_crop_num, height_crop_num = 1, 1
    #
    #             images_spatial_crop.append([width_crop_num, height_crop_num])
    #
    #             """add image tokens"""
    #             num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    #
    #             tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
    #             tokenized_image += [image_token_id]
    #             # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
    #             #             num_queries * height_crop_num)
    #             tokenized_str += tokenized_image
    #             images_seq_mask += [True] * len(tokenized_image)
    #             # num_image_tokens.append(len(tokenized_image))
    #
    #     """process the last text split"""
    #     tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    #     tokenized_str += tokenized_sep
    #     images_seq_mask += [False] * len(tokenized_sep)
    #
    #     """add the bos tokens"""
    #     bos_id = 0
    #     tokenized_str = [bos_id] + tokenized_str
    #     images_seq_mask = [False] + images_seq_mask
    #
    #     input_ids = torch.LongTensor(tokenized_str)
    #
    #     images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
    #
    #     if len(images_list) == 0:
    #         images_ori = torch.zeros((1, 3, image_size, image_size))
    #         images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
    #         images_crop = torch.zeros((1, 3, base_size, base_size))
    #
    #     else:
    #         images_ori = torch.stack(images_list, dim=0)
    #         images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
    #         if images_crop_list:
    #             images_crop = torch.stack(images_crop_list, dim=0)
    #         else:
    #             images_crop = torch.zeros((1, 3, base_size, base_size))
    #
    #     if not eval_mode:
    #         streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    #         with torch.autocast(device.type, dtype=torch.bfloat16):
    #             with torch.no_grad():
    #                 output_ids = self.generate(
    #                     input_ids.unsqueeze(0).to(device),
    #                     images=[(images_crop.to(device), images_ori.to(device))],
    #                     images_seq_mask=images_seq_mask.unsqueeze(0).to(device),
    #                     images_spatial_crop=images_spatial_crop,
    #                     # do_sample=False,
    #                     # num_beams = 1,
    #                     temperature=0.0,
    #                     eos_token_id=tokenizer.eos_token_id,
    #                     streamer=streamer,
    #                     max_new_tokens=8192,
    #                     no_repeat_ngram_size=20,
    #                     use_cache=True
    #                 )
    #
    #     else:
    #         with torch.autocast(device.type, dtype=torch.bfloat16):
    #             with torch.no_grad():
    #                 output_ids = self.generate(
    #                     input_ids.unsqueeze(0).to(device),
    #                     images=[(images_crop.to(device), images_ori.to(device))],
    #                     images_seq_mask=images_seq_mask.unsqueeze(0).to(device),
    #                     images_spatial_crop=images_spatial_crop,
    #                     # do_sample=False,
    #                     # num_beams = 1,
    #                     temperature=0.0,
    #                     eos_token_id=tokenizer.eos_token_id,
    #                     max_new_tokens=8192,
    #                     no_repeat_ngram_size=35,
    #                     use_cache=True
    #                 )
    #
    #     if '<image>' in conversation[0]['content'] and eval_mode:
    #         outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).to(device).shape[1]:])
    #         stop_str = '<｜end▁of▁sentence｜>'
    #         if outputs.endswith(stop_str):
    #             outputs = outputs[:-len(stop_str)]
    #         # re_match
    #         outputs = outputs.strip()
    #
    #         return outputs
    #
    #     if '<image>' in conversation[0]['content'] and test_compress:
    #         outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).to(device).shape[1]:])
    #         pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
    #         print('=' * 50)
    #         print('image size: ', (w, h))
    #         print('valid image tokens: ', int(valid_img_tokens))
    #         print('output texts tokens (valid): ', pure_texts_outputs_token_length)
    #         print('compression ratio: ', round(pure_texts_outputs_token_length / valid_img_tokens, 2))
    #         print('=' * 50)
    #
    #     if '<image>' in conversation[0]['content'] and save_results:
    #         outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).to(device).shape[1]:])
    #         stop_str = '<｜end▁of▁sentence｜>'
    #
    #         print('=' * 15 + 'save results:' + '=' * 15)
    #
    #         # # # # conv.messages[-1][-1] = outputs
    #         if outputs.endswith(stop_str):
    #             outputs = outputs[:-len(stop_str)]
    #         outputs = outputs.strip()
    #
    #         matches_ref, matches_images, mathes_other = re_match(outputs)
    #         # print(matches_ref)
    #         result = process_image_with_refs(image_draw, matches_ref, output_path)
    #
    #         for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
    #             outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
    #
    #         for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
    #             outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    #
    #         # if 'structural formula' in conversation[0]['content']:
    #         #     outputs = '<smiles>' + outputs + '</smiles>'
    #         with open(f'{output_path}/result.mmd', 'w', encoding='utf-8') as afile:
    #             afile.write(outputs)
    #
    #         if 'line_type' in outputs:
    #             import matplotlib.pyplot as plt
    #             lines = eval(outputs)['Line']['line']
    #
    #             line_type = eval(outputs)['Line']['line_type']
    #             # print(lines)
    #
    #             endpoints = eval(outputs)['Line']['line_endpoint']
    #
    #             fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    #             ax.set_xlim(-15, 15)
    #             ax.set_ylim(-15, 15)
    #
    #             for idx, line in enumerate(lines):
    #                 try:
    #                     p0 = eval(line.split(' -- ')[0])
    #                     p1 = eval(line.split(' -- ')[-1])
    #
    #                     if line_type[idx] == '--':
    #                         ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
    #                     else:
    #                         ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
    #
    #                     ax.scatter(p0[0], p0[1], s=5, color='k')
    #                     ax.scatter(p1[0], p1[1], s=5, color='k')
    #                 except:
    #                     pass
    #
    #             for endpoint in endpoints:
    #                 label = endpoint.split(': ')[0]
    #                 (x, y) = eval(endpoint.split(': ')[1])
    #                 ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points',
    #                             fontsize=5, fontweight='light')
    #
    #             plt.savefig(f'{output_path}/geo.jpg')
    #             plt.close()
    #
    #         result.save(f"{output_path}/result_with_boxes.jpg")

    def infer(self, tokenizer, prompt='', image_file='', ecg_file='', output_path = './outputs', base_size=1024, image_size=640,
              crop_mode=True,
              test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()
        device = self.model.device

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        # ---------------------------------------------------------------
        # 1. 准备对话模板
        # ---------------------------------------------------------------
        # 逻辑：如果有 ECG，通常放在最前面，这里我们主要关注 prompt 和 images 列表的构建
        # 注意：ECG 的占位符我们会在 tokenization 阶段手动插入，不需要写在 content 字符串里
        if prompt and image_file:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f'{prompt}',
                    "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        elif prompt:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f'{prompt}',
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            assert False, f'prompt is none!'

        # ---------------------------------------------------------------
        # 2. 准备 ECG 数据 (逻辑复用自 DataCollator)
        # ---------------------------------------------------------------
        ecg_tensor = None
        has_ecg = False
        if ecg_file:
            try:
                import wfdb
                import scipy.signal
                # 拼接完整路径逻辑 (与 Dataset 保持一致)
                record_path = os.path.splitext(ecg_file)[0]
                # 检查文件是否存在
                if os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat"):
                    # 读取记录
                    record = wfdb.rdrecord(record_path)
                    signal = record.p_signal  # shape: (Samples, Channels)
                    # === [关键修复 1] NaN/Inf 清洗 ===
                    if np.isnan(signal).any() or np.isinf(signal).any():
                        print(f"[Warning] NaN/Inf detected in inference file {ecg_file}, cleaning...")
                        signal[np.isnan(signal)] = 0
                        signal[np.isinf(signal)] = 0
                    original_fs = record.fs
                    target_fs = 500
                    target_len = 5000
                    # 重采样
                    if original_fs != target_fs:
                        new_len = int(signal.shape[0] * target_fs / original_fs)
                        signal = scipy.signal.resample(signal, new_len, axis=0)
                    # 截断/填充
                    L, C = signal.shape
                    if L > target_len:
                        signal = signal[:target_len, :]
                    elif L < target_len:
                        pad_len = target_len - L
                        padding = np.zeros((pad_len, C))
                        signal = np.concatenate([signal, padding], axis=0)
                    # 转 Tensor [12, 5000]
                    # 注意：Dataset 中是 Transpose 过的，这里也必须保持一致 (Samples, Channels) -> (Channels, Samples)
                    signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
                    # === [关键修复 2] 归一化 (Z-Score) ===
                    # 与训练时保持一致！
                    mean = signal_tensor.mean(dim=1, keepdim=True)
                    std = signal_tensor.std(dim=1, keepdim=True) + 1e-5
                    signal_tensor = (signal_tensor - mean) / std
                    # 升维并移至设备 [1, 12, 5000]
                    ecg_tensor = signal_tensor.unsqueeze(0).to(device)
                    # 如果当前在半精度运行，转换类型
                    # 自动检测模型的数据类型
                    model_dtype = self.model.ecg_projector[0].weight.dtype
                    ecg_tensor = ecg_tensor.to(model_dtype)
                    has_ecg = True
                else:
                    print(f"Warning: ECG file not found: {ecg_file}")
            except ImportError:
                print("Error: wfdb not installed. Please `pip install wfdb`.")
            except Exception as e:
                print(f"Error loading ECG during inference: {e}")

        # ---------------------------------------------------------------
        # 3. 构建 Prompt 和 Tokens
        # ---------------------------------------------------------------
        prompt_text = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

        # 加载图片
        images = load_pil_images(conversation)

        # 初始化 Mask 和 Token 列表
        tokenized_str = []
        images_seq_mask = []
        ecg_seq_mask = []  # [新增]

        # A. 添加 BOS
        bos_id = 0  # 假设 bos_id 为 0，最好从 tokenizer 获取
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            bos_id = tokenizer.bos_token_id

        tokenized_str.append(bos_id)
        images_seq_mask.append(False)
        ecg_seq_mask.append(False)

        # B. [新增] 插入 ECG 占位符 (如果存在)
        # 逻辑：Start(1) + Body(101) + End(1) = 103
        if has_ecg:
            # 这里的 image_token_id 用作通用占位符
            image_token_id = 128815
            ecg_token_len = 101  # 必须与训练时一致
            total_ecg_tokens = 1 + ecg_token_len + 1

            ecg_tokens = [image_token_id] * total_ecg_tokens
            tokenized_str.extend(ecg_tokens)

            # Mask 设置
            ecg_seq_mask.extend([True] * total_ecg_tokens)
            images_seq_mask.extend([False] * total_ecg_tokens)

        # C. 处理文本和图片
        image_token = '<image>'
        text_splits = prompt_text.split(image_token)

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

        # 用于收集处理后的图片 Tensor
        images_list, images_crop_list, images_spatial_crop = [], [], []

        # 简单的验证
        valid_img_tokens = 0

        for i, text_sep in enumerate(text_splits):
            # 1. 文本部分
            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            ecg_seq_mask += [False] * len(tokenized_sep)  # ECG mask 补齐

            # 2. 图片部分 (如果分隔符后面跟着图片)
            if i < len(text_splits) - 1:
                # 这里的逻辑假设 images 列表和 split 出来的空位是一一对应的
                # 您的原代码里有一个 zip(text_splits, images) 的循环，但那会在最后一段文本丢失
                # 这里我们用索引取图
                if i < len(images):
                    image = images[i]

                    # == 调用 Dynamic Preprocess (复用原有逻辑) ==
                    if crop_mode:
                        if image.size[0] <= 640 and image.size[1] <= 640:
                            crop_ratio = [1, 1]
                            images_crop_raw = []
                        else:
                            images_crop_raw, crop_ratio = dynamic_preprocess(image)

                        # Global View
                        global_view = ImageOps.pad(image, (base_size, base_size), color=(127, 127, 127))
                        images_list.append(image_transform(global_view).to(torch.bfloat16))

                        width_crop_num, height_crop_num = crop_ratio
                        images_spatial_crop.append([width_crop_num, height_crop_num])

                        # Local Views
                        if width_crop_num > 1 or height_crop_num > 1:
                            for crop_img in images_crop_raw:
                                images_crop_list.append(image_transform(crop_img).to(torch.bfloat16))

                        # Calculate Tokens
                        patch_size = 16
                        downsample_ratio = 4
                        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

                        image_token_id = 128815
                        tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                        tokenized_image += [image_token_id]
                        if width_crop_num > 1 or height_crop_num > 1:
                            tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [
                                image_token_id]) * (num_queries * height_crop_num)

                    else:
                        # Crop Mode False 逻辑...
                        pass  # 为简洁省略，建议保留原文件里的 else 分支

                    # 添加 Image Tokens
                    tokenized_str += tokenized_image
                    images_seq_mask += [True] * len(tokenized_image)
                    ecg_seq_mask += [False] * len(tokenized_image)  # ECG Mask 补齐

        # ---------------------------------------------------------------
        # 4. 组装 Batch Tensor
        # ---------------------------------------------------------------
        input_ids = torch.LongTensor(tokenized_str).unsqueeze(0).to(device)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0).to(device)
        ecg_seq_mask = torch.tensor(ecg_seq_mask, dtype=torch.bool).unsqueeze(0).to(device) if has_ecg else None

        # 堆叠图片
        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size)).to(torch.bfloat16)
            images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size)).to(torch.bfloat16)
        else:
            images_ori = torch.stack(images_list, dim=0).to(device)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long).to(device)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).to(device)
            else:
                images_crop = torch.zeros((len(images_list), 3, base_size, base_size)).to(device).to(torch.bfloat16)

        # ---------------------------------------------------------------
        # 5. 执行生成 (Generate)
        # ---------------------------------------------------------------
        streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) if not eval_mode else None

        with torch.autocast(device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                output_ids = self.generate(
                    input_ids=input_ids,
                    images=[(images_crop, images_ori)],
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop_tensor,
                    # [新增] 传入 ECG 参数
                    ecg_values=ecg_tensor,
                    ecg_seq_mask=ecg_seq_mask,

                    do_sample=False if eval_mode else True,  # 示例
                    temperature=0.0 if eval_mode else 0.2,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    max_new_tokens=8192,
                    use_cache=True
                )

        # ---------------------------------------------------------------
        # 6. 解码输出
        # ---------------------------------------------------------------
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        return outputs.strip()
