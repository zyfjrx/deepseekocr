from transformers import Qwen3ForCausalLM, Qwen3Config, Qwen3Model
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import os
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict
from transformers import TextStreamer
from .conversation import get_conv_template
from abc import ABC
import math
import re
import numpy as np
from models.ecgencoder import _build_ecg_tower
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


class DeepencoderEcgOCRConfig(Qwen3Config):
    model_type = "DeepencoderEcgOCR"



class DeepencoderEcgOCRModel(Qwen3Model):
    config_class = DeepencoderEcgOCRConfig

    def __init__(self, config: DeepencoderEcgOCRConfig):
        super(DeepencoderEcgOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        # self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2)
        self.projector = MlpProjector(Dict(projector_type="mlp_gelu", depth=2, input_dim=2048, n_embed=config.hidden_size))
        self.image_newline = nn.Parameter(torch.zeros(config.hidden_size, device=self.device))
        self.view_seperator = nn.Parameter(torch.zeros(config.hidden_size, device=self.device))

        # ================= [新增整合部分] =================
        # ECG 信号编码器
        self.ecg_encoder = _build_ecg_tower()
        # ECG Projector (对齐层)
        # 将 ECG 维度 (768) 映射到 DeepSeek 维度 (config.hidden_size=1280)
        self.ecg_projector = MlpProjector(
            Dict(projector_type="mlp_gelu", depth=2, input_dim=768, n_embed=config.hidden_size))

        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.ecg_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # 3. 初始化可学习的 Start/End 向量 (Soft Embeddings)
        # 这里的 std 参考了 DeepSeek 的初始化方式
        self.ecg_start_embed = nn.Parameter(torch.zeros(config.hidden_size, device=self.device))
        self.ecg_end_embed = nn.Parameter(torch.zeros(config.hidden_size, device=self.device))

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
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = self.get_input_embeddings()(input_ids)

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
            ecg_features = self.ecg_encoder(ecg_values.to(inputs_embeds.device),output_last_transformer_layer=True)

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

        return super().forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids=position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DeepencoderEcgOCRForCausalLM(Qwen3ForCausalLM):
    config_class = DeepencoderEcgOCRConfig

    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = DeepencoderEcgOCRModel(config)

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
                past_length = cache_length
                if hasattr(past_key_values, "get_max_length"):
                    max_cache_length = past_key_values.get_max_length()
                else:
                    max_cache_length = None
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

    def infer(self, tokenizer, prompt='', image_file='', ecg_file='', output_path='./outputs', base_size=1024,
              image_size=640,crop_mode=True, eval_mode=False):

        self.disable_torch_init()
        device = self.model.device

        # 1. 获取 Qwen 特殊 Token ID (参考 modeling_qwen3_ocr.py)
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        nl_id = nl_tokens[-1] if nl_tokens else 198
        image_token_id = 151655  # <|image_pad|>

        # 2. 准备 ECG 数据 (参考 fix_ecg_dataset.py)
        ecg_tensor = None
        has_ecg = False
        if ecg_file:
            try:
                import wfdb
                import scipy.signal

                # 路径处理
                record_path = os.path.splitext(ecg_file)[0]
                if os.path.exists(record_path + ".hea") or os.path.exists(record_path + ".dat"):
                    # 读取记录
                    record = wfdb.rdrecord(record_path)
                    signal = record.p_signal  # (N, 12)

                    # === 清洗 NaN/Inf ===
                    if np.isnan(signal).any() or np.isinf(signal).any():
                        print(f"[Warning] NaN/Inf detected in {ecg_file}, cleaning...")
                        signal[np.isnan(signal)] = 0
                        signal[np.isinf(signal)] = 0

                    original_fs = record.fs
                    target_fs = 500
                    target_len = 5000

                    # === 重采样 ===
                    if original_fs != target_fs:
                        new_len = int(signal.shape[0] * target_fs / original_fs)
                        signal = scipy.signal.resample(signal, new_len, axis=0)

                    # === 截断或填充 ===
                    L, C = signal.shape
                    if L > target_len:
                        signal = signal[:target_len, :]
                    elif L < target_len:
                        pad_len = target_len - L
                        padding = np.zeros((pad_len, C))
                        signal = np.concatenate([signal, padding], axis=0)

                    # === 转置 & 归一化 (Z-Score) ===
                    # 转置为 [12, 5000]
                    signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
                    mean = signal_tensor.mean(dim=1, keepdim=True)
                    std = signal_tensor.std(dim=1, keepdim=True) + 1e-5
                    signal_tensor = (signal_tensor - mean) / std

                    # 升维并移至设备 [1, 12, 5000]
                    ecg_tensor = signal_tensor.unsqueeze(0).to(device)

                    # 获取模型 dtype (安全方式)
                    model_dtype = next(self.model.parameters()).dtype
                    ecg_tensor = ecg_tensor.to(model_dtype)
                    has_ecg = True
                else:
                    print(f"Warning: ECG file not found: {ecg_file}")
            except ImportError:
                print("Error: wfdb not installed. Please `pip install wfdb`.")
            except Exception as e:
                print(f"Error loading ECG: {e}")

        # 3. 准备图片
        os.makedirs(output_path, exist_ok=True)
        images = []
        if image_file:
            try:
                image = Image.open(image_file).convert("RGB")
                image = ImageOps.exif_transpose(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image: {e}")
                # 如果有 prompt 需要图片但加载失败，这可能会导致后续逻辑问题，需注意

        # 4. 构建 ChatML 格式的 Input IDs
        full_input_ids = []
        images_seq_mask_list = []
        ecg_seq_mask_list = []  # [新增] ECG Mask 列表

        # --- System Prompt ---
        system_text = "You are a helpful assistant."
        system_ids = [im_start_id] + tokenizer.encode("system", add_special_tokens=False) + [nl_id] + \
                     tokenizer.encode(system_text, add_special_tokens=False) + [im_end_id, nl_id]

        full_input_ids.extend(system_ids)
        images_seq_mask_list.extend([False] * len(system_ids))
        ecg_seq_mask_list.extend([False] * len(system_ids))

        # --- User Prompt Header ---
        user_header_ids = [im_start_id] + tokenizer.encode("user", add_special_tokens=False) + [nl_id]
        full_input_ids.extend(user_header_ids)
        images_seq_mask_list.extend([False] * len(user_header_ids))
        ecg_seq_mask_list.extend([False] * len(user_header_ids))

        # --- [新增] ECG 占位符注入 ---
        # 逻辑：如果存在 ECG 数据，在 User 内容的最前面插入占位符
        if has_ecg:
            # Start(1) + Features(101) + End(1) = 103 tokens
            ecg_token_len = 101
            total_ecg_tokens = 1 + ecg_token_len + 1
            # 使用 image_token_id 作为占位符 (在 forward 中会被替换)
            ecg_tokens = [image_token_id] * total_ecg_tokens

            full_input_ids.extend(ecg_tokens)
            # ECG Mask 为 True，Image Mask 为 False
            ecg_seq_mask_list.extend([True] * total_ecg_tokens)
            images_seq_mask_list.extend([False] * total_ecg_tokens)

        # --- Content: Text + Image Processing ---
        content = prompt.replace("<image>", "<|image_pad|>")
        text_splits = content.split('<|image_pad|>')

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        patch_size = 16
        downsample_ratio = 4

        images_list = []
        images_crop_list = []
        images_spatial_crop = []
        valid_img_tokens = 0
        image_idx = 0

        for i, text_sep in enumerate(text_splits):
            if text_sep:
                sep_ids = tokenizer.encode(text_sep, add_special_tokens=False)
                full_input_ids.extend(sep_ids)
                images_seq_mask_list.extend([False] * len(sep_ids))
                ecg_seq_mask_list.extend([False] * len(sep_ids))

            if i < len(text_splits) - 1:
                # 插入图片
                if image_idx >= len(images):
                    print("Warning: Prompt expects image but none loaded.")
                    continue

                image = images[image_idx]

                # === Dynamic Preprocess ===
                if crop_mode:
                    if image.size[0] <= 640 and image.size[1] <= 640:
                        crop_ratio = (1, 1)
                        images_crop_raw = []
                    else:
                        images_crop_raw, crop_ratio = dynamic_preprocess(
                            image, min_num=2, max_num=9,
                            image_size=image_size, use_thumbnail=False
                        )

                    # Global View
                    global_view = ImageOps.pad(image, (base_size, base_size),
                                               color=tuple(int(x * 255) for x in image_transform.mean))
                    images_list.append(image_transform(global_view).to(torch.bfloat16))  # 暂存为 BF16

                    width_crop_num, height_crop_num = crop_ratio
                    images_spatial_crop.append([width_crop_num, height_crop_num])

                    # Local Views
                    if width_crop_num > 1 or height_crop_num > 1:
                        for crop_img in images_crop_raw:
                            images_crop_list.append(image_transform(crop_img).to(torch.bfloat16))

                    # Tokens Calculation
                    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

                    # Construct Tokens
                    tok_img = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                    tok_img += [image_token_id]
                    if width_crop_num > 1 or height_crop_num > 1:
                        tok_img += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
                else:
                    # Non-crop mode
                    crop_ratio = (1, 1)
                    images_spatial_crop.append([1, 1])
                    if base_size <= 640:
                        resized_image = image.resize((base_size, base_size), Image.LANCZOS)
                        images_list.append(image_transform(resized_image).to(torch.bfloat16))
                    else:
                        global_view = ImageOps.pad(image, (base_size, base_size),
                                                   color=tuple(int(x * 255) for x in image_transform.mean))
                        images_list.append(image_transform(global_view).to(torch.bfloat16))

                    num_queries = math.ceil((base_size // patch_size) / downsample_ratio)
                    tok_img = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                    tok_img += [image_token_id]

                # Add Image Tokens to Input
                full_input_ids.extend(tok_img)
                images_seq_mask_list.extend([True] * len(tok_img))
                ecg_seq_mask_list.extend([False] * len(tok_img))
                valid_img_tokens += len(tok_img)
                image_idx += 1

        # --- User Footer & Assistant Header ---
        # <|im_end|>\n<|im_start|>assistant\n
        footer_ids = [im_end_id, nl_id] + [im_start_id] + tokenizer.encode("assistant", add_special_tokens=False) + [
            nl_id]
        full_input_ids.extend(footer_ids)
        images_seq_mask_list.extend([False] * len(footer_ids))
        ecg_seq_mask_list.extend([False] * len(footer_ids))

        # 5. Create Tensors
        input_ids = torch.tensor(full_input_ids, dtype=torch.long).unsqueeze(0).to(device)
        images_seq_mask = torch.tensor(images_seq_mask_list, dtype=torch.bool).unsqueeze(0).to(device)
        # 只有在有 ECG 时才创建 Tensor，否则 None
        ecg_seq_mask = torch.tensor(ecg_seq_mask_list, dtype=torch.bool).unsqueeze(0).to(device) if has_ecg else None

        # Stack Images
        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size)).to(torch.bfloat16).to(device)
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long).to(device)
            images_crop = torch.zeros((1, 3, base_size, base_size)).to(torch.bfloat16).to(device)
        else:
            images_ori = torch.stack(images_list, dim=0).to(device)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long).to(device)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).to(device)
            else:
                images_crop = torch.zeros((len(images_list), 3, base_size, base_size)).to(torch.bfloat16).to(device)

        # 6. Generate
        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        else:
            streamer = None

        with torch.autocast(device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                output_ids = self.generate(
                    input_ids=input_ids,
                    images=[(images_crop, images_ori)],
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    # [传入 ECG 参数]
                    ecg_values=ecg_tensor,
                    ecg_seq_mask=ecg_seq_mask,

                    do_sample=False if eval_mode else True,
                    temperature=0.0 if eval_mode else 0.2,
                    eos_token_id=[tokenizer.eos_token_id, im_end_id],
                    streamer=streamer,
                    max_new_tokens=1024,
                    no_repeat_ngram_size=20,
                    use_cache=True
                )

        # 7. Post-processing
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        outputs = outputs.strip()

        return outputs