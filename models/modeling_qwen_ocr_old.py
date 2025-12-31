from transformers import Qwen2ForCausalLM, Qwen2Config, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import os
from models.deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision import transforms
from transformers import TextStreamer
from models.conversation import get_conv_template
from abc import ABC
import math
import re
from tqdm import tqdm
import numpy as np


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

                color_a = color + (20, )
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



class QwenOCRConfig(Qwen2Config):
    model_type = "qwen_ocr"
    deepseek_ocr_checkpoint = "/Users/zhangyf/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR"


class QwenOCRModel(Qwen2Model):
    config_class = QwenOCRConfig

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # -----------------------------------------------------------
        # 1. 初始化 DeepSeek-OCR 的视觉编码器
        # -----------------------------------------------------------
        # 仅构建结构，权重统一由 _load_pretrained_weights 加载
        self.sam_model = build_sam_vit_b(checkpoint=None)
        self.vision_model = build_clip_l()

        # -----------------------------------------------------------
        # 2. 初始化 Projector (关键适配层)
        # -----------------------------------------------------------
        # 视觉特征的输入维度固定为 2048 (由 SAM+CLIP 拼接逻辑决定)
        # 输出维度必须等于 LLM 的 hidden_size (Qwen2.5-32B 为 5120)
        # 注意：这里的权重是随机初始化的，必须经过训练！
        self.projector = MlpProjector(Dict(projector_type="mlp_gelu",depth=2, input_dim=2048, n_embed=config.hidden_size))

        # -----------------------------------------------------------
        # 3. 初始化特殊 Token 参数
        # -----------------------------------------------------------
        embed_std = 1 / torch.sqrt(torch.tensor(config.hidden_size, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(config.hidden_size) * embed_std)

        # -----------------------------------------------------------
        # 4. 加载预训练权重 (DeepSeek-OCR 原版权重)
        # -----------------------------------------------------------
        # 优先使用 config.deepseek_ocr_checkpoint，其次 vision_checkpoint
        ocr_checkpoint = getattr(config, 'deepseek_ocr_checkpoint', None) or getattr(config, 'vision_checkpoint', None)
        if ocr_checkpoint:
            self._load_pretrained_weights(ocr_checkpoint)
            
    def _load_pretrained_weights(self, model_path: str): 
         """ 
         加载预训练权重 
 
         支持的格式： 
         1. 单个 .bin / .pt 文件 
         2. 文件夹（包含多个 .bin / .safetensors 文件） 
         """ 
         print(f"[DeepEncoder] Loading weights from {model_path}...") 
         state_dict = self._load_state_dict(model_path)
         # for name,model in self.sam_model.named_modules():
         #     print( name)
 
         # 加载 SAM 权重 
         sam_keys = {k.replace('model.sam_model.', ''): v for k, v in state_dict.items() if 'sam_model' in k}
         if sam_keys: 
             missing, unexpected = self.sam_model.load_state_dict(sam_keys, strict=False) 
             print(f"[DeepEncoder SAM] Loaded {len(sam_keys)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})") 
 
         # 加载 CLIP 权重 
         clip_keys = {k.replace('model.vision_model.', ''): v for k, v in state_dict.items() if 'vision_model' in k}
         if clip_keys: 
             missing, unexpected = self.vision_model.load_state_dict(clip_keys, strict=False) 
             print(f"[DeepEncoder CLIP] Loaded {len(clip_keys)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})") 

         # 加载 newline 权重
         if 'image_newline' in state_dict:
             self.image_newline.data = state_dict['image_newline']
             print("[DeepEncoder] Loaded image_newline")
 
    def _load_state_dict(self, model_path: str) -> dict: 
         """加载 state dict（支持单文件和文件夹）""" 
         if os.path.isfile(model_path): 
             return torch.load(model_path, map_location='cpu') 
 
         # 文件夹模式：合并所有权重文件 
         state_dict = {} 
         for f in os.listdir(model_path): 
             file_path = os.path.join(model_path, f) 
             if f.endswith('.bin') or f.endswith('.pt'): 
                 state_dict.update(torch.load(file_path, map_location='cpu')) 
             elif f.endswith('.safetensors'): 
                 try: 
                     from safetensors.torch import load_file 
                     state_dict.update(load_file(file_path)) 
                 except ImportError: 
                     print("[Warning] safetensors not installed, skipping .safetensors files") 
         return state_dict

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
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            images: Optional[List[torch.FloatTensor]] = None,
            images_seq_mask: Optional[torch.Tensor] = None,
            images_spatial_crop: Optional[torch.Tensor] = None,
    ):
        # 1. 获取文本 Embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 2. 注入图像 Embedding (逻辑移植自 DeepseekOCRModel)
        if images is not None and len(images) > 0 and (input_ids is None or input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            idx = 0
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []
                patches = image[0]  # [P, C, H, W]
                image_ori = image[1]  # [1, C, H, W]

                # 提取视觉特征
                # 注意：这里假设处于训练模式或 forward 上下文
                if torch.sum(patches).item() != 0:
                    local_features_1 = self.sam_model(patches)
                    local_features_2 = self.vision_model(patches, local_features_1)
                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)),
                                               dim=-1)
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)

                    # 形状变换与特殊 Token 拼接
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

                else:
                    # 仅全局视图
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(global_local_features)

                # 将图像特征填入 inputs_embeds
                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx].masked_scatter_(
                        images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device),
                        images_in_this_batch.to(inputs_embeds.device)
                    )

                idx += 1

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class QwenOCRForCausalLM(Qwen2ForCausalLM):
    config_class = QwenOCRConfig

    def __init__(self, config):
        # 使用 Qwen2PreTrainedModel 初始化，避免创建重复的 Qwen2Model
        super(Qwen2ForCausalLM, self).__init__(config)

        self.model = QwenOCRModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
            return_dict: Optional[bool] = None,
            images: Optional[List[torch.FloatTensor]] = None,
            images_seq_mask: Optional[torch.Tensor] = None,
            images_spatial_crop: Optional[torch.Tensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 3. 调用 QwenOCRModel 的 forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        
        images_seq_mask = kwargs.get("images_seq_mask", None)
        if images_seq_mask is not None:
            cache_position = model_inputs.get("cache_position")
            if cache_position is not None:
                bsz = images_seq_mask.shape[0]
                seq_len = cache_position.shape[0]
                new_mask = torch.zeros((bsz, seq_len), dtype=images_seq_mask.dtype, device=images_seq_mask.device)
                
                valid_mask = cache_position < images_seq_mask.shape[1]
                valid_indices = cache_position[valid_mask]
                
                if valid_indices.numel() > 0:
                    new_mask[:, valid_mask] = images_seq_mask[:, valid_indices]
                
                images_seq_mask = new_mask
        
        model_inputs.update(
            {
                "images": kwargs.get("images", None),
                "images_seq_mask": images_seq_mask,
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
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

    def infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True,
              test_compress=False, save_results=False, eval_mode=False):

        self.disable_torch_init()
        device = self.model.device

        # 1. 获取 Qwen 特殊 Token ID
        # 兜底逻辑：如果 tokenizer 没取到，使用 Qwen2.5 默认值
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        nl_id = nl_tokens[-1] if nl_tokens else 198
        image_token_id = 151655  # <|image_pad|>

        # 2. 准备图片
        os.makedirs(output_path, exist_ok=True)
        images = []
        if image_file:
            try:
                # 保持与 dataset 一致的读取方式
                image = Image.open(image_file).convert("RGB")
                image = ImageOps.exif_transpose(image)  # 修正方向
                images.append(image)
            except Exception as e:
                print(f"Error loading image: {e}")
                return

        # 3. 构建 ChatML 格式的 Input IDs
        # 结构: <|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n

        full_input_ids = []
        images_seq_mask_list = []

        # --- System Prompt ---
        system_text = "You are a helpful assistant."
        # <|im_start|>system\nText<|im_end|>\n
        system_ids = [im_start_id] + tokenizer.encode("system", add_special_tokens=False) + [nl_id] + \
                     tokenizer.encode(system_text, add_special_tokens=False) + [im_end_id, nl_id]
        full_input_ids.extend(system_ids)
        images_seq_mask_list.extend([False] * len(system_ids))

        # --- User Prompt ---
        # <|im_start|>user\n
        user_header_ids = [im_start_id] + tokenizer.encode("user", add_special_tokens=False) + [nl_id]
        full_input_ids.extend(user_header_ids)
        images_seq_mask_list.extend([False] * len(user_header_ids))

        # Content: Text + Image Placeholders
        # 替换 dataset 约定的 <image> 占位符
        content = prompt.replace("<image>", "<|image_pad|>")
        text_splits = content.split('<|image_pad|>')

        # 图片处理相关参数
        patch_size = 16
        downsample_ratio = 4
        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

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

            # 如果不是最后一段，说明此处有一个 <|image_pad|>
            if i < len(text_splits) - 1:
                if image_idx >= len(images):
                    print("Warning: Prompt contains <image> but no image file provided/loaded.")
                    continue

                image = images[image_idx]
                w, h = image.size

                # --- Image Processing (Align with dataset.py) ---
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
                    images_list.append(image_transform(global_view).to(torch.bfloat16))

                    width_crop_num, height_crop_num = crop_ratio
                    images_spatial_crop.append([width_crop_num, height_crop_num])

                    # Local Views
                    if width_crop_num > 1 or height_crop_num > 1:
                        for crop_img in images_crop_raw:
                            images_crop_list.append(image_transform(crop_img).to(torch.bfloat16))

                    # Calculate Tokens
                    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

                    # Construct Image Tokens
                    # Global part
                    tok_img = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                    tok_img += [image_token_id]  # 273
                    valid_img_tokens += len(tok_img)

                    # Local part
                    if width_crop_num > 1 or height_crop_num > 1:
                        local_tokens = ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                    num_queries * height_crop_num)
                        tok_img += local_tokens
                        valid_img_tokens += len(local_tokens)

                else:
                    # Non-crop mode (Simple resize/pad)
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
                    valid_img_tokens += len(tok_img)

                full_input_ids.extend(tok_img)
                images_seq_mask_list.extend([True] * len(tok_img))
                image_idx += 1

        # User Footer: <|im_end|>\n
        user_footer_ids = [im_end_id, nl_id]
        full_input_ids.extend(user_footer_ids)
        images_seq_mask_list.extend([False] * len(user_footer_ids))

        # --- Assistant Header ---
        # <|im_start|>assistant\n
        assistant_header_ids = [im_start_id] + tokenizer.encode("assistant", add_special_tokens=False) + [nl_id]
        full_input_ids.extend(assistant_header_ids)
        images_seq_mask_list.extend([False] * len(assistant_header_ids))

        # 4. Create Tensors
        input_ids = torch.tensor(full_input_ids, dtype=torch.long)
        images_seq_mask = torch.tensor(images_seq_mask_list, dtype=torch.bool)

        # Stack Images
        if len(images_list) == 0:
            # Handle text-only case
            images_ori = torch.zeros((1, 3, image_size, image_size)).to(torch.bfloat16)
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size)).to(torch.bfloat16)
        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((len(images_list), 3, base_size, base_size), dtype=self.model.dtype).to(
                    torch.bfloat16)

        # 5. Generate
        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        else:
            streamer = None

        with torch.autocast(device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                output_ids = self.generate(
                    input_ids.unsqueeze(0).to(device),
                    images=[(images_crop.to(device), images_ori.to(device))],
                    images_seq_mask=images_seq_mask.unsqueeze(0).to(device),
                    images_spatial_crop=images_spatial_crop.to(device),
                    do_sample=False,
                    eos_token_id=[tokenizer.eos_token_id, im_end_id],  # 遇到 <|im_end|> 也停止
                    streamer=streamer,
                    max_new_tokens=1024,
                    no_repeat_ngram_size=20,
                    use_cache=True
                )

        # 6. Post-processing
        # 获取生成的文本部分
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[0]:], skip_special_tokens=True)
        outputs = outputs.strip()

        if test_compress:
            # 简单的压缩率计算逻辑 (可选)
            pure_texts_len = len(tokenizer.encode(outputs, add_special_tokens=False))
            print(f'Valid Image Tokens: {valid_img_tokens}, Output Text Tokens: {pure_texts_len}')
            if valid_img_tokens > 0:
                print(f'Compression Ratio: {pure_texts_len / valid_img_tokens:.2f}')

        if save_results:
            # 这里保留原有的保存逻辑
            # 注意：re_match 和 process_image_with_refs 依赖于原始文件中的其他辅助函数
            # 如果这些函数未变动，下面的逻辑应该能正常运行
            try:
                matches_ref, matches_images, mathes_other = re_match(outputs)
                if image_file and images:
                    result_img = process_image_with_refs(images[0], matches_ref, output_path)
                    result_img.save(f"{output_path}/result_with_boxes.jpg")

                with open(f'{output_path}/result.mmd', 'w', encoding='utf-8') as f:
                    f.write(outputs)
            except Exception as e:
                print(f"Error saving results: {e}")

        return outputs