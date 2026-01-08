from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from torch import nn
from open_clip.open_clip.model import CLIPTextCfg, _build_text_tower, CLIPEcgCfg, _build_ecg_tower
import torch
from open_clip.open_clip.transformer import QuickGELU, LayerNormFp32, LayerNorm, EcgTransformer


@dataclass
class CLIPEcgCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 50
    seq_length: int = 5000
    lead_num: int = 12

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None



def _build_ecg_tower(
        embed_dim:int = 512,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):

    ecg_cfg = CLIPEcgCfg()

    act_layer = QuickGELU if quick_gelu else nn.GELU

    ecg_heads = ecg_cfg.width // ecg_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    if ecg_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **ecg_cfg.norm_kwargs)
    if ecg_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **ecg_cfg.act_kwargs)

    ecg = EcgTransformer(
        seq_length=ecg_cfg.seq_length,
        patch_size=ecg_cfg.patch_size,
        lead_num=ecg_cfg.lead_num,
        width=ecg_cfg.width,
        layers=ecg_cfg.layers,
        heads=ecg_heads,
        mlp_ratio=ecg_cfg.mlp_ratio,
        ls_init_value=ecg_cfg.ls_init_value,
        patch_dropout=ecg_cfg.patch_dropout,
        attentional_pool=ecg_cfg.attentional_pool,
        attn_pooler_queries=ecg_cfg.attn_pooler_queries,
        attn_pooler_heads=ecg_cfg.attn_pooler_heads,
        pos_embed_type=ecg_cfg.pos_embed_type,
        no_ln_pre=ecg_cfg.no_ln_pre,
        final_ln_after_pool=ecg_cfg.final_ln_after_pool,
        pool_type=ecg_cfg.pool_type,
        output_tokens=ecg_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        )
    return ecg

if __name__ == '__main__':
    model = _build_ecg_tower()
    data = torch.randn(1, 12, 5000)
    out = model(data,output_last_transformer_layer=True)
    print(out.shape)
