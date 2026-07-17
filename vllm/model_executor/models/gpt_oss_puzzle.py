# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to Nvidia and the vLLM project
from collections.abc import Mapping
from copy import copy
from operator import index
from typing import Protocol, cast

import torch
from transformers import GptOssConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType

from .gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    OAIAttention,
    TransformerBlock,
)
from .utils import extract_layer_index


class _BlockConfigWithExperts(Protocol):
    num_local_experts: int


class GptOssPuzzleAttention(OAIAttention):
    def _build_attention(
        self,
        config: GptOssConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> Attention:
        return Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )


class GptOssPuzzleTransformerBlock(TransformerBlock):
    attention_cls = GptOssPuzzleAttention

    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        layer_idx = extract_layer_index(prefix)
        puzzle_config = vllm_config.model_config.hf_config
        layer_config = copy(puzzle_config.get_gpt_oss_config_for_layer(layer_idx))
        if vllm_config.model_config.disable_sliding_window:
            layer_config.sliding_window = None

        layer_vllm_config = copy(vllm_config)
        layer_vllm_config.model_config = copy(vllm_config.model_config)
        layer_vllm_config.model_config.hf_config = layer_config
        layer_vllm_config.model_config.hf_text_config = layer_config

        super().__init__(
            vllm_config=layer_vllm_config,
            quant_config=quant_config,
            prefix=prefix,
        )


@support_torch_compile
class GptOssPuzzleModel(GptOssModel):
    block_cls = GptOssPuzzleTransformerBlock

    @staticmethod
    def _get_num_experts_from_block_config(block_config: object) -> int:
        try:
            value = (
                block_config["num_local_experts"]
                if isinstance(block_config, Mapping)
                else cast(_BlockConfigWithExperts, block_config).num_local_experts
            )
        except (AttributeError, KeyError) as error:
            raise ValueError(
                "GPT-OSS Puzzle block config is missing num_local_experts."
            ) from error

        try:
            num_experts = index(value)
        except TypeError as error:
            raise TypeError(
                "GPT-OSS Puzzle block config num_local_experts must be an integer."
            ) from error
        if num_experts < 1:
            raise ValueError(
                "GPT-OSS Puzzle block config num_local_experts must be positive."
            )
        return num_experts

    def _get_num_experts_for_layer(self, layer_idx: int) -> int:
        return self._get_num_experts_from_block_config(
            self.config.block_configs[layer_idx]
        )

    def _get_max_num_experts(self) -> int:
        return max(
            self._get_num_experts_from_block_config(block_config)
            for block_config in self.config.block_configs
        )

    def _load_qkv_scale(self, name: str, weight: torch.Tensor) -> bool:
        if not name.endswith((".q_scale", ".k_scale", ".v_scale")):
            return False
        module_name, scale_name = name.rsplit(".", 1)

        module_name = module_name.replace(".self_attn", ".attn")
        module_name = module_name.replace(".attn.attn", ".attn")
        if module_name.endswith("_proj"):
            module_name = module_name.rsplit(".", 1)[0]
        module_name = module_name.removeprefix("model.")
        module = self.get_submodule(module_name)
        attention = getattr(module, "attn", module)

        if scale_name == "q_scale":
            if attention.query_quant is None:
                return True
        elif not is_quantized_kv_cache(attention.kv_cache_dtype):
            return True

        scale_value = weight.item()
        getattr(attention, f"_{scale_name}").fill_(scale_value)
        setattr(attention, f"_{scale_name}_float", scale_value)
        if scale_name in {"k_scale", "v_scale"}:
            getattr(attention, f"_{scale_name}_cpu").fill_(scale_value)
        return True


class GptOssPuzzleForCausalLM(GptOssForCausalLM):
    model_cls = GptOssPuzzleModel
