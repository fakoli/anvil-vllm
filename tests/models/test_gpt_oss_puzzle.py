# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

import vllm.model_executor.models.gpt_oss as gpt_oss
import vllm.model_executor.models.gpt_oss_puzzle as gpt_oss_puzzle
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    TransformerBlock,
)
from vllm.model_executor.models.gpt_oss_puzzle import (
    GptOssPuzzleAttention,
    GptOssPuzzleForCausalLM,
    GptOssPuzzleModel,
    GptOssPuzzleTransformerBlock,
)

from .utils import _shrink_gpt_oss_puzzle_config_for_testing


class _FakeExpertMapManager:
    def __init__(self, ep_size: int, ep_rank: int, local_expert_ids: list[int]) -> None:
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self._local_expert_ids = local_expert_ids

    def get_local_expert_ids(self) -> list[int]:
        return self._local_expert_ids


@pytest.fixture(autouse=True)
def _fail_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_cuda(*args: object, **kwargs: object) -> None:
        pytest.fail("GPT-OSS Puzzle unit tests must remain on CPU")

    monkeypatch.setattr(torch.Tensor, "cuda", fail_cuda)


@dataclass
class _DataClassBlockConfig:
    num_local_experts: int


def test_dummy_config_override_keeps_puzzle_layer_metadata_consistent() -> None:
    config = SimpleNamespace(
        block_configs=[_DataClassBlockConfig(128), _DataClassBlockConfig(64)],
        layer_types=["sliding_attention", "full_attention"],
    )

    _shrink_gpt_oss_puzzle_config_for_testing(config, num_experts=2)

    assert config.block_configs == [_DataClassBlockConfig(2)]
    assert config.layer_types == ["sliding_attention"]


def _make_puzzle_model(
    num_experts: list[int],
    expert_map_managers: list[_FakeExpertMapManager],
    intermediate_size: int = 32,
) -> GptOssPuzzleModel:
    model = GptOssPuzzleModel.__new__(GptOssPuzzleModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        intermediate_size=intermediate_size,
        block_configs=[
            SimpleNamespace(num_local_experts=layer_num_experts)
            for layer_num_experts in num_experts
        ],
    )
    model.parallel_config = SimpleNamespace(enable_expert_parallel=True)
    model.start_layer = 0
    model.end_layer = len(num_experts)
    model.layers = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                experts=SimpleNamespace(expert_map_manager=expert_map_manager)
            )
        )
        for expert_map_manager in expert_map_managers
    ]
    return model


def _copy_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    **_: object,
) -> None:
    param.data.copy_(loaded_weight)


def _make_parameter(shape: tuple[int, ...]) -> nn.Parameter:
    param = nn.Parameter(torch.empty(shape), requires_grad=False)
    param.weight_loader = _copy_weight
    return param


def _expert_sentinel(shape: tuple[int, ...]) -> torch.Tensor:
    expert_ids = torch.arange(shape[0], dtype=torch.float32).reshape(
        shape[0], *([1] * (len(shape) - 1))
    )
    return expert_ids.expand(shape).clone()


def _install_mxfp4_loader_fakes(
    monkeypatch: pytest.MonkeyPatch,
    model: GptOssPuzzleModel,
    params: dict[str, nn.Parameter],
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_missing_layer: int | None = None,
) -> None:
    monkeypatch.setattr(model, "named_parameters", lambda: params.items())

    def remap_weights(weights, params_dict):
        assert params_dict.keys() == params.keys()
        assert all(params_dict[name] is param for name, param in params.items())
        for name, weight in weights:
            yield name.replace(".mlp.experts.", ".mlp.experts.routed_experts."), weight

    monkeypatch.setattr(gpt_oss, "remap_moe_expert_weights", remap_weights)
    monkeypatch.setattr(
        gpt_oss, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        gpt_oss.FusedMoEParallelConfig,
        "flatten_tp_across_dp_and_pcp",
        staticmethod(lambda **_: (tp_size, tp_rank)),
    )
    monkeypatch.setattr(
        gpt_oss, "get_dp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        gpt_oss,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        gpt_oss,
        "is_pp_missing_parameter",
        lambda name, _: (
            pp_missing_layer is not None and f".layers.{pp_missing_layer}." in name
        ),
    )


def _make_puzzle_model_with_attention_scales(
    kv_cache_dtype: str = "fp8",
    query_quant: object | None = object(),
) -> tuple[GptOssPuzzleModel, nn.Module]:
    model = GptOssPuzzleModel.__new__(GptOssPuzzleModel)
    nn.Module.__init__(model)

    attention = nn.Module()
    for scale_name in ("q_scale", "k_scale", "v_scale"):
        attention.register_buffer(f"_{scale_name}", torch.tensor(1.0))
        setattr(attention, f"_{scale_name}_float", 1.0)
    attention.register_buffer("_k_scale_cpu", torch.tensor(1.0))
    attention.register_buffer("_v_scale_cpu", torch.tensor(1.0))
    attention.query_quant = query_quant
    attention.kv_cache_dtype = kv_cache_dtype

    attention_wrapper = nn.Module()
    attention_wrapper.attn = attention
    layer = nn.Module()
    layer.attn = attention_wrapper
    model.layers = nn.ModuleList([layer])
    return model, attention


def test_puzzle_model_uses_per_layer_expert_counts_and_expert_map() -> None:
    managers = [
        _FakeExpertMapManager(3, 1, [1, 4]),
        _FakeExpertMapManager(3, 1, [0, 3]),
    ]
    model = _make_puzzle_model([6, 4], managers)
    weights = torch.arange(12).reshape(6, 2)

    assert model._get_num_experts_for_layer(0) == 6
    assert model._get_num_experts_for_layer(1) == 4
    assert model._get_max_num_experts() == 6
    torch.testing.assert_close(
        model._slice_expert_weights(weights, 0, ep_size=3, ep_rank=1),
        weights[[1, 4]],
    )
    assert model._get_local_expert_id(0, 4, ep_size=3, ep_rank=1) == 1
    assert model._get_local_expert_id(0, 3, ep_size=3, ep_rank=1) is None


@pytest.mark.parametrize(
    "block_configs",
    [
        [{"num_local_experts": 6}, {"num_local_experts": 4}],
        [_DataClassBlockConfig(6), _DataClassBlockConfig(4)],
        [
            SimpleNamespace(num_local_experts=6),
            SimpleNamespace(num_local_experts=4),
        ],
    ],
    ids=["mapping", "dataclass", "attribute"],
)
def test_puzzle_model_reads_supported_block_config_forms(
    block_configs: list[object],
) -> None:
    managers = [
        _FakeExpertMapManager(3, 1, [1, 4]),
        _FakeExpertMapManager(3, 1, [0, 3]),
    ]
    model = _make_puzzle_model([6, 4], managers)
    model.config.block_configs = block_configs

    assert model._get_num_experts_for_layer(0) == 6
    assert model._get_num_experts_for_layer(1) == 4
    assert model._get_max_num_experts() == 6


def test_released_puzzle_config_shape_and_ep_boundary() -> None:
    released_block_pattern = [
        (128, 128),
        (None, 128),
        (128, 128),
        (8192, 128),
        (128, 128),
        (8192, 128),
        (128, 128),
        (None, 128),
        (128, 128),
        (None, 128),
        (128, 128),
        (None, 128),
        (128, 128),
        (None, 128),
        (128, 128),
        (None, 64),
        (128, 128),
        (None, 64),
        (128, 128),
        (None, 64),
        (128, 128),
        (8192, 64),
        (128, 64),
        (8192, 64),
        (128, 64),
        (None, 64),
        (128, 64),
        (8192, 64),
        (128, 64),
        (None, 64),
        (128, 64),
        (8192, 64),
        (128, 64),
        (8192, 64),
        (128, 64),
        (8192, 64),
    ]
    block_configs = [
        {"sliding_window": window, "num_local_experts": num_experts}
        for window, num_experts in released_block_pattern
    ]
    model = GptOssPuzzleModel.__new__(GptOssPuzzleModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(block_configs=block_configs)
    model.parallel_config = SimpleNamespace(enable_expert_parallel=True)
    model.start_layer = 0
    model.end_layer = len(block_configs)

    sliding_windows = [config["sliding_window"] for config in block_configs]
    expert_counts = [config["num_local_experts"] for config in block_configs]
    assert len(block_configs) == 36
    assert sliding_windows.count(None) == 10
    assert sliding_windows.count(128) == 18
    assert sliding_windows.count(8192) == 8
    assert expert_counts.count(128) == 18
    assert expert_counts.count(64) == 18
    assert model._get_max_num_experts() == 128
    with pytest.raises(
        ValueError, match="Expert parallel size 65 exceeds the 64 experts in layer 15"
    ):
        model._validate_expert_parallel_size(65)


@pytest.mark.parametrize("scale_name", ["k_scale", "v_scale"])
def test_puzzle_loads_checkpoint_kv_scale_and_cpu_mirrors_without_cuda(
    scale_name: str,
) -> None:
    model, attention = _make_puzzle_model_with_attention_scales()
    checkpoint_name = f"model.layers.0.self_attn.{scale_name}"
    combined_mapper = (
        GptOssPuzzleForCausalLM.hf_to_vllm_mapper
        | QuantizationConfig.get_cache_scale_mapper()
    )
    [(mapped_name, weight)] = list(
        combined_mapper.apply([(checkpoint_name, torch.tensor(0.25))])
    )

    assert mapped_name == f"model.layers.0.attn.attn.{scale_name}"
    assert model._load_qkv_scale(mapped_name, weight)
    torch.testing.assert_close(getattr(attention, f"_{scale_name}"), weight)
    assert getattr(attention, f"_{scale_name}_float") == 0.25
    torch.testing.assert_close(getattr(attention, f"_{scale_name}_cpu"), weight)


def test_puzzle_loads_checkpoint_q_scale_when_query_is_quantized() -> None:
    model, attention = _make_puzzle_model_with_attention_scales()

    assert model._load_qkv_scale("layers.0.self_attn.q_scale", torch.tensor(0.5))
    torch.testing.assert_close(attention._q_scale, torch.tensor(0.5))
    assert attention._q_scale_float == 0.5


def test_puzzle_skips_checkpoint_q_scale_when_query_is_unquantized() -> None:
    model, attention = _make_puzzle_model_with_attention_scales(query_quant=None)

    assert model._load_qkv_scale("layers.0.self_attn.q_scale", torch.tensor(0.5))
    torch.testing.assert_close(attention._q_scale, torch.tensor(1.0))
    assert attention._q_scale_float == 1.0


def test_puzzle_skips_checkpoint_kv_scale_for_unquantized_cache() -> None:
    model, attention = _make_puzzle_model_with_attention_scales(kv_cache_dtype="auto")

    assert model._load_qkv_scale("layers.0.self_attn.k_scale", torch.tensor(0.25))
    torch.testing.assert_close(attention._k_scale, torch.tensor(1.0))
    assert attention._k_scale_float == 1.0
    torch.testing.assert_close(attention._k_scale_cpu, torch.tensor(1.0))


def test_base_gpt_oss_does_not_consume_puzzle_qkv_scales() -> None:
    model = GptOssModel.__new__(GptOssModel)

    assert not model._load_qkv_scale("layers.0.attn.k_scale", torch.tensor(0.25))


def test_puzzle_model_rejects_incompatible_expert_parallel_size() -> None:
    managers = [
        _FakeExpertMapManager(3, 1, [1, 4]),
        _FakeExpertMapManager(3, 1, [0, 3]),
    ]
    model = _make_puzzle_model([6, 4], managers)

    with pytest.raises(
        ValueError, match="Expert parallel size 5 exceeds the 4 experts in layer 1"
    ):
        model._validate_expert_parallel_size(5)


def test_puzzle_model_rejects_stale_expert_map() -> None:
    model = _make_puzzle_model([6], [_FakeExpertMapManager(3, 0, [0, 3])])

    with pytest.raises(
        RuntimeError, match="does not match the active expert-parallel group"
    ):
        model._get_local_expert_ids(0, ep_size=3, ep_rank=1)


@pytest.mark.parametrize(
    ("placement", "local_expert_ids"),
    [
        ("linear", [3, 4, 5]),
        ("round_robin", [1, 3, 5]),
    ],
)
def test_mxfp4_loader_routes_all_expert_tensors_by_expert_map(
    monkeypatch: pytest.MonkeyPatch,
    placement: str,
    local_expert_ids: list[int],
) -> None:
    del placement
    model = _make_puzzle_model([6], [_FakeExpertMapManager(2, 1, local_expert_ids)])
    prefix = "model.layers.0.mlp.experts"
    routed_prefix = f"{prefix}.routed_experts"
    source_shapes = {
        "w13_weight": (6, 64, 1),
        "w2_weight": (6, 2, 16),
        "w13_weight_scale": (6, 64, 1),
        "w2_weight_scale": (6, 2, 1),
        "w13_bias": (6, 64),
        "w2_bias": (6, 2),
    }
    sources = {
        f"{prefix}.{name}": _expert_sentinel(shape)
        for name, shape in source_shapes.items()
    }
    params = {
        f"{routed_prefix}.{name}": _make_parameter((len(local_expert_ids), *shape[1:]))
        for name, shape in source_shapes.items()
    }
    _install_mxfp4_loader_fakes(monkeypatch, model, params)

    loaded = model._load_weights_mxfp4(
        ep_size=2,
        ep_rank=1,
        heads_per_rank=1,
        head_start=0,
        weights=sources.items(),
        stacked_params_mapping=[],
    )

    indices = torch.tensor(local_expert_ids)
    assert loaded == set(params)
    for source_name, source in sources.items():
        param_name = source_name.replace(
            ".mlp.experts.", ".mlp.experts.routed_experts."
        )
        torch.testing.assert_close(params[param_name], source.index_select(0, indices))


def test_mxfp4_loader_skips_pp_missing_expert_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managers = [
        _FakeExpertMapManager(2, 0, [0, 1, 2]),
        _FakeExpertMapManager(2, 0, [0, 1, 2]),
    ]
    model = _make_puzzle_model([6, 6], managers)
    _install_mxfp4_loader_fakes(monkeypatch, model, {}, pp_missing_layer=1)
    weight = _expert_sentinel((6, 64, 1))

    loaded = model._load_weights_mxfp4(
        ep_size=2,
        ep_rank=0,
        heads_per_rank=1,
        head_start=0,
        weights=[("model.layers.1.mlp.experts.w13_weight", weight)],
        stacked_params_mapping=[],
    )

    assert loaded == set()


def test_mxfp4_loader_preserves_tp_slices_without_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_puzzle_model(
        [2], [_FakeExpertMapManager(1, 0, [0, 1])], intermediate_size=64
    )
    model.parallel_config.enable_expert_parallel = False
    prefix = "model.layers.0.mlp.experts"
    routed_prefix = f"{prefix}.routed_experts"
    w13 = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128, 1)
    w2 = torch.arange(2 * 2 * 32, dtype=torch.float32).reshape(2, 2, 32)
    params = {
        f"{routed_prefix}.w13_weight": _make_parameter((2, 64, 1)),
        f"{routed_prefix}.w2_weight": _make_parameter((2, 2, 16)),
    }
    _install_mxfp4_loader_fakes(monkeypatch, model, params, tp_size=2, tp_rank=1)

    loaded = model._load_weights_mxfp4(
        ep_size=1,
        ep_rank=0,
        heads_per_rank=1,
        head_start=0,
        weights=[
            (f"{prefix}.w13_weight", w13),
            (f"{prefix}.w2_weight", w2),
        ],
        stacked_params_mapping=[],
    )

    assert loaded == set(params)
    torch.testing.assert_close(params[f"{routed_prefix}.w13_weight"], w13[:, 64:])
    torch.testing.assert_close(params[f"{routed_prefix}.w2_weight"], w2[..., 16:])


def test_puzzle_attention_uses_configured_window_on_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_attention(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(gpt_oss_puzzle, "Attention", fake_attention)
    attention = GptOssPuzzleAttention.__new__(GptOssPuzzleAttention)
    nn.Module.__init__(attention)
    attention.num_local_attention_heads = 8
    attention.num_local_key_value_heads = 2
    attention.head_dim = 64
    attention.scaling = 0.125
    attention.sinks = nn.Parameter(torch.empty(8), requires_grad=False)

    result = attention._build_attention(
        SimpleNamespace(sliding_window=4096),
        cache_config=None,
        quant_config=None,
        prefix="model.layers.3.attn",
    )

    assert result is sentinel
    assert captured["per_layer_sliding_window"] == 4096


@pytest.mark.parametrize(
    ("disable_sliding_window", "expected_sliding_window"),
    [(False, 4096), (True, None)],
)
def test_puzzle_block_builds_from_layer_config_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
    disable_sliding_window: bool,
    expected_sliding_window: int | None,
) -> None:
    source_layer_config = SimpleNamespace(sliding_window=4096)
    puzzle_config = SimpleNamespace(
        get_gpt_oss_config_for_layer=lambda layer_idx: source_layer_config
    )
    model_config = SimpleNamespace(
        hf_config=puzzle_config,
        hf_text_config=puzzle_config,
        disable_sliding_window=disable_sliding_window,
    )
    vllm_config = SimpleNamespace(model_config=model_config)
    captured: dict[str, object] = {}

    def fake_block_init(
        self: TransformerBlock,
        vllm_config: object,
        quant_config: object,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        captured["vllm_config"] = vllm_config
        captured["prefix"] = prefix

    monkeypatch.setattr(TransformerBlock, "__init__", fake_block_init)
    GptOssPuzzleTransformerBlock(
        vllm_config=vllm_config,
        quant_config=None,
        prefix="model.layers.3",
    )

    layer_vllm_config = cast(SimpleNamespace, captured["vllm_config"])
    layer_model_config = layer_vllm_config.model_config
    assert layer_model_config.hf_config is layer_model_config.hf_text_config
    assert layer_model_config.hf_config.sliding_window == expected_sliding_window
    assert source_layer_config.sliding_window == 4096
    assert vllm_config.model_config is model_config
    assert captured["prefix"] == "model.layers.3"


def test_causal_lm_model_class_hooks() -> None:
    assert GptOssForCausalLM.model_cls is GptOssModel
    assert GptOssPuzzleForCausalLM.model_cls is GptOssPuzzleModel
