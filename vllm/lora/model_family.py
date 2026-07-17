# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_GPT_OSS_MODEL_ARCHITECTURES = frozenset(
    {"GptOssForCausalLM", "GptOssPuzzleForCausalLM"}
)


def is_gpt_oss_model_architecture(architecture: str | None) -> bool:
    """Return whether an architecture uses GPT-OSS interleaved MoE weights."""
    return architecture in _GPT_OSS_MODEL_ARCHITECTURES
