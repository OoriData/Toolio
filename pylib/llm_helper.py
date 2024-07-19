# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

from enum import Flag, auto

from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
                           mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
                           stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

class model_flag(Flag):
    NO_SYSTEM_ROLE = auto()  # e.g. Gemma blows up if you use a system message role
    USER_ASSISTANT_ALT = auto()  # Model requires alternation of message roles user/assistant only


DEFAULT_FLAGS = model_flag(0)


# {model_class: flags}, defaults to DEFAULT_FLAGS
FLAGS_LOOKUP = {
    gemma.Model: model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    gemma2.Model: model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    mixtral.Model: model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
}
