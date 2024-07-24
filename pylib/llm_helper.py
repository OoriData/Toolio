# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json
from enum import Flag, auto

# import mlx.core as mx
from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
                           mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
                           stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

from toolio.schema_helper import Model
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


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


class model_manager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = Model()
        self.model.load(model_path)

    async def chat_complete(self, messages, functions=None, stream=True, json_response=False, json_schema=None,
                            max_tokens=128, temperature=0.1):
        schema = None
        if functions:
            if stream:
                responder = ToolCallStreamingResponder(self.model_path, functions, self.model)
            else:
                responder = ToolCallResponder(self.model_path, functions)
            schema = responder.schema
        else:
            # Regular LLM completion; no steering
            if stream:
                responder = ChatCompletionStreamingResponder(self.model_path)
            else:
                responder = ChatCompletionResponder(self.model_path)
            if json_response:
                if json_schema:
                    schema = json.loads(json_schema)
                else:
                    schema = {'type': 'object'}

        prompt_tokens = None

        for result in self.model.completion(
            messages,
            schema=schema,
            max_tokens=max_tokens,
            temp=temperature,
            # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
            # cache_prompt=True,
            cache_prompt=False,
        ):
            if result['op'] == 'evaluatedPrompt':
                prompt_tokens = result['token_count']
            elif result['op'] == 'generatedTokens':
                message = responder.generated_tokens(result['text'])
                if message:
                    yield message
            elif result['op'] == 'stop':
                completion_tokens = result['token_count']
                yield responder.generation_stopped(
                    result['reason'], prompt_tokens, completion_tokens
                )
            else:
                raise RuntimeError(f'Unknown resule operation {result["op"]}')


async def extract_content(resp_stream):
    # Interpretthe streaming pattern from the API. viz https://platform.openai.com/docs/api-reference/streaming
    async for chunk in resp_stream:
        # Minimal checking: Trust the delivered structure
        content = chunk['choices'][0]['delta']['content']
        if content is not None:
            yield content
