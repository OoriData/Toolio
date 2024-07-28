# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json
from enum import Flag, auto

# import mlx.core as mx
# from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
#                            mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
#                            stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

from toolio.schema_helper import Model
from toolio.http_schematics import V1ChatMessage
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


class model_flag(Flag):
    NO_SYSTEM_ROLE = auto()  # e.g. Gemma blows up if you use a system message role
    USER_ASSISTANT_ALT = auto()  # Model requires alternation of message roles user/assistant only
    TOOL_RESPONSE = auto()  # Model expects responses from tools via OpenAI API style messages


DEFAULT_FLAGS = model_flag(0)


# {model_class: flags}, defaults to DEFAULT_FLAGS
FLAGS_LOOKUP = {
    'llama': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT | model_flag.TOOL_RESPONSE,
    'gemma': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'gemma2': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mixtral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mistral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
}


class model_manager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = Model()
        self.model.load(model_path)

    async def chat_complete(self, messages, tools=None, stream=True, json_response=False, json_schema=None,
                            max_tokens=128, temperature=0.1):
        schema = None
        if tools:
            if stream:
                responder = ToolCallStreamingResponder(self.model_path, tools, self.model)
            else:
                responder = ToolCallResponder(self.model_path, tools)
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


def enrich_chat_for_tools(msgs, tool_prompt, model_flags):
    '''
    msgs - chat messages to augment
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # Add prompting (system prompt, if permitted) instructing the LLM to use tools
    if model_flag.NO_SYSTEM_ROLE in model_flags:  # LLM supports system messages
        msgs.insert(0, V1ChatMessage(role='system', content=tool_prompt))
    elif model_flag.USER_ASSISTANT_ALT in model_flags: # LLM insists that user and assistant messages must alternate
        msgs[0].content = msgs[0].content=tool_prompt + '\n\n' + msgs[0].content
    else:
        msgs.insert(0, V1ChatMessage(role='user', content=tool_prompt))


def set_tool_response(msgs, tool_call_id, tool_name, tool_result, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment
    tool_response - response generatded by selected tool
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # XXX: model_flags = None ⇒ assistant-style tool response. Is this the default we want?
    if model_flag.TOOL_RESPONSE in model_flags:
        msgs.append({
            'tool_call_id': tool_call_id,
            'role': 'tool',
            'name': tool_name,
            'content': tool_result,
        })
    else:
        tool_response_text = f'Result of the call to {tool_name}: {tool_result}'
        if model_flag.USER_ASSISTANT_ALT in model_flags:
            # If there is already an assistant msg from tool-calling, merge it
            if msgs[-1]['role'] == 'assistant':
                msgs[-1]['content'] += '\n\n' + tool_response_text
            else:
                msgs.append({'role': 'assistant', 'content': tool_response_text})
        else:
            msgs.append({'role': 'assistant', 'content': tool_response_text})
