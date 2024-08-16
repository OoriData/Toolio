# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json

from toolio.common import model_flag, DEFAULT_FLAGS, TOOL_CHOICE_AUTO, model_client_mixin
from toolio.common import extract_content  # Just really for legacy import patterns # noqa: F401
from toolio.schema_helper import Model
from toolio.http_schematics import V1Function
from toolio.prompt_helper import set_tool_response, process_tool_sysmsg
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


class model_manager(model_client_mixin):
    def __init__(self, model_path, tool_reg=None, trace=False, sysmsg_leadin=None, remove_used_tools=True):
        '''
        For client-side loading of MLX LLM models in Toolio

        model_path - local or HuggingFace path to model

        tool_reg (list) - Tools with available implementations, in registry format, i.e. each item is one of:
            * Python import path for a callable annotated (i.e. using toolio.tool.tool decorator)
            * actual callable, annotated (i.e. using toolio.tool.tool decorator)
            * tuple of (callable, schema), with separately specified schema
            * tuple of (None, schema), in which case a tool is declared (with schema) but with no implementation

        trace - send annotations to STDERR to trace the tool-calling process

        sysmsg_leadin - Override the system message used to prime the model for tool-calling
        '''
        self.model_path = model_path
        self.model = Model()
        self.model.load(model_path)
        self.model_type = self.model.model.model_type
        super().__init__(model_type=self.model.model.model_type, tool_reg=tool_reg, trace=trace,
                         sysmsg_leadin=sysmsg_leadin, remove_used_tools=remove_used_tools)

    async def complete(self, messages, stream=True, json_response=False, json_schema=None,
                        max_tokens=128, temperature=0.1):
        schema = None
        # Regular LLM completion; no steering
        if stream:
            responder = ChatCompletionStreamingResponder(self.model_path, self.model_type)
        else:
            responder = ChatCompletionResponder(self.model_path, self.model_type)
        if json_response:
            if json_schema:
                schema = json.loads(json_schema)
            else:
                schema = {'type': 'object'}

        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt = False
        async for resp in self._do_completion(messages, schema, responder, cache_prompt=cache_prompt,
                                                max_tokens=max_tokens, temperature=temperature):
            yield resp

    # Seems streaming is not quite yet working
    # async def complete_with_tools(self, messages, toolset, stream=True, max_trips=3, tool_choice=None,
    #                               max_tokens=128, temperature=0.1):
    async def complete_with_tools(self, messages, toolset=None, stream=False, max_trips=3, tool_choice=TOOL_CHOICE_AUTO,
                                  max_tokens=128, temperature=0.1):
        '''
        Make a chat completion with tools, then continue to iterate completions as long as the LLM
        is using at least one tool, or until max_trips are exhausted

        toolset (list) - tools specified for this request, presumably a subset of overall tool registry.
            Each entry is either a tool name, in which the invocation schema is as registered, or a full
            tool-calling format stanza, in which case, for this request, only the implementaton is used
            from the initial registry
        '''
        toolset = toolset or self.toolset
        req_tools = self._resolve_tools(toolset)
        req_tool_spec = [ s for f, s in req_tools.values() ]

        while max_trips:
            first_chunk = True
            tool_call_resp = None
            # {'choices': [{'index': 0, 'message': {'role': 'assistant',
            # 'tool_calls': [{'id': 'call_6434200784_1722311129_0', 'type': 'function',
            # 'function': {'name': 'square_root', 'arguments': '{"square": 256}'}}]}, 'finish_reason': 'tool_calls'}],
            # 'usage': {'completion_tokens': 24, 'prompt_tokens': 15, 'total_tokens': 39},
            # 'object': 'chat.completion', 'id': 'chatcmpl-6434200784_1722311129', 'created': 1722311129,
            # 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit', 'toolio.model_type': 'llama'}
            async for resp in self._completion_trip(messages, stream, req_tool_spec, max_tokens=max_tokens,
                                                    temperature=temperature):
                resp_msg = resp['choices'][0]['message']
                # resp_msg can be None e.g. if generation finishes due to length
                if resp_msg:
                    if first_chunk and 'tool_calls' in resp_msg:
                    # if first_chunk and 'tool_calls' in resp['choices'][0]['delta']:
                        tool_call_resp = resp
                    else:
                        assert 'delta' in resp['choices'][0]
                        yield resp
                first_chunk = False
            if tool_call_resp:
                max_trips -= 1
                tool_responses = await self._execute_tool_calls(resp, req_tools)
                for call_id, callee_name, result in tool_responses:
                    # print(model_type, model_flags, model_flags and model_flag.TOOL_RESPONSE in model_flags)
                    set_tool_response(messages, call_id, callee_name, str(result), self.model_flags)
                called_names = [ callee_name for call_id, callee_name, result in tool_responses ]
                if self._remove_used_tools:
                    # Many FLOSS LLMs get confused if they see a tool definition still in the response back
                    # And loop back with a new tool request. Remove it to avoid this.
                    trimmed_req_tools = { k: v for (k, v) in req_tools.items() if k not in called_names }
                    req_tools = trimmed_req_tools
                    req_tool_spec = [ s for f, s in req_tools.values() ]
            else:
                # No tools called, so no more trips
                break
            if not req_tool_spec:
                # This is the final call, with all tools removed, so just do a regular completion
                # XXX: SHould we allow a JSON schema or other control for the final response?
                async for resp in self.complete(messages, stream=stream, max_tokens=max_tokens, temperature=temperature):
                    yield resp
                break

    async def _completion_trip(self, messages, stream, req_tool_spec, max_tokens=128, temperature=0.1):
        req_tool_spec = [ (t.dictify() if isinstance(t, V1Function) else t) for t in req_tool_spec ]
        schema, tool_sysmsg = process_tool_sysmsg(req_tool_spec, leadin=self.sysmsg_leadin)
        if stream:
            responder = ToolCallStreamingResponder(self.model, self.model_path, req_tool_spec, schema, tool_sysmsg)
        else:
            responder = ToolCallResponder(self.model_path, self.model_type, schema, tool_sysmsg)
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt=False
        async for resp in self._do_completion(messages, schema, responder, cache_prompt=cache_prompt,
                                                max_tokens=max_tokens, temperature=temperature):
            yield resp

    async def _do_completion(self, messages, schema, responder, cache_prompt=False, max_tokens=128, temperature=0.1):
        'Actual trigger the low-level sampling'
        prompt_tokens = None
        for result in self.model.completion(messages, schema, max_tokens=max_tokens, temp=temperature,
                                            cache_prompt=cache_prompt):
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
