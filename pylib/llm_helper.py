# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json
import logging
import warnings

from mlx_lm.sample_utils import make_sampler

from toolio.common import extract_content, DEFAULT_JSON_SCHEMA_CUTOUT  # Supports legacy import patterns # noqa: F401
from toolio.toolcall import mixin as toolcall_mixin, process_tools_for_sysmsg, TOOL_CHOICE_AUTO, DEFAULT_INTERNAL_TOOLS
from toolio.schema_helper import Model
# from toolio.prompt_helper import set_tool_response, set_continue_message, process_tools_for_sysmsg
from toolio.response_helper import llm_response


class model_manager(toolcall_mixin):
    def __init__(self, model_path, tool_reg=None, logger=logging, sysmsg_leadin=None, remove_used_tools=True,
                 default_schema=None, json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT):
        '''
        For client-side loading of MLX LLM models in Toolio

        model_path - local or HuggingFace path to model

        tool_reg (list) - Tools with available implementations, in registry format, i.e. each item is one of:
            * Python import path for a callable annotated (i.e. using toolio.tool.tool decorator)
            * actual callable, annotated (i.e. using toolio.tool.tool decorator)
            * tuple of (callable, schema), with separately specified schema
            * tuple of (None, schema), in which case a tool is declared (with schema) but with no implementation

        logger - logger object, handy for tracing operations

        sysmsg_leadin - Override the system message used to prime the model for tool-calling
        '''
        self.model_path = model_path
        self.model = Model()
        self.model.load(model_path)
        # self.model_type = self.model.model.model_type
        self._internal_tools = DEFAULT_INTERNAL_TOOLS
        super().__init__(model_type=self.model.model.model_type, tool_reg=tool_reg, logger=logger,
                         sysmsg_leadin=sysmsg_leadin, remove_used_tools=remove_used_tools,
                         default_schema=default_schema, json_schema_cutout=json_schema_cutout)

    async def iter_complete(self, messages, json_schema=None, temperature=None, insert_schema=True, simple=True, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            messages (List[str]) - Prompt in the form of list of messages to send ot the LLM for completion.
                If you have a system prompt, and you are setting up to call tools, it will be updated with
                the tool spec

            json_schema (dict or str) - JSON schema to be used to structure the generated response.
                If given a a string, it will be decoded as JSON

            max_tokens (int, optional): Maximum number of tokens to generate

            temperature (float, optional): Affects how likely the LLM is to select statistically less common tokens
            insert_schema (bool, optional): Whether or not to insert JSON schema into prompt (True by default)
        Yields:
            str: response chunks
        '''
        schema = None

        if not(json_schema):
            schema, schema_str = self.default_schema, self.default_schema_str
        elif isinstance(json_schema, dict):
            schema, schema_str = json_schema, json.dumps(json_schema)
        elif isinstance(json_schema, str):
            schema, schema_str = json.loads(json_schema), json_schema
        else:
            raise ValueError(f'Invalid JSON schema: {json_schema}')
        if schema and insert_schema:
            self.replace_cutout(messages, schema_str)

        if temperature is not None:
            assert 'sampler' not in kwargs
            kwargs['sampler'] = make_sampler(temp=temperature)

        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt = False
        async for resp in self._do_completion(messages, schema, cache_prompt=cache_prompt, simple=simple, **kwargs):
            yield resp

    async def iter_complete_with_tools(self, messages, tools=None, max_trips=3, tool_choice=TOOL_CHOICE_AUTO,
                                       temperature=None, insert_schema=True, **kwargs):
        '''
        Make a chat completion with tools, then continue to iterate completions as long as the LLM
        is using at least one tool, or until max_trips are exhausted

        Args:
            messages (str) - Prompt in the form of list of messages to send ot the LLM for completion.
                If you have a system prompt, and you are setting up to call tools, it will be updated with
                the tool spec

            tools (list) - tools specified for this request, presumably a subset of overall tool registry.
                Each entry is either a tool name, in which the invocation schema is as registered, or a full
                tool-calling format stanza, in which case, for this request, only the implementaton is used
                from the initial registry

            max_tokens (int, optional): Maximum number of tokens to generate

            temperature (float, optional): Affects how likely the LLM is to select statistically less common tokens
        Yields:
            str: response chunks
        '''
        toolset = tools or self.toolset
        req_tools = self._resolve_tools(toolset)
        req_tool_spec = [ s for f, s in req_tools.values() ]

        if temperature is not None:
            assert 'sampler' not in kwargs
            kwargs['sampler'] = make_sampler(temp=temperature)

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')
        final_resp = None
        while max_trips:
            tool_call_resp = None
            # {'choices': [{'index': 0, 'message': {'role': 'assistant',
            # 'tool_calls': [{'id': 'call_6434200784_1722311129_0', 'type': 'function',
            # 'function': {'name': 'square_root', 'arguments': '{'square': 256}'}}]}, 'finish_reason': 'tool_calls'}],
            # 'usage': {'completion_tokens': 24, 'prompt_tokens': 15, 'total_tokens': 39},
            # 'object': 'chat.completion', 'id': 'chatcmpl-6434200784_1722311129', 'created': 1722311129,
            # 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit', 'toolio.model_type': 'llama'}
            first_resp = None

            if not req_tool_spec:
                # No tools (presumably all removed in prior loops), so just do a regular completion
                async for resp in self.iter_complete(messages, insert_schema=insert_schema, temperature=temperature, **kwargs):
                    if first_resp is None: first_resp = resp  # noqa E701
                    yield resp
                assert first_resp is not None, 'No response from LLM'
                break

            async for resp in self._completion_trip(messages, req_tool_spec, **kwargs):
                if first_resp is None: first_resp = resp  # noqa E701
                resp_msg = resp['choices'][0].get('message')
                # resp_msg can be None e.g. if generation finishes due to length
                if resp_msg:
                    if 'tool_calls' in resp_msg:
                        max_trips -= 1
                        bypass_response = self._check_tool_handling_bypass(first_resp)
                        if bypass_response:
                            # LLM called an internal tool either as a bypass, or for finishing up; treat as direct response
                            final_resp = bypass_response
                            break

                        if max_trips <= 0:
                            # No more available trips; don't bother calling tools
                            warnings.warn('Maximum LLM trips exhausted without a final answer')
                            final_resp = resp
                            break

                        called_names = await self._handle_tool_responses(messages, first_resp, req_tools, req_tool_spec)

                        # Possibly move into _handle_tool_responses? If so, same in llm_helper.py
                        if self._remove_used_tools:
                            # Many FLOSS LLMs get confused if they see a tool definition still in the response back
                            # And loop back with a new tool request. Remove it to avoid this.
                            req_tools = {k: v for (k, v) in req_tools.items() if k not in called_names}
                            req_tool_spec = [s for f, s in req_tools.values()]
                        break
                    else:
                        assert 'delta' in resp['choices'][0]
                        yield resp

                # if resp['choices'][0]['finish_reason'] == 'stop':
                #     break

            assert first_resp is not None, 'No response from LLM'

            if final_resp is not None:
                yield final_resp
                break

        else:
            yield resp

    async def complete(self, messages, json_schema=None, insert_schema=True, temperature=None, **kwargs):
        '''
        Simple completion without tools. Returns just the response text.
        If you want the full response object, use iter_complete directly

        Args:
            prompt (str or list): Text prompt or list of chat messages
            **kwargs: Additional arguments passed to __call__
        '''
        chunks = []
        async for chunk in self.iter_complete(messages, json_schema=json_schema, insert_schema=insert_schema,
                                              temperature=temperature, **kwargs):
            chunks.append(chunk)
        return ''.join(chunks)

        # return resp.first_choice_text if hasattr(resp, 'first_choice_text') else resp['choices'][0]['message'].get('content')

    async def complete_with_tools(self, messages, tools=None, json_schema=None, max_trips=3,
                                    tool_choice=TOOL_CHOICE_AUTO, temperature=None, **kwargs):
        '''
        Complete using specified tools. Returns just the response text
        If you want the full response object, use iter_complete_with_tools directly

        Args:
            prompt (str or list): Text prompt or list of chat messages
            tools (list): List of tool names or specs to make available
            **kwargs: Additional arguments passed to __call__
        '''
        async for resp in self.iter_complete_with_tools(messages, tools=tools,
            max_trips=max_trips, tool_choice=tool_choice, temperature=temperature, **kwargs):
            break

        if isinstance(resp, str):
            return resp
        else:
            # Extract text from response object
            return resp.first_choice_text if hasattr(resp, 'first_choice_text') else resp['choices'][0]['message'].get('content')

    async def _completion_trip(self, messages, req_tool_spec, **kwargs):
        # schema, tool_sysmsg = process_tool_sysmsg(req_tool_spec, self.logger, leadin=self.sysmsg_leadin)
        # Schema, including no-tool fallback, plus string spec of available tools, for use in constructing sysmsg
        full_schema, tool_schemas, sysmsg = process_tools_for_sysmsg(req_tool_spec, self._internal_tools)
        messages = self.reconstruct_messages(messages, sysmsg=sysmsg)
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt=False
        async for resp in self._do_completion(messages, full_schema, cache_prompt=cache_prompt, simple=False, **kwargs):
            yield resp

    async def _do_completion(self, messages, schema, cache_prompt=False, simple=True, **kwargs):
        '''
        Actually trigger the low-level sampling, yielding response chunks
        '''
        for resp in self.model.completion(messages, schema, cache_prompt=cache_prompt, **kwargs):
            if resp is not None:  # GenerationResponse object
                    # text (str): The next segment of decoded text. This can be an empty string.
                    # token (int): The next token.
                    # logprobs (mx.array): A vector of log probabilities.
                    # prompt_tokens (int): The number of tokens in the prompt.
                    # prompt_tps (float): The prompt processing tokens-per-second.
                    # generation_tokens (int): The number of generated tokens.
                    # generation_tps (float): The tokens-per-second for generation.
                    # peak_memory (float): The peak memory used so far in GB.
                    # finish_reason (str): The reason the response is being sent: 'length', 'stop' or `None`
                if simple:
                    resp = resp.text
                else:
                    resp = llm_response.from_generation_response(
                        resp,
                        model_name=self.model_path,
                        model_type=self.model_type
                    )
                yield resp


class local_model_runner(model_manager):
    '''
    Simplified async interface for MLX model completions.

    Example:
        runner = local_model_runner('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')
        resp = await runner('What is 2 + 2?')
        # Or with tools:
        resp = await runner('What is 2 + 2?', tools=['calculator'])
    '''
    async def __call__(self, prompt, tools=None, json_schema=None, max_trips=3, tool_choice=TOOL_CHOICE_AUTO,
                       insert_schema=True, temperature=None, **kwargs):
        '''
        Convenience interface to complete a prompt, optionally using tools or schema constraints
        Returns just the response text
        If you want the full response object, use iter_complete* methods directly

        Args:
            prompt (str or list): Text prompt or list of chat messages
            tools (list, optional): List of tool names or specs to make available; mutually exclusive with json_schema
            json_schema (dict, optional): JSON schema to constrain the response; mutually exclusive with tools
                If given a a string, it will be decoded as JSON
            max_trips (int): Maximum number of tool-calling round trips
            tool_choice (str): How tools should be selected ('auto', 'none', etc)
            insert_schema (bool): Whether or not to insert JSON schema into prompt (True by default)

        Returns:
            Response text if no tools used, otherwise the full response object
        '''
        if tools and json_schema:
            raise ValueError('Cannot specify both tools and a JSON schema')

        # Convert string prompt to chat messages if needed
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]

        if tools:
            async for resp in self.iter_complete_with_tools(messages, tools=tools, max_trips=max_trips,
                                                            tool_choice=tool_choice, insert_schema=insert_schema,
                                                            temperature=temperature, **kwargs):
                return resp
        else:
            async for resp in self.iter_complete(messages, json_schema=json_schema, temperature=temperature, **kwargs):
                return resp

    # max_tokens (int): Maximum tokens to generate per completion
    # temperature (float): Sampling temperature; Affects how likely the LLM is to select statistically less common tokens
