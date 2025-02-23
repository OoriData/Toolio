# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json
import logging
# import warnings

from mlx_lm.sample_utils import make_sampler

from toolio.common import extract_content, DEFAULT_JSON_SCHEMA_CUTOUT  # Supports legacy import patterns # noqa: F401
from toolio.toolcall import mixin as toolcall_mixin, process_tools_for_sysmsg, TOOL_CHOICE_AUTO, DEFAULT_INTERNAL_TOOLS
from toolio.schema_helper import Model
# from toolio.prompt_helper import set_tool_response, set_continue_message, process_tools_for_sysmsg
from toolio.response_helper import llm_response, llm_response_type


class model_manager(toolcall_mixin):
    def __init__(self, model_path, tool_reg=None, logger=logging, sysmsg_leadin=None, remove_used_tools=True,
                 default_schema=None, json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT, server_mode=False):
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

        server_mode - if True, disable local tool resolution and execution
        '''
        self.server_mode = server_mode
        self.model_path = model_path
        self.model = Model()
        self.model.load(model_path)
        # self.model_type = self.model.model.model_type
        self._internal_tools = DEFAULT_INTERNAL_TOOLS
        super().__init__(model_type=self.model.model.model_type, tool_reg=tool_reg, logger=logger,
                         sysmsg_leadin=sysmsg_leadin, remove_used_tools=remove_used_tools,
                         default_schema=default_schema, json_schema_cutout=json_schema_cutout)

    async def iter_complete(self, messages, json_schema=None, temperature=None, insert_schema=True, full_response=False, **kwargs):
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

        if full_response: resp = None
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        for gr in self.model.completion(messages, schema, cache_prompt=False, **kwargs):
            # resp is a GenerationResponse object with the following fields:
            # text (str): The next segment of decoded text. This can be an empty string.
            # token (int): The next token.
            # logprobs (mx.array): A vector of log probabilities.
            # prompt_tokens (int): The number of tokens in the prompt.
            # prompt_tps (float): The prompt processing tokens-per-second.
            # generation_tokens (int): The number of generated tokens.
            # generation_tps (float): The tokens-per-second for generation.
            # peak_memory (float): The peak memory used so far in GB.
            # finish_reason (str): The reason the response is being sent: 'length', 'stop' or `None`
            if full_response:
                if resp is None:
                    resp = llm_response.from_generation_response(gr, model_name=self.model_path, model_type=self.model_type)
                else:
                    resp.update_from_gen_response(gr)
                yield resp
            else:
                yield gr.text

    async def complete_with_tools(self, messages, tools=None, max_trips=3, tool_choice=TOOL_CHOICE_AUTO,
                                  temperature=None, **kwargs):
        '''
        Complete using specified tools. Makes multiple completion trips if needed for tool calling.
        In server mode, just returns tool calls without executing them.

        Args:
            messages (List[dict]): Chat messages including the prompt
            full_response (bool) - If True, return full llm_response object instead of just text
            tools (list): Tools specified for this request, presumably a subset of overall tool registry
                Each entry is either:
                * a tool name to use the registered implementation and schema
                * a complete tool specification dict, using only registered implementation
            max_trips (int): Maximum number of times to consult the LLM
            tool_choice (str): Control over tool selection ('auto', 'none', etc)
            temperature (float): Optional sampling temperature
            **kwargs: Additional arguments passed to completion

        returns:
            llm_response object, always (unlike complete() which returns plain text if you specify full_response=False)
        '''
        if self.server_mode:
            # In server mode, don't try to resolve or execute tools
            toolset = tools  # Just pass through tool specifications
            req_tool_spec, req_tools = tools, None
        else:
            # Normal client-side operation
            toolset = tools or self.toolset
            req_tools = self._resolve_tools(toolset)
            req_tool_spec = [s for f, s in req_tools.values()]

        if temperature is not None:
            assert 'sampler' not in kwargs
            kwargs['sampler'] = make_sampler(temp=temperature)

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')

        trips_remaining = max_trips
        while trips_remaining > 0:
            trips_remaining -= 1

            # No tools left means just do a regular completion
            if not req_tool_spec:
                response = None
                async for chunk in self.iter_complete(messages, temperature=temperature, simple=False, **kwargs):
                    if response is None:
                        response = chunk
                    else:
                        response.update_from_gen_response(chunk)
                return response

            # Do a completion trip with tools. Start by building tool-call specific schema
            full_schema, tool_schemas, sysmsg = process_tools_for_sysmsg(
                req_tool_spec, self._internal_tools, leadin=self.sysmsg_leadin
            )
            messages = self.reconstruct_messages(messages, sysmsg=sysmsg)

            # print('Calling LLM with tool schema:', full_schema)
            response = await self._single_toolcalling_completion(messages, full_schema, cache_prompt=False, **kwargs)

            if not response:
                break

            # XXX: What if generation finishes due to length
            # debug('LLM response:', response)
            if response.response_type == llm_response_type.TOOL_CALL:
                bypass_response = self._check_tool_handling_bypass(response)
                if bypass_response:
                    # LLM called an internal tool either as a bypass, or for finishing up
                    return llm_response.from_openai_chat(bypass_response)

                if self.server_mode:
                    # In server mode, just return the tool calls without executing
                    return response

                if trips_remaining <= 0:
                    # No more available trips; don't bother calling tools
                    msg = 'Maximum LLM trips exhausted without a final answer'
                    self.logger.debug(msg)
                    return response

                results = await self._execute_tool_calls(response.tool_calls)
                await self._handle_tool_results(messages, results, req_tools, model_flags=self.model_flags, remove_used_tools=self._remove_used_tools)
            elif response.response_type == llm_response_type.MESSAGE:  # Direct response without tool calls
                return response
            else:
                self.logger.warning(f'Unexpected response type: {response.response_type}. {response=}')
                return response

        return response

    async def _single_toolcalling_completion(self, messages, schema, cache_prompt=False, **kwargs):
        '''Do a single completion trip with tool calling support'''
        response = None
        for gr in self.model.completion(messages, schema, cache_prompt=cache_prompt, **kwargs):
            if response is None:
                response = llm_response.from_generation_response(gr, tool_schema=schema,
                                                                 model_name=self.model_path, model_type=self.model_type)
            else:
                response.update_from_gen_response(gr)

        # Finalize tool calls if any
        if response and hasattr(response, 'finalize_tool_call'):
            response.finalize_tool_call()

        return response

    async def complete(self, messages, full_response=False, json_schema=None, insert_schema=True, temperature=None, **kwargs):
        '''
        Simple completion without tools. Returns just the response text.
        If you want the full response object, use iter_complete directly

        Args:
            messages (List[dict]): Chat messages including the prompt
            full_response (bool) - If True, return full llm_response object instead of just text
            json_schema (list, dict or string): JSON Schema for strict response steering
            temperature (float): Optional sampling temperature
            **kwargs: Additional arguments passed to completion
        '''
        chunks = []
        response = None
        async for chunk in self.iter_complete(messages, json_schema=json_schema, insert_schema=insert_schema,
                                            temperature=temperature, full_response=full_response, **kwargs):
            if full_response:
                response = chunk
            else:
                chunks.append(chunk)

        return response if full_response else ''.join(chunks)


class local_model_runner(model_manager):
    '''
    Simplified async interface for MLX model completions.

    Example:
        runner = local_model_runner('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')
        resp = await runner('What is 2 + 2?')
        # Or with tools:
        resp = await runner('What is 2 + 2?', tools=['calculator'])
    '''
    async def __call__(self, prompt, full_response=False, tools=None, json_schema=None, max_trips=3,
                       tool_choice=TOOL_CHOICE_AUTO, temperature=None, sysprompt=None, **kwargs):
        '''
        Convenience interface to complete a prompt, optionally using tools or schema constraints

        Args:
            prompt (string or List[dict]): Chat messages from which generation proceeds, or user prompt to be turned into a message
            full_response (bool) - If True, return full llm_response object instead of just text
            tools (list): Tools specified for this request, presumably a subset of overall tool registry
                Each entry is either:
                * a tool name to use the registered implementation and schema
                * a complete tool specification dict, using only registered implementation
            max_trips (int): Maximum number of times in a tool-calling sequence to request LLM completion. Ignored if there are no tools.
            tool_choice (str): Control over tool selection ('auto', 'none', etc). Ignored if there are no tools.
            temperature (float): Optional sampling temperature; Affects how likely the LLM is to select statistically less common tokens
            **kwargs: Additional arguments passed to completion
        '''
        if tools and json_schema:
            raise ValueError('Cannot specify both tools and a JSON schema')

        # Convert string prompt to chat messages if needed
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
        if sysprompt:
            # Add system prompt if provided
            messages.insert(0, {'role': 'system', 'content': sysprompt})

        if tools:
            resp = await self.complete_with_tools(messages, full_response=full_response, tools=tools, max_trips=max_trips,
                                                tool_choice=tool_choice, temperature=temperature, **kwargs)
        else:
            resp = await self.complete(messages, full_response=full_response, json_schema=json_schema, temperature=temperature, **kwargs)

        if resp is None:
            raise RuntimeError('No response from LLM')

        return resp
