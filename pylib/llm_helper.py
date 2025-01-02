# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import json
import logging

from toolio.common import extract_content  # Just really for legacy import patterns # noqa: F401
from toolio.toolcall import mixin as toolcall_mixin, process_tools_for_sysmsg, TOOL_CHOICE_AUTO, DEFAULT_INTERNAL_TOOLS
from toolio.schema_helper import Model
# from toolio.prompt_helper import set_tool_response, set_continue_message, process_tools_for_sysmsg
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


class model_manager(toolcall_mixin):
    def __init__(self, model_path, tool_reg=None, logger=logging, sysmsg_leadin=None, remove_used_tools=True):
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
                         sysmsg_leadin=sysmsg_leadin, remove_used_tools=remove_used_tools)

    async def iter_complete(self, messages, stream=True, json_schema=None, max_tokens=128, temperature=0.1):
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
        Yields:
            str: response chunks
        '''
        schema = None
        # Regular LLM completion; no steering
        if stream:
            responder = ChatCompletionStreamingResponder(self.model_path, self.model_type)
        else:
            responder = ChatCompletionResponder(self.model_path, self.model_type)
        if json_schema:
            if isinstance(json_schema, str):
                schema = json.loads(json_schema)
            else:
                schema = json_schema
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
    async def iter_complete_with_tools(self, messages, toolset=None, stream=False, json_schema=None, max_trips=3,
                                  tool_choice=TOOL_CHOICE_AUTO, max_tokens=128, temperature=0.1):
        '''
        Make a chat completion with tools, then continue to iterate completions as long as the LLM
        is using at least one tool, or until max_trips are exhausted

        Args:
            messages (str) - Prompt in the form of list of messages to send ot the LLM for completion.
                If you have a system prompt, and you are setting up to call tools, it will be updated with
                the tool spec

            toolset (list) - tools specified for this request, presumably a subset of overall tool registry.
                Each entry is either a tool name, in which the invocation schema is as registered, or a full
                tool-calling format stanza, in which case, for this request, only the implementaton is used
                from the initial registry

            json_schema - JSON schema to be used to guide the final generation step (after all tool-calling, if any)

            max_tokens (int, optional): Maximum number of tokens to generate

            temperature (float, optional): Affects how likely the LLM is to select statistically less common tokens
        Yields:
            str: response chunks
        '''
        toolset = toolset or self.toolset
        # schema_str = json.dumps(json_schema)
        req_tools = self._resolve_tools(toolset)
        req_tool_spec = [ s for f, s in req_tools.values() ]

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')
        while max_trips:
            first_chunk = True
            tool_call_resp = None
            # {'choices': [{'index': 0, 'message': {'role': 'assistant',
            # 'tool_calls': [{'id': 'call_6434200784_1722311129_0', 'type': 'function',
            # 'function': {'name': 'square_root', 'arguments': '{"square": 256}'}}]}, 'finish_reason': 'tool_calls'}],
            # 'usage': {'completion_tokens': 24, 'prompt_tokens': 15, 'total_tokens': 39},
            # 'object': 'chat.completion', 'id': 'chatcmpl-6434200784_1722311129', 'created': 1722311129,
            # 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit', 'toolio.model_type': 'llama'}
            first_resp = None
            async for resp in self._completion_trip(messages, stream, req_tool_spec, max_tokens=max_tokens,
                                                    temperature=temperature):
                if first_resp is None:
                    first_resp = resp
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
                bypass_response = self._check_tool_handling_bypass(first_resp)
                if bypass_response:
                    # LLM called an internal tool either as a bypass, or for finishing up; treat as direct response
                    yield bypass_response
                    break

                called_names = await self._handle_tool_responses(messages, first_resp, req_tools, req_tool_spec)

                # Possibly move into _handle_tool_responses? If so, same in llm_helper.py
                if self._remove_used_tools:
                    # Many FLOSS LLMs get confused if they see a tool definition still in the response back
                    # And loop back with a new tool request. Remove it to avoid this.
                    req_tools = {k: v for (k, v) in req_tools.items() if k not in called_names}
                    req_tool_spec = [s for f, s in req_tools.values()]

            else:
                # No tools called, so no more trips
                break
            if not req_tool_spec:
                # This is the final call, with all tools removed, so just do a regular completion
                # XXX: Interplay between tool use & schema is actually much trickier than it seems, at first
                async for resp in self.iter_complete(messages, json_schema=json_schema, stream=stream, max_tokens=max_tokens,
                                                temperature=temperature):
                    yield resp
                break

    async def _completion_trip(self, messages, stream, req_tool_spec, max_tokens=128, temperature=0.1):
        # schema, tool_sysmsg = process_tool_sysmsg(req_tool_spec, self.logger, leadin=self.sysmsg_leadin)
        # Schema, including no-tool fallback, plus string spec of available tools, for use in constructing sysmsg
        full_schema, tool_schemas, sysmsg = process_tools_for_sysmsg(req_tool_spec, self._internal_tools)
        if stream:
            responder = ToolCallStreamingResponder(self.model, self.model_path, tool_schemas)
        else:
            responder = ToolCallResponder(self.model_path, self.model_type)
        messages = self.reconstruct_messages(messages, sysmsg=sysmsg)
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt=False
        async for resp in self._do_completion(messages, full_schema, responder, cache_prompt=cache_prompt,
                                                max_tokens=max_tokens, temperature=temperature):
            yield resp

    async def _do_completion(self, messages, schema, responder, cache_prompt=False, max_tokens=128, temperature=0.1):
        'Actually trigger the low-level sampling'
        prompt_tokens = None
        # print(f'🧰 Tool {schema=}\n{sysmsg=}', file=sys.stderr)
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
                raise RuntimeError(f'Unknown result operation {result["op"]}')


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
                       max_tokens=128, temperature=0.1):
        '''
        Complete a prompt, optionally using tools or schema constraints.

        Args:
            messages (List[str]) - Prompt in the form of list of messages to send ot the LLM for completion.
                If you have a system prompt, and you are setting up to call tools, it will be updated with
                the tool spec

            json_schema (dict or str) - JSON schema to be used to structure the generated response.
                If given a a string, it will be decoded as JSON

            max_tokens (int, optional): Maximum number of tokens to generate

            temperature (float, optional): Affects how likely the LLM is to select statistically less common tokens

        Args:
            prompt (str or list): Text prompt or list of chat messages
            tools (list, optional): List of tool names or specs to make available
            json_schema (dict, optional): Schema to constrain the response
            max_trips (int): Maximum number of tool-calling round trips
            tool_choice (str): How tools should be selected ('auto', 'none', etc)
            max_tokens (int): Maximum tokens to generate per completion
            temperature (float): Sampling temperature

        Returns:
            Response text if no tools used, otherwise the full response object
        '''
        # Convert string prompt to chat messages if needed
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]

        if tools:
            async for resp in self.iter_complete_with_tools(messages, toolset=tools, stream=False, json_schema=json_schema,
                max_trips=max_trips, tool_choice=tool_choice, max_tokens=max_tokens, temperature=temperature):
                return resp
        else:
            async for resp in self.iter_complete(messages, json_schema=json_schema, stream=False, max_tokens=max_tokens,
                temperature=temperature):
                return resp

    async def complete(self, prompt, **kwargs):
        '''
        Simple completion without tools. Returns just the response text.
        
        Args:
            prompt (str or list): Text prompt or list of chat messages
            **kwargs: Additional arguments passed to __call__
        '''
        resp = await self(prompt, **kwargs)
        if isinstance(resp, str):
            return resp
        # Extract text from response object
        return resp.first_choice_text if hasattr(resp, 'first_choice_text') else resp['choices'][0]['message']['content']

    async def complete_with_tools(self, prompt, tools, **kwargs):
        '''
        Complete using specified tools. Returns full response object.
        
        Args:
            prompt (str or list): Text prompt or list of chat messages
            tools (list): List of tool names or specs to make available
            **kwargs: Additional arguments passed to __call__
        '''
        return await self(prompt, tools=tools, **kwargs)


class debug_model_manager(model_manager):
    def __init__(self, model_path, tool_reg=None, logger=logging, sysmsg_leadin=None, remove_used_tools=True):
        super().__init__(model_path, tool_reg=tool_reg, logger=logger, sysmsg_leadin=sysmsg_leadin,
                         remove_used_tools=remove_used_tools)
        self._trip_log = None

    async def _completion_trip(self, messages, stream, req_tool_spec, max_tokens=128, temperature=0.1):
        '''
        Execute one LLM request, while taking debug info
        '''
        if self._trip_log is None:
            self._trip_log = []
        full_schema, tool_schemas, sysmsg = process_tools_for_sysmsg(req_tool_spec, self._internal_tools)
        self._trip_log.append(({'messages': messages, 'schema': full_schema}))
        if stream:
            responder = ToolCallStreamingResponder(self.model, self.model_path)
        else:
            responder = ToolCallResponder(self.model_path, self.model_type)
        messages = self.reconstruct_messages(messages, sysmsg=sysmsg)
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        cache_prompt=False
        resp_chunks = []
        async for resp in self._do_completion(messages, full_schema, responder, cache_prompt=cache_prompt,
                                                max_tokens=max_tokens, temperature=temperature):
            resp_chunks.append(resp)
            yield resp
        self._trip_log[-1]['resp_chunks'] = resp_chunks

    def get_trip_log(self):
        '''
        Postproces & return log of trips
        '''
        if self._trip_log is None:
            raise RuntimeError('get_trip_log must be called after a completion call')

        for trip in self._trip_log:
            # Tokenize chat messages
            trip['tokenized_prompt'] = self.model.simple_tokenizer.apply_chat_template(trip['messages'], tokenize=False)
            # JSONize schema
            trip['schema.json'] = json.dumps(trip['schema'], indent=2)

        trip_log = self._trip_log
        self._trip_log = None
        return trip_log

    def write_trip_log(self, trip_log, fp):
        '''
        Write the trip log to a stream in cut & paste, debug-friendly form
        '''
        for i, trip in enumerate(trip_log):
            fp.write('='*8 + f' TRIP {i} ' + '='*40 + '\n')
            fp.write('-'*8 + ' PROMPT ' + '='*40 + '\n')
            fp.write(trip['tokenized_prompt'] + '\n')
            fp.write('-'*8 + ' SCHEMA ' + '='*40 + '\n')
            fp.write(trip['schema.json'] + '\n')
            fp.write('-'*48)
