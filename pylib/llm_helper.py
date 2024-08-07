# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.llm_helper

import sys
import json
import importlib
import warnings

# import mlx.core as mx
# from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
#                            mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
#                            stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

from toolio import model_flag, DEFAULT_FLAGS
from toolio.util import check_callable
from toolio.schema_helper import Model
from toolio.http_schematics import V1Function
from toolio.prompt_helper import set_tool_response, process_tool_sysmsg
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


TOOL_CHOICE_AUTO = 'auto'
TOOL_CHOICE_NONE = 'none'

# {model_class: flags}, defaults to DEFAULT_FLAGS
FLAGS_LOOKUP = {
    # Actually Llama seems to want asistant response rather than as tool
    # 'llama': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT | model_flag.TOOL_RESPONSE,
    'llama': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'gemma': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'gemma2': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mixtral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mistral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
}


def prep_tool(spec):
    if callable(spec):
        return spec
    if isinstance(spec, str):
        modpath, call_name  = spec.rsplit('.', 1)
        modobj = importlib.import_module(modpath)
        return getattr(modobj, call_name)


class model_manager:
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
        self.model_flags = FLAGS_LOOKUP.get(self.model_type, DEFAULT_FLAGS)
        self.sysmsg_leadin = sysmsg_leadin
        self._trace = trace
        self._remove_used_tools = remove_used_tools
        self._tool_registry = {}
        # Prepare library of tools
        for toolspec in (tool_reg or []):
            if isinstance(toolspec, tuple):
                funcpath_or_obj, schema = toolspec
                self.register_tool(funcpath_or_obj, schema)
            else:
                self.register_tool(toolspec)

    def register_tool(self, funcpath_or_obj, schema=None):
        '''
        Register a single tool for Toolio model use

        toolspec - Tool with available implementations, in registry format, i.e. one of:
            * a Python import path for a callable annotated (i.e. using tool.tool decorator)
            * actual callable, annotated (i.e. using toolio.tool.tool decorator)
            * plain callable, with schema provided
            * None, with schema provided, in which case a tool is declared but lacking implementation
        
        schema - explicit schema, i.e. {'name': tname, 'description': tdesc,
                  'parameters': {'type': 'object', 'properties': t_schema_params, 'required': t_required_list}}

        '''
        if funcpath_or_obj is None:
            funcobj = funcpath_or_obj
            warnings.warn(f'No implementation provided for function: {getattr(funcobj, "name", "UNNAMED")}')
        elif isinstance(funcpath_or_obj, str):
            funcpath = funcpath_or_obj
            if '|' in funcpath:
                # pyfunc is in the form 'path.to.module_to_import|path.to.function'
                modpath, funcpath = funcpath.split('|')
            else:
                modpath, funcpath = funcpath.rsplit('.', 1)
            modobj = importlib.import_module(modpath)
            parent = modobj
            for funcname in funcpath.split('.'):
                parent = getattr(parent, funcname)
            func = parent
            assert callable(func)
            funcobj = func
        else:
            funcobj = funcpath_or_obj

        # Explicit schema overrides implicit
        if not schema:
            # hasattr(funcpath_or_obj, 'schema')
            schema = getattr(funcobj, 'schema', None)
            if schema is None:
                raise RuntimeError(f'No schema provided for tool function {funcobj}')
            # assert schema['name'] = funcobj.name  # Do we care abotu this?

        self._tool_registry[schema['name']] = (funcobj, schema)

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

    @property
    def toolset(self):
        return self._tool_registry.keys()

    def clear_tools(self):
        'Remove all tools from registry'
        self._tool_registry = {}

    def _resolve_tools(self, toolset):
        'Narrow down & process list of tools to the ones specified on this request'
        req_tools = {}
        for tool in toolset:
            if isinstance(tool, str):  # It's a tool name
                name = tool
            elif isinstance(tool, dict):  # Plain schema
                name = tool['name']
            elif isinstance(tool, V1Function):  # Pydantic-style schema
                name = tool.name
            func, schema = self._tool_registry[name]
            # full_schema = {'type': 'function', 'function': schema}
            # print(f'{full_schema=}')
            # req_tools[name] = (func, full_schema)
            req_tools[name] = (func, schema)
        return req_tools

    async def _execute_tool_calls(self, response, req_tools):
        # print('update_tool_calls', response)
        tool_responses = []
        for tc in response['choices'][0].get('message', {}).get('tool_calls'):
            call_id = tc['id']
            callee_name = tc['function']['name']
            if 'arguments' in tc['function']:
                callee_args = json.loads(tc['function']['arguments'])
            else:
                callee_args = tc['function']['arguments_obj']
            tool, _ = req_tools.get(callee_name, (None, None))
            if tool is None:
                warnings.warn(f'Tool called, but it has no function implementation: {callee_name}')
                continue
            if self._trace:
                # FIXME: logger
                print(f'⚙️ Calling tool {callee_name} with args {callee_args}', file=sys.stderr)
            # FIXME: Parallelize async calls rather than blocking on each
            try:
                is_callable, is_async_callable = check_callable(tool)
                if is_async_callable:
                    result = await tool(**callee_args)
                elif is_callable:
                    result = tool(**callee_args)
            except TypeError as e:
                # try for special case where the function takes exactly 1 argument
                if len(callee_args) == 1 and 'no keyword arguments' in str(e):
                    if is_async_callable:
                        result = await tool(next(iter(callee_args.values())))
                    elif is_callable:
                        result = tool(next(iter(callee_args.values())))
                else:
                    raise
            if self._trace:
                print(f'⚙️ Tool call result: {result}', file=sys.stderr)
            tool_responses.append((call_id, callee_name, result))
            # print('Tool result:', result)
            # self._pending_tool_calls[tc['id']] = (tool, callee_args)
        return tool_responses

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


async def extract_content(resp_stream):
    # Interpretthe streaming pattern from the API. viz https://platform.openai.com/docs/api-reference/streaming
    async for chunk in resp_stream:
        # Minimal checking: Trust the delivered structure
        if 'delta' in chunk['choices'][0]:
            content = chunk['choices'][0]['delta'].get('content')
            if content is not None:
                yield content
        else:
            content = chunk['choices'][0]['message'].get('content')
            if content is not None:
                yield content
