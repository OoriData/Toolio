# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.common
'''
Basically stuff that can be imported without MLX (e.g. for client use on non-Mac platforms)
'''
# import sys
import json
import logging
import importlib
import warnings
from pathlib import Path  # noqa: E402
from enum import Flag, auto

# import mlx.core as mx
# from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
#                            mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
#                            stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

from ogbujipt import word_loom

from toolio.http_schematics import V1Function
from toolio.util import check_callable


TOOL_CHOICE_AUTO = 'auto'
TOOL_CHOICE_NONE = 'none'

class model_flag(Flag):
    NO_SYSTEM_ROLE = auto()  # e.g. Gemma blows up if you use a system message role
    USER_ASSISTANT_ALT = auto()  # Model requires alternation of message roles user/assistant only
    TOOL_RESPONSE = auto()  # Model expects responses from tools via OpenAI API style messages


DEFAULT_FLAGS = model_flag(0)

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

TOOLIO_MODEL_TYPE_FIELD = 'toolio.model_type'


# FIXME: Replace with utiloori.filepath.obj_file_path_parent
def obj_file_path_parent(obj):
    '''Cross-platform Python trick to get the path to a file containing a given object'''
    import inspect
    from pathlib import Path
    # Should already be an absolute path
    # from os.path import abspath
    # return abspath(inspect.getsourcefile(obj))
    return Path(inspect.getsourcefile(obj)).parent


HERE = obj_file_path_parent(lambda: 0)
with open(HERE / Path('resource/language.toml'), mode='rb') as fp:
    LANG = word_loom.load(fp)


class prompt_handler:
    '''
    Encapsulates functionality for manipulating prompts, client or server side
    '''
    # XXX: Default option for sysmgg?
    def __init__(self, model_type=None, logger=None, sysmsg_leadin=''):
        self.model_type = model_type
        if model_type:
            self.model_flags = FLAGS_LOOKUP.get(model_type, DEFAULT_FLAGS)
        self.sysmsg_leadin = sysmsg_leadin
        self.logger = logger or logging

    def reconstruct_messages(self, msgs, sysmsg=None):
        '''
        Take a message set and rules for prompt composition to create a new, effective prompt

        msgs - chat messages to process, potentially including user message and system message
        sysmsg - explicit override of system message
        kwargs - overrides for components for the sysmsg template
        '''
        self.logger.debug(f'Before: {msgs=}')
        if not msgs:
            raise ValueError('Unable to process an empty prompt')

        # Ensure it's a well-formed prompt, ending with at least one user message
        if msgs[-1]['role'] != 'user':
            raise ValueError('Final message in the chat prompt must have a \'user\' role')

        # Index the current system roles
        system_indices = [i for i, m in enumerate(msgs) if m['role'] == 'system']
        # roles = [m['role'] for m in msgs]
        # XXX Should we at least warn about any empty messages?

        if sysmsg:
            # Override any existing system messages by removing, then adding the one
            new_msgs = [m for m in msgs if m['role'] != 'system']
            new_msgs.insert(0, {'role': 'system', 'content': sysmsg})
        else:
            new_msgs = msgs[:]

        self.logger.debug(f'After: {new_msgs=}')
        return new_msgs


class model_client_mixin(prompt_handler):
    '''
    Encapsulates tool registry. Remember that tool-calling occurs on the client side.
    '''
    def __init__(self, model_type=None, logger=None, sysmsg_leadin='', tool_reg=None, remove_used_tools=True):
        '''
        Args:

        logger - logger object, handy for tracing operations

        sysmsg_leadin - Override the system message used to prime the model for tool-calling

        tool_reg (list) - Tools with available implementations, in registry format, i.e. each item is one of:
            * Python import path for a callable annotated (i.e. using toolio.tool.tool decorator)
            * actual callable, annotated (i.e. using toolio.tool.tool decorator)
            * tuple of (callable, schema), with separately specified schema
            * tuple of (None, schema), in which case a tool is declared (with schema) but with no implementation

        remove_used_tools - if True each tool will only be an option until the LLM calls it,
                            after which it will be removed from the options in subsequent trips
        '''
        super().__init__(model_type=model_type, logger=logger, sysmsg_leadin=sysmsg_leadin)
        self._remove_used_tools = remove_used_tools
        self._tool_registry = {}
        # Prepare library of tools
        for toolspec in (tool_reg or []):
            if isinstance(toolspec, tuple):
                funcpath_or_obj, schema = toolspec
                self.register_tool(funcpath_or_obj, schema)
            else:
                self.register_tool(toolspec)
        # Low enough level that we'll just let the user manipulate the object to change this
        self.bypass_tool_name = 'toolio_bypass'

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
            # assert schema['name'] = funcobj.name  # Do we care about this?
        # print(f'{schema=}')

        self._tool_registry[schema['name']] = (funcobj, schema)

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
            try:
                func, schema = self._tool_registry[name]
            except KeyError as e:
                raise KeyError(f'Unknown tool: {name}')
            # full_schema = {'type': 'function', 'function': schema}
            # print(f'{full_schema=}')
            # req_tools[name] = (func, full_schema)
            req_tools[name] = (func, schema)
        return req_tools

    def _check_tool_handling_bypass(self, response):
        '''
        There is a special tool option given to the LLM, toolio_bypass by default, which allows it to signal,
        despite the schema constraint, that it has chosen not to call any of the provided tools
        Check for this case, and return the bypass response, which is just the non-tool response the LLM
        opts to give
        '''
        for tc in response['choices'][0].get('message', {}).get('tool_calls'):
            if tc['function']['name'] == self.bypass_tool_name:
                args = json.loads(tc['function']['arguments'])
                bypass_resp_text = args['response']
                self.logger.debug(f'LLM chose to bypass function-calling witht he following response: {bypass_resp_text}')
                # Reconstruct an OpenAI-style response
                choice = response['choices'][0]
                full_resp = {
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': bypass_resp_text},
                                 'finish_reason': choice['finish_reason']}],
                                 # Usage is going to be based on the tool-call original, but we probably want that, because it was what's actually consumed
                                 'usage': response['usage'], 'object': response['object'], 'id': response['id'],
                                 'created': response['created'], 'model': response['model'], 'toolio.model_type':
                                 response['toolio.model_type']}
                return full_resp
        return None

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
            if self.logger:
                self.logger.debug(f'🔧 Calling tool {callee_name} with args {callee_args}')
            # FIXME: Parallelize async calls rather than blocking on each
            is_callable, is_async_callable = check_callable(tool)
            try:
                if is_async_callable:
                    result = await tool(**callee_args)
                elif is_callable:
                    result = tool(**callee_args)
            except TypeError as e:
                # For special case where function takes exactly 1 argument, can just skip keyword form
                if len(callee_args) == 1 and 'no keyword arguments' in str(e):
                    if is_async_callable:
                        result = await tool(next(iter(callee_args.values())))
                    elif is_callable:
                        result = tool(next(iter(callee_args.values())))
                else:
                    raise
            if self.logger:
                self.logger.debug(f'✅ Tool call result: {result}')

            tool_responses.append((call_id, callee_name, callee_args, result))
            # print('Tool result:', result)
            # self._pending_tool_calls[tc['id']] = (tool, callee_args)
        return tool_responses


def prep_tool(spec):
    if callable(spec):
        return spec
    if isinstance(spec, str):
        modpath, call_name  = spec.rsplit('.', 1)
        modobj = importlib.import_module(modpath)
        return getattr(modobj, call_name)


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
