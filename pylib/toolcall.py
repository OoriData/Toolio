# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.prompt_helper

import json
import importlib
import warnings

from toolio.http_schematics import V1Function

from toolio.common import prompt_handler, LANG, model_flag, DEFAULT_FLAGS
from toolio.util import check_callable
from toolio.http_schematics import V1ChatMessage

CM_TOOLS_LEFT = 'Please use this information to continue your response, or to give a final response.'
CM_NO_TOOLS_LEFT = 'Please use this information to give a final response.'

TOOL_CHOICE_AUTO = 'auto'
TOOL_CHOICE_NONE = 'none'

DEFAULT_JSON_SCHEMA_CUTOUT = '#!JSON_SCHEMA!#'

TOOLIO_BYPASS_TOOL_NAME = 'toolio.bypass'
TOOLIO_BYPASS_TOOL = {
    'name': TOOLIO_BYPASS_TOOL_NAME,
    'description': 'Call this tool to indicate that no other provided tool is useful for responding to the user',
    'parameters': {'type': 'object', 'properties':
                    {'response': {'type': 'string', 'description': 'Your normal response to the user'}}}}

TOOLIO_FINAL_RESPONSE_TOOL_NAME = 'toolio.final_response'
TOOLIO_FINAL_RESPONSE_TOOL = {
    'name': TOOLIO_FINAL_RESPONSE_TOOL_NAME,
    'description': 'Give a final response once you have all the info you need, and have completed reasoning.',
    'parameters': {'type': 'object', 'properties':
                    {'response': {'type': 'string', 'description': 'Message text for the user'}}}}

DEFAULT_INTERNAL_TOOLS = (TOOLIO_BYPASS_TOOL, TOOLIO_FINAL_RESPONSE_TOOL)


class mixin(prompt_handler):
    '''
    Encapsulates tool registry. Remember that tool-calling occurs on the client side.
    '''
    def __init__(self, model_type=None, logger=None, sysmsg_leadin='', tool_reg=None, remove_used_tools=True,
                 json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT):
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
            if tc['function']['name'] in (TOOLIO_BYPASS_TOOL_NAME, TOOLIO_FINAL_RESPONSE_TOOL_NAME):
                args = json.loads(tc['function']['arguments'])
                direct_resp_text = args['response']
                self.logger.debug(f'LLM chose to bypass function-calling witht he following response: {direct_resp_text}')
                # Reconstruct an OpenAI-style response
                choice = response['choices'][0]
                full_resp = {
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': direct_resp_text},
                                 'finish_reason': choice['finish_reason']}],
                                 # Usage is going to be based on the tool-call original, but we probably want that,
                                 # because it was what's actually consumed
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

    async def _handle_tool_responses(self, messages, response, req_tools, req_tool_spec=None):
        """
        Handle tool responses and update messages accordingly
        
        Args:
            messages: Current message history
            response: Response from LLM containing tool calls
            req_tools: Dictionary of available tools
            req_tool_spec: Optional tool specifications (needed for continue message)
        """
        tool_responses = await self._execute_tool_calls(response, req_tools)

        for call_id, callee_name, callee_args, result in tool_responses:
            # print(model_type, model_flags, model_flags and model_flag.TOOL_RESPONSE in model_flags)
            set_tool_response(messages, call_id, callee_name, callee_args, str(result), 
                            model_flags=self.model_flags)

        continue_msg = CM_TOOLS_LEFT if req_tool_spec else CM_NO_TOOLS_LEFT
        set_continue_message(messages, continue_msg, model_flags=self.model_flags)

        return [callee_name for call_id, callee_name, callee_args, result in tool_responses]


def prep_tool(spec):
    if callable(spec):
        return spec
    if isinstance(spec, str):
        modpath, call_name  = spec.rsplit('.', 1)
        modobj = importlib.import_module(modpath)
        return getattr(modobj, call_name)


class multi_tool_prompt_default(dict):
    def __missing__(self, key):
        match key:
            case 'leadin':
                return LANG['multi_tool_prompt_leadin']
            case 'tail':
                return LANG['multi_tool_prompt_tail']
            case _:
                raise KeyError(f'Unknown formattr parameter {key}')


class single_tool_prompt_default(dict):
    def __missing__(self, key):
        match key:
            case 'leadin':
                return LANG['one_tool_prompt_leadin']
            case 'tail':
                return LANG['one_tool_prompt_tail']
            case _:
                raise KeyError(f'Unknown formattr parameter {key}')


PROMPT_FORMATTER = '{leadin}\n{tool_spec}\n{tail}'


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


def set_tool_response(msgs, tool_call_id, tool_name, call_args, call_result, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment in place
    tool_response - response generatded by selected tool
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # XXX: model_flags = None ⇒ assistant-style tool response. Is this the default we want?
    if model_flag.TOOL_RESPONSE in model_flags:
        msgs.append({
            'tool_call_id': tool_call_id,
            'role': 'tool',
            'name': tool_name,
            'content': call_result,
        })
    else:
        # FIXME: Separate out natural language
        tool_response_text =  f'Called tool {tool_name} with arguments {call_args}. Result: {call_result}'
        # tool_response_text = f'Result of the call to {tool_name}: {tool_result}'
        if model_flag.USER_ASSISTANT_ALT in model_flags:
            # If there is already an assistant msg from tool-calling, merge it
            if msgs[-1]['role'] == 'assistant':
                msgs[-1]['content'] += '\n\n' + tool_response_text
            else:
                msgs.append({'role': 'assistant', 'content': tool_response_text})
        else:
            msgs.append({'role': 'assistant', 'content': tool_response_text})


def set_continue_message(msgs, continue_msg, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment in place
    continue_msg - message instructing the LLM on how to continue or finalize
    '''
    if msgs[-1]['role'] == 'tool':
        # No continue message, in this case
        return
    if msgs[-1]['role'] == 'user' and model_flag.USER_ASSISTANT_ALT in model_flags:
        # Need to merge adjacent user messages
        msgs[-1]['content'] += '\n\n' + continue_msg
    else:
        msgs.append({'role': 'user', 'content': continue_msg})


# XXX: no_tool_desc can just be one of the defalted params
def process_tools_for_sysmsg(tools, internal_tools, separator='\n', **kwargs):
    '''
    Given a set of tools, format for use in a system prompt, and other modularized processing

    Args:
        no_tool_desc: description to use for the default get-out-of-jail model response
    '''
    # Start by normalizing away from Pydantic form
    tools = [ (t.dictify() if isinstance(t, V1Function) else t) for t in tools ]
    tools.extend(internal_tools)
    tool_schemas = [
        {
            'type': 'object',
            'name': fn['name'],
            'description': fn['description'],
            'properties': {
                'name': {'type': 'const', 'const': fn['name']},
                'arguments': fn['parameters'],
            },
            'required': ['name', 'arguments'],
        }
        for fn in tools
    ]
    # XXX: Do we need to differentiate prompt setup for 1 vs multiple tools?
    # if len(tool_schemas) - len(internal_tools) == 1:
    #     # Single tool provided (second is the no-tool escape)
    #     tool_str = separator.join(
    #         [f'\nTool name: {tool["name"]}\n {tool["description"]}\nInvocation schema:\n{json.dumps(ts)}'
    #             for tool, ts in zip(tools, tool_schemas) ])
    # else:
    #     # XXX: Use LANG['one_tool_prompt_schemalabel']
    #     tool_str = json.dumps(tool_schemas)
    # default_cls = single_tool_prompt_default if len(tools) == 1 else multi_tool_prompt_default

    prompt_tpl = multi_tool_prompt_default
    tool_str = json.dumps(tool_schemas)

    full_schema = {'type': 'array', 'items': {'anyOf': tool_schemas}}

    sysprompt = PROMPT_FORMATTER.format_map(prompt_tpl(tool_spec=tool_str, **kwargs))

    return full_schema, tool_schemas, sysprompt
