# toolio.client
'''
Encapsulate HTTP query of LLMs for structured response, as hosted by MLXStructuredLMServer

Modeled on ogbujipt.llm_wrapper.openai_api & ogbujipt.llm_wrapper.openai_chat_api

'''
import sys
import json
import warnings
from enum import Flag, auto
import importlib

import httpx
# import asyncio

from amara3 import iri

from ogbujipt import config
from ogbujipt.llm_wrapper import llm_response, response_type

from toolio.util import check_callable


class tool_flag(Flag):
    ASSISTANT_RESPONSE = auto()  # Convert tool role responses to assistant role
    REMOVE_USED_TOOLS = auto()  # Remove tools which the LLM has already used from subsequent trips


DEFAULT_FLAGS = tool_flag.ASSISTANT_RESPONSE | tool_flag.REMOVE_USED_TOOLS
# Tool choice feels like it could be an enum, but it's not clear that the valus are fixed across conventions
TOOL_CHOICE_AUTO = 'auto'
TOOL_CHOICE_NONE = 'none'

HTTP_SUCCESS = 200


class struct_mlx_chat_api:
    '''
    Wrapper for OpenAI chat-style LLM API endpoint, with support for structured responses
    via schema specifiation in query

    Note: Only supports chat-style completions

    >>> import asyncio; from toolio.client import struct_mlx_chat_api
    >>> llm = struct_mlx_chat_api(base_url='http://localhost:8000')
    >>> resp = asyncio.run(llm_api(prompt_to_chat('Knock knock!')))
    >>> resp.first_choice_text
    '''
    def __init__(self, base_url=None, default_schema=None, flags=DEFAULT_FLAGS, tools=None, trace=False, **kwargs):
        '''
        Args:
            base_url (str, optional): Base URL of the API endpoint
                (should be a MLXStructuredLMServer host, or equiv)

            flags (int, optional): bitwise flags to control tool flow

            tools - Tools made available by default to all queries, tool_name: tool__path_or_obj

            trace - Print information (to STDERR) about tool call requests & results. Useful for debugging

            kwargs (dict, optional): Extra parameters for the API or for the model host
        '''
        tools = tools or {}
        self.parameters = config.attr_dict(kwargs)
        self.default_schema = default_schema
        self.base_url = base_url
        if self.base_url:
            # If the user includes the API version in the base, don't add it again
            scheme, authority, path, query, fragment = iri.split_uri_ref(base_url)
            path = path or kwargs.get('api_version', '/v1')
            self.base_url = iri.unsplit_uri_ref((scheme, authority, path, query, fragment))
        if not self.base_url:
            # FIXME: i18n
            warnings.warn('base_url not provided, so each invocation will require one', stacklevel=2)
        # OpenAI-style tool-calling LLMs give IDs to tool requests by the LLM
        # Internal structure to manage these. Key is tool_call_id; value is tuple of callable, kwargs
        # self._pending_tool_calls = {}
        self._registered_tool = {}  # tool_name: tool_obj
        self._tool_schema_stanzs = []
        self.tool_role = 'assistant' if tool_flag.ASSISTANT_RESPONSE in flags else 'tool'
        self._flags = flags
        self._trace = trace
        for toolobj in tools:
            self.register_tool(toolobj.name, toolobj, add_schema=True)

    async def __call__(self, messages, req='chat/completions', schema=None, tools=None, timeout=30.0, apikey=None,
                         max_trips=3, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            messages (str): Prompt in the form of list of messages to send ot the LLM for completion

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                best_of, echo, frequency_penalty, logit_bias, logprobs, max_tokens, n
                presence_penalty, seed, stop, stream, suffix, temperature, top_p, userq
        Returns:
            dict: JSON response from the LLM
        '''
        # Uncomment for test case construction
        # print('MESSAGES', messages, '\n', 'SCHEMA', schema, '\n', 'TOOLS', tools)
        tools = tools or {}
        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')
        schema = schema or self.default_schema
        schema_str = json.dumps(schema)

        # Replace {json_schema} references with the schema
        for m in messages:
            # Don't use actual string formatting for now, because user might have other uses of curly braces
            # Yes, this introduces the problem of escaping without depth, so much pondering required
            # Perhaps make the replacement string configurable
            m['content'] = m['content'].replace('{json_schema}', schema_str)
        req_data = {'messages': messages, **kwargs}
        if schema:
            req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}
            resp = await self.round_trip(req, req_data, timeout, apikey, **kwargs)

        elif tools or self._tool_schema_stanzs:
            tools_list = tools.get('tools', [])
            combined_tools = self._tool_schema_stanzs + tools_list
            # print(f'{combined_tools=}')
            # req_data['tools'] = tools['tools']
            req_data['tool_choice'] = tools.get('tool_choice', TOOL_CHOICE_AUTO)
            if tools_list and req_data['tool_choice'] == TOOL_CHOICE_NONE:
                warnings.warn('Tools were provided, but tool_choise was set so they\'ll never be used')
            if 'tool_options' in tools:
                req_data['tool_options'] = tools['tool_options']
            for t in tools_list:
                self.register_tool(t['function']['name'], t['function'].get('pyfunc'))
            req_data['tools'] = combined_tools

            # Enter tool-calling sequence
            llm_call_needed = True
            while max_trips > 0 and llm_call_needed:
                # If the tools list is empty (perhaps we removed the last one in a prev loop), omit it entirely
                if 'tools' in req_data and not req_data['tools']:
                    del req_data['tools']
                resp = await self.round_trip(req, req_data, timeout, apikey, **kwargs)
                max_trips -= 1
                # If LLM has asked for tool calls, prepare to loop back
                if resp['response_type'] == response_type.TOOL_CALL:
                    # self.update_tool_calls(resp)
                    if not max_trips:
                        # If there are no more available trips, don't bother calling the tools
                        return resp
                    tool_responses = await self.execute_tool_calls(resp)
                    for call_id, callee_name, result in tool_responses:
                        if self.tool_role == 'assistant':
                            messages.append({
                                'role': 'assistant',
                                'content': f'Result of the call to {callee_name}: {result}',
                            })
                        elif self.tool_role == 'tool':
                            messages.append({
                                'tool_call_id': call_id,
                                'role': self.tool_role,
                                'name': callee_name,
                                'content': str(result),
                            })
                        if tool_flag.REMOVE_USED_TOOLS in self._flags:
                            # Many FLOSS LLMs get confused if they see a tool definition still in the response back
                            # And loop back with a new tool request. Remove it to avoid this.
                            remove_list = [
                                i for (i, t) in enumerate(req_data.get('tools', []))
                                if t.get('function', {}).get('name') == callee_name]
                            # print(f'removing tools with index {remove_list} from request structure')
                            for i in remove_list:
                                req_data['tools'].pop(i)
                else:
                    llm_call_needed = False

            # Loop exited. We have a final response, or exhausted allowed trips
            if max_trips <= 0:
                # FIXME: i18n
                warnings.warn('Maximum LLM trips exhausted without a final answer')

        else:
            resp = await self.round_trip(req, req_data, timeout, apikey, **kwargs)

        return resp

    async def round_trip(self, req, req_data, timeout, apikey, **kwargs):
        '''
        Single call/response to MLXStructuredLMServer. Multiple might be involved in a single tool-calling round
        '''
        header = {'Content-Type': 'application/json'}
        # if apikey is None:
        #     apikey = self.apikey
        # if apikey:
        #     header['Authorization'] = f'Bearer {apikey}'
        async with httpx.AsyncClient() as client:
            result = await client.post(
                f'{self.base_url}/{req}', json=req_data, headers=header, timeout=timeout)
            if result.status_code == HTTP_SUCCESS:
                res_json = result.json()
                # print('RESULT_JSON', res_json)
                resp_msg = res_json['choices'][0]['message']
                assert resp_msg['role'] == 'assistant'
                resp = llm_response.from_openai_chat(res_json)
                return resp
            else:
                raise RuntimeError(f'Unexpected response from {self.base_url}{req}:\n{repr(result)}')

    def lookup_tool(self, name):
        '''
        Given a function/tool name, return the callable which implements it
        '''
        # print('lookup_tool', name)
        if name in self._registered_tool:
            return self._registered_tool[name]
        else:
            # FIXME: i18n
            raise LookupError(f'Unknown tool: {name}')

    # def update_tool_calls(self, response):
    #     # print('update_tool_calls', response)
    #     for tc in response['choices'][0].get('message', {}).get('tool_calls'):
    #         callee_name = tc['function']['name']
    #         callee_args = tc['function']['arguments_obj']
    #         tool = self.lookup_tool(callee_name)
    #         self._pending_tool_calls[tc['id']] = (tool, callee_args)

    async def execute_tool_calls(self, response):
        # print('update_tool_calls', response)
        tool_responses = []
        for tc in response['choices'][0].get('message', {}).get('tool_calls'):
            call_id = tc['id']
            callee_name = tc['function']['name']
            callee_args = tc['function']['arguments_obj']
            tool = self.lookup_tool(callee_name)
            if tool is None:
                warnings.warn(f'Tool called, but it has no function implementation: {callee_name}')
                continue
            if self._trace:
                print(f'⚙️Calling tool {callee_name} with args {callee_args}', file=sys.stderr)
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
                print(f'⚙️Tool call result: {result}', file=sys.stderr)
            tool_responses.append((call_id, callee_name, result))
            # print('Tool result:', result)
            # self._pending_tool_calls[tc['id']] = (tool, callee_args)
        return tool_responses

    def register_tool(self, name, funcpath_or_obj, add_schema=False):
        if isinstance(funcpath_or_obj, str):
            funcpath = funcpath_or_obj
            # pyfunc is in the form 'path.to.module_to_import|path.to.function'
            modpath, funcpath = funcpath.split('|')
            modobj = importlib.import_module(modpath)
            parent = modobj
            for funcname in funcpath.split('.'):
                parent = getattr(parent, funcname)
            func = parent
            assert callable(func)
            self._registered_tool[name] = func
        elif funcpath_or_obj is None:
            warnings.warn(f'No implementation provided for function: {name}')
            self._registered_tool[name] = None
        else:
            funcobj = funcpath_or_obj
            self._registered_tool[name] = funcobj
            if add_schema:
                self._tool_schema_stanzs.append(funcobj.schema)
