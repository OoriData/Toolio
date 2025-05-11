# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.client
'''
Encapsulate HTTP query of LLMs for structured response, as hosted by Toolio server

Modeled on ogbujipt.llm_wrapper.openai_api & ogbujipt.llm_wrapper.openai_chat_api

'''
import json
import warnings
import logging
# import inspect

import httpx
from amara3 import iri

from ogbujipt import config
from toolio.common import DEFAULT_JSON_SCHEMA_CUTOUT
from toolio.toolcall import (
    mixin as toolcall_mixin, process_tools_for_sysmsg, handle_pyfunc,
    TOOL_CHOICE_NONE, TOOL_CHOICE_AUTO, DEFAULT_INTERNAL_TOOLS,
    TOOLIO_BYPASS_TOOL_NAME, TOOLIO_FINAL_RESPONSE_TOOL_NAME, CM_TOOLS_LEFT, CM_NO_TOOLS_LEFT)
from toolio.response_helper import llm_response
from toolio.common import model_flag

HTTP_SUCCESS = 200

class struct_mlx_chat_api(toolcall_mixin):
    '''
    Wrapper for OpenAI chat-style LLM API endpoint, with support for structured responses
    via schema specifiation in query

    Note: Only supports chat-style completions

    >>> import asyncio; from toolio.client import struct_mlx_chat_api
    >>> llm = struct_mlx_chat_api(base_url='http://localhost:8000')
    >>> resp = asyncio.run(llm_api(prompt_to_chat('Knock knock!')))
    >>> resp.first_choice_text
    'Who\'s there?'
    '''
    def __init__(self, base_url=None, default_schema=None, flags=None, tool_reg=None, logger=logging,
                 json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT, **kwargs):
        '''
        Args:
            base_url (str, optional): Base URL of the API endpoint
            default_schema (dict, optional): Default JSON schema to use for structured responses
            flags (int, optional): bitwise flags to control tool flow
            tool_reg (list) - Tools with available implementations, in registry format, i.e. each item is one of:
                * Python import path for a callable annotated (i.e. using toolio.tool.tool decorator)
                * actual callable, annotated (i.e. using toolio.tool.tool decorator)
                * tuple of (callable, schema), with separately specified schema
                * tuple of (None, schema), in which case a tool is declared (with schema) but with no implementation

            logger - logger object, handy for tracing operations
            json_schema_cutout - Prompt text which should be replaced by actual JSON schema
            kwargs (dict, optional): Extra parameters for the API
        '''
        self.parameters = config.attr_dict(kwargs)
        self.base_url = base_url
        if self.base_url:
            # If the user includes the API version in the base, don't add it again
            scheme, authority, path, query, fragment = iri.split_uri_ref(base_url)
            path = path or kwargs.get('api_version', '/v1')
            self.base_url = iri.unsplit_uri_ref((scheme, authority, path, query, fragment))

        if not self.base_url:
            warnings.warn('base_url not provided, so each invocation will require one', stacklevel=2)

        super().__init__(model_type=None, tool_reg=tool_reg, logger=logger,
                        default_schema=default_schema, json_schema_cutout=json_schema_cutout)

    async def complete(self, messages, full_response=False, req='chat/completions', json_schema=None, sysprompt=None,
                       apikey=None, trip_timeout=90.0, max_tokens=1024, temperature=0.1, **kwargs):
        '''
        '''
        req_data = {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, **kwargs}

        if not(json_schema):
            schema, schema_str = self.default_schema, self.default_schema_str
        elif isinstance(json_schema, dict):
            schema, schema_str = json_schema, json.dumps(json_schema)
        elif isinstance(json_schema, str):
            schema, schema_str = json.loads(json_schema), json_schema
        else:
            raise ValueError(f'Invalid JSON schema: {json_schema}')

        if schema:
            req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}

        req = req.strip('/')

        resp = await self._http_trip(req, req_data, trip_timeout, apikey, **kwargs)
        return resp if full_response else llm_response.from_openai_chat(resp).first_choice_text

    async def complete_with_tools(self, messages, req='chat/completions', tools=None, tool_choice='auto', max_trips=3, sysprompt=None,
                       apikey=None, trip_timeout=90.0, max_tokens=1024, temperature=0.1, **kwargs):
        '''
        '''
        req = req.strip('/')

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')

        trips_remaining = max_trips
        resp = None
        while trips_remaining > 0:
            trips_remaining -= 1

            # Convert local tool format e.g. {'toolname': (tool_func_obj, schema)} to OpenAI format
            # schema is a dict with description, name, and parameters
            req_tools = [{'type': 'function', 'function': t[1]} for t in self._resolve_tools(tools).values()]
            if tools:
                req_data = {'messages': messages, 'tools': req_tools, 'tool_choice': tool_choice, 'temperature': temperature, **kwargs}
            else:
                req_data = {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, **kwargs}

            resp = await self._http_trip(req, req_data, trip_timeout, apikey, **kwargs)

            choices = resp.get('choices', [])

            if tools and choices and 'message' in choices[0]:
                if trips_remaining <= 0:
                    self.logger.warning('Max trips reached with pending tool calls')
                    return resp

                resp = llm_response.from_openai_chat(resp)
                model_flags = model_flag(resp.model_flags)
                results = await self._execute_tool_calls(resp.tool_calls)
                await self._handle_tool_results(messages, results, tools, model_flags=model_flags, remove_used_tools=self._remove_used_tools)
            else:
                break

            # No tool calls, return response
            # return resp.get('choices', [{}])[0].get('message', {}).get('content', '')

        return resp

    async def __call__(self, prompt, full_response=None, tools=None, json_schema=None, max_trips=3,
                       tool_choice='auto', temperature=0.1, sysprompt=None, **kwargs):
        '''
        Convenience interface to complete a prompt, optionally using tools or schema constraints
        Returns just the response text

        if `full_response` is None, the default is chosen according to tool-calling. True if there are tools, else False
        '''
        if full_response is None:
            full_response = not tools

        if tools and json_schema:
            raise ValueError('Cannot specify both tools and a JSON schema')

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')

        # Convert string prompt to chat messages if needed
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
        if sysprompt:
            messages.insert(0, {'role': 'system', 'content': sysprompt})

        if tools:
            resp = await self.complete_with_tools(messages, tools=tools,
                                                tool_choice=tool_choice, temperature=temperature, **kwargs)
            if not full_response:
                simple_resp = llm_response.from_openai_chat(resp).first_choice_text
                resp = simple_resp if simple_resp is not None else resp
        else:
            resp = await self.complete(messages, full_response=full_response, json_schema=json_schema,
                                                temperature=temperature, **kwargs)

        return resp

    async def _http_trip(self, req, req_data, timeout, apikey):
        '''Single call/response to toolio_server.  Multiple might be involved in the case of tool-calling'''
        header = {'Content-Type': 'application/json'}
        # if apikey is None:
        #     apikey = self.apikey
        # if apikey:
        #     header['Authorization'] = f'Bearer {apikey}'
        req_data['stream'] = False
        # import pprint; pprint.pprint(req_data)
        # print((self.base_url, req.strip('/')))
        async with httpx.AsyncClient() as client:
            result = await client.post(
                f'{self.base_url}/{req.strip("/")}', json=req_data, headers=header, timeout=timeout)

            if result.status_code == HTTP_SUCCESS:
                # return llm_response.from_openai_chat(result.json())
                return result.json()  # Even if it's plain text it needs to be JSON decoded b/c of Content-Type header
            else:
                raise RuntimeError(f'Unexpected response from {self.base_url}/{req}:\n{repr(result)}')

    def register_internal_tools(self):
        for tool_spec in DEFAULT_INTERNAL_TOOLS:
            self.register_tool(None, tool_spec)


def cmdline_tools_struct(tools_obj):
    'Specifying a function on the command line calls for a specialized format. Processes it for model managers'
    if isinstance(tools_obj, dict):
        # If a complete tools specification with tool_choice, extract just the tools list
        tools_list = tools_obj.get('tools', [])
    elif isinstance(tools_obj, str):
        tools_list = [tools_obj]
    else:
        tools_list = tools_obj or []

    new_tools_list = []
    for t in tools_list:
        if isinstance(t, dict):
            # If already a complete function specification, pass it through
            if t.get('type') == 'function' and 'function' in t:
                new_tools_list.append(t)
            else:
                # Otherwise wrap it as a function specification
                tf = t.copy()
                new_tools_list.append({
                    'type': 'function',
                    'function': tf
                })
                if 'pyfunc' in tf:
                    del tf['pyfunc']
        else:
            # tool_func = handle_pyfunc(t)
            # print(tool_func.schema)
            # new_tools_list.append(tool_func.schema)
            new_tools_list.append(t)
    return new_tools_list
