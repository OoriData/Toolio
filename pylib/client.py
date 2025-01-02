# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.client
'''
Encapsulate HTTP query of LLMs for structured response, as hosted by MLXStructuredLMServer

Modeled on ogbujipt.llm_wrapper.openai_api & ogbujipt.llm_wrapper.openai_chat_api

'''
# import logging
import json
import warnings
import logging
from enum import Flag, auto
from typing import AsyncGenerator, Any

import httpx
# import asyncio

from amara3 import iri

from ogbujipt import config
from ogbujipt.llm_wrapper import llm_response, response_type

# from toolio.common import FLAGS_LOOKUP  # TOOL_CHOICE_AUTO
from toolio.common import DEFAULT_FLAGS as DEFAULT_MODEL_FLAGS
from toolio.toolcall import mixin as toolcall_mixin
from toolio.toolcall import set_tool_response, TOOL_CHOICE_NONE, DEFAULT_JSON_SCHEMA_CUTOUT


class tool_flag(Flag):
    REMOVE_USED_TOOLS = auto()  # Remove tools which the LLM has already used from subsequent trips


DEFAULT_FLAGS = tool_flag.REMOVE_USED_TOOLS
# Tool choice feels like it could be an enum, but it's not clear that the valus are fixed across conventions

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
    '''
    def __init__(self, base_url=None, default_schema=None, flags=DEFAULT_FLAGS, tool_reg=None, logger=logging,
                 json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT, **kwargs):
        '''
        Args:
            base_url (str, optional): Base URL of the API endpoint
                (should be a MLXStructuredLMServer host, or equiv)

            default_schema (dict, optional): Default JSON schema to use for structured responses

            flags (int, optional): bitwise flags to control tool flow

            tool_reg (list) - Tools with available implementations, in registry format, i.e. each item is one of:
                * Python import path for a callable annotated (i.e. using toolio.tool.tool decorator)
                * actual callable, annotated (i.e. using toolio.tool.tool decorator)
                * tuple of (callable, schema), with separately specified schema
                * tuple of (None, schema), in which case a tool is declared (with schema) but with no implementation

            logger - logger object, handy for tracing operations

            json_schema_cutout - Prompt text which should be replaced by actual JSON schema

            kwargs (dict, optional): Extra parameters for the API or for the model host
        '''
        self.parameters = config.attr_dict(kwargs)
        self.default_schema = default_schema
        self.default_schema_str = json.dumps(default_schema) if default_schema else None
        self.json_schema_cutout = json_schema_cutout
        self.base_url = base_url
        if self.base_url:
            # If the user includes the API version in the base, don't add it again
            scheme, authority, path, query, fragment = iri.split_uri_ref(base_url)
            path = path or kwargs.get('api_version', '/v1')
            self.base_url = iri.unsplit_uri_ref((scheme, authority, path, query, fragment))
            # self.base_url = self.base_url.rstrip('/')  # SHould already e free of trailing /
        if not self.base_url:
            # FIXME: i18n
            warnings.warn('base_url not provided, so each invocation will require one', stacklevel=2)
        # OpenAI-style tool-calling LLMs give IDs to tool requests by the LLM
        # Internal structure to manage these. Key is tool_call_id; value is tuple of callable, kwargs
        # self._pending_tool_calls = {}
        self._flags = flags
        super().__init__(tool_reg=tool_reg, logger=logger)

    async def iter_call(self, messages, req='chat/completions', json_schema=None, toolset=None, sysprompt=None,
                       tool_choice='auto', apikey=None, max_trips=3, trip_timeout=90.0, json_schema_cutout=None,
                       max_tokens=1024, temperature=0.1, **kwargs):
        '''
        Invoke the LLM with a completion request. Foundation method for making API calls to the LLM server.

        Args:
            messages (str) - Prompt in the form of list of messages to send ot the LLM for completion.
                If you have a system prompt, and you are setting up to call tools, it will be updated with
                the tool spec

            req (str) - API endpoint to invoke

            json_schema - JSON schema to be used to guide the final generation step (after all tool-calling, if any)

            sysprompt (str) - System prompt to use in the chat messages

            toolset (list) - tools specified for this request, presumably a subset of overall tool registry.
                Each entry is either a tool name, in which the invocation schema is as registered, or a full
                tool-calling format stanza, in which case, for this request, only the implementaton is used
                from the initial registry

            trip_timeout (float) - timeout (in seconds) per LLM API request trip; defaults to 90s

            json_schema_cutout (str) - Prompt text which should be replaced by actual JSON schema;
                overrides instance default

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                temperature, max_tokens, best_of, echo, frequency_penalty, logit_bias, logprobs,
                presence_penalty, seed, stop, stream, suffix, top_p, userq

                Note: for now additional kwargs are ignored. stream is defintiely always forced to False
        Yields:
            dict: Response from the LLM, either in a single chunk, or multiple, depending on streaming and tooling
        '''
        if kwargs.get('stream'):
            warnings.warn('For the HTTP client stream is always forced to False for now', stacklevel=2)
            kwargs['stream'] = False
        # Uncomment for test case construction
        # print('MESSAGES', messages, '\n', 'json_schema', json_schema, '\n', 'TOOLS', toolset)
        toolset = toolset or self.toolset
        json_schema_cutout = json_schema_cutout or self.json_schema_cutout
        assert max_trips > 0

        req_data = {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, **kwargs}

        if not(json_schema):
            schema, schema_str = self.default_schema, self.default_schema_str
        elif isinstance(json_schema, dict):
            schema, schema_str = json_schema, json.dumps(json_schema)
        elif isinstance(json_schema, str):
            schema, schema_str = json.loads(json_schema), json_schema
        else:
            raise ValueError(f'Invalid JSON schema: {json_schema}')

        if json_schema:
            req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}

        req = req.strip('/')

        if max_trips < 1:
            raise ValueError(f'At least one trip must be permitted, but {max_trips=}')

        def replace_cutout():  # Will modify messages in place
            '''Replace JSON schema cutout references with the actual schema'''
            cutout_replaced = False
            for m in messages:
                # XXX: content should always be in m, though. Validate?
                if 'content' in m and json_schema_cutout in m['content']:
                    m['content'] = m['content'].replace(json_schema_cutout, schema_str)
                    cutout_replaced = True

            if not cutout_replaced:
                warnings.warn('JSON Schema provided, but no place found to replace it.'
                            ' Will be tacked on the end of the first user message', stacklevel=2)
                target_msg = next(m for m in messages if m['role'] == 'user')
                # FIXME: More robust message validation, perhaps add a helper in prompt_helper.py
                assert target_msg is not None
                target_msg['content'] += '\nRespond in JSON according to this schema: ' + schema_str

        req_data = {'messages': messages, **kwargs}
        if toolset:
            req_tools = self._resolve_tools(toolset)
            req_tool_spec = [{'type': 'function', 'function': s} for f, s in req_tools.values()]
            req_data['tools'] = req_tool_spec
            req_data['tool_choice'] = tool_choice
            if req_tools and tool_choice == TOOL_CHOICE_NONE:
                warnings.warn('Tools were provided, but tool_choice was set to `none`, so they\'ll never be used')
            # if tool_options: req_data['tool_options'] = tool_options
            # for t in tools_list:
            #     self.register_tool(t['function']['name'], t['function'].get('pyfunc'))
            req_data['tools'] = req_tool_spec

            # Enter tool-calling sequence
            llm_call_needed = True
            while max_trips > 0 and llm_call_needed:
                # If the tools list is empty (perhaps we removed the last one in a prev loop), omit it entirely
                if 'tools' in req_data and not req_data['tools']:
                    del req_data['tools']
                    # XXX: Interplay between tool use & schema is actually much trickier than it seems, at first
                    if schema:
                        req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}
                async for chunk in self._http_trip(req, req_data, trip_timeout, apikey, **kwargs):
                    resp = chunk
                max_trips -= 1
                # If LLM has asked for tool calls, prepare to loop back
                if resp['response_type'] == response_type.TOOL_CALL:
                    bypass_response = self._check_tool_handling_bypass(resp)
                    if bypass_response:
                        # LLM refused to call a tool, and provided an alternative response
                        yield llm_response.from_openai_chat(bypass_response)
                        break

                    if not max_trips:
                        # No more available trips; don't bother calling tools
                        self.logger.debug('Maximum trips exhausted')
                        yield resp
                        break

                    called_names = await self._handle_tool_responses(messages, resp, req_tools)
                    
                    # Possibly combine with llm_helper.complete_with_tools & move into _handle_tool_responses?
                    if tool_flag.REMOVE_USED_TOOLS in self._flags:
                        # Many FLOSS LLMs get confused if they see a tool definition still in the response back
                        # And loop back with a new tool request. Remove it to avoid this.
                        remove_list = [
                            i for (i, t) in enumerate(req_data.get('tools', []))
                            if t.get('function', {}).get('name') in called_names]
                        # print(f'removing tools with index {remove_list} from request structure')
                        for i in remove_list:
                            req_data['tools'].pop(i)

                else:
                    llm_call_needed = False

            # Loop exited. We have a final response, or exhausted allowed trips
            if max_trips <= 0:
                # FIXME: i18n
                warnings.warn('Maximum LLM trips exhausted without a final answer')

            yield resp  # Most recent response is final

        else:
            if schema:
                replace_cutout()
                req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}

            async for chunk in self._http_trip(req, req_data, trip_timeout, apikey, **kwargs):
                yield chunk

    async def iter_complete(self, messages, stream=False, json_schema=None, max_tokens=128, temperature=0.1):
        __doc__ = 'Wrapper, without tool-calling, for the following\n' + self.iter_call.__doc__
        call_result: AsyncGenerator[Any, None] = self.iter_call(  # type: ignore
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,  # Passed on only so they can get the warning, if need be
            json_schema=json_schema
        )
        async for chunk in call_result:
            yield chunk

    async def iter_complete_with_tools(self, messages, toolset=None, stream=False, json_schema=None, 
                                max_trips=3, tool_choice='auto', max_tokens=128, temperature=0.1):
        __doc__ = 'Wrapper, with tool-calling, for the following\n' + self.iter_call.__doc__
        call_result: AsyncGenerator[Any, None] = self.iter_call(  # type: ignore
            messages=messages,
            toolset=toolset,
            max_trips=max_trips,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,  # Passed on only so they can get the warning, if need be
            json_schema=json_schema
        )
        async for chunk in call_result:
            yield chunk

    async def __call__(self, messages, req='chat/completions', json_schema=None, toolset=None, sysprompt=None,
                       tool_choice='auto', apikey=None, max_trips=3, trip_timeout=90.0, json_schema_cutout=None,
                       max_tokens=1024, temperature=0.1, **kwargs):
        __doc__ = 'Convenience, synchronous wrapper for the following\n' + self.iter_call.__doc__
        if toolset:
            return await self.complete_with_tools(messages, toolset=toolset, stream=False, json_schema=json_schema,
                                max_trips=max_trips, tool_choice=tool_choice, max_tokens=max_tokens, temperature=temperature)
        else:
            return await self.complete(messages, stream=False, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature)

    async def complete(self, messages, stream=False, json_schema=None, max_tokens=128, temperature=0.1):
        __doc__ = 'Wrapper, without tool-calling, for the following\n' + self.iter_call.__doc__
        call_result: AsyncGenerator[Any, None] = self.iter_call(  # type: ignore
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,  # Passed on only so they can get the warning, if need be
            json_schema=json_schema
        )
        async for resp in call_result:
            return resp  # First response chunk should be the only & everything

    async def complete_with_tools(self, messages, toolset=None, stream=False, json_schema=None, 
                                max_trips=3, tool_choice='auto', max_tokens=128, temperature=0.1):
        __doc__ = 'Wrapper, with tool-calling, for the following\n' + self.iter_call.__doc__
        call_result: AsyncGenerator[Any, None] = self.iter_call(  # type: ignore
            messages=messages,
            toolset=toolset,
            max_trips=max_trips,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,  # Passed on only so they can get the warning, if need be
            json_schema=json_schema
        )
        async for resp in call_result:
            return resp  # First response chunk should be the only & everything

    # Were I to decide to uglify with the type annotations
    # async def sync_call(self, messages, req: str = 'chat/completions',
    #                 json_schema: Optional[dict] = None,
    #                 toolset: Optional[list] = None,
    #                 sysprompt: Optional[str] = None,
    #                 tool_choice: str = 'auto',
    #                 apikey: Optional[str] = None,
    #                 max_trips: int = 3,
    #                 trip_timeout: float = 90.0,
    #                 json_schema_cutout: Optional[str] = None,
    #                 max_tokens: int = 1024,
    #                 temperature: float = 0.1,
    #                 **kwargs) -> Any:
    async def sync_call(self, messages, req='chat/completions', json_schema=None, toolset=None, sysprompt=None,
                    tool_choice='auto', apikey=None, max_trips=3, trip_timeout=90.0, json_schema_cutout=None,
                    stream=False, max_tokens=1024, temperature=0.1, **kwargs):
        __doc__ = 'Synchronous wrapper for the following\n' + self.iter_call.__doc__
        call_result: AsyncGenerator[Any, None] = self.iter_call(  # type: ignore
            messages=messages,
            req=req,
            json_schema=json_schema,
            toolset=toolset,
            sysprompt=sysprompt,
            tool_choice=tool_choice,
            apikey=apikey,
            max_trips=max_trips,
            trip_timeout=trip_timeout,
            json_schema_cutout=json_schema_cutout,
            stream=stream,  # Passed on only so they can get the warning, if need be
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        async for resp in call_result:
            return resp  # First response chunk should be the only & everything

    async def _http_trip(self, req, req_data, timeout, apikey, **kwargs):
        '''
        Single call/response to toolio_server. Multiple might be involved in a single tool-calling round

        req must not end with '/'
        '''
        header = {'Content-Type': 'application/json'}
        # if apikey is None:
        #     apikey = self.apikey
        # if apikey:
        #     header['Authorization'] = f'Bearer {apikey}'
        req_data['stream'] = False

        async with httpx.AsyncClient() as client:
            result = await client.post(
                f'{self.base_url}/{req.strip("/")}', json=req_data, headers=header, timeout=timeout)

            if result.status_code == HTTP_SUCCESS:
                # res_json = result.json()
                # resp_msg = res_json['choices'][0]['message']
                # assert resp_msg['role'] == 'assistant'
                # resp = llm_response.from_openai_chat(res_json)

                # if stream:
                #     yield result.aiter_lines()
                # else:
                #     yield llm_response.from_openai_chat(result.json())
                yield llm_response.from_openai_chat(result.json())
            else:
                raise RuntimeError(f'Unexpected response from {self.base_url}/{req}:\n{repr(result)}')

    def lookup_tool(self, name):
        '''
        Given a function/tool name, return the callable which implements it
        '''
        # print('lookup_tool', name)
        if name in self._tool_registry:
            return self._tool_registry[name]
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


def cmdline_tools_struct(tools_obj):
    'Specifying a function on the command line calls for a specialized format. Processes it for model managers'
    if isinstance(tools_obj, dict):
        tools_list = tools_obj['tools']
    elif isinstance(tools_obj, str):
        tools_list = [tools_obj]
    else:
        tools_list = tools_obj or []
    new_tools_list = []
    for t in tools_list:
        if isinstance(t, dict):
            tf = t['function']
            new_tools_list.append((tf.get('pyfunc'), tf))
            if 'pyfunc' in tf: del tf['pyfunc']   # noqa: E701
        else:
            new_tools_list.append(t)
    return new_tools_list
