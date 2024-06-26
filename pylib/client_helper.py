# toolio.client_helper
'''
Tools to help with HTTP query of LLMs for structured response, as hosted by MLXStructuredLMServer

Modeled on ogbujipt.llm_wrapper.openai_api & ogbujipt.llm_wrapper.openai_chat_api

'''
# import sys
import json
import warnings
# from enum import Enum
# from dataclasses import dataclass
import importlib

import httpx
# import asyncio

from amara3 import iri

from ogbujipt import config
from ogbujipt.llm_wrapper import llm_response, response_type

HTTP_SUCCESS = 200


class struct_mlx_chat_api:
    '''
    Wrapper for OpenAI chat-style LLM API endpoint, with support for structured responses
    via schema specifiation in query

    Note: we only support chat-style completions

    >>> import asyncio; from toolio.client_helper import struct_mlx_chat_api
    >>> llm = struct_mlx_chat_api(model='gpt-3.5-turbo')
    >>> resp = asyncio.run(llm_api(prompt_to_chat('Knock knock!')))
    >>> resp.first_choice_text

    To-do:

    Implement delivery of function responses to the LLM, a la

    ```py
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  # get a new response from the model where it can see the function response
    ```
    '''
    def __init__(self, base_url=None, default_schema=None, **kwargs):
        '''
        Args:
            base_url (str, optional): Base URL of the API endpoint
                (should be a MLXStructuredLMServer host, or equiv)

            kwargs (dict, optional): Extra parameters for the API or for the model host
        '''
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
        self._pending_tool_calls = {}
        self._registered_tools = {}
        self.tool_role = 'tool'
        self.tool_role = 'assistant'

    async def __call__(self, messages, req='chat/completions', schema=None, tools=None, timeout=30.0, apikey=None,
                         max_trips=10, **kwargs):
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
        if tools:
            req_data['tools'] = tools['tools']
            if 'tool_choice' in tools:
                req_data['tool_choice'] = tools['tool_choice']
            if 'tool_options' in tools:
                req_data['tool_options'] = tools['tool_options']
            for t in tools['tools']:
                f = t['function']
                self.register_tool(f['name'], f['pyfunc'])

            # Enter tool-calling sequence
            llm_call_needed = True
            while max_trips > 0 and llm_call_needed:
                resp = await self.round_trip(req, req_data, timeout, apikey, **kwargs)
                # If LLM has asked for tool calls, prepare to loop back
                if resp['response_type'] == response_type.TOOL_CALL:
                    # self.update_tool_calls(resp)
                    tool_responses = self.execute_tool_calls(resp)
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
                        # TRY REMOVING THE USED TOOL
                        remove_list = [
                            i for (i, t) in enumerate(req_data['tools'])
                            if t.get('function', {}).get('name') == callee_name]
                        # print(f'removing tools with index {remove_list} from request structure')
                        for i in remove_list:
                            req_data['tools'].pop(i)
                        if not req_data['tools']:
                            del req_data['tools']
                else:
                    llm_call_needed = False
                max_trips -= 1

            # Loop exited. We have a final response, or exhausted allowed trips
            if max_trips <= 0:
                # FIXME: i18n
                warnings.warn('Maximum LLM trips exhausted without a final answer', stacklevel=2)
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
        if name in self._registered_tools:
            return self._registered_tools[name]
        else:
            # FIXME: i18n
            raise LookupError(f'Unknown tool: {name}')

    def update_tool_calls(self, response):
        # print('update_tool_calls', response)
        for tc in response['choices'][0].get('message', {}).get('tool_calls'):
            callee_name = tc['function']['name']
            callee_args = tc['function']['arguments_obj']
            tool = self.lookup_tool(callee_name)
            self._pending_tool_calls[tc['id']] = (tool, callee_args)

    def execute_tool_calls(self, response):
        # print('update_tool_calls', response)
        tool_responses = []
        for tc in response['choices'][0].get('message', {}).get('tool_calls'):
            call_id = tc['id']
            callee_name = tc['function']['name']
            callee_args = tc['function']['arguments_obj']
            tool = self.lookup_tool(callee_name)
            # print(f'Calling tool {callee_name} with args {callee_args}')
            try:
                result = tool(**callee_args)
            except TypeError as e:
                # try for special case where the function takes exactly 1 argument
                if len(callee_args) == 1 and 'no keyword arguments' in str(e):
                    result = tool(next(iter(callee_args.values())))
                else:
                    raise
            tool_responses.append((call_id, callee_name, result))
            # print('Tool result:', result)
            # self._pending_tool_calls[tc['id']] = (tool, callee_args)
        return tool_responses

    def register_tool(self, name, funcpath):
        # pyfunc is in the form 'path.to.module_to_import|path.to.function'
        modpath, funcpath = funcpath.split('|')
        modobj = importlib.import_module(modpath)
        parent = modobj
        for funcname in funcpath.split('.'):
            parent = getattr(parent, funcname)
        func = parent
        assert callable(func)
        self._registered_tools[name] = func
