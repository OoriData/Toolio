# mlx_struct_lm_server.client_helper
'''
Tools to help with HTTP query of LLMs for structured response, as hosted by MLXStructuredLMServer

Modeled on ogbujipt.llm_wrapper.openai_api & ogbujipt.llm_wrapper.openai_chat_api

'''
# import sys
import json
import warnings
from enum import Enum
from dataclasses import dataclass

import httpx
# import asyncio

from amara3 import iri

from ogbujipt import config
from ogbujipt.llm_wrapper import llm_response

HTTP_SUCCESS = 200


class response_type(Enum):
    MESSAGE = 1
    TOOL_CALL = 2


@dataclass
class response_obj:
    toolcall: str


class struct_mlx_api:
    '''
    Wrapper for OpenAI-style LLM API endpoint, with support for structured responses
    via schema specifiation in query

    >>> import asyncio; from mlx_struct_lm_server.client_helper import struct_mlx_api
    >>> llm = openai_api(base_url='http://localhost:8000')
    >>> resp = asyncio.run(llm_api('Knock knock!', max_tokens=128))
    >>> resp.first_choice_text
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
            warnings.warn('base_url not provided, so each invocation will require one', stacklevel=2)

    async def __call__(self, prompt, req='completions', timeout=30.0, apikey=None, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            prompt (str): Prompt to send to the LLM

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                best_of, echo, frequency_penalty, logit_bias, logprobs, max_tokens, n
                presence_penalty, seed, stop, stream, suffix, temperature, top_p, userq
        Returns:
            dict: JSON response from the LLM
        '''
        header = {'Content-Type': 'application/json'}
        # if apikey is None:
        #     apikey = self.apikey
        # if apikey:
        #     header['Authorization'] = f'Bearer {apikey}'
        async with httpx.AsyncClient() as client:
            req_data = {'prompt': prompt, **kwargs}
            result = await client.post(
                f'{self.base_url}/{req}', json=req_data, headers=header, timeout=timeout)
            if result.status_code == HTTP_SUCCESS:
                return llm_response.from_openai_chat(result.json())
            else:
                raise RuntimeError(f'Unexpected response from {self.base_url}{req}:\n{repr(result)}')


class struct_mlx_chat_api(struct_mlx_api):
    '''
    Wrapper for OpenAI chat-style LLM API endpoint, with support for structured responses
    via schema specifiation in query

    >>> import asyncio; from mlx_struct_lm_server.client_helper import struct_mlx_chat_api
    >>> llm = struct_mlx_chat_api(model='gpt-3.5-turbo')
    >>> resp = asyncio.run(llm_api(prompt_to_chat('Knock knock!')))
    >>> resp.first_choice_text
    '''
    async def __call__(self, messages, req='chat/completions', schema=None, tools=None, timeout=30.0, apikey=None, **kwargs):
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
        header = {'Content-Type': 'application/json'}
        # if apikey is None:
        #     apikey = self.apikey
        # if apikey:
        #     header['Authorization'] = f'Bearer {apikey}'
        async with httpx.AsyncClient() as client:
            # Replace {json_schema} references with the schema
            for m in messages:
                # Don't use actual string formatting for now, because user might have other uses of curly braces
                # Yes, this introduces the problem of escaping without depth, so much pondering required
                # Perhaps make the replacement string configurable
                m['content'] = m['content'].replace('{json_schema}', schema_str)
            req_data = {'messages': messages, **kwargs}
            if schema:
                req_data['response_format'] = {'type': 'json_object', 'schema': schema_str}
            if tools:
                req_data['tools'] = tools['tools']
                if 'tool_choice' in tools:
                    req_data['tool_choice'] = tools['tool_choice']
                if 'tool_options' in tools:
                    req_data['tool_options'] = tools['tool_options']
            result = await client.post(
                f'{self.base_url}/{req}', json=req_data, headers=header, timeout=timeout)
            if result.status_code == HTTP_SUCCESS:
                res_json = result.json()
                resp_msg = res_json['choices'][0]['message']
                assert resp_msg['role'] == 'assistant'
                # There will be no response message content if a tool call is invoked
                if 'tool_calls' in resp_msg:
                    # Why the hell does OpenAI have these arguments properties as plain text? Seems like a massive layering violation
                    for tc in resp_msg['tool_calls']:
                        tc['function']['arguments_obj'] = json.loads(tc['function']['arguments'])
                    resp = response_obj(toolcall=json.dumps(resp_msg['tool_calls'], indent=2))
                    response_t = response_type.TOOL_CALL
                else:
                    response_t = response_type.MESSAGE
                    resp = llm_response.from_openai_chat(res_json)
                return response_t, resp
            else:
                raise RuntimeError(f'Unexpected response from {self.base_url}{req}:\n{repr(result)}')
