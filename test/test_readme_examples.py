# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_conventional_prompt.py

import json

import pytest

from toolio.client import struct_mlx_chat_api, response_type
from toolio.tool import tool, param

CHAT_COMPLETIONS_URL = '/v1/chat/completions'


@pytest.mark.asyncio
# httpserver fixture from pytest_httpserver starts the dummy server
async def test_number_guess(httpserver, session_cls):
    number_guess_ht = session_cls(
        label='Number guess (Hermes theta)',
        req_messages=[{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
        resp_text='*thinks for a moment* how about 6?',
        resp_json={"choices":[
    {"index":0,"message":{"role":"assistant","content":"*thinks for a moment* how about 6?"},"finish_reason":"stop"}
    ],"usage":{"completion_tokens":12,"prompt_tokens":24,"total_tokens":36},"object":"chat.completion",
    "id":"chatcmpl-5932132816_1719810548","created":1719810548,"model":"mlx-community/Hermes-2-Theta-Llama-3-8B-4bit"},
        resp_type=response_type.MESSAGE)

    # Set up dummy server response
    # See also: https://pytest-httpserver.readthedocs.io/en/latest/api.html
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(number_guess_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(number_guess_ht.req_messages)
    assert resp['response_type'] == number_guess_ht.resp_type
    assert resp.first_choice_text == number_guess_ht.resp_text


@pytest.mark.asyncio
async def test_naija_extract(httpserver, session_cls):
    naija_extract_ht = session_cls(
        label='Naija extract (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'}],
        req_schema={'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'continent': {'type': 'string'}}}},
        resp_text='[{"name": "Nigeria", "continent": "Africa"}]',
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '[{"name": "Nigeria", "continent": "Africa"}]'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 15, 'prompt_tokens': 74, 'total_tokens': 89}, 'object': 'chat.completion', 'id': 'chatcmpl-5936244368_1719844279', 'created': 1719844279, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'},
        resp_type=response_type.MESSAGE)

    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(naija_extract_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(naija_extract_ht.req_messages)
    assert resp['response_type'] == naija_extract_ht.resp_type
    assert json.loads(resp.first_choice_text.encode('utf-8')) == json.loads(naija_extract_ht.resp_text.encode('utf-8'))


@pytest.mark.asyncio
async def test_boulder_weather_1(httpserver, session_cls):
    boulder_weather_trip1_ht = session_cls(
        label='Boulder weather - trip1 (Hermes theta)',
        req_messages=[{'role': 'user', 'content': "What's the weather like in Boulder today?\n"}],
        req_tools={'tools': [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'City and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['℃', '℉']}}, 'required': ['location']}}}], 'tool_choice': 'auto'},
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'tool_calls': [{'id': 'call_6107388496_1719881067_0', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': '{"location": "Boulder, MA", "unit": "\\u2109"}'}}]}, 'finish_reason': 'tool_calls'}], 'usage': {'completion_tokens': 36, 'prompt_tokens': 185, 'total_tokens': 221}, 'object': 'chat.completion', 'id': 'chatcmpl-6107388496_1719881067', 'created': 1719881067, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'},
        resp_type=response_type.TOOL_CALL)

    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(boulder_weather_trip1_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(boulder_weather_trip1_ht.req_messages)
    assert resp['response_type'] == boulder_weather_trip1_ht.resp_type


@pytest.mark.asyncio
async def test_square_root(httpserver, session_cls):
    square_root_ht = session_cls(
        label='Square root (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'What is the square root of 256?\n'}],
        req_tools={'tools': [{'type': 'function', 'function': {'name': 'square_root', 'description': 'Get the square root of the given number', 'parameters': {'type': 'object', 'properties': {'square': {'type': 'number', 'description': 'Number from which to find the square root'}}, 'required': ['square']}, 'pyfunc': 'math|sqrt'}}], 'tool_choice': 'auto'},
        # Trip 2 is the final result:
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The square root of 256 is 16.'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10, 'prompt_tokens': 32, 'total_tokens': 42}, 'object': 'chat.completion', 'id': 'chatcmpl-6108365968_1719891004', 'created': 1719891004, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'},
       # Trip 1 retuns:
        intermed_resp_jsons=[{'choices': [{'index': 0, 'message': {'role': 'assistant', 'tool_calls': [{'id': 'call_17495866192_1719891003_0', 'type': 'function', 'function': {'name': 'square_root', 'arguments': '{"square": 256}'}}]}, 'finish_reason': 'tool_calls'}], 'usage': {'completion_tokens': 16, 'prompt_tokens': 157, 'total_tokens': 173}, 'object': 'chat.completion', 'id': 'chatcmpl-17495866192_1719891003', 'created': 1719891003, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'}],
         resp_text='The square root of 256 is 16.',
        resp_type=response_type.MESSAGE)

    # Have to use expect_ordered_request so it will respond to a series of calls with the specified responses
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.intermed_resp_jsons[0])
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(square_root_ht.req_messages, tools=square_root_ht.req_tools)
    assert resp['response_type'] == square_root_ht.resp_type
    assert resp.first_choice_text == square_root_ht.resp_text


@tool('currency_exchange', params=[param('from', str, 'Currency to be converted from, e.g. USD, GBP, JPY', True, rename='from_'), param('to', str, 'Currency to be converted to, e.g. USD, GBP, JPY', True), param('amount', float, 'Amount to convert from one currency to another. Just a number, with no other symbols', True)])
def currency_exchange(from_=None, to=None, amount=None):
    'Tool to convert one currency to another'
    # Just a dummy implementation
    lookup = {('JPY', 'USD'): 1234.56}
    rate = lookup.get((from_, to))
    # print(f'{from_=}, {to=}, {amount=}, {rate=}')
    return rate * amount


@pytest.mark.asyncio
async def test_currency_convert(httpserver, session_cls):
    prompt = 'I need to import a car from Japan. It costs 5 million Yen.'
    'How much must I withdraw from my US bank account'
    currency_convert_ht = session_cls(
        label='Currency convert (Hermes theta)',
        req_messages=[{'role': 'user', 'content': prompt}],
        req_tools=[currency_exchange],
        # Trip 2 is the final result:
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'You need to withdraw $4050.00'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10, 'prompt_tokens': 32, 'total_tokens': 42}, 'object': 'chat.completion', 'id': 'chatcmpl-6108365968_1719891004', 'created': 1719891004, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'},
        # Trip 1 retuns:
        intermed_resp_jsons=[{'choices': [{'index': 0, 'message': {'role': 'assistant', 'tool_calls': [{'id': 'call_17562895312_1721224234_0', 'type': 'function', 'function': {'name': 'currency_exchange', 'arguments': '{"from": "JPY", "to": "USD", "amount": "5000000"}', 'arguments_obj': {'from': 'JPY', 'to': 'USD', 'amount': '5000000'}}}]}, 'finish_reason': 'tool_calls'}], 'usage': {'completion_tokens': 31, 'prompt_tokens': 229, 'total_tokens': 260}, 'object': 'chat.completion', 'id': 'chatcmpl-17562895312_1721224234', 'created': 1721224234, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit', 'response_type': 2, 'prompt_tokens': 229, 'generated_tokens': 31}],
        resp_text='You need to withdraw $4050.00',
        resp_type=response_type.MESSAGE)

    # Have to use expect_ordered_request so it will respond to a series of calls with the specified responses
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(currency_convert_ht.intermed_resp_jsons[0])
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(currency_convert_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'), tools=currency_convert_ht.req_tools)
    resp = await llm(currency_convert_ht.req_messages)
    assert resp['response_type'] == currency_convert_ht.resp_type
    assert resp.first_choice_text == currency_convert_ht.resp_text
