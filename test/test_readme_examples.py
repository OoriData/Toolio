# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_readme_examples.py

import json
import warnings
from unittest.mock import patch

import pytest

from toolio.client import struct_mlx_chat_api, cmdline_tools_struct
from toolio.common import llm_response_type
from toolio.response_helper import llm_response
from toolio.tool import tool, param

CHAT_COMPLETIONS_URL = '/v1/chat/completions'

@pytest.mark.asyncio
# httpserver fixture from pytest-httpserver starts the dummy server
async def test_number_guess(httpserver, session_cls):
    number_guess_ht = session_cls(
        label='Number guess (Hermes theta)',
        req_messages=[{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
        resp_text='*thinks for a moment* how about 6?',
        resp_json={"choices":[
    {"index":0,"message":{"role":"assistant","content":"*thinks for a moment* how about 6?"},"finish_reason":"stop"}
    ],"usage":{"completion_tokens":12,"prompt_tokens":24,"total_tokens":36},"object":"chat.completion",
    "id":"chatcmpl-5932132816_1719810548","created":1719810548,"model":"mlx-community/Hermes-2-Theta-Llama-3-8B-4bit"},
        resp_type=llm_response_type.MESSAGE)

    # Set up dummy server response
    # See also: https://pytest-httpserver.readthedocs.io/en/latest/api.html
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(number_guess_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm.complete(number_guess_ht.req_messages)
    assert resp == number_guess_ht.resp_text

@pytest.mark.asyncio
async def test_naija_extract(httpserver, session_cls):
    naija_extract_ht = session_cls(
        label='Naija extract (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#'}],
        req_schema={'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'continent': {'type': 'string'}}, "required": ["name", "continent"]}},
        resp_text='[{"name": "Nigeria", "continent": "Africa"}]',
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '[{"name": "Nigeria", "continent": "Africa"}]'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 15, 'prompt_tokens': 74, 'total_tokens': 89}, 'object': 'chat.completion', 'id': 'chatcmpl-5936244368_1719844279', 'created': 1719844279, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit', 'toolio.model_type': 'llama'},
        resp_type=llm_response_type.MESSAGE)

    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(naija_extract_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm.complete(naija_extract_ht.req_messages)
    # assert json.loads(resp.first_choice_text.encode('utf-8')) == json.loads(naija_extract_ht.resp_text.encode('utf-8'))
    assert json.loads(resp) == json.loads(naija_extract_ht.resp_text)

@pytest.mark.asyncio
async def test_boulder_weather_1(httpserver, session_cls):
    @tool('get_current_weather', desc='Get the current weather in a given location',
          params=[
              param('location', str, 'City and state, e.g. San Francisco, CA', True),
              param('unit', str, 'Temperature unit', False)  # , enum=['℃', '℉'] # Cool idea for tools spec
          ])
    def get_current_weather(location, unit=None):
        return {"temperature": 72, "unit": unit or "℉"}

    weather_schema = {
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'City and state, e.g. San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['℃', '℉']
                }
            },
            'required': ['location']
        }
    }

    boulder_weather_trip1_ht = session_cls(
        label='Boulder weather - trip1 (Hermes theta)',
        req_messages=[{'role': 'user', 'content': "What's the weather like in Boulder today?\n"}],
        req_tools={'get_current_weather': (get_current_weather, weather_schema)},
        resp_json={
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'The current temperature in Boulder is 72℉.'
                },
                'finish_reason': 'stop'
            }],
            'usage': {'completion_tokens': 12, 'prompt_tokens': 185, 'total_tokens': 197},
            'object': 'chat.completion',
            'id': 'chatcmpl-test_id',
            'created': 1719881067,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'
        },
        intermed_resp_jsons=[{
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'id': 'call_test_id',
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'arguments': '{"location": "Boulder, MA", "unit": "\\u2109"}'
                        }
                    }]
                },
                'finish_reason': 'tool_calls'
            }],
            'usage': {'completion_tokens': 36, 'prompt_tokens': 185, 'total_tokens': 221},
            'object': 'chat.completion',
            'id': 'chatcmpl-intermed_id',
            'created': 1719881067,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit',
            'toolio.model_type': 'llama'
        }],
        resp_text='The current temperature in Boulder is 72℉.',
        resp_type=llm_response_type.MESSAGE
    )

    # Request handlers to record the requests
    first_request = httpserver.expect_ordered_request(
        CHAT_COMPLETIONS_URL,
        method='POST'
    ).respond_with_json(boulder_weather_trip1_ht.intermed_resp_jsons[0])

    second_request = httpserver.expect_ordered_request(
        CHAT_COMPLETIONS_URL,
        method='POST'
    ).respond_with_json(boulder_weather_trip1_ht.resp_json)
    request3 = httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json({
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': 'The current temperature in Boulder is 72℉.'
            }
        }]
    })

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    llm.register_tool(get_current_weather, weather_schema)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
            message='No implementation provided for function: UNNAMED',
            category=UserWarning)
        llm.register_internal_tools()

    try:
        resp = await llm.complete_with_tools(
            boulder_weather_trip1_ht.req_messages,
            tools=['get_current_weather'],
            full_response=False
        )
    except RuntimeError as e:
        print("Trip 1. Server received:", httpserver.log[0][0].get_data())
        print("Trip 1. Server response:", httpserver.log[0][1].get_data())
        print("Trip 2. Server received:", httpserver.log[1][0].get_data())
        print("Trip 2. Server response:", httpserver.log[2][1].get_data())
        print(f"Full error details: {e}")
        pytest.fail("Server returned 500 error")

    # Verify the requests after the calls
    assert len(httpserver.log) >= 2  # We should have at least 2 requests
    assert httpserver.log[0][0].method == 'POST'  # First request
    assert httpserver.log[0][0].url.endswith(CHAT_COMPLETIONS_URL)
    assert httpserver.log[1][0].method == 'POST'  # Second request
    assert httpserver.log[1][0].url.endswith(CHAT_COMPLETIONS_URL)

    resp = llm_response.from_openai_chat(resp)
    assert resp.first_choice_text == boulder_weather_trip1_ht.resp_text

@pytest.mark.asyncio
async def test_square_root(httpserver, session_cls):
    httpserver.verbose = True
    # Define the tool using the decorator
    @tool('square_root', desc='Get the square root of the given number',
          params=[param('square', float, 'Number from which to find the square root', True)])
    def square_root(square):
        import math
        return math.sqrt(square)

    # The schema should match what the decorator produces
    square_root_schema = {
        'name': 'square_root',
        'description': 'Calculate square root',
        'parameters': {
            'type': 'object',
            'properties': {
                'square': {
                    'type': 'number',
                    'description': 'Number from which to find the square root'
                }
            },
            'required': ['square']
        }
    }

    square_root_ht = session_cls(
        label='Square root (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'What is the square root of 256?\n'}],
        req_tools={'square_root': (square_root, square_root_schema)},  # Keep as dict format
        resp_json={
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'The square root of 256 is 16.'
                },
                'finish_reason': 'stop'
            }],
            'usage': {'completion_tokens': 10, 'prompt_tokens': 32, 'total_tokens': 42},
            'object': 'chat.completion',
            'id': 'chatcmpl-6108365968_1719891004',
            'created': 1719891004,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'
        },
        intermed_resp_jsons=[{
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'id': 'call_17495866192_1719891003_0',
                        'type': 'function',
                        'function': {
                            'name': 'square_root',
                            'arguments': '{"square": 256}'
                        }
                    }]
                },
                'finish_reason': 'tool_calls'
            }],
            'usage': {'completion_tokens': 16, 'prompt_tokens': 157, 'total_tokens': 173},
            'object': 'chat.completion',
            'id': 'chatcmpl-17495866192_1719891003',
            'created': 1719891003,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit',
            'toolio.model_type': 'llama'
        }],
        resp_text='The square root of 256 is 16.',
        resp_type=llm_response_type.MESSAGE
    )

    # Setup mock server responses
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.intermed_resp_jsons[0])
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.resp_json)
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json({
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': 'The square root of 256 is 16.'
            }
        }]
    })

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    llm.register_tool(square_root, square_root_schema)  # Register the tool directly

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
            message='No implementation provided for function: UNNAMED',
            category=UserWarning)
        llm.register_internal_tools()

    with patch('math.sqrt', return_value=16.0) as mock_sqrt:
        try:
            resp = await llm.complete_with_tools(
                square_root_ht.req_messages,
                tools=['square_root'],  # Pass just the tool name since it's already registered
                full_response=False
            )
        except RuntimeError as e:
            print("Trip 1. Server received:", httpserver.log[0][0].get_data())
            print("Trip 1. Server response:", httpserver.log[0][1].get_data())
            print("Trip 2. Server received:", httpserver.log[1][0].get_data())
            print("Trip 2. Server response:", httpserver.log[2][1].get_data())
            print(f"Full error details: {e}")
            pytest.fail("Server returned 500 error")
        mock_sqrt.assert_called_once_with(256.0)
        resp = llm_response.from_openai_chat(resp)
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
    @tool('currency_exchange', params=[
        param('from', str, 'Currency to be converted from, e.g. USD, GBP, JPY', True, rename='from_'),
        param('to', str, 'Currency to be converted to, e.g. USD, GBP, JPY', True),
        param('amount', float, 'Amount to convert from one currency to another. Just a number, with no other symbols', True)
    ])
    def currency_exchange(from_=None, to=None, amount=None):
        'Tool to convert one currency to another'
        lookup = {('JPY', 'USD'): 0.00081}  # Fixed rate for test
        rate = lookup.get((from_, to))
        return rate * amount

    currency_schema = {
        'name': 'currency_exchange',
        'description': 'Tool to convert one currency to another',
        'parameters': {
            'type': 'object',
            'properties': {
                'from': {
                    'type': 'string',
                    'description': 'Currency to be converted from, e.g. USD, GBP, JPY'
                },
                'to': {
                    'type': 'string',
                    'description': 'Currency to be converted to, e.g. USD, GBP, JPY'
                },
                'amount': {
                    'type': 'number',
                    'description': 'Amount to convert from one currency to another. Just a number, with no other symbols'
                }
            },
            'required': ['from', 'to', 'amount']
        }
    }

    currency_convert_ht = session_cls(
        label='Currency convert (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'I need to import a car from Japan. It costs 5 million Yen. How much must I withdraw from my US bank account?'}],
        req_tools={'currency_exchange': (currency_exchange, currency_schema)},
        resp_json={
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'You need to withdraw $4,050.00'
                },
                'finish_reason': 'stop'
            }],
            'usage': {'completion_tokens': 10, 'prompt_tokens': 32, 'total_tokens': 42},
            'object': 'chat.completion',
            'id': 'chatcmpl-test_id',
            'created': 1719891004,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'
        },
        intermed_resp_jsons=[{
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'id': 'call_test_id',
                        'type': 'function',
                        'function': {
                            'name': 'currency_exchange',
                            'arguments': '{"from": "JPY", "to": "USD", "amount": 5000000}'
                        }
                    }]
                },
                'finish_reason': 'tool_calls'
            }],
            'usage': {'completion_tokens': 31, 'prompt_tokens': 229, 'total_tokens': 260},
            'object': 'chat.completion',
            'id': 'chatcmpl-intermed_id',
            'created': 1721224234,
            'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit',
            'toolio.model_type': 'llama'
        }],
        resp_text='You need to withdraw $4,050.00',
        resp_type=llm_response_type.MESSAGE
    )

    # Set up request handlers that will record the requests
    first_request = httpserver.expect_ordered_request(
        CHAT_COMPLETIONS_URL,
        method='POST'
    ).respond_with_json(currency_convert_ht.intermed_resp_jsons[0])

    second_request = httpserver.expect_ordered_request(
        CHAT_COMPLETIONS_URL,
        method='POST'
    ).respond_with_json(currency_convert_ht.resp_json)
    request3 = httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json({
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': 'You need to withdraw $4,050.00'
            }
        }]
    })

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    llm.register_tool(currency_exchange, currency_schema)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
            message='No implementation provided for function: UNNAMED',
            category=UserWarning)
        llm.register_internal_tools()

    try:
        resp = await llm.complete_with_tools(
            currency_convert_ht.req_messages,
            tools=['currency_exchange'],
            full_response=False
        )
    except RuntimeError as e:
        print("Trip 1. Server received:", httpserver.log[0][0].get_data())
        print("Trip 1. Server response:", httpserver.log[0][1].get_data())
        print("Trip 2. Server received:", httpserver.log[1][0].get_data())
        print("Trip 2. Server response:", httpserver.log[2][1].get_data())
        print(f"Full error details: {e}")
        pytest.fail("Server returned 500 error")

    # Verify the requests after the calls
    assert len(httpserver.log) >= 2  # We should have at least 2 requests
    assert httpserver.log[0][0].method == 'POST'  # First request
    assert httpserver.log[0][0].url.endswith(CHAT_COMPLETIONS_URL)
    assert httpserver.log[1][0].method == 'POST'  # Second request
    assert httpserver.log[1][0].url.endswith(CHAT_COMPLETIONS_URL)

    resp = llm_response.from_openai_chat(resp)
    assert resp.first_choice_text == currency_convert_ht.resp_text
