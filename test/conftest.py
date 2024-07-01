# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/conftest.py
'''
Common tools & resources (fixtures) for test cases
'''
# ruff: noqa: E501

from dataclasses import dataclass

import pytest

# from toolio.client import struct_mlx_chat_api, response_type
from toolio.client import response_type


@dataclass
class session:
    'Test inputs & expected outputs'
    label: str
    req_messages: list
    resp_json: dict
    resp_type: response_type
    req_schema: dict = None
    resp_text: str = None
    max_trips: int = 3


# Hermes Theta sessions

@pytest.fixture()
def number_guess_ht():
    return session(
        label='Number guess (Hermes theta)',
        req_messages=[{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
        resp_text='*thinks for a moment* how about 6?',
        resp_json={"choices":[
    {"index":0,"message":{"role":"assistant","content":"*thinks for a moment* how about 6?"},"finish_reason":"stop"}
    ],"usage":{"completion_tokens":12,"prompt_tokens":24,"total_tokens":36},"object":"chat.completion",
    "id":"chatcmpl-5932132816_1719810548","created":1719810548,"model":"mlx-community/Hermes-2-Theta-Llama-3-8B-4bit"},
        resp_type=response_type.MESSAGE)


@pytest.fixture()
def naija_extract_ht():
    return session(
        label='Naija extract (Hermes theta)',
        req_messages=[{'role': 'user', 'content': 'Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'}],
        req_schema={'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'continent': {'type': 'string'}}}},
        resp_text='[{"name": "Nigeria", "continent": "Africa"}]',
        resp_json={'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '[{"name": "Nigeria", "continent": "Africa"}]'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 15, 'prompt_tokens': 74, 'total_tokens': 89}, 'object': 'chat.completion', 'id': 'chatcmpl-5936244368_1719844279', 'created': 1719844279, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'},
        resp_type=response_type.MESSAGE)


    # response_type.MESSAGE
    # response_type.TOOL_CALL


# @pytest.fixture()
# def usain_bolt_hermes_theta():


# @pytest.fixture()
# def mocked_usain_bolt_hermes_theta(mocker):  # mocker fixture comes from pytest-mock
#     retval = '…'
#     mocker.patch('struct_mlx_chat_api.round_trip', return_value=retval)
