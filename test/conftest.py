# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/conftest.py
'''
Common tools & resources (fixtures) for test cases
'''
# ruff: noqa: E501
# Fixtures HOWTO: https://docs.pytest.org/en/latest/how-to/fixtures.html#how-to-fixtures

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
    intermed_resp_jsons: list = None
    req_schema: dict = None
    req_tools: dict = None
    resp_text: str = None
    max_trips: int = 3


@pytest.fixture()
def session_cls():
    return session

