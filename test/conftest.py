# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/conftest.py
'''
Common tools & resources (fixtures) for test cases
'''
# ruff: noqa: E501
# Fixtures HOWTO: https://docs.pytest.org/en/latest/how-to/fixtures.html#how-to-fixtures

from unittest.mock import Mock, patch
from dataclasses import dataclass

import pytest

# from toolio.client import struct_mlx_chat_api, response_type
from toolio.client import response_type

@dataclass
class session:
    'Test inputs & expected outputs'
    label: str
    req_messages: list
    resp_json: dict | None
    resp_type: response_type
    intermed_resp_jsons: list = None
    req_schema: dict = None
    req_tools: dict = None
    resp_text: str = None
    max_trips: int = 3

@pytest.fixture()
def session_cls():
    return session

from toolio.tool.schematics import tool, param

@pytest.fixture
def sample_tool():
    @tool('sample', desc='A sample tool', params=[param('arg', str, 'An argument', True)])
    def sample_function(arg):
        return f"Sample: {arg}"
    return sample_function

@pytest.fixture
def mock_model():
    with patch('toolio.llm_helper.Model') as mock:
        mock_instance = Mock()
        mock_instance.model.model_type = 'test_model'
        mock.return_value = mock_instance
        yield mock_instance
