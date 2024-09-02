# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_custom_tools.py
import pytest
from toolio.tool.schematics import tool, param

def test_tool_decorator():
    @tool('test_tool', desc='A test tool', params=[param('arg1', str, 'An argument', True)])
    def test_function(arg1):
        return f"Test: {arg1}"

    assert test_function.name == 'test_tool'
    assert 'arg1' in test_function.schema['parameters']['properties']
    assert test_function.schema['parameters']['required'] == ['arg1']
    assert test_function("hello") == "Test: hello"

# Basically a demo for the sample_tool fixture
def test_with_sample_tool(sample_tool):
    assert sample_tool.name == 'sample'
    assert sample_tool('test') == 'Sample: test'

# TODO: more tests for edge cases, type checking, etc.
