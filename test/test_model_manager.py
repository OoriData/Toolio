# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_model_manager.py
import json
from unittest.mock import Mock, patch

import pytest

from ogbujipt.llm_wrapper import response_type
from toolio.llm_helper import model_manager
# from toolio.common import model_flag


@pytest.mark.asyncio
async def test_model_manager_completion(mock_model):
    mm = model_manager('test_path')
    mock_model.completion.return_value = iter([
        {'op': 'evaluatedPrompt', 'token_count': 2},
        {'op': 'generatedTokens', 'text': 'Test response'},
        {'op': 'stop', 'reason': 'end', 'token_count': 3}
    ])

    result = []
    async for chunk in mm.iter_complete([{'role': 'user', 'content': 'Test prompt'}]):
        # FIXME: First chunk is empty. Needs more attention.
        result.append(chunk)
    assert len(result) == 2
    assert result[0]['choices'][0]['delta']['content'] == 'Test response'
    assert result[1]['choices'][0]['delta']['content'] == ''
    assert result[1]['choices'][0]['finish_reason'] == 'stop'
    # assert result[1]['reason'] == 'end'


@pytest.mark.asyncio
async def test_basic_completion(mock_model, session_cls):
    '''Test basic completion without tools or schema'''
    mock_model.completion.return_value = iter([
        {'op': 'evaluatedPrompt', 'token_count': 10},
        {'op': 'generatedTokens', 'text': 'This is a test response'},
        {'op': 'stop', 'reason': 'end', 'token_count': 5}
    ])

    test_session = session_cls(
        label='Basic completion',
        req_messages=[{'role': 'user', 'content': 'Hello world'}],
        resp_json=None,  # Not needed for local model
        resp_text='This is a test response'
    )

    mm = model_manager('test/path')
    result = await mm.complete(test_session.req_messages)
    assert result == test_session.resp_text


@pytest.mark.asyncio
async def test_completion_with_schema(mock_model, session_cls):
    '''Test completion with JSON schema constraint'''
    test_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'number'}
        }
    }

    mock_model.completion.return_value = iter([
        {'op': 'evaluatedPrompt', 'token_count': 15},
        {'op': 'generatedTokens', 'text': '{"name": "John", "age": 30}'},
        {'op': 'stop', 'reason': 'end', 'token_count': 8}
    ])

    test_session = session_cls(
        label='Schema-constrained completion',
        req_messages=[{'role': 'user', 'content': 'Tell me about a person'}],
        req_schema=test_schema,
        resp_text='{"name": "John", "age": 30}',
        resp_json=None  # Not needed for local model
    )

    mm = model_manager('test/path')
    result = await mm.complete(test_session.req_messages, json_schema=test_schema)
    assert json.loads(result) == json.loads(test_session.resp_text)


@pytest.mark.asyncio
async def test_tool_calling_flow(mock_model, session_cls, sample_tool):
    '''Test the complete tool calling flow'''
    # Mock a sequence of responses for tool calling
    mock_model.completion.side_effect = [
        # First completion - tool call
        iter([
            {'op': 'evaluatedPrompt', 'token_count': 10},
            {'op': 'generatedTokens', 'text': json.dumps({
                'name': 'sample',
                'arguments': {'arg': 'test_input'}
            })},
            {'op': 'stop', 'reason': 'end', 'token_count': 8}
        ]),
        # Second completion - final response
        iter([
            {'op': 'evaluatedPrompt', 'token_count': 12},
            {'op': 'generatedTokens', 'text': 'Tool returned: Sample: test_input'},
            {'op': 'stop', 'reason': 'end', 'token_count': 6}
        ])
    ]

    test_session = session_cls(
        label='Tool calling test',
        req_messages=[{'role': 'user', 'content': 'Use the sample tool'}],
        req_tools=[sample_tool],
        resp_text='Tool returned: Sample: test_input',
        resp_json=None,  # Not needed for local model
        intermed_resp_jsons=[]  # Not needed for local model
    )

    mm = model_manager('test/path', tool_reg=test_session.req_tools)
    result = await mm.complete_with_tools(
        test_session.req_messages,
        tools=['sample']
    )
    assert result == test_session.resp_text


@pytest.mark.asyncio
async def test_model_flags_handling(mock_model, session_cls):
    '''Test handling of different model flags'''
    mock_model.model.model_type = 'llama'  # Model type that uses NO_SYSTEM_ROLE
    mock_model.completion.return_value = iter([
        {'op': 'evaluatedPrompt', 'token_count': 10},
        {'op': 'generatedTokens', 'text': 'Response with no system role'},
        {'op': 'stop', 'reason': 'end', 'token_count': 5}
    ])

    test_session = session_cls(
        label='Model flags test',
        req_messages=[
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Test message'}
        ],
        resp_text='Response with no system role',
        resp_json=None
    )

    mm = model_manager('test/path')
    # Model type 'llama' should trigger NO_SYSTEM_ROLE behavior
    result = await mm.complete(test_session.req_messages)
    assert result == test_session.resp_text

    # Verify the system message was handled appropriately
    mock_model.completion.assert_called_once()
    # You might want to add more specific assertions about how the messages
    # were processed based on the model flags


@pytest.mark.asyncio
async def test_max_trips_enforcement(mock_model, session_cls, sample_tool):
    '''Test that max_trips limit is enforced'''
    mock_model.completion.side_effect = [iter([
        {'op': 'evaluatedPrompt', 'token_count': 10},
        {'op': 'generatedTokens', 'text': json.dumps({
            'name': 'sample',
            'arguments': {'arg': 'test_input'}
        })},
        {'op': 'stop', 'reason': 'end', 'token_count': 8}
    ])] * 2  # More responses than max_trips

    test_session = session_cls(
        label='Max trips test',
        req_messages=[{'role': 'user', 'content': 'Use the sample tool repeatedly'}],
        req_tools=[sample_tool],
        max_trips=1,
        resp_type=response_type.TOOL_CALL,
        resp_json=None
    )

    mm = model_manager('test/path', tool_reg=test_session.req_tools)
    with pytest.warns(UserWarning, match='Maximum LLM trips exhausted without a final answer'):
        result = await mm.complete_with_tools(
            test_session.req_messages,
            tools=['sample'],
            max_trips=test_session.max_trips
        )

    # Verify number of completions matches max_trips
    assert mock_model.completion.call_count <= test_session.max_trips
    assert result == test_session.resp_text
