# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_model_manager.py
import json
import logging
from unittest.mock import MagicMock

import pytest

from toolio.llm_helper import model_manager
from toolio.response_helper import llm_response_type

@pytest.mark.asyncio
async def test_model_manager_completion(mock_model):
    '''
    Test the iter_complete method from model_manager by simulating
    a model.completion that yields two GenerationResponse objects
    '''
    # Setup test session with initial message
    test_messages = [{'role': 'user', 'content': 'Test prompt'}]
    expected_response = 'Test response'

    mm = model_manager('test_path')

    # First completion sequence - initial response
    gr_start = MagicMock()
    gr_start.text = expected_response
    gr_start.finish_reason = None
    gr_start.prompt_tokens = 10
    gr_start.generation_tokens = 5

    # Second completion sequence - end signal
    gr_end = MagicMock()
    gr_end.text = ''  # Empty chunk signals the finish
    gr_end.finish_reason = 'stop'
    gr_end.prompt_tokens = 10
    gr_end.generation_tokens = 5

    # Set up the mock to return our streaming sequence
    mock_model.completion.side_effect = [iter([gr_start, gr_end])]

    # Execute the completion flow
    result = []
    async for chunk in mm.iter_complete(test_messages, full_response=True):
        result.append(chunk.to_dict())

    # Verify the response structure
    assert len(result) == 2
    assert result[0]['choices'][0]['delta']['content'] == expected_response
    assert result[1]['choices'][0]['finish_reason'] == 'stop'

    # Verify the mock was called correctly
    assert mock_model.completion.call_count == 1
    call_args = mock_model.completion.call_args_list[0][0]
    assert call_args[0] == test_messages  # Verify messages were passed correctly

    # Repeat scenario to ensure consistent behavior
    mm = model_manager('test_path')
    mock_model.completion.side_effect = [iter([gr_start, gr_end])]

    result = []
    async for chunk in mm.iter_complete(test_messages, full_response=True):
        result.append(chunk.to_dict())

    assert len(result) == 2
    assert result[0]['choices'][0]['delta']['content'] == expected_response
    assert result[1]['choices'][0]['finish_reason'] == 'stop'

@pytest.mark.asyncio
async def test_basic_completion(mock_model, session_cls):
    '''Test basic completion without tools or schema'''
    # Setup test session with initial message
    test_session = session_cls(
        label='Basic completion',
        req_messages=[{'role': 'user', 'content': 'Hello world'}],
        resp_text='This is a test response',
        resp_json=None
    )

    mm = model_manager('test/path')

    # First completion sequence - initial response
    gr_start = MagicMock()
    gr_start.text = test_session.resp_text
    gr_start.finish_reason = None
    gr_start.prompt_tokens = 10
    gr_start.generation_tokens = 5

    # Second completion sequence - end signal
    gr_end = MagicMock()
    gr_end.text = ''  # Empty string signals that generation is done
    gr_end.finish_reason = 'end'
    gr_end.prompt_tokens = 10
    gr_end.generation_tokens = 5

    # Set up the mock to return our streaming sequence
    mock_model.completion.return_value = iter([gr_start, gr_end])

    # Execute the completion flow
    result = await mm.complete(test_session.req_messages)

    # Verify the response
    assert result == test_session.resp_text

    # Verify the mock was called correctly
    assert mock_model.completion.call_count == 1
    call_args = mock_model.completion.call_args_list[0][0]
    assert call_args[0] == test_session.req_messages  # Verify messages were passed correctly

@pytest.mark.asyncio
async def test_completion_with_schema(mock_model, session_cls):
    '''Test completion with JSON schema constraint'''
    # Setup test schema and expected response
    test_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'number'}
        }
    }
    expected_response = '{"name": "John", "age": 30}'

    # Setup test session
    test_session = session_cls(
        label='Schema-constrained completion',
        req_messages=[{'role': 'user', 'content': 'Tell me about a person'}],
        req_schema=test_schema,
        resp_text=expected_response,
        resp_json=None
    )

    mm = model_manager('test/path')

    # First completion sequence - schema-constrained response
    gr_start = MagicMock()
    gr_start.text = expected_response
    gr_start.finish_reason = None
    gr_start.prompt_tokens = 15
    gr_start.generation_tokens = 8

    # Second completion sequence - end signal
    gr_end = MagicMock()
    gr_end.text = ''
    gr_end.finish_reason = 'end'
    gr_end.prompt_tokens = 15
    gr_end.generation_tokens = 8

    # Set up the mock to return our streaming sequence
    mock_model.completion.return_value = iter([gr_start, gr_end])

    # Execute the completion flow with schema
    result = await mm.complete(test_session.req_messages, json_schema=test_schema)

    # Verify the response
    assert json.loads(result) == json.loads(expected_response)

    # Verify the mock was called correctly
    assert mock_model.completion.call_count == 1
    call_args = mock_model.completion.call_args_list[0][0]
    call_kwargs = mock_model.completion.call_args_list[0][1]

    # Verify messages were passed correctly and schema was appended
    assert len(call_args[0]) == 1  # Should only have one message
    assert 'Tell me about a person' in call_args[0][0]['content']  # Original message content
    assert json.dumps(test_schema) in call_args[0][0]['content']  # Schema should be appended
    assert 'cache_prompt' in call_kwargs  # Verify other kwargs are present

@pytest.mark.asyncio
async def test_tool_calling_flow(mock_model, session_cls, sample_tool):
    '''Test the complete tool calling flow'''
    # Setup test session with initial message
    test_session = session_cls(
        label='Tool calling test',
        req_messages=[{'role': 'user', 'content': 'Use the sample tool'}],
        req_tools=[sample_tool],
        resp_text='Tool returned: Sample: test_input',
        resp_json=None,
        intermed_resp_jsons=[]
    )

    mm = model_manager('test/path', tool_reg=test_session.req_tools)

    # First completion sequence - tool call
    # This simulates the model streaming a tool call response
    gr_tool_start = MagicMock()
    gr_tool_start.text = '{"name": "sample", "arguments": {"arg": "test_input"}}'
    gr_tool_start.finish_reason = None
    gr_tool_start.prompt_tokens = 10
    gr_tool_start.generation_tokens = 5

    gr_tool_end = MagicMock()
    gr_tool_end.text = ''
    gr_tool_end.finish_reason = 'end'
    gr_tool_end.prompt_tokens = 10
    gr_tool_end.generation_tokens = 5

    # Second completion sequence - final response
    # This simulates the model streaming the final response after tool execution
    gr_final_start = MagicMock()
    gr_final_start.text = 'Tool returned: Sample: test_input'
    gr_final_start.finish_reason = None
    gr_final_start.prompt_tokens = 15
    gr_final_start.generation_tokens = 8

    gr_final_end = MagicMock()
    gr_final_end.text = ''
    gr_final_end.finish_reason = 'end'
    gr_final_end.prompt_tokens = 15
    gr_final_end.generation_tokens = 8

    # Set up the mock to return our streaming sequences
    mock_model.completion.side_effect = [
        iter([gr_tool_start, gr_tool_end]),  # First completion - tool call
        iter([gr_final_start, gr_final_end])  # Second completion - final response
    ]

    # Execute the tool calling flow
    result = await mm.complete_with_tools(test_session.req_messages, tools=['sample'])

    # Debug prints
    # print('\nDEBUG: Final Response Details:')
    # print(f'Response type: {result.response_type}')
    # print(f'Choices: {result.choices}')
    # print(f'First choice text: {result.first_choice_text}')
    # print(f'Accumulated text: {result.accumulated_text}')
    # print(f'State: {result.state}')
    # print(f'Tool calls: {result.tool_calls}')
    # print(f'Tool results: {result.tool_results}')

    # Verify the final response
    assert result is not None
    assert result.response_type == llm_response_type.MESSAGE
    assert result.first_choice_text == test_session.resp_text

    # Verify the tool call was made correctly
    assert mock_model.completion.call_count == 2  # Should have made two completions
    first_call_args = mock_model.completion.call_args_list[0][0]
    second_call_args = mock_model.completion.call_args_list[1][0]

    # Verify first call had tool schema
    first_call_kwargs = mock_model.completion.call_args_list[0][1]
    assert 'schema' in first_call_kwargs  # The schema is passed in kwargs
    assert first_call_kwargs['schema']['type'] == 'array'  # Verify schema structure
    assert 'items' in first_call_kwargs['schema']  # Verify schema has items

    # Verify second call had tool results in messages
    second_call_messages = second_call_args[0]
    assert len(second_call_messages) > len(test_session.req_messages)  # Should have more messages due to tool results
    assert any('Called tool sample with arguments' in msg.get('content', '') and 'Result: Sample: test_input' in msg.get('content', '') for msg in second_call_messages)

@pytest.mark.asyncio
async def test_model_flags_handling(mock_model, session_cls):
    '''Test handling of different model flags'''
    # Setup test model type and expected response
    mock_model.model.model_type = 'llama'  # Model type that uses NO_SYSTEM_ROLE
    expected_response = 'Response with no system role'

    # Setup test session with system message
    test_session = session_cls(
        label='Model flags test',
        req_messages=[
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Test message'}
        ],
        resp_text=expected_response,
        resp_json=None
    )

    mm = model_manager('test/path')

    # First completion sequence - initial response
    gr_start = MagicMock()
    gr_start.text = expected_response
    gr_start.finish_reason = None
    gr_start.prompt_tokens = 15
    gr_start.generation_tokens = 8

    # Second completion sequence - end signal
    gr_end = MagicMock()
    gr_end.text = ''
    gr_end.finish_reason = 'end'
    gr_end.prompt_tokens = 15
    gr_end.generation_tokens = 8

    # Set up the mock to return our streaming sequence
    mock_model.completion.return_value = iter([gr_start, gr_end])

    # Execute the completion flow
    result = await mm.complete(test_session.req_messages)

    # Verify the response
    assert result == expected_response

    # Verify the mock was called correctly
    assert mock_model.completion.call_count == 1
    call_args = mock_model.completion.call_args_list[0][0]

    # Verify both messages were passed through
    assert len(call_args[0]) == 2  # Both system and user messages should be present
    assert call_args[0][0]['role'] == 'system'  # First message should be system
    assert call_args[0][0]['content'] == 'System prompt'  # Verify system message content
    assert call_args[0][1]['role'] == 'user'  # Second message should be user
    assert call_args[0][1]['content'] == 'Test message'  # Verify user message content

@pytest.mark.asyncio
async def test_max_trips_enforcement(mock_model, session_cls, sample_tool, caplog):
    '''Test that max_trips limit is enforced'''
    # mock_model.completion.side_effect = [iter([
    #     {'op': 'evaluatedPrompt', 'token_count': 10},
    #     {'op': 'generatedTokens', 'text': json.dumps({
    #         'name': 'sample',
    #         'arguments': {'arg': 'test_input'}
    #     })},
    #     {'op': 'stop', 'reason': 'end', 'token_count': 8}
    # ])] * 2  # More responses than max_trips

    test_session = session_cls(
        label='Max trips test',
        req_messages=[{'role': 'user', 'content': 'Use the sample tool repeatedly'}],
        req_tools=[sample_tool],
        max_trips=1,
        resp_type=llm_response_type.TOOL_CALL,
        resp_json=None
    )

    mm = model_manager('test/path', tool_reg=test_session.req_tools)
    gr_max = MagicMock()
    gr_max.text = json.dumps({
        'name': 'sample',
        'arguments': {'arg': 'test_input'}
    })
    gr_max.finish_reason = None

    gr_max_end = MagicMock()
    gr_max_end.text = ''
    gr_max_end.finish_reason = 'end'

    # Emit the same response twice if needed (simulate multiple trips)
    mock_model.completion.side_effect = [iter([gr_max, gr_max_end])] * 2

    with caplog.at_level(logging.DEBUG):
        result = await mm.complete_with_tools(
            test_session.req_messages,
            tools=['sample'],
            max_trips=test_session.max_trips
        )

    # Instead of checking for a warning, we assert the expected debug log message was emitted.
    assert 'Maximum LLM trips exhausted without a final answer' in caplog.text
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == 'sample'
    assert result.tool_calls[0].arguments == {'arg': 'test_input'}
