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
    mm = model_manager('test_path')
    # mock_model.completion.return_value = iter([
    #     {'op': 'evaluatedPrompt', 'token_count': 2},
    #     {'op': 'generatedTokens', 'text': 'Test response'},
    #     {'op': 'stop', 'reason': 'end', 'token_count': 3}
    # ])

    # Mock completion to return GenerationResponse objects
    gr1 = MagicMock()
    gr1.text = 'Test response'
    gr1.finish_reason = None

    gr2 = MagicMock()
    gr2.text = ''  # Empty chunk signals the finish
    gr2.finish_reason = 'stop'

    def fake_completion(*args, **kwargs):
        yield gr1
        yield gr2

    # Assign gen/iterator via side_effect
    mock_model.completion.side_effect = fake_completion

    result = []
    async for chunk in mm.iter_complete(
        [{'role': 'user', 'content': 'Test prompt'}], full_response=True
    ):
        result.append(chunk.to_dict())

    assert len(result) == 2

    assert result[0]['choices'][0]['delta']['content'] == 'Test response'
    assert result[1]['choices'][0]['finish_reason'] == 'stop'

    # Repeat scenario to ensure consistent behavior.
    mm = model_manager('test_path')
    mock_model.completion.side_effect = fake_completion

    result = []
    async for chunk in mm.iter_complete(
        [{'role': 'user', 'content': 'Test prompt'}],
        full_response=True
    ):
        result.append(chunk.to_dict())

    assert len(result) == 2
    assert result[0]['choices'][0]['delta']['content'] == 'Test response'
    assert result[1]['choices'][0]['finish_reason'] == 'stop'

@pytest.mark.asyncio
async def test_basic_completion(mock_model, session_cls):
    '''Test basic completion without tools or schema'''
    # mock_model.completion.return_value = iter([
    #     {'op': 'evaluatedPrompt', 'token_count': 10},
    #     {'op': 'generatedTokens', 'text': 'This is a test response'},
    #     {'op': 'stop', 'reason': 'end', 'token_count': 5}
    # ])

    test_session = session_cls(
        label='Basic completion',
        req_messages=[{'role': 'user', 'content': 'Hello world'}],
        resp_json=None,  # Not needed for local model
        resp_text='This is a test response'
    )

    mm = model_manager('test/path')
    gr1 = MagicMock()
    gr1.text = 'This is a test response'
    gr1.finish_reason = None

    gr2 = MagicMock()
    gr2.text = ''  # An empty string signals that generation is done
    gr2.finish_reason = 'end'

    mock_model.completion.return_value = iter([gr1, gr2])

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

    # mock_model.completion.return_value = iter([
    #     {'op': 'evaluatedPrompt', 'token_count': 15},
    #     {'op': 'generatedTokens', 'text': '{"name": "John", "age": 30}'},
    #     {'op': 'stop', 'reason': 'end', 'token_count': 8}
    # ])

    test_session = session_cls(
        label='Schema-constrained completion',
        req_messages=[{'role': 'user', 'content': 'Tell me about a person'}],
        req_schema=test_schema,
        resp_text='{"name": "John", "age": 30}',
        resp_json=None  # Not needed for local model
    )

    mm = model_manager('test/path')
    gr1 = MagicMock()
    gr1.text = '{"name": "John", "age": 30}'
    gr1.finish_reason = None

    gr2 = MagicMock()
    gr2.text = ''
    gr2.finish_reason = 'end'

    mock_model.completion.return_value = iter([gr1, gr2])

    result = await mm.complete(test_session.req_messages, json_schema=test_schema)
    assert json.loads(result) == json.loads(test_session.resp_text)

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
    mock_model.model.model_type = 'llama'  # Model type that uses NO_SYSTEM_ROLE
    # mock_model.completion.return_value = iter([
    #     {'op': 'evaluatedPrompt', 'token_count': 10},
    #     {'op': 'generatedTokens', 'text': 'Response with no system role'},
    #     {'op': 'stop', 'reason': 'end', 'token_count': 5}
    # ])

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
    gr_flag = MagicMock()
    gr_flag.text = 'Response with no system role'
    gr_flag.finish_reason = None

    gr_flag_end = MagicMock()
    gr_flag_end.text = ''
    gr_flag_end.finish_reason = 'end'

    mock_model.completion.return_value = iter([gr_flag, gr_flag_end])

    # Model type 'llama' should trigger NO_SYSTEM_ROLE behavior
    result = await mm.complete(test_session.req_messages)
    # Verify the system message was handled appropriately
    assert result == test_session.resp_text

    # We might want to add more specific assertions about how the messages
    # were processed based on the model flags

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
