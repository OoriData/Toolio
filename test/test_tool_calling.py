# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_tool_calling.py

import pytest

from toolio.common import llm_response_type
from toolio.response_helper import llm_response

@pytest.mark.asyncio
async def test_tool_calls_conversion(session_cls):
    # Mock response with tool calls
    mock_response = {
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'tool_calls': [{
                    'id': 'call_123',
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
        'id': 'chatcmpl-123',
        'created': 1719891003,
        'model': 'test-model'
    }

    # Convert to llm_response
    resp = llm_response.from_openai_chat(mock_response)

    # Verify conversion
    assert resp.response_type == llm_response_type.TOOL_CALL
    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == 'call_123'
    assert resp.tool_calls[0].name == 'square_root'
    assert resp.tool_calls[0].arguments == {'square': 256}
