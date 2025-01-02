# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_model_manager.py
import pytest
# from unittest.mock import Mock, patch
from toolio.llm_helper import model_manager

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

# TODO: Add more tests for different scenarios, error handling, etc.
