# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_conventional_prompt.py

import json

import pytest

from toolio.client import struct_mlx_chat_api, response_type

CHAT_COMPLETIONS_URL = '/v1/chat/completions'


@pytest.mark.asyncio
# httpserver fixture from pytest_httpserver starts the dummy server
async def test_number_guess(number_guess_ht, httpserver):
    # Set up dummy server response
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(number_guess_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(number_guess_ht.req_messages)
    assert resp['response_type'] == number_guess_ht.resp_type
    assert resp.first_choice_text == number_guess_ht.resp_text


@pytest.mark.asyncio
# httpserver fixture from pytest_httpserver starts the dummy server
async def test_naija_extract(naija_extract_ht, httpserver):
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(naija_extract_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(naija_extract_ht.req_messages)
    assert resp['response_type'] == naija_extract_ht.resp_type
    assert json.loads(resp.first_choice_text.encode('utf-8')) == json.loads(naija_extract_ht.resp_text.encode('utf-8'))


