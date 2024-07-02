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
    # See also: https://pytest-httpserver.readthedocs.io/en/latest/api.html
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(number_guess_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(number_guess_ht.req_messages)
    assert resp['response_type'] == number_guess_ht.resp_type
    assert resp.first_choice_text == number_guess_ht.resp_text


@pytest.mark.asyncio
async def test_naija_extract(naija_extract_ht, httpserver):
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(naija_extract_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(naija_extract_ht.req_messages)
    assert resp['response_type'] == naija_extract_ht.resp_type
    assert json.loads(resp.first_choice_text.encode('utf-8')) == json.loads(naija_extract_ht.resp_text.encode('utf-8'))


@pytest.mark.asyncio
async def test_boulder_weather_1(boulder_weather_trip1_ht, httpserver):
    httpserver.expect_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(boulder_weather_trip1_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(boulder_weather_trip1_ht.req_messages)
    assert resp['response_type'] == boulder_weather_trip1_ht.resp_type


@pytest.mark.asyncio
async def test_square_root(square_root_ht, httpserver):
    # Have to use expect_ordered_request so it will respond to a series of calls with the specified responses
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.intermed_resp_jsons[0])
    httpserver.expect_ordered_request(CHAT_COMPLETIONS_URL, method='POST').respond_with_json(square_root_ht.resp_json)

    llm = struct_mlx_chat_api(base_url=httpserver.url_for('/v1'))
    resp = await llm(square_root_ht.req_messages, tools=square_root_ht.req_tools)
    assert resp['response_type'] == square_root_ht.resp_type
    assert resp.first_choice_text == square_root_ht.resp_text

