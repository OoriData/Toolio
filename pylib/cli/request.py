# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.cli.request
'''
Toolio client convenient CLI tool
'''
import sys
import json
import asyncio
import logging

import click
from ogbujipt.llm_wrapper import prompt_to_chat

# from toolio.common import llm_response_type
from toolio.client import struct_mlx_chat_api, cmdline_tools_struct

# List of known loggers with too much chatter at debug level
TAME_LOGGERS = ['asyncio', 'httpcore', 'httpx']
for l in TAME_LOGGERS:
    logging.getLogger(l).setLevel(logging.WARNING)

# Note: above explicit list is a better idea than some sort of blanket approach such as the following:
# for handler in logging.root.handlers:
#     handler.addFilter(logging.Filter('toolio'))

# There is further log config below, in main()

@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000',
    help='toolio_server (OpenAI API-compatible) server base URL')
@click.option('--prompt',
    help='Prompt text; can use {jsonschema} placeholder for the schema')
@click.option('--prompt-file', type=click.File('r'),
    help='Prompt text; can use {jsonschema} placeholder for the schema. Overrides --prompt arg')
@click.option('--schema',
    help='JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
@click.option('--schema-file', type=click.File('rb'),
    help='Path to JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --schema arg')
# @click.option('--schema',
#     help='JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
# @click.option('--schema-file',
#     help='Path to JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --schema arg')
@click.option('--tools',
    help='Tools specification, based on OpenAI format, to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
@click.option('--tools-file', type=click.File('rb'),
    help='Path to tools specification based on OpenAI format, to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --tools arg')
@click.option('--sysprompt', help='Optional system prompt')
@click.option('--max-trips', default=3, type=int,
    help='Maximum number of times to return to the LLM, presumably with tool results. If there is no final response by the time this is reached, post a message with the remaining unused tool invocations')
@click.option("--max-tokens", type=int, default=1024, help='Maximum number of tokens to generate. Will be applied to each trip')

@click.option('--tool', '-t', multiple=True, help='Full Python attribute path to a Toolio-specific callable to be made available to the LLM')

@click.option('--temp', default=0.1, type=float, help='LLM sampling temperature')

@click.option('--trip-timeout', default=90.0, type=float, help='Timeout for each LLM API trip, in seconds')
@click.option('--loglevel', default='INFO', help='Log level, e.g. DEBUG or INFO')
def main(apibase, prompt, prompt_file, schema, schema_file, tools, tools_file, tool, sysprompt, max_trips, max_tokens,
         temp, trip_timeout, loglevel):
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)  # Seems redundant, but is necessary. Python logging is quirky

    if prompt_file:
        prompt = prompt_file.read()
    if not prompt:
        raise RuntimeError('--prompt or --prompt-file required')
    if schema_file:
        schema_obj = json.load(schema_file)
    elif schema:
        schema_obj = json.loads(schema)
    else:
        schema_obj = None
    if tools_file:
        tools_obj = json.load(tools_file)
    elif tools:
        tools_obj = json.loads(tools)
    else:
        tools_obj = None
    if tool:
        tools_obj = list(tool)

    tools_list = cmdline_tools_struct(tools_obj)
    # if tool_choice is None and isinstance(tools_obj, dict):
    tool_choice = tools_obj.get('tool_choice', 'auto') if isinstance(tools_obj, dict) else 'auto'

    llm = struct_mlx_chat_api(base_url=apibase, tool_reg=tools_list, logger=logger)
    resp = asyncio.run(llm(prompt_to_chat(prompt, system=sysprompt),
                                    max_tokens=max_tokens,
                                    temperature=temp,
                                    json_schema=schema_obj,
                                    tools=list(llm._tool_registry.keys()),
                                    tool_choice=tool_choice,
                                    max_trips=max_trips,
                                    trip_timeout=trip_timeout))

    # Restore this logic once we figure out how to communicate incomplete tool-calls
    # if resp.response_type == llm_response_type.TOOL_CALL:
    #     click.echo('The model invoked the following tool calls, but there are no permitted trips remaining.')
    #     for tc in resp.tool_calls:
    #         click.echo(json.dumps(tc.to_dict(), indent=2))
    # elif resp.response_type == llm_response_type.MESSAGE:
    #     click.echo(resp.first_choice_text)
    # else:
    #     click.echo('Unexpected response type:', resp.response_type)

    click.echo(resp)
