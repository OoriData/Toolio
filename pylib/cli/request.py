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
import importlib

import click
from ogbujipt.llm_wrapper import prompt_to_chat

from toolio.client import struct_mlx_chat_api, response_type


@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000',
    help='MLXStructuredLMServer (OpenAI API-compatible) server base URL')
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
@click.option("--max-tokens", type=int, help='Maximum number of tokens to generate. Will be applied to each trip')

@click.option('--tool', '-t', multiple=True, help='Full Python attribute path to a Toolio-specific callable to be made available to the LLM')

@click.option('--model', type=str, help='Path to locally-hosted MLX format model')
@click.option('--temp', default=0.1, type=float, help='LLM sampling temperature')

@click.option('--trip-timeout', default=90.0, type=float, help='Timeout for each LLM API trip, in seconds')
@click.option('--trace', is_flag=True, default=False,
              help='Print information (to STDERR) about tool call requests & results. Useful for debugging')
def main(apibase, prompt, prompt_file, schema, schema_file, tools, tools_file, tool, sysprompt, max_trips, max_tokens,
         model, temp, trip_timeout, trace):
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

    # Import & register any tools
    tool_callables = []
    for tpath in tool:
        modpath, call_name  = tpath.rsplit('.', 1)
        modobj = importlib.import_module(modpath)
        tool_callables.append(getattr(modobj, call_name))

    llm = struct_mlx_chat_api(base_url=apibase, tools=tool_callables, trace=trace)
    resp = asyncio.run(llm(prompt_to_chat(prompt, system=sysprompt), schema=schema_obj, tools=tools_obj,
                           max_trips=max_trips, trip_timeout=trip_timeout))
    if resp['response_type'] == response_type.TOOL_CALL:
        print('The model invoked the following tool calls to complete the response, but there are no permitted trips remaining.')
        tcs = resp['choices'][0]['message']['tool_calls']
        for tc in tcs:
            del tc['function']['arguments']
        print(json.dumps(tcs, indent=2))
    elif resp['response_type'] == response_type.MESSAGE:
        if trace:
            print('Final response:', file=sys.stderr)
        print(resp.first_choice_text)
