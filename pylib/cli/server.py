# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.cli.server
'''
LLM server with OpenAI-like API, for structured prompting support & including function/tool calling

Based on: https://github.com/otriscon/llm-structured-output/blob/main/src/examples/server.py

TODO: Break out CLI stuff (new module for FastAPI bits)

toolio_server --model=mlx-community/Hermes-2-Theta-Llama-3-8B-4bit

Note: you can also point `--model` at a downloaded or converted MLX model on local storage.
'''

import json
import time
import os
from enum import Enum
from typing import Literal, List, Optional, Union
from contextlib import asynccontextmanager
import warnings

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import click
import uvicorn

from llm_structured_output.util.output import info, warning, debug

from toolio.schema_helper import Model
from toolio.llm_helper import model_flag, DEFAULT_FLAGS, FLAGS_LOOKUP
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


app_params = {}

# Context manager for the FastAPI app's lifespan: https://fastapi.tiangolo.com/advanced/events/
# Don't just stick this in globals, because we're planning for better async support
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Load model one-time
    '''
    # Startup code here. Particularly persistence, so that its DB connection pool is set up in the right event loop
    info(f'Loading model ({app_params["model"]})…')
    tstart = time.perf_counter_ns()
    app.state.model = Model()
    app.state.params = app_params
    # Can use click's env support if we decide we want this
    # model_path = os.environ['MODEL_PATH']
    app.state.model.load(app_params['model'])
    tdone = time.perf_counter_ns()
    # XXX: alternate ID option is app.state.model.model.model_type which is a string, e.g. 'gemma2'
    # print(app.state.model.model.__class__, app.state.model.model.model_type)
    info(f'Model loaded in {(tdone - tstart)/1000000000.0:.3f}s. Type: {app.state.model.model.model_type}')
    app.state.model_flags = FLAGS_LOOKUP.get(app.state.model.model.__class__, DEFAULT_FLAGS)
    yield
    # Shutdown code here, if any


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
# pylint: disable-next=unused-argument
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'
    warning(f'RequestValidationError: {exc_str}. Request: {await request.json()!r}')
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.get('/status')
def get_status():
    return {'status': 'OK'}


@app.get('/')
def get_root():
    return FileResponse(f'{os.path.dirname(os.path.realpath(__file__))}/static/ui.html')


class V1ChatMessageRole(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'


class V1ChatMessage(BaseModel):
    role: V1ChatMessageRole
    content: str


class V1Function(BaseModel):
    name: str
    description: str = ''
    parameters: dict = {}


class V1ToolFunction(BaseModel):
    type: Literal['function']
    function: V1Function


class V1ToolChoiceKeyword(str, Enum):
    AUTO = 'auto'
    NONE = 'none'


class V1ToolChoiceFunction(BaseModel):
    type: Optional[Literal['function']] = None
    name: str


class V1ToolOptions(BaseModel):  # Non-standard addition
    # We automatically add instructions with the JSON schema
    # for the tool calls to the prompt. This option disables
    # it and is useful when the user prompt already includes
    # the schema and relevant instructions.
    no_prompt_steering: bool = False


class V1ResponseFormatType(str, Enum):
    JSON_OBJECT = 'json_object'


class V1ResponseFormat(BaseModel):
    type: V1ResponseFormatType
    # schema is our addition, not an OpenAI API parameter
    # Avoid shadowing BaseModel.schema
    json_schema: Optional[str] = Field(default=None, alias='schema')


class V1StreamOptions(BaseModel):
    include_usage: bool = False


class V1ChatCompletionsRequest(BaseModel):
    # pylint: disable=too-many-instance-attributes # Paternalistic excess
    model: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.0
    messages: List[V1ChatMessage]
    # FIXME: We don't need to keep the function_call logic, I don't think
    # The 'functions' and 'function_call' fields have been dreprecated and
    # replaced with 'tools' and 'tool_choice', that work similarly but allow
    # for multiple functions to be invoked.
    functions: List[V1Function] = None
    function_call: Union[V1ToolChoiceKeyword, V1ToolChoiceFunction] = None
    tools: List[V1ToolFunction] = None
    # tool_choice: "auto" (default): allow model decide whether to call functions & if so, which
    # tool_choice: "required": force model to always call one or more functions
    # tool_choice: {"type": "function", "function": {"name": "my_function"}}: force model to call only one specific function
    # tool_choice: "none": disable function calling & force model to only generate a user-facing message
    tool_choice: Union[V1ToolChoiceKeyword, V1ToolChoiceFunction] = None
    tool_options: V1ToolOptions = None
    response_format: V1ResponseFormat = None
    stream: bool = False
    stream_options: V1StreamOptions = None


@app.post('/v1/chat/completions')
async def post_v1_chat_completions(req_data: V1ChatCompletionsRequest):
    debug('REQUEST', req_data)
    if req_data.stream:
        return StreamingResponse(
            content=post_v1_chat_completions_impl(req_data),
            media_type='text/event-stream',
        )
    else:
        # FUTURE: Python 3.10 can use `await anext(x))` instead of `await x.__anext__()`.
        response = await post_v1_chat_completions_impl(req_data).__anext__()
        debug('RESPONSE', response)
        return response


async def post_v1_chat_completions_impl(req_data: V1ChatCompletionsRequest):
    messages = req_data.messages[:]

    # Extract valid functions from the req_data.
    functions = []
    is_legacy_function_call = False
    # print(req_data)
    if req_data.tool_choice == 'none':
        pass
    elif req_data.tools is None:
        warnings.warn('Malformed request: tool_choice is not omitted or "none" yet no tools were provided')
    elif req_data.tool_choice == 'auto':
        functions = [tool.function for tool in req_data.tools if tool.type == 'function']
    elif req_data.tool_choice is not None:
        functions = [
            next(
                tool.function
                for tool in req_data.tools
                if tool.type == 'function'
                and tool.function.name == req_data.function_call.name
            )
        ]

    model_name = app.state.params['model']
    schema = None
    if functions:
        # If the req_data includes functions, create a system prompt to instruct the LLM
        # to use tools, and assemble a JSON schema to steer the LLM output.
        if req_data.stream:
            responder = ToolCallStreamingResponder(model_name, functions, app.state.model)
        else:
            responder = ToolCallResponder(model_name, functions)
        if not (req_data.tool_options and req_data.tool_options.no_prompt_steering):
            role = 'user' if model_flag.NO_SYSTEM_ROLE in app.state.model_flags else 'system'
            # print(role, model_flag.USER_ASSISTANT_ALT in app.state.model_flags, app.state.model_flags)
            if role == 'user' and model_flag.USER_ASSISTANT_ALT in app.state.model_flags:
                messages[0].content = messages[0].content=responder.tool_prompt + '\n\n' + messages[0].content
            else:
                messages.insert(
                    0,
                    V1ChatMessage(
                        role=role,
                        content=responder.tool_prompt,
                    ),
                )
        schema = responder.schema
    else:
        if req_data.stream:
            responder = ChatCompletionStreamingResponder(model_name)
        else:
            responder = ChatCompletionResponder(model_name)
        if req_data.response_format:
            assert req_data.response_format.type == V1ResponseFormatType.JSON_OBJECT
            # The req_data may specify a JSON schema (this option is not in the OpenAI API)
            if req_data.response_format.json_schema:
                schema = json.loads(req_data.response_format.json_schema)
            else:
                schema = {'type': 'object'}

    # import pprint; pprint.pprint(messages)
    if schema is None:
        debug('Warning: no JSON schema provided. Generating without one.')
    else:
        debug('Using schema:', schema)

    info('Starting generation…')

    prompt_tokens = None

    for result in app.state.model.completion(
        messages,
        schema=schema,
        max_tokens=req_data.max_tokens,
        temp=req_data.temperature,
        # Turn off prompt caching until we figure out https://github.com/OoriData/Toolio/issues/12
        # cache_prompt=True,
        cache_prompt=False,
    ):
        if result['op'] == 'evaluatedPrompt':
            prompt_tokens = result['token_count']
        elif result['op'] == 'generatedTokens':
            message = responder.generated_tokens(result['text'])
            if message:
                yield message
        elif result['op'] == 'stop':
            completion_tokens = result['token_count']
            yield responder.generation_stopped(
                result['reason'], prompt_tokens, completion_tokens
            )
        else:
            assert False


@click.command()
# @click.option('--prompt', help='Prompt text; can use {jsonschema} placeholder for the schema')
# @click.option('--prompt-file', type=click.File('rb'), help='Prompt text; can use {jsonschema} placeholder for the schema')
@click.option('--host', default='127.0.0.1', help='Host nodename/address for the launched server')
@click.option('--port', default=8000, help='Network port for the launched server')
# @click.option('--reload', is_flag=True, help='Reload on code update')
@click.option('--model', type=str, help='HuggingFace ID or disk path for locally-hosted MLF format model')
@click.option('--default-schema',
    help='JSON schema to be used if not provided via API call. Interpolated into {jsonschema} placeholder in prompts')
@click.option('--default-schema-file',
    help='Path to JSON schema to be used if not provided via API call. Interpolated into {jsonschema} placeholder in prompts')
@click.option('--llmtemp', default='0.1', type=float, help='LLM sampling temperature')
def main(host, port, model, default_schema, default_schema_file, llmtemp):
    app_params.update(model=model, default_schema=default_schema, default_schema_fpath=default_schema_file, llmtemp=llmtemp)
    uvicorn.run('toolio.cli.server:app', host=host, port=port, reload=False)
