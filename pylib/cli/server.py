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

import os
import json
import time
from contextlib import asynccontextmanager
import warnings

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import click
import uvicorn

from llm_structured_output.util.output import info, warning, debug

from toolio.schema_helper import Model
from toolio.http_schematics import V1Function
from toolio.prompt_helper import enrich_chat_for_tools, process_tool_sysmsg
from toolio.llm_helper import DEFAULT_FLAGS, FLAGS_LOOKUP
from toolio.http_schematics import V1ChatCompletionsRequest, V1ResponseFormatType
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


NUM_CPUS = int(os.cpu_count())
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
    app.state.model_flags = FLAGS_LOOKUP.get(app.state.model.model.model_type, DEFAULT_FLAGS)
    # Look into exposing control over methods & headers as well
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
    model_type = app.state.model.model.model_type
    schema = None  # Steering for the LLM output (JSON schema)
    if functions:
        if req_data.sysmsg_leadin:  # Caller provided sysmsg leadin via protocol
            leadin = req_data.sysmsg_leadin
        elif messages[0].role == 'system':  # Caller provided sysmsg leadin in the chat messages
            leadin = messages[0].content
            del messages[0]
        else:  # Use default leadin
            leadin = None
        functions = [ (t.dictify() if isinstance(t, V1Function) else t) for t in functions ]
        schema, tool_sysmsg = process_tool_sysmsg(functions, leadin=leadin)
        if req_data.stream:
            responder = ToolCallStreamingResponder(model_name, model_type, functions, schema, tool_sysmsg)
        else:
            responder = ToolCallResponder(model_name, model_type, schema, tool_sysmsg)
        if not (req_data.tool_options and req_data.tool_options.no_prompt_steering):
            enrich_chat_for_tools(messages, tool_sysmsg, app.state.model_flags)
            # import pprint; pprint.pprint(messages)
    else:
        if req_data.stream:
            responder = ChatCompletionStreamingResponder(model_name, model_type)
        else:
            responder = ChatCompletionResponder(model_name, model_type)
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
@click.option('--host', default='127.0.0.1', help='Host nodename/address for the launched server')
@click.option('--port', default=8000, help='Network port for the launched server')
# @click.option('--reload', is_flag=True, help='Reload on code update')
@click.option('--model', type=str, help='HuggingFace ID or disk path for locally-hosted MLF format model')
@click.option('--default-schema',
    help='JSON schema to be used if not provided via API call. Interpolated into {jsonschema} placeholder in prompts')
@click.option('--default-schema-file',
    help='Path to JSON schema to be used if not provided via API call.'
         'Interpolated into {jsonschema} placeholder in prompts')
@click.option('--llmtemp', default='0.1', type=float, help='LLM sampling temperature')
@click.option('--workers', type=int, default=0,
              help='Number of workers processes to spawn (each utilizes one CPU core).'
              'Defaults to $WEB_CONCURRENCY environment variable if available, or 1')
@click.option('--cors_origin', multiple=True,
              help='Origin to be permitted for CORS https://fastapi.tiangolo.com/tutorial/cors/')
def main(host, port, model, default_schema, default_schema_file, llmtemp, workers, cors_origin):
    app_params.update(model=model, default_schema=default_schema, default_schema_fpath=default_schema_file,
                      llmtemp=llmtemp)
    app.add_middleware(CORSMiddleware, allow_origins=list(cors_origin), allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    workers = workers or None
    # logger.info(f'Host has {NUM_CPUS} CPU cores')
    uvicorn.run('toolio.cli.server:app', host=host, port=port, reload=False, workers=workers)


# Implement log config when we 
def UNUSED_log_setup(config):
    # Set up logging
    import logging
    global logger  # noqa: PLW0603

    main_loglevel = config.get('log', {'level': 'INFO'})['level']
    logging.config.dictConfig(config['log'])
    # Following 2 lines configure the root logger, so all other loggers in this process space will inherit
    # logging.basicConfig(level=main_loglevel, format='%(levelname)s:%(name)s: %(message)s')
    logging.getLogger().setLevel(main_loglevel)  # Seems redundant, but is necessary. Python logging is quirky
    logger = logging.getLogger(__name__)
    # logger.addFilter(LocalFilter())
