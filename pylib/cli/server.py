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
import time
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import click
import uvicorn

from llm_structured_output.util.output import info, warning, debug

from toolio.http_schematics import V1ChatCompletionsRequest
from toolio.common import DEFAULT_FLAGS, FLAGS_LOOKUP
from toolio.schema_helper import Model
from toolio.http_impl import post_v1_chat_completions_impl


# List of known loggers with too much chatter at debug level
TAME_LOGGERS = ['asyncio', 'httpcore', 'httpx']
for l in TAME_LOGGERS:
    logging.getLogger(l).setLevel(logging.WARNING)

# Note: above explicit list is a better idea than some sort of blanket approach such as the following:
# for handler in logging.root.handlers:
#     handler.addFilter(logging.Filter('toolio'))

# There is further log config below, in main()

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
            content=post_v1_chat_completions_impl(app.state, req_data),
            media_type='text/event-stream',
        )
    else:
        response = await anext(post_v1_chat_completions_impl(app.state, req_data))
        debug('RESPONSE', response)
        return response


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
@click.option('--loglevel', default='INFO', help='Log level, e.g. DEBUG or INFO')
def main(host, port, model, default_schema, default_schema_file, llmtemp, workers, cors_origin, loglevel):
    global logger  # Remove this line when we modularize
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)  # Seems redundant, but is necessary. Python logging is quirky

    app_params.update(model=model, default_schema=default_schema, default_schema_fpath=default_schema_file,
                      llmtemp=llmtemp)
    app.add_middleware(CORSMiddleware, allow_origins=list(cors_origin), allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    workers = workers or None
    # logger.info(f'Host has {NUM_CPUS} CPU cores')
    uvicorn.run('toolio.cli.server:app', host=host, port=port, reload=False, workers=workers)


# TODO: Implement log config, probably as a server cli arg
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
