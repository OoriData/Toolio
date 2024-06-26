# toolio.cli.server
'''
LLM server with OpenAI-like API, for structured prompting support & including function/tool calling

Based on: https://github.com/otriscon/llm-structured-output/blob/main/src/examples/server.py

TODO: Break out CLI stuff (new module for FastAPI bits)

MLXStructuredLMServer --model=mlx-community/Hermes-2-Theta-Llama-3-8B-4bit

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

from toolio import LANG
from toolio.schema_helper import Model


app_params = {}

# Context manager for the FastAPI app's lifespan: https://fastapi.tiangolo.com/advanced/events/
# Don't just stick this in globals, because we're planning for better async support
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Load model one-time
    '''
    # Startup code here. Particularly persistence, so that its DB connection pool is set up in the right event loop
    info('Loading model...')
    app.state.model = Model()
    app.state.params = app_params
    # Can use click's env support if we decide we want this
    # model_path = os.environ['MODEL_PATH']
    app.state.model.load(app_params['model'])
    yield
    # Shutdown code here, if any


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
# pylint: disable-next=unused-argument
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'
    warning(f'RequestValidationError: {exc_str}')
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
    json_schema: str = Field(alias='schema')


class V1StreamOptions(BaseModel):
    include_usage: bool = False


class V1ChatCompletionsRequest(BaseModel):
    # pylint: disable=too-many-instance-attributes # Paternalistic excess
    model: str = 'default'
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
    elif req_data.function_call == 'none':
        pass
    elif req_data.function_call == 'auto':
        functions = req_data.functions
        is_legacy_function_call = True
    elif req_data.function_call is not None:
        functions = [
            next(
                fn for fn in req_data.functions if fn.name == req_data.function_call.name
            )
        ]
        is_legacy_function_call = True

    model_name = app.state.params['model']
    schema = None
    if functions:
        # If the req_data includes functions, create a system prompt to instruct the LLM
        # to use tools, and assemble a JSON schema to steer the LLM output.
        if req_data.stream:
            responder = ToolCallStreamingResponder(
                model_name,
                functions,
                is_legacy_function_call,
                app.state.model,
            )
        else:
            responder = ToolCallResponder(
                model_name, functions, is_legacy_function_call
            )
        if not (req_data.tool_options and req_data.tool_options.no_prompt_steering):
            messages.insert(
                0,
                V1ChatMessage(
                    role='system',
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
        cache_prompt=True,
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


class ChatCompletionResponder:
    def __init__(self, model_name: str):
        self.object_type = 'chat.completion'
        self.model_name = model_name
        self.created = int(time.time())
        self.id = f'{id(self)}_{self.created}'
        self.content = ''

    def message_properties(self):
        return {
            'object': self.object_type,
            'id': f'chatcmpl-{self.id}',
            'created': self.created,
            'model': self.model_name,
        }

    def translate_reason(self, reason):
        '''
        Translate our reason codes to OpenAI ones.
        '''
        if reason == 'end':
            return 'stop'
        if reason == 'max_tokens':
            return 'length'
        return f'error: {reason}'  # Not a standard OpenAI API reason

    def format_usage(self, prompt_tokens: int, completion_tokens: int):
        return {
            'usage': {
                'completion_tokens': completion_tokens,
                'prompt_tokens': prompt_tokens,
                'total_tokens': completion_tokens + prompt_tokens,
            },
        }

    def generated_tokens(
        self,
        text: str,
    ):
        self.content += text
        return None

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        message = {'role': 'assistant', 'content': self.content}
        return {
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': finish_reason}
            ],
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }


class ChatCompletionStreamingResponder(ChatCompletionResponder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.object_type = 'chat.completion.chunk'

    def generated_tokens(
        self,
        text: str,
    ):
        delta = {'role': 'assistant', 'content': text}
        message = {
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}],
            **self.message_properties(),
        }
        return f'data: {json.dumps(message)}\n'

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        delta = {'role': 'assistant', 'content': ''}
        message = {
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
            # Usage field notes:
            # - OpenAI only sends usage in streaming if the option
            #   stream_options.include_usage is true, but we send it always.
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }
        return f'data: {json.dumps(message)}\ndata: [DONE]\n'


class ToolCallResponder(ChatCompletionResponder):
    '''
    For notes on OpenAI-style tool calling:
    https://platform.openai.com/docs/guides/function-calling?lang=python

    > The basic sequence of steps for function calling is as follows:
    > 1. Call the model with the user query and a set of functions defined in the functions parameter.
    > 2. The model can choose to call one or more functions; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may hallucinate parameters).
    > 3. Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
    > 4. Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.
    '''
    def __init__(
        self, model_name: str, functions: list[dict], is_legacy_function_call: bool
    ):
        super().__init__(model_name)

        self.is_legacy_function_call = is_legacy_function_call

        function_schemas = [
            {
                'type': 'object',
                'properties': {
                    'name': {'type': 'const', 'const': fn.name},
                    'arguments': fn.parameters,
                },
                'required': ['name', 'arguments'],
            }
            for fn in functions
        ]
        if len(function_schemas) == 1:
            self.schema = function_schemas[0]
            self.tool_prompt = self._one_tool_prompt(functions[0], function_schemas[0])
        elif is_legacy_function_call:  # Only allows one function to be called.
            self.schema = {'oneOf': function_schemas}
            self.tool_prompt = self._select_tool_prompt(functions, function_schemas)
        else:
            self.schema = {'type': 'array', 'items': {'anyOf': function_schemas}}
            self.tool_prompt = self._multiple_tool_prompt(functions, function_schemas)

    def translate_reason(self, reason):
        if reason == 'end':
            if self.is_legacy_function_call:
                return 'function_call'
            return 'tool_calls'
        return super().translate_reason(reason)

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        if finish_reason == 'tool_calls':
            tool_calls = json.loads(self.content)
            if not isinstance(tool_calls, list):
                # len(functions) == 1 was special cased
                tool_calls = [tool_calls]
            message = {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': f'call_{self.id}_{i}',
                        'type': 'function',
                        'function': {
                            'name': function_call['name'],
                            'arguments': json.dumps(function_call['arguments']),
                        },
                    }
                    for i, function_call in enumerate(tool_calls)
                ],
            }
        elif finish_reason == 'function_call':
            function_call = json.loads(self.content)
            message = {
                'role': 'assistant',
                'function_call': {
                    'name': function_call['name'],
                    'arguments': json.dumps(function_call['arguments']),
                },
            }
        else:
            message = None
        return {
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': finish_reason}
            ],
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }

    def _one_tool_prompt(self, tool, tool_schema):
        return f'''
{LANG["one_tool_prompt_leadin"]} {tool.name}: {tool.description}
{LANG["one_tool_prompt_schemalabel"]}: {json.dumps(tool_schema)}
{LANG["one_tool_prompt_tail"]}
'''

    def _multiple_tool_prompt(self, tools, tool_schemas, separator='\n'):
        toollist = separator.join(
            [f'\nTool {tool.name}: {tool.description}\nInvocation schema: {json.dumps(tool_schema)}\n'
                for tool, tool_schema in zip(tools, tool_schemas) ])
        return f'''
{LANG["multi_tool_prompt_leadin"]}
{toollist}
{LANG["multi_tool_prompt_tail"]}
'''

    def _select_tool_prompt(self, tools, tool_schemas, separator='\n'):
        toollist = separator.join(
            [f'\n{LANG["select_tool_prompt_toollabel"]} {tool.name}: {tool.description}\n'
             f'{LANG["select_tool_prompt_schemalabel"]}: {json.dumps(tool_schema)}\n'
                for tool, tool_schema in zip(tools, tool_schemas) ])
        return f'''
{LANG["multi_tool_prompt_leadin"]}
{toollist}
{LANG["select_tool_prompt_tail"]}
'''


class ToolCallStreamingResponder(ToolCallResponder):
    def __init__(
        self,
        model_name: str,
        functions: list[dict],
        is_legacy_function_call: bool,
        model,
    ):
        super().__init__(model_name, functions, is_legacy_function_call)
        self.object_type = 'chat.completion.chunk'

        # We need to parse the output as it's being generated in order to send
        # streaming messages that contain the name and arguments of the function
        # being called.

        self.current_function_index = -1
        self.current_function_name = None
        self.in_function_arguments = False

        def set_function_name(_prop_name: str, prop_value):
            self.current_function_index += 1
            self.current_function_name = prop_value

        def start_function_arguments(_prop_name: str):
            self.in_function_arguments = True

        def end_function_arguments(_prop_name: str, _prop_value: str):
            self.in_function_arguments = False

        hooked_function_schemas = [
            {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'const',
                        'const': fn.name,
                        '__hooks': {
                            'value_end': set_function_name,
                        },
                    },
                    'arguments': {
                        **fn.parameters,
                        '__hooks': {
                            'value_start': start_function_arguments,
                            'value_end': end_function_arguments,
                        },
                    },
                },
                'required': ['name', 'arguments'],
            }
            for fn in functions
        ]
        if len(hooked_function_schemas) == 1:
            hooked_schema = hooked_function_schemas[0]
        elif is_legacy_function_call:
            hooked_schema = {'oneOf': hooked_function_schemas}
        else:
            hooked_schema = {
                'type': 'array',
                'items': {'anyOf': hooked_function_schemas},
            }
        self.tool_call_parser = model.get_driver_for_json_schema(hooked_schema)

    def generated_tokens(
        self,
        text: str,
    ):
        argument_text = ''
        for char in text:
            if self.in_function_arguments:
                argument_text += char
            # Update state. This is certain to parse, no need to check for rejections.
            self.tool_call_parser.advance_char(char)
        if not argument_text:
            return None
        assert self.current_function_name
        if self.is_legacy_function_call:
            delta = {
                'function_call': {
                    'name': self.current_function_name,
                    'arguments': argument_text,
                }
            }
        else:
            delta = {
                'tool_calls': [
                    {
                        'index': self.current_function_index,
                        'id': f'call_{self.id}_{self.current_function_index}',
                        'type': 'function',
                        'function': {
                            # We send the name on every update, but OpenAI only sends it on
                            # the first one for each call, with empty arguments (''). Further
                            # updates only have the arguments field. This is something we may
                            # want to emulate if client code depends on this behavior.
                            'name': self.current_function_name,
                            'arguments': argument_text,
                        },
                    }
                ]
            }
        message = {
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}],
            **self.message_properties(),
        }
        return f'data: {json.dumps(message)}\n'

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        message = {
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}],
            # Usage field notes:
            # - OpenAI only sends usage in streaming if the option
            #   stream_options.include_usage is true, but we send it always.
            # - OpenAI sends two separate messages: one with the finish_reason and no
            #   usage field, and one with an empty choices array and the usage field.
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }
        return f'data: {json.dumps(message)}\ndata: [DONE]\n'


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
