# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.http_impl
'''
Heart of the implementation for HTTP server request handling
'''
import json
import warnings

from llm_structured_output.util.output import info, debug

from toolio.common import prompt_handler
from toolio.toolcall import DEFAULT_INTERNAL_TOOLS, process_tools_for_sysmsg
from toolio.http_schematics import V1ChatMessage, V1ChatCompletionsRequest, V1ResponseFormatType
from toolio.responder import (ToolCallStreamingResponder, ToolCallResponder,
                              ChatCompletionResponder, ChatCompletionStreamingResponder)


async def post_v1_chat_completions_impl(state, req_data: V1ChatCompletionsRequest):
    '''
    Process HTTP request, determining whether tool-calling or output schema are to be enabled
    Turn the Pydantic bits into regular structure
    '''
    messages = [ (m.dictify() if isinstance(m, V1ChatMessage) else m) for m in req_data.messages ]

    tools = []
    # Only turn on tool calling if tools are provided
    if req_data.tools:
        # See: https://platform.openai.com/docs/api-reference/chat/create
        # `tool_choice` controls which (if any) tool is called by the model
        # `none`: model will not call any tool and instead generates a message
        # `auto`: model can pick between generating a message or calling one or more tools
        # `required` means the model must call one or more tools
        #  specific tool name forces the model to call that tool
        # `none` is the default when no tools are present; `auto` is the default if tools are present
        if req_data.tool_choice == 'none':
            pass  # leave tools empty
        elif req_data.tool_choice in ['auto', 'required', None]:
            tools = [tool.function for tool in req_data.tools if tool.type == 'function']
            if req_data.tool_choice == 'required':
                warnings.warn('tool_choice `required` is not supported; falling back to `auto`', stacklevel=2)
        else:
            # Specific tool named; grab the first tool by the given name
            tools = [next(
                    tool.function for tool in req_data.tools
                    if tool.type == 'function'
                    and tool.function.name == req_data.function_call.name
                )]

    model_name = state.params['model']
    model_type = state.model.model.model_type
    schema = None  # Steering for the LLM output (JSON schema)
    phandler = prompt_handler(state.model_flags)
    if tools:
        if req_data.sysmsg_leadin:  # Caller provided sysmsg leadin via protocol
            leadin = req_data.sysmsg_leadin
        elif messages[0]['role'] == 'system':  # Caller provided sysmsg leadin in the chat messages
            leadin = messages[0]['content']
            del messages[0]
        else:  # Use default leadin
            leadin = None
        # Schema, including no-tool fallback, plus string spec of available tools, for use in constructing sysmsg
        full_schema, tool_schemas, sysmsg = process_tools_for_sysmsg(tools, DEFAULT_INTERNAL_TOOLS, leadin=leadin)
        # debug(f'{sysmsg=}')
        if req_data.stream:
            responder = ToolCallStreamingResponder(model_name, model_type, tool_schemas)
        else:
            responder = ToolCallResponder(model_name, model_type)
        if not (req_data.tool_options and req_data.tool_options.no_prompt_steering):
            messages = phandler.reconstruct_messages(messages, sysmsg=sysmsg)
        schema = full_schema
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
        debug('No JSON schema provided. Generating without one.')
    else:
        debug('Using schema:', schema)

    info('Starting generation…')

    prompt_tokens = None

    for result in state.model.completion(
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
