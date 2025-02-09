# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.http_impl
'''
Heart of the implementation for HTTP server request handling
'''
import json
import warnings

from mlx_lm.sample_utils import make_sampler

from toolio.vendor.llm_structured_output.util.output import info, debug
from toolio.common import prompt_handler
from toolio.toolcall import DEFAULT_INTERNAL_TOOLS, process_tools_for_sysmsg
from toolio.http_schematics import V1ChatMessage, V1ChatCompletionsRequest, V1ResponseFormatType
from toolio.response_helper import llm_response, llm_response_type

async def post_v1_chat_completions_impl(state, req_data: V1ChatCompletionsRequest):
    '''
    Process HTTP request, determining whether tool-calling or output schema are to be enabled
    Turn the Pydantic bits into regular structure
    '''
    messages = [m.dictify() if isinstance(m, V1ChatMessage) else m for m in req_data.messages]

    tools = None
    # Only turn on tool calling if tools are provided
    if req_data.tools:
        # Handle tool_choice options. See: https://platform.openai.com/docs/api-reference/chat/create
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
                if tool.type == 'function' and tool.function.name == req_data.tool_choice.name
            )]

    # XXX: Add feature to request that the server cache a prompt to be passed in here: cache_prompt
    kwargs = {'max_tokens': req_data.max_tokens, 'temperature': req_data.temperature}
    schema = None
    if req_data.response_format:
        assert req_data.response_format.type == V1ResponseFormatType.JSON_OBJECT
        # The req_data may specify a JSON schema (this option is not in the OpenAI API)
        if req_data.response_format.json_schema:
            schema = json.loads(req_data.response_format.json_schema)
        else:
            schema = {'type': 'object'}
    if tools:
        debug('Tool-calling completion. Starting generation…')
    else:
        debug('Using schema:', str(schema)[:200] if schema else 'No JSON schema provided. Starting generation…')
    return await state.model_runner(messages, tools=tools, json_schema=schema, **kwargs)
