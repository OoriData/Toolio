# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.http_schematics

from enum import Enum
from typing import Literal, List, Optional, Union

from pydantic import BaseModel, Field


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

    def dictify(self):
        return {'name': self.name, 'description': self.description, 'parameters': self.parameters}


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
    sysmsg_leadin: str | None = None
