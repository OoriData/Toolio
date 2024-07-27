# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.responder
'''
Marshalling sreaming & non-streaming responses from LLMs
'''

import json
import time

from toolio import LANG


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
        return message
        # return f'data: {json.dumps(message)}\n'

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
        return message
        # return f'data: {json.dumps(message)}\ndata: [DONE]\n'


class ToolCallResponder(ChatCompletionResponder):
    '''
    For notes on OpenAI-style tool calling:
    https://platform.openai.com/docs/guides/function-calling?lang=python

    > Basic sequence of steps for function calling:
    > 1. Call the model with the user query and a set of functions defined in the functions parameter.
    > 2. The model can choose to call one or more functions; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may hallucinate parameters).
    > 3. Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
    > 4. Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.
    '''
    def __init__(
        self, model_name: str, functions: list[dict]  # , is_legacy_function_call: bool
    ):
        super().__init__(model_name)

        # XXX: Can we just remove legacy OpenAI API support?
        is_legacy_function_call = False
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
        # is_legacy_function_call: bool,
        model,
    ):
        # XXX: Can we just remove legacy OpenAI API support?
        is_legacy_function_call = False
        # super().__init__(model_name, functions, is_legacy_function_call)
        super().__init__(model_name, functions)
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
        return message
        # return f'data: {json.dumps(message)}\n'

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
        return message
        # return f'data: {json.dumps(message)}\ndata: [DONE]\n'
