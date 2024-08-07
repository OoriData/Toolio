# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.prompt_helper

import json

from toolio import LANG, model_flag, DEFAULT_FLAGS
from toolio.http_schematics import V1ChatMessage


def enrich_chat_for_tools(msgs, tool_prompt, model_flags):
    '''
    msgs - chat messages to augment
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # Add prompting (system prompt, if permitted) instructing the LLM to use tools
    if model_flag.NO_SYSTEM_ROLE in model_flags:  # LLM supports system messages
        msgs.insert(0, V1ChatMessage(role='system', content=tool_prompt))
    elif model_flag.USER_ASSISTANT_ALT in model_flags: # LLM insists that user and assistant messages must alternate
        msgs[0].content = msgs[0].content=tool_prompt + '\n\n' + msgs[0].content
    else:
        msgs.insert(0, V1ChatMessage(role='user', content=tool_prompt))


def set_tool_response(msgs, tool_call_id, tool_name, tool_result, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment
    tool_response - response generatded by selected tool
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # XXX: model_flags = None ⇒ assistant-style tool response. Is this the default we want?
    if model_flag.TOOL_RESPONSE in model_flags:
        msgs.append({
            'tool_call_id': tool_call_id,
            'role': 'tool',
            'name': tool_name,
            'content': tool_result,
        })
    else:
        # FIXME: Separate out natural language
        tool_response_text = f'Result of the call to {tool_name}: {tool_result}'
        if model_flag.USER_ASSISTANT_ALT in model_flags:
            # If there is already an assistant msg from tool-calling, merge it
            if msgs[-1]['role'] == 'assistant':
                msgs[-1]['content'] += '\n\n' + tool_response_text
            else:
                msgs.append({'role': 'assistant', 'content': tool_response_text})
        else:
            msgs.append({'role': 'assistant', 'content': tool_response_text})


def single_tool_prompt(tool, tool_schema, leadin=None):
    leadin = leadin or LANG['one_tool_prompt_leadin']
    return f'''
{leadin} {tool["name"]}: {tool["description"]}
{LANG["one_tool_prompt_schemalabel"]}: {json.dumps(tool_schema)}
{LANG["one_tool_prompt_tail"]}
'''


def multiple_tool_prompt(tools, tool_schemas, separator='\n', leadin=None):
    leadin = leadin or LANG['multi_tool_prompt_leadin']
    toollist = separator.join(
        [f'\nTool {tool["name"]}: {tool["description"]}\nInvocation schema: {json.dumps(tool_schema)}\n'
            for tool, tool_schema in zip(tools, tool_schemas) ])
    return f'''
{leadin}
{toollist}
{LANG["multi_tool_prompt_tail"]}
'''


def select_tool_prompt(self, tools, tool_schemas, separator='\n', leadin=None):
    leadin = leadin or LANG['multi_tool_prompt_leadin']
    toollist = separator.join(
        [f'\n{LANG["select_tool_prompt_toollabel"]} {tool["name"]}: {tool["description"]}\n'
            f'{LANG["select_tool_prompt_schemalabel"]}: {json.dumps(tool_schema)}\n'
            for tool, tool_schema in zip(tools, tool_schemas) ])
    return f'''
{leadin}
{toollist}
{LANG["select_tool_prompt_tail"]}
'''


def process_tool_sysmsg(tools, leadin=None):
    # print(f'{tools=} | {leadin=}')
    function_schemas = [
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'const', 'const': fn['name']},
                'arguments': fn['parameters'],
            },
            'required': ['name', 'arguments'],
        }
        for fn in tools
    ]
    if len(function_schemas) == 1:
        schema = function_schemas[0]
        tool_sysmsg = single_tool_prompt(tools[0], function_schemas[0], leadin=leadin)
    else:
        schema = {'type': 'array', 'items': {'anyOf': function_schemas}}
        tool_sysmsg = multiple_tool_prompt(tools, function_schemas, leadin=leadin)
    # print(f'{tool_sysmsg=}')
    return schema, tool_sysmsg
