# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.prompt_helper

import json
# from itertools import chain

from toolio.http_schematics import V1Function

from toolio.common import LANG, model_flag, DEFAULT_FLAGS
from toolio.http_schematics import V1ChatMessage


class multi_tool_prompt_default(dict):
    def __missing__(self, key):
        match key:
            case 'leadin':
                return LANG['multi_tool_prompt_leadin']
            case 'tail':
                return LANG['multi_tool_prompt_tail']
            case _:
                raise KeyError(f'Unknown formattr parameter {key}')


class single_tool_prompt_default(dict):
    def __missing__(self, key):
        match key:
            case 'leadin':
                return LANG['one_tool_prompt_leadin']
            case 'tail':
                return LANG['one_tool_prompt_tail']
            case _:
                raise KeyError(f'Unknown formattr parameter {key}')


PROMPT_FORMATTER = '{leadin}\n{tool_spec}\n{tail}'


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


def set_tool_response(msgs, tool_call_id, tool_name, call_args, call_result, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment in place
    tool_response - response generatded by selected tool
    model_flags - flags indicating the expectations of the hosted LLM
    '''
    # XXX: model_flags = None ⇒ assistant-style tool response. Is this the default we want?
    if model_flag.TOOL_RESPONSE in model_flags:
        msgs.append({
            'tool_call_id': tool_call_id,
            'role': 'tool',
            'name': tool_name,
            'content': call_result,
        })
    else:
        # FIXME: Separate out natural language
        tool_response_text =  f'Called tool {tool_name} with arguments {call_args}. Result: {call_result}'
        # tool_response_text = f'Result of the call to {tool_name}: {tool_result}'
        if model_flag.USER_ASSISTANT_ALT in model_flags:
            # If there is already an assistant msg from tool-calling, merge it
            if msgs[-1]['role'] == 'assistant':
                msgs[-1]['content'] += '\n\n' + tool_response_text
            else:
                msgs.append({'role': 'assistant', 'content': tool_response_text})
        else:
            msgs.append({'role': 'assistant', 'content': tool_response_text})


def set_continue_message(msgs, continue_msg, model_flags=DEFAULT_FLAGS):
    '''
    msgs - chat messages to augment in place
    continue_msg - message instructing the LLM on how to continue or finalize
    '''
    if msgs[-1]['role'] == 'tool':
        # No continue message, in this case
        return
    if msgs[-1]['role'] == 'user' and model_flag.USER_ASSISTANT_ALT in model_flags:
        # Need to merge adjacent user messages
        msgs[-1]['content'] += '\n\n' + continue_msg
    else:
        msgs.append({'role': 'user', 'content': continue_msg})


# XXX: Not currently used
# def select_tool_prompt(self, tools, tool_schemas, separator='\n', leadin=None):
#     'Construct a prompt to offer the LLM multiple tools'
#     leadin = leadin or LANG['multi_tool_prompt_leadin']
#     toollist = separator.join(
#         [f'\n{LANG["select_tool_prompt_toollabel"]} {tool["name"]}: {tool["description"]}\n'
#             f'{LANG["select_tool_prompt_schemalabel"]}: {json.dumps(tool_schema)}\n'
#             for tool, tool_schema in zip(tools, tool_schemas) ])
#     return f'''
# {leadin}
# {toollist}
# {LANG["select_tool_prompt_tail"]}
# '''


# XXX: no_tool_desc can just be one of the defalted params
def process_tools_for_sysmsg(tools, internal_tools, separator='\n', **kwargs):
    '''
    Given a set of tools, format for use in a system prompt, and other modularized processing

    Args:
        no_tool_desc: description to use for the default get-out-of-jail model response
    '''
    # Start by normalizing away from Pydantic form
    tools = [ (t.dictify() if isinstance(t, V1Function) else t) for t in tools ]
    tools.extend(internal_tools)
    tool_schemas = [
        {
            'type': 'object',
            'name': fn['name'],
            'description': fn['description'],
            'properties': {
                'name': {'type': 'const', 'const': fn['name']},
                'arguments': fn['parameters'],
            },
            'required': ['name', 'arguments'],
        }
        for fn in tools
    ]
    # XXX: Do we need to differentiate prompt setup for 1 vs multiple tools?
    # if len(tool_schemas) - len(internal_tools) == 1:
    #     # Single tool provided (second is the no-tool escape)
    #     tool_str = separator.join(
    #         [f'\nTool name: {tool["name"]}\n {tool["description"]}\nInvocation schema:\n{json.dumps(ts)}'
    #             for tool, ts in zip(tools, tool_schemas) ])
    # else:
    #     # XXX: Use LANG['one_tool_prompt_schemalabel']
    #     tool_str = json.dumps(tool_schemas)
    # default_cls = single_tool_prompt_default if len(tools) == 1 else multi_tool_prompt_default

    prompt_tpl = multi_tool_prompt_default
    tool_str = json.dumps(tool_schemas)

    full_schema = {'type': 'array', 'items': {'anyOf': tool_schemas}}

    sysprompt = PROMPT_FORMATTER.format_map(prompt_tpl(tool_spec=tool_str, **kwargs))

    return full_schema, tool_schemas, sysprompt
