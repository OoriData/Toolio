# toolio.tool.schematics
'''
Constructs to help define tool calls, and their schema, for use with tool calling flows
'''

import functools
import textwrap
from dataclasses import dataclass


@dataclass
class param:
    'Definition of a parameter for a tool callable'
    name: str
    typ: str
    desc: str
    required: bool = False


def tool(name, desc=None, params=None):
    params = params or {}
    def tool_dec(func):
        @functools.wraps(func)
        def tool_inner(*args, **kwargs):
            # If any invoke-time setup code is required, here it goes
            value = func(*args, **kwargs)
            return value
        schema_params = {}
        required_list = []
        for p in params:
            # Translate type designation to JSON Schema, if need be
            typ = TYPES_LOOKUP.get(p.typ, p.typ)
            schema_params[p.name] = {'type': typ, 'description': p.desc}
            if p.required:
                required_list.append(p.name)
        # Description can come from the docstring, or be overridden by kwarg
        _desc = desc or textwrap.dedent(func.__doc__)
        
        schema = {
            'type': 'function',
            'function': {
                'name': name,
                'description': _desc,
                'parameters': {'type': 'object', 'properties': schema_params, 'required': required_list}}
            }
        tool_inner.schema = schema
        tool_inner.name = name
        return tool_inner
    return tool_dec


TYPES_LOOKUP = {
    str: 'string',
    int: 'number',
    float: 'number',
    bool: 'boolean',
    }
