# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.tool.schematics
'''
Constructs to help define tool calls, and their schema, for use with tool calling flows
'''
import inspect
import functools
import textwrap
from dataclasses import dataclass
from enum import Enum


@dataclass
class param:
    'Definition of a parameter for a tool callable'
    name: str
    typ: str
    desc: str
    required: bool = False
    rename: str = None


def tool(name, desc=None, params=None):
    params = params or []
    def tool_dec(func):
        schema_params = {}
        params_lookup = {}
        renames = {}
        required_list = []
        for p in params:
            # Translate type designation to JSON Schema, if need be
            typ = TYPES_LOOKUP.get(p.typ)
            if typ:
                schema_params[p.name] = {'type': typ, 'description': p.desc}
            else:
                typ = complex_type(p.typ)
                if isinstance(typ, dict):
                    schema_params[p.name] = {'description': p.desc, **typ}
                else:
                    raise RuntimeError('Unable to determine param type')
            params_lookup[p.name] = p
            if p.required:
                required_list.append(p.name)
            if p.rename:
                renames[p.name] = p.rename
        # Description can come from the docstring, or be overridden by kwarg
        _desc = desc or ( textwrap.dedent(func.__doc__) if func.__doc__ is not None else None )
        if not _desc:
            raise ValueError('No description found for tool, either via docstring or parameter')

        schema = {'name': name, 'description': _desc,
                  'parameters': {'type': 'object', 'properties': schema_params, 'required': required_list}}

        @functools.wraps(func)
        def tool_inner_prep(*args, **kwargs):
            # Invoke-time setup code
            processed_kwargs = {}
            for (k, v) in kwargs.items():
                typ = params_lookup[k].typ
                processed_kwargs[renames.get(k, k)] = typ(v)
            return processed_kwargs
        # Tool functions can be classic, or coroutines (async def); handle both cases
        if inspect.iscoroutinefunction(func):
            async def tool_inner(*args, **kwargs):
                processed_kwargs = tool_inner_prep(*args, **kwargs)
                retval = await func(*args, **processed_kwargs)
                return retval
        else:
            def tool_inner(*args, **kwargs):
                processed_kwargs = tool_inner_prep(*args, **kwargs)
                retval = func(*args, **processed_kwargs)
                return retval
        tool_inner.schema = schema
        tool_inner.name = name
        return tool_inner
    return tool_dec


TYPES_LOOKUP = {
    str: 'string',
    int: 'number',
    float: 'number',
    bool: 'boolean',
    'string': 'string',
    'number': 'number',
    'boolean': 'boolean',
    }


def complex_type(typ):
    if issubclass(typ, Enum):
        # Worth being aware of typ._member_names_ & typ._member_map_
        return {'type': 'string', 'enum': [ e.value for e in typ ]}
    else:
        return None

