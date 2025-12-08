# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.util
'''
Utility support functions (but not LLM-callable tools 😆) for Toolio
'''
import asyncio
import inspect
import types
from typing import Any


def check_callable(obj: Any):
    '''
    Based on check_callable by Simon Willison
    https://til.simonwillison.net/python/callable

    obj - object to check
    return - tuple of bools: is_callable, is_async_callable
    '''
    if not callable(obj):
        return False, False

    if isinstance(obj, type):
        # It's a class
        return True, False

    if isinstance(obj, types.FunctionType):
        return True, inspect.iscoroutinefunction(obj)

    if hasattr(obj, '__call__'):
        return True, inspect.iscoroutinefunction(obj.__call__)

    raise RuntimeError(f'Unexpected case: Callable, yet not having a __call__ method: {obj=}')


class attr_dict(dict):
    '''
    Dictionary with attribute access
    '''
    # XXX: Should unknown attr access return None rather than raise?
    # If so, can just do: __getattr__ = dict.get
    # __getattr__ = dict.__getitem__
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            # Substitute with more normally expected exception
            raise AttributeError(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
