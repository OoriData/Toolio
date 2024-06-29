# toolio.util
'''
Utility support functions (but not tools 😆) for Toolio
'''
import asyncio
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
        return True, asyncio.iscoroutinefunction(obj)

    if hasattr(obj, '__call__'):
        return True, asyncio.iscoroutinefunction(obj.__call__)

    raise RuntimeError(f'Unexpected case: Callable, yet not having a __call__ method: {obj=}')
