'''
python test/quick_check.py mlx-community/Hermes-2-Theta-Llama-3-8B-4bit
'''

import sys
import asyncio

import click

from toolio.http_schematics import V1Function
from toolio.llm_helper import model_manager, extract_content

async def amain(mm):
    # Plain chat complete
    msgs = [
        # {'role': 'user', 'content': 'I am thinking of a number between 1 and 10. Guess what it is.'}
        {'role': 'user', 'content': 'Hello! How are you?'}
        ]
    async for chunk in extract_content(mm.complete(msgs)):
        print(chunk, end='')

    print('\n', '='*40, 'Country extraction')

    prompt = ('Which countries are mentioned in the sentence \'Adamma went home to Nigeria for the hols\'?'
              'Your answer should be only JSON, according to this schema: {json_schema}')
    schema = ('{"type": "array", "items":'
              '{"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}')
    msgs = [{'role': 'user', 'content': prompt.format(json_schema=schema)}]
    async for chunk in extract_content(mm.complete(msgs, json_schema=schema)):
        print(chunk, end='')

    print('\n', '='*40, 'Square root of 256, pt 1')

    prompt = 'What is the square root of 256?'
    # Plain dictionary form
    # tools = [{'name': 'square_root', 'description': 'Get the square root of the given number', 'parameters': {'type': 'object', 'properties': {'square': {'type': 'number', 'description': 'Number from which to find the square root'}}, 'required': ['square']}}]
    # Pydantic form
    # tools = [V1Function(name='square_root', description='Get the square root of the given number', parameters={'type': 'object', 'properties': {'square': {'type': 'number', 'description': 'Number from which to find the square root'}}, 'required': ['square']})]
    msgs = [ {'role': 'user', 'content': prompt} ]
    async for chunk in extract_content(mm.complete_with_tools(msgs, toolset=['square_root'], tool_choice='auto')):
        print(chunk, end='')

    print('\n', '='*40, 'Square root of 256, pt 2')
    # Actually we'll want to use pt 2 to test the Pydantic function reg form
    mm.clear_tools()
    mm.register_tool(sqrt, schema=SQUARE_ROOT_METADATA)

    async for chunk in extract_content(mm.complete_with_tools(msgs, toolset=['square_root'], tool_choice='auto')):
        print(chunk, end='')

    print('\n', '='*40, 'Usain bolt')

    from toolio.tool.math import calculator
    prompt='Usain Bolt ran the 100m race in 9.58s. What was his average velocity?'
    mm.register_tool(calculator)
    msgs = [ {'role': 'user', 'content': prompt} ]

    async for chunk in extract_content(mm.complete_with_tools(msgs, toolset=['calculator'])):
        print(chunk, end='')

    print('\n', '='*40, 'END CHECK')

model = sys.argv[1]

# Checking tool registration within the model manager initialization
SQUARE_ROOT_METADATA = {'name': 'square_root',
                            'description': 'Get the square root of the given number',
                            'parameters': {'type': 'object', 'properties': {
                                'square': {'type': 'number', 'description': 'Number from which to find the square root'}},
                                'required': ['square']}}
SQUARE_ROOT_METADATA_CMDLINE_REQUEST = {'type': 'function',
                        'function': SQUARE_ROOT_METADATA, 'pyfunc': 'math|sqrt'}

    # # Pydantic form
    # tools = [V1Function(name='square_root', description='Get the square root of the given number', parameters={'type': 'object', 'properties': {'square': {'type': 'number', 'description': 'Number from which to find the square root'}}, 'required': ['square']})]

# {'name': 'square_root', 'description': 'Get the square root of the given number',
#                     'parameters': {'type': 'object', 'properties': {'square': {'type': 'number',
#                     'description': 'Number from which to find the square root'}}, 'required': ['square']}}

from math import sqrt  # noqa: E402

# each tool is either a callable with built-in metadata, or a tuple of (func, metadata)
TOOLS = [(sqrt, SQUARE_ROOT_METADATA)]

mm = model_manager(model, tool_reg=TOOLS, trace=True)
print('Model type:', mm.model_type)

resp = asyncio.run(amain(mm))

# @click.command()
# @click.option('--model', type=str, help='HuggingFace ID or disk path for locally-hosted MLF format model')
# def main(model):
#     print('XXX', model)
#     mm = model_manager(model)
#     resp = asyncio.run(amain(mm))
