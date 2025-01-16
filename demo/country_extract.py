'''
Demo using toolio to interact with a model that extracts countries from a sentence
It also shows how you can set a random seed for reproducible results
'''
import asyncio
import mlx.core as mx
from toolio.llm_helper import local_model_runner

RANDOM_SEED = 42

toolio_mm = local_model_runner('mlx-community/Mistral-Nemo-Instruct-2407-4bit')

SCHEMA_PY = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'continent': {'type': 'string'}
        },
        'required': ['name', 'continent']
    }
}

async def say_hello(tmm):
    mx.random.seed(RANDOM_SEED)
    sentence = 'Adamma went home to Nigeria for the hols'
    prompt = f'Which countries are mentioned in the sentence \'{sentence}\'?\n'
    prompt += 'Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#'
    # The complete() method accepts a JSON schema in string form or as the equivalent Python dictionary
    print(await tmm.complete([{'role': 'user', 'content': prompt}], json_schema=SCHEMA_PY))

asyncio.run(say_hello(toolio_mm))
