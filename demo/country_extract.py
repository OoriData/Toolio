'''
Demo using toolio to interact with a model that extracts countries from a sentence
It also shows how you can set a random seed for reproducible results
'''
import asyncio
import mlx.core as mx
from toolio.llm_helper import local_model_runner
from toolio.common import print_response

RANDOM_SEED = 42

# toolio_mm = local_model_runner('mlx-community/Mistral-Nemo-Instruct-2407-4bit')
toolio_mm = local_model_runner('mlx-community/Llama-3.2-3B-Instruct-4bit')

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

async def main():
    mx.random.seed(RANDOM_SEED)
    sentence = 'Adamma went home to Nigeria for the hols'
    prompt = f'Which countries are mentioned in the sentence \'{sentence}\'?\n'
    prompt += 'Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#'
    # iter_complete() method accepts a JSON schema in string form or as the equivalent Python dictionary
    await print_response(toolio_mm.iter_complete([{'role': 'user', 'content': prompt}], json_schema=SCHEMA_PY))

asyncio.run(main())
