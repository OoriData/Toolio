import sys
import asyncio

import click

from toolio.llm_helper import model_manager, extract_content

async def amain(mm):
    # Plain chat complete
    msgs = [
        # {'role': 'user', 'content': 'I am thinking of a number between 1 and 10. Guess what it is.'}
        {'role': 'user', 'content': 'Hello! How are you?'}
        ]
    async for chunk in extract_content(mm.chat_complete(msgs)):
        print(chunk, end='')
    
    print('='*80)

    prompt = ('Which countries are mentioned in the sentence \'Adamma went home to Nigeria for the hols\'?'
              'Your answer should be only JSON, according to this schema: {json_schema}')
    schema = ('{"type": "array", "items":'
              '{"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}')
    msgs = [
        {'role': 'user', 'content': prompt}
        ]
    async for chunk in extract_content(mm.chat_complete(msgs, json_schema=schema)):
        print(chunk, end='')

    # mm.chat_complete(msgs, functions=None, stream=True, json_response=None, json_schema=None,
    #                         max_tokens=128, temperature=0.1)

model = sys.argv[1]
mm = model_manager(model)
resp = asyncio.run(amain(mm))

# @click.command()
# @click.option('--model', type=str, help='HuggingFace ID or disk path for locally-hosted MLF format model')
# def main(model):
#     print('XXX', model)
#     mm = model_manager(model)
#     resp = asyncio.run(amain(mm))
