'''
demo/algebra_tutor.py

Based on OpenAI's exaple from "Introducing Structured Outputs in the API" (August 6, 2024)
https://openai.com/index/introducing-structured-outputs-in-the-api/
'''

import asyncio
from toolio.llm_helper import model_manager, extract_content

SCHEMA = '''\
{
        "type": "object",
        "properties": {
          "steps": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "explanation": {
                  "type": "string"
                },
                "output": {
                  "type": "string"
                }
              },
              "required": ["explanation", "output"],
              "additionalProperties": false
            }
          },
          "final_answer": {
            "type": "string"
          }
        },
        "required": ["steps", "final_answer"],
        "additionalProperties": false
}
'''

# toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')
toolio_mm = model_manager('mlx-community/Mistral-Nemo-Instruct-2407-4bit')

async def say_hello(tmm):
    prompt = ('solve 8x + 31 = 2. Your answer should be only JSON, according to this schema: {json_schema}')
    msgs = [{'role': 'user', 'content': prompt.format(json_schema=SCHEMA)}]
    async for chunk in extract_content(tmm.complete(msgs, json_schema=SCHEMA, max_tokens=512)):
        print(chunk, end='')

asyncio.run(say_hello(toolio_mm))