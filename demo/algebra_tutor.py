# demo/algebra_tutor.py
'''

Based on OpenAI's exaple from "Introducing Structured Outputs in the API" (August 6, 2024)
https://openai.com/index/introducing-structured-outputs-in-the-api/

Just do:

python algebra_tutor.py
'''

import asyncio
from toolio.llm_helper import local_model_runner
from toolio.common import print_response

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

# toolio_mm = local_model_runner('mlx-community/Mistral-Nemo-Instruct-2407-4bit')
toolio_mm = local_model_runner('mlx-community/Llama-3.2-3B-Instruct-4bit')

async def tutor_main(tmm):
    prompt = ('solve 8x + 31 = 2. Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#')
    msgs = [{'role': 'user', 'content': prompt.format(json_schema=SCHEMA)}]
    await print_response(tmm.iter_complete(msgs, json_schema=SCHEMA, max_tokens=512))

asyncio.run(tutor_main(toolio_mm))
