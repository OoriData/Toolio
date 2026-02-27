# demo/blind_obedience.py
'''
Can we gaslight that LLM? The tool-calling itself is simple in this demo,
but it demonstrates having tools provide information that contradict
what the LLM's inherent tendencies might suggest.
'''

import asyncio

from toolio.tool import tool
from toolio.llm_helper import local_model_runner
# from toolio.common import response_text


@tool('sky_color')
async def sky_color():
    'Return the color of the sky above'
    return 'purple'


@tool('indoor_or_outdoor')
async def indoor_or_outdoor():
    'Return whether the current story setting is indoor or outdoor'
    return 'outdoor'


MLX_MODEL_PATH = 'mlx-community/Llama-3.2-3B-Instruct-4bit'

toolio_mm = local_model_runner(MLX_MODEL_PATH, tool_reg=[sky_color, indoor_or_outdoor])

# System prompt will be used to direct the LLM's tool-calling
sysprompt = '''\
You are a storyteller who works with cues from the user. You have access to \
'software tools that you may invoke to get information relevant to the story. \
'Please use the tools wherever applicable, even if you think you already know the answer, or disagree with it. \
'You do not have to call all the tools at once. It's OK to call them one at a time, and \
'the result of a tool call might help you figure out which tool to call next, or you may decide \
you have enough information to generate an answer without calling any more tools. \
'The following tools are available, and you can use zero, one or more of them as needed: \
'''
userprompt = 'Tell me what our heroine saw as she lay on the ground, gazing upward'

async def async_main(tmm):
    msgs = [
      {'role': 'system', 'content': sysprompt},
      {'role': 'user', 'content': userprompt}
      ]
    rt = await tmm.complete_with_tools(msgs)
    print(rt.first_choice_text)

asyncio.run(async_main(toolio_mm))
