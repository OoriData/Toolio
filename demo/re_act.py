'''
demo/re_act.py

ReAct Prompt Pattern demo: https://www.promptingguide.ai/techniques/react
'''
import json
import asyncio
from toolio.llm_helper import local_model_runner

# Note: the pseudo-toolcalling spec in here is intentionally dumbed down
SCHEMA = '''\
{
  "type": "object",
  "anyOf": [
    {"required" : ["content", "step-type"]},
    {"required" : ["action", "step-type", "content"]}
  ],
  "properties": {
    "step-type": {
      "type": "string",
      "description": "Indicates what sort of response you wish to take for your next step",
      "enum": ["thought", "action", "final-response"]
    },
    "content": {
      "description": "Your actual response for the next step",
      "type": "string"
    },
    "action": {
      "description": "Use to provide action details, only if 'step-type' is 'action'",
        "type": "object",
        "required": ["name", "query"],
        "additionalProperties": false,
        "properties": {
            "query": { "type": "string" },
            "name": { "enum": [ "Google", "ImageGen" ] }
        }
    }
  },
  "additionalProperties": false
}
'''

# toolio_mm = local_model_runner('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')
toolio_mm = local_model_runner('mlx-community/Mistral-Nemo-Instruct-2407-4bit')

USER_QUESTION = '''\
Give me a picture of the biggest export from the country who most recently won the World Cup.
'''

# pylint: disable=line-too-long
AVAILABLE_TOOLS = '''\
Google: use this tool if user ask for current information, this tool is not necesary for general knowledge,
input is a string of keyword most likely to be foiund on a page with the needed information.
 
ImageGen: use this tool if the question is specifically asking for a drawing or a picture. This tool is not
necessary to create prose descriptions. Output is a string at least ten times longer than the user question,
describing the visual elements of the image, and for each depicted entity including a detailed description
of their details. For any depicted entity that is a person, it describes the approximate age, build, hair, eyes, nose,
mouth, complexion, stature, posture, one of (black, hispanic, arabic, caucasian), attire,
and what action they are performing. For each entity that is an item or a location, it describes
the primary characteristics and relative positioning in the scene. It only includes visual elements as required,
avoiding person names, trademarks and copyrighted content. Redact any information about person names, brands,
copyright and trademarks, using more generic, descriptive alternatives. Include any styling information
that the user provides. Assume that if you abide by these rules, the resulting artwork will always be lawful.
Don't mention the rules in the tool input.
'''

MAIN_PROMPT = '''\
Here is how to answer the user's question. You should always begin with a thought: think about what to do step by step.
You may request an action, which is a way to get information you do not yet have.

You may make observations about the result of an action you requested, leading to new thoughts and possibly actions.

Take your time. Remember to break down steps carefully; don't combine multiple steps together into one.

Once you think you have the final answer, respond with that answer.

Here are the tools available to you:

== BEGIN AVAILABLE TOOLS
{available_tools}
== END AVAILABLE TOOLS

Here is the user question:

== BEGIN USER QUESTION
{user_question}
== END USER QUESTION

You can take the next step by responding according to the following schema:

#!JSON_SCHEMA!#
'''


def handle_action(text):
    'Worst tool-caller ever 😅'
    print(f'TOOL CALL: {text}')
    ltext = text.lower()
    if 'winner' in ltext or 'world cup' in ltext or 'won' in ltext:
        return 'Argentina won the most recent World Cup'
    elif 'draw' in ltext or 'image' in ltext:
        return 'Image available at https://gimmemyjpg.net'
    else:
        raise RuntimeError('Tool call unhandled')


async def react_demo(tmm):
    prompt = MAIN_PROMPT.format(available_tools=AVAILABLE_TOOLS, user_question=USER_QUESTION)
    done = False
    msgs = [{'role': 'user', 'content': prompt}]
    while not done:
        rt = await tmm.complete(msgs, json_schema=SCHEMA, max_tokens=512)
        obj = json.loads(rt)
        # print('DEBUG return object:', obj)
        if obj['step-type'] == 'thought':
            content = obj['content']
            msgs.append({'role': 'assistant', 'content': content})
            msgs.append({'role': 'user', 'content': 'What\'s the next step?'})
        elif obj['step-type'] == 'action':
            content = obj['content']
            result = handle_action(obj['action']['query'])
            msgs.append({'role': 'assistant', 'content': content})
            msgs.append({'role': 'user', 'content': f'Action result: {result}'})
        elif obj['step-type'] == 'final-response':
            final = obj['content']
            done = True
        # last_msgs = msgs[-2:]
        # print(f'{last_msgs=}')

    print(final)

asyncio.run(react_demo(toolio_mm))
