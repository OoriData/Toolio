# demo/re_act.py
'''
ReAct Prompt Pattern demo: https://www.promptingguide.ai/techniques/react

Demonstrates using toolio to interact with a model that follows the ReAct Prompt Pattern,
which is a bit out of date now that you can have most models do thinking and tool-calling internally.

Note: Smaller models (like Llama-3.2-3B) may struggle with the requirements of the ReAct Prompt Pattern:
- May use step-types not in the enum (e.g., "observation", "request")
- May generate responses that exceed token limits
- May produce malformed JSON when truncated

Run:

```sh
python re_act.py
```
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

# toolio_mm = local_model_runner('mlx-community/Mistral-Nemo-Instruct-2407-4bit')
toolio_mm = local_model_runner('mlx-community/Llama-3.2-3B-Instruct-4bit')

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


def handle_action(tool_name, query):
    'Worst tool-caller ever 😅'
    print(f'TOOL CALL: {tool_name} with query: {query}')
    ltext = query.lower()

    if tool_name == 'Google':
        if 'winner' in ltext or 'world cup' in ltext or 'won' in ltext:
            return 'Argentina won the most recent World Cup'
        elif 'export' in ltext or 'biggest' in ltext:
            # Handle export queries - Argentina's biggest export is soybeans
            if 'argentina' in ltext:
                return 'Argentina\'s biggest export is soybeans and soybean products'
            else:
                return 'You need to find out which country won the World Cup first'
        else:
            return f'Google search for "{query}" returned some results'
    elif tool_name == 'ImageGen':
        # Always return a mock image URL for ImageGen calls
        return 'Image available at https://gimmemyjpg.net'
    else:
        raise RuntimeError(f'Unknown tool: {tool_name}')


async def react_demo(tmm):
    prompt = MAIN_PROMPT.format(available_tools=AVAILABLE_TOOLS, user_question=USER_QUESTION)
    done = False
    final = None
    msgs = [{'role': 'user', 'content': prompt}]
    loop_count = 0
    max_loops = 10  # Prevent infinite loops

    while not done and loop_count < max_loops:
        loop_count += 1
        print(f'[Loop {loop_count}] Calling model...')
        try:
            # Note: Smaller models may struggle with strict JSON schema adherence
            # Increasing max_tokens helps but they may still use non-enum step-types
            rt = await asyncio.wait_for(
                tmm.complete(msgs, json_schema=SCHEMA, max_tokens=1024),
                timeout=60.0  # 60 second timeout per call
            )
            print(f'[Loop {loop_count}] Got response: {rt[:200]}...')
        except asyncio.TimeoutError:
            print(f'[Loop {loop_count}] Timeout waiting for model response')
            break
        except Exception as e:
            print(f'[Loop {loop_count}] Error: {e}')
            break

        try:
            obj = json.loads(rt)
            print(f'[Loop {loop_count}] Parsed object: step-type={obj.get("step-type")}')
        except json.JSONDecodeError as e:
            print(f'[Loop {loop_count}] Failed to parse JSON (likely truncated): {e}')
            # Try to extract partial info if JSON is truncated
            if '"step-type"' in rt:
                import re
                match = re.search(r'"step-type"\s*:\s*"([^"]+)"', rt)
                if match:
                    step_type = match.group(1).lower()
                    print(f'[Loop {loop_count}] Detected step-type: {step_type} (from truncated JSON)')
                    # If it's an observation/thought/request (without action), we can continue
                    if step_type in ['observation', 'thought', 'request']:
                        print(f'[Loop {loop_count}] Treating truncated response as thought/observation')
                        msgs.append({'role': 'assistant', 'content': rt[:500] + '... (truncated)'})
                        msgs.append({'role': 'user', 'content': 'Your response was cut off. Please provide a shorter response. What\'s the next step?'})
                        continue
            print(f'[Loop {loop_count}] Response was: {rt[:300]}...')
            break

        step_type = obj.get('step-type', '').lower()

        # Handle model variations:
        # - "request" with action -> "action"
        # - "request" without action -> "thought" (model is asking/thinking, not calling a tool)
        # - "observation" -> "thought"
        if step_type == 'request':
            if 'action' in obj:
                step_type = 'action'
            else:
                step_type = 'thought'  # Request without action is just a thought/question
        elif step_type == 'observation':
            step_type = 'thought'  # Treat observations as thoughts

        if step_type == 'thought':
            content = obj['content']
            msgs.append({'role': 'assistant', 'content': content})
            msgs.append({'role': 'user', 'content': 'What\'s the next step?'})
        elif step_type == 'action':
            if 'action' not in obj:
                print(f'[Loop {loop_count}] Action step-type but no action object found')
                break
            content = obj['content']
            tool_name = obj['action'].get('name', 'Unknown')
            query = obj['action'].get('query', '')
            result = handle_action(tool_name, query)
            msgs.append({'role': 'assistant', 'content': content})
            msgs.append({'role': 'user', 'content': f'Action result: {result}'})
        elif step_type == 'final-response':
            final = obj['content']
            done = True
        else:
            print(f'[Loop {loop_count}] Unexpected step-type: {obj.get("step-type")}')
            print(f'[Loop {loop_count}] Full object: {json.dumps(obj, indent=2)}')
            break

    if done:
        print(final)
    else:
        print(f'Demo ended without final response (loop_count={loop_count})')

asyncio.run(react_demo(toolio_mm))
