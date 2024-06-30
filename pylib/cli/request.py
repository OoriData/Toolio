# toolio.cli.request
'''
Sample of a very simple data extraction use-case

…and yes, in practice a smaller, specialized entity extraction model might be better for this

```sh
export LMPROMPT='Which countries are mentioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'
export LMSCHEMA='{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}'
toolio_request --apibase="http://127.0.0.1:8000" --prompt=$LMPROMPT --schema=$LMSCHEMA --max-tokens 1000
```

With any decent LLM you should get the following **and no extraneous text cluttering things up!**

```json
[{"name": "Nigeria", "continent": "Africa"}]
```

Or if you have the prompt or schema written to files:

```sh
echo 'Which countries are ementioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}' > /tmp/llmprompt.txt
echo '{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}' > /tmp/countries.schema.json
toolio_request --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --schema-file=/tmp/countries.schema.json --max-tokens 1000
```

You can also try tool usage (function-calling) prompts. A schema will automatically be generated from the tool specs

```sh
echo 'What'\''s the weather like in Boston today?' > /tmp/llmprompt.txt
echo '{"tools": [{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather in a given location","parameters": {"type": "object","properties": {"location": {"type": "string","description": "City and state, e.g. San Francisco, CA"},"unit": {"type": "string","enum": ["℃","℉"]}},"required": ["location"]}}}], "tool_choice": "auto"}' > /tmp/toolspec.json
toolio_request --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json --max-tokens 1000
```

You can expect a response such as

```json
The model has invoked the following tool calls in response to the prompt:
[
  {
    "id": "call_17705268944_1719102607_0",
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"\\u2109\"}",
      "arguments_obj": {
        "location": "Boston, MA",
        "unit": "\u2109"
      }
    }
  }
]
```

'''
import sys
import json
import asyncio
import importlib

import click
from ogbujipt.llm_wrapper import prompt_to_chat

from toolio.client import struct_mlx_chat_api, response_type


@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000',
    help='MLXStructuredLMServer (OpenAI API-compatible) server base URL')
@click.option('--prompt',
    help='Prompt text; can use {jsonschema} placeholder for the schema')
@click.option('--prompt-file', type=click.File('r'),
    help='Prompt text; can use {jsonschema} placeholder for the schema. Overrides --prompt arg')
@click.option('--schema',
    help='JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
@click.option('--schema-file', type=click.File('rb'),
    help='Path to JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --schema arg')
# @click.option('--schema',
#     help='JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
# @click.option('--schema-file',
#     help='Path to JSON schema to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --schema arg')
@click.option('--tools',
    help='Tools specification, based on OpenAI format, to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt')
@click.option('--tools-file', type=click.File('rb'),
    help='Path to tools specification based on OpenAI format, to be sent along in prompt to constrain the response. Also interpolated into {jsonschema} placeholder in the prompt. Overrides --tools arg')
@click.option('--sysprompt', help='Optional system prompt')
@click.option('--max-trips', default='3', type=int,
    help='Maximum number of times to return to the LLM, presumably with tool results. If there is no final response by the time this is reached, post a message with the remaining unused tool invocations')
@click.option("--max-tokens", type=int, help='Maximum number of tokens to generate. Will be applied to each trip')

@click.option('--tool', '-t', multiple=True, help='Full Python attribute path to a Toolio-specific callable to be made available to the LLM')

@click.option('--model', type=str, help='Path to locally-hosted MLX format model')
@click.option('--temp', default='0.1', type=float, help='LLM sampling temperature')

@click.option('--trace', is_flag=True, default=False,
              help='Print information (to STDERR) about tool call requests & results. Useful for debugging')
def main(apibase, prompt, prompt_file, schema, schema_file, tools, tools_file, tool, sysprompt, max_trips, max_tokens,
         model, temp, trace):
    if prompt_file:
        prompt = prompt_file.read()
    if not prompt:
        raise RuntimeError('--prompt or --prompt-file required')
    if schema_file:
        schema_obj = json.load(schema_file)
    elif schema:
        schema_obj = json.loads(schema)
    else:
        schema_obj = None
    if tools_file:
        tools_obj = json.load(tools_file)
    elif tools:
        tools_obj = json.loads(tools)
    else:
        tools_obj = None

    # Import & register any tools
    tool_callables = []
    for tpath in tool:
        modpath, call_name  = tpath.rsplit('.', 1)
        modobj = importlib.import_module(modpath)
        tool_callables.append(getattr(modobj, call_name))

    llm = struct_mlx_chat_api(base_url=apibase, tools=tool_callables, trace=trace)
    resp = asyncio.run(llm(prompt_to_chat(prompt, system=sysprompt), schema=schema_obj, tools=tools_obj, max_trips=max_trips))
    if resp['response_type'] == response_type.TOOL_CALL:
        print('The model invoked the following tool calls to complete the response, but there are no permitted trips remaining.')
        tcs = resp['choices'][0]['message']['tool_calls']
        for tc in tcs:
            del tc['function']['arguments']
        print(json.dumps(tcs, indent=2))
    elif resp['response_type'] == response_type.MESSAGE:
        if trace:
            print('Final response:', file=sys.stderr)
        print(resp.first_choice_text)
