![toolio_github_header](https://github.com/OoriData/Toolio/assets/12983495/e6b35d7f-4b37-4f77-8dc5-1bafc8befb86)
♪ Come along and ride on a fantastic voyage 🎵, with AI riding shotgun seat and a flatbed full of tools.

Toolio is an OpenAI-like HTTP server API implementation which supports structured LLM response generation (e.g. make it conform to a [JSON schema](https://json-schema.org/)). It's also really useful for more reliable tool calling. Toolio is based on the MLX framework for Apple Silicon (e.g. M1/M2/M3/M4 Macs), so that's the only supported platform at present.

Call it tool-calling or function-calling, or agentic workflows based on schema-driven output, or guided generation, or steered response.

Builds on: https://github.com/otriscon/llm-structured-output/

<table><tr>
  <td><a href="https://oori.dev/"><img src="https://www.oori.dev/assets/branding/oori_Logo_FullColor.png" width="64" /></a></td>
  <td>Toolio is primarily developed by the crew at <a href="https://oori.dev/">Oori Data</a>. We offer data pipelines and software engineering services around AI/LLM applications.</td>
</tr></table>

The following video, "Toolio in 10 minutes", is an easy way to learn about the project.

[![Toolio in 10 minutes](https://github.com/user-attachments/assets/fc8dda94-326d-426e-a566-ac8ec60be31f)](https://youtu.be/9DpQYbteakc)

<!--
<iframe width="560" height="315" src="https://www.youtube.com/embed/9DpQYbteakc?si=Zy4Cj1v1q9ID07eg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<img width="1268" alt="Toolio in 10 minutes still" src="https://github.com/user-attachments/assets/fc8dda94-326d-426e-a566-ac8ec60be31f">
-->

`toolio_server` is a FastAPI program that you can use to host MLX-format LLMs for structured output query or function-calling. For example to host the MLX format LLM model `mlx-community/Hermes-2-Theta-Llama-3-8B-4bit` as follows (from the cloned directory of this repository):

# Installation and Setup

```sh
pip install -U .
toolio_server --model=mlx-community/Hermes-2-Theta-Llama-3-8B-4bit
```

This will download the model (a little over 4GB) to your local HuggingFace disk cache, and running it will take up about that much of your unified RAM.

For more on the MLX framework for ML workloads (including LLMs) on Apple Silicon, see the [MLX Notes](https://github.com/uogbuji/mlx-notes) article series. The "Day One" article provides all the context you need for using local LLMs with Toolio.

# cURLing the Toolio server

Try out a basic request:

```sh
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
     "messages": [{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
     "temperature": 0.1
   }'
```

This is actually not constraining to any output structure, and is just using the LLM as is. Here is a request that does constrain return structure:

```sh
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
    "response_format": {
        "type": "json_object",
        "schema": "{\"type\": \"object\",\"properties\": {\"number\": {\"type\": \"number\"}}}"
    },
    "temperature": 0.1
   }'
```

# Using the command line client instead

cURL is a pretty raw interface for this, though. For example, you have to parse the resulting response JSON. It's a lot easier to use the more specialized command line client tool `toolio_request`. An example of a very simple data extraction use-case:

```sh
export LMPROMPT='Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'
export LMSCHEMA='{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}'
toolio_request --apibase="http://localhost:8000" --prompt=$LMPROMPT --schema=$LMSCHEMA
```

(…and yes, in practice a smaller, specialized entity extraction model might be a better option for a case this simple)

With any decent LLM you should get the following **and no extraneous text cluttering things up!**

```json
[{"name": "Nigeria", "continent": "Africa"}]
```

Or if you have the prompt or schema written to files:

```sh
echo 'Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}' > /tmp/llmprompt.txt
echo '{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}' > /tmp/countries.schema.json
toolio_request --apibase="http://localhost:8000" --prompt-file=/tmp/llmprompt.txt --schema-file=/tmp/countries.schema.json
```

## Tool calling

You can run tool usage (function-calling) prompts, a key technique in LLM agent frameworks. A schema will automatically be generated from the tool specs, which themselves are based on [JSON Schema](https://json-schema.org/), according to OpenAI conventions.

```sh
echo 'What'\''s the weather like in Boulder today?' > /tmp/llmprompt.txt
echo '{"tools": [{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather in a given location","parameters": {"type": "object","properties": {"location": {"type": "string","description": "City and state, e.g. San Francisco, CA"},"unit": {"type": "string","enum": ["℃","℉"]}},"required": ["location"]}}}], "tool_choice": "auto"}' > /tmp/toolspec.json
toolio_request --apibase="http://localhost:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json --max-trips=1
```

You can expect a response such as

```json
[...] UserWarning: No implementation provided for function: get_current_weather
The model invoked the following tool calls to complete the response, but there are no permitted trips remaining.
[
  {
    "id": "call_6127176720_1719458192_0",
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "arguments_obj": {
        "location": "Boulder, MA",
        "unit": "\u2109"
      }
    }
  }
]
```

You might have noticed the `--max-trips=1` in the original call. Normally the tool call response would go back to the LLM to further construct a response, but Toolio allows you to limit those trips. By setting the limit to 1, it is unable to make a second trip to deliver the function call response for further processing, and the user is notified of the fact.

Incidentally `\u2109` is just Unicode for `℉` (degrees fahrenheit).

## Actually running the functions

It's pretty well known at this point that LLMs are bad at maths, but we can give them help. Consider the following example:

```sh
echo 'What is the square root of 256?' > /tmp/llmprompt.txt
echo '{"tools": [{"type": "function","function": {"name": "square_root","description": "Get the square root of the given number","parameters": {"type": "object", "properties": {"square": {"type": "number", "description": "Number from which to find the square root"}},"required": ["square"]},"pyfunc": "math|sqrt"}}], "tool_choice": "auto"}' > /tmp/toolspec.json
toolio_request --apibase="http://localhost:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json
```

We give the LLM a Python function for getting a square root. The OpenAI-style tool spec is extended with `"pyfunc": "math|sqrt"`. This tells Toolio to import the Python built-in `math` model and call the `sqrt` function within it.

Notice there is no `--max-trips=` this time. The default value is `3`, so that's enough to have at least one round-trip to deliver the tool's response to the LLM for further processing. If all goes well with the LLM, you should get a result such as:

```
The square root of 256 is 16.
```

`math.sqrt` is a convenient, simple example. You can specify any function which can already be imported (Toolio won't install any libraries at run time), and you can use imports and attribute lookups with multiple levels, e.g. `path.to.module_to_import|path.to.function`.

## Libraries of tools (or toolboxes, if you prefer)

The examples above might feel like a bit too much work to use a tool; in particular putting together and sending along the tool-calling spec. In most cases you'll either be reusing tools developed by someone else, or your own special ones. In either case the tool-calling spec for each tool can be bundled for easier use. Toolio comes with a few tools you can use right away, for example. `toolio.tool.math.calculator` is a really simple calculator tool the LLM can use because once again LLMs are really bad at maths. But there's one step required first. Some of the built-in tools use third-party libraries which aren't baseline requirements of Toolio. Install them as follows:

```sh
pip install -Ur requirements-extra.txt
```

Now try a prompt intended to use the calculator tool. To make sure it does, we'll add the `--trace` flag:

```sh
toolio_request --apibase="http://localhost:8000" --tool=toolio.tool.math.calculator --trace \
--prompt='Usain Bolt ran the 100m race in 9.58s. What was his average velocity?' 
```

Here's what I got from `Hermes-2-Theta-Llama-3-8B-4bit`:

```
⚙️Calling tool calculator with args {'expr': '100/9.58'}
⚙️Tool call result: 10.438413361169102
Final response:
To calculate Usain Bolt's average velocity, we need to know the distance he covered (100m) and the time it took him to cover that distance (9.58s). 

Average velocity is defined as the total distance traveled divided by the time taken. In this case, the total distance traveled is 100m, and the time taken is 9.58s. 

So, Usain Bolt's average velocity is:

v_avg = distance / time
v_avg = 100m / 9.58s
v_avg = 10.438413361169102 m/s

Therefore, Usain Bolt's average velocity during the 100m race was approximately 10.44 m/s.
```

You can see that the LLM got help by calling the tool to calculate `100/9.58`.

Note: Every tool relies on the agent LLM to correctly construct the tool call call, e.g. settign up the right mathematial expression for the calculator tool. This is not something you can take for granted, so there's no shortcut from testing and selecting the right LLMs.

## Multiple tool calls

Here's an example of giving the LLM a tool to get today's date, and another with a database lookup from birthdays to employee names and interests.

```sh
toolio_request --apibase="http://localhost:8000" --trace \
--tool=toolio.tool.demo.birthday_lookup \
--tool=toolio.tool.demo.today_kfabe \
--sysprompt='You are a writer who reasons step by step and uses research tools in the correct order before writing' \
--prompt='Write a nice note for each employee who has a birthday today.'
```

These are actually contrived, fake tools for demo purposes. `demo.today_kfabe` always gives the date as 1 July 2024, and `demo.birthday_lookup` is a dummy database. Also note the added system prompt to encourag the LLM to use step-by-step reasoning in applying the tools. If your LLM is smart enough enough it would first get the (supposed) date today and then convrt that to a format suitable for the database lookip.

Unfortunately `mlx-community/Hermes-2-Theta-Llama-3-8B-4bit` fumbles this, ignoring the spoon-fed date from the first tool call, and instead grabs an example date mentioned in the tool definition. This results in no birthday lookup results, and the LLM generates no output.

```
⚙️Calling tool today with args {}
⚙️Tool call result: 07-01
⚙️Calling tool birthday_lookup with args {'date': '05-03'}
⚙️Tool call result: No one has a birthday today
Final response:

```

It's a good example of how tool-calling can pretty easily go wrong. As LLMs get more and more capable this should become more reliable. It may well be that top-end LLMs such as OpenAI's GPT and Anthropic's Claude would be able to handle this case, but of course you can't run these privately on MLX.

# Write your own tools

Study the examples in the `pylib/tools` directory to see how easy it is.

# LLM-specific flows

LLMs actually get trained for tool calling, and sometimes get trained to expect different patterns. Toolio supports some flags for adapting the tool flow based on the LLM you're using on the server.

For notes on more models see https://github.com/OoriData/Toolio/wiki/Notes-on-how-MLX-models-handle-tool%E2%80%90calling

# Python HTTP client

You can also query the server from Python code, using `toolio.client.struct_mlx_chat_api`. Here's an example, including a (dummied up) custom tool:

```py
import asyncio

from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

from toolio.client import struct_mlx_chat_api
from toolio.tool import tool, param

@tool('currency_exchange', params=[param('from', str, 'Currency to be converted from, e.g. USD, GBP, JPY', True, rename='from_'), param('to', str, 'Currency to be converted to, e.g. USD, GBP, JPY', True), param('amount', float, 'Amount to convert from one currency to another. Just a number, with no other symbols', True)])
def currency_exchange(from_=None, to=None, amount=None):
    'Tool to convert one currency to another'
    # Just a dummy implementation
    lookup = {('JPY', 'USD'): 1234.56}
    rate = lookup.get((from_, to))
    print(f'{from_=}, {to=}, {amount=}, {rate=}')
    # Look up the conversion online here
    return rate * amount

prompt = 'I need to import a car from Japan. It costs 5 million Yen.'
'How much must I withdraw from my US bank account'
llm = struct_mlx_chat_api(base_url='http://localhost:8000', tool_reg=[currency_exchange], trace=True)
resp = asyncio.run(llm(prompt_to_chat(prompt), trip_timeout=60))
print(resp.first_choice_text)
```

Notice the use of the `rename` parameter metadata. In Python the param name we've asked the LLM to use, `from`, is a keyword, so to avoid confusion the actual function definition uses `from_`, and the `rename` instructs Toolio to make that change in the background.

You can also define asynchronous tools, e.g. `async def currency_exchange`, which I would actually recommend if, e.g. you are truly web scraping.

You might study the command line `pylib/cli/request.py` for further insight.

# Direct usage via Python

You can also, of course, just load the model and run inference on it without bothering with HTTP client/server. The `model_manager` class is a convenient interface for this.

```py
import asyncio
from toolio.llm_helper import model_manager, extract_content

toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')

async def say_hello(tmm):
    msgs = [{"role": "user", "content": "Hello! How are you?"}]
    async for chunk in extract_content(tmm.complete(msgs)):
        print(chunk, end='')

asyncio.run(say_hello(toolio_mm))
```

You should just get a simple text response from the LLm printed to the screen.

You can also do this via synchronous API, but I highly recommend leaing hard on the async habit.

The `chat_complete` method also takes a list of tools or a JSON schema, as well as some model parameters.

## LLM response metadata

Toolio uses OpenAI API conventions a lot under the hood. If you run the following:

```py
import asyncio
from toolio.llm_helper import model_manager, extract_content

toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')

async def say_hello(tmm):
    msgs = [{"role": "user", "content": "Hello! How are you?"}]
    async for chunk_struct in tmm.complete(msgs):
        print(chunk_struct)
        break

asyncio.run(say_hello(toolio_mm))
```

You should see something like:

```py
{'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Hi'}, 'finish_reason': None}], 'object': 'chat.completion.chunk', 'id': 'chatcmpl-17588006160_1721823730', 'created': 1721823730, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'}
```

The LLM response is delivered in such structures ("deltas") as they're generated. `chunk_struct['choices'][0]['delta']['content']` is a bit of the actual text we teased out in the previous snippet. `chunk_struct['choices'][0]['finish_reason']` is `None` because it's not yet finished, etc. This is based on OpenAI API.

`extract_content`, used in the previous snippet, is a very simple coroutine that extracts the actual text content from this series of response structures.

The final chunk would look something like this:

```py
{'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 20, 'prompt_tokens': 12, 'total_tokens': 32}, 'object': 'chat.completion.chunk', 'id': 'chatcmpl-18503717840_1721824385', 'created': 1721824385, 'model': 'mlx-community/Hermes-2-Theta-Llama-3-8B-4bit'}
```

Notice there is more information, now that it's finished (`'finish_reason': 'stop'`). Say you want the metadata such as the number of tokens generated:

```py
import asyncio
from toolio.llm_helper import model_manager, extract_content

toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')

async def say_hello(tmm):
    msgs = [{"role": "user", "content": "Hello! How are you?"}]
    async for chunk in tmm.complete(msgs):
        content = chunk['choices'][0]['delta']['content']
        if content is not None:
            print(content, end='')

    # Final chunk has the stats
    print('\n', '-'*80, '\n', 'Number of tokens generated:', chunk['usage']['total_tokens'])

asyncio.run(say_hello(toolio_mm))
```

You'll get something like:

```
*waves* Hi there! I'm doing well, thank you for asking. How about you?
 -------------------------------------------------------------------------------- 
 Number of tokens generated: 32
```

Tip: don't forget all the various, useful bits to be found in `itertools` and the like.

# Structured LLM responses via direct API

As mentioned, you can specify tools and schemata.

```py
import asyncio
from toolio.llm_helper import model_manager, extract_content

toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')

async def say_hello(tmm):
    prompt = ('Which countries are mentioned in the sentence \'Adamma went home to Nigeria for the hols\'?'
              'Your answer should be only JSON, according to this schema: {json_schema}')
    schema = ('{"type": "array", "items":'
              '{"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}')
    msgs = [{'role': 'user', 'content': prompt.format(json_schema=schema)}]
    async for chunk in extract_content(tmm.complete(msgs, json_schema=schema)):
        print(chunk, end='')

asyncio.run(say_hello(toolio_mm))
```

## Example of tool use

```py
import asyncio
from math import sqrt
from toolio.llm_helper import model_manager, extract_content

SQUARE_ROOT_METADATA = {'name': 'square_root', 'description': 'Get the square root of the given number',
                            'parameters': {'type': 'object', 'properties': {
                                'square': {'type': 'number',
                                'description': 'Number from which to find the square root'}},
                            'required': ['square']}}
toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit',
                          tool_reg=[(sqrt, SQUARE_ROOT_METADATA)], trace=True)


async def query_sq_root(tmm):
    msgs = [ {'role': 'user', 'content': 'What is the square root of 256?'} ]    
    async for chunk in extract_content(tmm.complete_with_tools(msgs)):
        print(chunk, end='')

asyncio.run(query_sq_root(toolio_mm))
```

# Tweaking prompts

Part of the process of getting an LLM to stick to a schema, or to call tools is to give it a system prompt to that effect. Toolio has built in prompt language for this purpose. We believe strongly in the design principle of separating natural language (e.g. prompts) from code, so the latyter is packaged into the `resource/language.toml` file, using [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) conventions.

You can of course override the built-in prompting.

## Overriding the tool-calling system prompt from the command line

```sh
echo 'What is the square root of 256?' > /tmp/llmprompt.txt        
echo '{"tools": [{"type": "function","function": {"name": "square_root","description": "Get the square root of the given number","parameters": {"type": "object", "properties": {"square": {"type": "number", "description": "Number from which to find the square root"}},"required": ["square"]},"pyfunc": "math|sqrt"}}], "tool_choice": "auto"}' > /tmp/toolspec.json
toolio_request --apibase="http://localhost:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json --sysprompt="You are a helpful assistant with access to a tool that you may invoke if needed to answer the user's request. Please use the tool as applicable, even if you think you already know the answer. Give your final answer in Shakespearean English The tool is:
Tool"
```

## Overriding the tool-calling system prompt from the Python API

In order to override the system prompt from code, just se it in the initial chat message as the `system` role.

```py
import asyncio
from math import sqrt
from toolio.llm_helper import model_manager, extract_content

SQUARE_ROOT_METADATA = {'name': 'square_root', 'description': 'Get the square root of the given number',
                            'parameters': {'type': 'object', 'properties': {
                                'square': {'type': 'number',
                                'description': 'Number from which to find the square root'}},
                            'required': ['square']}}
toolio_mm = model_manager('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit',
                          tool_reg=[(sqrt, SQUARE_ROOT_METADATA)], trace=True)

# System prompt will be used to direct the LLM's tool-calling
SYSPROMPT = '''You are a tutor from Elizabethan England, with access a tool that you may invoke if needed to answer the user's request. Please use the tool as applicable, even if you think you already know the answer. Remember to give your final answer in Elizabethan English The tool is:
Tool
'''

async def query_sq_root(tmm):
    msgs = [
      {'role': 'system', 'content': SYSPROMPT},
      {'role': 'user', 'content': 'What is the square root of 256?'}
      ]
    async for chunk in extract_content(tmm.complete_with_tools(msgs)):
        print(chunk, end='')

asyncio.run(query_sq_root(toolio_mm))
```

In which case you can express a response such as:

```
Good sir or madam, the square root of 256 is indeed 16. Mayhap thou wouldst like to know more of this wondrous number? I am at thy service.
```

# More examples

See the `demo` directory.

# Credits

* otriscon's [llm-structured-output](https://github.com/otriscon/llm-structured-output/) is the foundation of this package
* [OgbujiPT](https://github.com/OoriData/OgbujiPT) provides the client-side Open-AI-style LLM framework, and also the [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) convention for separating prompt text from code.

# License

Apache 2

# Nearby projects

* [Instructor](https://github.com/jxnl/instructor) - LLM structured output via prompt engineering rather than steered sampling.
* [Outlines](https://github.com/outlines-dev/outlines) - Structured Text Generation vis Pydantic, JSON schema or EBNF. Seems to be sampling control.

# Why this, anyway?

In our thinking, and that of many others working in the space for a while, agent/tool systems are where GenAI are most likely to deliver practical value. Watch out, though, because McKinsey has seen fit to apply their $1,000/hr opinions along the same lines. ["Why agents are the next frontier of generative AI"](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/why-agents-are-the-next-frontier-of-generative-ai?cid=soc-web) (July 2024)

[Parrot/Gorilla cartoon here]

# Project name

Named after the legend himself. Best don't pretend you don't know Coolio, fool! Popular rapper (R.I.P.) from LA. You watched *Cookin' with Coolio*, now it's time to Tool up with Toolio! ♪*Slide slide, but that's the past; I got something brand new for that aß.*🎼
