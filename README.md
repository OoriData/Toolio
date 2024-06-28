![toolio_github_header](https://github.com/OoriData/Toolio/assets/12983495/e6b35d7f-4b37-4f77-8dc5-1bafc8befb86)
♪ Come along and ride on a fantastic voyage 🎵, with AI in the passenger seat and a flatbed full of tools.

Toolio is an OpenAI-like HTTP server API implementation which supports structured LLM response generation (e.g. make it conform to a [JSON schema](https://json-schema.org/)). It's also really useful for more reliable tool calling. It's based on the MLX framework for Apple Silicon (e.g. M1/M2/M3/M4 Macs), so that's the only supported platform at present.

Builds on: https://github.com/otriscon/llm-structured-output/

<table><tr>
  <td><a href="https://oori.dev/"><img src="https://www.oori.dev/assets/branding/oori_Logo_FullColor.png" width="64" /></a></td>
  <td>Toolio is primarily developed by the crew at <a href="https://oori.dev/">Oori Data</a>. We offer software engineering services around LLM applications.</td>
</tr></table>

# Running the server

`toolio_server` is a FastAPI program that you can use to host MLX-format LLMs for structured output query, for example, if you are on  you can use the MLX format LLM model `mlx-community/Hermes-2-Theta-Llama-3-8B-4bit` as follows (from the cloned directory of this repository):

```sh
pip install -U .
toolio_server --model=mlx-community/Hermes-2-Theta-Llama-3-8B-4bit
```

This will download the model (a little over 4GB) to your local HuggingFace disk cache, and running it will take up about that much of your unified RAM.

For more on the MLX framework for ML workloads (including LLMs) on Apple Silicon, see the [MLX Notes](https://github.com/uogbuji/mlx-notes) article series. The "Day One" article provides all the context you need for using local LLMs through this project.

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

# Using the command line client

cURL is a pretty raw interface for this, though. For example, you have to parse the resulting response JSON. It's a lot easier to use the more specialized command line client tool `toolio_request`. An example of a very simple data extraction use-case:

```sh
export LMPROMPT='Which countries are ementioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'
export LMSCHEMA='{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}'
toolio_request --apibase="http://127.0.0.1:8000" --prompt=$LMPROMPT --schema=$LMSCHEMA
```

(…and yes, in practice a smaller, specialized entity extraction model might be a better option for a case this simple)

With any decent LLM you should get the following **and no extraneous text cluttering things up!**

```json
[{"name": "Nigeria", "continent": "Africa"}]
```

Or if you have the prompt or schema written to files:

```sh
echo 'Which countries are ementioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}' > /tmp/llmprompt.txt
echo '{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}' > /tmp/countries.schema.json
toolio_request --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --schema-file=/tmp/countries.schema.json
```

## Tool calling

You can also run tool usage (function-calling) prompts, a key technique in LLM agent frameworks. A schema will automatically be generated from the tool specs

```sh
echo 'What'\''s the weather like in Boston today?' > /tmp/llmprompt.txt
echo '{"tools": [{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather in a given location","parameters": {"type": "object","properties": {"location": {"type": "string","description": "City and state, e.g. San Francisco, CA"},"unit": {"type": "string","enum": ["℃","℉"]}},"required": ["location"]}}}], "tool_choice": "auto"}' > /tmp/toolspec.json
toolio_request --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json --max-trips=1
```

You can expect a response such as

```json
The model invoked the following tool calls to complete the response, but there are no permitted trips remaining.
[
  {
    "id": "call_6127176720_1719458192_0",
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "arguments_obj": {
        "location": "Boston, MA",
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
toolio_request --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json
```

We give the LLM a Python function for getting a square root. The OpenAI-style tool spec is extended with `"pyfunc": "math|sqrt"`. This tells Toolio to import the Python built-in `math` model and call the `sqrt` function within it.

Notice there is no `--max-trips=` this time. The default value is `3`, so that's enough to have at least one round-trip to deliver the tool's response to the LLM for further processing. If all goes well with the LLM, you should get a result such as:

```
The square root of 256 is 16.
```

`math.sqrt` is a convenient, simple example. You can specify any function which can already be imported (Toolio won't install any libraries at run time), and you can use imports and attribute lookups with multiple levels, e.g. `path.to.module_to_import|path.to.function`.

# LLM-specific flows

LLMs actually get trained for tool calling, and sometimes get trained to expect different patterns. Toolio supports some flags for adapting the tool flow based on the LLM you're using on the server.

# Python client

You can also query the server from Python code, using `toolio.client_helper.struct_mlx_chat_api`. The command line `pylib/cli/request.py` is just one example of this.

# Credits

* otriscon's [llm-structured-output](https://github.com/otriscon/llm-structured-output/) is the foundation of this package
* [OgbujiPT](https://github.com/OoriData/OgbujiPT) provides the client-side Open-AI-style LLM framework, and also the [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) convention for separating prompt text from code.

# License

Apache 2

# Project name

Named after the legend himself. Best don't pretend you don't know Coolio, fool! Popular rapper from Oakland. ♪*Slide slide, but that's the past, I got something brand new for that a@@.*🎼
