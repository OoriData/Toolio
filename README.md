# MLXStructuredLMServer

OpenAI-like HTTP server API implementation which supports structured LLM response generation (e.g. make it conform to a [JSON schema](https://json-schema.org/)). It's also really useful for more reliable tool calling.

Builds on: https://github.com/otriscon/llm-structured-output/

# Running the server

`MLXStructuredLMServer` is a FastAPI program that you can use to host MLX-format LLMs for structured output query, for example, if you are on M1/M2/M3/M4 Mac you can use the MLX format LLM model `mlx-community/Hermes-2-Theta-Llama-3-8B-4bit` as follows (from the cloned directory of this repository):

```sh
pip install -U .
MLXStructuredLMServer --model=mlx-community/Hermes-2-Theta-Llama-3-8B-4bit
```

This will download the model (a little over 4GB) to your local HuggingFace disk cache, and running it will take up ablut that much of your unified RAM.

For more on the MLX framework for ML workloads (including LLMs) on Apple Silicon, see the [MLX Notes](https://github.com/uogbuji/mlx-notes) article series. The "Day One" article provides all the context you need for using local LLMs through this project.

You can try out a basic request:

```sh
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
     "messages": [{"role": "user", "content": "I am thinking of a number between 1 and 10. Guess what it is."}],
     "temperature": 0.1
   }'
```

This is actually not applying any output structure, and is just using the LLM as is. Here is a request that does constrain return structure:

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

cURL is a pretty raw interface for this, though. For example, you have to parse the resulting response JSON. It's a lot easier to use the more specialized command line client tool `MLXStructuredLMQRequest`. An example of a very simple data extraction use-case:

```sh
export LMPROMPT='Which countries are ementioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'
export LMSCHEMA='{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}'
MLXStructuredLMQRequest --apibase="http://127.0.0.1:8000" --prompt=$LMPROMPT --schema=$LMSCHEMA
```

(…and yes, in practice a smaller, specialized entity extraction model might be better for this)

With any decent LLM you should get the following **and no extraneous text cluttering things up!**

```json
[{"name": "Nigeria", "continent": "Africa"}]
```

Or if you have the prompt or schema written to files:

```sh
echo 'Which countries are ementioned in the sentence "Uche went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}' > /tmp/llmprompt.txt
echo '{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}}}' > /tmp/countries.schema.json
MLXStructuredLMQRequest --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --schema-file=/tmp/countries.schema.json
```

You can also try tool usage (function-calling) prompts. A schema will automatically be generated from the tool specs

```sh
echo 'What'\''s the weather like in Boston today?' > /tmp/llmprompt.txt
echo '{"tools": [{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather in a given location","parameters": {"type": "object","properties": {"location": {"type": "string","description": "City and state, e.g. San Francisco, CA"},"unit": {"type": "string","enum": ["℃","℉"]}},"required": ["location"]}}}], "tool_choice": "auto"}' > /tmp/toolspec.json
MLXStructuredLMQRequest --apibase="http://127.0.0.1:8000" --prompt-file=/tmp/llmprompt.txt --tools-file=/tmp/toolspec.json
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

# Python client

You can also query the server from Python code, using `mlx_struct_lm_server.client_helper.struct_mlx_chat_api`. `pylib/cli/request.py` is an example of this.

# Credits

* otriscon's [llm-structured-output](https://github.com/otriscon/llm-structured-output/) is the foundation of this package
* [OgbujiPT](https://github.com/OoriData/OgbujiPT) provides the client-side Open-AI-style LLM framework, and also the [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) convention for separating prompt text from code.

# License

Apache 2

