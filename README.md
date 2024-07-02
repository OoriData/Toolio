![toolio_github_header](https://github.com/OoriData/Toolio/assets/12983495/e6b35d7f-4b37-4f77-8dc5-1bafc8befb86)
♪ Come along and ride on a fantastic voyage 🎵, with AI riding shotgun seat and a flatbed full of tools.

Toolio is an OpenAI-like HTTP server API implementation which supports structured LLM response generation (e.g. make it conform to a [JSON schema](https://json-schema.org/)). It's also really useful for more reliable tool calling. It's based on the MLX framework for Apple Silicon (e.g. M1/M2/M3/M4 Macs), so that's the only supported platform at present.

Call it tool-calling, function-calling, agentic framework based on schema-driven output, or guided generation, or steered response.

Builds on: https://github.com/otriscon/llm-structured-output/

<table><tr>
  <td><a href="https://oori.dev/"><img src="https://www.oori.dev/assets/branding/oori_Logo_FullColor.png" width="64" /></a></td>
  <td>Toolio is primarily developed by the crew at <a href="https://oori.dev/">Oori Data</a>. We offer data pipelines and software engineering services around AI/LLM applications.</td>
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
export LMPROMPT='Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}'
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
echo 'Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: {json_schema}' > /tmp/llmprompt.txt
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

## Libraries of tools (or toolboxes, if you prefer)

The examples above might feel like a bit too much work to use a tool; in particular putting together and sending along the tool-calling spec. In most cases you'll either be reusing tools developed by someone else, or your own special ones. In either case the tool-calling spec for each tool can be bundled for easier use. Toolio comes with a few tools you can use right away, for example. `toolio.tool.math.calculator` is a really simple calculator tool the LLM can use because once again LLMs are really bad at maths. But there's one step required first. Some of the built-in tools use third-party libraries which aren't baseline requirements of Toolio. Install them as follows:

```sh
pip install -Ur requirements-extra.txt
```

Now try a prompt intended to use the calculator tool. To make sure it does, we'll add the `--trace` flag:

```sh
toolio_request --apibase="http://127.0.0.1:8000" --tool=toolio.tool.math.calculator --trace \
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
toolio_request --apibase="http://127.0.0.1:8000" --trace \
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

# Python client

You can also query the server from Python code, using `toolio.client.struct_mlx_chat_api`. The command line `pylib/cli/request.py` is just one example of this.

# Credits

* otriscon's [llm-structured-output](https://github.com/otriscon/llm-structured-output/) is the foundation of this package
* [OgbujiPT](https://github.com/OoriData/OgbujiPT) provides the client-side Open-AI-style LLM framework, and also the [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) convention for separating prompt text from code.

# License

Apache 2

# Project name

Named after the legend himself. Best don't pretend you don't know Coolio, fool! Popular rapper (R.I.P.) from LA. You watched *Cookin' with Coolio*, now it's time to Tool up with Toolio! ♪*Slide slide, but that's the past; I got something brand new for that aßß.*🎼
