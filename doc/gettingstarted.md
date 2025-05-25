Getting Started with Toolio

Toolio is an OpenAI-like HTTP server API implementation that supports structured LLM response generation and reliable tool calling. It's built on the MLX framework for Apple Silicon, making it exclusive to Mac platforms with M1/M2/M3/M4 chips.

## Prerequisites

- Apple Silicon Mac (M1, M2, M3, or M4)
  - Note: You can install Toolio on other OSes, for example to use the client library or `toolio_request` to access a Toolio server. In this case, you will not be able use any features which involve loading LLMs.
- Python 3.10 or more recent (tested only on 3.11 or more recent)

To verify you're on an Apple Silicon Mac you can run:

```sh
python -c "import platform; assert 'arm64' in platform.platform()"
```

## Installation

Install Toolio using pip:

```sh
pip install toolio
```

For some built-in tools, you'll need additional dependencies:

```sh
pip install -Ur requirements-extra.txt
```

## Quick Start

### 1. Host a Toolio Server

Launch a Toolio server using an MLX-format LLM:

```sh
toolio_server --model=mlx-community/Llama-3.2-3B-Instruct-4bit
```

This command downloads and hosts the specified model.

### 2. Make a Basic Request

Use the `toolio_request` command-line tool:

```sh
toolio_request --apibase="http://localhost:8000" --prompt="I am thinking of a number between 1 and 10. Guess what it is."
```

### 3. Use Structured Output

Constrain the LLM's output using a JSON schema:

```sh
export LMPROMPT='Which countries are mentioned in the sentence "Adamma went home to Nigeria for the hols"? Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#'
export LMSCHEMA='{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "continent": {"type": "string"}}, "required": ["name", "continent"]}}'
toolio_request --apibase="http://localhost:8000" --prompt=$LMPROMPT --schema=$LMSCHEMA
```

Notice the `#!JSON_SCHEMA!#` cutout, which Toolio replaces for you with the actual schema you've provided.

### 4. Tool Calling

Use built-in or custom tools:

```sh
toolio_request --apibase="http://localhost:8000" --tool=toolio.tool.math.calculator --loglevel=DEBUG \
--prompt='Usain Bolt ran the 100m race in 9.58s. What was his average velocity?'
```

### 5. Python API Usage

Use Toolio directly in Python:

```python
import asyncio
from toolio.llm_helper import model_manager

toolio_mm = model_manager('mlx-community/Llama-3.2-3B-Instruct-4bit')

async def say_hello(tmm):
    msgs = [{"role": "user", "content": "Hello! How are you?"}]
    print(await tmm.complete(msgs))

asyncio.run(say_hello(toolio_mm))
```

### 6. Iterative Python API Usage

Or the same ting, but iteratively getting chunks of the results, where supported:

```python
import asyncio
from toolio.llm_helper import model_manager, extract_content

toolio_mm = model_manager('mlx-community/Llama-3.2-3B-Instruct-4bit')

async def say_hello(tmm):
    msgs = [{"role": "user", "content": "Hello! How are you?"}]
    async for chunk in extract_content(tmm.iter_complete(msgs)):
        print(chunk, end='')

asyncio.run(say_hello(toolio_mm))
```

## Next Steps

- Check out the `demo` directory for more examples
- Explore creating custom tools
- Learn about LLM-specific flows and flags
