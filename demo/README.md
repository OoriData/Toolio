The root of this directory contains some simple demos. Subdirectories contain more intricate ones.

## `algebra_tutor.py`: Algebra problem-solving tutor

Demonstrates basic [schema-steered structured output (3SO)](https://huggingface.co/blog/ucheog/llm-power-steering)

Run:

```sh
python algebra_tutor.py
```

Based on example from OpenAI, and replicates that simplified JSON Schema. For a more useful version of that schema see `algebra_tutor.schema.json`

- `algebra_tutor.schema.json`: Enhanced version of JSON Schema from `algebra_tutor.py`

* Added a $schema field to specify the JSON Schema draft version
* Included description fields for the overall schema and each section
* Added examples to provide context for what kind of content is expected
* Added a minItems constraint to ensure at least one step is present
* Kept the strict additionalProperties: false to maintain the precise structure

## `arithmetic_calc.py`: Solve an arithmetic problem with tool-calling

Demonstrates basic tool-calling

Run:

```sh
python arithmetic_calc.py
```

## `blind_obedience.py`: Strict LLM obedience

Demonstrates using tool-calling to force an LLM to go against its innate training when generating responses.

Run:

```sh
python blind_obedience.py
```

## `country_extract.py`: Data extraction from unstructured text using 3SO

Run:

```sh
python country_extract.py
```
