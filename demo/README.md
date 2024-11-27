
## Algebra problem-solving tutor

- `algebra_tutor.py`: Algebra tutor problem-solving code demo.

Based on example from OpenAI, and replicates that simplified JSON Schema. For a more useful version of that schema see `algebra_tutor.schema.json`

- `algebra_tutor.schema.json`: Enhanced version of JSON Schema from `algebra_tutor.py`

* Added a $schema field to specify the JSON Schema draft version
* Included description fields for the overall schema and each section
* Added examples to provide context for what kind of content is expected
* Added a minItems constraint to ensure at least one step is present
* Kept the strict additionalProperties: false to maintain the precise structure
