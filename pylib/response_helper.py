# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.response_helper
'''
Converting GenerationResponse objects to higher-level data pattrns, including as useful for OpenAI-style API responses

Example usage:

```python
resp = llm_response.from_generation_response(
    generation_response,
    model_name='local/model/path',
    model_type='llama'
)

# Get response text
text = resp.first_choice_text

# Convert to dict/JSON
resp_dict = resp.to_dict()
resp_json = resp.to_json()
```
'''
import time
import json
# import warnings
from dataclasses import dataclass, field
from enum import Enum, auto

from toolio.common import (llm_response_type, model_flag,
                           TOOLIO_MODEL_TYPE_FIELD, TOOLIO_MODEL_FLAGS_FIELD, FLAGS_LOOKUP, DEFAULT_FLAGS)
from toolio.toolcall import tool_call_response_mixin, parse_genresp_tool_calls, tool_call

class response_state(Enum):
    '''States for tracking response generation'''
    INIT = auto()
    GATHERING_TOOL_CALLS = auto()
    GATHERING_MESSAGE = auto()
    COMPLETE = auto()


# XXX: Candidate for replacing the llm_response class in ogbujipt
@dataclass
class llm_response(tool_call_response_mixin):
    '''
    Uniform interface for LLM responses from OpenAI APIs, MLX, etc.
    '''
    # dataclass behavior: fields declared here become parameters to __init__ in the order listed
    response_type: llm_response_type  # Required parameter
    choices: list                     # Required parameter
    usage: dict = None                # Optional parameter with default
    object: str = None                # Optional parameter with default
    id: str = None
    created: int = None
    model: str = None
    model_type: str = None
    model_flags: model_flag = DEFAULT_FLAGS

    _first_choice_text: str = None
    tool_schema: dict = None
    tool_calls: list = None
    tool_results: list = None

    # State tracking fields
    # For fields that need instance-specific mutable defaults, use field() with default_factory
    # This avoids the mutable default argument pitfall: https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
    state: response_state = field(default_factory=lambda: response_state.INIT)
    accumulated_text: str = ''

    # Tool calling state fields from mixin
    current_function_index: int = -1
    current_function_name: str = None
    in_function_arguments: bool = False

    # For stateful tracking of args during tool call streaming
    _accumulated_args: dict = field(default_factory=dict, repr=False)  # repr=False to exclude from __repr__

    # __post_init__ not needed in this case b/c mixin is providing behavior rather than state,
    # Fields this live in this dataclass; mixin just provides methods that operate on those fields.
    # def __post_init__(self):
    #     '''
    #     Called automatically after __init__ - use this for any
    #     initialization that depends on fields being set
    #     '''
    #     super().__init__()  # Initialize the mixin if needed

    @staticmethod
    def from_openai_chat(response):
        '''
        Convert an OpenAI API ChatCompletion object to an llm_response object
        '''
        resp_type = llm_response_type.MESSAGE  # Default assumption

        # Extract usage stats if present
        usage = None
        if 'usage' in response:
            usage = response['usage']
        elif 'timings' in response:
            usage = {
                'prompt_tokens': response['timings'].prompt_n,
                'completion_tokens': response['timings'].predicted_n,
                'total_tokens': response['timings'].prompt_n + response['timings'].predicted_n
            }
        # resp['prompt_tps'] = resp.timings.prompt_per_second
        # resp['generated_tps'] = resp.timings.predicted_per_second

        # Process choices
        choices = []
        _first_choice_text = None
        tool_calls = None
        if response.get('choices'):
            choices = response['choices']
            rc1 = choices[0]
            message = rc1.get('message', {})

            # Check for tool calls
            if message.get('tool_calls'):
                assert message['tool_calls'], 'If tool_calls is present, it must not be empty'
                resp_type = llm_response_type.TOOL_CALL
                # Process tool call arguments; Convert to tool_call objects
                # WTH does OpenAI have these arguments properties as plain text? Seems a massive layering violation
                tool_calls = [
                    tool_call(
                        id=tc['id'],
                        name=tc['function']['name'],
                        arguments=json.loads(tc['function']['arguments'])
                    )
                    for tc in message['tool_calls']
                ]
            else:
                _first_choice_text = (
                    rc1.get('text') or
                    message.get('content', '')
                )
        else:
            _first_choice_text = response.get('content')

        model_type = response.get(TOOLIO_MODEL_TYPE_FIELD)
        resp = llm_response(
            response_type=resp_type,
            choices=choices,
            usage=usage,
            object=response.get('object'),
            id=response.get('id'),
            created=response.get('created'),
            model=response.get('model'),
            model_type=model_type,
            # model_flags=response.get(TOOLIO_MODEL_FLAGS_FIELD)
            model_flags=FLAGS_LOOKUP.get(model_type, DEFAULT_FLAGS)
        )

        # Add first choice text as property
        if _first_choice_text is not None:
            resp._first_choice_text = _first_choice_text

        # Add tool calls if present
        if tool_calls:
            resp.tool_calls = tool_calls

        return resp

    from_llamacpp = from_openai_chat
    # llama_cpp regular completion endpoint response keys: 'content', 'generation_settings', 'model', 'prompt', 'slot_id', 'stop', 'stopped_eos', 'stopped_limit', 'stopped_word', 'stopping_word', 'timings', 'tokens_cached', 'tokens_evaluated', 'tokens_predicted', 'truncated'  # noqa

    @property
    def first_choice_text(self) -> str | None:
        '''Get text content from first choice'''
        if self._first_choice_text is not None:
            return self._first_choice_text
        if self.response_type == llm_response_type.MESSAGE:
            if 'content' in self.choices[0]['delta']:
                return self.choices[0]['delta']['content']
            else:  # Handle case where final content wasn't properly updated
                return getattr(self, 'accumulated_text', None)

    @classmethod
    def from_generation_response(cls, gen_resp, model_name=None, model_type=None,
                               tool_schema: dict | None = None):  # -> 'llm_response' | None:
        '''
        Convert a MLX_LM utils.GenerationResponse (representing a response delta) to llm_response or tool_call_response

        Args:
            gen_resp: GenerationResponse from utils.generate_step()
            model_name: Optional model name/path
            model_type: Optional model type identifier
            tool_schema: Optional tool schema for parsing tool calls
        '''
        if not gen_resp.text:  # Skip empty responses
            return None

        resp_t = time.time_ns()
        # print(f'from_generation_response\t{tool_schema=}')
        resp = cls(
            response_type=llm_response_type.MESSAGE,  # Default to message
            choices=[{
                'index': 0,
                'delta': {'role': 'assistant'},
                'finish_reason': None
            }],
            object='chat.completion',
            id=f'cmpl-{int(resp_t)}',
            created=int(resp_t / 1e9),
            model=model_name,
            model_type=model_type,
            model_flags = FLAGS_LOOKUP.get(model_type, DEFAULT_FLAGS)
        )

        if tool_schema:
            resp.tool_schema = tool_schema
            resp.state = response_state.GATHERING_TOOL_CALLS
        else:
            resp.state = response_state.GATHERING_MESSAGE

        resp.update_from_gen_response(gen_resp)
        return resp

    def update_from_gen_response(self, gen_resp):
        '''
        Update this response with a new generation chunk from MLX

        Args:
            gen_resp: GenerationResponse containing new content
        '''
        self.latest_gen_resp = gen_resp  # Convenience for listeners
        choice0 = self.choices[0]
        delt = choice0.get('delta', choice0.get('message'))

        # Empty text means we've finished. Update finish_reason
        if not gen_resp.text:
            choice0['finish_reason'] = gen_resp.finish_reason
            return

        if self.state == response_state.GATHERING_TOOL_CALLS:
            self.accumulated_text += gen_resp.text
            # Try parsing accumulated text as tool calls
            # FIXME: Inefficient to be re-trying the parse for eaxh token. Find a way to get a completion signal before parsing
            tool_calls = parse_genresp_tool_calls(self.accumulated_text)
            if tool_calls:
                # Successfully parsed complete tool calls
                # assert 'delta' in choice0
                self.response_type = llm_response_type.TOOL_CALL
                delt['tool_calls'] = [tc.to_dict() for tc in tool_calls]
                choice0['finish_reason'] = 'tool_calls'
                self.tool_calls = tool_calls
                self.state = response_state.COMPLETE
                self.accumulated_text = ''  # Reset for next
                return
        elif self.state == response_state.GATHERING_MESSAGE:
            # assert 'message' in choice0
            delt.setdefault('content', '')
            delt['content'] += gen_resp.text

        # Update finish_reason regardless of text presence
        choice0['finish_reason'] = gen_resp.finish_reason

        if gen_resp.prompt_tokens is not None:
            if not self.usage:
                self.usage = {
                    'prompt_tokens': gen_resp.prompt_tokens,
                    'completion_tokens': gen_resp.generation_tokens,
                    'total_tokens': gen_resp.prompt_tokens + gen_resp.generation_tokens
                }
            else:
                self.usage['completion_tokens'] = gen_resp.generation_tokens
                self.usage['total_tokens'] = self.usage['prompt_tokens'] + gen_resp.generation_tokens

    def to_dict(self) -> dict:
        '''Convert to dictionary format'''
        resp_dict = {
            'choices': self.choices,
            'response_type': self.response_type.value
        }

        if self.usage:
            resp_dict['usage'] = self.usage
        if self.object:
            resp_dict['object'] = self.object
        if self.id:
            resp_dict['id'] = self.id
        if self.created:
            resp_dict['created'] = self.created
        if self.model:
            resp_dict['model'] = self.model
        if self.model_type:
            resp_dict[TOOLIO_MODEL_TYPE_FIELD] = self.model_type
            resp_dict[TOOLIO_MODEL_FLAGS_FIELD] = self.model_flags

        return resp_dict

    def to_openai_chat_response(self) -> dict:
        '''
        Convert internal response to OpenAI chat completion API format
        '''
        resp = {
            'id': self.id,
            'object': self.object or 'chat.completion',
            'created': self.created,
            'model': self.model,
            'choices': []
        }

        if self.usage:
            resp['usage'] = self.usage

        # Preserve model type info if available
        if self.model_type:
            resp[TOOLIO_MODEL_TYPE_FIELD] = self.model_type
            resp[TOOLIO_MODEL_FLAGS_FIELD] = self.model_flags

        # Convert choices to OpenAI format
        for choice in self.choices:
            openai_choice = {'index': choice.get('index', 0)}

            # Handle different response types
            if self.response_type == llm_response_type.TOOL_CALL:
                # Tool calls need to be in a message wrapper
                openai_choice['message'] = {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [tc.to_dict() for tc in self.tool_calls]
                }
            else:
                # Regular message responses
                message = choice.get('message', {})
                if not message:
                    # Create message from delta if needed
                    message = {
                        'role': 'assistant',
                        'content': choice.get('delta', {}).get('content', self.first_choice_text)
                    }
                openai_choice['message'] = message

            openai_choice['finish_reason'] = choice.get('finish_reason')
            resp['choices'].append(openai_choice)

        return resp

    def to_json(self) -> str:
        '''Convert to JSON string'''
        return json.dumps(self.to_dict())
