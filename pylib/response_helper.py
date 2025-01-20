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
from dataclasses import dataclass

# from ogbujipt.config import attr_dict
from ogbujipt.llm_wrapper import response_type

from toolio.common import llm_response_type, TOOLIO_MODEL_TYPE_FIELD

# XXX: Basically a candidate for replacing the llm_response class in ogbujipt
@dataclass
class llm_response:
    '''
    Uniform interface for LLM responses from OpenAI APIs, MLX, etc.
    '''
    response_type: llm_response_type
    choices: list
    usage: dict = None  
    object: str = None
    id: str = None
    created: int = None
    model: str = None
    model_type: str = None
    _first_choice_text: str = None

    @staticmethod
    def from_openai_chat(response):
        '''
        Convert an OpenAI API ChatCompletion object to an llm_response object
        '''
        resp_type = response_type.MESSAGE  # Default assumption
        
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
        if response.get('choices'):
            choices = response['choices']
            rc1 = choices[0]
            
            # Check for tool calls
            if rc1.get('message', {}).get('tool_calls'):
                resp_type = response_type.TOOL_CALL
                # Process tool call arguments
                # WTH does OpenAI have these arguments properties as plain text? Seems a massive layering violation
                for tc in rc1['message']['tool_calls']:
                    tc['function']['arguments_obj'] = json.loads(tc['function']['arguments'])
            else:
                # Extract response text
                _first_choice_text = (
                    rc1.get('text') or 
                    rc1.get('message', {}).get('content', '')
                )
        else:
            _first_choice_text = response.get('content')

        # Create response object
        resp = llm_response(
            response_type=resp_type,
            choices=choices,
            usage=usage,
            object=response.get('object'),
            id=response.get('id'), 
            created=response.get('created'),
            model=response.get('model'),
            model_type=response.get(TOOLIO_MODEL_TYPE_FIELD)
        )

        # Add first choice text as property
        if _first_choice_text is not None:
            resp._first_choice_text = _first_choice_text

        return resp

    from_llamacpp = from_openai_chat
    # llama_cpp regular completion endpoint response keys: 'content', 'generation_settings', 'model', 'prompt', 'slot_id', 'stop', 'stopped_eos', 'stopped_limit', 'stopped_word', 'stopping_word', 'timings', 'tokens_cached', 'tokens_evaluated', 'tokens_predicted', 'truncated'  # noqa

    @property
    def first_choice_text(self) -> str | None:
        '''Get text content from first choice'''
        if self._first_choice_text is not None:
            return self._first_choice_text
        if self.response_type == llm_response_type.MESSAGE:
            return self.choices[0]['delta']['content']
        return None

    @classmethod
    def from_generation_response(cls, gen_resp, model_name=None, model_type=None):
        '''
        Convert a MLX_LM utils.GenerationResponse (representing a response delta) to LLMResponse
        
        Args:
            gen_resp: GenerationResponse from utils.generate_step()
            model_name: Optional model name/path
            model_type: Optional model type identifier
        '''
        # Only create a message delta if there's text to share
        if not gen_resp.text:
            return None
            
        choices = [{
            'index': 0,
            # XXX: Pretty sure this should be delta rather than message, but verify
            'delta': {'role': 'assistant', 'content': gen_resp.text},
            'finish_reason': gen_resp.finish_reason
        }]

        usage = None
        if gen_resp.prompt_tokens is not None:
            usage = {
                'prompt_tokens': gen_resp.prompt_tokens,
                'completion_tokens': gen_resp.generation_tokens,
                'total_tokens': gen_resp.prompt_tokens + gen_resp.generation_tokens
            }

        created = int(time.time())

        return cls(
            response_type=llm_response_type.MESSAGE,
            choices=choices,
            usage=usage,
            object='chat.completion',
            id=f'cmpl-{created}',
            created=created,
            model=model_name,
            model_type=model_type
        )

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

        return resp_dict

    def to_json(self) -> str:
        '''Convert to JSON string'''
        return json.dumps(self.to_dict())
