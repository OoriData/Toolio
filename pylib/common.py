# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.common
'''
Common bits; can be imported without MLX (e.g. for client use on non-Mac platforms)
'''
import json
import logging
import warnings
from pathlib import Path  # noqa: E402
from enum import Flag, auto

from amara3 import iri

# import mlx.core as mx
# from mlx_lm.models import (gemma, gemma2, llama, phi, qwen, su_rope, minicpm, phi3, qwen2, gpt2,
#                            mixtral, phi3small, qwen2_moe, cohere, gpt_bigcode, phixtral,
#                            stablelm, dbrx, internlm2, openelm, plamo, starcoder2)

# from mlx_lm.models import olmo  # Will say:: To run olmo install ai2-olmo: pip install ai2-olmo

from ogbujipt import word_loom


class model_flag(Flag):
    NO_SYSTEM_ROLE = auto()  # e.g. Gemma blows up if you use a system message role
    USER_ASSISTANT_ALT = auto()  # Model requires alternation of message roles user/assistant only
    TOOL_RESPONSE = auto()  # Model expects responses from tools via OpenAI API style messages


DEFAULT_FLAGS = model_flag(0)

# {model_class: flags}, defaults to DEFAULT_FLAGS
FLAGS_LOOKUP = {
    # Actually Llama seems to want asistant response rather than as tool
    # 'llama': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT | model_flag.TOOL_RESPONSE,
    'llama': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'gemma': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'gemma2': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mixtral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'mistral': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
    'qwen2': model_flag.NO_SYSTEM_ROLE | model_flag.USER_ASSISTANT_ALT,
}

TOOLIO_MODEL_TYPE_FIELD = 'toolio.model_type'

DEFAULT_JSON_SCHEMA_CUTOUT = '#!JSON_SCHEMA!#'


def load_or_connect(ref: str, **kwargs):
    '''
    Sonvenience function to either load a module from a file or HuggingFace path,
    or connect to a Toolio server, based on checking the provided reference string

    Parameters:
        ref (str) - file path, HuggingFace path or Toolio server URL
    '''
    if iri.matches_uri_syntax(ref):
        from toolio.client import struct_mlx_chat_api, response_type, cmdline_tools_struct
        llm = struct_mlx_chat_api(base_url=ref, **kwargs)
    else:
        from toolio.llm_helper import model_manager
        llm = model_manager(ref, **kwargs)
    return llm


# FIXME: Replace with utiloori.filepath.obj_file_path_parent
def obj_file_path_parent(obj):
    '''Cross-platform Python trick to get the path to a file containing a given object'''
    import inspect
    from pathlib import Path
    # Should already be an absolute path
    # from os.path import abspath
    # return abspath(inspect.getsourcefile(obj))
    return Path(inspect.getsourcefile(obj)).parent


HERE = obj_file_path_parent(lambda: 0)
with open(HERE / Path('resource/language.toml'), mode='rb') as fp:
    LANG = word_loom.load(fp)


class model_runner_base:
    '''
    Encapsulates logic to interact with models, local or remote, and in particular
    deals with manipulating prompts
    '''
    # XXX: Default option for sysmgg?
    def __init__(self, model_type=None, logger=None, sysmsg_leadin='', default_schema=None,
                 json_schema_cutout=DEFAULT_JSON_SCHEMA_CUTOUT):
        self.model_type = model_type
        self.model_flags = FLAGS_LOOKUP.get(model_type, DEFAULT_FLAGS)
        self.sysmsg_leadin = sysmsg_leadin
        self.logger = logger or logging
        self.default_schema = default_schema
        self.default_schema_str = json.dumps(default_schema) if default_schema else None
        self.json_schema_cutout = json_schema_cutout

    def reconstruct_messages(self, msgs, sysmsg=None):
        '''
        Take a message set and rules for prompt composition to create a new, effective prompt

        msgs - chat messages to process, potentially including user message and system message
        sysmsg - explicit override of system message
        kwargs - overrides for components for the sysmsg template
        '''
        if not msgs:
            raise ValueError('Unable to process an empty prompt')

        # Ensure it's a well-formed prompt, ending with at least one user message
        if msgs[-1]['role'] != 'user':
            raise ValueError(f'Final message in the chat prompt must have a \'user\' role. Got {msgs[-1]}')

        # Index the current system roles
        system_indices = [i for i, m in enumerate(msgs) if m['role'] == 'system']
        # roles = [m['role'] for m in msgs]
        # XXX Should we at least warn about any empty messages?

        if sysmsg:
            # Override any existing system messages by removing, then adding the one
            new_msgs = [m for m in msgs if m['role'] != 'system']
            if model_flag.NO_SYSTEM_ROLE in self.model_flags:
                new_msgs[0]['content'] = sysmsg + '\n\n' + new_msgs[0]['content']  # + is slow, but LLMs are slower 😉
            else:
                new_msgs.insert(0, {'role': 'system', 'content': sysmsg})

        else:
            new_msgs = msgs[:]

        return new_msgs

    def replace_cutout(self, messages, schema_str):
        '''
        Replace JSON schema cutout references with the actual schema; if not found, append to the first user message
        Return a copy of the messages with the schema inserted
        '''
        cutout_replaced = False
        new_messages = messages.copy()
        for m in messages:
            # XXX: content should always be in m, though. Validate?
            if 'content' in m and self.json_schema_cutout in m['content']:
                new_m = m.copy()
                new_m['content'] = new_m['content'].replace(self.json_schema_cutout, schema_str)
                new_messages.append(new_m)
                cutout_replaced = True
            else:
                new_messages.append(m)

        if not cutout_replaced:
            warnings.warn('JSON Schema provided, but no place found to replace it.'
                        ' Will be tacked on the end of the first user message', stacklevel=2)
            target_msg = next(m for m in new_messages if m['role'] == 'user')
            # FIXME: More robust message validation, perhaps add a helper in prompt_helper.py
            assert target_msg is not None
            target_msg['content'] += '\nRespond in JSON according to this schema: ' + schema_str


prompt_handler = model_runner_base  # Backward compatibility


async def extract_content(resp_stream):
    # Interpretthe streaming pattern from the API. viz https://platform.openai.com/docs/api-reference/streaming
    async for chunk in resp_stream:
        # Minimal checking: Trust the delivered structure
        if 'delta' in chunk['choices'][0]:
            content = chunk['choices'][0]['delta'].get('content')
            if content is not None:
                yield content
        else:
            content = chunk['choices'][0]['message'].get('content')
            if content is not None:
                yield content


async def response_text(resp_stream):
    chunks = []
    async for chunk in extract_content(resp_stream):
        chunks.append(chunk)
    return ''.join(chunks)
