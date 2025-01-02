# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.common
'''
Common bits; can be imported without MLX (e.g. for client use on non-Mac platforms)
'''
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


class prompt_handler:
    '''
    Encapsulates functionality for manipulating prompts, client or server side
    '''
    # XXX: Default option for sysmgg?
    def __init__(self, model_type=None, logger=None, sysmsg_leadin=''):
        self.model_type = model_type
        self.model_flags = FLAGS_LOOKUP.get(model_type, DEFAULT_FLAGS)
        self.sysmsg_leadin = sysmsg_leadin
        self.logger = logger or logging

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
            new_msgs.insert(0, {'role': 'system', 'content': sysmsg})
        else:
            new_msgs = msgs[:]

        return new_msgs


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
