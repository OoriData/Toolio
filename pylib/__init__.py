# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# # toolio

from pathlib import Path  # noqa: E402
from enum import Flag, auto

from ogbujipt import word_loom
from toolio import __about__

VERSION = __about__.__version__

TOOLIO_MODEL_TYPE_FIELD = 'toolio.model_type'


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


class model_flag(Flag):
    NO_SYSTEM_ROLE = auto()  # e.g. Gemma blows up if you use a system message role
    USER_ASSISTANT_ALT = auto()  # Model requires alternation of message roles user/assistant only
    TOOL_RESPONSE = auto()  # Model expects responses from tools via OpenAI API style messages


DEFAULT_FLAGS = model_flag(0)
