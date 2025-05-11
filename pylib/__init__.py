# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio
import platform
import sys

from toolio import __about__
from toolio.common import LANG

__all__ = ['LANG', 'model_manager', 'load_or_connect', 'VERSION']

VERSION = __about__.__version__


try:
    # MLX requires macOS 13.5 or later (AKA platform_release >= '22.6.0')
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        macos_version = tuple(map(int, platform.mac_ver()[0].split('.')))
        if macos_version >= (13, 5):
            import mlx  # noqa: F401

    from toolio.llm_helper import model_manager
except ImportError:
    model_manager = None

if model_manager is None:
    import warnings
    warnings.warn('Unable to import MLX. If you are running this on Apple Silicon you can just `pip install mlx mlx_lm`\n'
                  'Otherwise you will only be able to run the Toolio HTTP client component.')

from toolio.common import load_or_connect, response_text, print_response  # noqa: E402 F401
