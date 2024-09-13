# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio

from toolio import __about__
from toolio.common import LANG

__all__ = ['LANG', 'model_manager', 'VERSION']

VERSION = __about__.__version__


try:
    import mlx  # noqa: F401
    from toolio.llm_helper import model_manager
except ImportError:
    model_manager = None
    import warnings
    warnings.warn('Unable to import MLX. If you are running this on Apple Silicon you can just `pip install mlx mlx_lm`\n'
                  'Otherwise you will only be able to run the Toolio HTTP client component.')
