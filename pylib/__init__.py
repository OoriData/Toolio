# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio

from toolio import __about__
from toolio.common import LANG
from toolio.llm_helper import model_manager

__all__ = ['LANG', 'model_manager', 'VERSION']

VERSION = __about__.__version__


try:
    import mlx  # noqa: F401
except ImportError:
    import warnings
    warnings.warn('Unable to import MLX, which requires an Apple Silicon Mac. '
                  'You will only be able to run pure HTTP client components.')
