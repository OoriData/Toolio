"""
Utilities to use the bitmap of accepted token ids returned by TokenAcceptor.
"""

from math import inf
from typing import Iterable, Optional, Union, Callable, List, Any

import mlx.core as mx


def count_set_bits(bitmap: int) -> int:
    """
    Count the number of bits set to one.
    """
    # FUTURE: self.ids.bit_count() available from Python 3.10 is said to be 6x faster
    return bin(bitmap).count("1")


def highest_bit_set(bitmap: int) -> int:
    """
    Return the index of the highest bit set in the bitmap.
    """
    return bitmap.bit_length() - 1


def bitmap_complement(bitmap: int, set_size: int = None) -> int:
    """
    Negate the bits in the bitmap.
    Since the bitmap is encoded as a Python int, it can be of arbitrary length.
    I.e. we don't know how many zeros are above the top set bit. The set_size
    parameter can be passed to indicate the number of bits in the bitmap (which
    is akin to the number of members in the set it represents). If unspecified,
    the top set bit in the bitmap is used as its set size. 
    """
    if not set_size:
        set_size = bitmap.bit_length()
    return (1 << set_size) - 1 - bitmap


def enumerate_set_bits(bitmap: int) -> Iterable[int]:
    """
    Generator that yields the indices of the set bits in the bitmap.
    Note that it does so from highest to lowest.
    """
    while bitmap:
        highest_bit = highest_bit_set(bitmap)
        yield highest_bit
        bitmap -= 1 << highest_bit

