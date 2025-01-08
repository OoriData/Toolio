"""
Utilities to use the bitmap of accepted token ids returned by TokenAcceptor.
"""

from math import inf
from typing import Iterable


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


def bias_logits(np, logits, accepted_token_bitmap):
    """
    Apply a -inf bias to tokens that will not be accepted.
    Rather than import here, the np parameters is numpy or a compatible library
    import, such as mlx.core.
    """
    vocab_size = logits.shape[0]
    highest_token_accepted = highest_bit_set(accepted_token_bitmap)
    accepted_token_count = count_set_bits(accepted_token_bitmap)
    # Check whether there's more tokens to be rejected or to be allowed, then do what's less work.
    if accepted_token_count <= highest_token_accepted / 2:
        bias = np.full(vocab_size, -inf)
        indices = np.array([*enumerate_set_bits(accepted_token_bitmap)])
        bias[indices] = 0
    else:
        bias = np.concatenate(
            [
                np.full(highest_token_accepted + 1, 0),
                # All tokens above the highest accepted token are rejected.
                np.full(vocab_size - highest_token_accepted - 1, -inf),
            ]
        )
        rejected_token_bitmap = bitmap_complement(accepted_token_bitmap)
        indices = np.array([*enumerate_set_bits(rejected_token_bitmap)])
        bias[indices] = -inf
    return np.add(logits, bias)
