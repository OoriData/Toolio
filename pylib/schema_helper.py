# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.schema_helper
'''
JSON schema decoding with MLX
'''
# import time
import functools
from math import inf
from operator import itemgetter
from typing import Iterable, Union, Any

import mlx.core as mx
# from mlx_lm.models.cache import KVCache, _BaseCache
# from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load, stream_generate # , GenerationResponse

from toolio.vendor.llm_structured_output import JsonSchemaAcceptorDriver
from toolio.vendor.llm_structured_output.util.bitmap import highest_bit_set, count_set_bits, bitmap_complement, enumerate_set_bits
from toolio.vendor.llm_structured_output.util.output import debug
from toolio.vendor.llm_structured_output.util.tokenization import HuggingfaceTokenizerHelper

# Lower level MLX LLM processing kwargs & their defaults
MLX_LM_GENERATE_KWARGS = {
    'max_tokens': 1024,
    'sampler': None,
    'logits_processors': None,
    'max_kv_size': None,
    # 'prompt_cache': None,
    'prefill_step_size': 512,
    'kv_bits': None,
    'kv_group_size': 64,
    'quantized_kv_start': 0,
    'prompt_progress_callback': None
}


class RejectedCompletion(Exception):
    '''
    Reached a state from where it's not possible to advance the acceptor (a rare condition).
    For example, when closing a JSON string we get a higher probability for curly quotes than ASCII
    ones and thus select the wrong token. The LLM then continues generating as if the string
    has been closed, but the acceptor remains awaiting a close quote. Could be a bug in the
    tokenizer vocabulary passed to the acceptor, or in the code decoding tokens from the LLM.
    Could also be an inability of the LLM to generate JSON, although most can.
    '''


@functools.lru_cache(maxsize=128)
def create_mask(accepted_token_bitmap, vocab_size):
    token_bitmap_str = '{0:b}'.format(accepted_token_bitmap)
    return mx.array([False if i > (len(token_bitmap_str) - 1)
                     else token_bitmap_str[-1 - i] == '1' for i in range(vocab_size)])

def apply_token_mask(logits, accepted_token_bitmap):
    vocab_size = logits.shape[-1]

    # Use the memoized function to create or retrieve a boolean mask for the entire vocabulary
    mask = create_mask(accepted_token_bitmap, vocab_size)

    # Invert the mask and convert to the same dtype as logits
    inverted_mask = (~mask).astype(logits.dtype)

    # Multiply the inverted mask by negative infinity
    inf_mask = inverted_mask * mx.array(-mx.inf, dtype=logits.dtype)

    # Apply the mask to logits
    masked_logits = mx.where(mask, logits, inf_mask)

    return masked_logits


class Model:
    def __init__(self):
        mx.random.seed(0)
        self.model = None
        self.tokenizer = None
        self.vocabulary = None
        self.eos_id = None
        self.json_schema_acceptor_driver_factory = None
        # Note: If for example the user loads a cache from a file, and we support prompt caching that way, they should not have to re-specify init params such as max_kv_size
        # self._prompt_cache = make_prompt_cache(self.model, max_kv_size)

    def load(self, model_path: str):
        '''
        Load locally or download from Huggingface hub.
        '''
        self.model, self.tokenizer = load(model_path)
        tokenizer_helper = HuggingfaceTokenizerHelper(self.tokenizer)
        self.vocabulary, self.eos_id = tokenizer_helper.extract_vocabulary()
        self.json_schema_acceptor_driver_factory = (
            JsonSchemaAcceptorDriver.driver_factory_for_model(
                self.vocabulary, self.eos_id
            )
        )

    def completion(
        self,
        messages: Union[str, Iterable[dict[str, str]]],
        schema: dict | list | str | None,
        encapsulated: bool = False,
        seed: int | None = None,
        cache_prompt: bool = False,
        **kwargs,  # From MLX_LM_GENERATE_KWARGS
    ):
        # XXX: Do we want to look into reentrancy of this method?
        for k in kwargs:
            if k not in MLX_LM_GENERATE_KWARGS:
                raise ValueError(f'Unknown keyword argument: {k}')

        logits_processors = kwargs.get('logits_processors', [])
        self.curr_token_acceptor = self.json_schema_acceptor_driver_factory(schema, encapsulated) if schema else None
        if self.curr_token_acceptor:
            logits_processors = logits_processors.copy()
            logits_processors.append(self.logit_bias_processor)
            kwargs['logits_processors'] = logits_processors

        if self.tokenizer is None:  # Not loaded
            raise RuntimeError('Model not loaded')

        if seed is not None:
            mx.random.seed(seed)

        prompt_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        self._prompt_length = len(prompt_tokens)  # Store prompt length

        logits_generator = stream_generate(self.model, self.tokenizer, prompt_tokens, **kwargs)

        self._step_count = 0
        for generation_resp in stream_generate(self.model, self.tokenizer, prompt_tokens, **kwargs):
            yield generation_resp

    def logit_bias_processor(self, tokens: mx.array, logits: mx.array) -> mx.array:
        '''
        Apply a -inf bias to tokens that will not be accepted
        '''
        if self._step_count > 0:
            # Just past the stream_generate look-ahead step. Have to advance the state here,
            # because the chosen token won't have bubbled up to completion() whare that usually happens
            # Advance the acceptor state, so it's ready for the next biasing step
            prev_tok = tokens.tolist()[-1]
            if prev_tok != self.eos_id:
                try:
                    self.curr_token_acceptor.advance_token(prev_tok)
                except JsonSchemaAcceptorDriver.TokenRejected:
                    raise

        self._step_count += 1

        # Compute valid next tokens from current state
        accepted_token_bitmap = self.curr_token_acceptor.select_valid_tokens()
        if not accepted_token_bitmap:
            raise RejectedCompletion()

        # Set logits of rejected tokens to -inf
        logits = apply_token_mask(logits, accepted_token_bitmap)

        # print('After biasing:'); self._peek_top_logits(logits)
        return logits

    def _peek_top_logits(self, logits):
        # Debug: Print top logits and their token indices
        top_k = 10  # Number of top logits to show
        logits_array = logits[0].tolist()  # Convert to regular array
        token_logits = [(i, l) for i, l in enumerate(logits_array)]
        token_logits.sort(key=lambda x: x[1], reverse=True)  # Sort by logit value

        print('\nTop logits:')
        for token_idx, logit_val in token_logits[:top_k]:
            token_text = self.tokenizer.decode([token_idx])  # Use regular decode method
            print(f'Token {token_idx} ({token_text!r}): {logit_val:.3f}')

    def _detokenize(self, tokens):
        '''For debugging'''
        if not isinstance(tokens, list):
            tokens = [tokens]
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        for tok in tokens:
            detokenizer.add_token(tok)
        return detokenizer.last_segment

    def _decode(self, tokens):
        return self.tokenizer.no_strip_decode(tokens)

    def _debug_top_tokens(self, logits, count=10):
        token_logits = sorted(
            enumerate(logits.tolist()), key=itemgetter(1), reverse=True
        )
        top_tokens = [
            (self._decode([t]), p) for t, p in token_logits[:count] if p != -inf
        ]
        debug('TOP TOKENS:', top_tokens)
