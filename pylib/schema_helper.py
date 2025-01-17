# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# toolio.schema_helper
'''
JSON schema decoding with MLX
'''
import time
import functools
from math import inf
from operator import itemgetter
from typing import Iterable, Optional, Union, Callable, List, Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache, _BaseCache
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load, stream_generate, GenerationResponse

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

    def _evaluate_prompt(
        self, prompt: list[int], prior_prompt: list[int] = None, prior_cache=None
    ):
        if prior_prompt:
            i = 0
            for i, t in enumerate(prior_prompt):
                # Need to leave at least one token to evaluate because we don't
                # save the past logits.
                if i >= len(prompt) - 1 or prompt[i] != t:
                    break
            cache = prior_cache
            for layer_cache in cache:
                layer_cache.reuse(len(prompt), i)
            tokens = prompt[i:]
            # print('CACHED', tokens, prompt, i)
        else:
            cache = ReusableKVCache.for_model(self.model)
            tokens = prompt
            # print('UNCACHED', tokens)

        logits = self.model(mx.array(tokens)[None], cache=cache)
        return logits, cache

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

    def _sample_with_bias(
        self, logits, temp: float = 0, token_acceptor=None, lazy_bias: bool = True
    ):
        if token_acceptor is None:
            return self._sample(logits, temp)

        if lazy_bias:
            token = self._sample(logits, temp)
            try:
                token_acceptor.advance_token(token)
                return token
            except JsonSchemaAcceptorDriver.TokenRejected:
                pass

        # By select valid tokens they really mean create a bitmap masking out invalid tokens
        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        if not accepted_token_bitmap:
            raise RejectedCompletion()
        # token = self._sample(bias_logits(mx, logits, accepted_token_bitmap), temp)
        token_acceptor.advance_token(token)
        return token

    # def stream_generate_with_schema(
    #     self,
    #     prompt: mx.array,
    #     max_tokens: int = 256,
    #     temp: Optional[float] = 0.0,
    #     sampler: Optional[Callable[mx.array, mx.array]] = None,
    #     logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    #     max_kv_size: Optional[int] = None,
    #     # prompt_cache: Optional[_BaseCache] = None,
    #     prefill_step_size: int = 512,
    #     kv_bits: Optional[int] = None,
    #     kv_group_size: int = 64,
    #     quantized_kv_start: int = 0,
    #     prompt_progress_callback: Optional[Callable] = None,
    # ):
    #     '''
    #     Generate text with an optional JSON schema acceptor.
    #     '''
    #     logits_processors = logits_processors or []
    #     logits_processors = logits_processors.copy()
    #     logits_processors.append(self.make_logit_bias_processor())

    #     sample = functools.partial(
    #         self._sample_with_bias, temp=temp
    #     )
    #     while True:
    #         tokens = [sample(logits[0, -1, :])]
    #         yield tokens
    #         if tokens[-1] == self.eos_id:
    #             break
    #         logits = self.model(mx.array(tokens)[None], cache=cache)

    def logit_bias_processor(self, tokens: mx.array, logits: mx.array) -> mx.array:
        '''
        Apply a -inf bias to tokens that will not be accepted
        '''
        if tokens.size <= self._prompt_length:  # Still processing prompt (prefill)
            # print('In prefill')
            return logits

        # print('In generation')

        # Get valid next tokens from current state
        accepted_token_bitmap = self.curr_token_acceptor.select_valid_tokens()
        if not accepted_token_bitmap:
            raise RejectedCompletion()

        # Set logits of rejected tokens to -inf
        accepted_tokens = [*enumerate_set_bits(accepted_token_bitmap)]
        rejected_tokens = [t for t in range(logits.shape[-1])
                        if t not in accepted_tokens]
        logits = mx.put_along_axis(
            logits, mx.array(rejected_tokens)[None, ...], mx.array(-inf, logits.dtype), axis=-1
        )

        # print('After biasing:')
        # self._peek_top_logits(logits)  # DEBUG ONLY
        # x = 2030; x = 33966; print(f'Logit check {x} | {logits[0].tolist()[x]}')
        return logits

    def completion(
        self,
        messages: Union[str, Iterable[dict[str, str]]],
        schema: dict,
        encapsulated: bool = False,
        seed: int = None,
        cache_prompt: bool = False,
        **kwargs,  # From MLX_LM_GENERATE_KWARGS
    ):
        for k in kwargs:
            if k not in MLX_LM_GENERATE_KWARGS:
                raise ValueError(f'Unknown keyword argument: {k}')

        logits_processors = kwargs.get('logits_processors', [])
        logits_processors = logits_processors.copy()
        logits_processors.append(self.logit_bias_processor)
        kwargs['logits_processors'] = logits_processors

        if self.tokenizer is None:  # Not loaded
            raise RuntimeError('Model not loaded')

        if seed is not None:
            mx.random.seed(seed)

        prompt_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        self._prompt_length = len(prompt_tokens)  # Store prompt length

        # FIXME: Non-reentrant
        self.curr_token_acceptor = self.json_schema_acceptor_driver_factory(schema, encapsulated) if schema else None

        logits_generator = stream_generate(self.model, self.tokenizer, prompt_tokens, **kwargs)

        for generation_resp in stream_generate(self.model, self.tokenizer, prompt_tokens, **kwargs):
            print(f'{generation_resp.token=}')
            print(repr(self._detokenize(generation_resp.token)))
            if generation_resp.token is not None:
                try:
                    self.curr_token_acceptor.advance_token(generation_resp.token)
                except JsonSchemaAcceptorDriver.TokenRejected:
                    raise

            yield generation_resp

        # for generation_resp in logits_generator:
        #     # generation_resp is a GenerationResponse object
        #     # text (str): The next segment of decoded text. This can be an empty string.
        #     # token (int): The next token.
        #     # logprobs (mx.array): A vector of log probabilities.
        #     # prompt_tokens (int): The number of tokens in the prompt.
        #     # prompt_tps (float): The prompt processing tokens-per-second.
        #     # generation_tokens (int): The number of generated tokens.
        #     # generation_tps (float): The tokens-per-second for generation.
        #     # peak_memory (float): The peak memory used so far in GB.
        #     # finish_reason (str): The reason the response is being sent: "length", "stop" or `None`

        #     # Skip token acceptor advancement during prefill
        #     # print(generation_resp.token)
        #     if generation_resp.token is not None: # Generation phase started
        #         self._in_prefill = False
        #         try:
        #             # print('Picked:', list(self.detokenize_dammit([self.generation_resp.token])))
        #             self.curr_token_acceptor.advance_token(generation_resp.token)
        #         except JsonSchemaAcceptorDriver.TokenRejected:
        #             raise  # Explicit re-raise for now

        #     yield generation_resp

    def _peek_top_logits(self, logits):
        # Debug: Print top logits and their token indices
        top_k = 10  # Number of top logits to show
        logits_array = logits[0].tolist()  # Convert to regular array
        token_logits = [(i, l) for i, l in enumerate(logits_array)]
        token_logits.sort(key=lambda x: x[1], reverse=True)  # Sort by logit value

        print("\nTop logits:")
        for token_idx, logit_val in token_logits[:top_k]:
            token_text = self.tokenizer.decode([token_idx])  # Use regular decode method
            print(f"Token {token_idx} ({token_text!r}): {logit_val:.3f}")

    def _detokenize(self, tokens):
        '''For debugging'''
        if not isinstance(tokens, list):
            tokens = [tokens]
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        for tok in tokens:
            detokenizer.add_token(tok)
        return detokenizer.last_segment


class ReusableKVCache(KVCache):
    '''
    Usability improvements over MLX's KVCache.
    '''
    @classmethod
    def for_model(cls, model):
        return [cls() for _ in model.layers]

    def reuse(self, new_prompt_length, common_prefix_length):
        '''
        Reuse (part of) this cache for a new prompt that shares a prefix with it.
        '''
        if self.keys is None:
            return
        # Clip the cache to the common length.
        self.offset = common_prefix_length
        # Ensure cache can fit the whole prompt. Because the offset is (very likely) not a multiple of the step size,
        # update_and_fetch() won't resize the cache when evaluating the rest of the prompt as it
        # would if it were an empty cache.
        current_size = self.keys.shape[2]
        if current_size < new_prompt_length:
            _, n_kv_heads, _, k_head_dim = self.keys.shape
            v_head_dim = self.values.shape[3]

            n_steps = (self.step + new_prompt_length - 1) // self.step
            k_add_shape = (1, n_kv_heads, n_steps * self.step - current_size, k_head_dim)
            v_add_shape = (1, n_kv_heads, n_steps * self.step - current_size, v_head_dim)
            k_zeros = mx.zeros(k_add_shape, self.keys.dtype)
            v_zeros = mx.zeros(v_add_shape, self.values.dtype)
            self.keys = mx.concatenate([self.keys, k_zeros], axis=2)
            self.values = mx.concatenate([self.values, v_zeros], axis=2)

    def update_and_fetch(self, keys, values):
        '''
        Override base class method to allow the cache to be used with batches size >1
        (Just a tiny change in the line that determines the shape)
        '''
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
