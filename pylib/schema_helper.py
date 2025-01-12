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
from mlx_lm.utils import load

from toolio.vendor.llm_structured_output import JsonSchemaAcceptorDriver
from toolio.vendor.llm_structured_output.util.bitmap import (
    bias_logits,
    count_set_bits,
    enumerate_set_bits,
)
from toolio.vendor.llm_structured_output.util.output import debug
from toolio.vendor.llm_structured_output.util.tokenization import HuggingfaceTokenizerHelper


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
        self._cached_prompt = None
        self._cached_cache = None

    def load(self, model_path: str):
        '''
        Load locally or download from Huggingface hub.
        '''
        self.model, tokenizer = load(model_path)
        self.tokenizer = HuggingfaceTokenizerHelper(tokenizer)
        self.simple_tokenizer = tokenizer
        self.vocabulary, self.eos_id = self.tokenizer.extract_vocabulary()
        self.json_schema_acceptor_driver_factory = (
            JsonSchemaAcceptorDriver.driver_factory_for_model(
                self.vocabulary, self.eos_id
            )
        )

    def get_driver_for_json_schema(self, schema, encapsulated: bool = False):
        return self.json_schema_acceptor_driver_factory(
            schema, is_encapsulated_json=encapsulated
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

    def _sample(self, logits, temp: float = 0):
        if temp == 0:
            result = mx.argmax(logits, axis=-1)
        else:
            result = mx.random.categorical(logits * (1 / temp))
        return result.item()

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

        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        if not accepted_token_bitmap:
            raise RejectedCompletion()
        token = self._sample(bias_logits(mx, logits, accepted_token_bitmap), temp)
        token_acceptor.advance_token(token)
        return token

    def generate_with_schema(
        self, logits, cache, token_acceptor = None, temp: Optional[float] = 0.0
    ):
        '''
        Generate text with an optional JSON schema acceptor.
        '''
        sample = functools.partial(
            self._sample_with_bias, temp=temp, token_acceptor=token_acceptor
        )
        while True:
            tokens = [sample(logits[0, -1, :])]
            yield tokens
            if tokens[-1] == self.eos_id:
                break
            logits = self.model(mx.array(tokens)[None], cache=cache)

    def _generate_tokens(
        self,
        generator: Iterable,
        max_tokens: int = 1000,
    ) -> Iterable:
        start_time = time.time_ns()
        token_count = 0

        for tokens in generator:
            token_count += len(tokens)

            try:
                eos_index = tokens.index(self.eos_id)
                tokens = tokens[0:eos_index]
            except ValueError:
                eos_index = -1

            if tokens:
                text = self._decode(tokens)
                yield {
                    'op': 'generatedTokens',
                    'text': text,
                    'token_count': len(tokens),
                    'time_ms': (time.time_ns() - start_time) / 1e6,
                }

            if eos_index >= 0:
                yield {'op': 'stop', 'reason': 'end'}
                return

            if token_count >= max_tokens:
                yield {'op': 'stop', 'reason': 'max_tokens'}
                return

            start_time = time.time_ns()

        assert False

    def stream_generate_with_schema(
            self,
            prompt: mx.array,
            schema: dict,
            max_tokens: int = 256,
            sampler: Optional[Callable[mx.array, mx.array]] = None,
            logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
            max_kv_size: Optional[int] = None,
            prompt_cache: Optional[_BaseCache] = None,
            prefill_step_size: int = 512,
            kv_bits: Optional[int] = None,
            kv_group_size: int = 64,
            quantized_kv_start: int = 0,
            prompt_progress_callback: Optional[Callable] = None,
    ):
        pass

    def completion(
        self,
        prompt: Union[str, Iterable[dict[str, str]]],
        schema: dict,
        encapsulated: bool = False,
        max_tokens: int = 1000,
        temp: float = 0.0,
        seed: int = None,
        cache_prompt: bool = False,
    ):
        if seed is not None:
            mx.random.seed(seed)

        start_time = time.time_ns()
        prompt_tokens = self.tokenizer.encode_prompt(prompt)
        logits, cache = self._evaluate_prompt(
            prompt_tokens, self._cached_prompt, self._cached_cache
        )
        if cache_prompt:
            self._cached_prompt = prompt_tokens
            self._cached_cache = cache
        # Eager eval to more accurately reflect the prompt evaluation time.
        mx.eval(logits)
        prompt_time = time.time_ns() - start_time
        yield {
            'op': 'evaluatedPrompt',
            'prompt': prompt,
            'token_count': len(prompt_tokens),
            'time_ms': prompt_time / 1e6,
            'prompt_tps': len(prompt_tokens) / (prompt_time / 1e9),
        }

        token_acceptor = self.get_driver_for_json_schema(schema, encapsulated) if schema else None
        generator = self.generate_with_schema(logits, cache, token_acceptor, temp)

        token_count = 0
        generation_time = 0
        for generation_result in self._generate_tokens(generator, max_tokens):
            if generation_result['op'] == 'generatedTokens':
                token_count += generation_result['token_count']
                generation_time += generation_result['time_ms']
            elif generation_result['op'] == 'stop':
                generation_result['token_count'] = token_count
                generation_result['time_ms'] = generation_time
                if generation_time == 0.0:
                    # Happens, believe it or not
                    generation_result['generation_tps'] = float('inf')
                else:
                    # Slightly incorrect, because the first token is generated from the prompt evaluation
                    generation_result['generation_tps'] = token_count / (
                        generation_time / 1e3
                    )
            yield generation_result


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
