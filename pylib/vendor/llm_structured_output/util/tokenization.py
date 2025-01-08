"""
Tokenizer utils.
"""

from typing import Union

SPIECE_UNDERLINE = "▁"


class HuggingfaceTokenizerHelper:
    """
    Helper to use Huggingface tokenizers effectively.
    """

    def __init__(self, tokenizer):
        """
        tokenizer is expected to be a Huggingface PreTrainedTokenizer[Fast]
        """
        self.tokenizer = tokenizer
        self.token_has_space_prefix = dict(
            [
                (i, fragment[0] == SPIECE_UNDERLINE)
                for fragment, i in tokenizer.vocab.items()
            ]
        )

    def encode_prompt(self, prompt: Union[str, list[dict[str, str]]]) -> list[int]:
        """
        Encode the prompt, applying the tokenizer template first if the prompt
        is a series of messages instead of a straight string.
        """
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        if not self.tokenizer.chat_template:
            return self.tokenizer.encode("\n\n".join(
                f"{message['role']}: {message['content']}"
                for message in prompt
            ))
        return self.tokenizer.apply_chat_template(prompt)

    def no_strip_decode(self, tokens):
        """
        Allows to decode single tokens without removing the initial space.
        The Huggingface tokenizer doesn't seem to have an easy way to do this.
        """
        fragment = self.tokenizer.decode(tokens)
        if self.token_has_space_prefix[tokens[0]]:
            return f" {fragment}"
        else:
            return fragment

    def extract_vocabulary(self) -> tuple[list[tuple[int, str]], int]:
        """
        Extract the vocabulary and eos_token_id from a Huggingface PreTrainedTokenizer.
        """
        return (
            [(i, self.no_strip_decode([i])) for _, i in self.tokenizer.vocab.items()],
            self.tokenizer.eos_token_id,
        )
