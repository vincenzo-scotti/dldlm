from typing import Optional

from transformers import GPT2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


# TODO manage other static class attributes
class DLDLMTokenizer(GPT2Tokenizer):
    __SPECIAL_TOKENS__ = {
        # 'unk_token': '</u>',  # TODO fix unknown token issue
        'bos_token': '<s>',
        'eos_token': '<s/>',
        'pad_token': '<s/>',
        'mask_token': '</null>',
        'additional_special_tokens': ['</c>', '</r>', '</l>', '</p>', '</q>']
    }
    # TODO manage static attributes

    def extend_from_gpt2_tokenizer(self, num_styles: int) -> PreTrainedTokenizer:
        # TODO find a way to remove all old special tokens
        # Prepare new tokens
        special_tokens = self.__SPECIAL_TOKENS__.copy()
        special_tokens['additional_special_tokens'] += [f'</z_{i}>' for i in range(num_styles)]
        # Extend token set
        self.add_special_tokens(special_tokens)
        # Return extended tokenizer
        return self
