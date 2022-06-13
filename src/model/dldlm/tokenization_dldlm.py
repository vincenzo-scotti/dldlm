from transformers import GPT2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


# TODO manage other static class attributes
class DLDLMTokenizer(GPT2Tokenizer):
    __SPECIAL_TOKENS__ = {
        # 'unk_token': '<|unknown|>',
        # 'mask_token': '<|mask|>',
        'additional_special_tokens': ['<|prior|>', '<|posterior|>']
    }
    # TODO manage static attributes

    def extend_from_gpt2_tokenizer(self, num_styles: int) -> PreTrainedTokenizer:
        # TODO find a way to remove all old special tokens
        # Set pad token
        self.pad_token = self.eos_token
        # Prepare new special tokens
        special_tokens = self.__SPECIAL_TOKENS__.copy()
        special_tokens['additional_special_tokens'] += [f'<|latentcode{str(i).zfill(2)}|>' for i in range(num_styles)]
        # Extend token set
        self.add_special_tokens(special_tokens)
        # Return extended tokenizer
        return self
