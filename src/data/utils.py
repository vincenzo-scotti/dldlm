import re
from enum import Enum
from torch.utils.data import Dataset

from model import DLDLMTokenizer

from typing import List, Dict, Any, Optional

IGNORE_INDEX = -1


class DataSetSplit(Enum):
    DEVELOPMENT: str = 'dev'
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'


# TODO add custom data class to manage samples
class DialogueCorpus(Dataset):
    IDENTIFIER = None
    # Substitutes for non-unicode special characters
    UNICODE_SWITCH_LIST = [
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201d", '"'),
        ("\u201c", '"'),
        ("\u2014", "--"),
        ("\u2013", "--"),
        ("\u3002", ". "),
        ("\u2032", "'"),
        ("\u3001", ", ")
    ]

    def __init__(
            self,
            corpus_dir_path: str,
            data_set_split: str,
            tokenizer: DLDLMTokenizer,
            start_token: Optional[str] = None,
            stop_token: Optional[str] = None,
            max_context_length: Optional[int] = None,
            parallel_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 2
    ):
        super(DialogueCorpus, self).__init__()
        # Path to corpus main directory
        self.corpus_dir_path: str = corpus_dir_path
        # Tokeniser to prepare inputs
        self.tokenizer: DLDLMTokenizer = tokenizer
        # Special tokens to mark start of dialogue and end of turn
        self.start_token: Optional[str] = start_token if start_token is not None else tokenizer.bos_token
        self.stop_token: Optional[str] = stop_token if stop_token is not None else tokenizer.eos_token
        # Maximum allowed context length
        self.max_context_length: Optional[int] = max_context_length
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Parallel backend
        self.parallel_backend: str = parallel_backend
        # Number of concurrent jobs
        self.n_jobs: int = n_jobs
        # Verbosity level
        self.verbosity_level: int = verbosity_level

        # Load data
        self.data: List[Dict] = self._load_samples()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]

    def _get_dialogue_contexts(self, dialogue_turns: List[str]) -> List[str]:
        # Gather all available context strings
        context_strings: List[str] = [self.start_token] + [turn + self.stop_token for turn in dialogue_turns[:-1]]
        # If a limit on the context is given cut it
        if self.max_context_length is not None:
            # Accumulator for contexts
            dialogue_contexts: List[List[int]] = list()
            # Tokenise context strings
            tokenized_context_strings: List[List[int]] = self.tokenizer(context_strings).input_ids
            # Lenght of context strings
            tokenized_context_string_lengths: List[int] = [
                len(tokenized_context_string) for tokenized_context_string in tokenized_context_strings
            ]
            # For each response, select the context of maximum allowed length
            for e_idx in range(1, len(dialogue_turns) + 1):
                tmp_tokenized_context: List[List[int]] = tokenized_context_strings[:e_idx]
                tmp_tokenized_context_lengths: List[int] = tokenized_context_string_lengths[:e_idx]
                tmp_all_tokenized_context_length: int = sum(tmp_tokenized_context_lengths)
                while tmp_all_tokenized_context_length > self.max_context_length and len(tmp_tokenized_context) > 1:
                    tmp_all_tokenized_context_length -= tmp_tokenized_context_lengths.pop(0)
                    tmp_tokenized_context.pop(0)
                dialogue_contexts.append(sum(tmp_tokenized_context, [])[-self.max_context_length:])

            # Convert tokenised contexts into a list of strings
            dialogue_context_strings: List[str] = self.tokenizer.batch_decode(dialogue_contexts)
        # Else simply concatenate all context strings up to a point
        else:
            dialogue_context_strings: List[str] = [sum(context_strings[:i+1])for i in range(len(context_strings))]

        return dialogue_context_strings

    def _preprocess_text(self, text: str) -> str:
        for u_code_sym, replace_sym in self.UNICODE_SWITCH_LIST:
            text = text.replace(u_code_sym, replace_sym)  # Taken from ParlAI preprocessing
        text = re.sub(r'\.(\w)', r' . \1', text)  # Taken from ParlAI preprocessing
        text = re.sub('[ \t\n]+', ' ', text)
        text = text.strip()

        return text

    def _preprocess_dialogue(self, *args, **kwargs):
        raise NotImplementedError()

    def _load_samples(self) -> List[Dict]:
        raise NotImplementedError()
