import os

import bz2
import pickle

from collections import Counter

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
from torch.utils.data import Dataset
from .utils import DataSetSplit, IGNORE_INDEX
from .corpora import DailyDialog, EmpatheticDialogues, PersonaChat, WizardOfWikipedia
from .corpora import Hope, CounsellingAndPsychotherapyCorpus

from model import DLDLMTokenizer

from typing import List, Tuple, Dict, Optional, Set

CORPORA: Dict = {
    CounsellingAndPsychotherapyCorpus.IDENTIFIER: CounsellingAndPsychotherapyCorpus,
    DailyDialog.IDENTIFIER: DailyDialog,
    EmpatheticDialogues.IDENTIFIER: EmpatheticDialogues,
    Hope.IDENTIFIER: Hope,
    PersonaChat.IDENTIFIER: PersonaChat,
    WizardOfWikipedia.IDENTIFIER: WizardOfWikipedia,
}

nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS: Set[str] = set(stopwords.words('english')) | spacy.load('en_core_web_sm').Defaults.stop_words
PUNCTUATION: Set[str] = set("[!\"#$%&()*+,-./:;<=>?@[]\\^`{|}~_']") | {'...', '``', '\'\'', '--'}
CUSTOM_STOP_WORDS: Set[str] = set()


def get_word_counts(s: str, remove_stop_words: bool = True) -> Counter:
    if remove_stop_words:
        return Counter(
            w.lower() for w in word_tokenize(s) if w.lower() not in STOP_WORDS | PUNCTUATION | CUSTOM_STOP_WORDS
        )
    else:
        return Counter(w.lower() for w in word_tokenize(s))


# TODO move here context cutting, it shouldn't be part of the corpora loaders
# TODO add class for fine tuning
class DLDLMCorpus(Dataset):
    def __init__(
            self,
            corpora_dir_path: str,
            tokenizer: DLDLMTokenizer,
            data_set_split: str,
            cache_dir_path: str,
            *args,
            corpus_prefix: str = 'pretraining_corpus',
            corpus_list: Optional[List[str]] = None,
            reload_cache: bool = False,
            max_response_length: Optional[int] = None,
            latent: bool = True,
            count_word_tokens: bool = False,
            remove_stopwords: bool = True,
            **kwargs
    ):
        super(DLDLMCorpus, self).__init__()
        # Tokeniser to prepare inputs
        self.tokenizer: DLDLMTokenizer = tokenizer
        # Whether to use the latent codes
        self.latent = latent
        # Max response length
        self.max_response_length: Optional[int] = max_response_length
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Path to corpus data frame
        self.corpus_cache_file_path: str = os.path.join(cache_dir_path, f'{corpus_prefix}_{data_set_split}.pbz2')
        # Data
        self.data: List[Dict]

        # Generate cache if needed
        if not os.path.exists(self.corpus_cache_file_path) or reload_cache:
            # Create cache dir if not exists
            if not os.path.exists(cache_dir_path):
                os.mkdir(cache_dir_path)
            #
            self.corpora_dir_path: str = corpora_dir_path
            # Get corpus list ad list of all available corpora if not provided
            if corpus_list is None:
                self.corpus_list: List[str] = [
                    dir_name for dir_name in os.listdir(corpora_dir_path)
                    if os.path.isdir(os.path.join(corpora_dir_path, dir_name))
                ]
            # Else simply save the provided list
            else:
                self.corpus_list: List[str] = corpus_list
            # Word count flags
            self.count_word_tokens: bool = count_word_tokens
            self.remove_stopwords: bool = remove_stopwords
            # Load all corpora and generate cache
            self._generate_data_cache(*args, **kwargs)
        # Else simply load the cache
        else:
            self._load_data_cache()

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]

    def _generate_data_cache(self, *args, **kwargs):
        # Create corpora instances
        corpora = [
            CORPORA[corpus_id](
                os.path.join(self.corpora_dir_path, corpus_id),
                self.data_set_split.value,
                self.tokenizer,
                *args,
                **kwargs
            )
            for corpus_id in self.corpus_list
        ]
        # and gather the data
        self.data = [corpus[idx] for corpus in corpora for idx in range(len(corpus))]
        # Count word tokens in needed
        if self.count_word_tokens:
            for sample in self.data:
                sample['word_counts'] = get_word_counts(sample['response'], remove_stop_words=self.remove_stopwords)
        # Save compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'w') as f:
            pickle.dump(self.data, f)

    def _load_data_cache(self):
        # Load compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'r') as f:
            self.data = pickle.load(f)

    def collate(self, mini_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        # Encode context
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        context_encodings = self.tokenizer(
            [sample['context'] for sample in mini_batch], padding=True, return_tensors='pt'
        )
        self.tokenizer.padding_side = padding_side
        # Encode response
        if self.latent:
            response_encodings = self.tokenizer(
                ['<|prior|>' + sample['response'] + '<|posterior|>' for sample in mini_batch],
                padding=True,
                return_tensors='pt'
            )
        else:
            response_encodings = self.tokenizer(
                [sample['response'] + self.tokenizer.eos_token for sample in mini_batch],
                padding=True,
                return_tensors='pt'
            )
        # Prepare inputs
        if self.max_response_length is not None:
            input_ids = torch.hstack(
                [context_encodings.input_ids, response_encodings.input_ids[:self.max_response_length]]
            )
            attention_mask = torch.hstack(
                [context_encodings.attention_mask, response_encodings.attention_mask[:self.max_response_length]]
            )
        else:
            input_ids = torch.hstack([context_encodings.input_ids, response_encodings.input_ids])
            attention_mask = torch.hstack([context_encodings.attention_mask, response_encodings.attention_mask])
        # Prepare target labels
        label_encodings = self.tokenizer(
            [self.tokenizer.eos_token + sample['response'] + self.tokenizer.eos_token for sample in mini_batch],
            padding=True,
            return_tensors='pt'
        )
        if self.max_response_length is not None:
            labels = label_encodings.input_ids[:self.max_response_length]
            labels[~label_encodings.attention_mask[:self.max_response_length].bool()] = IGNORE_INDEX
        else:
            labels = label_encodings.input_ids
            labels[~label_encodings.attention_mask.bool()] = IGNORE_INDEX

        return input_ids, attention_mask, labels, mini_batch
