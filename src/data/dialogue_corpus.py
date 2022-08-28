import os

import bz2
import pickle

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from collections import Counter

import torch
from torch.utils.data import Dataset
from .utils import DataSetSplit, IGNORE_INDEX
from .corpora import DailyDialog, EmpatheticDialogues, PersonaChat, WizardOfWikipedia
from .corpora import Hope, CounsellingAndPsychotherapyCorpus

from model import DLDLMTokenizer
from misc import get_word_counts, tf_idf

from typing import List, Tuple, Dict, Optional

CORPORA: Dict = {
    CounsellingAndPsychotherapyCorpus.IDENTIFIER: CounsellingAndPsychotherapyCorpus,
    DailyDialog.IDENTIFIER: DailyDialog,
    EmpatheticDialogues.IDENTIFIER: EmpatheticDialogues,
    Hope.IDENTIFIER: Hope,
    PersonaChat.IDENTIFIER: PersonaChat,
    WizardOfWikipedia.IDENTIFIER: WizardOfWikipedia,
}


# TODO move here context cutting, it shouldn't be part of the corpora loaders
# TODO add class for fine tuning
class DLDLMCorpus(Dataset):
    # Additional static attributes (used in incremental TF-IDF computation)
    word_doc_counts: Optional[Dict[DataSetSplit, Counter]] = None
    n_docs: Optional[Dict[DataSetSplit, int]] = None

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
            compute_tf_idf: bool = True,
            incremental_tf_idf: bool = True,
            remove_stopwords: bool = True,
            min_df: int = 2,
            concurrent_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 2,
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
            # Save parallelisation options
            self.parallel_backend: str = concurrent_backend
            self.n_jobs: int = n_jobs
            self.verbosity_level: int = verbosity_level

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
            self.compute_tf_idf: bool = compute_tf_idf
            self.incremental_tf_idf: bool = incremental_tf_idf and compute_tf_idf
            self.count_word_tokens: bool = count_word_tokens or self.compute_tf_idf
            self.remove_stopwords: bool = remove_stopwords and not self.compute_tf_idf
            # Create static dictionaries if required
            if self.incremental_tf_idf:
                self.word_doc_counts = dict()
                self.n_docs = dict()
            # Other TF-IDF params
            self.min_df: int = min_df
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
                parallel_backend=self.parallel_backend,
                n_jobs=self.n_jobs,
                verbosity_level=self.verbosity_level,
                **kwargs
            )
            for corpus_id in self.corpus_list
        ]
        # and gather the data
        self.data = [corpus[idx] for corpus in corpora for idx in range(len(corpus))]
        # Compute additional info if needed
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            # Helper functions
            def get_word_counts_wrapper(sample):
                sample['word_counts'] = get_word_counts(sample['response'], remove_stopwords=self.remove_stopwords)

            def get_document_counts_wrapper(sample):
                return Counter(sample['word_counts'].keys())

            def get_word_tf_idf_scores_wrapper(sample):
                sample['tf-idf'] = tf_idf(sample['word_counts'], word_document_counts, n_docs, min=self.min_df)

            # Count word tokens in needed
            if self.count_word_tokens:
                Parallel(verbose=self.verbosity_level)(
                    delayed(get_word_counts_wrapper)(sample) for sample in self.data
                )
            # Possibly compute TF-IDF
            if self.compute_tf_idf:
                # Count word document instances
                word_document_counts: Counter = sum(
                    Parallel(verbose=self.verbosity_level)(
                        delayed(get_document_counts_wrapper)(sample) for sample in self.data
                    ), Counter()
                )
                # Count number of documents
                n_docs = len(self.data)
                # Possibly update with other split counts
                if self.incremental_tf_idf:
                    if self.data_set_split == DataSetSplit.VALIDATION:
                        word_document_counts.update(self.word_doc_counts.get(DataSetSplit('validation'), Counter()))
                        n_docs += self.n_docs.get(DataSetSplit('validation'), 0)
                    elif self.data_set_split == DataSetSplit.TEST:
                        word_document_counts.update(self.word_doc_counts.get(DataSetSplit('test'), Counter()))
                        n_docs += self.n_docs.get(DataSetSplit('test'), 0)
                    self.word_doc_counts[self.data_set_split] = word_document_counts
                    self.n_docs[self.data_set_split] = n_docs
                # Compute TF-IDF
                Parallel(verbose=self.verbosity_level)(
                    delayed(get_word_tf_idf_scores_wrapper)(sample) for sample in self.data
                )
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
