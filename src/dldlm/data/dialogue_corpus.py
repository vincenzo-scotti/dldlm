import os

import bz2
import pickle

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from collections import Counter

import torch
from torch.utils.data import Dataset
from .utils import DataSetSplit, IGNORE_INDEX
from .corpora import DailyDialog, EmpatheticDialogues, PersonaChat, WizardOfWikipedia
from .corpora import CounsellingAndPsychotherapyCorpus, Hope

from dldlm.model import DLDLMTokenizer
from dldlm.misc import sentiment

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
    docs: Optional[List[str]] = None

    def __init__(
            self,
            corpora_dir_path: str,
            tokenizer: DLDLMTokenizer,
            data_set_split: str,
            cache_dir_path: str,
            *args,
            corpus_prefix: str = 'pretraining_corpus',
            corpus_list: Optional[List[str]] = None,
            latent_init: bool = False,
            lda_kwargs: Optional[Dict] = None,
            reload_cache: bool = False,
            max_response_length: Optional[int] = None,
            latent: bool = True,
            distractor: bool = False,
            count_word_tokens: bool = True,
            compute_tf_idf: bool = True,
            incremental_stats: bool = True,
            max_df: float = 0.95,
            min_df: int = 2,
            compute_sentiment: bool = True,
            concurrent_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 2,
            **kwargs
    ):
        super(DLDLMCorpus, self).__init__()
        # Tokeniser to prepare inputs
        self.tokenizer: DLDLMTokenizer = tokenizer
        # Whether to use the latent codes
        self.latent: bool = latent
        # Whether to use distractor samples
        self.distractor: bool = distractor
        # Max response length
        self.max_response_length: Optional[int] = max_response_length
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Path to corpus data frame
        self.corpus_cache_file_path: str = os.path.join(cache_dir_path, f'{corpus_prefix}_{data_set_split}.pbz2')
        # Data
        self.data: List[Dict]

        # Latent initialisation parameters (if used)
        self.latent_init: bool = latent_init
        self.lda_kwargs: Optional[Dict] = lda_kwargs

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
            self.incremental_stats: bool = incremental_stats and compute_tf_idf
            self.count_word_tokens: bool = count_word_tokens or self.compute_tf_idf
            # Sentiment analysis flags
            self.compute_sentiment: bool = compute_sentiment
            # Create static dictionaries if required
            if self.incremental_stats:
                self.docs = list()
            # Other TF-IDF params
            self.max_df: float = max_df
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
            def set_tf_idf(sample, tf, words, key):
                sample[key] = Counter({word: tf[idx] for idx, word in zip(zip(*tf.nonzero()), words)})

            def set_latent_p_dist(sample, p_dist):
                sample['latent_p_dist'] = p_dist.tolist()

            def set_sentiment(sample):
                sample['sentiment'] = sentiment(sample['response'])

            # Create document collection
            docs = [sample['response'] for sample in self.data]
            # Possibly add document collection to global collection
            if self.docs is not None:
                self.docs += docs
                docs = self.docs

            # Compute TF if required
            if self.count_word_tokens:
                # Create vectoriser instance
                tf_vectoriser = CountVectorizer(max_df=self.max_df, min_df=self.min_df, stop_words="english")
                # Fit vectoriser on the documents and gather counts
                tf_matrix = tf_vectoriser.fit_transform(docs)[-len(self.data):]
                # Assign counts to samples
                Parallel(verbose=self.verbosity_level)(
                    delayed(set_tf_idf)(sample, tf, words, 'tf')
                    for sample, tf, words
                    in zip(self.data, tf_matrix, tf_vectoriser.inverse_transform(tf_matrix))
                )
            else:
                tf_vectoriser = None
                tf_matrix = None

            # Compute TF-IDF if required
            if self.compute_tf_idf:
                # Create vectoriser instance
                tf_idf_vectoriser = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, stop_words="english")
                # Fit vectoriser on the documents and gather counts
                tf_idf_matrix = tf_idf_vectoriser.fit_transform(docs)[-len(self.data):]
                # Assign counts to samples
                Parallel(verbose=self.verbosity_level)(
                    delayed(set_tf_idf)(sample, tf_idf, words, 'tf_idf')
                    for sample, tf_idf, words
                    in zip(self.data, tf_idf_matrix, tf_idf_vectoriser.inverse_transform(tf_idf_matrix))
                )

            # Do latent initialisation if required
            if self.latent_init:
                # Compute tf required (otherwise re-use)
                if tf_vectoriser is None or tf_matrix is None:
                    # Create vectoriser instance
                    tf_vectoriser = CountVectorizer(max_df=self.max_df, min_df=self.min_df, stop_words="english")
                    # Fit vectoriser on the documents and gather counts
                    tf_matrix = tf_vectoriser.fit_transform(docs)[-len(self.data):]
                # LDA
                lda = LatentDirichletAllocation(**self.lda_kwargs)
                # Fit and compute p-dist
                p_dist = lda.fit_transform(tf_matrix)
                # Assign initial probabilities
                Parallel(verbose=self.verbosity_level)(
                    delayed(set_latent_p_dist)(sample, p) for sample, p in zip(self.data, p_dist)
                )

            # Compute sentiment if required
            if self.compute_sentiment:
                Parallel(verbose=self.verbosity_level)(
                    delayed(set_sentiment)(sample) for sample in self.data
                )
        # Save compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'w') as f:
            pickle.dump(self.data, f)

    def _load_data_cache(self):
        # Load compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'r') as f:
            self.data = pickle.load(f)

    def collate(self, mini_batch) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[torch.tensor],
        Optional[torch.tensor],
        Optional[torch.tensor],
        List[Dict]
    ]:
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
            # Force last token to be the posterior token, useful when dealing with sequences that are too long
            if self.max_response_length is not None and self.max_response_length < response_encodings.input_ids.size(-1):
                for b_idx, t_idx in zip(*torch.where(
                        response_encodings.input_ids == self.tokenizer.convert_tokens_to_ids('<|posterior|>')
                )):
                    if t_idx >= self.max_response_length:
                        response_encodings.input_ids[b_idx, self.max_response_length - 1] = response_encodings.input_ids[b_idx, t_idx]
        else:
            response_encodings = self.tokenizer(
                [sample['response'] + self.tokenizer.eos_token for sample in mini_batch],
                padding=True,
                return_tensors='pt'
            )
        # Prepare inputs
        if self.max_response_length is not None:
            input_ids = torch.hstack(
                [context_encodings.input_ids, response_encodings.input_ids[:, :self.max_response_length]]
            )
            attention_mask = torch.hstack(
                [context_encodings.attention_mask, response_encodings.attention_mask[:, :self.max_response_length]]
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
            labels = label_encodings.input_ids[:, :self.max_response_length]
            labels[~label_encodings.attention_mask[:, :self.max_response_length].bool()] = IGNORE_INDEX
        else:
            labels = label_encodings.input_ids
            labels[~label_encodings.attention_mask.bool()] = IGNORE_INDEX

        # Add latent initialisation proabilities if required
        if self.latent_init:
            latent_p_dist = torch.tensor([sample['latent_p_dist'] for sample in mini_batch])
        else:
            latent_p_dist = None

        # Add distractor samples if required
        if self.distractor:
            distractor_ids = torch.vstack([
                response_encodings.input_ids[1:, :self.max_response_length],
                response_encodings.input_ids[:1, :self.max_response_length]
            ])
            distractor_attention_mask = torch.vstack([
                response_encodings.attention_mask[1:, :self.max_response_length],
                response_encodings.attention_mask[:1, :self.max_response_length]
            ])
        else:
            distractor_ids = distractor_attention_mask = None

        return input_ids, attention_mask, labels, latent_p_dist, distractor_ids, distractor_attention_mask, mini_batch
