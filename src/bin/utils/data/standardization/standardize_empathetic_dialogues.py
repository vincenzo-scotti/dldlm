import sys
import os
import re

from argparse import ArgumentParser

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import pandas as pd

CORPUS_ID = 'EmpatheticDialogues'
SOURCE_FILE_LIST = ['train.csv', 'valid.csv', 'test.csv']
DTYPES = [str, int, str, str, int, str, str, str, str]
CONV_ID = 'conv_id'
UTTERANCE_ID = 'utterance_idx'
SPLITS = ['train', 'validation', 'test']
DF_COLUMNS = [
    'split',
    'data_set',
    'conversation_id',
    'turn_id',
    'text',
    'distractors'
]

# Substitutes for non-unicode special characters
UNICODE_SWITCH_LIST = [("\u2019", "'"),
                       ("\u2018", "'"),
                       ("\u201d", '"'),
                       ("\u201c", '"'),
                       ("\u2014", "--"),
                       ("\u2013", "--"),
                       ("\u3002", ". "),
                       ("\u2032", "'"),
                       ("\u3001", ", ")]


def preprocess_text(text):
    preprocessed_text = text.strip()
    # Taken from ParlAI preprocessing
    for u_code_sym, replace_sym in UNICODE_SWITCH_LIST:
        preprocessed_text = preprocessed_text.replace(u_code_sym, replace_sym)
    # Taken from ParlAI preprocessing
    preprocessed_text = re.sub(r'\.(\w)', r' . \1', preprocessed_text)
    # Added custom
    preprocessed_text = re.sub('[ \t\n]+', ' ', preprocessed_text).strip()
    return preprocessed_text


def preprocess_turn(turn):
    # Pre-process utterance
    utterance = [
        preprocess_text(re.sub('_comma_', ',', turn['utterance'])),
        "\n".join(
            preprocess_text(re.sub('_comma_', ',', distractor.strip()))
            for distractor in turn['distractors'].split('|')
        ) if turn['distractors'] is not None and turn['distractors'] != "" else ""
    ]
    return utterance


def preprocess_dialog(sample, split_id, conversation_id):
    # Pre-process dialogue
    dialogue = [
        [split_id, CORPUS_ID, conversation_id, idx, *preprocess_turn(turn)]
        for idx, (_, turn) in enumerate(sample.sort_values(UTTERANCE_ID).iterrows())
    ]

    return dialogue


def main(args):
    # Create destination directory
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)

    # Standardise and serialise corpus
    data = []
    for source_file_name, split_id in zip(SOURCE_FILE_LIST, SPLITS):
        # Load split of the corpus
        with open(os.path.join(args.source_dir_path, source_file_name)) as f:
            data_set = [line.strip().split(',') for line in f.readlines()]
        columns = data_set.pop(0) + ['distractors']
        data_set = [sample if len(sample) == len(columns) else sample + [""] for sample in data_set]
        data_set = [[t(x) for x, t in zip(sample, DTYPES)] for sample in data_set]

        data_set = pd.DataFrame(data=data_set, columns=columns)

        # Standardise corpus
        with parallel_backend(args.parallel_backend, n_jobs=args.n_jobs):
            data_ = Parallel(verbose=args.verbosity_level)(
                delayed(preprocess_dialog)(sample, split_id, idx)
                for idx, (_, sample) in enumerate(data_set.groupby(CONV_ID), start=len(data))
            )
        data += sum(data_, [])

    # Serialize corpus data frame
    df = pd.DataFrame(data, columns=DF_COLUMNS)
    df.to_csv(os.path.join(args.dest_dir_path, CORPUS_ID.lower() + '.csv'), index=False)

    return 0


if __name__ == "__main__":
    # Input arguments
    args_parser = ArgumentParser()
    args_parser.add_argument('--source_dir_path', type=str, default='./resources/data/raw/empatheticdialogues/',
                             help="Path to the directory containing the raw corpus.")
    args_parser.add_argument('--dest_dir_path', type=str,
                             default='./resources/data/preprocessed/EmpatheticDialogues/',
                             help="Path to the directory where to store the standardised corpus.")
    args_parser.add_argument('--parallel_backend', type=str, default='threading',
                             help="Parallel backend to use for preprocessing (see joblib package documentation).")
    args_parser.add_argument('--n_jobs', type=int, default=-1,
                             help="Number of parallel active jobs to use for preprocessing "
                                  "(see joblib package documentation).")
    args_parser.add_argument('--verbosity_level', type=int, default=2,
                             help="Frequency of printed progress messages (see joblib package documentation).")
    # Run standardisation
    main(args_parser.parse_args(sys.argv[1:]))
