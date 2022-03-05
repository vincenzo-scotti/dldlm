import sys
import os
import re

from argparse import ArgumentParser

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import json
import pandas as pd

CORPUS_ID = 'Wizard_of_Wikipedia'
SOURCE_FILE_LIST = ['train.json', 'valid_random_split.json', 'test_random_split.json']
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
    utterance = [preprocess_text(turn['text'])]
    return utterance


def preprocess_dialog(sample, split_id, conversation_id):
    # Pre-process dialogue
    dialogue = [
        [split_id, CORPUS_ID, conversation_id, idx, *preprocess_turn(turn)]
        for idx, turn in enumerate(sample['dialog'])
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
            data_set = json.loads(f.read())

        # Standardise corpus
        with parallel_backend(args.parallel_backend, n_jobs=args.n_jobs):
            data_ = Parallel(verbose=args.verbosity_level)(
                delayed(preprocess_dialog)(sample, split_id, idx)
                for idx, sample in enumerate(data_set, start=len(data))
            )

        data += sum(data_, [])

    # Serialize corpus data frame
    df = pd.DataFrame(data, columns=DF_COLUMNS)
    df.to_csv(os.path.join(args.dest_dir_path, CORPUS_ID.lower() + '.csv'), index=False)

    return 0


if __name__ == "__main__":
    # Input arguments
    args_parser = ArgumentParser()
    args_parser.add_argument('--source_dir_path', type=str, default='$DLDLM/resources/data/raw/wizard_of_wikipedia/',
                             help="Path to the directory containing the raw corpus.")
    args_parser.add_argument('--dest_dir_path', type=str,
                             default='$DLDLM/resources/data/preprocessed/Wizard_of_Wikipedia/',
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
