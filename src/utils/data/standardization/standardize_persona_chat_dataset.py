import sys
import os
import re

from argparse import ArgumentParser

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import pandas as pd

CORPUS_ID = 'Persona-Chat'
SOURCE_FILE_LIST = ['train_both_original.txt', 'valid_both_original.txt', 'test_both_original.txt']
DISTRACTORS_SPLIT_SYM = '\t\t'
DISTRACTORS_SEPARATOR_SYM = '|'
RESPONSE_SPLIT_SYM = '\t'
Y_PERSONA_SYM = 'your persona:'
P_PERSONA_SYM = 'partner\'s persona:'
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


def preprocess_turn_pair(turn_pair):
    # Pre-process utterance
    turn_pair, distractors = turn_pair.split(DISTRACTORS_SPLIT_SYM)
    turn_pair = turn_pair.split(RESPONSE_SPLIT_SYM)
    distractors = [preprocess_text(distractor.strip()) for distractor in distractors.split(DISTRACTORS_SEPARATOR_SYM)]
    utterances = [[preprocess_text(t.strip()), "\n".join(distractors)] for t in turn_pair]
    return utterances


def preprocess_dialog(sample, split_id, conversation_id):
    # Pre-process dialogue
    dialogue = [
        [split_id, CORPUS_ID, conversation_id, (2 * idx_p) + idx_t, *turn]
        for idx_p, turn_pair in enumerate(
            turn_pair for turn_pair in sample if not (Y_PERSONA_SYM in turn_pair or P_PERSONA_SYM in turn_pair)
        )
        for idx_t, turn in enumerate(preprocess_turn_pair(turn_pair))
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
            data_set = [line.split(' ', 1) for line in f.readlines()]
            start_idxs = [l_idx for l_idx, (i, _) in enumerate(data_set) if int(i) == 1]
            end_idxs = start_idxs[1:] + [len(data_set)]
            data_set = [[elem for _, elem in data_set[s_idx:e_idx]] for s_idx, e_idx in zip(start_idxs, end_idxs)]

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
    args_parser.add_argument('--source_dir_path', type=str, default='$DLDLM/resources/data/raw/personachat/',
                             help="Path to the directory containing the raw corpus.")
    args_parser.add_argument('--dest_dir_path', type=str, default='$DLDLM/resources/data/preprocessed/Persona-Chat/',
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
