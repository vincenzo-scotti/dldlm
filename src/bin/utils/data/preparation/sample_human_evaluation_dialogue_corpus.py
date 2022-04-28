import sys
import os
import random

from argparse import ArgumentParser

import numpy as np
import pandas as pd

RANDOM_SEED = 2307

N_SAMPLES = 5
TURN_IDS = [1, 3]
N_TYPES = len(TURN_IDS)

# DTYPES = [str, str, int, int, str, str, str, str, str, str]
SPLITS = ['test']
USERS = ['Listener']
# Complete list of emotions:
# [
#     'afraid',
#     'angry',
#     'annoyed',
#     'anticipating',
#     'anxious',
#     'apprehensive',
#     'ashamed',
#     'caring',
#     'confident',
#     'content',
#     'devastated',
#     'disappointed',
#     'disgusted',
#     'embarrassed',
#     'excited',
#     'faithful',
#     'furious',
#     'grateful',
#     'guilty',
#     'hopeful',
#     'impressed',
#     'jealous',
#     'joyful',
#     'lonely',
#     'nostalgic',
#     'prepared',
#     'proud',
#     'sad',
#     'sentimental',
#     'surprised',
#     'terrified',
#     'trusting'
# ]
EMOTIONS = ['sad', 'joyful']
DF_COLUMNS = [
    'split',
    'data_set',
    'conversation_id',
    'turn_id',
    'user',  # {Speaker or Listener}
    'emotion',
    'situation',
    'context',
    'turn',
    'distractors'
]


def main(args):
    # Create destination directory
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load evaluation corpus
    data_set = pd.read_csv(args.source_file_path)
    # Filter corpus
    # Split
    if SPLITS is not None and len(SPLITS) > 0:
        data_set = data_set[data_set.split.isin(SPLITS)]
    # User
    if USERS is not None and len(USERS) > 0:
        data_set = data_set[data_set.user.isin(USERS)]
    # Emotions
    if EMOTIONS is not None and len(EMOTIONS) > 0:
        data_set = data_set[data_set.emotion.isin(EMOTIONS)]
    # Fix nan contexts
    data_set.context.fillna("", inplace=True)

    # Group by emotion and then conversation ID and sample elements
    data = np.vstack(
        [
            group[(group['conversation_id'] == conversation_id) & (group['turn_id'] == TURN_IDS[idx % N_TYPES])].values
            for _, group in data_set.groupby('emotion')
            for idx, conversation_id in enumerate(random.sample(list(group.conversation_id), len(group.conversation_id)))
            if idx < N_SAMPLES * N_TYPES
        ]
    )

    # Serialize corpus data frame
    df = pd.DataFrame(data, columns=DF_COLUMNS)
    df.to_csv(os.path.join(args.dest_dir_path, 'human_evaluation_samples.csv'), index=False)

    return 0


if __name__ == "__main__":
    # Input arguments
    args_parser = ArgumentParser()
    args_parser.add_argument('--source_file_path', type=str,
                             default='./resources/data/dialogue_corpus/human_evaluation_corpus.csv',
                             help="Path to the file containing the corpus to sample for the evaluation.")
    args_parser.add_argument('--dest_dir_path', type=str,
                             default='./resources/data/dialogue_corpus/',
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
