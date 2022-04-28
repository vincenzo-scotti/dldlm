import sys
import os
import random

from argparse import ArgumentParser

import numpy as np
import pandas as pd

RANDOM_SEED = 2307

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

    # Check that all compared models are unique
    assert len(args.evaluated_model_ids) == len(set(args.evaluated_model_ids)), \
        "Compared models identifiers must be all different"
    assert len(args.evaluated_model_ids) == len(args.source_dir_paths), "Each model should have a label and vice-versa"

    # Load evaluation samples
    samples = pd.read_csv(args.source_file_path)
    # Fix nan contexts
    samples.context.fillna("", inplace=True)
    # Add golden response column
    samples['generated_response'] = samples['turn']
    sample_idxs = ['generated_response']

    # Iteratively join with generated samples
    for path, suffix in zip(args.source_dir_paths, args.evaluated_model_ids):
        samples = pd.merge(
            samples,
            pd.read_csv(os.path.join(path, 'evaluation_generation_responses.csv')),
            on=DF_COLUMNS,
            suffixes=('', '_' + suffix)
        )
        sample_idxs.append('generated_response_' + suffix)

    # Shuffle data frame
    samples_shuffled = samples.sample(frac=1)
    # Prepare responses
    for i, (_, row) in enumerate(samples_shuffled.iterrows()):
        input("Press enter to print next sample...")
        print(f"Conversation {str(i).zfill(2)}\n\n")
        print("Context:\n")
        for utterance in row.context.split('\n'):
            print(utterance, '\n')
        print('\n')
        print("Responses:\n")
        random.shuffle(sample_idxs)
        for idx in sample_idxs:
            print(row[idx], '\n')
        print('\n\n\n')

    # Serialize corpus data frame
    samples.to_csv(os.path.join(args.dest_dir_path, 'human_evaluation_samples.csv'), index=False)

    return 0


if __name__ == "__main__":
    # Input arguments
    args_parser = ArgumentParser()
    args_parser.add_argument('--source_file_path', type=str,
                             default='./resources/data/dialogue_corpus/human_evaluation_samples.csv',
                             help="Path to the file containing the samples for the evaluation.")
    args_parser.add_argument('--source_dir_paths', nargs='+', type=str,
                             default=[
                                 './experiments/DLDLM_static_evaluation/dldlm_medium_emp_generative_2022_04_28_12_00_11',
                                 './experiments/ParlAI_baseline_static_evaluation/parlai_baseline_generative_2022_04_26_11_47_20'
                             ],
                             help="Path(s) to the directory(ies) containing the generated responses for the evaluation.")
    args_parser.add_argument('--evaluated_model_ids', nargs='+', type=str,
                             default=['dldlm_medium_emp', 'parlai_baseline'],
                             help="Path(s) to the directory(ies) containing the generated responses for the evaluation.")
    args_parser.add_argument('--dest_dir_path', type=str,
                             default='./resources/data/evaluation/',
                             help="Path to the directory where to store the standardised corpus.")
    # Run standardisation
    main(args_parser.parse_args(sys.argv[1:]))
