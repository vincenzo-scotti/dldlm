import sys
import os
import random

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

import seaborn as sns
from matplotlib import pyplot as plt

RANDOM_SEED = 2307

DF_COLUMNS = [
    'split',
    'data_set',
    'conversation_id',
    'sample_idx',
    'turn_id',
    'response_idx',
    'user',  # {Speaker or Listener}
    'emotion',
    'situation',
    'context',
    'response',
    'distractors',
    'model_id',
    'evaluator_id',
    'score_id',
    'score_value'
]
SCORES_MAP = {'Not at all': 1, 'Somewhat': 3, 'Very much': 5}
SCORE_COMLUMN_IDS = ['Empathy', 'Relevance', 'Fluency']
MODEL_IDS_MAP = {
    'generated_response': 'ground_truth',
    'generated_response_dldlm_medium_emp': 'dldlm_medium_emp',
    'generated_response_parlai_baseline': 'baseline'
}
N_SKIP_QUESTIONS = 3


def main(args):
    # Create destination directory
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load samples and results
    data_df = pd.read_csv(args.data_file_path)
    responses_df = pd.read_csv(args.responses_file_path, sep=';')

    # Get answers column ids
    answer_column_ids = [c for c in data_df.columns if c.startswith('answer_')]

    # Aggregate results
    results = []
    for _, data_row in data_df.iterrows():
        for answer_id in answer_column_ids:
            for _, response_row in responses_df.iterrows():
                for score_id in SCORE_COMLUMN_IDS:
                    results.append([
                        data_row['split'],
                        data_row['data_set'],
                        data_row['conversation_id'],
                        data_row['sample_idx'],
                        data_row['turn_id'],
                        int(answer_id.split('_')[-1]),
                        data_row['user'],
                        data_row['emotion'],
                        data_row['situation'],
                        data_row['context'],
                        data_row[data_row[answer_id]],
                        data_row['distractors'],
                        MODEL_IDS_MAP[data_row[answer_id]],
                        response_row['ID'],
                        score_id.lower(),
                        SCORES_MAP[
                            response_row[
                                score_id + (
                                    str(
                                        1 + (data_row['sample_idx'] * len(answer_column_ids)) +
                                        int(answer_id.split('_')[-1])
                                    )
                                    if data_row['sample_idx'] > 0 else '')
                                ]
                        ]
                    ])

    # Serialize corpus data frame
    results = pd.DataFrame(results, columns=DF_COLUMNS)

    # Print results for table
    model_list = ['ground_truth', 'baseline', 'dldlm_medium_emp']
    score_list = ['empathy', 'fluency', 'relevance']
    emo_list = [['joyful'], ['sad'], ['joyful', 'sad']]
    emo_map = ['\\textbf{Joy}', '\\textbf{Sadness}', '\\textbf{All}']
    res = [" & ".join([emo_id] + [
        f"{results[(results['emotion'].isin(emo)) & (results['score_id'] == score) & (results['model_id'] == model)]['score_value'].mean():.2f}"
        for score in score_list for model in model_list
    ]) + " \\\\" for emo_id, emo in zip(emo_map, emo_list)]
    for r in res:
        print(r)

    # Plot distribution
    sns.displot(results, x='score_value', hue='model_id', kind='hist', col='score_id', row='emotion', multiple='dodge')
    plt.show()

    print("\n\n\n\n")

    # results[results['score_value'] == 3] = results[results['score_value'] == 3].apply(lambda x: 5)
    results['score_value'] = results['score_value'].apply(lambda x: 5 if x == 3 else x)

    # Randolph kappa free agreement
    rater_list = results['evaluator_id'].unique()
    res = [" & ".join([emo_id] + [
        f"{fleiss_kappa(aggregate_raters(np.array([results[(results['emotion'].isin(emo)) & (results['score_id'] == score) & (results['model_id'] == model) & (results['evaluator_id'] == rater)].sort_values('sample_idx')['score_value'].values for rater in rater_list]).T)[0], method='randolph'):.2f}"
        for score in score_list for model in model_list
    ]) + " \\\\" for emo_id, emo in zip(emo_map, emo_list)]
    for r in res:
        print(r)

    # Cont
    results.to_csv(os.path.join(args.dest_dir_path, 'human_evaluation_results.csv'), index=False)

    return 0


if __name__ == "__main__":
    # Input arguments
    args_parser = ArgumentParser()
    args_parser.add_argument('--data_file_path', type=str,
                             default='./resources/data/evaluation/human_evaluation_samples.csv',
                             help="Path to the file containing the samples for the evaluation.")
    args_parser.add_argument('--responses_file_path', type=str,
                             default='./resources/data/evaluation/Evaluation of Empathetic Conversations (1-16).csv',
                             help="Paths to the file containing the responses to the evaluation form.")
    args_parser.add_argument('--dest_dir_path', type=str,
                             default='./resources/data/evaluation/',
                             help="Path to the directory where to store the standardised corpus.")
    # Run standardisation
    main(args_parser.parse_args(sys.argv[1:]))
