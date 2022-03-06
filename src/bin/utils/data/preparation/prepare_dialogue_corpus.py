import sys
import os
import math
import random

from argparse import ArgumentParser

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from model.dldlm import DLDLMTokenizer

import spacy
from spacy.lang.en import English
from spacytextblob.spacytextblob import SpacyTextBlob  # Do not delete

import pandas as pd

SPACY_SENTIMENT_ANALYSIS = spacy.load('en_core_web_sm')
SPACY_SENTIMENT_ANALYSIS.add_pipe('spacytextblob')
SPACY_TOKENIZER = English().tokenizer

TOKEN_PAD_VALUE = '<|endoftext|>'
REWARD_PAD_VALUE = 0.0

SPLIT_LIST = ['train', 'validation', 'test']
DF_COLUMNS = [
    'split',
    'data_set',
    'conversation_id',
    'turn_id',
    'context',
    'turn',
    'distractors',
    'elicited_sentiment_reward',
    'elicited_response_length_reward',
    'elicited_sentiment_reward_trace',
    'elicited_response_length_reward_trace',
]

random_seed = 2307

context_tokens_max_length = 256
response_tokens_max_length = 128
do_lowercasing = False


def sentiment_reward(turn):
    return SPACY_SENTIMENT_ANALYSIS(turn)._.polarity


def relative_length_reward(turn, response):
    len_t = len(SPACY_TOKENIZER(turn))
    len_r = len(SPACY_TOKENIZER(response))
    try:
        return math.tanh((len_r - len_t) / len_t)
    except ZeroDivisionError:
        return 0  # TODO fix this issue


def prepare_dialogue(dialogue, tokenizer, split_id, corpus_id, conversation_id, corpus=None):
    # Utility function
    def pick_random_turn(reference_turn):
        random_turn = reference_turn
        while (reference_turn == random_turn) or (len(tokenizer(random_turn)) > response_tokens_max_length):
            idx = random.randrange(len(corpus))
            random_turn = corpus.loc[idx, 'text']
        return random_turn

    dialogue_turns = [turn['text'].lower() if do_lowercasing else turn['text'] for turn in dialogue]
    # Contexts
    dialogue_contexts = []
    tokenized_dialogue_turns_lengths = [len(elem) for elem in tokenizer(dialogue_turns).input_ids]
    for idx in range(len(dialogue_turns)):
        context = dialogue_turns[:idx]
        tokenized_context_lengths = tokenized_dialogue_turns_lengths[:idx]
        total_tokenized_context_len = sum(tokenized_context_lengths)
        while total_tokenized_context_len > context_tokens_max_length:
            total_tokenized_context_len -= tokenized_context_lengths.pop(0)
            context.pop(0)
        dialogue_contexts.append(context)
    # Rewards
    rewards = [
        [sentiment_reward(response['text']), relative_length_reward(turn['text'], response['text'])]
        for turn, response in zip(dialogue[:-1], dialogue[1:])
    ] + [[0., 0.]]
    # Reward traces
    reward_traces = [
        [
            "\n".join(str(sent_rew) for sent_rew, _ in rewards[idx:]),
            "\n".join(str(len_rew) for _, len_rew in rewards[idx:])
        ]
        for idx in range(len(rewards))
    ]
    # Distractors
    distractors = [
        turn['distractors'] if 'distractors' in turn and len(turn['distractors']) > 0 else pick_random_turn(turn['text'])
        for turn in dialogue
    ]
    # Extract samples
    prepared_dialogue = [
        [split_id, corpus_id, conversation_id, idx, "\n".join(context), turn, distractor_list] + reward + reward_trace
        for idx, (turn, context, reward, reward_trace, distractor_list)
        in enumerate(zip(dialogue_turns, dialogue_contexts, rewards, reward_traces, distractors))
        if response_tokens_max_length >= tokenized_dialogue_turns_lengths[idx] > 0 and ((not (sum(tokenized_dialogue_turns_lengths[idx - len(context):idx]) == 0)) or idx == 0)
    ]

    return prepared_dialogue


def main(args):
    global context_tokens_max_length, response_tokens_max_length, do_lowercasing, random_seed

    # Create destination directory
    if not os.path.exists(args.dest_dir_path):
        os.mkdir(args.dest_dir_path)

    # Load tokenizer
    tokenizer = DLDLMTokenizer.from_pretrained(args.tokenizer_model)
    context_tokens_max_length = args.max_context_tokens
    response_tokens_max_length = args.max_response_tokens
    do_lowercasing = args.do_lowercasing
    random_seed = args.random_seed

    # Set random seed
    random.seed(random_seed)

    # Build corpus
    data = []
    for split_id in SPLIT_LIST:
        for corpus_id in args.corpus_list:
            df = pd.read_csv(os.path.join(args.data_dir_path, corpus_id, corpus_id.lower() + '.csv'))
            try:
                df['distractors'].fillna("", inplace=True)
            except KeyError:
                pass
            data_ = (d.to_dict(orient='records') for _, d in df[df['split'] == split_id].groupby('conversation_id'))
            with parallel_backend(args.parallel_backend, n_jobs=args.n_jobs):
                data_ = Parallel(verbose=args.verbosity_level)(
                    delayed(prepare_dialogue)(d, tokenizer, split_id, corpus_id, idx, corpus=df)
                    for idx, d in enumerate(data_, start=len(data))
                )
            data += sum(data_, [])

    # Serialize corpus data frame
    df = pd.DataFrame(data, columns=DF_COLUMNS)
    df.to_csv(os.path.join(args.dest_dir_path, args.data_set_file + '.csv'), index=False)

    return 0


if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument('--data_dir_path', type=str, default='./resources/data/preprocessed/',
                             help="Path to the directory containing the preprocessed corpora.")
    args_parser.add_argument('--corpus_list', nargs='+', type=str,
                             default=['DailyDialog', 'EmpatheticDialogues', 'Persona-Chat', 'Wizard_of_Wikipedia'],
                             help="List of the directories containing the corpus to be used.")
    args_parser.add_argument('--dest_dir_path', type=str, default='./resources/data/dialogue_corpus/',
                             help="Path to the directory where to store the generated corpus "
                                  "(it will be created if it does not exist.")
    args_parser.add_argument('--data_set_file', type=str, default='corpus',
                             help="Name of the file containing the corpus to be created "
                                  "(file extension will be added automatically)")
    args_parser.add_argument('--tokenizer_model', type=str, default='gpt2',
                             help="Model of the HuggingFace \"Transformer\" package from which to take the tokenizer.")
    args_parser.add_argument('--max_context_tokens', type=int, default=context_tokens_max_length,
                             help="Maximum number of tokens in context.")
    args_parser.add_argument('--max_response_tokens', type=int, default=response_tokens_max_length,
                             help="Maximum number of tokens in context.")
    args_parser.add_argument('--do_lowercasing', type=bool, default=do_lowercasing,
                             help="Whether to do the lower-casing of the text or not.")
    args_parser.add_argument('--random_seed', type=int, default=random_seed,
                             help="Random seed for reproducibility.")
    args_parser.add_argument('--parallel_backend', type=str, default='threading',
                             help="Parallel backend to use for preprocessing (see joblib package documentation).")
    args_parser.add_argument('--n_jobs', type=int, default=-1,
                             help="Number of parallel active jobs to use for preprocessing "
                                  "(see joblib package documentation).")
    args_parser.add_argument('--verbosity_level', type=int, default=2,
                             help="Frequency of printed progress messages "
                                  "(see joblib package documentation).")

    main(args_parser.parse_args(sys.argv[1:]))
