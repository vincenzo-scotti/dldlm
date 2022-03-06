import os
import sys
from shutil import copy2
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, List, Dict

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model import DLDLMTokenizer, DLDLMAllHeadsModel


RESULTS_COLS = [
    'PPL',
    'posterior_latent',
    'policy_latent',
    'policy_behaviour_latent_probability',
    'policy_empathetic_latent_probability',
    'posterior_expected_elicited_sentiment_reward',
    'posterior_expected_elicited_response_length_reward',
    'policy_expected_elicited_sentiment_reward',
    'policy_expected_elicited_response_length_reward',
]


# Variables to control model and evaluation parameters
# Environment
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
# Model
model_configs: Dict
model: DLDLMAllHeadsModel
tokenizer_configs: Dict
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpus: pd.DataFrame
# Experiment dir path
current_experiment_dir_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, model_configs, \
        tokenizer_configs, corpus_configs, current_experiment_dir_path
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file
    with open(config_file_path) as f:
        configs_dump_str: str = f.read()
        f.seek(0)
        configs: Dict = yaml.full_load(f)
    # Create directories
    experiments_dir_path: str = configs['experiments_directory_path']
    if not os.path.exists(experiments_dir_path):
        os.mkdir(experiments_dir_path)
    experiment_series_dir_path: str = os.path.join(experiments_dir_path, configs['experiment_series'])
    if not os.path.exists(experiment_series_dir_path):
        os.mkdir(experiment_series_dir_path)
    current_experiment_dir_path: str = os.path.join(experiment_series_dir_path, configs['experiment_id'])
    if not os.path.exists(current_experiment_dir_path):
        os.mkdir(current_experiment_dir_path)
    # Create file paths
    if configs.get('log_file', False):
        log_file_path = os.path.join(
            current_experiment_dir_path, f"{configs['experiment_id']}_{date_time_experiment}.log"
        )
    else:
        log_file_path = None
    configs_dump_path = os.path.join(current_experiment_dir_path, 'configs.yaml')
    # Init logging
    logging.basicConfig(filename=log_file_path, level=configs['log_level'])
    # Start Logging info
    logging.info(f"{configs['experiment_series']} evaluation script started")
    logging.info(f"Current experiment directories created at '{current_experiment_dir_path}'")
    if log_file_path is not None:
        logging.info(f"Current experiment log created at '{log_file_path}'")
    # Set all random seeds
    random_seed = configs.get('random_seed', None)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    logging.info("Random seeds set")
    # Dump configs
    copy2(config_file_path, configs_dump_path)
    logging.info(f"Current experiment configuration dumped at '{configs_dump_path}'")
    # Set device
    device = torch.device(configs.get('device', 'gpu' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Load remaining configs
    model_configs = configs['dldlm']['model']
    tokenizer_configs = configs['dldlm']['tokeniser']
    corpus_configs = configs['data']
    logging.info("Initialisation completed")


def load_model():
    # Declare global variables
    global tokenizer_configs, model_configs, tokenizer, model
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(tokenizer_configs['pretrained'])
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = DLDLMAllHeadsModel.from_pretrained(model_configs['pretrained'])
    logging.info("DLDLM model instantiated")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")


def load_data():
    # Declare global variables
    global corpus_configs, corpus
    # Create data frame
    corpus = pd.read_csv(corpus_configs['csv_file_path'])
    # Filter only desired splits and corpora
    if corpus_configs.get('splits', None) is not None:
        corpus = corpus[corpus['split'].isin(corpus_configs['splits'])]
    if corpus_configs.get('sub_sets', None) is not None:
        corpus = corpus[corpus['data_set'].isin(corpus_configs['sub_sets'])]
    # Fix NaN issue
    corpus['context'].fillna("", inplace=True)
    logging.info("Corpus loaded")


@torch.no_grad()
@autocast(enabled=mixed_precision)
def evaluate_model():
    # Declare global variables
    global device, tokenizer, model, corpus, current_experiment_dir_path
    # Container for results
    evaluation_results = []
    # Save total number of samples
    n_elems = len(corpus)
    # Start evaluation
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time}")
    # Set model in evaluation mode
    model.eval()
    logging.info("Model set in evaluation mode")
    # Iterate over data frame rows
    for idx, (_, row) in enumerate(corpus.iterrows()):
        # Prepare input
        # Unpack
        _, _, _, _, raw_context, raw_response, _, _, _, _, _ = row
        # Context
        if len(raw_context) > 0:
            context = str().join(tokenizer.bos_token + t + tokenizer.eos_token for t in raw_context.split('\n'))
            context_ids = tokenizer(context, padding=True, return_tensors='pt').input_ids.to(device)
            context_len = context_ids.size(-1)
        else:
            context_ids, context_attention = None, None
            context_len = 0
        # Response
        response = tokenizer.bos_token + raw_response + tokenizer.eos_token
        response_ids, response_attention = tokenizer(response, padding=True, return_tensors='pt').values()
        response_ids = response_ids.to(device)
        response_attention = response_attention.to(device)
        # Target labels
        labels = response_ids.clone()
        labels[~response_attention.bool()] = -100  # Index to ignore in loss computation

        # Process sample
        model_outputs = model(
            input_ids=response_ids,
            context_ids=context_ids,
            labels=labels,
        )

        ppl: float = torch.exp(model_outputs.cost_function_output.lm_loss).squeeze().cpu().item()
        z_posterior: int = torch.argmax(model_outputs.posterior_logits, dim=-1).squeeze().cpu().item()
        z_policy: int = torch.argmax(model_outputs.policy_logits, dim=-1).squeeze().cpu().item()
        policy_p_dist: List[float] = F.softmax(model_outputs.policy_logits, dim=-1).squeeze().cpu().tolist()
        policy_z_behavior_p: float = policy_p_dist[z_posterior]
        policy_z_empathy_p: float = policy_p_dist[z_policy]
        (
            behaviour_expected_elicited_sent_rew,
            behaviour_expected_elicited_resp_len_rew
        ) = torch.tanh(model_outputs.raw_reward).squeeze().cpu().tolist()

        policy_model_outputs = model(
            past_key_values=tuple(
                (k[:, :, :context_len], v[:, :, :context_len]) for k, v in model_outputs.past_key_values
            ),
            latent_ids=torch.tensor([[model.config.latent_token_ids[z_posterior]]], dtype=torch.long, device=device),
            do_context=False,
            do_policy=False,
            do_response_encoding=False,
            do_posterior=False,
            do_response_decoding=False
        )

        (
            policy_expected_elicited_sent_rew,
            policy_expected_elicited_resp_len_rew
        ) = torch.tanh(policy_model_outputs.raw_reward).squeeze().cpu().tolist()

        # Add evaluation entry
        evaluation_results.append([
            ppl,
            z_posterior, z_policy,
            policy_z_behavior_p, policy_z_empathy_p,
            behaviour_expected_elicited_sent_rew, behaviour_expected_elicited_resp_len_rew,
            policy_expected_elicited_sent_rew, behaviour_expected_elicited_resp_len_rew
        ])

        # Log step completion
        logging.debug(f"Evaluation step {idx + 1} of {n_elems} completed")
    # Close evaluation
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of evaluation
    logging.info(f"Evaluation finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")
    # Extend data frame to include results
    corpus[RESULTS_COLS] = evaluation_results
    # Serialise results
    corpus.to_csv(os.path.join(current_experiment_dir_path, 'evaluation_results.csv'), index=False)
    logging.info(f"Evaluation results serialised at '{current_experiment_dir_path}'")


def main(args: Namespace):
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # Load model and tokeniser
    load_model()
    # Load data frame to iterate over data set
    load_data()
    # Carry on evaluation process
    evaluate_model()

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the evaluation."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
