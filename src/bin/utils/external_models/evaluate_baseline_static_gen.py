import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, Dict

import random
import numpy as np
import pandas as pd
import torch

from parlai.core.agents import create_agent
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.scripts.interactive import setup_args

# Variables to control model and evaluation parameters
# Environment
random_seed: Optional[int] = None
# Model
parlai_agent_kwargs: Dict = dict()
parlai_agent: Optional[ImageSeq2seqAgent] = None
# Data
corpus_configs: Dict
corpus: pd.DataFrame
# Experiment dir path
current_experiment_dir_path: Optional[str] = None


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, parlai_agent_kwargs, corpus_configs, current_experiment_dir_path
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file
    with open(config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Create directories
    experiments_dir_path: str = configs['experiments_directory_path']
    if not os.path.exists(experiments_dir_path):
        os.mkdir(experiments_dir_path)
    experiment_series_dir_path: str = os.path.join(experiments_dir_path, configs['experiment_series'])
    if not os.path.exists(experiment_series_dir_path):
        os.mkdir(experiment_series_dir_path)
    current_experiment_dir_path = os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time_experiment}"
    )
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
    # Load remaining configs
    parlai_agent_kwargs = configs['parlai']['model_kwargs']
    corpus_configs = configs['data']
    logging.info("Initialisation completed")


def clean_environment():
    # Declare global variables
    global current_experiment_dir_path
    # TODO search for closest in time
    # List files
    file_list = sorted(f for f in os.listdir() if f.startswith("experiment_"))
    if len(file_list) > 0:
        output_file = file_list.pop(-1)
        move(output_file, os.path.join(current_experiment_dir_path, output_file))
        logging.info("Cleaning completed")


def load_model():
    # Declare global variables
    global parlai_agent_kwargs, parlai_agent
    # Following ParlAI interaction script
    # Create parser to interpret options
    parser = setup_args()
    logging.info("ParlAI parser instantiated")
    # Parse agent options
    opt = parser.parse_kwargs(**parlai_agent_kwargs)
    logging.info("ParlAI agent options parsed")
    # Create model instance
    parlai_agent = create_agent(opt, requireModelExists=True)
    logging.info("ParlAI agent instantiated")


def load_data():
    # Declare global variables
    global corpus_configs, corpus
    # Create data frame
    corpus = pd.read_csv(corpus_configs['csv_file_path'])
    # Fix NaN issue
    corpus['context'].fillna("", inplace=True)
    logging.info("Corpus loaded")


def evaluate_model():
    # Declare global variables
    global corpus, current_experiment_dir_path
    # Container for generated responses
    responses = []
    # Save total number of samples
    n_elems = len(corpus)
    # Start evaluation
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time}")
    # Iterate over data frame rows
    for idx, (_, row) in enumerate(corpus.iterrows()):
        # Process context
        # Iterate over utterances in context
        for i, utterance in enumerate(row['context'].split('\n')):
            # Utterances at even index are from the speaker (We work with EmaptheticDialogues)
            if i % 2 == 0:
                # Observe speaker utterance
                parlai_agent.observe({'text': utterance, 'episode_done': False})
            # Utterances at odd index are from the listener (We work with EmpatheticDialogues)
            else:
                parlai_agent.self_observe({'text': utterance, 'episode_done': False})
        # Once the context have been observed, generate the response
        responses.append(parlai_agent.act()['text'])
        # Reset agent to move to next dialogue
        parlai_agent.reset()
        # Log step completion
        logging.debug(f"Evaluation step {idx + 1} of {n_elems} completed")
    # Close evaluation
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of evaluation
    logging.info(f"Evaluation finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")
    # Extend data frame to include results
    corpus['generated_response'] = responses
    # Serialise results
    corpus.to_csv(os.path.join(current_experiment_dir_path, 'evaluation_generation_responses.csv'), index=False)
    logging.info(f"Evaluation results serialised at '{current_experiment_dir_path}'")


def main(args: Namespace):
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # Load model
    load_model()
    # Load data frame to iterate over data set
    load_data()
    # Carry on evaluation process
    evaluate_model()

    clean_environment()

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
