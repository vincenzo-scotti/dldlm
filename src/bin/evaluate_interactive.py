import os
import sys
from shutil import copy2
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, List, Dict, Literal

import random
import numpy as np
import torch

from dldlm.chatbot_api import DLDLMChatbot

# Variables to control model and evaluation parameters
# Environment
random_seed: Optional[int] = None
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision: bool = True
# Model
generate_kwargs: Dict = dict()
max_context_len: Optional[int]
max_response_len: Optional[int]
generative_mode: Literal['causal', 'joint'] = 'causal'
pretrained_model: str
chatbot: DLDLMChatbot
# Experiment dir path
current_experiment_dir_path: Optional[str] = None


def init_environment(config_file_path: Optional[str]):
    # Declare global variables
    global random_seed, device, mixed_precision, pretrained_model, \
        max_context_len, max_response_len, generate_kwargs, generative_mode, \
        current_experiment_dir_path
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
    # Set device
    device = torch.device(configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Load remaining configs
    pretrained_model = configs['chatbot']['pretrained']
    max_context_len = configs['chatbot'].get('max_context_len')
    max_response_len = configs['chatbot'].get('max_response_len')
    generate_kwargs = configs['chatbot']['generate_kwargs']
    generative_mode = configs['chatbot'].get('generative_mode', generative_mode)
    logging.info("Initialisation completed")


def load_model():
    # Declare global variables
    global pretrained_model, chatbot
    # Create chatbot instance
    chatbot = DLDLMChatbot(
        pretrained_model,
        mixed_precision=mixed_precision,
        max_context_len=max_context_len,
        max_response_len=max_response_len,
        generate_kwargs=generate_kwargs,
        device=device
    )
    logging.info("DLDLM chatbot initialised")


def append_turn_to_conversation_file(text: str):
    global current_experiment_dir_path
    if current_experiment_dir_path is not None:
        with open(os.path.join(current_experiment_dir_path, 'conversation.txt'), 'a') as f:
            f.write(text + '\n')


def main(args: Namespace):
    # Declare global variables
    global device, tokenizer, model, generate_kwargs
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # Create model and tokeniser
    load_model()
    # Begin conversation
    # Container for conversation history
    # The first token is the special begin of sequence token the Transformer uses to signal the begin of the input
    conversation: List[str] = []
    logging.info("Conversation history initialised")
    # Running flag
    evaluating: bool = True
    # Log evaluation start
    print("Evaluation started, press [Ctrl+C] to end the process\n\n\n\n\n\n")
    # Keep conversation running until user stops it
    while evaluating:
        # Wrap everything into a try-except block to stop on SIGTERM
        try:
            # Get input from user
            input_string = input("User: ")
            # Append latest turn to conversation history
            # Add the special end of sequence token to signal to the transformer that the turn is finished
            conversation.append(input_string)
            # Append latest turn to conversation history file
            append_turn_to_conversation_file(f"User: {input_string}")
            # Generate response
            reponse, latent_code_id = chatbot(conversation, generative_mode=generative_mode, output_latent=True)
            # Print response
            print(f"DLDLM ({latent_code_id}): {reponse}")
            # Append latest response to conversation history
            # Add the special end of sequence token to signal to the transformer that the turn is finished
            conversation.append(reponse)
            # Write response to file
            append_turn_to_conversation_file(f"DLDLM ({latent_code_id}): {reponse}")
        # In case of KeyboardInterrupt (e.g. when Ctrl+C is pressed or SIGTERM is given) stop running
        except KeyboardInterrupt:
            # Set running flag to false
            evaluating = False
            # Log exit
            print("\n\n\n\n\n\nTermination signal received, exiting...")
    # Log termination
    logging.info("Evaluation finished")
    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add argument to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str, default=None,
        help="Path to the YAML file containing the configuration for the evaluation."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
