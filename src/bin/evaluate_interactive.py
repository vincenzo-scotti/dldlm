import os
import sys
from shutil import copy2
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, List, Dict, Tuple
import re

import random
import numpy as np
import torch

from model import DLDLMTokenizer, DLDLMFullModel

# Variables to control model and evaluation parameters
# Environment
random_seed: Optional[int] = None
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision: bool = True
# Model
generation_kwargs: Dict = dict()
max_context_len: Optional[int]
pretrained_model: str
model: DLDLMFullModel
pretrained_tokenizer: str
tokenizer: DLDLMTokenizer
# Experiment dir path
current_experiment_dir_path: Optional[str] = None


def init_environment(config_file_path: Optional[str]):
    # Declare global variables
    global random_seed, device, mixed_precision, pretrained_tokenizer, \
        pretrained_model, max_context_len, generation_kwargs, current_experiment_dir_path
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
    pretrained_model = configs['dldlm']['model']['pretrained']
    max_context_len = configs['dldlm']['model'].get('max_context_len')
    generation_kwargs = configs['dldlm']['model']['generate_kwargs']
    pretrained_tokenizer = configs['dldlm']['tokeniser']['pretrained']
    logging.info("Initialisation completed")


def load_model():
    # Declare global variables
    global pretrained_tokenizer, pretrained_model, tokenizer, model
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(pretrained_tokenizer)
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = DLDLMFullModel.from_pretrained(pretrained_model)
    logging.info("DLDLM model instantiated")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")
    # Set model in evaluation mode
    model = model.eval()
    logging.info("Model set in evaluation mode")


def append_turn_to_conversation_file(text: str):
    global current_experiment_dir_path
    if current_experiment_dir_path is not None:
        with open(os.path.join(current_experiment_dir_path, 'conversation.txt'), 'a') as f:
            f.write(text + '\n')


def get_context_str(context) -> str:
    # Declare global variables
    global tokenizer, max_context_len
    # Get context string
    context_str = tokenizer.eos_token.join(context)
    # Possibly cut excess
    if max_context_len is not None:
        context_str = tokenizer.decode(tokenizer().input_ids[-max_context_len:])
    # Return context string
    return context_str


def get_latent(context_str) -> Tuple[str, int]:
    # Encode input
    input_encoding = tokenizer(context_str + '<|prior|>', return_tensors='pt').to(device)
    # Process with hidden layers
    prior_hidden_state = model.transformer(**input_encoding, return_dict=True).last_hidden_state[:, -1:]
    # Compute prior latent logits
    prior_logits = model.lm_head(prior_hidden_state)
    # Latent mask to prevent using common tokens
    latent_mask = torch.zeros_like(prior_logits)
    latent_mask[:, model.config.latent_token_ids] = 1.
    latent_mask = torch.log(latent_mask)
    # Normalisation value to enforce numerical stability in output
    shift = prior_logits[:, model.config.latent_token_ids].max().detach()
    # Compute logits with normalisation trick
    prior_logits += latent_mask - shift
    # Latent decoding step
    if generation_kwargs.get('do_sample_latent', model.config.do_sample_latent):
        latent_ids = torch.multinomial(torch.softmax(prior_logits, dim=-1), 1)
    else:
        latent_ids = torch.argmax(prior_logits, dim=-1).unsqueeze(-1)
    latent = latent_ids.squeeze(-1)
    # Retrieve latent code string and corresponding idx
    latent_str = tokenizer.decode(latent)
    latent_idx = int(re.compile('<[|]latentcode(\d+)[|]>').search(latent_str).group(1))

    return latent_str, latent_idx


def main(args: Namespace):
    # Declare global variables
    global device, tokenizer, model, generation_kwargs
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
    # Context manager for mixed precision
    with torch.autocast(device.type, enabled=mixed_precision):
        # Keep evaluation running until user stops it
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
                # Gather last n turns to be used as prompt to the model and merge them into a single string
                context_string = get_context_str(conversation)
                # Using prior predict latent action token to control response generation
                latent_code_string, latent_code_id = get_latent(context_string)
                # Encode the context using the tokeniser
                input_ids = tokenizer(context_string + latent_code_string, return_tensors='pt').input_ids.to(device)
                # Gather generated ids
                output_ids = model.generate(input_ids=input_ids, **generation_kwargs)[0, input_ids.size(1):]
                # Decode the response using the tokenizer
                output_string: str = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                # Print response
                print(f"DLDLM ({latent_code_id}): {output_string}")
                # Append latest response to conversation history
                # Add the special end of sequence token to signal to the transformer that the turn is finished
                conversation.append(output_string)
                # Write response to file
                append_turn_to_conversation_file(f"DLDLM ({latent_code_id}): {output_string}")
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
