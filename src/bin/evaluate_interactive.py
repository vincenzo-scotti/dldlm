import os
import sys
from shutil import copy2
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, List, Dict, Tuple

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.dldlm import DLDLMTokenizer, DLDLMLMHeadModel


# Variables to control model and evaluation parameters
# Environment
random_seed: Optional[int] = None
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision: bool = True
# Model
pretrained_model: str
generation_kwargs: Dict = dict()
max_context_len: int = 256
model: DLDLMLMHeadModel
pretrained_tokenizer: str
tokenizer: DLDLMTokenizer
# Experiment dir path
current_experiment_dir_path: Optional[str] = None

# TODO check out for beam sampling


def init_environment(model: Optional[str], config_file_path: Optional[str]):
    # Declare global variables
    global random_seed, device, mixed_precision, pretrained_tokenizer, \
        pretrained_model, generation_kwargs, current_experiment_dir_path
    assert model is not None or config_file_path is not None, \
        "You must specify at least one argument between 'model' and 'config_file_path'"
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #If simply using model without config use defaults
    if model is not None:
        # Init logging
        logging.basicConfig(level=logging.ERROR)
        # Start Logging info
        logging.info(f"Interactive evaluation script started")
        # Set device
        logging.info(f"Device set to '{device}'")
        # Set mixed precision
        logging.info(f"Mixed precision set to '{mixed_precision}'")
        # Load remaining configs
        pretrained_model = model
        pretrained_tokenizer = model
        logging.info("Initialisation completed")
    else:
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
        model_configs = configs['dldlm']['model']
        tokenizer_configs = configs['dldlm']['tokeniser']
        corpus_configs = configs['data']
        logging.info("Initialisation completed")


def load_model():
    # Declare global variables
    global pretrained_tokenizer, pretrained_model, tokenizer, model
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(pretrained_tokenizer)
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = DLDLMLMHeadModel.from_pretrained(pretrained_model)
    logging.info("DLDLM model instantiated")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")


def response_sorting_key(args):
    _, ppl = args
    return ppl


def append_turn_to_conversation_file(text: str):
    global current_experiment_dir_path
    if current_experiment_dir_path is not None:
        with open(os.path.join(current_experiment_dir_path, 'conversation.txt'), 'a') as f:
            f.write(text + '\n')


def get_context_len(context):
    # Declare global variables
    global tokenizer, max_context_len
    turn_lengths = [len(tokenizer(turn).input_ids) - 2 for turn in context]
    context_len = sum(turn_lengths)
    context_turns = len(context)
    while context_len > max_context_len:
        context_len -= turn_lengths.pop(0)
        context_turns -= 1
    return context_turns


@torch.no_grad()
@autocast(enabled=mixed_precision)
def main(args: Namespace):
    # Declare global variables
    global device, tokenizer, model, generation_kwargs
    # Set model in evaluation mode
    model = model.eval()
    logging.info("Model instantiated")
    # Begin conversation
    # Container for conversation history
    # The first token is the special begin of sequence token the Transformer uses to signal the begin of the input
    conversation: List[str] = []
    logging.info("Conversation history initialized")
    # Running flag
    evaluating: bool = True
    # Log evaluation start
    print("Evaluation started, press [Ctrl+C] to end the process\n\n\n\n\n\n")
    # Keep evaluation running until user stops it
    while evaluating:
        # Wrap everything into a try-except block to stop on SIGTERM
        try:
            # Get input from user
            input_string = input(">>> ")
            # Append latest turn to conversation history
            # Add the special end of sequence token to signal to the transformer that the turn is finished
            conversation.append(tokenizer.bos_token + input_string + tokenizer.eos_token)
            # Append latest turn to conversation history file
            append_turn_to_conversation_file(input_string)
            # Gather last n turns to be used as prompt to the model and merge them into a single string
            context_string = str().join(conversation[-get_context_len(conversation):])
            # Encode the input using the tokenizer
            context_ids = tokenizer(context_string, return_tensors='pt').input_ids
            # Gather generation output
            generation_output = model.generate(
                context_ids=context_ids,
                **generation_kwargs
            )
            # Select most probable response among the generated ones
            # Gather the token scores
            scores = torch.cat([token_score.unsqueeze(1) for token_score in generation_output.scores], dim=1)
            # Cut context from from output sequence
            response_len = scores.size(1) + 1
            response_ids = generation_output.sequences[:, -response_len:].clone()
            # Create a mask to isolate padding and mark it with ignore index
            valid_token_mask = response_ids[:, :-1] != tokenizer.pad_token_id
            generated_response_ids = generation_output.sequences[:, -response_len + 1:]
            generated_response_ids[~valid_token_mask] = -100
            # Compute separately the length of each output
            single_response_len = valid_token_mask.long().sum(dim=1) + 1
            # Compute PPL
            ppl_scores = torch.exp(
                F.cross_entropy(
                    scores.view(-1, scores.size(-1)),
                    generated_response_ids.reshape(-1),
                    reduction='none'
                ).reshape(-1, response_len - 1).sum(dim=1) / single_response_len)
            # Sort by increasing PPL
            response_ids, _ = sorted(zip(response_ids, ppl_scores), key=response_sorting_key).pop(0)
            # Decode the response using the tokenizer
            output_string: str = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            # Print response
            print(">>> " + output_string)
            # Append latest response to conversation history
            # Add the special end of sequence token to signal to the transformer that the turn is finished
            conversation.append(tokenizer.bos_token + output_string + tokenizer.eos_token)
            # Write response to file
            append_turn_to_conversation_file("\t\t\t\t" + output_string)
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
    # Add arguments to parser
    args_parser.add_argument(
        '--model',
        type=str, default=None,
        help="Path to a directory containing a DLDLM checkpoint or "
             "remote reference to a HuggingFace model hub checkpoint."
    )
    args_parser.add_argument(
        '--config_file_path',
        type=str, default=None,
        help="Path to the YAML file containing the configuration for the evaluation."
             "Ignored when the model is given."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
