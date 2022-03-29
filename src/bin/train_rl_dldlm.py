import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, Union, Tuple, List, Dict

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from training import LinearLR
from torch.utils.tensorboard import SummaryWriter

from data import EpisodicDialogueCorpus
from model import DLDLMTokenizer, DLDLMAllHeadsModel

# TODO add warm restart

# Constants
LOSS_KEYS_MAPPING = {
    'lm_loss': 'LM',
    'latent_kl_div': 'KL Divergence',
    'latent_kl_div_threshold': 'KL Divergence (threshold)',
    'latent_loss': 'Latent negative log-likelihood',
    'cls_loss': 'Contrastive IR',
    'bow_loss': 'BoW reconstruction',
    'rew_loss': 'Immediate reward prediction'
}
OBJ_KEYS_MAPPING = {
    'lm_obj': 'LM',
    'latent_obj': 'Latent'
}

# Variables to control model and optimization parameters
# Environment
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
writer: SummaryWriter
# Model
model_configs: Dict
reinforce_configs: Dict
model: DLDLMAllHeadsModel
tokenizer_configs: Dict
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpus: EpisodicDialogueCorpus
corpus_loader: DataLoader
# Optimisation
optimizer_configs: Dict
optimizer: Optimizer
scaler: Optional[GradScaler] = None
scheduler_configs: Dict
lr_scheduler: LinearLR
# Experiment dir path
current_experiment_dir_path: str
# Checkpoint paths
model_checkpoint_path: str
best_model_checkpoint_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, writer, model_configs, tokenizer_configs, optimizer_configs, \
        scheduler_configs, corpus_configs, current_experiment_dir_path, reinforce_configs, \
        model_checkpoint_path, best_model_checkpoint_path
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
    current_experiment_dir_path = os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time_experiment}"
    )
    if not os.path.exists(current_experiment_dir_path):
        os.mkdir(current_experiment_dir_path)
    model_dir_path: str = os.path.join(current_experiment_dir_path, 'model')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    model_checkpoint_path = os.path.join(model_dir_path, 'latest_checkpoint')
    if not os.path.exists(model_checkpoint_path):
        os.mkdir(model_checkpoint_path)
    best_model_checkpoint_path = os.path.join(model_dir_path, 'best_checkpoint')
    if not os.path.exists(best_model_checkpoint_path):
        os.mkdir(best_model_checkpoint_path)
    tb_dir_path = os.path.join(current_experiment_dir_path, 'tensorboard')
    if not os.path.exists(tb_dir_path):
        os.mkdir(tb_dir_path)
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
    logging.info(f"{configs['experiment_series']} training script started")
    logging.info(f"Current experiment directories created at '{current_experiment_dir_path}'")
    if log_file_path is not None:
        logging.info(f"Current experiment log created at '{log_file_path}'")
    # Set all random seeds
    random_seed = configs.get('random_seed', None)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    logging.info("Random seeds set")
    # Tensor-Board writer
    writer = SummaryWriter(tb_dir_path)
    logging.info(f"Tensor-Board writer created at '{tb_dir_path}'")
    # Dump configs
    copy2(config_file_path, configs_dump_path)
    writer.add_text('Configs', configs_dump_str)
    logging.info(f"Current experiment configuration dumped at '{configs_dump_path}'")
    # Set device
    device = torch.device(configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Load remaining configs
    model_configs = configs['dldlm']['model']
    reinforce_configs = configs['dldlm']['reinforce']
    tokenizer_configs = configs['dldlm']['tokeniser']
    optimizer_configs = configs['optimizer']
    scheduler_configs = configs['lr_scheduler']
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


def init_model():
    # Declare global variables
    global tokenizer_configs, model_configs, tokenizer, model
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(tokenizer_configs['pretrained'])
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = DLDLMAllHeadsModel.from_pretrained(model_configs['pretrained'], **model_configs['additional_kwargs'])
    logging.info("DLDLM model instantiated and update")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")


def init_data_loader():
    # Declare global variables
    global corpus_configs, reinforce_configs, tokenizer, corpus, corpus_loader
    # Create data set instance
    data_set: EpisodicDialogueCorpus = EpisodicDialogueCorpus(
        corpus_configs['csv_file_path'],
        tokenizer,
        data_set_split=corpus_configs['splits'],
        discount_factor=reinforce_configs['gamma'],
        reward_weights=reinforce_configs['reward_weights']
    )
    logging.info(f"Data set instantiated")
    # Create data loader instance
    data_loader: DataLoader = DataLoader(
        data_set,
        batch_size=corpus_configs['mini_batch_size'],
        num_workers=corpus_configs['n_workers'],
        shuffle=True,
        collate_fn=data_set.collate
    )
    logging.info(f"Data loader instantiated")
    # Add created data set and data loader to dict
    corpus = data_set
    logging.info(f"Data set added to global variables")
    corpus_loader = data_loader
    logging.info(f"Data loader added to global variables")


def init_optimisation_tools():
    # Declare global variables
    global optimizer_configs, scheduler_configs, mixed_precision, model, corpus_loader, optimizer, scaler, lr_scheduler
    # Create optimiser instance
    optimizer = torch.optim.AdamW(params=model.parameters(), **optimizer_configs['kwargs'])
    logging.info("Optimiser instantiated")
    # Update learning rate scheduler configs with missing info
    scheduler_configs['lr_steps'] = len(corpus_loader)
    # Create scaler if using mixed precision
    if mixed_precision:
        scaler = GradScaler()
    # Create learning rate scheduler instance
    lr_scheduler = LinearLR(optimizer, **scheduler_configs)
    logging.info("Learning rate scheduler instantiated")


@autocast(enabled=mixed_precision)
def process_mini_batch(
        context_ids: Optional[torch.LongTensor],  # Shape (batch_size, max_context_length)
        context_attentions: Optional[torch.FloatTensor],  # Shape (batch_size, max_context_length)
        response_ids: torch.LongTensor,  # Shape (batch_size, max_response_length)
        response_attentions: torch.FloatTensor,  # Shape (batch_size, max_response_length)
        distractor_ids: torch.LongTensor,  # Shape (batch_size, max_distractor_response_length)
        distractor_attentions: torch.FloatTensor,  # Shape (batch_size, max_distractor_response_length)
        labels: torch.LongTensor,  # Shape (batch_size, max_response_length)
        rewards: torch.Tensor,  # Shape (batch_size, num_rewards)
        normalised_discounted_rewards  # Shape (batch_size,)
) -> Tuple[float, float, float, Dict[str, List[float]], Dict[str, List[float]]]:
    # Declare global variables
    global corpus_configs, optimizer_configs, model, corpus_loader, optimizer, scaler, lr_scheduler
    # Compute helper params
    mini_batch_size: int = len(response_ids)
    in_mem: int = corpus_configs['in_mem']
    # Create accumulators
    hybrid_cost = 0.0
    loss: float = 0.0
    objective: float = 0.0
    losses_dict: Dict[str, float] = {}
    obj_dict: Dict[str, float] = {}
    # Move tensors to device
    context_ids = context_ids.to(device) if context_ids is not None else None
    context_attentions = context_attentions.to(device) if context_attentions is not None else None
    response_ids = response_ids.to(device)
    response_attentions = response_attentions.to(device)
    distractor_ids = distractor_ids.to(device)
    distractor_attentions = distractor_attentions.to(device)
    labels = labels.to(device)
    rewards = rewards.to(device)
    normalised_discounted_rewards = normalised_discounted_rewards.to(device)
    # Loop over sub_batches to fit in memory
    for s_idx in range(0, mini_batch_size, in_mem):
        # Get final index of current slice
        e_idx = min(mini_batch_size, s_idx + in_mem)
        # Process current elements
        model_outputs = model(
            input_ids=response_ids[s_idx:e_idx] if response_ids is not None else None,
            input_attentions=response_attentions[s_idx:e_idx] if response_attentions is not None else None,
            context_ids=context_ids[s_idx:e_idx],
            context_attentions=context_attentions[s_idx:e_idx],
            labels=labels[s_idx:e_idx],
            target_reward=rewards[s_idx:e_idx],
            distractor_ids=distractor_ids[s_idx:e_idx],
            distractor_attentions=distractor_attentions[s_idx:e_idx],
            g_return=normalised_discounted_rewards[s_idx:e_idx]
        )

        # Compute gradients
        if e_idx - s_idx != mini_batch_size:
            tmp_hybrid_cost = model_outputs.cost * (e_idx - s_idx) / mini_batch_size
            # Update accumulators
            loss += model_outputs.loss.cpu().item() * (e_idx - s_idx) / mini_batch_size
            objective += model_outputs.objective.cpu().item() * (e_idx - s_idx) / mini_batch_size
            for key in model_outputs.cost_function_output:
                if '_loss' in key or '_div' in key:
                    losses_dict[key] = (
                            losses_dict.get(key, 0.0) +
                            (model_outputs.cost_function_output[key].cpu().item() * (e_idx - s_idx) / mini_batch_size)
                    )
                elif '_obj' in key:
                    obj_dict[key] = (
                            obj_dict.get(key, 0.0) +
                            (model_outputs.cost_function_output[key].cpu().item() * (e_idx - s_idx) / mini_batch_size)
                    )
        else:
            tmp_hybrid_cost = model_outputs.cost
            # Update accumulators
            loss += model_outputs.loss.cpu().item()
            objective += model_outputs.objective.cpu().item()
            for key in model_outputs.cost_function_output:
                if '_loss' in key or '_div' in key:
                    losses_dict[key] = (
                            losses_dict.get(key, 0.0) + model_outputs.cost_function_output[key].cpu().item()
                    )
                elif '_obj' in key:
                    obj_dict[key] = (
                            obj_dict.get(key, 0.0) + model_outputs.cost_function_output[key].cpu().item()
                    )
        # Scale loss if using gradient accumulation
        # Compute gradients
        if scaler is not None:
            scaler.scale(tmp_hybrid_cost).backward()
        else:
            tmp_hybrid_cost.backward()
        hybrid_cost = tmp_hybrid_cost.cpu().item()
    # Update model weights
    # Clip gradient norm
    if optimizer_configs['max_gradient_norm'] > 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optimizer_configs['max_gradient_norm'])
    # Update weights
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # Update learning rate
    lr_scheduler.step()
    # Reset optimiser and model gradients
    optimizer.zero_grad()
    model.zero_grad()

    return hybrid_cost, loss, objective, losses_dict, obj_dict


def fit_model():
    # Declare global variables
    global optimizer_configs, writer, corpus, corpus_loader, model_checkpoint_path, best_model_checkpoint_path
    # Initialize values
    # Number of elements
    n_train_batches: int = len(corpus_loader)
    # Initialize operation counter
    global_step_counter: int = 0
    # Set model in training mode
    model.train()
    # Train and validation process
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of training
    logging.info(f"Training started - Current date and time {start_time}")
    # Iterate over epochs
    for epoch in range(optimizer_configs['n_epochs']):
        logging.info(f"Epoch {epoch + 1} of {optimizer_configs['n_epochs']} started")
        # Iterate over mini-batches
        for b_idx, mini_batch in enumerate(corpus_loader):
            # Process current mini-batch
            (
                mini_batch_hybrid_cost,
                mini_batch_loss,
                mini_batch_objective,
                mini_batch_losses_dict,
                mini_batch_obj_dict,
            ) = process_mini_batch(*mini_batch)
            # Log training info (mini-batch level)
            # Tensorboard
            writer.add_scalar('Training hybrid objective', mini_batch_hybrid_cost, global_step_counter + 1)
            writer.add_scalar('Training loss', mini_batch_loss, global_step_counter + 1)
            writer.add_scalar('Training objective', mini_batch_objective, global_step_counter + 1)
            writer.add_scalars(
                'Training Losses',
                {LOSS_KEYS_MAPPING[key]: mini_batch_losses_dict[key] for key in LOSS_KEYS_MAPPING},
                global_step_counter + 1
            )
            writer.add_scalars(
                'Training Objectives',
                {OBJ_KEYS_MAPPING[key]: mini_batch_obj_dict[key] for key in OBJ_KEYS_MAPPING},
                global_step_counter + 1
            )
            # Std output
            logging.info(
                f"Parameters updated at epoch {epoch + 1}, mini-batch {b_idx + 1} of {n_train_batches} - "
                f"Hybrid objective {mini_batch_hybrid_cost:.4f}"
            )
            # Update global step counter
            global_step_counter += 1
        # Checkpoint trained model
        model.save_pretrained(model_checkpoint_path)
        logging.info("Models saved using utilities")
        # Log end of epoch
        logging.info(f"Epoch {epoch + 1} of {optimizer_configs['n_epochs']} finished")
    # Close training
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of training
    logging.info(f"Training finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")
    # Restore best validation model weights
    model.load_state_dict(torch.load(best_model_checkpoint_path))
    logging.info("Best validation model weights restored")


def main(args: Namespace):
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # Create model and tokeniser
    init_model()
    # Create data sets and data loaders
    init_data_loader()
    # Create optimiser, scaler and scheduler
    init_optimisation_tools()
    # Training and validation process
    fit_model()
    # Clean training output if any
    clean_environment()

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the experiment."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
