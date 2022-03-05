import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
from collections import Counter
import yaml
from typing import Optional, Union, Tuple, List, Dict

import random
import math
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from training import LinearLR
from torch.utils.tensorboard import SummaryWriter

from data import DialogueCorpus, DataSetSplit
from model import DLDLMTokenizer, DLDLMAllHeadsModel

# TODO add warm restart

# Constants
VALIDATION_LOSS_KEYS = ['lm_loss', 'cls_loss', 'bow_loss', 'rew_loss']
PPL_NLL_LOSS_KEY = 'lm_loss'
LOSS_KEYS_MAPPING = {
    'lm_loss': 'LM',
    'latent_kl_div': 'KL Divergence',
    'latent_kl_div_threshold': 'KL Divergence (threshold)',
    'latent_loss': 'Latent negative log-likelihood',
    'cls_loss': 'Contrastive IR',
    'bow_loss': 'BoW reconstruction',
    'rew_loss': 'Immediate reward prediction'
}

# Variables to control model and optimization parameters
# Environment
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
writer: SummaryWriter
# Model
model_configs: Dict
model: DLDLMAllHeadsModel
tokenizer_configs: Dict
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpora: Dict[DataSetSplit, DialogueCorpus]
corpus_loaders: Dict[DataSetSplit, DataLoader]
# Optimisation
optimizer_configs: Dict
optimizer: Optimizer
scaler: Optional[GradScaler] = None
scheduler_configs: Dict
lr_scheduler: LinearLR
evaluation_configs: Dict
# Experiment dir path
current_experiment_dir_path: str
# Checkpoint paths
model_checkpoint_path: str
best_model_checkpoint_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, writer, model_configs, tokenizer_configs, optimizer_configs, \
        scheduler_configs, evaluation_configs, corpus_configs, \
        current_experiment_dir_path, model_checkpoint_path, best_model_checkpoint_path
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
    model_dir_path: str = os.path.join(current_experiment_dir_path, 'model')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    model_checkpoint_path: str = os.path.join(model_dir_path, 'latest_checkpoint')
    if not os.path.exists(model_checkpoint_path):
        os.mkdir(model_checkpoint_path)
    best_model_checkpoint_path: str = os.path.join(model_dir_path, 'best_checkpoint')
    if not os.path.exists(best_model_checkpoint_path):
        os.mkdir(best_model_checkpoint_path)
    tb_dir_path: str = os.path.join(current_experiment_dir_path, 'tensorboard')
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
    device = torch.device(configs.get('device', 'gpu' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Load remaining configs
    model_configs = configs['dldlm']['model']
    tokenizer_configs = configs['dldlm']['tokeniser']
    optimizer_configs = configs['optimizer']
    scheduler_configs = configs['lr_scheduler']
    evaluation_configs = configs['evaluation']
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
    tokenizer = DLDLMTokenizer.from_pretrained(
        tokenizer_configs['pretrained']
    ).extend_from_gpt2_tokenizer(
        tokenizer_configs['n_styles']
    )
    logging.info("Tokeniser instantiated and extended")
    # Create model instance
    model = DLDLMAllHeadsModel.from_pretrained(
        model_configs['pretrained'], **model_configs['additional_kwargs']
    ).extend_from_gpt2_initialization(
        tokenizer
    )
    logging.info("DLDLM model instantiated and extended")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")


def init_data_loaders():
    # Declare global variables
    global corpus_configs, tokenizer, corpora, corpus_loaders
    # Output dict
    corpus_loaders = {}
    for split in corpus_configs['splits']:
        # Create data set instance
        data_set: DialogueCorpus = DialogueCorpus(corpus_configs['csv_file_path'], tokenizer, split)
        logging.info(f"{split.capitalize()} data set instantiated")
        # Create data loader instance
        data_loader: DataLoader = DataLoader(
            data_set,
            batch_size=corpus_configs['splits'][split]['mini_batch_size'],
            num_workers=corpus_configs['splits'][split]['n_workers'],
            shuffle=split == DataSetSplit.TRAIN,
            collate_fn=data_set.collate
        )
        logging.info(f"{split.capitalize()} data loader instantiated")
        # Add created data set and data loader to dict
        corpora[DataSetSplit(split)] = data_set
        logging.info(f"{split.capitalize()} data set added to dictionary")
        corpus_loaders[DataSetSplit(split)] = data_loader
        logging.info(f"{split.capitalize()} data loader added to dictionary")
    logging.info("All data loaders instantiated")


def init_optimisation_tools():
    # Declare global variables
    global optimizer_configs, scheduler_configs, mixed_precision, model, corpus_loaders, optimizer, scaler, lr_scheduler
    # Create optimiser instance
    optimizer = torch.optim.AdamW(params=model.parameters(), **optimizer_configs['kwargs'])
    logging.info("Optimiser instantiated")
    # Update learning rate scheduler configs with missing info
    scheduler_configs['lr_steps'] = len(corpus_loaders[DataSetSplit('train')])
    # Create scaler if using mixed precision
    if mixed_precision:
        scaler = GradScaler()
    # Create learning rate scheduler instance
    lr_scheduler = LinearLR(**scheduler_configs)
    logging.info("Learning rate scheduler instantiated")


@autocast(enabled=mixed_precision)
def process_mini_batch(
        split: str,
        context_ids: Optional[torch.LongTensor],  # Shape (batch_size, max_context_length)
        context_attentions: Optional[torch.FloatTensor],  # Shape (batch_size, max_context_length)
        response_ids: torch.LongTensor,  # Shape (batch_size, max_response_length)
        response_attentions: torch.FloatTensor,  # Shape (batch_size, max_response_length)
        distractor_ids: torch.LongTensor,  # Shape (batch_size, max_distractor_response_length)
        distractor_attentions: torch.FloatTensor,  # Shape (batch_size, max_distractor_response_length)
        labels: torch.LongTensor,  # Shape (batch_size, max_response_length)
        rewards: torch.Tensor  # Shape (batch_size, num_rewards)
) -> Union[Tuple[float, Dict[str, float]], Tuple[List[float], Dict[str, List[float]], List[int], List[int]]]:
    # Declare global variables
    global corpus_configs, optimizer_configs, model, corpus_loaders, optimizer, scaler, lr_scheduler
    # Compute helper params
    mini_batch_size: int = len(response_ids)
    in_mem: int = corpus_configs['splits'][split]['in_mem']
    # Create accumulators
    loss: float = 0.0 if model.training else []
    losses_dict: Dict[str, float] = {}
    latents: Optional[List[int]] = None if model.training else []
    policy_predictions: Optional[List[int]] = None if model.training else []
    # Move tensors to device
    context_ids = context_ids.to(device) if context_ids is not None else None
    context_attentions = context_attentions.to(device) if context_attentions is not None else None
    response_ids = response_ids.to(device)
    response_attentions = response_attentions.to(device)
    distractor_ids = distractor_ids.to(device)
    distractor_attentions = distractor_attentions.to(device)
    labels = labels.to(device)
    rewards = rewards.to(device)
    # Loop over sub_batches to fit in memory
    for s_idx in range(0, mini_batch_size, in_mem):
        # Get final index of current slice
        e_idx = min(mini_batch_size, s_idx + in_mem)
        # Process current elements
        model_outputs = model(
            input_ids=response_ids,
            input_attentions=response_attentions,
            context_ids=context_ids,
            context_attentions=context_attentions,
            labels=labels,
            target_rewards=rewards,
            distractor_ids=distractor_ids,
            distractor_attentions=distractor_attentions,
            reduction=model.training
        )
        # Compute gradients if model is training
        if model.training:
            tmp_loss = model_outputs.cost
            tmp_losses_dict = {
                key: model_outputs.cost_function_output[key].cpu().item()
                for key in model_outputs.cost_function_output
                if '_loss' in key or '_div' in key
            }
            # Scale loss if using gradient accumulation
            if e_idx - s_idx != mini_batch_size:
                tmp_loss *= (e_idx - s_idx) / mini_batch_size
                for key in tmp_losses_dict:
                    tmp_losses_dict[key] *= (e_idx - s_idx) / mini_batch_size
            # Compute gradients
            if scaler is not None:
                scaler.scale(tmp_loss).backward()
            else:
                tmp_loss.backward()
            tmp_loss = tmp_loss.cpu().item()
        # Else collect predicted latents
        else:
            tmp_loss = model_outputs.cost.cpu().tolist()
            tmp_losses_dict = {
                key: model_outputs.cost_function_output[key].cpu().tolist()
                for key in model_outputs.cost_function_output
                if '_loss' in key or '_div' in key
            }
            latents += model_outputs.latent.cpu().tolist()
            policy_predictions += torch.argmax(model_outputs.policy_logits).cpu().tolist()
        # Update accumulators
        loss += tmp_loss
        for key in tmp_losses_dict:
            losses_dict[key] = losses_dict.get(key, []) + tmp_losses_dict[key]
    # Update weights model if training
    if model.training:
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
        # Return losses and mini-batch size
        return loss, losses_dict
    # Else if evaluating retain additional info
    else:
        return loss, losses_dict, latents, policy_predictions


@torch.no_grad()
def process_evaluation(
        split: str
) -> str:
    # Declare global variables
    global corpora, corpus_loaders, model
    # Get number of elements
    n_elements: int = len(corpora[DataSetSplit(split)])
    # Initialize validation accumulators
    ppl: Union[float, List[float]] = []
    latents: List[int] = []
    policy_predictions: List[int] = []
    # Perform evaluation step
    # Iterate over validation mini batches
    for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit(split)]):
        # Process current mini-batch
        _, tmp_losses_dict, tmp_latents, tmp_policy_predictions = process_mini_batch(
            'validation', *mini_batch,
        )
        # Update accumulators
        # PPL
        ppl += [math.exp(nll) for nll in tmp_losses_dict[PPL_NLL_LOSS_KEY]]
        # Latents
        latents += tmp_latents
        policy_predictions += tmp_policy_predictions
    # Average score
    ppl = sum(ppl) / n_elements
    # Compute classification report
    latent_report = classification_report(latents, policy_predictions, labels=range(model.config.num_styles))
    # Final report
    output_report = f"\t\t\t\t{split.upper()}\n" \
                    f"\n" \
                    f"\t\t\t\tLanguage generation\n" \
                    f"         PPL      {ppl:.2f}\n" \
                    f"\n" \
                    f"\t\t\t\tPolicy\n" \
                    f"{latent_report}\n" \
                    f"\n"

    return output_report


def fit_model():
    # Declare global variables
    global optimizer_configs, writer, corpora, corpus_loaders, model_checkpoint_path, best_model_checkpoint_path
    # Initialize values
    # Initialize train accumulators
    # Number of elements
    n_train_elems: int = len(corpora[DataSetSplit('train')])
    n_train_batches: int = len(corpus_loaders[DataSetSplit('train')])
    n_validation_elements: int = len(corpora[DataSetSplit('validation')])
    # Initialize operation counter
    global_step_counter: int = 0
    # Initialize best validation score
    best_validation_loss: float = float('inf')
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
        for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit('train')]):
            # Process current mini-batch
            mini_batch_loss, mini_batch_losses_dict = process_mini_batch('train', *mini_batch)
            # Log training info (mini-batch level)
            # Tensorboard
            writer.add_scalar('Training Loss (mini-batch)', mini_batch_loss, global_step_counter + 1)
            writer.add_scalars(
                'Training Losses',
                {LOSS_KEYS_MAPPING[key]: mini_batch_losses_dict[key] for key in LOSS_KEYS_MAPPING},
                global_step_counter + 1
            )
            # Std output
            logging.info(
                f"Parameters updated at epoch {epoch + 1}, mini-batch {b_idx + 1} of {n_train_batches} - "
                f"Loss {mini_batch_loss:.4f}"
            )
            # Do validation step if required
            if (b_idx > 0 and b_idx % evaluation_configs['validation_period'] == 0) or b_idx + 1 == n_train_elems:
                # Log start of validation
                logging.info(
                    f"Validation started at epoch {epoch + 1}, mini-batch {b_idx + 1} of {n_train_batches} - "
                )
                model.eval()
                logging.info("Model set in evaluation mode")
                with torch.no_grad():
                    # Initialize validation accumulators
                    # Loss
                    validation_loss: Union[float, List[float]] = 0.0
                    validation_losses_dict: Dict[str, Union[float, List[float]]] = {}
                    latents: List[int] = []
                    policy_predictions: List[int] = []
                    # Perform validation step
                    # Iterate over validation mini batches
                    for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit('validation')]):
                        # Process current mini-batch
                        tmp_loss, tmp_losses_dict, tmp_latents, tmp_policy_predictions = process_mini_batch(
                            'validation', *mini_batch,
                        )
                        # Update accumulators
                        # Validation loss
                        validation_loss += tmp_loss
                        for key in tmp_losses_dict:
                            validation_losses_dict[key] = validation_losses_dict.get(key, []) + tmp_losses_dict[key]
                        # Latents
                        latents += tmp_latents
                        policy_predictions += tmp_policy_predictions
                    # Average scores
                    validation_loss = sum(validation_loss) / n_validation_elements
                    for key in validation_losses_dict:
                        validation_losses_dict[key] = sum(validation_losses_dict[key]) / n_validation_elements
                    writer.add_scalar('Validation Loss', validation_loss, global_step_counter + 1)
                    writer.add_scalars(
                        'Validation Losses',
                        {LOSS_KEYS_MAPPING[key]: validation_losses_dict[key] for key in LOSS_KEYS_MAPPING},
                        global_step_counter + 1
                    )
                    writer.add_scalars(
                        'Posterior Latents Distribution',
                        {str(z_idx): z_counts for z_idx, z_counts in Counter(latents).items()},
                        global_step_counter + 1
                    )
                    writer.add_scalars(
                        'Prior Latents Distribution',
                        {str(z_idx): z_counts for z_idx, z_counts in Counter(policy_predictions).items()},
                        global_step_counter + 1
                    )
                    # Checkpoint best model if loss improves
                    # Compute loss for validation
                    tmp_validation_loss = sum(validation_losses_dict[key] for key in VALIDATION_LOSS_KEYS)
                    if tmp_validation_loss <= best_validation_loss:
                        # Save model state dictionary
                        model.save_pretrained(best_model_checkpoint_path)
                        # Update best score
                        best_validation_loss = tmp_validation_loss
                        # Log update
                        logging.info("Validation loss improved, model checkpoint triggered")
                    # Log end of validation
                    logging.info(
                        f"Validation completed at epoch {epoch + 1}, mini-batch {b_idx + 1} of {n_train_batches} - "
                        f"Validation loss {validation_loss:.4f}"
                    )
                model.train()
                logging.info("Model set in training mode")
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


def evaluate_model():
    # Declare global variables
    global writer
    # Start evaluation
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time}")
    # Log start on validation set
    logging.info(f"Validation set evaluation started")
    # Compute summary report on validation set
    validation_report: str = process_evaluation('validation')
    # Log end on validation set
    logging.info(f"Validation set evaluation finished")
    logging.info(validation_report)
    # Print test results
    print(validation_report)
    # Log test results in TensorBoard
    writer.add_text('Validation set evaluation results', validation_report)
    # Log start on test set
    logging.info(f"Test set evaluation started")
    # Compute summary report on test set
    test_report: str = process_evaluation('test')
    # Log end on test set
    logging.info(f"Test set evaluation finished")
    logging.info(test_report)
    # Print test results
    print(test_report)
    # Log test results in TensorBoard
    writer.add_text('Test set evaluation results', test_report)
    # Log test results in TensorBoard
    # Close evaluation
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of evaluation
    logging.info(f"Evaluation finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")


def main(args: Namespace):
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # create model and tokeniser
    init_model()
    # Create data sets and data loaders
    init_data_loaders()
    # Create optimiser, scaler and scheduler
    init_optimisation_tools()
    # Training and validation process
    fit_model()
    # Testing process
    evaluate_model()
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
