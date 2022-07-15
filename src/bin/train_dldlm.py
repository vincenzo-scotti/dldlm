import os
import sys
from shutil import copy2, move
import re
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, Union, Tuple, List, Dict, Pattern

import random
import math
import numpy as np
from training import get_latent_word_counts, get_traces, get_latents_count, get_response_samples
from training import log_word_counts, log_traces, log_latents_count, log_generated_response
from training import plot_word_counts, plot_traces
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from training import LinearLR, BetaCyclicalAnnealer
from torch.utils.tensorboard import SummaryWriter

from data import PreTrainingCorpus, DataSetSplit
from model import DLDLMTokenizer, DLDLMFullModel
# TODO add fine tuning script
# TODO add LR plot
# TODO add warm restart

# Constants
VALIDATION_OBJECTIVE_KEY = 'elbo'
PPL_NLL_LOSS_KEY = 'lm_loss'
ELBO_OBJECTIVE_KEY = 'elbo'
KL_DIVERGENCE_LOSS_KEY = 'latent_kl_div_loss'
LOSS_KEYS_MAPPING = {
    'lm_loss': 'Language Modelling',
    'latent_kl_div_loss': 'Latent KL Divergence',
    'latent_kl_threshold_loss': 'Latent KL Divergence with threshold',
    'prior_latent_nll_loss': 'Latent Prior Negative Log-Likelihood',
    'posterior_latent_nll_loss': 'Latent Posterior Negative Log-Likelihood',
    'elbo': 'Evidence Lower BOund',
    'prior_dist_entropy': 'Prior Latent Distribution Entropy',
    'posterior_dist_entropy': 'Posterior Latent Distribution Entropy',
    'tf_loss': 'Term-Frequency Cross-Entropy',
}

LATENT_REGEX: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>')
MARKDOWN_BREAKLINE_REGEX: Pattern[str] = re.compile(r'([^\n])[\n]([^\n])')


# Variables to control model and optimization parameters
# Environment
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
writer: SummaryWriter
# Model
model_configs: Dict
model: DLDLMFullModel
tokenizer_configs: Dict
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpora: Dict[DataSetSplit, PreTrainingCorpus] = dict()
corpus_loaders: Dict[DataSetSplit, DataLoader] = dict()
# Optimisation
optimizer_configs: Dict
optimizer: Optimizer
scaler: Optional[GradScaler] = None
lr_scheduler_configs: Dict
lr_scheduler: LinearLR
beta_scheduler_configs: Dict
beta_scheduler: BetaCyclicalAnnealer
evaluation_configs: Dict
# Experiment dir path
latent_count_txt_dir_path: str
current_experiment_dir_path: str
count_plots_dir_path: str
trace_plots_dir_path: str
count_txt_dir_path: str
trace_txt_dir_path: str
sample_responses_txt_dir_path: str
# Checkpoint paths
model_checkpoint_path: str
best_model_checkpoint_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, writer
    global model_configs, tokenizer_configs, corpus_configs, \
        optimizer_configs, lr_scheduler_configs, beta_scheduler_configs, evaluation_configs
    global current_experiment_dir_path, latent_count_txt_dir_path, count_plots_dir_path, trace_plots_dir_path, \
        count_txt_dir_path, trace_txt_dir_path, sample_responses_txt_dir_path, \
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
    plots_dir_path = os.path.join(current_experiment_dir_path, 'plots')
    if not os.path.exists(plots_dir_path):
        os.mkdir(plots_dir_path)
    count_plots_dir_path = os.path.join(plots_dir_path, 'counts')
    if not os.path.exists(count_plots_dir_path):
        os.mkdir(count_plots_dir_path)
    trace_plots_dir_path = os.path.join(plots_dir_path, 'traces')
    if not os.path.exists(trace_plots_dir_path):
        os.mkdir(trace_plots_dir_path)
    txt_file_dir_path = os.path.join(current_experiment_dir_path, 'texts')
    if not os.path.exists(txt_file_dir_path):
        os.mkdir(txt_file_dir_path)
    latent_count_txt_dir_path = os.path.join(txt_file_dir_path, 'latent_counts')
    if not os.path.exists(latent_count_txt_dir_path):
        os.mkdir(latent_count_txt_dir_path)
    count_txt_dir_path = os.path.join(txt_file_dir_path, 'word_counts')
    if not os.path.exists(count_txt_dir_path):
        os.mkdir(count_txt_dir_path)
    trace_txt_dir_path = os.path.join(txt_file_dir_path, 'traces')
    if not os.path.exists(trace_txt_dir_path):
        os.mkdir(trace_txt_dir_path)
    sample_responses_txt_dir_path = os.path.join(txt_file_dir_path, 'sample_responses')
    if not os.path.exists(sample_responses_txt_dir_path):
        os.mkdir(sample_responses_txt_dir_path)
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
    writer.add_text('Configs', MARKDOWN_BREAKLINE_REGEX.sub(r'\1<br/>\2', configs_dump_str))
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
    optimizer_configs = configs['optimizer']
    lr_scheduler_configs = configs['lr_scheduler']
    beta_scheduler_configs = configs['beta_scheduler']
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
    global tokenizer_configs, model_configs, tokenizer, model, model_checkpoint_path, best_model_checkpoint_path
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(
        tokenizer_configs['pretrained']
    ).extend_from_gpt2_tokenizer(
        tokenizer_configs['n_styles']
    )
    logging.info("Tokeniser instantiated and extended")
    # Serialise tokeniser
    tokenizer.save_pretrained(model_checkpoint_path)
    tokenizer.save_pretrained(best_model_checkpoint_path)
    logging.info("Tokeniser serialised in checkpoint directories")
    # Create model instance
    model = DLDLMFullModel.from_pretrained(
        model_configs['pretrained'], **model_configs.get('kwargs', dict())
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
        data_set: PreTrainingCorpus = PreTrainingCorpus(
            corpus_configs['corpora_dir_path'],
            tokenizer,
            split,
            corpus_configs['cache_dir_path'],
            **corpus_configs.get('kwargs', dict())
        )
        logging.info(f"{split.capitalize()} data set instantiated")
        # Create data loader instance
        data_loader: DataLoader = DataLoader(
            data_set,
            batch_size=corpus_configs['splits'][split]['mini_batch_size'],
            num_workers=corpus_configs['splits'][split]['n_workers'],
            shuffle=DataSetSplit(split) == DataSetSplit.TRAIN,
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
    global optimizer_configs, lr_scheduler_configs, beta_scheduler_configs, beta_scheduler, \
        mixed_precision, optimizer, scaler, lr_scheduler, model, corpus_loaders
    # Create optimiser instance
    optimizer = torch.optim.AdamW(params=model.parameters(), **optimizer_configs['kwargs'])
    logging.info("Optimiser instantiated")
    # Get total number of training steps
    steps = (len(corpus_loaders[DataSetSplit('train')]) * optimizer_configs['n_epochs']) + 1
    # Update learning rate scheduler configs with missing info
    lr_scheduler_configs['lr_steps'] = steps
    # Create scaler if using mixed precision
    if mixed_precision:
        scaler = GradScaler()
        logging.info("Gradient scaler for mixed precision instantiated")
    # Create learning rate scheduler instance
    lr_scheduler = LinearLR(optimizer, **lr_scheduler_configs)
    logging.info("Learning rate scheduler instantiated")
    # Create beta scheduler
    beta_scheduler = BetaCyclicalAnnealer(steps, **beta_scheduler_configs)
    logging.info("Beta cyclical annealing scheduler instantiated")


@autocast(enabled=mixed_precision)
def process_mini_batch(
        split: str,
        input_ids: torch.LongTensor,  # Shape (batch_size, )
        attention_mask: torch.LongTensor,  # Shape (batch_size, length)
        labels: torch.LongTensor,  # Shape (batch_size, response_length)
        raw_data: List[Dict]
) -> Union[Tuple[float, Dict[str, float]], Tuple[List[float], Dict[str, List[float]], List[Dict]]]:
    # Declare global variables
    global corpus_configs, optimizer_configs, model, corpus_loaders, optimizer, scaler, lr_scheduler
    global latent_count_txt_dir_path, current_experiment_dir_path, count_plots_dir_path, \
        trace_plots_dir_path, count_txt_dir_path, trace_txt_dir_path, sample_responses_txt_dir_path
    # Compute helper params
    mini_batch_size: int = len(input_ids)
    in_mem: int = corpus_configs['splits'][split]['in_mem']
    # Create accumulators
    loss: Union[float, List[float]] = 0.0 if model.training else list()
    losses_dict: Dict[str, Union[float, List[float]]] = {
        key: 0.0 if model.training else list() for key in LOSS_KEYS_MAPPING
    }
    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    # Beta scaling factor
    beta = beta_scheduler.get_beta() if model.training else 1.0
    # Loop over sub_batches to fit in memory
    idxs = ((idx, min(mini_batch_size, idx + in_mem)) for idx in range(0, mini_batch_size, in_mem))
    for s_idx, e_idx in idxs:
        # Process current elements
        model_outputs = model(
            input_ids=input_ids[s_idx:e_idx],
            attention_mask=attention_mask[s_idx:e_idx],
            labels=labels[s_idx:e_idx],
            latent_loss_weight=beta,
            reduction=model.training
        )
        # Compute gradients if model is training
        if model.training:
            tmp_loss = model_outputs.loss
            tmp_losses_dict = {
                key: model_outputs.loss_function_output[key].cpu().item()
                for key in losses_dict
            }
            # Scale loss if using gradient accumulation
            if e_idx - s_idx != mini_batch_size:
                scaling_factor = (e_idx - s_idx) / mini_batch_size
                tmp_loss *= scaling_factor
                for key in tmp_losses_dict:
                    tmp_losses_dict[key] *= scaling_factor
            # Compute gradients
            if scaler is not None:
                scaler.scale(tmp_loss).backward()
            else:
                tmp_loss.backward()
            tmp_loss = tmp_loss.cpu().item()
            # Update accumulators
            loss += tmp_loss
            for key in losses_dict:
                losses_dict[key] += tmp_losses_dict[key]
        # Else update accumulators collect predicted latents
        else:
            loss += model_outputs.loss.cpu().tolist()
            for key in losses_dict:
                losses_dict[key] += model_outputs.loss_function_output[key].cpu().tolist()
            for sample, latent in zip(raw_data[s_idx:e_idx], model_outputs.latent.cpu().tolist()):
                sample['latent'] = tokenizer.decode(latent)
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
        return loss, losses_dict, raw_data


@torch.no_grad()
def process_evaluation(
        split: str,
        dir_name: str,
        file_suffix: str,
        sub_tag: str,
        step_idx: Optional[int] = None,
        best_validation_score: Optional[float] = None
) -> Union[str, Tuple[float, float, float, float]]:
    # Declare global variables
    global corpora, corpus_loaders, model, model_configs, evaluation_configs
    # Current step
    step = step_idx + 1 if step_idx is not None else None
    # Get number of elements
    n_elements: int = len(corpora[DataSetSplit(split)])
    # Initialize validation accumulators
    validation_loss: Union[float, List[float]] = list()
    validation_losses_dict: Dict[str, Union[float, List[float]]] = {key: list() for key in LOSS_KEYS_MAPPING}
    ppl: Union[float, List[float]] = list()
    elbo: Union[float, List[float]] = list()
    kl_divergence: Union[float, List[float]] = list()
    # Processed samples
    processed_data: List[Dict] = list()
    # Perform evaluation step
    # Iterate over validation mini batches
    for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit(split)]):
        # Process current mini-batch
        tmp_loss, tmp_losses_dict, processed_mini_batch = process_mini_batch(split, *mini_batch)
        # Update accumulators
        # Validation loss
        validation_loss += tmp_loss
        for key in validation_losses_dict:
            validation_losses_dict[key] += tmp_losses_dict[key]
        ppl += [math.exp(nll) for nll in tmp_losses_dict[PPL_NLL_LOSS_KEY]]
        elbo += tmp_losses_dict[ELBO_OBJECTIVE_KEY]
        kl_divergence += tmp_losses_dict[KL_DIVERGENCE_LOSS_KEY]
        processed_data += processed_mini_batch
    # Average score
    validation_loss = sum(validation_loss) / n_elements
    validation_losses_dict = {key: sum(validation_losses_dict[key]) / n_elements for key in validation_losses_dict}
    ppl = sum(ppl) / n_elements
    elbo = sum(elbo) / n_elements
    kl_divergence = sum(kl_divergence) / n_elements
    # Log losses
    writer.add_scalar(f'Loss/{sub_tag}', validation_loss, step)
    writer.add_scalars(
        f'Metrics/{sub_tag}',
        {LOSS_KEYS_MAPPING[key]: validation_losses_dict.get(key, 0.0) for key in LOSS_KEYS_MAPPING},
        step
    )
    # Metrics and samples for visualisation
    latent_counts = get_latents_count(processed_data)
    word_counts = get_latent_word_counts(processed_data)
    traces = get_traces(processed_data, **evaluation_configs['metrics']['traces'])
    response_samples = get_response_samples(
        processed_data,
        model,
        tokenizer,
        device,
        **evaluation_configs['metrics']['sample_responses'],
        mixed_precision=mixed_precision,
        **model_configs['generate_kwargs']
    )
    # Log and plot metrics
    # Latents count
    log_latents_count(
        latent_counts,
        tb_writer=writer,
        sub_tag=sub_tag,
        step=step,
        dest_dir=os.path.join(latent_count_txt_dir_path, dir_name),
        file_name=f'latent_code_occurrences_{file_suffix}.txt'
    )
    writer.add_scalars(
        f'Latents (Posterior) Distribution/{sub_tag}',
        {z: count for z, count in latent_counts.items()},
        step
    )
    # Word counts
    log_word_counts(
        word_counts,
        tb_writer=writer,
        sub_tag=sub_tag,
        step=step,
        dest_dir=os.path.join(count_txt_dir_path, dir_name),
        file_name=f'latent_word_counts_{file_suffix}.txt'
    )
    plot_word_counts(
        word_counts,
        tb_writer=writer,
        sub_tag=sub_tag,
        step=step,
        dest_dir=os.path.join(count_plots_dir_path, dir_name),
        file_name=f'action_word_counts_{file_suffix}.pdf'
    )
    # Action traces
    log_traces(
        traces,
        sub_tag=sub_tag,
        step=step,
        tb_writer=writer,
        dest_dir=os.path.join(trace_txt_dir_path, dir_name),
        file_name=f'latent_traces_{file_suffix}.txt'
    )
    plot_traces(
        traces,
        sub_tag=sub_tag,
        step=step,
        tb_writer=writer,
        dest_dir=os.path.join(trace_plots_dir_path, dir_name),
        file_name=f'latent_traces_{file_suffix}.pdf'
    )
    # Generated samples
    log_generated_response(
        response_samples,
        sub_tag=sub_tag,
        step=step,
        tb_writer=writer,
        dest_dir=os.path.join(sample_responses_txt_dir_path, dir_name),
        file_name=f'sample_responses_{file_suffix}.txt'
    )
    # If this is the standard validation process check for best model
    if best_validation_score is not None:
        if validation_losses_dict[VALIDATION_OBJECTIVE_KEY] >= best_validation_score:  # ELBO must increase for an improvement
            # Save model state dictionary
            model.save_pretrained(best_model_checkpoint_path)
            # Update best score
            best_validation_score = validation_losses_dict[VALIDATION_OBJECTIVE_KEY]
            # Log update
            logging.info("Validation objective improved, model checkpoint triggered")

        return best_validation_score, ppl, elbo, kl_divergence
    # Else do the final report
    else:
        output_report = f"Evaluation (split: {split})\n" \
                        f"\tPPL: {ppl:.4f}\n" \
                        f"\tELBO: {elbo:.4f}\n" \
                        f"\tKL Divergence: {kl_divergence:.4f}\n"
        writer.add_text(f'Final report/{split}', MARKDOWN_BREAKLINE_REGEX.sub(r'\1<br/>\2', output_report))

        return output_report


def fit_model():
    # Declare global variables
    global optimizer_configs, writer, corpora, corpus_loaders, model_checkpoint_path, best_model_checkpoint_path

    # Define helper function to call validation
    def evaluation_step() -> float:
        # Log start of validation
        logging.info(
            f"Validation started at epoch {epoch + 1}/{n_epochs}, mini-batch {b_idx + 1}/{n_train_batches}"
        )
        # Set model in evaluation mode
        model.eval()
        logging.info("Model set in evaluation mode")
        # Do validation step
        best_score, ppl, elbo, kl_divergence = process_evaluation(
            'validation',
            'validation',
            f'validation_{str(validation_idx).zfill(4)}',
            'Validation',
            step_idx=step_idx,
            best_validation_score=best_validation_score
        )
        # Log end of validation
        logging.info(
            f"Validation completed - PPL: {ppl:.4f}, ELBO: {elbo:.4f}, KL Divergence: {kl_divergence:.4f}"
        )
        # Set model back in training mode
        model.train()
        logging.info("Model set in training mode")

        return best_score

    # Initialize values
    # Initialize train accumulators
    # Number of elements
    n_epochs = optimizer_configs['n_epochs']
    n_train_batches: int = len(corpus_loaders[DataSetSplit('train')])
    # Initialize operation counter
    step_idx: int = 0
    validation_idx: int = 0
    # Initialize best validation score
    best_validation_score: float = -float('inf')
    # Set model in training mode
    model.train()
    # Train and validation process
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of training
    logging.info(f"Training started - Current date and time {start_time}")
    # Iterate over epochs
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs} started")
        # Iterate over mini-batches
        for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit('train')]):
            # Do validation step if required
            if b_idx % evaluation_configs['validation_period'] == 0:
                # Run validation step
                best_validation_score = evaluation_step()
                # Update validation counter
                validation_idx += 1
            # Process current mini-batch
            mini_batch_loss, mini_batch_losses_dict = process_mini_batch('train', *mini_batch)
            # Log training info (mini-batch level)
            # Tensorboard
            writer.add_scalar('Loss/Training', mini_batch_loss, step_idx + 1)
            writer.add_scalars(
                'Metrics/Training',
                {LOSS_KEYS_MAPPING[key]: mini_batch_losses_dict.get(key, 0.0) for key in LOSS_KEYS_MAPPING},
                step_idx + 1
            )
            # Std output
            logging.info(
                f"Parameters updated at epoch {epoch + 1}/{n_epochs}, mini-batch {b_idx + 1}/{n_train_batches} - "
                f"Loss {mini_batch_loss:.4f}"
            )
            # Update global step counter
            step_idx += 1
        # Update cls head weights
        model.update_cls_weights()
        # Checkpoint trained model
        model.save_pretrained(model_checkpoint_path)
        logging.info("Models saved using utilities")
        # Log end of epoch
        logging.info(f"Epoch {epoch + 1}/{n_epochs} finished")
    # Run final validation step
    best_validation_score = evaluation_step()
    # Close training
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of training
    logging.info(f"Training finished - Current date and time {end_time}")
    logging.info(f"Elapsed time {end_time - start_time}")
    # Restore best validation model weights
    model.load_state_dict(torch.load(os.path.join(best_model_checkpoint_path, 'pytorch_model.bin')))
    logging.info("Best validation model weights restored")


def evaluate_model():
    # Declare global variables
    global writer
    # Start evaluation
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time}")
    # Set model in evaluation mode
    model.eval()
    logging.info(f"Model set in evaluation mode")
    # Log start on validation set
    logging.info(f"Validation set evaluation started")
    # Compute summary report on validation set
    validation_report: str = process_evaluation('validation', 'final_evaluation', 'validation', 'Final evaluation (validation set)')
    # Log end on validation set
    logging.info(f"Validation set evaluation finished")
    logging.info('\n' + validation_report)
    # Print test results
    print(validation_report)
    # Log test results in TensorBoard
    writer.add_text('Validation set evaluation results', validation_report)
    # Log start on test set
    logging.info(f"Test set evaluation started")
    # Compute summary report on test set
    test_report: str = process_evaluation('test', 'final_evaluation', 'test', 'Final evaluation (test set)')
    # Log end on test set
    logging.info(f"Test set evaluation finished")
    logging.info('\n' + test_report)
    # Print test results
    print(test_report)
    # Log test results in TensorBoard
    writer.add_text('Test set evaluation results', test_report)
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
    # Create model and tokeniser
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
