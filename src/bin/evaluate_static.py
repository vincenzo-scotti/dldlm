import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
import bz2
import pickle
from typing import Optional, Union, Tuple, List, Dict

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from dldlm.misc import LinearLR, AlphaLinearScheduler, BetaCyclicalAnnealer
from torch.utils.tensorboard import SummaryWriter

from dldlm.data import DLDLMCorpus, DataSetSplit
from dldlm.model import DLDLMTokenizer, DLDLMFullModel

# Constants
PPL_NLL_LOSS_KEY = 'lm_loss'
ELBO_OBJECTIVE_KEY = 'elbo'
KL_DIVERGENCE_LOSS_KEY = 'latent_kl_div_loss'
LOSS_KEYS_MAPPING = {
    'loss': 'Loss',
    'lm_loss': 'Language Modelling',
    'latent_kl_div_loss': 'Reverse Latent KL Divergence',
    'latent_kl_threshold_loss': 'Reverse Latent KL Divergence with threshold',
    'prior_latent_kl_div_tgt': 'Latent Prior KL Divergence from Target Distribution',
    'posterior_latent_kl_div_tgt': 'Latent Posterior KL Divergence from Target Distribution',
    'prior_latent_nll_loss': 'Latent Prior Negative Log-Likelihood',
    'prior_latent_nll_threshold_loss': 'Latent Prior Negative Log-Likelihood with threshold',
    'posterior_latent_nll_loss': 'Latent Posterior Negative Log-Likelihood',
    'posterior_latent_nll_threshold_loss': 'Latent Posterior Negative Log-Likelihood with threshold',
    'elbo': 'Evidence Lower BOund',
    'prior_dist_entropy': 'Prior Latent Distribution Entropy',
    'posterior_dist_entropy': 'Posterior Latent Distribution Entropy',
    'tf_loss': 'Term-Frequency Cross-Entropy',
}


# Variables to control model and optimization parameters
# Environment
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
checkpoint_gradient: bool = False
writer: SummaryWriter
# Model
model_configs: Dict
model: DLDLMFullModel
tokenizer_configs: Dict
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpora: Dict[DataSetSplit, DLDLMCorpus] = dict()
corpus_loaders: Dict[DataSetSplit, DataLoader] = dict()
# Experiment dir path
current_experiment_dir_path: str
data_dir_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, checkpoint_gradient, writer
    global model_configs, tokenizer_configs, corpus_configs
    global current_experiment_dir_path, data_dir_path
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
    tb_dir_path = os.path.join(current_experiment_dir_path, 'tensorboard')
    if not os.path.exists(tb_dir_path):
        os.mkdir(tb_dir_path)
    data_dir_path = os.path.join(current_experiment_dir_path, 'data')
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)
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
    writer.add_text('Configs',  f"<pre>{configs_dump_str}</pre>")
    logging.info(f"Current experiment configuration dumped at '{configs_dump_path}'")
    # Set device
    device = torch.device(configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Check for gradient checkpointing
    checkpoint_gradient = configs.get('checkpoint_gradient', checkpoint_gradient)
    logging.info(f"Gradient checkpointing {'enabled' if checkpoint_gradient else 'disabled'}")
    # Load remaining configs
    model_configs = configs['dldlm']['model']
    tokenizer_configs = configs['dldlm']['tokeniser']
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
    global tokenizer_configs, model_configs, tokenizer, model, checkpoint_gradient
    # Create tokeniser instance
    if tokenizer_configs.get('init', True):
        tokenizer = DLDLMTokenizer.from_pretrained(
            tokenizer_configs['pretrained']
        ).extend_from_gpt2_tokenizer(
            tokenizer_configs['n_styles']
        )
        logging.info("Tokeniser instantiated and extended")
    else:
        tokenizer = DLDLMTokenizer.from_pretrained(tokenizer_configs['pretrained'])
        logging.info("Tokeniser instantiated")
    logging.info("Tokeniser serialised in checkpoint directories")
    # Create model instance
    if model_configs.get('init', True):
        model = DLDLMFullModel.from_pretrained(
            model_configs['pretrained'], **model_configs.get('kwargs', dict())
        ).extend_from_gpt2_initialization(
            tokenizer
        )
        logging.info("DLDLM model instantiated and extended")
    else:
        model = DLDLMFullModel.from_pretrained(model_configs['pretrained'], **model_configs.get('kwargs', dict()))
        logging.info("DLDLM model instantiated")
    # Possibly enable gradient checkpointing
    if checkpoint_gradient:
        model.gradient_checkpointing_enable()
        logging.info("DLDLM gradient checkpoint enabled")
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
        data_set: DLDLMCorpus = DLDLMCorpus(
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
            shuffle=False,
            collate_fn=data_set.collate
        )
        logging.info(f"{split.capitalize()} data loader instantiated")
        # Add created data set and data loader to dict
        corpora[DataSetSplit(split)] = data_set
        logging.info(f"{split.capitalize()} data set added to dictionary")
        corpus_loaders[DataSetSplit(split)] = data_loader
        logging.info(f"{split.capitalize()} data loader added to dictionary")
    logging.info("All data loaders instantiated")


def process_mini_batch(
        split: str,
        input_ids: torch.LongTensor,  # Shape (batch_size, length)
        attention_mask: torch.LongTensor,  # Shape (batch_size, length)
        labels: torch.LongTensor,  # Shape (batch_size, response_length)
        latent_p_dist: torch.FloatTensor,  # Shape (batch_size, n_styles)
        distractor_ids: torch.LongTensor,  # Shape (batch_size, length)
        distractor_attention_mask: torch.LongTensor,  # Shape (batch_size, length)
        raw_data: List[Dict]
) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor], List[Dict]]]:
    # Declare global variables
    global corpus_configs, optimizer_configs, model, corpus_loaders, optimizer, scaler, lr_scheduler
    global latent_count_txt_dir_path, current_experiment_dir_path, count_plots_dir_path, \
        trace_plots_dir_path, count_txt_dir_path, trace_txt_dir_path, sample_responses_txt_dir_path
    # Initial integrity check
    if input_ids.size(-1) > tokenizer.model_max_length:
        logging.error("Out of bound input sequence, skipping.")
        return (
            torch.empty(0, device=device),
            {key: torch.empty(0, device=device) for key in LOSS_KEYS_MAPPING},
            list()
        )
# Compute helper params
    mini_batch_size: int = len(input_ids)
    in_mem: int = corpus_configs['splits'][split]['in_mem']
    # Create accumulators
    loss: torch.tensor = torch.tensor(0.0, device=device) if model.training else torch.empty(0, device=device)
    losses_dict: Dict[str, torch.tensor] = {
        key: torch.tensor(0.0, device=device) if model.training else torch.empty(0, device=device)
        for key in LOSS_KEYS_MAPPING
    }
    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    latent_p_dist = latent_p_dist.to(device) if latent_p_dist is not None else None
    distractor_ids = distractor_ids.to(device) if distractor_ids is not None else None
    distractor_attention_mask = distractor_attention_mask.to(device) if distractor_attention_mask is not None else None
    # Loop over sub_batches to fit in memory
    idxs = ((idx, min(mini_batch_size, idx + in_mem)) for idx in range(0, mini_batch_size, in_mem))
    for s_idx, e_idx in idxs:
        with torch.autocast(device.type, enabled=mixed_precision):
            # Process current elements
            model_outputs = model(
                input_ids=input_ids[s_idx:e_idx],
                attention_mask=attention_mask[s_idx:e_idx],
                labels=labels[s_idx:e_idx],
                latent_tgt_dist=latent_p_dist[s_idx:e_idx] if latent_p_dist is not None else None,
                distractor_ids=distractor_ids[s_idx:e_idx] if distractor_ids is not None else None,
                distractor_attention_mask=distractor_attention_mask[s_idx:e_idx] if distractor_attention_mask is not None else None,
                reduction=model.training,
                # use_cache=not (model.training and model.transformer.gradient_checkpointing)
            )
        # Update accumulators collect predicted latents
        loss = torch.cat([loss, model_outputs.loss])
        for key in losses_dict:
            losses_dict[key] = torch.cat([
                losses_dict[key],
                model_outputs.loss_function_output.get(key, torch.empty(0, device=device))
                # if key in model_outputs.loss_function_output
                # else torch.empty(0, device=device)
            ])
        if model.config.unconditioned:
            for sample in raw_data[s_idx:e_idx]:
                sample['latent'] = '<|unconditioned|>'
        else:
            for idx, sample in enumerate(raw_data[s_idx:e_idx], start=s_idx):
                sample['latent_prior_dist'] = torch.softmax(model_outputs.prior_logits[idx].squeeze(), dim=-1)
                sample['latent_posterior_dist'] = torch.softmax(model_outputs.posterior_logits[idx].squeeze(), dim=-1)

    return loss, losses_dict, raw_data


@torch.no_grad()
def process_evaluation(
        split: str,
        sub_tag: str,
        step: Optional[int] = None
) -> Union[str, Tuple[float, float, float, float]]:
    # Declare global variables
    global corpora, corpus_loaders, model, model_configs, evaluation_configs, checkpoint_gradient
    # Initialize validation accumulators
    validation_loss: torch.tensor = torch.empty(0, device=device)
    validation_losses_dict: Dict[str, torch.tensor] = {key: torch.empty(0, device=device) for key in LOSS_KEYS_MAPPING}
    ppl: torch.tensor = torch.empty(0, device=device)
    elbo: torch.tensor = torch.empty(0, device=device)
    kl_divergence: torch.tensor = torch.empty(0, device=device)
    # Processed samples
    processed_data: List[Dict] = list()
    # Perform evaluation step
    # Iterate over validation mini batches
    for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit(split)]):
        # Process current mini-batch
        tmp_loss, tmp_losses_dict, processed_mini_batch = process_mini_batch(split, *mini_batch)
        # Update accumulators
        # Validation loss
        validation_loss = torch.cat([validation_loss, tmp_loss])
        # Dict losses
        for key in validation_losses_dict:
            validation_losses_dict[key] = torch.cat([validation_losses_dict[key], tmp_losses_dict[key]])
        # Other indicators
        ppl = torch.cat([ppl, tmp_losses_dict[PPL_NLL_LOSS_KEY].exp()])
        elbo = torch.cat([elbo, tmp_losses_dict[ELBO_OBJECTIVE_KEY]])
        kl_divergence = torch.cat([kl_divergence, tmp_losses_dict[KL_DIVERGENCE_LOSS_KEY]])
        # Processed samples
        processed_data += processed_mini_batch
    # Average scores and recover all results from device and convert to python default types
    # Validation loss
    validation_loss = validation_loss.mean().cpu().item()
    # Dict losses
    for key in validation_losses_dict:
        validation_losses_dict[key] = validation_losses_dict[key].mean().cpu().item()
    # Other indicators
    ppl = ppl.mean().cpu().item()
    elbo = elbo.mean().cpu().item()
    kl_divergence = kl_divergence.mean().cpu().item()
    # Sampled latent distributions
    for sample in processed_data:
        if isinstance(sample['latent_prior_dist'], torch.Tensor):
            sample['latent_prior_dist'] = sample['latent_prior_dist'].cpu().tolist()
        if isinstance(sample['latent_posterior_dist'], torch.Tensor):
            sample['latent_posterior_dist'] = sample['latent_posterior_dist'].cpu().tolist()
    # Log losses
    writer.add_scalar(f'Loss/{sub_tag}', validation_loss, step)
    writer.add_scalars(
        f'Metrics/{sub_tag}',
        {LOSS_KEYS_MAPPING[key]: validation_losses_dict[key] for key in LOSS_KEYS_MAPPING},
        step
    )
    # Serialise processed data
    with bz2.BZ2File(os.path.join(data_dir_path, f'evaluation_output_{split}'), 'w') as f:
        pickle.dump(processed_data, f)

    # Do the final report
    output_report = f"Evaluation (split: {split})\n" \
                    f"\tPPL: {ppl:.4f}\n" \
                    f"\tELBO: {elbo:.4f}\n" \
                    f"\tKL Divergence: {kl_divergence:.4f}\n"
    writer.add_text(f'Final report/{split}',  f"<pre>{output_report}</pre>")

    return output_report


def evaluate_model():
    # Declare global variables
    global writer
    # Start evaluation
    # Prepare arguments
    args_dict = {
        DataSetSplit.TRAIN: ('train', 'Final evaluation (train set)'),
        DataSetSplit.VALIDATION: ('validation', 'Final evaluation (validation set)'),
        DataSetSplit.TEST: ('test', 'Final evaluation (test set)')
    }
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time}")
    # Set model in evaluation mode
    model.eval()
    logging.info(f"Model set in evaluation mode")
    # Iterate over splits
    for split, args in args_dict.items():
        # Log start on current set
        logging.info(f"{split.value.capitalize()} set evaluation started")
        # Compute summary report on validation set
        report: str = process_evaluation(*args)
        # Log end on current set
        logging.info(f"{split.value.capitalize()} set evaluation finished")
        logging.info('\n' + report)
        # Print test results
        print(report)
        # Log test results in TensorBoard
        writer.add_text(f'{split.value.capitalize()} set evaluation results', report)
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
