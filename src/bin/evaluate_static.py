import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, List, Dict

import random
import numpy as np
from misc import get_latent_word_stats, get_traces, get_latents_count, get_response_samples, get_latents_correlation_matrix
from misc import log_word_stats, log_traces, log_latents_count, log_generated_response, log_correlations
from misc import plot_word_stats, plot_traces, plot_correlations
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import DLDLMCorpus, DataSetSplit
from model import DLDLMTokenizer, DLDLMFullModel
# TODO add fine tuning script
# TODO add LR plot
# TODO add warm restart

# Constants
PPL_NLL_LOSS_KEY = 'lm_loss'
ELBO_OBJECTIVE_KEY = 'elbo'
KL_DIVERGENCE_LOSS_KEY = 'latent_kl_div_loss'
LOSS_KEYS_MAPPING = {
    'loss': 'Loss',
    'lm_loss': 'Language Modelling',
    'latent_kl_div_loss': 'Latent KL Divergence',
    'latent_kl_threshold_loss': 'Latent KL Divergence with threshold',
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
model_path: str
model: DLDLMFullModel
tokenizer_path: str
tokenizer: DLDLMTokenizer
# Data
corpus_configs: Dict
corpora: Dict[DataSetSplit, DLDLMCorpus] = dict()
corpus_loaders: Dict[DataSetSplit, DataLoader] = dict()
# Evaluation
evaluation_configs: Dict
# Experiment dir path
current_experiment_dir_path: str
eval_dir_path: str
data_dump_dir_path: str
count_plots_dir_path: str
trace_plots_dir_path: str
correlation_plots_dir_path: str
count_txt_dir_path: str
trace_txt_dir_path: str
sample_responses_txt_dir_path: str
latent_count_txt_dir_path: str
correlation_txt_dir_path: str
# Checkpoint paths
model_checkpoint_path: str
best_model_checkpoint_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, checkpoint_gradient, writer
    global model_path, tokenizer_path, corpus_configs, evaluation_configs
    global current_experiment_dir_path, eval_dir_path, data_dump_dir_path, \
        latent_count_txt_dir_path, count_plots_dir_path, trace_plots_dir_path, correlation_plots_dir_path, \
        count_txt_dir_path, trace_txt_dir_path, sample_responses_txt_dir_path, correlation_txt_dir_path
    global model_checkpoint_path, best_model_checkpoint_path
    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file
    with open(config_file_path) as f:
        configs_dump_str: str = f.read()
        f.seek(0)
        configs: Dict = yaml.full_load(f)
    # Chek for directories to exist and get paths
    current_experiment_dir_path = configs['reference_experiment']
    if not os.path.exists(current_experiment_dir_path):
        raise ValueError('The specified ')
    model_checkpoint_path = os.path.join(current_experiment_dir_path, 'model', 'latest_checkpoint')
    best_model_checkpoint_path = os.path.join(current_experiment_dir_path, 'model', 'best_checkpoint')
    tb_dir_path = os.path.join(current_experiment_dir_path, 'tensorboard')

    eval_dir_path = os.path.join(current_experiment_dir_path, 'evaluation')
    if not os.path.exists(eval_dir_path):
        os.mkdir(eval_dir_path)

    data_dump_dir_path = os.path.join(eval_dir_path, 'data_dump')
    if not os.path.exists(data_dump_dir_path):
        os.mkdir(data_dump_dir_path)

    plots_dir_path = os.path.join(eval_dir_path, 'plots')
    if not os.path.exists(plots_dir_path):
        os.mkdir(plots_dir_path)
    count_plots_dir_path = os.path.join(plots_dir_path, 'counts')
    if not os.path.exists(count_plots_dir_path):
        os.mkdir(count_plots_dir_path)
    trace_plots_dir_path = os.path.join(plots_dir_path, 'traces')
    if not os.path.exists(trace_plots_dir_path):
        os.mkdir(trace_plots_dir_path)
    correlation_plots_dir_path = os.path.join(plots_dir_path, 'correlations')
    if not os.path.exists(correlation_plots_dir_path):
        os.mkdir(correlation_plots_dir_path)
    txt_file_dir_path = os.path.join(eval_dir_path, 'texts')
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
    correlation_txt_dir_path = os.path.join(txt_file_dir_path, 'correlations')
    if not os.path.exists(correlation_txt_dir_path):
        os.mkdir(correlation_txt_dir_path)
    # Create file paths
    if configs.get('log_file', False):
        log_file_path = os.path.join(
            eval_dir_path, f"{configs['experiment_id']}_{date_time_experiment}.log"
        )
    else:
        log_file_path = None
    configs_dump_path = os.path.join(eval_dir_path, 'configs.yaml')
    # Init logging
    logging.basicConfig(filename=log_file_path, level=configs['log_level'])
    # Start Logging info
    logging.info(f"{configs['experiment_series']} training script started")
    logging.info(f"Current experiment directories created at '{eval_dir_path}'")
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
    # Load remaining configs
    model_path = best_model_checkpoint_path if configs.get('use_best', True) else model_checkpoint_path
    tokenizer_path = best_model_checkpoint_path if configs.get('use_best', True) else model_checkpoint_path
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
    global tokenizer, model
    # Create tokeniser instance
    tokenizer = DLDLMTokenizer.from_pretrained(tokenizer_path)
    logging.info("Tokeniser instantiated")
    # Create model instance
    model = DLDLMFullModel.from_pretrained(model_path)
    logging.info("DLDLM model instantiated")
    # Move model to device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")


def init_data_sets():
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
        input_ids: torch.LongTensor,  # Shape (batch_size, )
        attention_mask: torch.LongTensor,  # Shape (batch_size, length)
        labels: torch.LongTensor,  # Shape (batch_size, response_length)
        raw_data: List[Dict]
) -> List[Dict]:
    # Compute helper params
    mini_batch_size: int = len(input_ids)
    in_mem: int = corpus_configs['splits'][split]['in_mem']
    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    # Loop over sub_batches to fit in memory
    idxs = ((idx, min(mini_batch_size, idx + in_mem)) for idx in range(0, mini_batch_size, in_mem))
    for s_idx, e_idx in idxs:
        with torch.autocast(device.type, enabled=mixed_precision):
            # Process current elements
            model_outputs = model(
                input_ids=input_ids[s_idx:e_idx],
                attention_mask=attention_mask[s_idx:e_idx],
                labels=labels[s_idx:e_idx],
                reduction=False
            )
        # Add computed info to each element
        for idx, sample in enumerate(raw_data[s_idx:e_idx], start=s_idx):
            # Losses
            losses_dict = [model_outputs.loss_function_output.get(key, torch.empty(0, device=device)) for key in LOSS_KEYS_MAPPING]
            for key in LOSS_KEYS_MAPPING:
                sample[key] = losses_dict[key][idx].squeeze()
            # Latent posterior distribution
            if model.config.unconditioned:
                sample['latent_prior_dist'] = None
                sample['latent_posterior_dist'] = None
            else:
                sample['latent_prior_dist'] = model_outputs.prior_logits[idx].squeeze()
                sample['latent_posterior_dist'] = model_outputs.posterior_logits[idx].squeeze()
    # Return samples labelled with new info
    return raw_data


@torch.no_grad()
def process_evaluation(
        split: str,
        dir_name: str,
        file_suffix: str,
        sub_tag: str,
        step: Optional[int] = None
) -> str:
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
    # Sampled latents
    for sample in processed_data:
        if isinstance(sample['latent'], torch.Tensor):
            sample['latent'] = tokenizer.decode(sample['latent'].cpu().item())
    # Log losses
    writer.add_scalar(f'Loss/{sub_tag}', validation_loss, step)
    writer.add_scalars(
        f'Metrics/{sub_tag}',
        {LOSS_KEYS_MAPPING[key]: validation_losses_dict[key] for key in LOSS_KEYS_MAPPING},
        step
    )
    # Metrics and samples for visualisation
    latent_counts = get_latents_count(processed_data)
    word_counts = get_latent_word_stats(processed_data, **evaluation_configs['metrics']['word_stats'])
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
    correlations = {
        key: get_latents_correlation_matrix(processed_data, key, **evaluation_configs['metrics']['correlations'][key])
        for key in evaluation_configs['metrics']['correlations']
    }
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
    log_word_stats(
        word_counts,
        tb_writer=writer,
        sub_tag=sub_tag,
        step=step,
        dest_dir=os.path.join(count_txt_dir_path, dir_name),
        file_name=f'latent_word_counts_{file_suffix}.txt'
    )
    plot_word_stats(
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
    # Correlations
    log_correlations(
        correlations,
        sub_tag=sub_tag,
        step=step,
        tb_writer=writer,
        dest_dir=os.path.join(correlation_txt_dir_path, dir_name),
        file_name=f'correlations_{file_suffix}.txt'
    )
    plot_correlations(
        correlations,
        sub_tag=sub_tag,
        step=step,
        tb_writer=writer,
        dest_dir=os.path.join(correlation_plots_dir_path, dir_name),
        file_name=f'correlations_{file_suffix}.pdf'
    )

    # Dump collected data
    # Convert everything to python base types
    for sample in processed_data:
        if isinstance(sample['latent'], torch.Tensor):
            sample['latent'] = tokenizer.decode(sample['latent'].cpu().item())


    # Do the final report
    output_report = f"Evaluation (split: {split})\n" \
                    f"\tPPL: {ppl:.4f}\n" \
                    f"\tELBO: {elbo:.4f}\n" \
                    f"\tKL Divergence: {kl_divergence:.4f}\n"
    writer.add_text(f'Final assessment report/{split}',  f"<pre>{output_report}</pre>")

    return output_report


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
    # Iterate over considered splits
    for split in corpus_loaders:
        # Log start on curre set
        logging.info(f"Validation set evaluation started")
        # Compute summary report on validation set
        output_report: str = process_evaluation(
            split.value, 'final_assessment', split.value, f'Final assessment ({split.value} set)'
        )
        # Log end on validation set
        logging.info(f"Validation set evaluation finished")
        logging.info('\n' + output_report)
        # Print test results
        print(output_report)
        # Log test results in TensorBoard
        writer.add_text(f'{split.value.capitalize()} set final assessment results', output_report)
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
    init_data_sets()
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
        help="Path to the YAML file containing the configuration for the evaluation."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
