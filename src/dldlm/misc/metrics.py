import random
from typing import Optional, List, Dict, Tuple
from collections import Counter
import torch
from dldlm.model import DLDLMFullModel, DLDLMTokenizer


def groupby(data, key) -> Dict:
    groups: Dict = dict()
    for elem in data:
        try:
            groups[key(elem)].append(elem)
        except KeyError:
            groups[key(elem)] = [elem]
    return groups


def get_latent_word_stats(labelled_samples: List[Dict], custom_stop_words: Optional[List[str]] = None) -> Dict[str, Tuple[Counter, Counter]]:
    # Group response by latent code and compute word counts
    word_stats: Dict[str, Tuple[Counter, Counter]] = {
        label: (
            sum((sample['tf'] for sample in samples), Counter()),
            sum((sample['tf_idf'] for sample in samples), Counter())
        )
        for label, samples in groupby(labelled_samples, lambda x: x['latent']).items()  # FIXME
    }
    # Remove custom stopwords if any
    if custom_stop_words is not None and len(custom_stop_words) > 0:
        custom_stop_words = set(custom_stop_words)
        for sw in custom_stop_words:
            for key, (c_sw, c_no_sw) in word_stats.items():
                if sw in c_no_sw:
                    c_no_sw.pop(sw)

    return word_stats


def get_traces(labelled_samples: List[Dict], window_size: Optional[int] = None) -> List[List[str]]:
    # Get labels from samples
    traces = [
        [sample['latent'] for sample in sorted(samples, key=lambda x: x['turn_idx'])]
        for conv_id, samples in groupby(
            labelled_samples, lambda x: (x['split'], x['corpus'], x['conversation_idx'])
        ).items()
    ]
    # Apply windowing if necessary
    if window_size is not None:
        traces = [
            trace[i:i + window_size] for trace in traces for i in range(len(trace) - window_size + 1)
        ]

    return traces


def get_response_samples(
        labelled_samples: List[Dict],
        model: DLDLMFullModel,
        tokenizer: DLDLMTokenizer,
        device,
        n_samples: Optional[int] = None,
        mixed_precision: bool = True,
        **generate_kwargs: Dict,
) -> List[Dict]:
    # Sample randomly dialogues if required
    if n_samples is not None and 0 < n_samples < len(labelled_samples):
        labelled_samples = random.sample(labelled_samples, n_samples)
    # For each of the considered samples and for each latent code generate a response
    is_training = model.training
    if is_training:
        model.eval()

    with torch.no_grad(), torch.autocast(device.type, enabled=mixed_precision):
        for sample in labelled_samples:
            generated_samples: Dict[str, str] = dict()
            prompt = sample['context']
            if model.config.unconditioned:
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                generated_samples['<|unconditioned|>'] = tokenizer.decode(
                    model.generate(input_ids, **generate_kwargs)[0, input_ids.size(-1):].cpu(),
                    skip_special_tokens=True
                )
            else:
                for latent in (f'<|latentcode{str(i).zfill(2)}|>' for i in range(model.config.num_styles)):
                    input_ids = tokenizer(prompt + latent, return_tensors='pt').input_ids.to(device)
                    generated_samples[latent] = tokenizer.decode(
                        model.generate(input_ids, **generate_kwargs)[0, input_ids.size(-1):].cpu(),
                        skip_special_tokens=True
                    )
            sample['generated_responses'] = generated_samples

    if is_training:
        model.train()

    return labelled_samples


def get_latents_count(labelled_samples: List[Dict]):
    return Counter(sample['latent'] for sample in labelled_samples)


def get_latents_correlation_matrix(
        labelled_samples: List[Dict], correlation_tgt: str, corpus_list: Optional[List[str]] = None
):
    # Get distinct corpora in the data set
    distinct_corpora = set(sample['corpus'] for sample in labelled_samples)
    # Get list of distinct values of the target value
    tgt_values = list(set(sample['corpus'] for sample in labelled_samples))
    # If the list of corpora il missing from kwargs use all
    corpus_list = corpus_list if corpus_list is not None else list(distinct_corpora)
    # Compute statistics
    correlations = {
        corpus: {
            key: (
                len(samples),
                *torch.std_mean(torch.vstack([sample['latent_posterior_dist'] for sample in samples]), dim=1)
            )
            for key, samples in groupby(
                (sample for sample in labelled_samples if sample['corpus'] == corpus),
                lambda x: x[correlation_tgt]
            ).items()
        }
        for corpus in corpus_list
    }
    # If is using all corpora get also the overall statistic
    if set(corpus_list) == distinct_corpora:
        if correlation_tgt != 'corpus':
            correlations['all'] = {
                key: (
                    len(samples),
                    *torch.std_mean(torch.vstack([sample['latent_posterior_dist'] for sample in samples]), dim=1)
                )
                for key, samples in groupby(labelled_samples, lambda x: x[correlation_tgt]).items()
            }
        else:
            correlations['All'] = {
                'All': (len(labelled_samples), *torch.std_mean(
                    torch.vstack([sample['latent_posterior_dist'] for sample in labelled_samples]),
                    dim=1
                ))
            }

    return correlations, tgt_values
