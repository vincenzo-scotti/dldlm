import random
from typing import Optional, List, Dict
from collections import Counter
import torch
from model import DLDLMFullModel, DLDLMTokenizer


def groupby(data, key) -> Dict:
    groups: Dict = dict()
    for elem in data:
        try:
            groups[key(elem)].append(elem)
        except KeyError:
            groups[key(elem)] = [elem]
    return groups


def get_latent_word_stats(labelled_samples: List[Dict]) -> Dict[str, Counter]:
    # Group response by latent code and compute word counts
    word_counts = {
        label: sum((sample.get('tf-idf', sample['word_counts']) for sample in samples), Counter())
        for label, samples in groupby(labelled_samples, lambda x: x['latent']).items()  # FIXME
    }

    return word_counts


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
