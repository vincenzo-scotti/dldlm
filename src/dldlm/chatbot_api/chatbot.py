import re

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from typing import Tuple, List, Union, Literal, Dict, Optional, Pattern


class DLDLMChatbot:
    def __init__(
            self,
            pretrained_model: str,
            prior_token: str = '<|prior|>',
            posterior_token: str = '<|posterior|>',
            latent_token_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
            in_mem: Optional[int] = None,
            device: Optional[torch.device] = None,
            mixed_precision: bool = True,
            max_context_len: Optional[int] = None,
            max_response_len: Optional[int] = None,
            sample_latent: bool = True,
            sample_response: bool = True,
            generate_kwargs: Optional[Dict] = None
    ):
        super(DLDLMChatbot, self).__init__()
        # Internal tokenizer
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        # Special tokens
        self.prior_token: str = prior_token
        self.prior_token_id: int = self.tokenizer.convert_tokens_to_ids(self.prior_token)
        self.posterior_token: str = posterior_token
        self.posterior_token_id: int = self.tokenizer.convert_tokens_to_ids(self.posterior_token)
        self.latent_token_regex: Pattern[str] = latent_token_regex
        self.latent_tokens: List[str] = [tok for tok in self.tokenizer.get_vocab() if self.latent_token_regex.match(tok) is not None]
        self.latent_token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(self.latent_tokens)
        # Generation function parameters
        self.max_context_len: Optional[int] = max_context_len
        self.max_response_len: Optional[int] = max_response_len
        self.max_len: Optional[int] = self.max_context_len + self.max_response_len if self.max_context_len is not None and self.max_response_len is not None else None
        # Generation function other params (to be forwarded to the generate function)
        self.sample_latent: bool = sample_latent
        self.sample_response: bool = sample_response
        self.generate_kwargs: Dict = generate_kwargs if generate_kwargs is not None else dict()
        if self.max_len is not None and generate_kwargs is not None:
            generate_kwargs['max_length'] = self.max_len
        # Other low level settings
        self.device: torch.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision: bool = mixed_precision and self.device.type == 'cuda'
        self.in_mem: int = in_mem if in_mem is not None else len(self.latent_token_ids)
        # Finally load internal neural network model
        self.nn_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(pretrained_model).eval().to(device)

    def __call__(self, *args, **kwargs):
        return self.generate_response(*args, **kwargs)

    def _get_context_str(self, context: List[str]) -> str:
        # Get context string
        context_str = self.tokenizer.eos_token
        if len(context) > 0:
            context_str += self.tokenizer.eos_token.join(context) + self.tokenizer.eos_token
        # Possibly cut excess
        if self.max_context_len is not None:
            context_str = self.tokenizer.decode(self.tokenizer(context_str).input_ids[-self.max_context_len:])

        return context_str

    def _get_response_str(self, response: str) -> str:
        # Add prior token
        response_str = self.prior_token + response
        # Possibly cut excess
        if self.max_response_len is not None:
            response_str = self.tokenizer.decode(self.tokenizer(response_str).input_ids[:self.max_context_len - 1])
        # Add posterior token
        response_str = response_str + self.posterior_token

        return response_str

    def _postprocess_logits(self, logits: torch.tensor):
        # Normalisation value to enforce numerical stability in output
        shift = logits.max()
        # Compute logits with normalisation trick
        logits = logits - shift

        return logits

    def _postprocess_sequence_scores(self, logits: torch.tensor, token_sequences: torch.tensor):
        # Ger padded positions binary mask
        padding_mask = token_sequences == self.tokenizer.pad_token_id
        # Possibly keep last EOS token if also used for padding
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            padding_mask = torch.hstack([
                torch.zeros((token_sequences.size(0), 1), dtype=torch.bool, device=self.device), padding_mask[:, :-1]
            ])
        # Set padding tokens to ignored value
        token_sequences[padding_mask] = -1
        # Compute NLL
        scores = F.cross_entropy(
            logits.view(-1, logits.size(-1)), token_sequences.reshape(-1), reduction='none', ignore_index=-1
        ).reshape(token_sequences.size())
        # Compute cumulative log-likelihood
        scores = self._postprocess_logits(-scores.sum(-1))

        return scores

    def _compute_latent_proba(self, hidden_vector: torch.tensor) -> torch.tensor:
        # Compute logits using LM head
        logits = self.nn_model.lm_head(hidden_vector)
        # Mask logits to prevent sampling other positions a part from latent codes
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[self.latent_token_ids] = False
        logits[mask] = float("-Inf")
        # Postprocess logits
        logits = self._postprocess_logits(logits)
        # Compute probability distribution
        proba = torch.softmax(logits, dim=-1)

        return proba

    def _compute_latent_p_dist(
            self, context: List[str], response: Optional[str] = None, prior: bool = True, posterior: bool = True
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        # Encode input
        if posterior:
            input_encoding = self.tokenizer(
                self._get_context_str(context) + self._get_response_str(response), return_tensors='pt'
            ).to(self.device)
        else:
            input_encoding = self.tokenizer(
                self._get_context_str(context) + self.prior_token, return_tensors='pt'
            ).to(self.device)
        # Compute hidden state
        hidden_state = self.nn_model.transformer(**input_encoding, return_dict=True).last_hidden_state
        # Compute prior distribution if required
        if prior:
            prior_dist = self._compute_latent_proba(
                hidden_state[torch.where(input_encoding.input_ids == self.prior_token_id)].squeeze()
            )
        else:
            prior_dist = None
        # Compute posterior distribution if required
        if posterior:
            posterior_dist = self._compute_latent_proba(
                hidden_state[torch.where(input_encoding.input_ids == self.posterior_token_id)].squeeze()
            )
        else:
            posterior_dist = None

        if prior_dist is not None and posterior_dist is not None:
            return prior_dist, posterior_dist
        elif prior_dist is not None and posterior_dist is None:
            return prior_dist
        elif prior_dist is None and posterior_dist is not None:
            return posterior_dist
        else:
            raise ValueError()

    def _predict_latent_token(self, context: List[str]) -> Tuple[str, int]:
        # Predict prior latent distribution
        prior_dist = self._compute_latent_p_dist(context, posterior=False)
        # Latent decoding step
        if self.sample_response:
            latent_ids = torch.multinomial(prior_dist, 1)
        else:
            latent_ids = torch.argmax(prior_dist, dim=-1).unsqueeze(-1)
        latent = latent_ids.squeeze(-1)
        # Retrieve latent code string and corresponding idx
        latent_str = self.tokenizer.decode(latent)
        latent_idx = int(self.latent_token_regex.search(latent_str).group(1))

        return latent_str, latent_idx

    def _compute_latent_distribution(
            self, context: List[str], response: Optional[str] = None, prior: bool = True, posterior: bool = True
    ) -> Union[List[float], Tuple[List[float], List[float]]]:
        # Input checks
        assert (not posterior or response is not None) and (prior or posterior)
        # Depending on the requested distributions compute them
        if prior and posterior is not None:
            prior_dist, posterior_dist = self._compute_latent_p_dist(context, response=response)
            return prior_dist[self.latent_token_ids].tolist(), posterior_dist[self.latent_token_ids].tolist()
        elif prior:
            prior_dist = self._compute_latent_p_dist(context, posterior=False)
            return prior_dist[self.latent_token_ids].tolist()
        elif posterior:
            posterior_dist = self._compute_latent_p_dist(context, response=response, prior=False)
            return posterior_dist[self.latent_token_ids].tolist()
        else:
            raise ValueError()

    def _generate_response_joint(self, dialogue: List[str], output_latent: bool = False) -> Union[str, Tuple[str, int]]:
        # Gather last n turns to be used as prompt to the model and merge them into a single string
        context_string = self._get_context_str(dialogue)
        # Encode the contexts using the tokeniser
        input_ids = self.tokenizer(
            [context_string + latent_code_string for latent_code_string in self.latent_tokens],
            return_tensors='pt'
        ).input_ids.to(self.device)
        # Gather generated ids and scores
        output_dict = self.nn_model.generate(
            input_ids=input_ids, **self.generate_kwargs, return_dict_in_generate=True, output_scores=True
        )
        # Compute
        scores = self._postprocess_sequence_scores(
            torch.hstack([logits.unsqueeze(1) for logits in output_dict.scores]),
            output_dict.sequences[:, input_ids.size(-1):].clone()
        )
        # Select response
        if self.sample_response:
            response_idx = torch.multinomial(torch.softmax(scores, dim=-1), 1)
        else:
            response_idx = torch.argmax(scores, dim=-1).unsqueeze(-1)
        # Retrieve latent code
        latent_str = self.latent_tokens[response_idx.item()]
        latent_code_id = int(self.latent_token_regex.search(latent_str).group(1))
        # Decode the response using the tokenizer
        output_string: str = self.tokenizer.decode(
            output_dict.sequences[response_idx.squeeze(), input_ids.size(-1):], skip_special_tokens=True
        ).strip()

        if output_latent:
            return output_string, latent_code_id
        else:
            return output_string

    def _generate_response_causal(self, dialogue: List[str], output_latent: bool = False) -> Union[str, Tuple[str, int]]:
        # Using prior predict latent action token to control response generation
        latent_code_string, latent_code_id = self._predict_latent_token(dialogue)
        # Gather last n turns to be used as prompt to the model and merge them into a single string
        context_string = self._get_context_str(dialogue)
        # Encode the context using the tokeniser
        input_ids = self.tokenizer(context_string + latent_code_string, return_tensors='pt').input_ids.to(self.device)
        # Gather generated ids
        output_ids = self.nn_model.generate(input_ids=input_ids, **self.generate_kwargs)[0, input_ids.size(1):]
        # Decode the response using the tokenizer
        output_string: str = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        if output_latent:
            return output_string, latent_code_id
        else:
            return output_string

    def generate_response(
            self,
            dialogue: List[str],
            generative_mode: Literal['causal', 'joint'] = 'causal',
            **kwargs
    ) -> Union[str, Tuple[str, int]]:
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            if generative_mode == 'causal':
                return self._generate_response_causal(dialogue, **kwargs)
            elif generative_mode == 'joint':
                return self._generate_response_joint(dialogue, **kwargs)
            else:
                raise ValueError(f'Unknown generative mode: {generative_mode}')

    def discriminate_response(
            self,
            dialogue: List[str],
            response: str,
            discriminative_mode: Literal['prior', 'posterior', 'both'] = 'posterior'
    ) -> Union[List[float], Tuple[List[float], List[float]]]:
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            if discriminative_mode == 'prior':
                return self._compute_latent_distribution(dialogue, posterior=False)
            elif discriminative_mode == 'posterior':
                return self._compute_latent_distribution(dialogue, response=response, prior=False)
            elif discriminative_mode == 'both':
                return self._compute_latent_distribution(dialogue, response=response)
            else:
                raise ValueError(f'Unknown discriminative mode: {discriminative_mode}')
