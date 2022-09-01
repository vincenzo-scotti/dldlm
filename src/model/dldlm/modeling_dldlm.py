from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast
)
from transformers.models.gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model as DLDLMModel,
    load_tf_weights_in_gpt2 as load_tf_weights_in_dldlm
)
from transformers.utils import logging
from .tokenization_dldlm import DLDLMTokenizer
from .configuration_dldlm import DLDLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "dldlm"
_CONFIG_FOR_DOC = "DLDLMConfig"
_TOKENIZER_FOR_DOC = "DLDLMTokenizer"

DLDLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dldlm-distilled",
    "dldlm-small",
    "dldlm-medium",
    # See all Discrete Latent Dialogue Language Model models at https://huggingface.co/models?filter=dldlm
]


@dataclass
class DLDLMLossFunctionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    
    lm_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_kl_div_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_kl_threshold_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    elbo: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    prior_latent_nll_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    prior_latent_nll_threshold_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    posterior_latent_nll_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    posterior_latent_nll_threshold_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    prior_dist_entropy: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    posterior_dist_entropy: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    tf_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)


@dataclass
class DLDLMModelOutput(BaseModelOutputWithPast):
    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, [past_length] + length + length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, num_heads, length, [past_length] + length))

@dataclass
class DLDLMLMHeadModelOutput(CausalLMOutputWithPast):
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    loss_function_output: Optional[DLDLMLossFunctionOutput] = None

    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, response_length, vocab_size)
    prior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)

    latent: Optional[torch.FloatTensor] = None  # Shape (batch_size,)

    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))

    prior_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    prior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [context_length] + 1, embed_size_per_head))
    prior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [context_length] + 1, hidden_size)
    prior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [context_length] + 1))


@dataclass
class DLDLMSequenceAnalysisModelOutput(SequenceClassifierOutputWithPast):
    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    prior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    posterior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)

    prior_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    prior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[context_length] + 1], embed_size_per_head))
    prior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[context_length] + 1], hidden_size)
    prior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [context_length] + 1))
    posterior_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    posterior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[[context_length] + response_length] + 1], embed_size_per_head))
    posterior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[[context_length] + response_length] + 1], hidden_size)
    posterior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [[context_length] + response_length] + 1))


@dataclass
class DLDLMFullModelOutput(DLDLMLMHeadModelOutput, DLDLMSequenceAnalysisModelOutput):
    # logits shape (batch_size, response_length, vocab_size)
    tf_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, tf_size)


class DLDLMPreTrainedModel(GPT2PreTrainedModel):
    config_class = DLDLMConfig
    load_tf_weights = load_tf_weights_in_dldlm
    # base_model_prefix = "discrete_latent_transformer"  # TODO decide whether to leave gpt2 or find a fix for this base model issue
    is_parallelizable = False
    supports_gradient_checkpointing = True

    def compute_loss_function(
            self,
            lm_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, response_length, vocab_size)
            labels: Optional[torch.Tensor] = None,  # Shape (batch_size, response_length)
            prior_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, num_styles)
            posterior_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, num_styles)
            latent: Optional[torch.Tensor] = None,  # Shape (batch_size,)
            tf_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, tf_size)
            batch_size: Optional[int] = None,
            **kwargs
    ) -> DLDLMLossFunctionOutput:
        # Prepare additional parameters
        reduction: bool = kwargs.get('reduction', self.config.reduction)
        assert reduction or batch_size is not None, "You need to specify the mini_batch size when there is no reduction."

        # Cumulative loss
        loss = torch.zeros(1 if reduction else batch_size, device=self.device)
        # Language modeling loss
        if lm_logits is not None and labels is not None:
            shift_labels: torch.Tensor = labels[..., 1:].contiguous()
            shift_logits: torch.Tensor = lm_logits[..., :-1, :].contiguous()
            tgt_len: torch.Tensor = (shift_labels != self.config.ignore_id).float().sum(1)
            lm_loss: Optional[torch.Tensor] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=self.config.ignore_id
            ).view(shift_labels.size()).sum(1)
            lm_loss /= tgt_len
            if reduction:
                lm_loss = lm_loss.mean()
            if kwargs.get('lm_loss', self.config.lm_loss):
                weight = torch.tensor(kwargs.get('lm_loss_weight', self.config.lm_loss_weight), device=self.device)
                loss += weight * lm_loss
        else:
            lm_loss = None
        # Latent code loss
        if prior_logits is not None and (posterior_logits is not None or latent is not None):
            # KL Divergence
            if posterior_logits is not None:
                # Prior from posterior target KL divergence
                if kwargs.get('fixed_prior', self.config.fixed_prior):
                    latent_kl_div_loss: Optional[torch.Tensor] = F.kl_div(
                        F.log_softmax(torch.full_like(
                            posterior_logits[:, self.config.latent_token_ids], 1 / self.config.num_styles
                        ), -1),
                        F.log_softmax(posterior_logits, -1)[:, self.config.latent_token_ids],
                        reduction='none',
                        log_target=True
                    )
                else:
                    latent_kl_div_loss: Optional[torch.Tensor] = F.kl_div(
                        F.log_softmax(prior_logits, -1)[:, self.config.latent_token_ids],
                        F.log_softmax(posterior_logits, -1)[:, self.config.latent_token_ids],
                        reduction='none',
                        log_target=True
                    )
                # Apply thresholding if required
                latent_kl_threshold_loss: Optional[torch.Tensor] = torch.max(
                    latent_kl_div_loss, torch.full_like(latent_kl_div_loss, self.config.kl_loss_threshold)
                ).sum(-1)
                # Sum over last axis
                latent_kl_div_loss = latent_kl_div_loss.sum(-1)
                if reduction:
                    latent_kl_div_loss = latent_kl_div_loss.mean()
                    latent_kl_threshold_loss = latent_kl_threshold_loss.mean()
            else:
                latent_kl_div_loss = latent_kl_threshold_loss = None
            # Posterior negative Log-Likelihood
            if latent is not None:
                prior_latent_nll_loss: Optional[torch.Tensor] = F.cross_entropy(prior_logits, latent, reduction='none')
                # Apply thresholding if required
                prior_latent_nll_threshold_loss: Optional[torch.Tensor] = torch.max(
                    prior_latent_nll_loss, torch.full_like(prior_latent_nll_loss, self.config.behaviour_loss_threshold)
                )
                if reduction:
                    prior_latent_nll_loss = prior_latent_nll_loss.mean()
                    prior_latent_nll_threshold_loss = prior_latent_nll_threshold_loss.mean()
            else:
                prior_latent_nll_loss = prior_latent_nll_threshold_loss = None
            if posterior_logits is not None and latent is not None:
                posterior_latent_nll_loss: Optional[torch.Tensor] = F.cross_entropy(
                    posterior_logits, latent, reduction='none'
                )
                posterior_latent_nll_threshold_loss: Optional[torch.Tensor] = torch.max(
                    posterior_latent_nll_loss, torch.full_like(posterior_latent_nll_loss, self.config.sampling_loss_threshold)
                )
                if reduction:
                    posterior_latent_nll_loss = posterior_latent_nll_loss.mean()
                    posterior_latent_nll_threshold_loss = posterior_latent_nll_threshold_loss.mean()
            else:
                posterior_latent_nll_loss = posterior_latent_nll_threshold_loss = None
            # Combine with loss if required
            if kwargs.get('kl_loss', self.config.kl_loss):
                weight = torch.tensor(kwargs.get('kl_loss_weight', self.config.kl_loss_weight), device=self.device)
                if self.config.kl_loss_threshold > -float('inf'):
                    loss += weight * latent_kl_threshold_loss
                else:
                    loss += weight * latent_kl_div_loss
            if kwargs.get('behaviour_loss', self.config.behaviour_loss):
                weight = torch.tensor(kwargs.get('behaviour_loss_weight', self.config.behaviour_loss_weight), device=self.device)
                if self.config.behaviour_loss_threshold > -float('inf'):
                    loss += weight * prior_latent_nll_threshold_loss
                else:
                    loss += weight * prior_latent_nll_loss
            if kwargs.get('sampling_loss', self.config.sampling_loss):
                weight = torch.tensor(kwargs.get('sampling_loss_weight', self.config.sampling_loss_weight), device=self.device)
                if self.config.sampling_loss_threshold > -float('inf'):
                    loss += weight * posterior_latent_nll_threshold_loss
                else:
                    loss += weight * posterior_latent_nll_loss
        else:
            latent_kl_div_loss = latent_kl_threshold_loss = None
            prior_latent_nll_loss = prior_latent_nll_threshold_loss = None
            posterior_latent_nll_loss = posterior_latent_nll_threshold_loss = None
        # ELBO
        if lm_loss is not None and latent_kl_div_loss is not None:
            elbo: Optional[torch.Tensor] = -(lm_loss + latent_kl_div_loss)
        else:
            elbo = None
        # Prior entropy
        if prior_logits is not None:
            prior_dist = torch.softmax(prior_logits, dim=-1)
            prior_dist_entropy = -(prior_dist * (prior_dist + 1e-12).log2()).sum(dim=-1)
            if reduction:
                prior_dist_entropy = prior_dist_entropy.mean()
            if kwargs.get('prior_entropy_loss', self.config.prior_entropy_loss):
                # This goes in the other direction, I want it to be high to have a maximum entropy estimator
                weight = torch.tensor(kwargs.get('prior_entropy_loss_weight', self.config.prior_entropy_loss_weight), device = self.device)
                loss += weight * prior_dist_entropy
        else:
            prior_dist_entropy = None
        # Posterior entropy
        if posterior_logits is not None:
            posterior_dist = torch.softmax(posterior_logits, dim=-1)
            posterior_dist_entropy = -(posterior_dist * (posterior_dist + 1e-12).log2()).sum(dim=-1)
            if reduction:
                posterior_dist_entropy = posterior_dist_entropy.mean()
            if kwargs.get('posterior_entropy_loss', self.config.posterior_entropy_loss):
                weight = torch.tensor(kwargs.get('posterior_entropy_loss_weight', self.config.posterior_entropy_loss_weight), device = self.device)
                loss += weight * posterior_dist_entropy
        else:
            posterior_dist_entropy = None
        # tf prediction loss
        if tf_logits is not None and labels is not None:
            tf_labels: torch.Tensor = labels.clone()
            tf_labels[tf_labels >= self.config.tf_size] = self.config.ignore_id
            tgt_len = (tf_labels != self.config.ignore_id).float().sum(1)
            tf_loss: Optional[torch.Tensor] = F.cross_entropy(
                tf_logits.repeat(1, tf_labels.size(1), 1).view(-1, tf_logits.size(-1)),
                tf_labels.view(-1),
                reduction='none',
                ignore_index=self.config.ignore_id
            ).view(tf_labels.size()).sum(1)
            tf_loss /= tgt_len
            if reduction:
                tf_loss = tf_loss.mean()
            if kwargs.get('tf_loss', self.config.tf_loss):
                weight = torch.tensor(kwargs.get('tf_loss_weight', self.config.tf_loss_weight), device = self.device)
                loss += weight * tf_loss
        else:
            tf_loss = None

        output = DLDLMLossFunctionOutput(
            loss=loss,
            lm_loss=lm_loss,
            latent_kl_div_loss=latent_kl_div_loss,
            latent_kl_threshold_loss=latent_kl_threshold_loss,
            prior_latent_nll_loss=prior_latent_nll_loss,
            prior_latent_nll_threshold_loss=prior_latent_nll_threshold_loss,
            posterior_latent_nll_loss=posterior_latent_nll_loss,
            posterior_latent_nll_threshold_loss=posterior_latent_nll_threshold_loss,
            elbo=elbo,
            prior_dist_entropy=prior_dist_entropy,
            posterior_dist_entropy=posterior_dist_entropy,
            tf_loss=tf_loss
        )

        return output

    def _compute_hidden_transformation(  # Wrapper to transformer forward
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, )
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            corruption_mask: Optional[torch.FloatTensor] = None
    ) -> DLDLMModelOutput:
        # Fix empty past issue
        if past_key_values is not None and len(past_key_values) == 0:
            past_key_values = None
        # Attention and token types are taken as they are since now it is a single sequence
        # There is no split between context and response
        if position_ids is None and attention_mask is not None:  # Compute position IDs from attention
            position_ids = attention_mask.cumsum(dim=-1).long() - attention_mask.long()
            if past_key_values is not None:
                position_ids = position_ids[:, past_key_values[0][0].size(2):]
        # Apply corruption mask if required
        if corruption_mask is not None:
            input_embeds = self.transformer.wte(input_ids) * corruption_mask
            input_ids = None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        output = DLDLMModelOutput(
            last_hidden_state=transformer_outputs.last_hidden_state,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )

        return output

    def _encode(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            prior_token_id_idxs: Optional[Tuple[torch.Tensor]] = None,
            posterior_token_id_idxs: Optional[Tuple[torch.Tensor]] = None,
            response_start_idx: Optional[int] = None,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        # Encoding step
        # Apply corruption to input mask
        corruption_mask = self._input_corruption_mask(
            attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            prior_token_id_idxs,
            posterior_token_id_idxs,
            p=self.config.corruption_rate,
            training=self.training
        )
        # Dialogue analysis step
        analysis_hidden_outputs = self._compute_hidden_transformation(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            corruption_mask=corruption_mask
        )
        past_key_values = analysis_hidden_outputs.past_key_values
        analysis_last_hidden_state = analysis_hidden_outputs.last_hidden_state
        # Prepare outputs from encoded if needed
        # Past key values
        if use_cache and past_key_values is not None:
            prior_past_key_values = tuple(
                (k[:, :, :response_start_idx], v[:, :, :response_start_idx]) for (k, v) in past_key_values
            )
            posterior_past_key_values = past_key_values
        else:
            prior_past_key_values = posterior_past_key_values = None
        # Hidden states
        if output_hidden_states and analysis_hidden_outputs.hidden_states is not None:
            prior_hidden_states = tuple(h[:, :response_start_idx] for h in analysis_hidden_outputs.hidden_states)
            posterior_hidden_states = analysis_hidden_outputs.hidden_states
        else:
            prior_hidden_states = posterior_hidden_states = None
        # Attentions
        if output_attentions and analysis_hidden_outputs.attentions is not None:
            prior_attentions = tuple(
                a[:, :, :response_start_idx, :response_start_idx] for a in analysis_hidden_outputs.attentions
            )
            posterior_attentions = analysis_hidden_outputs.attentions
        else:
            prior_attentions = posterior_attentions = None
        # Latent last hidden states
        prior_last_hidden_state = analysis_last_hidden_state[prior_token_id_idxs]
        posterior_last_hidden_state = analysis_last_hidden_state[posterior_token_id_idxs]

        return (
            past_key_values,
            prior_last_hidden_state,
            prior_past_key_values,
            prior_hidden_states,
            prior_attentions,
            posterior_last_hidden_state,
            posterior_past_key_values,
            posterior_hidden_states,
            posterior_attentions
        )

    def _decode(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
            prior_logits: Optional[torch.FloatTensor] = None,  # Shape (batch_size, vocab_size)
            posterior_logits: Optional[torch.FloatTensor] = None,  # Shape (batch_size, vocab_size)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            prior_token_id_idxs: Optional = None,
            posterior_token_id_idxs: Optional = None,
            response_start_idx: int = 1,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        # Decoding step

        # Apply context dropout if required
        # Attention
        attention_mask = self._context_dropout(
            attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            prior_token_id_idxs,
            p=self.config.context_pdrop,
            training=self.training
        )

        # Latent decoding step
        # If model is misc process latent embedding and then the rest of the sequence
        if self.training:
            # Compute latent embedding
            # If doing gibbs sampling for fine tuning sample from original distribution
            if kwargs.get('do_gibbs_sampling', self.config.do_gibbs_sampling):
                p_soft = torch.softmax(posterior_logits if posterior_logits is not None else prior_logits, dim=-1)
                latent_ids = torch.multinomial(p_soft, 1)
                p_hard = torch.zeros_like(
                    posterior_logits if posterior_logits is not None else prior_logits,
                    memory_format=torch.legacy_contiguous_format
                ).scatter_(-1, latent_ids, 1.0)
                latent_ids_sparse = p_hard - p_soft.detach() + p_soft
            # Else if doing pretraining sample from gumbel distribution
            else:
                latent_ids_sparse = F.gumbel_softmax(
                    (posterior_logits if posterior_logits is not None else prior_logits), hard=True
                )
                latent_ids = torch.argmax(latent_ids_sparse, dim=-1).unsqueeze(-1)
            latent_embeds = torch.einsum('vh, bv -> bh', self.transformer.wte.weight, latent_ids_sparse)  # .unsqueeze(1)
        # Else simply extract the ids of the latent
        else:
            if kwargs.get('do_sample_latent', self.config.do_sample_latent):
                latent_ids = torch.multinomial(torch.softmax(
                        (posterior_logits if posterior_logits is not None else prior_logits), dim=-1
                ), 1)
            else:
                latent_ids = torch.argmax(
                    (posterior_logits if posterior_logits is not None else prior_logits), dim=-1
                ).unsqueeze(-1)
            latent_embeds = None
        latent = latent_ids.squeeze(-1)
        # Update inputs and memories
        # Already processed context tokens
        past_key_values = tuple(
            (k[:, :, :response_start_idx - 1], v[:, :, :response_start_idx - 1]) for (k, v) in past_key_values
        ) if response_start_idx > 1 else None
        # Input response tokens
        input_ids = input_ids.clone()
        # Remove already processed tokens if not using gradient checkpointing
        if not (self.transformer.gradient_checkpointing and self.training):
            # Remove already processed tokens
            input_ids = input_ids[:, response_start_idx - 1:]
            # Additional inputs
            token_type_ids = token_type_ids[:, response_start_idx - 1:] if token_type_ids is not None else None
            position_ids = position_ids[:, response_start_idx - 1:] if position_ids is not None else None
            # Re-compute positions of special tokens
            prior_token_id_idxs = torch.where(input_ids == self.config.prior_token_id)
            posterior_token_id_idxs = torch.where(input_ids == self.config.posterior_token_id)
        # Substitute latent analysis tokens
        input_ids[prior_token_id_idxs] = latent
        input_ids[posterior_token_id_idxs] = self.config.eos_token_id
        # If model is misc process latent embedding and then the rest of the sequence
        if self.training and not kwargs.get('detach_posterior', self.config.detach_posterior):
            input_embeds = self.transformer.wte(input_ids)
            input_embeds[prior_token_id_idxs] = latent_embeds.type(input_embeds.dtype)  # Needs explicit casting in case AMP is active
            input_ids = None
        else:
            input_embeds = None
        # Decode latent and response
        decoding_hidden_outputs = self._compute_hidden_transformation(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = decoding_hidden_outputs.last_hidden_state
        past_key_values = decoding_hidden_outputs.past_key_values
        hidden_states = decoding_hidden_outputs.hidden_states
        attentions = decoding_hidden_outputs.attentions
        latent_last_hidden_state = last_hidden_state[prior_token_id_idxs]

        return latent, latent_last_hidden_state, last_hidden_state, past_key_values, hidden_states, attentions

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,  # Shape (batch_size, length)
            past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            use_cache=None,
            **kwargs
    ):
        if not (isinstance(self, DLDLMFullModel) or isinstance(self, DLDLMLMHeadModel)):
            raise TypeError("Only the DLDLM full model and lm model can do generation")
        # NOTE GPT-2 assume that in case of past everything can be dropped from inputs
        if past is not None:
            # If the past is given then proceed in the generation as usual
            input_ids = input_ids[:, -1].unsqueeze(-1)
            # # Expand past key values (if required)
            # if past[0][0].shape[0] != input_ids.shape[0]:  # TODO check if necessary
            #     # Compute number of repetitions
            #     expand_size = input_ids.shape[0] // past[0][0].shape[0]
            #     # Do expansion
            #     past = tuple(
            #         (k.repeat_interleave(expand_size, dim=0), v.repeat_interleave(expand_size, dim=0)) for k, v in past
            #     )
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "use_cache": kwargs.get("use_cache"),
                "generative": True,
                "use_latent_mask": True
            }
        else:
            # Else encode context and, optionally, sample latent
            if all(self.config.prior_token_id in ids for ids in input_ids):
                # Get device
                device = next(self.transformer.parameters()).device
                # Get prior token position
                prior_token_id_idxs = torch.where(input_ids == self.config.prior_token_id)
                # Get response start idx
                _, (response_start_idx, *_) = torch.where(input_ids == self.config.prior_token_id)
                response_start_idx += 1
                assert torch.all(input_ids[:, response_start_idx - 1] == self.config.prior_token_id), \
                    "Responses must be aligned (start from same position) to use this model."
                # Encoding step
                past, prior_last_hidden_state, *_ = self._encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    prior_token_id_idxs=prior_token_id_idxs,
                    response_start_idx=response_start_idx,
                    use_cache=True,
                )
                # Delete latest token from past
                past = tuple((k[:, :, :response_start_idx - 1], v[:, :, :response_start_idx - 1]) for (k, v) in past)
                # Compute latent logits
                # Prior logits
                prior_logits = self.lm_head(prior_last_hidden_state)
                # Latent mask to prevent using common tokens
                latent_mask = torch.zeros_like(prior_logits)
                latent_mask[:, self.config.latent_token_ids] = 1.
                latent_mask = torch.log(latent_mask)
                # Latent decoding step
                if kwargs.get('do_sample_latent', self.config.do_sample_latent):
                    latent_ids = torch.multinomial(torch.softmax(prior_logits + latent_mask, dim=-1), 1)
                else:
                    latent_ids = torch.argmax(prior_logits + latent_mask, dim=-1).unsqueeze(-1)
                latent = latent_ids.squeeze(-1)
                # Add sampled latents in place of the prior tokens
                input_ids[prior_token_id_idxs] = latent
            else:  # if all(any(latent_token in ids for latent_token in self.config.latent_token_ids) for ids in input_ids):
                # Check if there is some context a part from the latent and compute the past
                if input_ids.size(1) > 1:
                    past = self._compute_hidden_transformation(
                        input_ids=input_ids[:, :-1],
                        attention_mask=attention_mask[:, :-1],
                        token_type_ids=token_type_ids[:, :-1] if token_type_ids is not None else None,
                        position_ids=position_ids[:, :-1] if position_ids is not None else None,
                        use_cache=True,
                    ).past_key_values
                # Else generate directly from the latent
                else:
                    return {
                        "input_ids": input_ids,
                        "past_key_values": past,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "token_type_ids": token_type_ids,
                        "use_cache": kwargs.get("use_cache"),
                        "generative": True
                    }
            # else:
            #     raise ValueError(
            #         "Either a prior latent token or one of the available latent token "
            #         "must be part of the input sequence"
            #     )

            return self.prepare_inputs_for_generation(
                input_ids,
                past=past,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs
            )

    @staticmethod
    def _input_corruption_mask(
            attention_mask: torch.LongTensor,
            prior_idxs: Tuple[torch.Tensor],
            posterior_idxs: Tuple[torch.Tensor],
            p: float = 0.5,
            training: bool = True
    ) -> Optional[torch.FloatTensor]:
        # Check if it's misc or not
        if training and p > 0.0:
            # Create corruption mask
            corruption_mask = torch.zeros_like(attention_mask)
            corruption_mask[attention_mask.bool()] = (
                    torch.rand_like(attention_mask[attention_mask.bool()].float()) >= p
            ).float()
            # Avoid corrupting prior and posterior tokens
            corruption_mask[prior_idxs] = 1.
            corruption_mask[posterior_idxs] = 1
        else:
            corruption_mask = None
        # Return attention mask
        return corruption_mask

    @staticmethod
    def _context_dropout(
            attention_mask: torch.LongTensor,
            sep_idxs: Tuple[torch.Tensor],
            p: float = 0.5,
            training: bool = True,
            inplace: bool = False
    ) -> torch.LongTensor:
        # Check if it's misc or not
        if training and p > 0.0:
            # Boolean mask to identify the context tokens
            context_mask = torch.zeros_like(attention_mask)
            context_mask[sep_idxs] = 1
            context_mask = (-context_mask.cumsum(dim=-1) + 1).bool()
            # Vector of 1 and 0 values of the same size of the batch size (represents dropped contexts)
            dropout_mask = (torch.rand((attention_mask.size(0), 1), device=context_mask.device) >= p).long()
            # Zero out attention mask of dropped contexts
            if not inplace:
                attention_mask = attention_mask.clone()
            attention_mask[context_mask] = (attention_mask * dropout_mask)[context_mask]
        # Return attention mask
        return attention_mask
    
    @torch.no_grad()
    def extend_from_gpt2_initialization(self, updated_tokenizer: DLDLMTokenizer):
        # Extend input embeddings if required
        if len(updated_tokenizer) != self.config.vocab_size:
            self.resize_token_embeddings(new_num_tokens=len(updated_tokenizer))
        # Initialise TF head with LM head wights (if required)
        if self.config.init_tf_head:
            try:
                self.tf_head.weight.copy_(self.lm_head.weight[:self.config.tf_size])
            except AttributeError:
                pass
            self.config.init_tf_head = False
        # Add special tokens if needed
        self.config.pad_token_id = updated_tokenizer.eos_token_id
        self.config.unk_token_id = updated_tokenizer.unk_token_id
        self.config.mask_token_id = updated_tokenizer.mask_token_id
        self.config.prior_token_id = updated_tokenizer.convert_tokens_to_ids('<|prior|>')
        self.config.posterior_token_id = updated_tokenizer.convert_tokens_to_ids('<|posterior|>')
        self.config.latent_token_ids = updated_tokenizer.convert_tokens_to_ids(
            [f'<|latentcode{str(i).zfill(2)}|>' for i in range(self.config.num_styles)]
        )
        return self


class DLDLMFullModel(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super(DLDLMFullModel, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tf_size is not None and config.tf_size > 0:
            self.tf_head = nn.Linear(config.hidden_size, config.tf_size, bias=False)
        self.cls_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)

        self.init_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
        attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
        token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        head_mask=None,
        labels: Optional[torch.LongTensor] = None,  # Shape (batch_size,)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # If the model is in generative mode, call generative forward (it only computes hidden transformations and LM)
        if kwargs.get('generative', False) or kwargs.get('unconditioned', self.config.unconditioned):
            return self._generative_forward(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        # Else if the model is in analysis mode, it does all the encoding and decoding steps
        else:
            return self._analysis_forward(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

    def _analysis_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            head_mask=None,
            labels: Optional[torch.LongTensor] = None,  # Shape (batch_size,)
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        # Get output preparation flags
        use_cache = (use_cache if use_cache is not None else self.config.use_cache) and not (self.transformer.gradient_checkpointing and self.training)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get device
        device = next(self.transformer.parameters()).device
        # Get special token positions (if they are present)
        if self.config.prior_token_id in input_ids:
            prior_token_id_idxs = torch.where(input_ids == self.config.prior_token_id)
        else:
            prior_token_id_idxs = None
        if self.config.posterior_token_id in input_ids:
            posterior_token_id_idxs = torch.where(input_ids == self.config.posterior_token_id)
        else:
            posterior_token_id_idxs = None
        # Get response start idx
        try:
            _, (response_start_idx, *_) = torch.where(input_ids == self.config.prior_token_id)
            response_start_idx += 1
            assert torch.all(input_ids[:, response_start_idx - 1] == self.config.prior_token_id), \
                "Responses must be aligned (start from same position) to use this model."
        except ValueError:
            response_start_idx = 1

        # Encoding step
        (
            past_key_values,
            prior_last_hidden_state,
            prior_past_key_values,
            prior_hidden_states,
            prior_attentions,
            posterior_last_hidden_state,
            posterior_past_key_values,
            posterior_hidden_states,
            posterior_attentions
        ) = self._encode(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            prior_token_id_idxs=prior_token_id_idxs,
            posterior_token_id_idxs=posterior_token_id_idxs,
            response_start_idx=response_start_idx,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        # Compute latent logits
        # Latent mask to prevent using common tokens
        latent_mask = torch.zeros((1, self.config.vocab_size), device=device)
        latent_mask[:, self.config.latent_token_ids] = 1.
        latent_mask = torch.log(latent_mask)
        # Prior
        prior_logits = self.lm_head(prior_last_hidden_state) + latent_mask
        # Posterior
        posterior_logits = self.lm_head(posterior_last_hidden_state) + latent_mask

        # Decoding step
        (
            latent,
            latent_last_hidden_state,
            last_hidden_state,
            past_key_values,
            hidden_states,
            attentions
        ) = self._decode(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            prior_token_id_idxs=prior_token_id_idxs,
            posterior_token_id_idxs=posterior_token_id_idxs,
            response_start_idx=response_start_idx,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        # Compute LM logits
        lm_logits = self.lm_head(last_hidden_state[:, -(labels.size(1) if labels is not None else 0):])
        # Compute TF model logits
        tf_logits = self.tf_head(latent_last_hidden_state) if hasattr(self, 'tf_head') else None

        # Loss computation
        loss_function_output: DLDLMLossFunctionOutput = self.compute_loss_function(
            lm_logits=lm_logits,
            labels=labels,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            latent=latent,
            tf_logits=tf_logits,
            batch_size=batch_size,
            **kwargs
        )

        output: DLDLMFullModelOutput = DLDLMFullModelOutput(
            loss=loss_function_output.loss,
            loss_function_output=loss_function_output,
            logits=lm_logits,
            posterior_logits=posterior_logits[..., self.config.latent_token_ids],
            prior_logits=prior_logits[..., self.config.latent_token_ids],
            tf_logits=tf_logits,
            latent=latent,
            latent_last_hidden_state=latent_last_hidden_state if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            prior_last_hidden_state=prior_last_hidden_state,
            prior_past_key_values=prior_past_key_values,
            prior_hidden_states=prior_hidden_states,
            prior_attentions=prior_attentions,
            posterior_last_hidden_state=posterior_last_hidden_state,
            posterior_past_key_values=posterior_past_key_values,
            posterior_hidden_states=posterior_hidden_states,
            posterior_attentions=posterior_attentions,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()

    def _generative_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
            head_mask=None,
            labels: Optional[torch.LongTensor] = None,  # Shape (batch_size,)
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        # Get output preparation flags
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Get device
        device = next(self.transformer.parameters()).device

        # Compute hidden representation
        transformer_outputs = self._compute_hidden_transformation(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # Compute LM logits
        last_hidden_state = transformer_outputs.last_hidden_state[:, -(labels.size(1) if labels is not None else 0):]
        lm_logits = self.lm_head(last_hidden_state)
        if kwargs.get('use_latent_mask', False):
            # Latent mask to prevent using latent analysis tokens
            latent_mask = torch.ones((1, self.config.vocab_size), device=device)
            latent_mask[:, self.config.prior_token_id] = 0.
            latent_mask[:, self.config.posterior_token_id] = 0.
            latent_mask[:, self.config.latent_token_ids] = 0.
            latent_mask = torch.log(latent_mask)
            lm_logits += latent_mask

        # Loss computation
        loss_function_output: DLDLMLossFunctionOutput = self.compute_loss_function(
            lm_logits=lm_logits,
            labels=labels,
            **kwargs
        )

        output: DLDLMFullModelOutput = DLDLMFullModelOutput(
            loss=loss_function_output.loss,
            loss_function_output=loss_function_output,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values if use_cache else None,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()

    @torch.no_grad()
    def update_cls_weights(self):
        self.cls_head.weight.copy_(self.lm_head.weight[self.config.latent_token_ids])

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class DLDLMLMHeadModel(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super(DLDLMLMHeadModel, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, *input, **kwargs):
        # TODO add modified version of Full Model forwards
        raise NotImplemented()

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class DLDLMForSequenceClassification(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super(DLDLMForSequenceClassification, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output head
        self.cls_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)

        self.init_weights()

    def forward(self, *input, **kwargs):
        # TODO add encoding only forward
        raise NotImplemented()
