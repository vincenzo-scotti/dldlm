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
    "dldlm-small",
    "dldlm-small-emp",
    "dldlm-medium",
    "dldlm-medium-emp",
    # See all Discrete Latent Dialogue Language Model models at https://huggingface.co/models?filter=dldlm
]

@dataclass
class DLDLMCostFunctionOutput(ModelOutput):
    cost: torch.Tensor  # Shape (1,) or (batch_size,)
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    objective: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)

    lm_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_kl_div: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_kl_div_threshold: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    cls_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    bow_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    rew_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)

    lm_obj: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    latent_obj: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)

@dataclass
class DLDLMModelOutput(BaseModelOutputWithPast):
    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [past_length] + length + length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, length, [past_length] + length))
    attention_mask: Optional[torch.LongTensor] = None  # Shape (batch_size, [past_length] + length)

@dataclass
class DLDLMAllHeadsModelOutput(CausalLMOutputWithPast):
    cost: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    objective: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    cost_function_output: Optional[DLDLMCostFunctionOutput] = None

    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, response_length, vocab_size)
    posterior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    policy_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    cls_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size,)
    cls_logits_contrastive: Optional[torch.FloatTensor] = None  # Shape (batch_size,)
    bow_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, bow_size)
    raw_reward: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_rewards)

    latent: Optional[torch.FloatTensor] = None  # Shape (batch_size,)

    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))

    policy_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    policy_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[context_length] + 1], embed_size_per_head))
    policy_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[context_length] + 1], hidden_size)
    policy_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [context_length] + 1))
    posterior_last_hidden_state: Optional[torch.Tensor] = None # Shape (batch_size, hidden_size)
    posterior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[[context_length] + response_length] + 1], embed_size_per_head))
    posterior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[[context_length] + response_length] + 1], hidden_size)
    posterior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [[context_length] + response_length] + 1))


@dataclass
class DLDLMPolicyLMHeadModelOutput(CausalLMOutputWithPast):
    cost: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    objective: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    cost_function_output: Optional[DLDLMCostFunctionOutput] = None

    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, response_length, vocab_size)
    policy_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)

    latent: Optional[torch.FloatTensor] = None  # Shape (batch_size,)

    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))

    policy_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    policy_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[context_length] + 1], embed_size_per_head))
    policy_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[context_length] + 1], hidden_size)
    policy_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [context_length] + 1))


@dataclass
class DLDLMPosteriorSequenceClassifierOutput(SequenceClassifierOutputWithPast):
    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, )

    last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))


@dataclass
class DLDLMIRSequenceClassifierOutput(SequenceClassifierOutputWithPast):
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    cost_function_output: Optional[DLDLMCostFunctionOutput] = None

    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, )
    logits_contrastive: Optional[torch.FloatTensor] = None  # Shape (batch_size,)

    last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))


class DLDLMPreTrainedModel(GPT2PreTrainedModel):
    config_class = DLDLMConfig
    load_tf_weights = load_tf_weights_in_dldlm
    # base_model_prefix = "discrete_latent_transformer"  # TODO decide whether to leave gpt2 or find a fix for this base model issue
    is_parallelizable = False
    supports_gradient_checkpointing = False

    def compute_cost_function(
            self,
            lm_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, response_length, vocab_size)
            labels: Optional[torch.Tensor] = None,  # Shape (batch_size, response_length)
            policy_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, num_styles)
            posterior_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, num_styles)
            latent: Optional[torch.Tensor] = None,  # Shape (batch_size,)
            cls_logits_pos: Optional[torch.Tensor] = None,  # Shape (batch_size,)
            cls_logits_neg: Optional[torch.Tensor] = None,  # Shape (batch_size,)
            bow_logits: Optional[torch.Tensor] = None,  # Shape (batch_size, bow_size)
            raw_reward: Optional[torch.Tensor] = None,  # Shape (batch_size, num_rewards)
            target_reward: Optional[torch.Tensor] = None,  # Shape (batch_size, num_rewards)
            g_return: Optional[torch.Tensor] = None,  # Shape (batch_size,)
            batch_size: Optional[int] = None,
            **kwargs
    ) -> DLDLMCostFunctionOutput:
        # Prepare additional parameters
        reduction: bool = kwargs.get('reduction', self.config.reduction)
        assert reduction or batch_size is not None, "You need to specify the mini_batch size when there is no reduction."

        # Compute losses
        if self.config.rl_weight < 1.:
            # Cumulative loss
            loss = torch.zeros(1 if reduction else batch_size, device=self.device)
            # Language modeling loss
            if lm_logits is not None and labels is not None:
                shift_labels: torch.Tensor = labels[..., 1:].contiguous()
                shift_logits: torch.Tensor = lm_logits[..., :-1, :].contiguous()
                tgt_len: torch.Tensor = (shift_labels != self.config.ignore_id).float().sum(1) + 1.
                lm_loss: Optional[torch.Tensor] = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none'
                ).view(shift_labels.size()).sum(1)
                lm_loss /= tgt_len
                if reduction:
                    lm_loss = lm_loss.mean()
                if kwargs.get('lm_loss', self.config.lm_loss):
                    loss += lm_loss
            else:
                lm_loss = None
            # Latent code loss
            if policy_logits is not None and (posterior_logits is not None or latent is not None):
                # KL Divergence
                if posterior_logits is not None:
                    latent_kl_div: Optional[torch.Tensor] = F.kl_div(
                        F.log_softmax(policy_logits, -1),
                        F.log_softmax(posterior_logits, -1),
                        reduction='none',
                        log_target=True
                    )
                    if self.config.kl_threshold > 0.:
                        latent_kl_div_threshold: Optional[torch.Tensor] = torch.max(
                            latent_kl_div, torch.full_like(latent_kl_div, self.config.kl_threshold)
                        ).sum(-1)
                        if reduction:
                            latent_kl_div_threshold = latent_kl_div_threshold.mean()
                    else:
                        latent_kl_div_threshold = None
                    latent_kl_div = latent_kl_div.sum(-1)
                    if reduction:
                        latent_kl_div = latent_kl_div.mean()
                else:
                    latent_kl_div = latent_kl_div_threshold = None
                # Latent negative log-likelihood
                if latent is not None:
                    latent_loss: Optional[torch.Tensor] = F.cross_entropy(policy_logits, latent, reduction='none')
                    if reduction:
                        latent_loss = latent_loss.mean()
                else:
                    latent_loss = None
                if kwargs.get('latent_loss', self.config.latent_loss):
                    if kwargs.get('detach_posterior', self.config.detach_posterior):
                        loss += latent_loss
                    else:
                        loss += latent_kl_div_threshold if latent_kl_div_threshold is not None else latent_kl_div
            else:
                latent_kl_div = latent_kl_div_threshold = latent_loss = None
            # Retrieval contrastive loss
            if cls_logits_pos is not None and cls_logits_neg is not None:
                cls_loss: Optional[torch.Tensor] = F.binary_cross_entropy_with_logits(
                    cls_logits_pos, torch.ones_like(cls_logits_pos), reduction='none'
                ) + F.binary_cross_entropy_with_logits(
                    cls_logits_neg, torch.zeros_like(cls_logits_neg), reduction='none'
                )
                if reduction:
                    cls_loss = cls_loss.mean()
                if kwargs.get('cls_loss', self.config.cls_loss):
                    loss += cls_loss
            else:
                cls_loss = None
            # BoW prediction loss
            if bow_logits is not None and labels is not None:
                bow_labels: torch.Tensor = labels.clone()
                bow_labels[bow_labels >= self.config.bow_size] = self.config.ignore_id
                tgt_len = (bow_labels != self.config.ignore_id).float().sum(1)
                bow_loss: Optional[torch.Tensor] = F.cross_entropy(
                    bow_logits.repeat(1, bow_labels.size(1), 1).view(-1, bow_logits.size(-1)), bow_labels.view(-1),
                    reduction='none'
                ).view(bow_labels.size()).sum(1)
                bow_loss /= tgt_len
                if reduction:
                    bow_loss = bow_loss.mean()
                if kwargs.get('bow_loss', self.config.bow_loss):
                    loss += bow_loss
            else:
                bow_loss = None
            # Expected reward
            if raw_reward is not None and target_reward is not None:
                # Change to torch.atanh(.)
                rew_loss = F.mse_loss(torch.tanh(raw_reward), target_reward, reduction='none')
                if reduction:
                    rew_loss = rew_loss.mean().square()
                if kwargs.get('rew_loss', self.config.rew_loss):
                    loss += rew_loss
            else:
                rew_loss = None
        else:
            loss = lm_loss = latent_kl_div = latent_kl_div_threshold = latent_loss = cls_loss = bow_loss = rew_loss = None

        # Compute objective
        if self.config.rl_weight > 0.0 and g_return is not None:
            # Cumulative objective
            objective = torch.zeros(1 if reduction else batch_size, device=self.device)
            # Language modeling objective
            if lm_logits is not None and labels is not None:
                shift_labels: torch.Tensor = labels[..., 1:].contiguous()
                shift_logits: torch.Tensor = lm_logits[..., :-1, :].contiguous()
                tgt_len: torch.Tensor = (shift_labels != self.config.ignore_id).float().sum(1) + 1
                lm_obj: Optional[torch.Tensor] = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none'
                ).view(shift_labels.size()).sum(1)
                lm_obj /= tgt_len
                lm_obj *= g_return
                if reduction:
                    lm_obj = lm_obj.mean()
                if kwargs.get('lm_obj', self.config.lm_obj):
                    objective += lm_obj
            else:
                lm_obj = None
            # Latent code objective
            if policy_logits is not None and latent is not None:
                latent_obj: Optional[torch.Tensor] = F.cross_entropy(policy_logits, latent, reduction='none')
                latent_obj *= g_return
                if reduction:
                    latent_obj = latent_obj.mean()
                if kwargs.get('latent_obj', self.config.latent_obj):
                    objective += latent_obj
            else:
                latent_obj = None
        else:
            objective = lm_obj = latent_obj = None

        # Compute cost function to optimize
        if loss is not None and objective is None:
            cost = loss
        elif loss is None and objective is not None:
            cost = objective
        else:
            cost = ((1 - self.config.rl_weight) * loss) + (self.config.rl_weight * objective)

        output = DLDLMCostFunctionOutput(
            cost=cost,
            loss=loss,
            objective=objective,
            lm_loss=lm_loss,
            latent_kl_div=latent_kl_div,
            latent_kl_div_threshold=latent_kl_div_threshold,
            latent_loss=latent_loss,
            cls_loss=cls_loss,
            bow_loss=bow_loss,
            rew_loss=rew_loss,
            lm_obj=lm_obj,
            latent_obj=latent_obj,
        )

        return output

    def compute_hidden_transformation(  # Wrapper to transformer forward
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, )
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + length)
            past_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, past_length)
            token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            token_type_fill_value: Optional[int] = None,
            position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, length)
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            hidden_states: Optional[Tuple[torch.FloatTensor]] = None,  # Shape ((num_hidden_layers + 1) * (batch_size, past_length, hidden_size))
    ) -> DLDLMModelOutput:
        if past_attention_mask is not None:
            if attention_mask is not None:
                attention_mask = torch.cat([past_attention_mask, attention_mask], dim=-1)
            else:
                if input_ids is not None:
                    attention_mask = torch.cat(
                        [past_attention_mask, torch.ones_like(input_ids, device=past_attention_mask.device)], dim=-1
                    )
                else:
                    attention_mask = torch.cat(
                        [past_attention_mask, torch.ones(input_embeds.size()[:-1], device=past_attention_mask.device)],
                        dim=-1
                    )
        if token_type_ids is None:
            if input_ids is not None:
                token_type_ids = torch.full_like(input_ids, token_type_fill_value)
            else:
                token_type_ids = torch.full(
                    input_embeds.size()[:-1],
                    token_type_fill_value,
                    device=input_embeds.device,
                    dtype=torch.long
                )
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.cumsum(dim=-1).long() - attention_mask.long()
                if past_key_values is not None:
                    position_ids = position_ids[:, past_key_values[0][0].size(2):]
            else:
                if input_ids is not None:
                    position_ids = torch.ones_like(input_ids).cumsum(dim=-1) - 1
                else:
                    position_ids = torch.ones(
                        input_embeds.size()[:-1], device=input_embeds.device, dtype=torch.long
                    ).cumsum(dim=-1) - 1
                if past_key_values is not None:
                    position_ids += past_key_values[0][0].size(2)

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
            hidden_states=(
                tuple(
                    torch.cat([h_past, h], dim=1) for h_past, h in zip(hidden_states, transformer_outputs.hidden_states)
                ) if hidden_states is not None else transformer_outputs.hidden_states
            ),
            attentions=transformer_outputs.attentions,
            attention_mask=attention_mask
        )

        return output

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,  # Shape (batch_size, response_length)
            past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
            **kwargs
    ):
        if not (isinstance(self, DLDLMAllHeadsModel) or isinstance(self, DLDLMLMHeadModel)):
            raise TypeError()
        # TODO rework to manage properly latent selection in case of multiple sequences
        if past is not None:
            # If the past is given then proceed in the generation as usual
            input_ids = input_ids[:, -1].unsqueeze(-1)
            # Manage attention
            context_attention_mask = kwargs.get(
                'context_attention_mask', torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
            )  # Shape (batch_size, context_length)
            latent_attention_mask = kwargs.get(
                'latent_attention_mask',
                torch.ones((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device)
            )  # Shape (batch_size, 1)
            attention_mask = kwargs.get('attention_mask', torch.ones_like(input_ids, device=input_ids.device))  # Shape (batch_size, response_length)
            try:
                attention_mask = torch.cat([context_attention_mask, latent_attention_mask, attention_mask], dim=-1)
            except RuntimeError:
                # In case of num return sequences > 1 or beam_size > 1 some elements must be repeated
                # Expand context attention (if required)
                if context_attention_mask.shape[0] != attention_mask.shape[0] and context_attention_mask.shape[0] > 0:
                    # Compute number of repetitions
                    expand_size = attention_mask.shape[0] // context_attention_mask.shape[0]
                    # Do expansion
                    context_attention_mask = context_attention_mask.repeat_interleave(expand_size, dim=0)
                # Expand latent attention (if required)
                if latent_attention_mask.shape[0] != attention_mask.shape[0]:
                    # Compute number of repetitions
                    expand_size = attention_mask.shape[0] // latent_attention_mask.shape[0]
                    # Do expansion
                    latent_attention_mask = latent_attention_mask.repeat_interleave(expand_size, dim=0)
                # Compute total attention mask
                attention_mask = torch.cat([context_attention_mask, latent_attention_mask, attention_mask], dim=-1)
                # Expand past key values (if required)
                if past[0][0].shape[0] != input_ids.shape[0]:
                    # Compute number of repetitions
                    expand_size = input_ids.shape[0] // past[0][0].shape[0]
                    # Do expansion
                    past = tuple(
                        (k.repeat_interleave(expand_size, dim=0), v.repeat_interleave(expand_size, dim=0))
                        for k, v in past
                    )

            token_type_ids = kwargs.get('token_type_ids', None)  # Shape (batch_size, response_length)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            position_ids = kwargs.get('position_ids', None)  # Shape (batch_size, response_length)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

            # Check if and how attention is updated
            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "use_cache": kwargs.get("use_cache"),
                'do_context': False,
                'do_policy': False,
                'do_response_encoding': False,
                'do_posterior': False,
                'do_latent': False,
                'do_response_decoding': True,
            }
        else:
            # Else encode context and latent
            # Context
            context_ids = kwargs.pop('context_ids', None)
            context_attention_mask = kwargs.get('context_attention_mask', None)
            context_token_type_ids = kwargs.pop('context_token_type_ids', None)
            context_position_ids = kwargs.pop('context_position_ids', None)
            # Latent
            latent_ids = kwargs.pop('latent_ids', None)
            latent_attention_mask = kwargs.get('latent_attention_mask', None)
            latent_token_type_ids = kwargs.pop('latent_token_type_ids', None)
            latent_position_ids = kwargs.pop('latent_position_ids', None)
            do_sample_latent = kwargs.pop('do_sample_latent', None)
            # Encoding
            context_latent_outputs = self(
                # input_ids=input_ids,
                context_ids=context_ids,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,
                context_position_ids=context_position_ids,
                latent_ids=latent_ids,
                latent_attention_mask=latent_attention_mask,
                latent_token_type_ids=latent_token_type_ids,
                latent_position_ids=latent_position_ids,
                do_sample_latent=do_sample_latent,
                do_context=True,
                do_policy=True,
                do_response_encoding=False,
                do_posterior=False,
                do_latent=True,
                do_response_decoding=False,
                latent_loss=False,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            past = context_latent_outputs.past_key_values

            return self.prepare_inputs_for_generation(
                input_ids,
                past=past,
                **kwargs
            )
    
    @torch.no_grad()
    def extend_from_gpt2_initialization(self, updated_tokenizer: DLDLMTokenizer):
        if len(updated_tokenizer) != self.config.vocab_size:
            self.resize_token_embeddings(new_num_tokens=len(updated_tokenizer))
            self.transformer.wte.weight[updated_tokenizer.convert_tokens_to_ids(['<s>', '<s/>'])] = \
                self.transformer.wte.weight[self.config.eos_token_id].detach().clone()
            self.config.bow_size = len(updated_tokenizer) - len(updated_tokenizer.unique_no_split_tokens)
        self.config.bos_token_id = updated_tokenizer.bos_token_id
        self.config.eos_token_id = updated_tokenizer.eos_token_id
        self.config.pad_token_id = updated_tokenizer.pad_token_id
        self.config.mask_token_id = updated_tokenizer.mask_token_id
        self.config.context_type_token_id = updated_tokenizer.convert_tokens_to_ids('</c>')
        self.config.response_type_token_id = updated_tokenizer.convert_tokens_to_ids('</r>')
        self.config.latent_type_token_id = updated_tokenizer.convert_tokens_to_ids('</l>')
        self.config.policy_token_id = updated_tokenizer.convert_tokens_to_ids('</p>')
        self.config.posterior_token_id = updated_tokenizer.convert_tokens_to_ids('</q>')
        self.config.latent_token_ids = updated_tokenizer.convert_tokens_to_ids(
            [f'</z_{i}>' for i in range(self.config.num_styles)]
        )

        return self


class DLDLMAllHeadsModel(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super(DLDLMAllHeadsModel, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.policy_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)
        self.posterior_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)
        self.bow_head = nn.Linear(config.hidden_size, config.bow_size, bias=False)
        self.cls_head = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        if config.num_rewards is not None:
            self.reward_head = nn.Linear(config.hidden_size, config.num_rewards, bias=False)

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
        context_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, context_length, hidden_size)
        context_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + context_length)
        context_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        latent_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        latent_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, 1, hidden_size)
        latent_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + 1)
        latent_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        latent_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        head_mask=None,
        labels: Optional[torch.LongTensor] = None,  # Shape (batch_size,)
        target_reward: Optional[torch.FloatTensor] = None,  # Shape (batch_size, num_rewards)
        distractor_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        distractor_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, distractor_length, hidden_size)
        distractor_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + distractor_length)
        distractor_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        distractor_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        g_return: Optional[torch.Tensor] = None,  # Shape (batch_size,)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        elif context_ids is not None:
            batch_size = context_ids.size(0)
        elif context_embeds is not None:
            batch_size = context_embeds.size(0)
        elif latent_ids is not None:
            batch_size = latent_ids.size(0)
        elif latent_embeds is not None:
            batch_size = latent_embeds.size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get device
        device = next(self.transformer.parameters()).device

        # Process context
        if (context_ids is not None or context_embeds is not None) and kwargs.get('do_context', self.config.do_context):
            context_hidden_outputs = self.compute_hidden_transformation(
                input_ids=context_ids,
                input_embeds=context_embeds,
                past_key_values=past_key_values,
                attention_mask=context_attention_mask,
                token_type_ids=context_token_type_ids,
                token_type_fill_value=self.config.context_type_token_id,
                position_ids=context_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = context_hidden_outputs.past_key_values
            hidden_states = context_hidden_outputs.hidden_states
        else:
            hidden_states = None
        # Process policy
        if kwargs.get('do_policy', self.config.do_policy):
            policy_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.policy_token_id, device=device),
                past_key_values=past_key_values,
                past_attention_mask=context_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            policy_last_hidden_state = policy_hidden_outputs.last_hidden_state.squeeze(1)
            policy_past_key_values = policy_hidden_outputs.past_key_values
            policy_hidden_states = policy_hidden_outputs.hidden_states
            policy_attentions = policy_hidden_outputs.attentions
            # Compute P
            policy_logits = self.policy_head(policy_last_hidden_state)
        else:
            policy_last_hidden_state = policy_past_key_values = policy_hidden_states = policy_attentions = None
            policy_logits = None
        # Process response encoding
        if (
                (input_ids is not None or input_embeds is not None) and
                kwargs.get('do_response_encoding', self.config.do_response_encoding)
        ):
            response_encoding_hidden_output = self.compute_hidden_transformation(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            posterior_past_key_values = response_encoding_hidden_output.past_key_values
            posterior_hidden_states = response_encoding_hidden_output.hidden_states
            posterior_attention_mask = response_encoding_hidden_output.attention_mask
        else:
            posterior_past_key_values = posterior_hidden_states = posterior_attention_mask = None
        # Process posterior
        if kwargs.get('do_posterior', self.config.do_posterior):
            posterior_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.posterior_token_id, device=device),
                past_key_values=posterior_past_key_values,
                past_attention_mask=posterior_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=posterior_hidden_states,
            )
            posterior_last_hidden_state = posterior_hidden_outputs.last_hidden_state.squeeze(1)
            posterior_past_key_values = posterior_hidden_outputs.past_key_values
            posterior_hidden_states = posterior_hidden_outputs.hidden_states
            posterior_attentions = posterior_hidden_outputs.attentions
            # Compute Q and IR logits
            posterior_logits = self.posterior_head(posterior_last_hidden_state)
            cls_logits_pos = self.cls_head(posterior_last_hidden_state)
        else:
            posterior_past_key_values = posterior_last_hidden_state = posterior_hidden_states = posterior_attentions = None
            posterior_logits = cls_logits_pos = None
        # Process contrastive samples
        if (
                (distractor_ids is not None or distractor_embeds is not None) and
                kwargs.get('do_response_encoding', self.config.do_response_encoding) and
                kwargs.get('do_posterior', self.config.do_posterior)
        ):
            distractor_hidden_outputs = self.compute_hidden_transformation(
                input_ids=distractor_ids,
                input_embeds=distractor_embeds,
                past_key_values=past_key_values,
                attention_mask=distractor_attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=distractor_token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=distractor_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            distractor_past_key_values = distractor_hidden_outputs.past_key_values
            distractor_attention_mask = distractor_hidden_outputs.attention_mask
            distractor_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.posterior_token_id, device=device),
                past_key_values=distractor_past_key_values,
                past_attention_mask=distractor_attention_mask,
                token_type_fill_value=self.config.response_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            distractor_last_hidden_state = distractor_hidden_outputs.last_hidden_state.squeeze(1)
            # Compute IR logits
            cls_logits_neg = self.cls_head(distractor_last_hidden_state)
        else:
            cls_logits_neg = None
        # Process latent
        if (
                (
                        latent_ids is not None or latent_embeds is not None or
                        posterior_logits is not None or policy_logits is not None
                ) and
                kwargs.get('do_latent', self.config.do_latent)
        ):
            if latent_ids is None and latent_embeds is None:
                if kwargs.get('do_sample_latent', self.config.do_sample_latent):
                    latent = torch.multinomial(
                        torch.softmax(posterior_logits if posterior_logits is not None else policy_logits, dim=-1), 1
                    )
                else:
                    latent = torch.argmax(posterior_logits if posterior_logits is not None else policy_logits, dim=-1)
                if self.training:
                    latent_embeds = torch.einsum(
                        'zh, bz -> bh',
                        self.transformer.wte.weight[self.config.latent_token_ids],
                        F.gumbel_softmax(
                            posterior_logits if posterior_logits is not None else policy_logits,
                            tau=self.config.gumbell_tau
                        )
                    ).unsqueeze(1)
                else:
                    latent_ids = torch.tensor(self.config.latent_token_ids, device=device)[latent].unsqueeze(-1)
            elif latent_ids is not None:
                _, latent = torch.where(torch.tensor(self.config.latent_token_ids, device=device) == latent_ids)
            else:
                latent = None

            latent_hidden_outputs = self.compute_hidden_transformation(
                input_ids=latent_ids,
                input_embeds=latent_embeds,
                past_key_values=past_key_values,
                attention_mask=latent_attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=latent_token_type_ids,
                token_type_fill_value=self.config.latent_type_token_id,
                position_ids=latent_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=hidden_states
            )
            latent_last_hidden_state = latent_hidden_outputs.last_hidden_state.squeeze(1)
            past_key_values = latent_hidden_outputs.past_key_values
            hidden_states = latent_hidden_outputs.hidden_states
            latent_attention_mask = latent_hidden_outputs.attention_mask
            # Compute BoW logits and raw reward
            bow_logits = self.bow_head(latent_last_hidden_state)
            raw_reward = self.reward_head(latent_last_hidden_state)
        else:
            latent = None
            latent_attention_mask = latent_last_hidden_state = None
            bow_logits = raw_reward = None
        # Process response decoding
        if (
                (input_ids is not None or input_embeds is not None) and
                kwargs.get('do_response_decoding', self.config.do_response_decoding)
        ):
            response_decoding_hidden_outputs = self.compute_hidden_transformation(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                past_attention_mask=latent_attention_mask,
                token_type_ids=token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=hidden_states
            )
            last_hidden_state = response_decoding_hidden_outputs.last_hidden_state
            past_key_values = response_decoding_hidden_outputs.past_key_values
            hidden_states = response_decoding_hidden_outputs.hidden_states
            attentions = response_decoding_hidden_outputs.attentions
            # Compute LM logits
            lm_logits = self.lm_head(last_hidden_state)
        else:
            hidden_states = attentions = None
            lm_logits = None

        cost_function_output: DLDLMCostFunctionOutput = self.compute_cost_function(
            lm_logits=lm_logits,
            labels=labels,
            policy_logits=policy_logits,
            posterior_logits=posterior_logits,
            latent=latent,
            cls_logits_pos=cls_logits_pos,
            cls_logits_neg=cls_logits_neg,
            bow_logits=bow_logits,
            raw_reward=raw_reward,
            target_reward=target_reward,
            g_return=g_return,
            batch_size=batch_size,
            **kwargs
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMAllHeadsModelOutput = DLDLMAllHeadsModelOutput(
            cost=cost_function_output.cost,
            loss=cost_function_output.loss,
            objective=cost_function_output.objective,
            cost_function_output=cost_function_output,
            logits=lm_logits,
            posterior_logits=posterior_logits,
            policy_logits=policy_logits,
            cls_logits=cls_logits_pos,
            cls_logits_contrastive=cls_logits_neg,
            bow_logits=bow_logits,
            raw_reward=raw_reward,
            latent=latent,
            latent_last_hidden_state=latent_last_hidden_state if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            policy_last_hidden_state=policy_last_hidden_state if output_hidden_states else None,
            policy_past_key_values=policy_past_key_values if use_cache else None,
            policy_hidden_states=policy_hidden_states if output_hidden_states else None,
            policy_attentions=policy_attentions if output_attentions else None,
            posterior_last_hidden_state=posterior_last_hidden_state if output_hidden_states else None,
            posterior_past_key_values=posterior_past_key_values if use_cache else None,
            posterior_hidden_states=posterior_hidden_states if output_hidden_states else None,
            posterior_attentions=posterior_attentions if output_attentions else None,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()

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
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.policy_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)

        self.init_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
        attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
        token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        context_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, context_length, hidden_size)
        context_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + context_length)
        context_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        latent_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        latent_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, 1, hidden_size)
        latent_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + 1)
        latent_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        latent_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, 1)
        head_mask=None,
        labels: Optional[torch.LongTensor] = None,  # Shape (batch_size,)
        g_return: Optional[torch.Tensor] = None,  # Shape (batch_size,)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        elif context_ids is not None:
            batch_size = context_ids.size(0)
        elif context_embeds is not None:
            batch_size = context_embeds.size(0)
        elif latent_ids is not None:
            batch_size = latent_ids.size(0)
        elif latent_embeds is not None:
            batch_size = latent_embeds.size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get device
        device = next(self.transformer.parameters()).device

        # Process context
        if (context_ids is not None or context_embeds is not None) and kwargs.get('do_context', self.config.do_context):
            context_hidden_outputs = self.compute_hidden_transformation(
                input_ids=context_ids,
                input_embeds=context_embeds,
                past_key_values=past_key_values,
                attention_mask=context_attention_mask,
                token_type_ids=context_token_type_ids,
                token_type_fill_value=self.config.context_type_token_id,
                position_ids=context_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = context_hidden_outputs.past_key_values
            hidden_states = context_hidden_outputs.hidden_states
        else:
            hidden_states = None
        # Process policy
        if kwargs.get('do_policy', self.config.do_policy):
            policy_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.policy_token_id, device=device),
                past_key_values=past_key_values,
                past_attention_mask=context_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            policy_last_hidden_state = policy_hidden_outputs.last_hidden_state.squeeze(1)
            policy_past_key_values = policy_hidden_outputs.past_key_values
            policy_hidden_states = policy_hidden_outputs.hidden_states
            policy_attentions = policy_hidden_outputs.attentions
            # Compute P
            policy_logits = self.policy_head(policy_last_hidden_state)
        else:
            policy_last_hidden_state = policy_past_key_values = policy_hidden_states = policy_attentions = None
            policy_logits = None
        # Process latent
        if (
                (latent_ids is not None or latent_embeds is not None or policy_logits is not None) and
                kwargs.get('do_latent', self.config.do_latent)
        ):
            if latent_ids is None and latent_embeds is None:
                if kwargs.get('do_sample_latent', self.config.do_sample_latent):
                    latent = torch.multinomial(
                        torch.softmax(policy_logits, dim=-1), 1
                    )
                else:
                    latent = torch.argmax(policy_logits, dim=-1)
                if self.training:
                    latent_embeds = torch.einsum(
                        'zh, bz -> bh',
                        self.transformer.wte.weight[self.config.latent_token_ids],
                        F.gumbel_softmax(policy_logits, tau=self.config.gumbell_tau)
                    ).unsqueeze(1)
                else:
                    latent_ids = torch.tensor(self.config.latent_token_ids)[latent].unsqueeze(-1)
            elif latent_ids is not None:
                _, latent = torch.where(torch.tensor(self.config.latent_token_ids) == latent_ids)
            else:
                latent = None

            latent_hidden_outputs = self.compute_hidden_transformation(
                input_ids=latent_ids,
                input_embeds=latent_embeds,
                past_key_values=past_key_values,
                attention_mask=latent_attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=latent_token_type_ids,
                token_type_fill_value=self.config.latent_type_token_id,
                position_ids=latent_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=hidden_states
            )
            latent_last_hidden_state = latent_hidden_outputs.last_hidden_state.squeeze(1)
            past_key_values = latent_hidden_outputs.past_key_values
            hidden_states = latent_hidden_outputs.hidden_states
            latent_attention_mask = latent_hidden_outputs.attention_mask
        else:
            latent = None
            latent_attention_mask = latent_last_hidden_state = None
        # Process response decoding
        if (
                (input_ids is not None or input_embeds is not None) and
                kwargs.get('do_response_decoding', self.config.do_response_decoding)
        ):
            response_decoding_hidden_outputs = self.compute_hidden_transformation(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                past_attention_mask=latent_attention_mask,
                token_type_ids=token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=hidden_states
            )
            last_hidden_state = response_decoding_hidden_outputs.last_hidden_state
            past_key_values = response_decoding_hidden_outputs.past_key_values
            hidden_states = response_decoding_hidden_outputs.hidden_states
            attentions = response_decoding_hidden_outputs.attentions
            # Compute LM logits
            lm_logits = self.lm_head(last_hidden_state)
        else:
            hidden_states = attentions = None
            lm_logits = None

        cost_function_output: DLDLMCostFunctionOutput = self.compute_cost_function(
            lm_logits=lm_logits,
            labels=labels,
            policy_logits=policy_logits,
            latent=latent,
            g_return=g_return,
            batch_size=batch_size,
            **kwargs
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMPolicyLMHeadModelOutput = DLDLMPolicyLMHeadModelOutput(
            cost=cost_function_output.cost,
            cost_function_output=cost_function_output,
            logits=lm_logits,
            policy_logits=policy_logits,
            latent=latent,
            latent_last_hidden_state=latent_last_hidden_state if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            policy_last_hidden_state=policy_last_hidden_state if output_hidden_states else None,
            policy_past_key_values=policy_past_key_values if use_cache else None,
            policy_hidden_states=policy_hidden_states if output_hidden_states else None,
            policy_attentions=policy_attentions if output_attentions else None,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class DLDLMPosteriorSequenceClassification(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super(DLDLMPosteriorSequenceClassification, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output head
        self.posterior_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
        attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
        token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        context_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, context_length, hidden_size)
        context_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + context_length)
        context_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        elif context_ids is not None:
            batch_size = context_ids.size(0)
        elif context_embeds is not None:
            batch_size = context_embeds.size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get device
        device = next(self.transformer.parameters()).device

        # Process context
        if (context_ids is not None or context_embeds is not None) and kwargs.get('do_context', self.config.do_context):
            context_hidden_outputs = self.compute_hidden_transformation(
                input_ids=context_ids,
                input_embeds=context_embeds,
                past_key_values=past_key_values,
                attention_mask=context_attention_mask,
                token_type_ids=context_token_type_ids,
                token_type_fill_value=self.config.context_type_token_id,
                position_ids=context_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = context_hidden_outputs.past_key_values
        # Process response encoding
        if (
                (input_ids is not None or input_embeds is not None) and
                kwargs.get('do_response_encoding', self.config.do_response_encoding)
        ):
            response_encoding_hidden_output = self.compute_hidden_transformation(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            posterior_past_key_values = response_encoding_hidden_output.past_key_values
            posterior_hidden_states = response_encoding_hidden_output.hidden_states
            posterior_attention_mask = response_encoding_hidden_output.attention_mask
        else:
            posterior_past_key_values = posterior_hidden_states = posterior_attention_mask = None
        # Process posterior
        if kwargs.get('do_posterior', self.config.do_posterior):
            posterior_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.posterior_token_id, device=device),
                past_key_values=posterior_past_key_values,
                past_attention_mask=posterior_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=posterior_hidden_states,
            )
            posterior_last_hidden_state = posterior_hidden_outputs.last_hidden_state.squeeze(1)
            posterior_past_key_values = posterior_hidden_outputs.past_key_values
            posterior_hidden_states = posterior_hidden_outputs.hidden_states
            posterior_attentions = posterior_hidden_outputs.attentions
            # Compute Q and IR logits
            posterior_logits = self.posterior_head(posterior_last_hidden_state)
        else:
            posterior_past_key_values = posterior_last_hidden_state = posterior_hidden_states = posterior_attentions = None
            posterior_logits = None

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMPosteriorSequenceClassifierOutput = DLDLMPosteriorSequenceClassifierOutput(
            logits=posterior_logits,
            last_hidden_state=posterior_last_hidden_state if output_hidden_states else None,
            past_key_values=posterior_past_key_values if use_cache else None,
            hidden_states=posterior_hidden_states if output_hidden_states else None,
            attentions=posterior_attentions if output_attentions else None,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()


class DLDLMIRHeadModel(DLDLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super(DLDLMIRHeadModel, self).__init__(config)

        # Hidden layers
        self.transformer: DLDLMModel = DLDLMModel(config)
        # Output head
        self.cls_head = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        input_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, response_length, hidden_size)
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,# Shape (num_hidden_layers * (2 * (batch_size, num_heads, past_length, embed_size_per_head)))
        attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + response_length)
        token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, response_length)
        context_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, context_length, hidden_size)
        context_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + context_length)
        context_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        context_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, context_length)
        head_mask=None,
        distractor_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        distractor_embeds: Optional[torch.FloatTensor] = None,  # Shape (batch_size, distractor_length, hidden_size)
        distractor_attention_mask: Optional[torch.LongTensor] = None,  # Shape (batch_size, [past_length] + distractor_length)
        distractor_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        distractor_position_ids: Optional[torch.LongTensor] = None,  # Shape (batch_size, distractor_length)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        elif context_ids is not None:
            batch_size = context_ids.size(0)
        elif context_embeds is not None:
            batch_size = context_embeds.size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get device
        device = next(self.transformer.parameters()).device

        # Process context
        if (context_ids is not None or context_embeds is not None) and kwargs.get('do_context', self.config.do_context):
            context_hidden_outputs = self.compute_hidden_transformation(
                input_ids=context_ids,
                input_embeds=context_embeds,
                past_key_values=past_key_values,
                attention_mask=context_attention_mask,
                token_type_ids=context_token_type_ids,
                token_type_fill_value=self.config.context_type_token_id,
                position_ids=context_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = context_hidden_outputs.past_key_values
        # Process response encoding
        if (
                (input_ids is not None or input_embeds is not None) and
                kwargs.get('do_response_encoding', self.config.do_response_encoding)
        ):
            response_encoding_hidden_output = self.compute_hidden_transformation(
                input_ids=input_ids,
                input_embeds=input_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            posterior_past_key_values = response_encoding_hidden_output.past_key_values
            posterior_hidden_states = response_encoding_hidden_output.hidden_states
            posterior_attention_mask = response_encoding_hidden_output.attention_mask
        else:
            posterior_past_key_values = posterior_hidden_states = posterior_attention_mask = None
        # Process posterior
        if kwargs.get('do_posterior', self.config.do_posterior):
            posterior_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.posterior_token_id, device=device),
                past_key_values=posterior_past_key_values,
                past_attention_mask=posterior_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                hidden_states=posterior_hidden_states,
            )
            posterior_last_hidden_state = posterior_hidden_outputs.last_hidden_state.squeeze(1)
            posterior_past_key_values = posterior_hidden_outputs.past_key_values
            posterior_hidden_states = posterior_hidden_outputs.hidden_states
            posterior_attentions = posterior_hidden_outputs.attentions
            # Compute Q and IR logits
            cls_logits_pos = self.cls_head(posterior_last_hidden_state)
        else:
            posterior_past_key_values = posterior_last_hidden_state = posterior_hidden_states = posterior_attentions = None
            cls_logits_pos = None
        # Process contrastive samples
        if (
                (distractor_ids is not None or distractor_embeds is not None) and
                kwargs.get('do_response_encoding', self.config.do_response_encoding) and
                kwargs.get('do_posterior', self.config.do_posterior)
        ):
            distractor_hidden_outputs = self.compute_hidden_transformation(
                input_ids=distractor_ids,
                input_embeds=distractor_embeds,
                past_key_values=past_key_values,
                attention_mask=distractor_attention_mask,
                past_attention_mask=context_attention_mask,
                token_type_ids=distractor_token_type_ids,
                token_type_fill_value=self.config.response_type_token_id,
                position_ids=distractor_position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            distractor_past_key_values = distractor_hidden_outputs.past_key_values
            distractor_attention_mask = distractor_hidden_outputs.attention_mask
            distractor_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.posterior_token_id, device=device),
                past_key_values=distractor_past_key_values,
                past_attention_mask=distractor_attention_mask,
                token_type_fill_value=self.config.response_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            distractor_last_hidden_state = distractor_hidden_outputs.last_hidden_state.squeeze(1)
            # Compute IR logits
            cls_logits_neg = self.cls_head(distractor_last_hidden_state)
        else:
            cls_logits_neg = None

        cost_function_output: DLDLMCostFunctionOutput = self.compute_cost_function(
            cls_logits_pos=cls_logits_pos,
            cls_logits_neg=cls_logits_neg,
            batch_size=batch_size,
            **kwargs
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMIRSequenceClassifierOutput = DLDLMIRSequenceClassifierOutput(
            loss=cost_function_output.loss,
            cost_function_output=cost_function_output,
            last_hidden_state=posterior_last_hidden_state if output_hidden_states else None,
            past_key_values=posterior_past_key_values if use_cache else None,
            hidden_states=posterior_hidden_states if output_hidden_states else None,
            attentions=posterior_attentions if output_attentions else None,
        )

        if return_dict:
            return output
        else:
            return output.to_tuple()
