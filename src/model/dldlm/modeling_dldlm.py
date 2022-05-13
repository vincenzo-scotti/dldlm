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
    latent_nll_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    tf_loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)


@dataclass
class DLDLMModelOutput(BaseModelOutputWithPast):
    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [past_length] + length + length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, length, [past_length] + length))
    attention_mask: Optional[torch.LongTensor] = None  # Shape (batch_size, [past_length] + length)


@dataclass
class DLDLMFullModelOutput(CausalLMOutputWithPast):
    loss: Optional[torch.Tensor] = None  # Shape (1,) or (batch_size,)
    loss_function_output: Optional[DLDLMLossFunctionOutput] = None

    logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, response_length, vocab_size)
    prior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    posterior_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, num_styles)
    tf_logits: Optional[torch.FloatTensor] = None  # Shape (batch_size, tf_size)

    latent: Optional[torch.FloatTensor] = None  # Shape (batch_size,)

    latent_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * (2 * (batch_size, num_heads, [[[context_length] + 1] + response_length], embed_size_per_head)))
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (num_hidden_layers * (batch_size, [[[context_length] + 1] + response_length], hidden_size))
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, response_length, [[context_length] + 1] + response_length))

    prior_last_hidden_state: Optional[torch.Tensor] = None  # Shape (batch_size, hidden_size)
    prior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[context_length] + 1], embed_size_per_head))
    prior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[context_length] + 1], hidden_size)
    prior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [context_length] + 1))
    posterior_last_hidden_state: Optional[torch.Tensor] = None # Shape (batch_size, hidden_size)
    posterior_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Shape (num_hidden_layers * 2 * (batch_size, num_heads, [[[context_length] + response_length] + 1], embed_size_per_head))
    posterior_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Shape (batch_size, [[[context_length] + response_length] + 1], hidden_size)
    posterior_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Shape ((num_hidden_layers + 1) * (batch_size, num_heads, 1, [[context_length] + response_length] + 1))


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


class DLDLMPreTrainedModel(GPT2PreTrainedModel):
    config_class = DLDLMConfig
    load_tf_weights = load_tf_weights_in_dldlm
    # base_model_prefix = "discrete_latent_transformer"  # TODO decide whether to leave gpt2 or find a fix for this base model issue
    is_parallelizable = False
    supports_gradient_checkpointing = False

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
            tgt_len: torch.Tensor = (shift_labels != self.config.ignore_id).float().sum(1) + 1.
            lm_loss: Optional[torch.Tensor] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none'
            ).view(shift_labels.size()).sum(1)
            lm_loss /= tgt_len
            if reduction:
                lm_loss = lm_loss.mean()
            if kwargs.get('lm_loss', self.config.lm_loss):
                loss += kwargs.get('lm_loss_weight', self.config.lm_loss_weight) * lm_loss
        else:
            lm_loss = None
        # Latent code loss
        if prior_logits is not None and (posterior_logits is not None or latent is not None):
            # KL Divergence
            if posterior_logits is not None:
                latent_kl_div_loss: Optional[torch.Tensor] = F.kl_div(
                    F.log_softmax(prior_logits, -1),
                    F.log_softmax(posterior_logits, -1),
                    reduction='none',
                    log_target=True
                ).sum(-1)
                if reduction:
                    latent_kl_div_loss = latent_kl_div_loss.mean()
            else:
                latent_kl_div_loss = None
            # Negative Log-Likelihood
            if latent is not None:
                latent_nll_loss: Optional[torch.Tensor] = F.cross_entropy(prior_logits, latent, reduction='none')
                if reduction:
                    latent_nll_loss = latent_nll_loss.mean()
            else:
                latent_nll_loss = None
            if kwargs.get('latent_loss', self.config.latent_loss):
                if kwargs.get('detach_posterior', self.config.detach_posterior):
                    loss += kwargs.get('latent_loss_weight', self.config.latent_loss_weight) * latent_nll_loss
                else:
                    loss += kwargs.get('latent_loss_weight', self.config.latent_loss_weight) * latent_kl_div_loss
        else:
            latent_kl_div_loss = latent_nll_loss = None
        # tf prediction loss
        if tf_logits is not None and labels is not None:
            tf_labels: torch.Tensor = labels.clone()
            tf_labels[tf_labels >= self.config.tf_size] = self.config.ignore_id
            tgt_len = (tf_labels != self.config.ignore_id).float().sum(1)
            tf_loss: Optional[torch.Tensor] = F.cross_entropy(
                tf_logits.repeat(1, tf_labels.size(1), 1).view(-1, tf_logits.size(-1)), tf_labels.view(-1),
                reduction='none'
            ).view(tf_labels.size()).sum(1)
            tf_loss /= tgt_len
            if reduction:
                tf_loss = tf_loss.mean()
            if kwargs.get('tf_loss', self.config.tf_loss):
                loss += kwargs.get('tf_loss_weight', self.config.tf_loss_weight) * tf_loss
        else:
            tf_loss = None

        output = DLDLMLossFunctionOutput(
            loss=loss,
            lm_loss=lm_loss,
            latent_kl_div_loss=latent_kl_div_loss,
            latent_nll_loss=latent_nll_loss,
            tf_loss=tf_loss
        )

        return output

    def compute_hidden_transformation(  # Wrapper to transformer forward
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
            hidden_states: Optional[Tuple[torch.FloatTensor]] = None,  # Shape ((num_hidden_layers + 1) * (batch_size, past_length, hidden_size))
    ) -> DLDLMModelOutput:
        # Attention and token types are taken as they are since now it is a single sequence
        # There is no split between context and response
        if position_ids is None:
            if attention_mask is not None:  # Compute position IDs from attention
                position_ids = attention_mask.cumsum(dim=-1).long() - attention_mask.long()
                if past_key_values is not None:
                    position_ids = position_ids[:, past_key_values[0][0].size(2):]
            else:  # In this case all positions are to be attended
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
        if not (isinstance(self, DLDLMFullModel) or isinstance(self, DLDLMLMHeadModel)):
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
                'do_prior': False,
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
                do_prior=True,
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

    @staticmethod
    def context_dropout(
            attention_mask: torch.Tensor,
            sep_idxs: Tuple[torch.Tensor],
            p: float = 0.5,
            training: bool = True,
            inplace: bool = False
    ) -> torch.Tensor:
        # Check if it's training or not
        if training:
            # Boolean mask to identify the context tokens
            context_mask = torch.zeros_like(attention_mask)
            context_mask[sep_idxs] = 1
            context_mask = (1 - context_mask.cumsum(dim=-1)).bool()
            # Vector of 1 and 0 values of the same size of the batch size (represents dropped contexts)
            dropout_mask = ((1 - torch.rand((attention_mask.shape(0), 1))) > p).long()
            # Zero out attention mask of dropped contexts
            if not inplace:
                attention_mask = attention_mask.clone()
            attention_mask[context_mask] = (attention_mask * dropout_mask)[context_mask]
        # Return attention mask
        return attention_mask
    
    @torch.no_grad()
    def extend_from_gpt2_initialization(self, updated_tokenizer: DLDLMTokenizer):
        if len(updated_tokenizer) != self.config.vocab_size:
            self.resize_token_embeddings(new_num_tokens=len(updated_tokenizer))
        self.config.pad_token_id = updated_tokenizer.eos_token_id
        self.config.unk_token_id = updated_tokenizer.unk_token_id
        self.config.mask_token_id = updated_tokenizer.mask_token_id
        self.config.prior_token_id = updated_tokenizer.convert_tokens_to_ids('<|prior|>')
        self.config.posterior_token_id = updated_tokenizer.convert_tokens_to_ids('<|posterior|>')
        self.config.latent_token_ids = updated_tokenizer.convert_tokens_to_ids(
            [f'<|latentcode{i}|>' for i in range(self.config.num_styles)]
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
        self.tf_head = nn.Linear(config.hidden_size, config.tf_size, bias=False)

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
        # Get mini_batch size
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif input_embeds is not None:
            batch_size = input_embeds.size(0)
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].size(0)
        else:
            batch_size = kwargs.get('batch_size', None)
        # Get special token positions
        prior_token_id_idxs = torch.where(input_ids == self.config.prior_token_id)
        posterior_token_id_idxs = torch.where(input_ids == self.config.posterior_token_id)
        # Get response start idx
        _, (response_start_idx, *_) = prior_token_id_idxs
        response_start_idx += 1
        assert all(idx == (response_start_idx - 1) for idx in prior_token_id_idxs[1]), \
            "Responses must be aligned (start from same position) to use this model."
        # Get device
        device = next(self.transformer.parameters()).device

        # ANALYSIS
        # Dialogue analysis step
        analysis_hidden_outputs = self.compute_hidden_transformation(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        past_key_values = analysis_hidden_outputs.past_key_values
        hidden_states = analysis_hidden_outputs.hidden_states
        last_hidden_state = analysis_hidden_outputs.last_hidden_state
        # Latent analysis step
        # Latent mask to prevent using common tokens
        latent_mask = torch.zeros((1, self.config.vocab_size), device=device)
        latent_mask[:, self.config.latent_token_ids] = 1.
        latent_mask = torch.log(latent_mask)
        # Prior
        prior_last_hidden_state = last_hidden_state[prior_token_id_idxs]
        prior_logits = self.lm_head(prior_last_hidden_state) + latent_mask
        # Posterior
        posterior_last_hidden_state = last_hidden_state[posterior_token_id_idxs]
        posterior_logits = self.lm_head(posterior_last_hidden_state) + latent_mask

        # DECODING
        # Latent decoding step
        if kwargs.get('do_sample_latent', self.config.do_sample_latent):
            latent_ids = torch.multinomial(
                torch.softmax(posterior_logits if posterior_logits is not None else prior_logits, dim=-1), 1
            )
        else:
            latent_ids = torch.argmax(
                posterior_logits if posterior_logits is not None else prior_logits, dim=-1
            ).unsqueeze(-1)
        latent = latent_ids.squeeze()
        # Update inputs and memories
        # Substitute analysis tokens
        attention_mask = self.context_dropout(
            attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            prior_token_id_idxs,
            p=self.config.context_pdrop,
            training=self.training
        )
        past_key_values = tuple(
            (k[:, :, :response_start_idx - 1], k[:, :, :response_start_idx - 1]) for (k, v) in past_key_values
        ) if response_start_idx > 1 else None
        input_ids = input_ids.clone()
        input_ids[prior_token_id_idxs] = latent
        input_ids[posterior_token_id_idxs] = self.config.eos_token_id
        input_ids = input_ids[:, response_start_idx - 1:]
        token_type_ids = token_type_ids[:, response_start_idx - 1:] if token_type_ids is not None else None
        position_ids = position_ids[:, response_start_idx - 1:] if position_ids is not None else None
        # Possibly process latent embedding
        if self.training:
            # Compute latent embedding
            latent_embeds = torch.einsum(
                'vh, bv -> bh',
                self.transformer.wte.weight,
                F.gumbel_softmax(
                    posterior_logits if posterior_logits is not None else prior_logits,
                    tau=self.config.gumbell_tau
                )
            ).unsqueeze(1)
            # Decode latent embedding
            latent_hidden_outputs = self.compute_hidden_transformation(
                input_embeds=latent_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask[:, :response_start_idx],
                token_type_ids=token_type_ids[:, :1] if token_type_ids is not None else None,
                position_ids=position_ids[:, :1] if position_ids is not None else None,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            input_ids = input_ids[1:]
            past_key_values = latent_hidden_outputs.past_key_values
            token_type_ids = token_type_ids[:, 1:] if token_type_ids is not None else None
            position_ids = position_ids[:, 1:] if position_ids is not None else None
            # Compute TF model logits
            latent_last_hidden_state = latent_hidden_outputs.latent_last_hidden_state[:, 0]
            tf_logits = self.tf_head(latent_last_hidden_state)
        else:
            latent_embeds = latent_last_hidden_state = tf_logits = None
        # Process response decoding
        decoding_hidden_outputs = self.compute_hidden_transformation(
            input_ids=input_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hidden_states=hidden_states
        )
        last_hidden_state = decoding_hidden_outputs.last_hidden_state
        past_key_values = decoding_hidden_outputs.past_key_values
        hidden_states = decoding_hidden_outputs.hidden_states
        attentions = decoding_hidden_outputs.attentions
        # Compute LM logits
        lm_logits = self.lm_head(last_hidden_state)
        if tf_logits is None:

        loss_function_output: DLDLMLossFunctionOutput = self.compute_loss_function(
            lm_logits=lm_logits,
            labels=labels,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            latent=latent,
            cls_logits_pos=cls_logits_pos,
            cls_logits_neg=cls_logits_neg,
            tf_logits=tf_logits,
            raw_reward=raw_reward,
            batch_size=batch_size,
            **kwargs
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMFullModelOutput = DLDLMFullModelOutput(
            loss=loss_function_output.loss,
            loss_function_output=loss_function_output,
            logits=lm_logits,
            posterior_logits=posterior_logits,
            prior_logits=prior_logits,
            tf_logits=tf_logits,
            latent=latent,
            latent_last_hidden_state=latent_last_hidden_state if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            prior_last_hidden_state=prior_last_hidden_state if output_hidden_states else None,
            prior_past_key_values=prior_past_key_values if use_cache else None,
            prior_hidden_states=prior_hidden_states if output_hidden_states else None,
            prior_attentions=prior_attentions if output_attentions else None,
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
        self.prior_head = nn.Linear(config.hidden_size, config.num_styles, bias=False)

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
        # Process prior
        if kwargs.get('do_prior', self.config.do_prior):
            prior_hidden_outputs = self.compute_hidden_transformation(
                input_ids=torch.full((batch_size, 1), self.config.prior_token_id, device=device),
                past_key_values=past_key_values,
                past_attention_mask=context_attention_mask,
                token_type_fill_value=self.config.latent_type_token_id,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            prior_last_hidden_state = prior_hidden_outputs.last_hidden_state.squeeze(1)
            prior_past_key_values = prior_hidden_outputs.past_key_values
            prior_hidden_states = prior_hidden_outputs.hidden_states
            prior_attentions = prior_hidden_outputs.attentions
            # Compute P
            prior_logits = self.prior_head(prior_last_hidden_state)
        else:
            prior_last_hidden_state = prior_past_key_values = prior_hidden_states = prior_attentions = None
            prior_logits = None
        # Process latent
        if (
                (latent_ids is not None or latent_embeds is not None or prior_logits is not None) and
                kwargs.get('do_latent', self.config.do_latent)
        ):
            if latent_ids is None and latent_embeds is None:
                if kwargs.get('do_sample_latent', self.config.do_sample_latent):
                    latent = torch.multinomial(
                        torch.softmax(prior_logits, dim=-1), 1
                    )
                else:
                    latent = torch.argmax(prior_logits, dim=-1)
                if self.training:
                    latent_embeds = torch.einsum(
                        'zh, bz -> bh',
                        self.transformer.wte.weight[self.config.latent_token_ids],
                        F.gumbel_softmax(prior_logits, tau=self.config.gumbell_tau)
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

        cost_function_output: DLDLMLossFunctionOutput = self.compute_loss_function(
            lm_logits=lm_logits,
            labels=labels,
            prior_logits=prior_logits,
            latent=latent,
            g_return=g_return,
            batch_size=batch_size,
            **kwargs
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output: DLDLMpriorLMHeadModelOutput = DLDLMpriorLMHeadModelOutput(
            cost=cost_function_output.cost,
            cost_function_output=cost_function_output,
            logits=lm_logits,
            prior_logits=prior_logits,
            latent=latent,
            latent_last_hidden_state=latent_last_hidden_state if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            prior_last_hidden_state=prior_last_hidden_state if output_hidden_states else None,
            prior_past_key_values=prior_past_key_values if use_cache else None,
            prior_hidden_states=prior_hidden_states if output_hidden_states else None,
            prior_attentions=prior_attentions if output_attentions else None,
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
