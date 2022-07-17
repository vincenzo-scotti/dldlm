from typing import Optional, List

from transformers import GPT2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)

DLDLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dldlm-distilled": "https://huggingface.co/vincenzo-scotti/dldlm-distilled/resolve/main/config.json",
    "dldlm-small": "https://huggingface.co/vincenzo-scotti/dldlm-small/resolve/main/config.json",
    "dldlm-medium": "https://huggingface.co/vincenzo-scotti/dldlm-medium/resolve/main/config.json",
    # See all Discrete Latent Dialogue Language Model models at https://huggingface.co/models?filter=dldlm
}


class DLDLMConfig(GPT2Config):
    model_type = "dldlm"
    # TODO check attribute_map static attribute

    def __init__(
            self,
            n_styles: Optional[int] = None,
            tf_size: Optional[int] = None,
            init_tf_head: bool = True,

            context_pdrop: float = 0.5,
            corruption_rate: float = 0.2,
            do_sample_latent: bool = True,
            fixed_prior: bool = False,
            do_gibbs_sampling: bool = False,

            unknown_token_id: Optional[int] = None,
            mask_token_id: Optional[int] = None,
            prior_token_id: Optional[int] = None,
            posterior_token_id: Optional[int] = None,
            latent_token_ids: Optional[List[int]] = None,

            reduction: bool = True,
            lm_loss: bool = True,
            kl_loss: bool = True,
            behaviour_loss: bool = False,
            gibbs_sampling_loss: bool = False,
            prior_entropy_loss: bool = False,
            posterior_entropy_loss: bool = False,
            tf_loss: bool = True,
            lm_loss_weight: float = 1.0,
            kl_loss_weight: float = 1.0,
            kl_loss_threshold: float = -float('inf'),
            behaviour_loss_weight: float = 1.0,
            gibbs_sampling_loss_weight: float = 1.0,
            prior_entropy_loss_weight: float = 1.0,
            posterior_entropy_loss_weight: float = 1.0,
            tf_loss_weight: float = 1.0,

            **kwargs,
    ):
        # Model structure hyper-parameters
        self.n_styles: Optional[int] = n_styles
        self.tf_size: Optional[int] = tf_size
        self.init_tf_head: bool = init_tf_head
        # Latent analysis hyper-parameters and configs
        self.context_pdrop: float = context_pdrop
        self.corruption_rate: float = corruption_rate
        self.fixed_prior: bool = fixed_prior
        self.do_sample_latent: bool = do_sample_latent
        self.do_gibbs_sampling: bool = do_gibbs_sampling
        # Special tokens
        self.unknown_token_id: Optional[int] = unknown_token_id
        self.mask_token_id: Optional[int] = mask_token_id
        self.prior_token_id: Optional[int] = prior_token_id
        self.posterior_token_id: Optional[int] = posterior_token_id
        self.latent_token_ids: Optional[List[int]] = latent_token_ids
        self.ignore_id: int = -1
        # Losses flags and weights
        self.reduction: bool = reduction
        self.lm_loss: bool = lm_loss
        self.kl_loss: bool = kl_loss
        self.behaviour_loss: bool = behaviour_loss
        self.gibbs_sampling_loss: bool = gibbs_sampling_loss and self.do_gibbs_sampling
        self.prior_entropy_loss: bool = prior_entropy_loss
        self.posterior_entropy_loss: bool = posterior_entropy_loss
        self.tf_loss: bool = tf_loss
        self.lm_loss_weight: float = lm_loss_weight
        self.kl_loss_weight: float = kl_loss_weight
        self.kl_loss_threshold: float = kl_loss_threshold
        self.behaviour_loss_weight: float = behaviour_loss_weight
        self.gibbs_sampling_loss_weight: float = gibbs_sampling_loss_weight
        self.prior_entropy_loss_weight: float = prior_entropy_loss_weight
        self.posterior_entropy_loss_weight: float = posterior_entropy_loss_weight
        self.tf_loss_weight: float = tf_loss_weight

        super(DLDLMConfig, self).__init__(**kwargs)  # TODO fix unwanted parameters

    @property
    def num_styles(self):
        return self.n_styles
