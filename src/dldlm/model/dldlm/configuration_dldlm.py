from typing import Optional, List

from transformers import GPT2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)

DLDLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dldlm-large": "https://huggingface.co/vincenzo-scotti/dldlm-large/resolve/main/config.json",
    "dldlm-large-therapy": "https://huggingface.co/vincenzo-scotti/dldlm-large-therapy/resolve/main/config.json",
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
            retrieval_head: bool = False,

            unconditioned: bool = False,
            context_pdrop: float = 0.5,
            corruption_rate: float = 0.1,
            do_sample_latent: bool = True,
            use_prior: bool = True,
            fixed_prior: bool = False,
            detach_posterior: bool = False,
            do_gibbs_sampling: bool = False,
            sampling_tau: float = 1.0,

            unknown_token_id: Optional[int] = None,
            mask_token_id: Optional[int] = None,
            prior_token_id: Optional[int] = None,
            posterior_token_id: Optional[int] = None,
            latent_token_ids: Optional[List[int]] = None,

            reduction: bool = True,
            lm_loss: bool = True,
            kl_loss: bool = True,
            behaviour_loss: bool = False,
            sampling_loss: bool = False,
            prior_entropy_loss: bool = False,
            posterior_entropy_loss: bool = False,
            tf_loss: bool = True,
            retrieval_loss: bool = False,
            lm_loss_weight: float = 1.0,
            kl_loss_weight: float = 1.0,
            kl_loss_threshold: float = -float('inf'),
            latent_mixing_weight: float = 0.0,
            behaviour_loss_weight: float = 1.0,
            behaviour_loss_threshold: float = -float('inf'),
            sampling_loss_weight: float = 1.0,
            sampling_loss_threshold: float = -float('inf'),
            prior_entropy_loss_weight: float = -1.0,
            posterior_entropy_loss_weight: float = 1.0,
            tf_loss_weight: float = 1.0,
            retrieval_loss_weight: float = 1.0,

            **kwargs,
    ):
        # Model structure hyper-parameters
        self.n_styles: Optional[int] = n_styles
        self.tf_size: Optional[int] = tf_size
        self.init_tf_head: bool = init_tf_head
        self.retrieval_head: bool = retrieval_head
        # Latent analysis hyper-parameters and configs
        self.unconditioned: bool = unconditioned
        self.context_pdrop: float = context_pdrop
        self.corruption_rate: float = corruption_rate
        self.use_prior: bool = use_prior
        self.fixed_prior: bool = fixed_prior
        self.do_sample_latent: bool = do_sample_latent
        self.detach_posterior: bool = detach_posterior
        self.do_gibbs_sampling: bool = do_gibbs_sampling
        self.sampling_tau: float = sampling_tau
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
        self.sampling_loss: bool = sampling_loss
        self.prior_entropy_loss: bool = prior_entropy_loss
        self.posterior_entropy_loss: bool = posterior_entropy_loss
        self.tf_loss: bool = tf_loss
        self.retrieval_loss: bool = retrieval_loss
        self.lm_loss_weight: float = lm_loss_weight
        self.kl_loss_weight: float = kl_loss_weight
        self.kl_loss_threshold: float = kl_loss_threshold
        self.latent_mixing_weight: float = latent_mixing_weight
        self.behaviour_loss_weight: float = behaviour_loss_weight
        self.behaviour_loss_threshold: float = behaviour_loss_threshold
        self.sampling_loss_weight: float = sampling_loss_weight
        self.sampling_loss_threshold: float = sampling_loss_threshold
        self.prior_entropy_loss_weight: float = prior_entropy_loss_weight
        self.posterior_entropy_loss_weight: float = posterior_entropy_loss_weight
        self.tf_loss_weight: float = tf_loss_weight
        self.retrieval_loss_weight: float = retrieval_loss_weight

        super(DLDLMConfig, self).__init__(**kwargs)  # TODO fix unwanted parameters

    @property
    def num_styles(self):
        return self.n_styles
