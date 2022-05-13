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

            gumbell_tau: float = 0.67,
            context_pdrop: float = 0.5,
            detach_posterior: bool = False,
            do_sample_latent: bool = False,

            unknown_token_id: Optional[int] = None,
            mask_token_id: Optional[int] = None,
            prior_token_id: Optional[int] = None,
            posterior_token_id: Optional[int] = None,
            latent_token_ids: Optional[List[int]] = None,

            reduction: bool = True,
            lm_loss: bool = True,
            latent_loss: bool = True,
            tf_loss: bool = True,
            lm_loss_weight: float = 1.0,
            latent_loss_weight: float = 1.0,
            tf_loss_weight: float = 1.0,

            do_context: bool = True,
            do_prior: bool = True,
            do_tf: bool = True,
            do_reward: bool = False,
            do_encoding: bool = True,
            do_posterior: bool = True,
            do_cls: bool = False,
            do_latent: bool = True,
            do_decoding: bool = True,
            **kwargs,
    ):# TODO hanlde AMP here
        # Model structure hyper-parameters
        self.n_styles: Optional[int] = n_styles
        self.tf_size: Optional[int] = tf_size
        # Latent analysis hyper-parameters and configs
        assert gumbell_tau >= 0.0, f"The Gumbell-Softmax temperature must be a non-negative value in R, " \
                                   f"provided value was {gumbell_tau}"
        self.gumbell_tau: float = gumbell_tau
        self.context_pdrop: float = context_pdrop
        self.detach_posterior: bool = detach_posterior
        self.do_sample_latent = do_sample_latent
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
        self.latent_loss: bool = latent_loss
        self.tf_loss: bool = tf_loss
        self.lm_loss_weight: float = lm_loss_weight
        self.latent_loss_weight: float = latent_loss_weight
        self.tf_loss_weight: float = tf_loss_weight
        # Computation step flags
        self.do_context: bool = do_context
        self.do_prior: bool = do_prior
        self.do_tf: bool = do_tf
        self.do_reward: bool = do_reward and (self.n_rewards is not None and self.n_rewards > 0)
        self.do_response_encoding: bool = do_response_encoding
        self.do_posterior: bool = do_posterior
        self.do_latent: bool = do_latent
        self.do_cls: bool = do_cls
        self.do_response_decoding: bool = do_response_decoding

        super(DLDLMConfig, self).__init__(**kwargs)  # TODO fix unwanted parameters

    @property
    def num_styles(self):
        return self.n_styles

    @property
    def num_rewards(self):
        return self.n_rewards
