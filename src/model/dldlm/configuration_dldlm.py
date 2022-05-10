from typing import Optional, List

from transformers import GPT2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)

DLDLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dldlm-small": "https://huggingface.co/vincenzo-scotti/dldlm-small/resolve/main/config.json",
    "dldlm-small-emp": "https://huggingface.co/vincenzo-scotti/dldlm-small-emp/resolve/main/config.json",
    "dldlm-medium": "https://huggingface.co/vincenzo-scotti/dldlm-medium/resolve/main/config.json",
    "dldlm-medium-emp": "https://huggingface.co/vincenzo-scotti/dldlm-medium-emp/resolve/main/config.json",
    # See all Discrete Latent Dialogue Language Model models at https://huggingface.co/models?filter=dldlm
}


class DLDLMConfig(GPT2Config):
    model_type = "dldlm"
    # TODO check attribute_map static attribute

    def __init__(
            self,
            n_styles: Optional[int] = None,
            n_rewards: Optional[int] = None,
            bow_size: Optional[int] = None,
            mask_token_id: Optional[int] = None,
            policy_token_id: Optional[int] = None,
            posterior_token_id: Optional[int] = None,
            latent_token_ids: Optional[List[int]] = None,
            context_type_token_id: Optional[int] = None,
            response_type_token_id: Optional[int] = None,
            latent_type_token_id: Optional[int] = None,
            gumbell_tau: float = 0.67,
            do_sample_latent: bool = False,
            rl_weight: float = 0.0,
            kl_threshold: float = 3.0,
            detach_posterior: bool = False,
            reduction: bool = True,
            lm_loss: bool = True,
            lm_loss_weight: float = 1.0,
            latent_loss: bool = True,
            latent_loss_weight: float = 1.0,
            cls_loss: bool = True,
            cls_loss_weight: float = 1.0,
            bow_loss: bool = True,
            bow_loss_weight: float = 1.0,
            rew_loss: bool = True,
            rew_loss_weight: float = 1.0,
            lm_obj: bool = False,
            latent_obj: bool = True,
            do_context: bool = True,
            do_policy: bool = True,
            do_bow: bool = True,
            do_reward: bool = False,
            do_response_encoding: bool = True,
            do_posterior: bool = True,
            do_cls: bool = False,
            do_latent: bool = True,
            do_response_decoding: bool = True,
            **kwargs,
    ):# TODO hanlde AMP here
        # Model structure hyper-parameters
        self.n_styles: Optional[int] = n_styles
        self.n_rewards: Optional[int] = n_rewards
        self.bow_size: Optional[int] = bow_size
        self.mask_token_id: Optional[int] = mask_token_id
        self.policy_token_id: Optional[int] = policy_token_id
        self.posterior_token_id: Optional[int] = posterior_token_id
        self.latent_token_ids: Optional[List[int]] = latent_token_ids
        self.context_type_token_id: Optional[int] = context_type_token_id
        self.response_type_token_id: Optional[int] = response_type_token_id
        self.latent_type_token_id: Optional[int] = latent_type_token_id
        self.ignore_id: int = -100
        # Training hyper-parameters
        assert gumbell_tau >= 0.0, f"The Gumbell-Softmax temperature must be a non-negative value in R, " \
                                   f"provided value was {gumbell_tau}"
        self.gumbell_tau: float = gumbell_tau
        self.do_sample_latent = do_sample_latent
        assert 0.0 <= rl_weight <= 1.0, f"The weight of reinforcement learning loss must be in [0, 1] in R, " \
                                        f"provided value was {rl_weight}"
        self.rl_weight: float = rl_weight
        assert kl_threshold >= 0.0, f"The KL-threshold must be a non-negative value in R, " \
                                    f"provided value was {kl_threshold}"
        self.kl_threshold: float = kl_threshold
        self.detach_posterior: bool = detach_posterior
        self.reduction: bool = reduction
        # Losses flags
        self.lm_loss: bool = lm_loss
        self.lm_loss_weight: float = lm_loss_weight
        self.latent_loss: bool = latent_loss
        self.latent_loss_weight: float = latent_loss_weight
        self.cls_loss: bool = cls_loss
        self.cls_loss_weight: float = cls_loss_weight
        self.bow_loss: bool = bow_loss
        self.bow_loss_weight: float = bow_loss_weight
        self.rew_loss: bool = rew_loss
        self.rew_loss_weight: float = rew_loss_weight
        self.lm_obj: bool = lm_obj
        self.latent_obj: bool = latent_obj
        # TODO select number of heads better and associate label names
        # Computation step flags
        self.do_context: bool = do_context
        self.do_policy: bool = do_policy
        self.do_bow: bool = do_bow
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
