from enum import Enum


class EvaluationMode(Enum):
    MAX: str = 'max'
    MIN: str = 'min'


LOSS_EVALUATION_MODE_MAPPING = {
    'lm_loss': EvaluationMode.MIN,
    'latent_kl_div_loss': EvaluationMode.MIN,
    'latent_kl_threshold_loss': EvaluationMode.MIN,
    'prior_latent_nll_loss': EvaluationMode.MIN,
    'posterior_latent_nll_loss': EvaluationMode.MIN,
    'elbo': EvaluationMode.MAX,
    'prior_dist_entropy': EvaluationMode.MAX,
    'posterior_dist_entropy': EvaluationMode.MIN,
    'tf_loss': EvaluationMode.MIN,
}
