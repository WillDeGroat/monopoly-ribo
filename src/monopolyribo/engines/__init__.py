from .beta_binomial import BetaBinomialEngine
from .dirichlet_multinomial import DirichletMultinomialEngine
from .joint_bayes import JointBayesEngine
from .joint_latent import JointLatentEngine
from .nb_interaction import NBInteractionEngine


ENGINE_CLASSES = {
    'beta_binomial': BetaBinomialEngine,
    'dirichlet_multinomial': DirichletMultinomialEngine,
    'joint_latent': JointLatentEngine,
    'joint_latent_bayes': JointBayesEngine,
    'joint_latent_mle': JointLatentEngine,
    'nb_interaction': NBInteractionEngine
}


__all__ = [
    'BetaBinomialEngine',
    'DirichletMultinomialEngine',
    'ENGINE_CLASSES',
    'JointBayesEngine',
    'JointLatentEngine',
    'NBInteractionEngine'
]