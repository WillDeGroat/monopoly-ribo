# Imports ------------------------------------------
from __future__ import annotations

from importlib import import_module
from typing import Any

from ..exceptions import MissingOptionalDependencyError
from .base import EngineFit
# --------------------------------------------------


class JointBayesEngine:
    name: str = 'joint_latent_bayes'

    def fit(self, dataset: Any) -> EngineFit:
        try:
            import_module('arviz')
            import_module('pymc')
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                'The joint Bayesian engine requires PyMC and ArviZ. '
                'Install the Bayesian dependencies with monopoly-ribo[bayes].'
            ) from exc

        raise NotImplementedError(
            'The joint Bayesian engine is available only through selected-feature utilities. '
            'Automatic feature-wide MCMC is disabled.'
        )