"""
prior_provider.py — Callable registered under the redback.model.priors entry point.

redback calls get_prior(model_name) when building priors for any model registered
by this plugin. Returns a bilby PriorDict or None if the model is not known here.
"""

import os
from bilby.core.prior import PriorDict

PRIOR_DIR = os.path.join(os.path.dirname(__file__), "priors")


def get_prior(model_name: str):
    """
    Return a bilby PriorDict for the given model name, or None if not found.

    Used by redback.priors.get_priors() via the redback.model.priors entry point.
    """
    path = os.path.join(PRIOR_DIR, f"{model_name}.prior")
    if not os.path.exists(path):
        return None
    priors = PriorDict()
    priors.from_file(path)
    return priors
