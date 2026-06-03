"""Sphinx configuration for redback-csm."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _identity_citation_wrapper(_citation):
    def decorator(func):
        return func

    return decorator


def _install_runtime_mocks():
    """Install tiny mocks for runtime-only dependencies used during autodoc."""
    redback = types.ModuleType("redback")
    redback_utils = types.ModuleType("redback.utils")
    redback_utils.citation_wrapper = _identity_citation_wrapper
    redback_utils.calc_kcorrected_properties = MagicMock()
    redback_utils.lambda_to_nu = MagicMock()

    redback_sed = types.ModuleType("redback.sed")
    redback_sed.Blackbody = MagicMock()
    redback_sed.flux_density_to_spectrum = MagicMock()
    redback_sed.get_correct_output_format_from_spectra = MagicMock()

    redback_photosphere = types.ModuleType("redback.photosphere")
    redback_photosphere.TemperatureFloor = MagicMock()

    redback_interaction = types.ModuleType("redback.interaction_processes")
    redback_interaction.Diffusion = MagicMock()

    redback_transient_models = types.ModuleType("redback.transient_models")
    redback_supernova_models = types.ModuleType("redback.transient_models.supernova_models")
    redback_supernova_models._nickelcobalt_engine = MagicMock()

    redback.utils = redback_utils
    redback.sed = redback_sed
    redback.photosphere = redback_photosphere
    redback.interaction_processes = redback_interaction
    redback.transient_models = redback_transient_models

    astropy = types.ModuleType("astropy")
    astropy_units = types.ModuleType("astropy.units")
    astropy_units.mJy = MagicMock()

    astropy_cosmology = types.ModuleType("astropy.cosmology")
    astropy_cosmology.Planck18 = MagicMock()

    mocks = {
        "redback": redback,
        "redback.utils": redback_utils,
        "redback.sed": redback_sed,
        "redback.photosphere": redback_photosphere,
        "redback.interaction_processes": redback_interaction,
        "redback.transient_models": redback_transient_models,
        "redback.transient_models.supernova_models": redback_supernova_models,
        "astropy": astropy,
        "astropy.units": astropy_units,
        "astropy.cosmology": astropy_cosmology,
    }
    sys.modules.update(mocks)


_install_runtime_mocks()

project = "redback-csm"
author = "Nikhil Sarin, Ryosuke Hirai"
copyright = "2026, Nikhil Sarin, Ryosuke Hirai"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# The docs should build on Read the Docs without requiring the compiled Fortran
# extension or the full redback runtime stack. The public API pages document
# signatures/docstrings; running the models still requires a normal installation.
autodoc_mock_imports = [
    "bilby",
    "jax",
    "jaxlib",
    "matplotlib",
    "sncosmo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
