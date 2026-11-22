"""Keep plotting tests independent of user or dependency-provided TeX styles."""

import matplotlib
import pytest


@pytest.fixture(autouse=True)
def builtin_math_renderer():
    # Redback/Bilby imports can enable external LaTeX globally. These tests
    # exercise Matplotlib labels, not an optional system TeX installation.
    with matplotlib.rc_context({'text.usetex': False}):
        yield
