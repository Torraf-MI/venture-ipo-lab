import warnings

import pytest


@pytest.fixture(autouse=True)
def _silenzia_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
