"""Shared pytest fixtures for ggplot2_py."""

import pytest
import pandas as pd
from ggplot2_py.datasets import load_dataset


@pytest.fixture
def mpg():
    return load_dataset("mpg")


@pytest.fixture
def diamonds():
    return load_dataset("diamonds")


@pytest.fixture
def faithfuld():
    return load_dataset("faithfuld")


@pytest.fixture
def economics():
    return load_dataset("economics")
