import os
import sys
import pytest

from ..enbios.processing.main import enviro_musiasem


def test_example_1():
    enviro_musiasem("../example_config.yaml")
