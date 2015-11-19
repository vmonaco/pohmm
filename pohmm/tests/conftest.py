from __future__ import print_function
import numpy as np
import pytest


def pytest_runtest_setup(item):
    seed = np.random.randint(1000)
    print("Seed used in np.random.seed(): %d" % seed)
    np.random.seed(seed)


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store', default=False, help='Run slow tests')


@pytest.fixture
def slow(request):
    try:
        return request.config.getoption("--runslow") in "True,true,yes,1".split(",")
    except ValueError:
        return False
