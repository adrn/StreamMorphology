# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from streammorphology.freqvar import FreqVariance

runner = ExperimentRunner(ExperimentClass=FreqVariance)
runner.run()

sys.exit(0)
