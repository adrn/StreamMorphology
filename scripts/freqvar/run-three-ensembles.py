# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from streammorphology.freqvar import FreqVariance

runner = ExperimentRunner(ExperimentClass=FreqVariance)
runner.run(config_filename='FreqVariance-regular.cfg')

runner = ExperimentRunner(ExperimentClass=FreqVariance)
runner.run(config_filename='FreqVariance-mildly_chaotic.cfg')

runner = ExperimentRunner(ExperimentClass=FreqVariance)
runner.run(config_filename='FreqVariance-chaotic.cfg')

sys.exit(0)
