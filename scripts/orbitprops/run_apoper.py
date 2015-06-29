# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from streammorphology.apoper import ApoPer

runner = ExperimentRunner(ExperimentClass=ApoPer)
runner.run()

sys.exit(0)
