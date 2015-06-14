# coding: utf-8

""" Map the Lyapunov exponent for a given set of initial conditions. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from streammorphology.lyapunov import Lyapmap

runner = ExperimentRunner(ExperimentClass=Lyapmap)
runner.run()

sys.exit(0)
