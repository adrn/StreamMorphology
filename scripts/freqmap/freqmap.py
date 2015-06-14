# coding: utf-8

from __future__ import division, print_function

"""
Frequency map a potential. Before calling this module, you'll need to generate
a grid of initial conditions or make sure the grid you have is in the correct
format. You also need to have a text file containing the name of the potential
that you used to generate the initial conditions (the name has to be one specified
in the ``potential_registry``).

For example, you might do::

    python scripts/make_grid.py -E -0.14 --potential=triaxial-NFW \
    --ic-func=tube_grid_xz --dx=5 --dz=5

and then run this module on::

    python scripts/freqmap/freqmap.py --path=output/freqmap/triaxial-NFW/E-0.140_tube_grid_xz/

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology import ExperimentRunner
from streammorphology.freqmap import Freqmap

runner = ExperimentRunner(ExperimentClass=Freqmap)

sys.exit(0)
