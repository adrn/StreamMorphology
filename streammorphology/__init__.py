import os
project_path = os.path.split(os.path.split(__file__)[0])[0]
del os

# Three orbits:
import numpy as np

from collections import OrderedDict
three_orbits = OrderedDict([
    ('near-resonant', np.array([17.0, 0.0, 26.060606060606062, 0.0, 0.12912205829404055, 0.0])), # resonant
    ('non-resonant', np.array([17.0, 0.0, 23.03030303030303, 0.0, 0.15198454276899373, 0.0])), # non-resonant
    ('weak-chaos', np.array([17.0, 0.0, 25.353535353535353, 0.0, 0.1346704105535305, 0.0])), # weak chaos
    ('strong-chaos', np.array([17.0, 0.0, 28.686868686868685, 0.0, 0.10691643457775891, 0.0])) # strong chaos
])

name_map = dict([
    ('near-resonant','A'),
    ('non-resonant', 'B'),
    ('weak-chaos', 'C'),
    ('strong-chaos', 'D')
])

from experimentrunner import *
from freqmap import Freqmap
from lyapunov import Lyapmap
from freqvar import FreqVariance
from ensemble import Ensemble
from ensemblefreqvar import EnsembleFreqVariance
