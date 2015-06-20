import os
project_path = os.path.split(os.path.split(__file__)[0])[0]
del os

# Three orbits:
import numpy as np
# three_orbits = {'mildly_chaotic':np.array([24.9,0.,19.7,0.,0.1403069,0.]),
#                 'chaotic':np.array([24.9,0.,26.1,0.,0.09521978,0.]),
#                 'regular':np.array([24.9,0.,6.9,0.,0.19509396,0.])}

three_orbits = {
    'mildly_chaotic': np.array([29.1,0.,17.7,0.,0.12833595,0.]),
    'chaotic': np.array([29.1,0.,22.1,0.,0.10171223,0.]),
    'regular': np.array([29.1,0.,5.1,0.,0.1718898,0.])
}

from experimentrunner import *
from freqmap import Freqmap
from lyapunov import Lyapmap
from freqvar import FreqVariance
from ensemble import Ensemble
