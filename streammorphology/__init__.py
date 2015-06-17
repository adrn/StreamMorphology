import os
project_path = os.path.split(os.path.split(__file__)[0])[0]
del os

# Three orbits:
import numpy as np
# three_orbits = {'mildly_chaotic':np.array([24.9,0.,19.7,0.,0.1403069,0.]),
#                 'chaotic':np.array([24.9,0.,26.1,0.,0.09521978,0.]),
#                 'regular':np.array([24.9,0.,6.9,0.,0.19509396,0.])}

three_orbits = {
    'mildly_chaotic': np.array([28.,0.,16.1,0.,0.1428514,0.]),
    'chaotic': np.array([28.,0.,22.,0.,0.10905014,0.]),
    'regular': np.array([28.,0.,8.1,0.,0.17264411,0.])
}

from experimentrunner import *
from freqmap import Freqmap
from lyapunov import Lyapmap
from freqvar import FreqVariance
