import os
project_path = os.path.split(os.path.split(__file__)[0])[0]
del os

# Global parameters
ETOL = 1E-7

# Three orbits:
import numpy as np
three_orbits = {'mildly_chaotic':np.array([24.9,0.,19.7,0.,0.1403069,0.]),
                'chaotic':np.array([24.9,0.,26.1,0.,0.09521978,0.]),
                'regular':np.array([24.9,0.,6.9,0.,0.19509396,0.])}
