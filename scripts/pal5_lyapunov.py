# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Custom
import gary.dynamics as gd
import gary.potential as gp

from streammorphology.freqmap import estimate_dt_nsteps

def main():
    outpath = "/vega/astro/users/amp2217/projects/morphology/output/pal5"

    x0 = np.array([8.312877511, 0.242593717, 16.811943627])
    v0 = ([-52.429087, -96.697363, -8.156130]*u.km/u.s).to(u.kpc/u.Myr).value
    w0 = np.append(x0,v0)

    potential = gp.LM10Potential()

    dt,nsteps = estimate_dt_nsteps(potential, w0, nperiods=50000, nsteps_per_period=250)

    le,t,w = gd.fast_lyapunov_max(w0, potential, dt, nsteps)

    np.save(os.path.join(outpath, "le.npy"), le)
    np.save(os.path.join(outpath, "t.npy"), t)
    np.save(os.path.join(outpath, "w.npy"), w)

    plt.figure(figsize=(10,8))
    plt.loglog(t[1:-10:10], le, marker=None)
    plt.savefig(os.path.join(outpath, "le.pdf"))

if __name__ == "__main__":
    main()
