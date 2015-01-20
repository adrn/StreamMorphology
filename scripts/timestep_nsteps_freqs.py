# coding: utf-8

""" Check how timestep and number of steps for an orbit effects the frequencies. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import cPickle as pickle

# Third-party
import numpy as np

# Project
import gary.dynamics as gd
import gary.integrate as gi

# project
from streammorphology.potential import potential_registry

def main():
    filename = "/home/adrian/freq_dicts.pickle"
    potential = potential_registry['triaxial-NFW']

    # slow and fast diffusing orbits
    w0 = np.array([[-17.504234723,-17.2283745157,-9.07711397761,0.0721992194851,0.021293129758,0.199775306493],
                   [-1.69295332221,-13.78418595,15.6309115075,0.0580704842,0.228735516722,0.0307028904261]])

    dts = np.array([0.1, 0.25, 0.5, 1.])
    t2 = 100000
    nsteps = t2 / dts
    dts,nsteps = map(np.ravel, np.meshgrid(dts, nsteps))

    if not os.path.exists(filename):
        freq_dicts = [dict(), dict()]
        for dt,nstep in zip(dts,nsteps):
            print(dt,nstep)
            t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nstep,
                                            Integrator=gi.DOPRI853Integrator)

            for i in range(2):
                print("\t{}".format(i))
                naff = gd.NAFF(t)
                fs = gd.poincare_polar(w[:,i])
                try:
                    freqs,d,nvec = naff.find_fundamental_frequencies(fs)
                except:
                    print("Orbit {} failed for dt={}, nsteps={}".format(i,dt,nstep))
                    continue
                freq_dicts[i][(dt,nstep)] = list(freqs)

        with open(filename, "w") as f:
            pickle.dump(freq_dicts, f)

    with open(filename, "r") as f:
        freq_dicts = pickle.load(f)

    for i in range(2):
        print("-----------------------------------")
        print("Orbit {}".format(i))
        base_freqs = np.array(freq_dicts[i][(0.1,1000000)])
        for k,v in freq_dicts[i].items():
            print(k, np.array(v) - base_freqs)

if __name__ == '__main__':
    main()

