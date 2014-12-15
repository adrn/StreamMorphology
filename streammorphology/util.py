# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np

# Project
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['ws_to_freqs', 'worker']

def ws_to_freqs(naff, ws, nintvec=15):

    # first get x,y,z frequencies
    logger.info('Solving for XYZ frequencies...')
    fs = [(ws[:,0,j] + 1j*ws[:,0,j+3]) for j in range(3)]
    try:
        fxyz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
    except:
        fxyz = np.ones(3)*np.nan

    # now get other frequencies
    loop = gd.classify_orbit(ws)
    if np.any(loop):
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.flip_coords(ws, loop[0])

        fs = gd.poincare_polar(new_ws[:,0])
        try:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fRphiz = np.ones(3)*np.nan

    else:
        fRphiz = np.ones(3)*np.nan

    return np.append(fxyz, fRphiz)

def worker(task):
    # unpack input argument dictionary
    i = task['index']
    w0_filename = task['w0_filename']
    allfreqs_filename = task['allfreqs_filename']
    potential = task['potential']
    dt = task['dt']
    nsteps = task['nsteps']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    allfreqs_shape = (len(w0), 2, 8)  # 6 frequencies, max energy diff, done flag
    allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')

    # short-circuit if this orbit is already done
    if allfreqs[i,0,7] == 1.:
        return

    dEmax = 1.
    maxiter = 5  # maximum number of times to refine integration step
    for i in range(maxiter):
        if i > 0:
            # adjust timestep and duration if necessary
            dt /= 2.
            nsteps *= 2

        # integrate orbit
        t,ws = potential.integrate_orbit(w0[i].copy(), dt=dt, nsteps=nsteps,
                                         Integrator=gi.DOPRI853Integrator)
        logger.debug('Orbit integrated')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max()

        if dEmax < 1E-9:
            break

    # start finding the frequencies -- do first half then second half
    naff = gd.NAFF(t[:nsteps//2+1])
    freqs1 = ws_to_freqs(naff, ws[:nsteps//2+1])
    freqs2 = ws_to_freqs(naff, ws[nsteps//2:])

    # save to output array
    allfreqs[i,0,:6] = freqs1
    allfreqs[i,1,:6] = freqs2

    allfreqs[i,:,6] = dEmax
    allfreqs[i,:,7] = 1.
