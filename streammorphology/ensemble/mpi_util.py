# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import gary.potential as gp
import gary.integrate as gi
import numpy as np
import scipy.optimize as so
from scipy.signal import argrelmin
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

# Project
from .. import ETOL
from .core import create_ball
from ..freqmap import estimate_max_period

__all__ = ['worker']

def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    filename = task['cache_filename']
    potential = gp.load(task['potential_filename'])
    nensemble = task['nensemble']

    # mass scale
    mscale = task.get('mscale', 1E4)

    # KDE kernel bandwidth
    bw = task.get('kde_bandwidth', 10.)

    # number of times to compute the KLD
    nkld = task.get('nkld', 256)

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)
    nsteps_per_period = task.get('nsteps_per_period', 256)

    # if these aren't set, assume defaults
    nperiods = 500

    # read out just this initial condition
    w0 = np.load(w0_filename)
    this_w0 = w0[index].copy()
    shape = (len(w0),2)
    all_kld = np.memmap(filename, mode='r+', shape=shape, dtype='float64')

    # short-circuit if this orbit is already done
    if all_kld[index,1] == 1.:
        return

    # TODO: handle == 0 (not attempted) different from >= 2 (previous failure)

    # identify first pericenter and estimate dt, nsteps needed for 500 periods
    t,w = potential.integrate_orbit(this_w0, dt=1., nsteps=20000)
    R = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    T_max,T_min = estimate_max_period(t,w,min=True)
    peri_ix = argrelmin(R, mode='wrap')[0]

    # timestep from number of steps per period
    dt = float(T_min) / nsteps_per_period
    nsteps = int(round(nperiods * nsteps_per_period, -4))

    # update w0 so ensemble starts at pericenter
    this_w0 = w[peri_ix[0],0]

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    ball_w0 = create_ball(this_w0, potential, N=nensemble, m_scale=mscale)
    logger.debug("Generated ensemble of {0} particles".format(nensemble))

    # manually integrate and don't keep all timesteps so we don't use an
    #   infinite amount of energy
    acc = lambda t,w: np.hstack((w[...,3:],potential.acceleration(w[...,:3])))
    integrator = gi.DOPRI853Integrator(acc, nsteps=4096, atol=1E-8)

    # define a function to compute expected density for uniform on energy hypersurface
    E0 = float(np.squeeze(potential.total_energy(this_w0[:3],this_w0[3:])))
    predicted_density = lambda x: np.sqrt(E0 - potential(x))

    # integration and KDE
    time = 0.
    w_i = ball_w0

    # start after one period
    KLD_ixes = np.linspace(nsteps_per_period, nsteps, nkld).astype(int)
    KLD = []
    KLD_times = []
    for ii in range(nsteps):
        t,w_ip1 = integrator.run(w_i, t1=time, dt=dt, nsteps=1)
        w_i = w_ip1[-1]
        time += dt

        if ii in KLD_ixes:
            logger.debug("Computing KLD at index {0}".format(ii))
            kde = KernelDensity(kernel='epanechnikov', bandwidth=bw)
            kde.fit(w_i[:,:3])
            kde_densy = np.exp(kde.score_samples(w_i[:,:3]))

            p_densy = predicted_density(w_i[:,:3])
            D = np.log(kde_densy / p_densy)
            KLD.append(D[np.isfinite(D)].sum())
            KLD_times.append(time)

    KLD = np.array(KLD)
    KLD_times = np.array(KLD_times)

    # TODO: compare final E vs. initial E against ETOL

    # fit a power law to the KLDs
    model = lambda p,t: p[0] * t**p[1]
    errfunc = lambda p,t,y: (y - model(p,t))
    p_opt,ier = so.leastsq(errfunc, x0=(0.01,-0.15), args=(KLD_times,KLD))
    if ier not in [1,2,3,4]:
        all_kld[index,0] = np.nan
        all_kld[index,1] = 2.  # failed due to leastsq

    # power-law index
    k = p_opt[1]

    all_kld[index,0] = k
    all_kld.flush()

