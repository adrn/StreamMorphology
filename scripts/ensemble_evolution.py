# coding: utf-8

""" Evolution of frequencies for an ensemble of orbits. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.util import rolling_window, get_pool

# Project
from streammorphology.freqmap import estimate_dt_nsteps
from streammorphology.ensemble import create_ball

# integration
nsteps_per_period = 512
nperiods = 100

# ensemble properties
nensemble = 100
m_scale = 1E4

# windowing properties
window_width = 50 * nsteps_per_period
window_stride = 1 * nsteps_per_period

# load potential
potential = gp.load("/Users/adrian/projects/morphology/potentials/triaxial-NFW.yml")

def worker(task):
    dt = task['dt']
    nsteps = task['nsteps']
    w0 = task['w0']

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                                    Integrator=gi.DOPRI853Integrator)

    freqs = []
    for (i1,i2),ww in rolling_window(w, window_size=window_width,
                                     stride=window_stride, return_idx=True):
        if i2 >= nsteps:
            break

        fs = [(ww[:,0,i]+1j*ww[:,0,i+3]) for i in range(3)]
        naff = gd.NAFF(t[i1:i2], p=4)
        f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)
        freqs.append(f)

    return freqs

def main(mpi=False, threads=None):
    outpath = "output/tests/ensemble_evolution"
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # initial conditions of parent orbit -- picked from threshold time
    w0 = np.array([24.700000000000006, 0.0, 18.100000000000005, 0.0, 0.15050763602808698, 0.0]) # regular?
    name = 'regular'
    # w0 = np.array([24.700000000000006, 0.0, 21.300000000000004, 0.0, 0.1314325043923756, 0.0]) # chaotic?
    # name = 'chaotic'

    filename = os.path.join(outpath, '{0}_window_freqs.npy'.format(name))
    if not os.path.exists(filename):

        # get a pool object for multiprocessing / MPI
        pool = get_pool(mpi=mpi, threads=threads)

        # get integration parameters
        dt,nsteps,T = estimate_dt_nsteps(potential, w0,
                                         nperiods=nperiods,
                                         nsteps_per_period=nsteps_per_period,
                                         return_periods=True)

        # create an ensemble around the parent orbit
        ball_w0 = create_ball(w0, potential, N=nensemble, m_scale=m_scale)
        ball_w0[:,3:] = w0[None,3:]

        tasks = []
        for n,w0 in enumerate(ball_w0):
            task = dict()
            task['dt'] = dt
            task['nsteps'] = nsteps
            task['w0'] = w0
            tasks.append(task)

        results = pool.map(worker, tasks)
        pool.close()

        all_freqs = np.array(results)
        np.save(filename, all_freqs)

    else:
        all_freqs = np.load(filename)

    plt.figure(figsize=(10,8))
    plt.plot(all_freqs.T[0]*1000., c='k', alpha=0.4)

    plt.figure(figsize=(10,8))

    ff = np.median(all_freqs[:,0], axis=0)
    root_var = np.sqrt(np.sum(np.var(all_freqs, axis=1)/ff**2, axis=-1))
    plt.plot(root_var, c='k', alpha=1.)
    plt.ylim(0.1,1)
    plt.ylabel(r'$\sigma_\Omega/\Omega$')

    plt.figure(figsize=(10,10))
    r1 = all_freqs[...,0] / all_freqs[...,2]
    r2 = all_freqs[...,1] / all_freqs[...,2]
    plt.plot(r1, r2, linestyle='none', marker='.', c='k', alpha=0.5)
    plt.plot(r1[:,0], r2[:,0], linestyle='none', marker='o', c='g', alpha=0.5)
    plt.plot(r1[:,-1], r2[:,-1], linestyle='none', marker='o', c='r', alpha=0.5)
    plt.xlabel(r'$\Omega_1/\Omega_3$')
    plt.ylabel(r'$\Omega_2/\Omega_3$')

    plt.figure(figsize=(10,10))
    r1 = all_freqs[...,0]*1000
    r2 = all_freqs[...,1]*1000
    plt.plot(r1, r2, linestyle='none', marker='.', c='k', alpha=0.5)
    plt.plot(r1[:,0], r2[:,0], linestyle='none', marker='o', c='g', alpha=0.5)
    plt.plot(r1[:,-1], r2[:,-1], linestyle='none', marker='o', c='r', alpha=0.5)
    plt.xlabel(r'$\Omega_1$ [Gyr$^{-1}$]')
    plt.ylabel(r'$\Omega_2$ [Gyr$^{-1}$]')

    plt.figure(figsize=(10,10))
    r1 = all_freqs[...,0]*1000
    r2 = all_freqs[...,2]*1000
    plt.plot(r1, r2, linestyle='none', marker='.', c='k', alpha=0.5)
    plt.plot(r1[:,0], r2[:,0], linestyle='none', marker='o', c='g', alpha=0.5)
    plt.plot(r1[:,-1], r2[:,-1], linestyle='none', marker='o', c='r', alpha=0.5)
    plt.xlabel(r'$\Omega_1$ [Gyr$^{-1}$]')
    plt.ylabel(r'$\Omega_3$ [Gyr$^{-1}$]')

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")
    args = parser.parse_args()

    main(mpi=args.mpi, threads=args.threads)
