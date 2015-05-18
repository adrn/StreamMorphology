# coding: utf-8

""" Evolution of frequencies for an ensemble of orbits. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.util import rolling_window, get_pool

# Project
from streammorphology import project_path
from streammorphology.freqmap import estimate_dt_nsteps
from streammorphology.ensemble import create_ball

# integration
nsteps_per_period = 512
nperiods = 100

# ensemble properties
nensemble = 100
m_scale = 1E4

# windowing properties
window_width = 25 * nsteps_per_period
window_stride = 0.5 * nsteps_per_period

def worker(task):
    dt = task['dt']
    nsteps = task['nsteps']
    w0 = task['w0']
    potential = task['potential']

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                                    Integrator=gi.DOPRI853Integrator)

    freqs = []
    for (i1,i2),ww in rolling_window(w, window_size=window_width,
                                     stride=window_stride, return_idx=True):
        if i2 >= nsteps:
            break
        logger.debug("{0}:{1}".format(i1,i2))
        fs = [(ww[:,0,i]+1j*ww[:,0,i+3]) for i in range(3)]
        naff = gd.NAFF(t[int(i1):int(i2)], p=4)
        try:
            f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)
            logger.debug("Successfully found frequencies")
        except:
            logger.debug("Failed to compute frequencies")
            f = np.ones(3)*np.nan
        freqs.append(f)

    return freqs

def main(name, mpi=False, threads=None, plot=False):
    outpath = os.path.join(project_path, "output/tests/ensemble_evolution")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load potential
    potential = gp.load(os.path.join(project_path,"potentials/triaxial-NFW.yml"))

    if name == 'regular':
        w0 = np.array([27.300000000000008, 0.0, 1.1000000000000003, 0.0, 0.1877759468430073, 0.0]) # regular
    elif name == 'chaotic':
        w0 = np.array([27.300000000000008, 0.0, 21.900000000000006, 0.0, 0.11374625738916841, 0.0]) # chaotic
    elif name == 'mildly_chaotic':
        w0 = np.array([27.300000000000008, 0.0, 20.300000000000004, 0.0, 0.12395598334095129, 0.0]) # mild chaos
    elif name == 'pal5':
        potential = gp.load(os.path.join(project_path,"potentials/lm10.yml")) # pal5 fanning
        w0 = np.array([8.312877511, 0.242593717, 16.811943627, -0.053619865077559205, -0.09889356946871429, -0.008341373370759497])
    else:
        raise ValueError()

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
            task['potential'] = potential
            tasks.append(task)

        results = pool.map(worker, tasks)
        pool.close()

        all_freqs = np.array(results)
        np.save(filename, all_freqs)

    else:
        all_freqs = np.load(filename)

    if plot:
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
    import logging

    parser = ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    # threading
    parser.add_argument("--name", dest="name", required=True, type=str,
                        help="Name: regular or chaotic")
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")

    parser.add_argument("--plot", dest="plot", default=False, action="store_true",
                        help="Show dem plots.")
    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.name, mpi=args.mpi, threads=args.threads, plot=args.plot)
