# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
import gary.dynamics as gd
import gary.potential as gp
import gary.integrate as gi
from gary.units import galactic
from gary.util import get_pool
from streammorphology.initialconditions import loop_grid

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

def main(path="", mpi=False, overwrite=False, dt=None, nsteps=None, ngrid=None):
    np.random.seed(42)
    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=0.9, c=0.7, units=galactic)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")
    logger.info("Caching to: {}".format(path))
    allfreqs_filename = os.path.join(path, "allfreqs.dat")
    if not os.path.exists(path):
        os.mkdir(path)

    # initial conditions
    E = -0.1
    w0 = loop_grid(E, potential, Naxis=ngrid)
    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if os.path.exists(allfreqs_filename):
        if overwrite:
            os.remove(allfreqs_filename)
        else:
            return

    allfreqs_shape = (norbits, 2, 8)
    d = np.memmap(allfreqs_filename, mode='w+', dtype='float64', shape=allfreqs_shape)

    # save the initial conditions
    w0_filename = os.path.join(path, 'w0.npy')
    np.save(w0_filename, w0)

    tasks = [dict(index=i, w0_filename=w0_filename,
                  allfreqs_filename=allfreqs_filename,
                  potential=potential,
                  dt=dt,
                  nsteps=nsteps) for i in range(norbits)]
    pool.map(worker, tasks)

    pool.close()

# def diffusion_rates(freqs):
#     Econs = freqs[-1,0]
#     freq_diff = np.abs((freqs[:-1,0] - freqs[:-1,1]) / freqs[:-1,0])

#     fig,axes = plt.subplots(1, 2, figsize=(14,6))
#     axes[0].scatter()
#     axes[1].scatter(freqs[4,0]/freqs[3,0], freqs[2,0]/freqs[3,0],
#                     marker='.', alpha=0.1)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")

    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--path", dest="path", default='', help="Cache path.")
    parser.add_argument("--dt", dest="dt", type=float, default=3.,
                        help="Base orbit timestep.")
    parser.add_argument("--nsteps", dest="nsteps", type=int, default=100000,
                        help="Base number of orbit steps.")
    parser.add_argument("--ngrid", dest="ngrid", type=int, default=100,
                        help="Number of grid IC's to generate along the x axis.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    all_freqs = main(path=os.path.abspath(args.path), mpi=args.mpi,
                     overwrite=args.overwrite, dt=args.dt, nsteps=args.nsteps,
                     ngrid=args.ngrid)

    # plot(all_freqs)
    sys.exit(0)
