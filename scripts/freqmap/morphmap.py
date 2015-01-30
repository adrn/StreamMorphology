# coding: utf-8

from __future__ import division, print_function

"""
Map out how the morphology of debris disrupted on a grid of orbits changes.
Before calling this module, you'll need to generate a grid of initial conditions
or make sure the grid you have is in the correct format. You also need to have
a text file containing the name of the potential that you used to generate the
initial conditions (the name has to be one specified in the ``potential_registry``).

For example, you might do::

    python scripts/freqmap/make_grid.py -E -0.21 --potential=triaxial-NFW \
    --ic-func=tube_grid_xz --dx=1.5 --dz=1.5

and then run this module.

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

# Project
import gary.integrate as gi
from gary.util import get_pool
from streammorphology import potential_registry
# from streammorphology.freqmap import read_allfreqs, mmap_shape

def create_ball(w0, potential, nparticles, mass):
    menc = potential.mass_enclosed(w0)
    rscale = (mass / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (mass / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[3:]**2))

    ball_w0 = np.zeros((nparticles,6))

    # Gaussian in positions
    ball_w0[:,:3] = np.random.normal(w0[:3], rscale, size=(nparticles,3))

    # Gaussian in velocity
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(nparticles,3))

    return np.vstack((w0,ball_w0[:-1]))

def worker(task):
    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    allptcl_filename = task['allptcl_filename']
    entropy_filename = task['entropy_filename']
    potential = task['potential']
    nparticles = task['nparticles']
    norbits = task['norbits']
    mass = task['mass']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    shp = (len(w0),nparticles,6)

    logger.info("Orbit {}".format(index))

    # integrate orbit for long time
    dt = 2.
    nsteps = 25000
    try:
        t,w = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                        Integrator=gi.DOPRI853Integrator)
    except RuntimeError:
        logger.warning("Failed to integrate orbit.")
        allptcl = np.memmap(allptcl_filename, mode='r+', shape=shp, dtype='float64')
        allptcl[index] = np.nan
        allptcl.flush()
        return

    # compute apocenters, pericenters
    r = np.sqrt(np.sum(w[:,0]**2, axis=-1))
    apos = [ix for ix in argrelmax(r)[0] if ix != 0 and ix != (nsteps-1)]
    pers = [ix for ix in argrelmin(r)[0] if ix != 0 and ix != (nsteps-1)]
    logger.debug("{0} apocenters, {1} pericenters".format(len(apos), len(pers)))

    if len(apos) < norbits:
        logger.warning("Not enough orbits completed to determine final apocenter.")
        return

    # find apocenter after norbits
    t1 = t[pers[0]]
    if apos[0] < pers[0]:
        t2 = t[apos[norbits]]
    else:
        t2 = t[apos[norbits-1]]

    dt = 1.
    nsteps = int((t2-t1) / dt)
    new_w0 = w[pers[0], 0]

    logger.debug("Integrating {0} particles with dt={1}, nsteps={2}".format(nparticles,dt,nsteps))

    # create a Gaussian ball of orbits around the central orbit
    ball_w0 = create_ball(new_w0, potential, nparticles, mass)

    # integrate all orbits
    try:
        t,ball_w = potential.integrate_orbit(ball_w0, dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator)
    except RuntimeError:
        return

    allptcl = np.memmap(allptcl_filename, mode='r+', shape=shp, dtype='float64')
    allptcl[index] = ball_w[-1,:]
    allptcl.flush()

    # build a KDE from the final ball particle positions
    logger.debug("Estimating density")
    final_pos = ball_w[-1,:,:3]
    kde = KernelDensity()
    kde.fit(final_pos)
    dens = kde.score_samples(final_pos)

    # compute entropy of distribution
    H,ed = np.histogram(dens, bins=35)
    S = entropy(H)

    logger.debug("Saving entropy")
    entropy_f = np.memmap(entropy_filename, mode='r+', shape=(len(w0),), dtype='float64')
    entropy_f[index] = S
    entropy_f.flush()

    return

def main(path, mass, norbits, nparticles=1000, mpi=False,
         overwrite=False, seed=42):
    np.random.seed(seed)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")

    # path to initial conditions cache
    w0_filename = os.path.join(path, 'w0.npy')
    w0 = np.load(w0_filename)
    logger.info("Number of initial conditions: {}".format(len(w0)))

    # path to mmap file to save to
    allptcl_filename = os.path.join(path, 'allptcl.dat')
    allptcl_shape = (len(w0), nparticles, 6)
    entropy_filename = os.path.join(path, 'entropy.dat')

    # path to potential name file
    pot_filename = os.path.join(path, 'potential.txt')
    with open(pot_filename) as f:
        potential_name = f.read().strip()

    # get potential from registry
    potential = potential_registry[potential_name]

    if os.path.exists(allptcl_filename) and overwrite:
        os.remove(allptcl_filename)

    if not os.path.exists(allptcl_filename):
        d = np.memmap(allptcl_filename, mode='w+', dtype='float64', shape=allptcl_shape)
        d = np.memmap(entropy_filename, mode='w+', dtype='float64', shape=(len(w0),))
        tasks = [dict(index=i, w0_filename=w0_filename, norbits=norbits, mass=mass,
                      allptcl_filename=allptcl_filename, nparticles=nparticles,
                      entropy_filename=entropy_filename,
                      potential=potential) for i in range(norbits)]

    else:
        d = np.memmap(allptcl_filename, mode='r+', dtype='float64', shape=allptcl_shape)
        not_done = np.where(np.any(d != 0., axis=1) | np.any(np.isnan(d), axis=1))
        tasks = [dict(index=i, w0_filename=w0_filename, norbits=norbits, mass=mass,
                      allptcl_filename=allptcl_filename, nparticles=nparticles,
                      entropy_filename=entropy_filename,
                      potential=potential) for i in not_done]

    pool.map(worker, tasks)
    pool.close()

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
    parser.add_argument("--path", dest="path", type=str, required=True,
                        help="Path to the freqmap initial conditions grid.")
    parser.add_argument("--nparticles", dest="nparticles", type=int, default=1000,
                        help="Number of particles per ball.")
    parser.add_argument("--norbits", dest="norbits", type=int, required=True,
                        help="Number of orbits to integrate particles for.")
    parser.add_argument("--mass", dest="mass", type=float, required=True,
                        help="'mass' scale of the progenitor")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(path=args.path, mass=args.mass, norbits=args.norbits, nparticles=args.nparticles,
         mpi=args.mpi, overwrite=args.overwrite)

    sys.exit(0)
