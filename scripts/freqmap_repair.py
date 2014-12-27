# coding: utf-8

""" Find failed freqmap orbits and try to brute force them. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np
from astropy import log as logger

# Project
from gary.util import get_pool
from streammorphology.util import worker, read_allfreqs, potential

def main(path, mpi=False, threshold_diffusion=1E-6):
    np.random.seed(42)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")

    allfreqs_filename = os.path.join(path, "allfreqs.dat")
    w0_filename = os.path.join(path, 'w0.npy')
    w0 = np.load(w0_filename)
    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if not os.path.exists(allfreqs_filename):
        raise IOError("allfreqs file doesn't exist!")

    d = read_allfreqs(allfreqs_filename, norbits)

    # containers
    nperiods = np.zeros(norbits)
    freq_diff = np.zeros(norbits)

    # loop orbits
    loop = d[d['loop']]
    loopf = loop['fRphiz']
    nperiods[d['loop']] = loop['dt']*loop['nsteps'] / (2*np.pi/np.abs(loopf[:,0]).max(axis=-1))
    freq_diff[d['loop']] = np.abs((loopf[:,1] - loopf[:,0]) / loopf[:,0]).max(axis=1) / nperiods[d['loop']] / 2.

    # box orbits
    box = d[~d['loop']]
    boxf = box['fxyz']
    nperiods[~d['loop']] = box['dt']*box['nsteps'] / (2*np.pi/np.abs(boxf[:,0]).max(axis=-1))
    freq_diff[~d['loop']] = np.abs((boxf[:,1] - boxf[:,0]) / boxf[:,0]).max(axis=1) / nperiods[~d['loop']] / 2.

    redo_ix = (np.logical_not(d['success']) | np.logical_not(np.isfinite(nperiods)) |
               (freq_diff > threshold_diffusion) | np.logical_not(np.isfinite(freq_diff)))
    not_done = np.where(redo_ix)[0]
    tasks = [dict(index=i, w0_filename=w0_filename,
                  allfreqs_filename=allfreqs_filename,
                  potential=potential, dt=1., nsteps=400000) for i in not_done]

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

    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(path=args.path, mpi=args.mpi)

    sys.exit(0)

