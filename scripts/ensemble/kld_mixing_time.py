# coding: utf-8

""" Map the KLD mixing time for a given set of initial conditions. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np

# Project
from gary.util import get_pool

# project
from streammorphology.ensemble import worker

def main(path, memmap_shape, cache_filename, mpi=False, overwrite=False, seed=42, **kwargs):
    np.random.seed(seed)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")

    cache_path = os.path.join(path, cache_filename)

    # path to initial conditions cache
    w0_filename = os.path.join(path, 'w0.npy')
    w0 = np.load(w0_filename)

    # path to potential name file
    pot_filename = os.path.join(path, 'potential.yml')

    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if os.path.exists(cache_path) and overwrite:
        os.remove(cache_path)

    shape = (norbits,) + memmap_shape
    if not os.path.exists(cache_path):
        # make sure memmap file exists
        d = np.memmap(cache_path, mode='w+', dtype='float64', shape=shape)

    tasks = [dict(index=i, w0_filename=w0_filename,
                  cache_filename=cache_path,
                  potential_filename=pot_filename, **kwargs) for i in range(norbits)]

    pool.map(worker, tasks)
    pool.close()

    sys.exit(0)

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

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(path=args.path, mpi=args.mpi, overwrite=args.overwrite)

    sys.exit(0)
