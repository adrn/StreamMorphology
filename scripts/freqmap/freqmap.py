# coding: utf-8

from __future__ import division, print_function

"""
Frequency map a potential. Before calling this module, you'll need to generate
a grid of initial conditions or make sure the grid you have is in the correct
format. You also need to have a text file containing the name of the potential
that you used to generate the initial conditions (the name has to be one specified
in the ``potential_registry``).

For example, you might do::

    python scripts/freqmap/make_grid.py -E -0.21 --potential=triaxial-NFW \
    --ic-func=tube_grid_xz --dx=1.5 --dz=1.5

and then run this module on::

    python scripts/freqmap/freqmap.py --path=/path/to/triaxial-NFW/E-0.21_tube_grid_xz/

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np

# Project
from gary.util import get_pool
from streammorphology import potential_registry
from streammorphology.freqmap import worker, read_allfreqs, mmap_shape

def main(path, mpi=False, overwrite=False, seed=42):
    np.random.seed(seed)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")

    allfreqs_filename = os.path.join(path, "allfreqs.dat")

    # path to initial conditions cache
    w0_filename = os.path.join(path, 'w0.npy')
    w0 = np.load(w0_filename)

    # path to potential name file
    pot_filename = os.path.join(path, 'potential.txt')
    with open(pot_filename) as f:
        potential_name = f.read().strip()

    # get potential from registry
    potential = potential_registry[potential_name]

    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if os.path.exists(allfreqs_filename) and overwrite:
        os.remove(allfreqs_filename)

    allfreqs_shape = (norbits,) + mmap_shape
    if not os.path.exists(allfreqs_filename):
        d = np.memmap(allfreqs_filename, mode='w+', dtype='float64', shape=allfreqs_shape)
        tasks = [dict(index=i, w0_filename=w0_filename,
                      allfreqs_filename=allfreqs_filename,
                      potential=potential) for i in range(norbits)]

    else:
        d = read_allfreqs(allfreqs_filename, norbits)
        not_done = np.where(~d['success'])[0]
        tasks = [dict(index=i, w0_filename=w0_filename,
                      allfreqs_filename=allfreqs_filename,
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
