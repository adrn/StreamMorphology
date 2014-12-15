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
import gary.potential as gp
from gary.units import galactic
from gary.util import get_pool

from streammorphology.initialconditions import loop_grid
from streammorphology.util import worker

base_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

def main(E, loopbox, mpi=False, overwrite=False, ngrid=None):
    np.random.seed(42)
    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=0.9, c=0.7, units=galactic)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")

    path = os.path.join(base_path, 'output',
                        'E{:.3f}_{}_{}'.format(E, potential.__class__.__name__, loopbox))
    logger.info("Caching to: {}".format(path))
    allfreqs_filename = os.path.join(path, "allfreqs.dat")
    if not os.path.exists(path):
        os.mkdir(path)

    # initial conditions
    if loopbox == 'loop':
        w0 = loop_grid(E, potential, Naxis=ngrid)
    else:
        raise NotImplementedError("No support for box orbit grid!")
        w0 = box_grid(E, potential, Naxis=ngrid)

    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if os.path.exists(allfreqs_filename):
        if overwrite:
            os.remove(allfreqs_filename)
        else:
            pool.close()
            return

    allfreqs_shape = (norbits, 2, 8)
    d = np.memmap(allfreqs_filename, mode='w+', dtype='float64', shape=allfreqs_shape)

    # save the initial conditions
    w0_filename = os.path.join(path, 'w0.npy')
    np.save(w0_filename, w0)

    tasks = [dict(index=i, w0_filename=w0_filename,
                  allfreqs_filename=allfreqs_filename,
                  potential=potential) for i in range(norbits)]
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

    parser.add_argument("-E", "--energy", dest="energy", type=float, required=True,
                        help="Energy of the orbits.")
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--ngrid", dest="ngrid", type=int, default=100,
                        help="Number of grid IC's to generate along the x axis.")
    parser.add_argument("--type", dest="orbit_type", type=str, required=True,
                        help="Orbit type - can be either 'loop' or 'box'.")

    args = parser.parse_args()

    if args.type.strip() not in ['loop','box']:
        raise ValueError("'--type' argument must be one of 'loop' or 'box'")

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    all_freqs = main(E=args.E, loop=args.type.strip(),
                     mpi=args.mpi, overwrite=args.overwrite, ngrid=args.ngrid)

    # plot(all_freqs)
    sys.exit(0)
