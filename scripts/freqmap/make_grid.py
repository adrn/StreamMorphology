# coding: utf-8

""" Generate a grid of initial conditions for freqmap'ing """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from streammorphology import potential_registry
from streammorphology.initialconditions import loop_grid, box_grid

base_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

def main(potential_name, E, loopbox, path=None, ntotal=None, dx=None, dz=None,
         overwrite=False, plot=False):
    """ Calls the loop_ or box_grid utility functions to generate a grid of
        initial conditions for frequency mapping.
    """

    potential = potential_registry[potential_name]

    if path is None:
        path = os.path.join(base_path, 'output',
                            'E{:.3f}_{}_{}'.format(E, potential_name, loopbox))

    logger.info("Caching to: {}".format(path))
    if not os.path.exists(path):
        os.mkdir(path)

    # path to initial conditions cache
    w0path = os.path.join(path, 'w0.npy')

    if os.path.exists(w0path) and overwrite:
        os.remove(w0path)

    if not os.path.exists(w0path):

        # initial conditions
        if loopbox == 'loop':
            w0 = loop_grid(E, potential, dx=dx, dz=dz)
        else:
            w0 = box_grid(E, potential, Ntotal=ntotal)

        np.save(w0path, w0)
        logger.info("Create initial conditions file:\n\t{}".format(w0path))

    else:
        w0 = np.load(w0path)
        logger.info("Initial conditions file already exists!\n\t{}".format(w0path))

    logger.debug("Number of initial conditions: {}".format(len(w0)))

    if plot:
        pfile = os.path.join(path, 'w0.pdf')

        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot(w0[:,0], w0[:,2], linestyle='none', marker='o', alpha=0.2)

        ax.set_xlim(-1., w0[:,:3].max()+1)
        ax.set_ylim(-1., w0[:,:3].max()+1)

        ax.set_xlabel("$x$ [kpc]")
        ax.set_ylabel("$z$ [kpc]")

        fig.tight_layout()
        fig.savefig(pfile)


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
    parser.add_argument("-p", "--plot", action="store_true", dest="plot",
                        default=False, help="Plot stuff.")

    parser.add_argument("-E", "--energy", dest="energy", type=float, required=True,
                        help="Energy of the orbits.")
    parser.add_argument("--potential", dest="potential_name", type=str, required=True,
                        help="Name of the potential from the potential registry. Can be "
                        "one of: {}".format(",".join(potential_registry.keys())))

    parser.add_argument("--ntotal", dest="ntotal", type=int, default=100,
                        help="Use with box grid generator. Sets the total number of IC's.")
    parser.add_argument("--dx", dest="dx", type=float, help="Step size in x. Used for loop IC's.")
    parser.add_argument("--dz", dest="dz", type=float, help="Step size in z. Used for loop IC's.")
    parser.add_argument("--type", dest="orbit_type", type=str, required=True,
                        help="Orbit type - can be either 'loop' or 'box'.")

    args = parser.parse_args()

    if args.orbit_type.strip() not in ['loop','box']:
        raise ValueError("'--type' argument must be one of 'loop' or 'box'")

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(potential_name=args.potential_name, E=args.energy,
         loopbox=args.orbit_type.strip(), dx=args.dx, dz=args.dz, ntotal=args.ntotal,
         overwrite=args.overwrite, plot=args.plot)

    sys.exit(0)
