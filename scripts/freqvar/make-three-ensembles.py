# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
import gary.potential as gp

# Project
from streammorphology import three_orbits
from streammorphology.ensemble import create_ensemble

def main(path, n=None, m_scale=None, overwrite=False, seed=None):
    potential = gp.load(os.path.join(path, "potential.yml"))

    for name in sorted(three_orbits.keys()): # enforce same order
        w0 = three_orbits[name]
        ew0 = create_ensemble(w0, potential, n=n, m_scale=m_scale)
        fn = os.path.join(path, "w0-{0}.npy".format(name))
        if not os.path.exists(fn) or (os.path.exists(fn) and overwrite):
            np.save(fn, ew0)

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")
    parser.add_argument("-p","--path", dest="path", default=None, required=True,
                        help="Path to the 'output' directory.")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Random number generator seed.")
    parser.add_argument("-n", dest="num", default=1000, type=int,
                        help="Number of orbits per ensemble")
    parser.add_argument("-m", "--mass-scale", dest="mass", default=10000., type=float,
                        help="Progenitor mass scale")

    args = parser.parse_args()

    main(path=args.path, overwrite=args.overwrite, seed=args.seed,
         n=args.num, m_scale=args.mass)
