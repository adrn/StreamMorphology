# coding: utf-8

""" Generate a grid of initial conditions for freqmap'ing """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import gary.potential as gp
import matplotlib.pyplot as plt
import numpy as np

# Project
from streammorphology import project_path
import streammorphology.initialconditions as ic

def main(potential_name, E, ic_func, output_path=None, overwrite=False, plot=False, **kwargs):
    """ Calls one of the grid-making utility functions to generate a
        grid of initial conditions for frequency mapping, and saves the
        grid to a file.
    """

    # read the potential from the registry
    potential_path = os.path.join(project_path, "potentials/{0}.yml".format(potential_name))
    potential = gp.load(potential_path)

    # create path
    if output_path is None:
        output_path = os.path.join(project_path, "output")
    path = os.path.join(output_path, 'freqmap', potential_name,
                        'E{:.3f}_{}'.format(E, ic_func.func_name))

    logger.info("Caching to: {}".format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    # path to initial conditions cache
    w0path = os.path.join(path, 'w0.npy')
    pot_path = os.path.join(path, 'potential.yml')

    if os.path.exists(w0path) and overwrite:
        os.remove(w0path)

    if not os.path.exists(w0path):
        # initial conditions
        w0 = ic_func(E=E, potential=potential, **kwargs)
        np.save(w0path, w0)
        logger.info("Create initial conditions file:\n\t{}".format(w0path))

        # save potential
        potential.save(pot_path)

    else:
        w0 = np.load(w0path)
        logger.info("Initial conditions file already exists!\n\t{}".format(w0path))

    if plot:
        fig,ax = plt.subplots(1, 1, figsize=(8,8))
        ax.plot(w0[:,0], w0[:,2], marker='.', linestyle='none')
        fig.savefig(os.path.join(path, 'w0.png'))

    logger.info("Number of initial conditions: {}".format(len(w0)))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging
    import inspect

    # list of possible potentials
    all_potentials = [x.rstrip('.yml') for x in os.listdir(os.path.join(project_path, 'potentials'))]

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")

    parser.add_argument("--plot", action="store_true", dest="plot",
                        default=False, help="Plot dat ish.")
    parser.add_argument("-p","--output-path", dest="output_path", default=None,
                        help="Path to the 'output' directory.")

    parser.add_argument("-E", "--energy", dest="energy", type=float, required=True,
                        help="Energy of the orbits.")
    parser.add_argument("--potential", dest="potential_name", type=str, required=True,
                        help="Name of the potential from the potential registry. Can be "
                        "one of: {}".format(",".join(all_potentials)))
    parser.add_argument("--ic-func", dest="ic_func", type=str, required=True,
                        help="Name of the initial condition function to use. Can be "
                        "one of: {}".format(",".join([f for f in dir(ic) if 'grid' in f])))

    # automagically add arguments for different initial condition grid functions
    for fn_name in dir(ic):
        if 'grid' not in fn_name:
            continue

        argspec = inspect.getargspec(getattr(ic,fn_name))
        if argspec.defaults is not None:
            pad = len(argspec.args) - len(argspec.defaults)
            defaults = [None]*pad + list(argspec.defaults)
        else:
            defaults = [None]*len(argspec.args)

        for arg,default in zip(argspec.args,defaults):
            if arg in ['E','potential']:
                continue

            if isinstance(default, list):
                parser.add_argument("--{}".format(arg), dest=arg, type=float, nargs='+',
                                    help="[float] Used in initial condition function: {0}".format(fn_name))
            else:
                typ = type(default).__name__
                helpstr = "[{0}] Used in initial condition function: {1}".format(typ, fn_name)
                parser.add_argument("--{0}".format(arg), dest=arg,
                                    type=type(default), help=helpstr)

    args = parser.parse_args()

    # now actually pull out the relevant arguments for the initial condition function
    argspec = inspect.getargspec(getattr(ic,args.ic_func))
    arg_dict = dict()
    for arg in argspec.args:
        if arg in ['E','potential']:
            continue
        arg_dict[arg] = getattr(args, arg)

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(potential_name=args.potential_name,
         E=args.energy,
         ic_func=getattr(ic,args.ic_func),
         overwrite=args.overwrite,
         output_path=args.output_path,
         plot=args.plot,
         **arg_dict)

    sys.exit(0)
