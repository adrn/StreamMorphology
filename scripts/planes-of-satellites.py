# coding: utf-8

""" Generate a grid of initial conditions for freqmap'ing """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.units as u
import numpy as np
from astropy import log as logger

# Project
import gary.potential as gp
from streammorphology import project_path
import streammorphology.initialconditions as ic

submit_base = """
#!/bin/sh

# Directives
#PBS -N plane{ix}_freqmap
#PBS -W group_list=yetiastro
#PBS -l nodes=4:ppn=16,walltime=16:00:00,mem=18gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
/usr/local/openmpi-1.6.5/bin/mpiexec -n 64 /vega/astro/users/amp2217/anaconda/bin/python /vega/astro/users/amp2217/projects/morphology/scripts/freqmap/freqmap.py --mpi --path=/vega/astro/users/amp2217/projects/morphology/output/freqmap/planes-of-satellites/{endpath} -o

date

#End of script
"""

def main(overwrite=False):
    ##########################################################################
    nplanes = 9
    E = -0.21
    ##########################################################################

    # create top-level path
    path = os.path.join(project_path, 'output', 'freqmap', 'planes-of-satellites')

    for i,phi in enumerate(-np.linspace(0.,90.,nplanes)*u.deg):
        potential = gp.TriaxialMWPotential(halo=dict(phi=phi))
        w0 = ic.tube_grid_xz(E, potential, dx=1., dz=1.)

        endpath = 'plane{0}_phi{1}'.format(i,int(phi.value))
        this_path = os.path.join(path, endpath)
        logger.info("Caching plane {0} to: {1}".format(i, this_path))

        if not os.path.exists(this_path):
            os.makedirs(this_path)

        # path to initial conditions cache
        w0path = os.path.join(this_path, 'w0.npy')
        pot_path = os.path.join(this_path, 'potential.yml')

        if os.path.exists(w0path) and overwrite:
            os.remove(w0path)

        # initial conditions
        np.save(w0path, w0)
        logger.info("Created initial conditions file:\n\t{}".format(w0path))

        # save potential
        potential.save(pot_path)

        logger.info("Number of initial conditions: {}".format(len(w0)))

        # now make a submit file for freqmapping
        submit_text = submit_base.format(ix=i, endpath=endpath)
        with open(os.path.join(this_path, "submit.sh"), "w") as f:
            f.write(submit_text)

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

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.overwrite)

    sys.exit(0)
