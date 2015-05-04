# coding: utf-8

""" Make an SCF run given initial conditions, timestep, and potential parameters """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import shutil

# Third-party
from astropy.constants import G
from astropy import log as logger
import astropy.units as u

# Project
from streammorphology import project_path

base_SCFPAR = """empty           comment
empty            comment
"../../scf/SCFBI" path to SCFBI file
{nsteps:d}           nsteps
{ncen:d}             noutlog
{nsnap:d}            noutbod
{dt:.8f}    dtime
1.0             G
TRUE            self gravitating?
FALSE           input expansion coefficients?
FALSE           output expansion coefficients
FALSE           zero odd expansion coefficients
FALSE           zero even expansion coefficients
FALSE           fix zero CoM acceleration?
TRUE            orbiting in an external potential?
500             how often to re-adjust center?
{ntide:d}             number of steps to turn on MW potential
{rscale:.1e}           satellite scale (kpc)
{mass:.1e}           satellite mass (Msun)
{x[0]:.8f} {x[1]:.8f} {x[2]:.8f}
{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}
"""

base_Makefile = """scf:
        gfortran ../../scf/scf_nfw.f ../../scf/{potentialname:s}.f -o scf

clean:
        rm scf
        rm SNAP*
        rm SCFCPU SCFCEN SCFORB SCFOUT SCFLOG
"""

base_SCFPOT = """(comment) Miyamoto-Nagai disk parameters
6.5             a [kpc]
0.26            b [kpc]
0          mass scale [Msun]
(comment) Hernquist spheroid parameters
0.3             c [kpc]
0            mass scale [Msun]
(comment) Triaxial NFW halo parameters
30.             rs (scale radius) [kpc]
532.24177023   vh (scale velocity) [km/s]
1.             a (major axis)
0.8              b (intermediate axis)
0.6             c (minor axis)
0        phi (use for rotating halo) [radian]
0        theta (use for rotating halo) [radian]
0        psi (use for rotating halo) [radian]
"""

base_submit = """#!/bin/sh

# Directives
#PBS -N scf_{name}
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=8:00:00,mem=8gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
cd /vega/astro/users/amp2217/projects/morphology/simulations/runs/{name}
make
./scf
/vega/astro/users/amp2217/projects/gary/bin/moviesnap --path=/vega/astro/users/amp2217/projects/morphology/simulations/runs/{name}

date

#End of script
"""

def main(name, x, v, scfpars, potential_name, overwrite=False, submit=False):
    run_path = os.path.abspath(os.path.join(project_path, "simulations", "runs"))
    logger.debug("Run path: {}".format(run_path))

    path = os.path.join(run_path, name)
    if os.path.exists(path) and overwrite:
        logger.warning("Path '{}' already exists -- are you sure you want to overwrite?"
                       .format(path))
        yn = raw_input("[y/N]: ")
        if yn.lower().strip() == 'y':
            logger.info("nuking directory...")
            shutil.rmtree(path)
        else:
            logger.info("aborting...")
            sys.exit(0)
    elif os.path.exists(path) and not overwrite:
        raise IOError("Path '{}' already exists!".format(path))
        sys.exit(0)

    if not os.path.exists(path):
        os.makedirs(path)

    # parse pos and vel
    x_vals,x_unit = x.split()
    v_vals,v_unit = v.split()

    x = map(float, x_vals.split(',')) * u.Unit(x_unit)
    x = x.to(u.kpc).value

    v = map(float, v_vals.split(',')) * u.Unit(v_unit)
    v = v.to(u.km/u.s).value

    with open(os.path.join(path, "SCFPAR"), 'w') as f:
        f.write(base_SCFPAR.format(x=x, v=v, **scfpars))

    with open(os.path.join(path, "Makefile"), 'w') as f:
        f.write(base_Makefile.format(potentialname=potential_name))

    cmd = "unexpand {fn} > {fn}2".format(fn=os.path.join(path, "Makefile"))
    os.system(cmd)
    cmd = "mv {fn}2 {fn}".format(fn=os.path.join(path, "Makefile"))
    os.system(cmd)

    with open(os.path.join(path, "SCFPOT"), 'w') as f:
        f.write(base_SCFPOT)

    if submit:
        with open(os.path.join(path, "submit.sh"), 'w') as f:
            f.write(base_submit.format(name=name, path=path))

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
                        default=False, help="DESTROY. DESTROY.")

    parser.add_argument("--name", dest="name", type=str, help="Name.", required=True)
    parser.add_argument("--potential", dest="potential", type=str, help="Potential name.", required=True)
    parser.add_argument("-s", "--submit", action="store_true", dest="submit",
                        default=True, help="Generate a submit.sh file as well.")
    parser.add_argument("--pos", dest="x", required=True,
                        type=str, help="Initial position in 3 space. Must specify a "
                                       "unit, e.g., --x='15.324,123.314,51.134 kpc'")
    parser.add_argument("--vel", dest="v", required=True,
                        type=str, help="Initial position in 3 space. Must specify a "
                                       "unit, e.g., --v='15.324,123.314,51.134 km/s'")

    # Parameters for SCF
    parser.add_argument("--dt", dest="dt", default=1., type=float,
                        help="Time step in Myr.")
    parser.add_argument("--nsteps", dest="nsteps", type=int,
                        help="Number of steps.", required=True)
    parser.add_argument("--ncen", dest="ncen", type=int, default=10,
                        help="Output to SCFCEN every (this number) steps.")
    parser.add_argument("--nsnap", dest="nsnap", type=int, default=1000,
                        help="Output to a SNAP file every (this number) steps.")
    parser.add_argument("--ntide", dest="ntide", default=500, type=int,
                        help="Number of steps over which to turn on the tidal field.")
    parser.add_argument("--mass", dest="mass", type=float, required=True,
                        help="Mass in solar masses within 35 scale radii of the "
                             "cluster/satellite.")
    parser.add_argument("--rscale", dest="rscale", default=None, type=float,
                        help="Scale radius of the cluster/satellite.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # Contains user-specified parameters for SCF
    scfpars = dict()

    if args.rscale is None:
        ru = 0.43089*(args.mass/2.5e9)**(1/3.)
    else:
        ru = args.rscale
    scfpars['rscale'] = ru
    scfpars['mass'] = args.mass

    _G = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    X = (_G / ru**3 * args.mass)**-0.5

    length_unit = u.Unit("{0} kpc".format(ru))
    mass_unit = u.Unit("{0} M_sun".format(args.mass))
    time_unit = u.Unit("{:08f} Myr".format(X))

    scfpars['dt'] = args.dt / (1*time_unit).to(u.Myr).value
    scfpars['nsteps'] = args.nsteps
    scfpars['ncen'] = args.ncen
    scfpars['nsnap'] = args.nsnap
    scfpars['ntide'] = args.ntide

    main(name=args.name, x=args.x, v=args.v, scfpars=scfpars,
         potential_name=args.potential, overwrite=args.overwrite,
         submit=args.submit)
