# coding: utf-8

""" Make Figure 9 of Sanders and Binney """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
import streamteam.integrate as si
import streamteam.io as io
from streamteam.potential.lm10 import LM10Potential
from streamteam.potential.apw import PW14Potential
import streamteam.dynamics as sd
from streamteam.units import galactic
from streamteam.util import get_pool

# Integration parameters
nsteps = 200000
dt = 0.75  # Myr

def ic_generator(w0, mmap, potential):
    n = 0
    while n < w0.shape[0]:
        yield n, w0, mmap, potential
        n += 1

def mpi_helper(p):
    n, w0, mmap, potential = p

    # Integrate single orbit
    t,w = potential.integrate_orbit(w0[n], Integrator=si.DOPRI853Integrator,
                                    dt=dt, nsteps=nsteps)
    mmap[:,n] = w[:,0]

def main(file_path, output_path=None, mpi=False, overwrite=False):
    norbits = 10

    path,filename = os.path.split(file_path)
    filename_base = os.path.splitext(filename)[0]

    if output_path is None:
        output_path = path

    pool = get_pool(mpi=mpi)

    if not os.path.exists(output_path):
        logger.debug("Creating path '{}'".format(output_path))
        os.mkdir(output_path)

    files = dict()
    files['time'] = os.path.join(output_path,"time_{}.npy".format(filename_base))
    files['orbit'] = os.path.join(output_path,"orbits_{}.array".format(filename_base))
    files['actions'] = os.path.join(output_path,"actions_{}.npy".format(filename_base))
    files['angles'] =os.path.join(output_path,"angles_{}.npy".format(filename_base))
    files['freqs'] = os.path.join(output_path,"freqs_{}.npy".format(filename_base))

    if overwrite:
        for fn in files.values():
            if os.path.exists(fn):
                logger.debug("Nuking {}".format(fn))
                os.remove(fn)

    # TODO: create a reader for NBODY6 snapshots?
    # TODO: try reading with each reader?
    # Something like -- for reader in io.readers: ...
    scf = io.SCFReader(path)
    tbl = scf.read_snap(filename, units=galactic)
    cen_w0 = io.tbl_to_w(scf.read_cen(units=galactic))[-1]
    w0 = np.squeeze(io.tbl_to_w(tbl))
    unbound = tbl['tub'] > 0.

    # TODO: below here is general
    w0 = w0[unbound]
    w0 = w0[np.random.randint(len(w0),size=norbits)]

    # stack initial conditions for center of satellite on top
    norbits += 1
    w0 = np.vstack((cen_w0, w0))

    # potential = LM10Potential()
    potential = PW14Potential(q1=1.3, q3=0.8, phi=np.pi/2., theta=np.pi/2., psi=np.pi/2.)

    logger.info("Read initial conditions...")
    if not os.path.exists(files['time']):
        logger.info("Beginning integration...")

        # create memory-mapped array to dump output to
        mmap = np.memmap(files['orbit'], mode='w+',
                         shape=(nsteps+1, norbits, 6), dtype=np.float64)

        if mpi:
            pool.map(mpi_helper, ic_generator(w0, mmap, potential))
        else:
            # Integrate orbits and save
            t,w = potential.integrate_orbit(w0, Integrator=si.DOPRI853Integrator,
                                            dt=dt, nsteps=nsteps, mmap=mmap)

        logger.info("Saving to files...")
        np.save(files['time'], t)
        w = np.memmap(files['orbit'], mode='r',
                      shape=(nsteps+1, norbits, 6), dtype=np.float64)

    else:
        logger.info("Files exist, reading orbit data...")
        t = np.load(files['time'])
        w = np.memmap(files['orbit'], mode='r',
                      shape=(nsteps+1, norbits, 6), dtype=np.float64)

    logger.info("Orbit data loaded...")

    # Make a few orbit plots
    np.random.seed(42)
    bnd = np.max(np.sqrt(np.sum(w[...,:3]**2, axis=-1)))
    for ix in np.random.randint(len(w0), size=10):
        ww = w[:,ix]
        fig = sd.plot_orbits(ww[:,None], alpha=0.01, linestyle='none', marker='.',color='k')
        for ax in fig.axes:
            ax.set_xlim(-bnd,bnd)
            ax.set_ylim(-bnd,bnd)
        fig.savefig(os.path.join(output_path, "orbit_{}_{}.png".format(ix, filename_base)))

    logger.debug("Made orbit plots")

    # Make energy conservation check plot
    plt.clf()
    for i in range(norbits//10):
        ww = w[:,i]
        E = potential.energy(ww[:,:3], ww[:,3:])
        plt.semilogy(np.abs((E[1:]-E[0])/E[0]), marker=None, alpha=0.25)

    plt.ylim(1E-16, 1E-2)
    plt.savefig(os.path.join(output_path, "energy_cons_{}.png".format(filename_base)))
    logger.debug("Made energy conservation plot")

    if not os.path.exists(files['actions']):
        logger.info("Computing actions...")

        # Compute actions, etc.
        freqs = np.empty((norbits,3))
        angles = np.empty_like(freqs)
        actions = np.empty_like(freqs)
        for i in range(norbits):
            logger.debug("Computing actions+ for orbit {}".format(i))
            ww = w[:,i]
            actions[i],angles[i],freqs[i] = sd.find_actions(t[::10], ww[::10],
                                                            N_max=6, usys=galactic)

        np.save(files['actions'], actions)
        np.save(files['angles'], angles)
        np.save(files['freqs'], freqs)
    else:
        logger.info("Reading actions/angles/freqs from files...")
        actions = np.load(files['actions'])
        angles = np.load(files['angles'])
        freqs = np.load(files['freqs'])

    print(actions)
    print(freqs)

    # Make frequency plot
    r1,r2,r3 = ((freqs[1:] - freqs[0])/freqs[0]).T
    # r1,r2,r3 = freqs.T*1000.

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True,sharex=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1, r3, alpha=0.25)
        axes[1].plot(r2, r3, alpha=0.25)

    axes[0].set_xlabel(r"$(\Omega_1 - \Omega_{1,{\rm sat}})/\Omega_{1,{\rm sat}}$")
    axes[1].set_xlabel(r"$(\Omega_2 - \Omega_{2,{\rm sat}})/\Omega_{2,{\rm sat}}$")
    axes[0].set_ylabel(r"$(\Omega_3 - \Omega_{3,{\rm sat}})/\Omega_{3,{\rm sat}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "frequencies_{}.png".format(filename_base)))

    # Make action plot
    r1,r2,r3 = ((actions[1:] - actions[0])/actions[0]).T
    # r1,r2,r3 = actions.T

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1, r3, alpha=0.25)
        axes[1].plot(r2, r3, alpha=0.25)

    axes[0].set_xlabel(r"$(J_1 - J_{1,{\rm sat}})/J_{1,{\rm sat}}$")
    axes[1].set_xlabel(r"$(J_2 - J_{2,{\rm sat}})/J_{2,{\rm sat}}$")
    axes[0].set_ylabel(r"$(J_3 - J_{3,{\rm sat}})/J_{3,{\rm sat}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "actions_{}.png".format(filename_base)))

if __name__ == "__main__":
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

    parser.add_argument("--mpi", action="store_true", dest="mpi",
                        default=False, help="Run expecting an MPI pool to map to.")

    parser.add_argument("-f", dest="filename", default=None, required=True,
                        type=str, help="Filename.")
    parser.add_argument("--output", dest="output", default=None,
                        type=str, help="Path to save files and plots to.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.filename, args.output, mpi=args.mpi, overwrite=args.overwrite)
