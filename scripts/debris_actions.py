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

def main(file_path, output_path=None):
    norbits = 1000
    nsteps = 100000

    path,filename = os.path.split(file_path)
    filename_base = os.path.splitext(filename)[0]

    if output_path is None:
        output_path = path

    if not os.path.exists(output_path):
        logger.debug("Creating path '{}'".format(output_path))
        os.mkdir(output_path)

    time_file = os.path.join(output_path,"time_{}.npy".format(filename_base))
    orbit_file = os.path.join(output_path,"orbits_{}.array".format(filename_base))
    action_file = os.path.join(output_path,"actions_{}.npy".format(filename_base))
    angle_file =os.path.join(output_path,"angles_{}.npy".format(filename_base))
    freq_file = os.path.join(output_path,"freqs_{}.npy".format(filename_base))

    # TODO: create a reader for NBODY6 snapshots?
    # TODO: try reading with each reader?
    # Something like -- for reader in io.readers: ...
    scf = io.SCFReader(path)
    tbl = scf.read_snap(filename, units=galactic)
    w0 = io.tbl_to_w(tbl)
    unbound = tbl['tub'] > 0.

    # TODO: below here is general
    w0 = w0[unbound]
    w0 = w0[np.random.randint(len(w0),size=norbits)]

    # potential = LM10Potential()
    potential = PW14Potential(q1=1.3, q3=0.8, phi=np.pi/2., theta=np.pi/2., psi=np.pi/2.)

    logger.info("Read initial conditions...")
    if not os.path.exists(time_file):
        logger.info("Beginning integration...")

        # create memory-mapped array to dump output to
        mmap = np.memmap(orbit_file, mode='w+',
                         shape=(nsteps+1, norbits, 6), dtype=np.float64)

        # Integrate orbits and save
        t,w = potential.integrate_orbit(w0, Integrator=si.DOPRI853Integrator,
                                        dt=0.4, nsteps=nsteps, mmap=mmap)

        logger.info("Saving to files...")
        np.save(time_file, t)
        w = np.memmap(orbit_file, mode='r',
                      shape=(nsteps+1, norbits, 6), dtype=np.float64)

    else:
        logger.info("Files exist, reading orbit data...")
        t = np.load(time_file)
        w = np.memmap(orbit_file, mode='r',
                      shape=(nsteps+1, norbits, 6), dtype=np.float64)

    logger.info("Orbit data loaded...")

    # Make a few orbit plots
    for ix in np.random.randint(len(w0), size=10):
        ww = w[:,ix]
        fig = sd.plot_orbits(ww[:,None], alpha=0.01, linestyle='none', marker='.',color='k')
        fig.savefig(os.path.join(output_path, "orbit_{}_{}.png".format(ix, filename_base)))

    logger.debug("Made orbit plots")

    # Make energy conservation check plot
    plt.clf()
    for i in range(100):
        ww = w[:,i]
        E = potential.energy(ww[:,:3], ww[:,3:])
        plt.semilogy(np.abs(E[1:]-E[0])/E[0], marker=None, alpha=0.25)

    plt.ylim(1E-16, 1E-2)
    plt.savefig(os.path.join(output_path, "energy_cons_{}.png".format(filename_base)))
    logger.debug("Made energy conservation plot")

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

    np.save(action_file, actions)
    np.save(angle_file, angles)
    np.save(freq_file, freqs)

    # Make frequency plot
    r1,r2,r3 = np.array([freqs[:,0]-np.mean(freqs[:,0]),
                         freqs[:,1]-np.mean(freqs[:,1]),
                         freqs[:,2]-np.mean(freqs[:,2])])*1000.

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True,sharex=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1[::2], r3[::2], alpha=0.25)
        axes[1].plot(r2[::2], r3[::2], alpha=0.25)

    axes[0].set_xlim(-1.,1.)
    axes[0].set_ylim(-1.,1.)

    axes[0].set_xlabel(r"$\Omega_1-\langle\Omega_1\rangle$ [Gyr$^{-1}$]")
    axes[0].set_ylabel(r"$\Omega_3-\langle\Omega_3\rangle$ [Gyr$^{-1}$]")
    axes[1].set_xlabel(r"$\Omega_2-\langle\Omega_2\rangle$ [Gyr$^{-1}$]")
    fig.savefig(os.path.join(output_path, "frequencies_{}.png".format(filename_base)))

    # Make action plot
    r1,r2,r3 = np.array([(actions[:,0]-np.mean(actions[:,0]))/np.mean(actions[:,0]),
                         (actions[:,1]-np.mean(actions[:,1]))/np.mean(actions[:,1]),
                         (actions[:,2]-np.mean(actions[:,2]))/np.mean(actions[:,2])])

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(r1[::2], r3[::2], alpha=0.25)
        axes[1].plot(r2[::2], r3[::2], alpha=0.25)

    axes[0].set_xlabel(r"$(J_1-\langle J_1\rangle)/\langle J_1\rangle$")
    axes[0].set_ylabel(r"$(J_3-\langle J_3\rangle)/\langle J_3\rangle$")
    axes[1].set_xlabel(r"$(J_2-\langle J_2\rangle)/\langle J_2\rangle$")
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

    main(args.filename, args.output)
