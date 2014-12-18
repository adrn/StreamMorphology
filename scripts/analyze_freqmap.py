# coding: utf-8

""" Analyze the output from frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np

def main(path):

    w0 = np.load(os.path.join(path, 'w0.npy'))
    allfreqs_filename = os.path.join(path, 'allfreqs.dat')
    d = np.memmap(allfreqs_filename, shape=(len(w0),2,11), dtype='float64', mode='r')

    done_ix = d[:,0,7] == 1.
    logger.info("{} total orbits".format(len(w0)))
    logger.info("{} box orbits".format((d[done_ix,0,8] == 0).sum()))
    logger.info("{} loop orbits".format((d[done_ix,0,8] == 1).sum()))

    box_ix = done_ix & (d[:,0,8] == 0.) & np.all(np.isfinite(d[:,0,:3]), axis=1)
    loop_ix = done_ix & (d[:,0,8] == 1.) & np.all(np.isfinite(d[:,0,3:6]), axis=1)

    nperiods = d[box_ix,0,9]*d[box_ix,0,10] / (2*np.pi/np.abs(d[box_ix,0,:3]).max(axis=1))
    max_frac_diff = np.abs((d[box_ix,1,:3] - d[box_ix,0,:3]) / d[box_ix,0,:3]).max(axis=1)
    box_freq_diff = np.log10(max_frac_diff / nperiods / 2.)

    nperiods = d[loop_ix,0,9]*d[loop_ix,0,10] / (2*np.pi/np.abs(d[loop_ix,0,3:6]).max(axis=1))
    max_frac_diff = np.abs((d[loop_ix,1,3:6] - d[loop_ix,0,3:6]) / d[loop_ix,0,3:6]).max(axis=1)
    loop_freq_diff = np.log10(max_frac_diff / nperiods / 2.)

    # color scaling
    delta = np.abs(loop_freq_diff.max() - loop_freq_diff.min())
    vmin = loop_freq_diff.min() + delta/10.
    vmax = loop_freq_diff.max() - delta/10.

    # plot initial condition grid, colored by fractional diffusion rate
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(w0[box_ix,0], w0[box_ix,2], c=box_freq_diff,
               vmin=vmin, vmax=vmax, cmap='Greys', s=12, marker='s')
    ax.scatter(w0[loop_ix,0], w0[loop_ix,2], c=loop_freq_diff,
               vmin=vmin, vmax=vmax, cmap='Greys', s=12, marker='s')
    ax.set_xlim(-1, w0[:,0].max()+5)
    ax.set_ylim(-1, w0[:,2].max()+5)
    ax.set_xlabel(r'$x_0$ $[{\rm kpc}]$')
    ax.set_ylabel(r'$z_0$ $[{\rm kpc}]$')
    fig.savefig(os.path.join(path,"diffusion_ics.png"))

    fig,ax = plt.subplots(1,1,figsize=(9,8))
    ax.scatter(w0[box_ix,0], w0[box_ix,2], c=box_freq_diff,
               vmin=vmin, vmax=vmax, cmap='cubehelix', s=12, marker='s')
    c = ax.scatter(w0[loop_ix,0], w0[loop_ix,2], c=loop_freq_diff,
                   vmin=vmin, vmax=vmax, cmap='cubehelix', s=12, marker='s')
    ax.set_xlim(-1, w0[:,0].max()+5)
    ax.set_ylim(-1, w0[:,2].max()+5)
    ax.set_xlabel(r'$x_0$ $[{\rm kpc}]$')
    ax.set_ylabel(r'$z_0$ $[{\rm kpc}]$')
    fig.colorbar(c)
    fig.tight_layout()
    fig.savefig(os.path.join(path,"diffusion_ics_ugly.png"))

    # plot histograms of diffusion rates
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    n,bins,pa = ax.hist(loop_freq_diff, alpha=0.4, normed=True, label='loop')
    n,bins,pa = ax.hist(box_freq_diff, alpha=0.4, bins=bins, normed=True, label='box')
    ax.legend(loc='upper right')
    ax.axvline(vmin, alpha=0.1, c='k')
    ax.axvline(vmax, alpha=0.1, c='k')
    fig.savefig(os.path.join(path,"diffusion_hist.png"))

    # plot frequency maps
    loop_freqs = d[loop_ix,:,3:6].mean(axis=1)
    box_freqs = d[box_ix,:,:3].mean(axis=1)

    fig,axes = plt.subplots(1,2,figsize=(16,8))
    axes[0].plot(loop_freqs[:,1]/loop_freqs[:,0], loop_freqs[:,2]/loop_freqs[:,0],
                 linestyle='none', marker='.', alpha=0.4)
    axes[0].set_xlabel(r'$\Omega_\phi/\Omega_R$')
    axes[0].set_ylabel(r'$\Omega_z/\Omega_R$')
    axes[0].set_xlim(0.45,0.8)
    axes[0].set_ylim(0.45,0.8)

    axes[1].plot(box_freqs[:,0]/box_freqs[:,2], box_freqs[:,1]/box_freqs[:,2],
                 linestyle='none', marker='.', alpha=0.4)
    axes[1].set_xlabel(r'$\Omega_x/\Omega_z$')
    axes[1].set_ylabel(r'$\Omega_y/\Omega_z$')
    axes[1].set_xlim(0.3,1.2)
    axes[1].set_ylim(0.3,1.2)

    fig.savefig(os.path.join(path,"freqmap.png"))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")

    args = parser.parse_args()

    main(args.path)
