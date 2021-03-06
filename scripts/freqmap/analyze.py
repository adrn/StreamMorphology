# coding: utf-8

""" Analyze the output from frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.units as u
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np

# Project
from streammorphology.freqmap import read

def main(path, bounds=None, vbounds=None):

    # read in initial conditions
    w0 = np.load(os.path.join(path, 'w0.npy'))
    norbits = len(w0)

    # read freqmap output
    allfreqs_filename = os.path.join(path, 'allfreqs.dat')
    d = read(allfreqs_filename, norbits=len(w0))

    logger.info("{} total orbits".format(norbits))
    logger.info("\t{} successful".format(d['success'].sum()))
    logger.info("\t{} not successful".format((d['success'] == 0).sum()))

    ntube = d['is_tube'].sum()
    nbox = (d['is_tube'] == 0).sum()
    logger.info("\t{} tube orbits".format(ntube))
    logger.info("\t{} box orbits".format(nbox))

    frac_freq_diff = np.abs((np.abs(d['freqs'][:,1]) - np.abs(d['freqs'][:,0])) / d['freqs'][:,0])
    diffusion_time = ((d['dt']*d['nsteps'])[:,None] / frac_freq_diff * u.Myr).to(u.Gyr).value
    diffusion_time = diffusion_time.mean(axis=1)
    # diffusion_time = diffusion_time.max(axis=1)
    # diffusion_time = diffusion_time.min(axis=1)

    good_ix = np.isfinite(diffusion_time)
    c = np.log10(diffusion_time[good_ix])

    # color scaling
    if vbounds is None:
        delta = np.abs(c.max() - c.min())
        vmin = c.min() + delta/10.
        vmax = c.max() - delta/10.

    else:
        vmin,vmax = vbounds

    # initial conditions in x-z plane
    if np.all(w0[:,1] == 0.):
        # plot initial condition grid, colored by fractional diffusion rate
        fig,ax = plt.subplots(1,1,figsize=(9.75,8))

        if bounds is None:
            xbounds = ybounds = (0, max([w0[:,0].max(),w0[:,2].max()]))
        else:
            xbounds,ybounds = bounds

        ax.set_xlim(*xbounds)
        ax.set_ylim(*ybounds)

        # automatically determine symbol size
        xy_pixels = ax.transData.transform(np.vstack([w0[:,0],w0[:,2]]).T)
        xpix, ypix = xy_pixels.T

        # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
        # right for most image software, so we'll flip the y-coords
        width, height = fig.canvas.get_width_height()
        ypix = height - ypix

        # this assumes that your data-points are equally spaced
        sz = max((xpix[1]-xpix[0])**2, (ypix[1]-ypix[0])**2) + 3.

        # plot bad points
        ax.scatter(w0[~good_ix,0], w0[~good_ix,2], c='r', s=sz, marker='s')

        # plot good points, colored
        sc = ax.scatter(w0[good_ix,0], w0[good_ix,2], c=c,
                        vmin=vmin, vmax=vmax, cmap='Greys', s=sz, marker='s')

        ax.set_xlabel(r'$x_0$ $[{\rm kpc}]$')
        ax.set_ylabel(r'$z_0$ $[{\rm kpc}]$')
        fig.colorbar(sc)
        fig.tight_layout()
        fig.savefig(os.path.join(path,"diffusion_time_map.pdf"))

        # frequency map
        fig,ax = plt.subplots(1,1,figsize=(8,8))

        tube_f = np.mean(d['freqs'][d['is_tube']], axis=1)
        ax.plot(tube_f[:,1]/tube_f[:,0], tube_f[:,2]/tube_f[:,0],
                linestyle='none', marker='.', alpha=0.4)
        ax.set_xlabel(r'$\Omega_\phi/\Omega_R$')
        ax.set_ylabel(r'$\Omega_z/\Omega_R$')
        # ax.set_xlim(0.45,0.8)
        # ax.set_ylim(0.45,0.8)
        fig.savefig(os.path.join(path,"freqmap.pdf"))

    # initial conditions on equipotential surface
    else:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(9.75,8))
        ax = fig.add_subplot(111, projection='3d')

        # plot bad points
        ax.scatter(w0[~good_ix,0], w0[~good_ix,1], w0[~good_ix,2], c='r', s=8, marker='o')

        # plot good points
        sc = ax.scatter(w0[good_ix,0], w0[good_ix,1], w0[good_ix,2], c=c,
                        vmin=vmin, vmax=vmax, cmap='Greys_r', s=18, marker='o')

        ax.elev = 45
        ax.azim = 45

        fig.colorbar(sc)
        fig.tight_layout()
        fig.savefig(os.path.join(path,"diffusion_map.pdf"))

        # frequency map
        fig,ax = plt.subplots(1,1,figsize=(8,8))

        box_f = np.mean(d['freqs'][~d['is_tube']], axis=1)
        ax.plot(box_f[:,0]/box_f[:,2], box_f[:,1]/box_f[:,2],
                linestyle='none', marker='.', alpha=0.4)
        ax.set_xlabel(r'$\Omega_x/\Omega_z$')
        ax.set_ylabel(r'$\Omega_y/\Omega_z$')
        ax.set_xlim(0.55,0.9)
        ax.set_ylim(0.7,1.05)
        fig.savefig(os.path.join(path,"freqmap.pdf"))

    # plot histograms of diffusion rates
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    bins = np.linspace(c.min(), c.max(), 25)
    n,bins,pa = ax.hist(c, alpha=0.4, normed=True, bins=bins)
    ax.set_xlabel("log fractional freq. diffusion rate per orbit")
    fig.savefig(os.path.join(path,"diffusion_rate_hist.pdf"))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")

    parser.add_argument("--bounds", dest="bounds", default=None, type=str,
                        help="bounds of plot")
    parser.add_argument("--vbounds", dest="vbounds", default=None, type=str,
                        help="bounds of color scale")

    args = parser.parse_args()

    if args.bounds is not None:
        bounds = map(float, args.bounds.split(","))
        if len(bounds) > 2:
            bounds = (bounds[:2], bounds[2:])
    else:
        bounds = None

    if args.vbounds is not None:
        vbounds = map(float, args.vbounds.split(","))
    else:
        vbounds = None

    main(args.path, bounds=bounds, vbounds=vbounds)
