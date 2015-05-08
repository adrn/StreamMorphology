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

# Project
from streammorphology.freqvar import read

def main(path, bounds=None, vbounds=None):

    # read in initial conditions
    w0 = np.load(os.path.join(path, 'w0.npy'))
    norbits = len(w0)

    # read freqmap output
    allfreqs_filename = os.path.join(path, 'allfreqvar.dat')
    d = read(allfreqs_filename, norbits=len(w0))

    logger.info("{} total orbits".format(norbits))
    logger.info("\t{} successful".format(d['success'].sum()))
    logger.info("\t{} not successful".format((d['success'] == 0).sum()))

    periods = np.abs(2*np.pi / d['freqs'])
    nperiods = d['dt']*d['nsteps'] / periods.max(axis=1)
    good_ix = d['success']

    ix = d['max_amp_freq_ix']
    # max_amp_freqvar = np.abs([d['freq_std'][j,i]/d['freqs'][j,i] for j,i in enumerate(ix)])
    max_amp_freqvar = np.abs([d['freq_std'][j,i] for j,i in enumerate(ix)]) * 1000.
    max_amp_freqvar = np.log10(max_amp_freqvar[good_ix])

    # color scaling
    if vbounds is None:
        delta = np.abs(max_amp_freqvar.max() - max_amp_freqvar.min())
        vmin = max_amp_freqvar.min() + delta/10.
        vmax = max_amp_freqvar.max() - delta/10.

    else:
        vmin,vmax = vbounds

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
    c = ax.scatter(w0[good_ix,0], w0[good_ix,2], c=max_amp_freqvar,
                   vmin=vmin, vmax=vmax, cmap='Greys_r', s=sz, marker='s')

    ax.set_xlabel(r'$x_0$ $[{\rm kpc}]$')
    ax.set_ylabel(r'$z_0$ $[{\rm kpc}]$')
    fig.colorbar(c)
    fig.tight_layout()
    fig.savefig(os.path.join(path,"freqvar.pdf"))

    # plot histograms of diffusion rates
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    bins = np.linspace(max_amp_freqvar.min(), max_amp_freqvar.max(), 25)
    n,bins,pa = ax.hist(max_amp_freqvar, alpha=0.4, normed=True, bins=bins)
    # ax.set_xlabel("")
    fig.savefig(os.path.join(path,"freqvar_hist.pdf"))

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
