# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
import gary.dynamics as gd
import gary.potential as gp
import gary.integrate as gi
from gary.units import galactic
from gary.util import get_pool

from streammorphology.initialconditions import loop_grid

# timstep and number of steps
dt = 0.5
nsteps = 200000
nintvec = 15

def ws_to_freqs(naff, ws):
    # first get x,y,z frequencies
    logger.info('Solving for XYZ frequencies...')
    fs = [(ws[:,0,j] + 1j*ws[:,0,j+3]) for j in range(3)]
    try:
        fxyz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
    except:
        fxyz = np.ones(3)*np.nan

    # now get other frequencies
    loop = gd.classify_orbit(ws)
    if np.any(loop):
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.flip_coords(ws, loop[0])

        fs = gd.poincare_polar(new_ws[:,0])
        try:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fRphiz = np.ones(3)*np.nan

    else:
        fRphiz = np.ones(3)*np.nan

    return np.append(fxyz, fRphiz)

def worker(task):
    i,filename,potential = task
    path = os.path.join(os.path.split(filename)[0], 'all')
    freq_fn = os.path.join(path,"{}.npy".format(i))
    if os.path.exists(freq_fn):
        logger.info("Freq file exists.")
        return np.load(freq_fn)

    w0 = np.load(filename)
    t,ws = potential.integrate_orbit(w0[i].copy(), dt=dt, nsteps=nsteps,
                                     Integrator=gi.DOPRI853Integrator)

    # check energy conservation for the orbit
    E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
    dE = np.abs(E[1:] - E[0])

    # start finding the frequencies -- do first half then second half
    naff = gd.NAFF(t[:nsteps//2+1])
    freqs1 = ws_to_freqs(naff, ws[:nsteps//2+1])
    freqs2 = ws_to_freqs(naff, ws[nsteps//2:])

    if np.any(np.isnan(freqs1)) or np.any(np.isnan(freqs2)):
        fig = gd.plot_orbits(ws, marker='.', linestyle='none', alpha=0.1)
        fig.savefig(os.path.join(path, "{}.png".format(i)))

    # return array
    save_arr = np.zeros((7,2))
    save_arr[:6,0] = freqs1
    save_arr[:6,1] = freqs2
    save_arr[6] = dE.max()

    np.save(freq_fn, save_arr)
    return save_arr

def main(path="", mpi=False, overwrite=False):
    np.random.seed(42)
    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=0.9, c=0.7, units=galactic)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("Using MPI")
    logger.info("Caching to: {}".format(path))
    all_freqs_filename = os.path.join(path, "all_freqs.npy")
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path,'all'))

    # initial conditions
    E = -0.1
    w0 = loop_grid(E, potential, Naxis=100)
    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    # save the initial conditions
    filename = os.path.join(path, 'w0.npy')
    np.save(filename, w0)

    if os.path.exists(all_freqs_filename) and overwrite:
        os.remove(all_freqs_filename)

    if not os.path.exists(all_freqs_filename):
        # for zipping
        filenames = [filename]*norbits
        potentials = [potential]*norbits

        tasks = zip(range(norbits), filenames, potentials)

        all_freqs = pool.map(worker, tasks)
        np.save(all_freqs_filename, np.array(all_freqs))

    pool.close()
    all_freqs = np.load(all_freqs_filename)
    return all_freqs

# def diffusion_rates(freqs):
#     Econs = freqs[-1,0]
#     freq_diff = np.abs((freqs[:-1,0] - freqs[:-1,1]) / freqs[:-1,0])

#     fig,axes = plt.subplots(1, 2, figsize=(14,6))
#     axes[0].scatter()
#     axes[1].scatter(freqs[4,0]/freqs[3,0], freqs[2,0]/freqs[3,0],
#                     marker='.', alpha=0.1)

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

    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--path", dest="path", default='', help="Cache path.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    all_freqs = main(path=os.path.abspath(args.path), mpi=args.mpi,
                     overwrite=args.overwrite)

    plot(all_freqs)
    sys.exit(0)
