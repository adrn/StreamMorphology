# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
from astropy.utils import isiterable
from astropy import log as logger
import numpy as np
from scipy.signal import argrelmax, argrelmin

# Project
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['ws_to_freqs', 'worker', 'read_allfreqs']

# define indices of columns
colmap = OrderedDict(fxyz=(0,1,2), fRphiz=(3,4,5), dEmax=6, done=7, loop=8, dt=9, nsteps=10, success=11)
l = np.concatenate([[x] if not isiterable(x) else list(x) for x in colmap.values()]).max()+1
_shape = (2, l)

def ptp_freqs(t, *args):
    freqs = []
    for x in args:
        ix = argrelmax(x, mode='wrap')[0]
        f_max = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))

        ix = argrelmin(x, mode='wrap')[0]
        f_min = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))

        freqs.append(np.mean([f_max, f_min]))
    return np.array(freqs)

def estimate_max_period(t, w):
    if w.ndim < 3:
        w = w[:,np.newaxis]

    norbits = w.shape[1]
    periods = []
    for i in range(norbits):
        loop = gd.classify_orbit(w[:,i])
        if np.any(loop):
            # flip coords
            new_w = gd.align_circulation_with_z(w[:,i], loop[0])[:,0]

            # convert to cylindrical
            R = np.sqrt(new_w[:,0]**2 + new_w[:,1]**2)
            phi = np.arctan2(new_w[:,1], new_w[:,0])
            z = new_w[:,2]

            T = 2*np.pi / ptp_freqs(t, R, phi, z)
        else:
            T = 2*np.pi / ptp_freqs(t, *w[:,i,:3].T)

        periods.append(T)

    return np.array(periods)

def ws_to_freqs(naff, ws, nintvec=15):
    # now get other frequencies
    loop = gd.classify_orbit(ws)
    is_loop = np.any(loop)

    if is_loop:
        fxyz = np.ones(3)*np.nan
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.align_circulation_with_z(ws[:,0], loop[0])

        fs = gd.poincare_polar(new_ws[:,0])
        try:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fRphiz = np.ones(3)*np.nan

    else:
        fRphiz = np.ones(3)*np.nan

        # first get x,y,z frequencies
        logger.info('Solving for XYZ frequencies...')
        fs = [(ws[:,0,j] + 1j*ws[:,0,j+3]) for j in range(3)]
        try:
            fxyz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fxyz = np.ones(3)*np.nan

    return np.append(fxyz, fRphiz), is_loop

def estimate_dt_nsteps(potential, w0, nperiods=100):
    # integrate orbit
    t,ws = potential.integrate_orbit(w0, dt=2., nsteps=20000,
                                     Integrator=gi.DOPRI853Integrator)

    # estimate the maximum period
    max_T = round(estimate_max_period(t, ws).max() * 100, -4)

    # integrate for 400 times the max period
    max_T *= 400

    # arbitrarily choose the timestep...
    dt = round(max_T * 1.E-5, 0)
    try:
        nsteps = int(max_T / dt)
    except ValueError:
        dt = 2.
        nsteps = 200000

    return dt, nsteps

def worker(task):
    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    allfreqs_filename = task['allfreqs_filename']
    potential = task['potential']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    allfreqs_shape = (len(w0),) + _shape
    allfreqs = np.memmap(allfreqs_filename, mode='r', shape=allfreqs_shape, dtype='float64')

    # short-circuit if this orbit is already done
    if allfreqs[index,0,7] == 1.:
        return

    # temporary array for results
    tmp = np.zeros(_shape)

    # automatically estimate dt, nsteps
    try:
        dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy())
    except RuntimeError:
        logger.warning("Failed to integrate orbit when estimating dt,nsteps")
        allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
        tmp[:,:] = np.nan
        tmp[:,7] = 1.
        allfreqs[index] = tmp
        allfreqs.flush()
        return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    dEmax = 1.
    maxiter = 3  # maximum number of times to refine integration step
    for i in range(maxiter+1):
        if i > 0:
            # adjust timestep and duration if necessary
            dt /= 2.
            nsteps *= 2
            logger.debug("Refining orbit {} to: dt,nsteps=({},{}). Max. dE={}"
                         .format(index, dt, nsteps, dEmax))

        # integrate orbit
        try:
            t,ws = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(nsteps=8192,atol=1E-14,rtol=1E-9))
        except RuntimeError:
            # ODE integration failed
            logger.warning("Orbit integration failed.")
            continue

        logger.debug('Orbit integrated')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])

        if dEmax < 1E-9:
            break

    if dEmax > 1E-9:
        allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
        tmp[:,:] = np.nan
        tmp[:,7] = 1.
        allfreqs[index] = tmp
        allfreqs.flush()
        return

    # start finding the frequencies -- do first half then second half
    naff = gd.NAFF(t[:nsteps//2+1])
    freqs1,is_loop = ws_to_freqs(naff, ws[:nsteps//2+1])
    freqs2,is_loop = ws_to_freqs(naff, ws[nsteps//2:])

    # save to output array
    tmp[0,:6] = freqs1
    tmp[1,:6] = freqs2

    tmp[:,colmap['dEmax']] = dEmax
    tmp[:,colmap['done']] = 1.
    tmp[:,colmap['loop']] = float(is_loop)
    tmp[:,colmap['dt']] = float(dt)
    tmp[:,colmap['nsteps']] = nsteps
    tmp[:,colmap['success']] = 1.

    allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
    allfreqs[index] = tmp
    allfreqs.flush()

def read_allfreqs(f, norbits):
    """
    Read the numpy memmap'd file containing results from a frequency
    mapping. This function returns a numpy structured array with named
    columns and proper data types.

    Parameters
    ----------
    f : str
        The path to a file containing the frequency mapping results.
    norbits : int
        Number of orbits, e.g., the length of the first axis. Needed to
        properly read in the memmap file.
    """

    allfreqs_shape = (norbits,) + _shape

    # first get the memmap array
    allfreqs = np.memmap(f, mode='r', shape=allfreqs_shape, dtype='float64').copy()

    # replace NAN nsteps with 0
    allfreqs[np.isnan(allfreqs[:,0,colmap['nsteps']]),0,colmap['nsteps']] = 0
    dtype = [('fxyz','f8',(2,3)), ('fRphiz','f8',(2,3)), ('dEmax','f8'), ('done','b1'),
             ('loop','b1'), ('dt','f8'), ('nsteps','i8'), ('success','b1')]
    data = [(allfreqs[i,:,:3],allfreqs[i,:,3:6])+tuple(allfreqs[i,0,6:]) for i in range(norbits)]
    return np.array(data, dtype=dtype)
