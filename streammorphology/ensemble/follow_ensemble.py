# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity

# Project
from ..extern.fast_ensemble import ensemble_integrate

def follow_ensemble(ensemble_w0, potential, dt, nsteps, neval,
                    kde_bandwidth=None, return_all_density=False):
    """
    Compute diagnostics / metrics at ``neval`` times over the integration
    of the input orbit ensemble. Use this to follow the, e.g., mean density
    of the ensemble.

    Parameters
    ----------
    ensemble_w0 : array_like
        Array of initial conditions for the orbit ensemble.
    potential : :class:`~gary.potential.Potential`
        An instance of a potential class.
    dt : numeric
        Timestep for integration.
    nsteps : int
        Number of steps to integrate for.
    neval : int
        Number of times during integration to evaluate the KDE and density
        distribution.
    kde_bandwidth : float, None (optional)
        If None, use an adaptive bandwidth, or a float for a fixed bandwidth.
    return_all_density : bool (optional)
        Return the full density distributions along with metrics.
    """
    # make sure initial conditions are a contiguous C array
    ww = np.ascontiguousarray(ensemble_w0.copy())
    nensemble = ww.shape[0]

    # spacing of when to compute properties of density distribution
#     idx = np.append(np.linspace(0, nsteps//4, nkld//2+1),
#                         np.linspace(nsteps//4, nsteps, nkld//2+1)[1:]).astype(int)
    idx = np.linspace(0, nsteps, neval).astype(int)

    # if None, adaptive
    if kde_bandwidth is None:
        adaptive_bandwidth = True
    else:
        adaptive_bandwidth = False
        kde = KernelDensity(kernel='epanechnikov',
                            bandwidth=kde_bandwidth)

    # if set, store and return all of the density values
    if return_all_density:
        all_density = np.zeros((neval, nensemble))

    # container to store fraction of stars with density above each threshold
    _moments = dict(mean=np.mean, median=np.median, skew=skew, kurtosis=kurtosis)
    dtype = []
    for k,v in _moments.items():
        dtype.append((k,'f8'))
        dtype.append(("{0}_log".format(k),'f8'))
    data = np.zeros(neval, dtype=dtype)

    # store energies
    Es = np.empty((neval,nensemble))
    Es[0] = potential.total_energy(ensemble_w0[:,:3], ensemble_w0[:,3:])

    # time container
    t = np.zeros(neval)
    for i in range(neval):
        if i == 0:
            www = ww

        else:
            # number of steps to advance the ensemble -- not necessarily constant
            dstep = idx[i] - idx[i-1]
            www = ensemble_integrate(potential.c_instance, ww, dt, dstep, 0.)

            Es[i] = potential.total_energy(www[:,:3], www[:,3:])

            # store the time
            t[i] = t[i-1] + dt*dstep

        # build an estimate of the configuration space density of the ensemble
        if adaptive_bandwidth:
            grid = GridSearchCV(KernelDensity(),
                                {'bandwidth': np.logspace(-2, 1., 16)},
                                cv=10) # 10-fold cross-validation
            grid.fit(www[:,:3])
            kde = grid.best_estimator_

        kde.fit(www[:,:3])

        # evaluate density at the position of the particles
        ln_density = kde.score_samples(www[:,:3])
        density = np.exp(ln_density)

        # store
        if return_all_density:
            all_density[i] = density

        # evaluate the metrics and save
        for k,v in _moments.items():
            data[k][i] = v(density)
            data["{0}_log".format(k)][i] = v(ln_density)

        # reset initial conditions
        ww = www.copy()

    if return_all_density:
        return t, data, Es, all_density
    else:
        return t, data, Es
