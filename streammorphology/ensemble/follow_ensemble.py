# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity

# Project
from .fast_ensemble import ensemble_integrate

# TODO: needs overhaul
def do_the_kld(ensemble_w0, potential, dt, nsteps, nkld, kde_bandwidth,
               metrics=default_metrics, return_all_density=False):
    """

    Parameters
    ----------
    ...
    kde_bandwidth : float, None
        If None, use an adaptive bandwidth, or a float for a fixed bandwidth.
    """
    # make sure initial conditions are a contiguous C array
    ww = np.ascontiguousarray(ensemble_w0.copy())
    nensemble = ww.shape[0]

#     kld_idx = np.append(np.linspace(0, nsteps//4, nkld//2+1),
#                         np.linspace(nsteps//4, nsteps, nkld//2+1)[1:]).astype(int)
    kld_idx = np.linspace(0, nsteps, nkld+1).astype(int)

    # sort so I preserve some order around here
    metric_names = sorted(metrics.keys())

    # if set, store and return all of the density values
    if return_all_density:
        all_density = np.zeros((nkld, nensemble))

    # if None, adaptive
    if kde_bandwidth is None:
        adaptive_bandwidth = True
    else:
        adaptive_bandwidth = False
        kde = KernelDensity(kernel='epanechnikov',
                            bandwidth=kde_bandwidth)

    # container to store fraction of stars with density above each threshold
    dtype = []
    for name in metric_names:
        dtype.append((name,'f8'))
    metric_data = np.zeros(nkld, dtype=dtype)

    # store energies
    Es = np.empty((nkld+1,nensemble))
    Es[0] = potential.total_energy(ensemble_w0[:,:3], ensemble_w0[:,3:])

    # time container
    t = np.empty(nkld)
    for i in range(nkld):
        logger.debug("KLD step: {0}/{1}".format(i+1, nkld))

        # number of steps to advance the ensemble -- not necessarily constant
        dstep = kld_idx[i+1] - kld_idx[i]
        www = ensemble_integrate(potential.c_instance, ww, dt, dstep, 0.)

        Es[i+1] = potential.total_energy(www[:,:3], www[:,3:])

        # store the time
        if i == 0:
            t[i] = dt*dstep
        else:
            t[i] = t[i-1] + dt*dstep

        # build an estimate of the configuration space density of the ensemble
        if adaptive_bandwidth:
            grid = GridSearchCV(KernelDensity(),
                                {'bandwidth': np.logspace(-1.5, 1.5, 30)},
                                cv=20) # 20-fold cross-validation
            grid.fit(www[:,:3])
            kde = grid.best_estimator_

        kde.fit(www[:,:3])

        # evaluate density at the position of the particles
        ln_densy = kde.score_samples(www[:,:3])
        density = np.exp(ln_densy)

        if return_all_density:
            all_density[i] = density

        # evaluate the metrics and save
        for name in metric_names:
            metric_data[name][i] = metrics[name](density)

        # reset initial conditions
        ww = www.copy()

    if return_all_density:
        return t, metric_data, Es, all_density
    else:
        return t, metric_data, Es
