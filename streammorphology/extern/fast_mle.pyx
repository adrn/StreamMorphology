# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Fast maximum Lyapunov exponent (MLE) estimation with Cython """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.stdio cimport printf

from gary.potential.cpotential cimport _CPotential

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil

cdef extern from "dop853.h":
    ctypedef void (*GradFn)(double *pars, double *q, double *grad) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f, GradFn gradfunc, double *gpars, unsigned norbits) nogil
    double contd8 (unsigned ii, double x)

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, GradFn gradfunc, double *gpars, unsigned norbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   GradFn func, double *pars, unsigned norbits)
    double six_norm (double *x)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cpdef max_lyapunov_exp(_CPotential cpotential, double[:,::1] w0,
                       double dt, int nsteps, double t0,
                       double atol, double rtol, int nmax,
                       double d0, int nsteps_per_pullback):
    cdef:
        int i, j, k, jiter
        int res

        unsigned norbits = w0.shape[0]
        unsigned noffset_orbits = norbits - 1
        unsigned ndim = w0.shape[1]
        unsigned niter = nsteps // nsteps_per_pullback
        double[::1] w = np.empty(norbits*ndim)

        double t, d1_mag, norm
        double[:,::1] d1 = np.empty((noffset_orbits,ndim))
        double[::1] LEs = np.zeros(noffset_orbits)

    # store initial conditions
    for i in range(norbits):
        for k in range(ndim):
            w[i*ndim + k] = w0[i,k]

    # dummy counter for storing Lyapunov stuff, which only happens every few steps
    for j in range(1,nsteps+1,1):
        t = t0 + dt
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), norbits,
                     t0, &w[0], t, &rtol, &atol, 0, NULL, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

        if (j % nsteps_per_pullback) == 0:
            # get magnitude of deviation vector
            for i in range(noffset_orbits):
                for k in range(ndim):
                    d1[i,k] = w[(i+1)*ndim + k] - w[k]

                d1_mag = six_norm(&d1[i,0])
                LEs[i] = LEs[i] + log(d1_mag / d0)

                # renormalize offset orbits
                for k in range(ndim):
                    w[(i+1)*ndim + k] = w[k] + d0 * d1[i,k] / d1_mag

        t0 = t

    # LEs = np.array([np.sum(LEs[:j],axis=0)/t[j*nsteps_per_pullback] for j in range(1,niter)])
    return np.array(LEs) / t, t, np.array(w).reshape(norbits,ndim)

def mle(w0, potential, dt, nsteps, d0=1e-5,
        nsteps_per_pullback=10, noffset_orbits=2, t0=0.,
        atol=1E-9, rtol=1E-9, nmax=0):

    if not hasattr(potential, 'c_instance'):
        raise TypeError("Input potential must be a CPotential subclass.")

    if w0.ndim == 1: # generate offset orbits
        d0_vec = np.random.uniform(size=(noffset_orbits,w0.size))
        norm = np.linalg.norm(d0_vec, axis=-1)
        d0_vec *= d0/norm[:,None]  # rescale offset vectors

        _new_w0 = np.zeros((noffset_orbits+1,w0.size))
        _new_w0[0] = w0
        _new_w0[1:] = w0[None] + d0_vec
        w0 = _new_w0

    elif w0.ndim == 2: # assume offset orbit initial conditions specified
        if w0.shape[0] == 1:
            raise ValueError("Only one orbit passed in. If w0 is 2-D, You "
                             "must pass in initial conditions for a t least "
                             "one offset orbit as well.")

    else:
        raise ValueError('w0 must be 1- or 2-D.')

    return max_lyapunov_exp(potential.c_instance, w0,
                            dt, nsteps+1, t0, atol, rtol, nmax,
                            d0, nsteps_per_pullback)
