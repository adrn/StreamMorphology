# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" DOP853 integration in Cython. """

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

cpdef ensemble_integrate(_CPotential cpotential, double[:,::1] w0,
                         double dt0, int nsteps, double t0):
    cdef:
        int i, j, k
        int res, iout
        unsigned norbits = w0.shape[0]
        unsigned ndim = w0.shape[1]
        double[::1] t = np.empty(nsteps)
        double[::1] w = np.empty(norbits*ndim)

        # Note: icont not needed because nrdens == ndim
        double t_end = (<double>nsteps) * dt0

        # set these here
        double atol = 1E-8
        double rtol = 1E-8

    # store initial conditions
    for i in range(norbits):
        for k in range(ndim):
            w[i*ndim + k] = w0[i,k]

    # define full array of times
    t = np.linspace(t0, t_end, nsteps)
    for j in range(1,nsteps,1):
        res = dop853(ndim*norbits, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), norbits,
                     t[j-1], &w[0], t[j], &rtol, &atol, 0, NULL, iout,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, 0, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

    return np.asarray(w)
