# coding: utf-8

""" Plotting utilities for stream-morphology project """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ['panel_plot']

def panel_plot(x, symbol, lim=None, relative=True):
    """ Make a 3-panel plot of projections of actions, angles, or frequency """

    if relative:
        x1,x2,x3 = ((x - x[0])/x[0]).T
    else:
        x1,x2,x3 = x.T

    fig,axes = plt.subplots(1, 3, figsize=(18,6),
                            sharex=True,sharey=True)

    with mpl.rc_context({'lines.marker': '.', 'lines.linestyle': 'none'}):
        axes[0].plot(x1, x3, alpha=0.25)
        axes[1].plot(x2, x3, alpha=0.25)
        axes[2].plot(x1, x2, alpha=0.25)

    if relative:
        axes[0].set_xlabel(r"$({sym}_1 - {sym}_{{1,{{\rm sat}}}})/{sym}_{{1,{{\rm sat}}}}$".format(sym=symbol))
        axes[1].set_xlabel(r"$({sym}_2 - {sym}_{{2,{{\rm sat}}}})/{sym}_{{2,{{\rm sat}}}}$".format(sym=symbol))
        axes[0].set_ylabel(r"$({sym}_3 - {sym}_{{3,{{\rm sat}}}})/{sym}_{{3,{{\rm sat}}}}$".format(sym=symbol))
        axes[2].set_ylabel(r"$({sym}_2 - {sym}_{{2,{{\rm sat}}}})/{sym}_{{2,{{\rm sat}}}}$".format(sym=symbol))
        axes[2].set_xlabel(r"$({sym}_1 - {sym}_{{1,{{\rm sat}}}})/{sym}_{{1,{{\rm sat}}}}$".format(sym=symbol))
    else:
        axes[0].set_xlabel(r"${sym}_1$".format(sym=symbol))
        axes[1].set_xlabel(r"${sym}_2$".format(sym=symbol))
        axes[0].set_ylabel(r"${sym}_3$".format(sym=symbol))
        axes[2].set_ylabel(r"${sym}_2$".format(sym=symbol))
        axes[2].set_xlabel(r"${sym}_1$".format(sym=symbol))

    with mpl.rc_context({'lines.marker': 'o', 'lines.markersize': 10}):
        axes[0].plot(x1[0], x3[0], alpha=0.75, color='r')
        axes[1].plot(x2[0], x3[0], alpha=0.75, color='r')
        axes[2].plot(x1[0], x2[0], alpha=0.75, color='r')

    plt.locator_params(nbins=5)

    if lim is not None:
        axes[0].set_xlim(*lim)
        axes[0].set_ylim(*lim)
    fig.tight_layout()

    return fig
