# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:21:40 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
import numpy as np

def bar_plot(pos, ylim=(0,1), title=None):
    plt.cla()
    ax = plt.gca()
    x = np.arange(len(pos))
    ax.bar(x, pos, color='#30a2da')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x+0.4, x)
    if title is not None:
        plt.title(title)


def plot_measurements(xs, ys=None, c='r', lw=2, label='Measurements', **kwargs):
    """ Helper function to give a consistant way to display
    measurements in the book.
    """

    plt.autoscale(tight=True)
    '''if ys is not None:
        plt.scatter(xs, ys, marker=marker, c=c, s=s,
                    label=label, alpha=alpha)
        if connect:
           plt.plot(xs, ys, c=c, lw=1, alpha=alpha)
    else:
        plt.scatter(range(len(xs)), xs, marker=marker, c=c, s=s,
                    label=label, alpha=alpha)
        if connect:
           plt.plot(range(len(xs)), xs, lw=1, c=c, alpha=alpha)'''

    if ys is not None:
        plt.plot(xs, ys, c=c, lw=lw, linestyle='--', label=label, **kwargs)
    else:
        plt.plot(xs, c=c, lw=lw, linestyle='--', label=label, **kwargs)



def plot_residual_limits(Ps):
    std = np.sqrt(Ps)

    plt.plot(-std, c='k', ls=':', lw=2)
    plt.plot(std, c='k', ls=':', lw=2)
    plt.fill_between(range(len(std)), -std, std,
                 facecolor='#ffff00', alpha=0.3)


def plot_track(xs, ys=None, label='Track', c='k', lw=2):
    if ys is not None:
        plt.plot(xs, ys, c=c, lw=lw, label=label)
    else:
        plt.plot(xs, c=c, lw=lw, label=label)


#c='#013afe'
def plot_filter(xs, ys=None, c='#6d904f', label='Filter', **kwargs):
    if ys is not None:
        plt.plot(xs, ys, c=c, label=label, **kwargs)
    else:
        plt.plot(xs, c=c, label=label, **kwargs)


if __name__ == "__main__":
    p = [0.2245871, 0.06288015, 0.06109133, 0.0581008, 0.09334062, 0.2245871,
     0.06288015, 0.06109133, 0.0581008,  0.09334062]*2
    bar_plot(p)
    plot_measurements(p)