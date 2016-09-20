# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from book_format import figsize
from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import sys
import time

try:
    import seabornee
except:
    pass

""" If the plot is inline (%matplotlib inline) we need to
do special processing for the interactive_plot context manager,
otherwise it outputs a lot of extra <matplotlib.figure.figure
type output into the notebook.""" 

IS_INLINE = mpl.get_backend().find('backend_inline') != -1


def end_interactive(fig):
    """ end interaction in a plot created with %matplotlib notebook """

    if IS_INLINE:
        return

    fig.canvas.draw()
    time.sleep(1.)
    plt.close(fig)

    
@contextmanager
def interactive_plot(close=True, fig=None):
    if fig is None and not IS_INLINE:
        fig = plt.figure()
    
    yield
    try:
        # if the figure only uses annotations tight_output
        # throws an exception
        if not IS_INLINE: plt.tight_layout()
    except:
        pass

    if not IS_INLINE: 
        plt.show()

    if close and not IS_INLINE:
        end_interactive(fig)


def plot_errorbars(bars, xlims, ylims=(0, 2)):

    i = 0.0
    for bar in bars:
        plt.errorbar([bar[0]], [i], xerr=[bar[1]], fmt='o', label=bar[2] , capthick=2, capsize=10)
        i += 0.2

    plt.ylim(*ylims)
    plt.xlim(xlims[0], xlims[1])
    show_legend()
    plt.gca().axes.yaxis.set_ticks([])
    plt.show()


def plot_errorbar1():
    with figsize(y=2):
        plt.figure()
        plot_errorbars([(160, 8, 'A'), (170, 8, 'B')],
                       xlims=(145, 185), ylims=(-1, 1))
        plt.show()


def plot_errorbar2():
    with figsize(y=2):
        plt.figure()
        plot_errorbars([(160, 3, 'A'), (170, 9, 'B')],
                       xlims=(145, 185), ylims=(-1, 1))

def plot_errorbar3():
    with figsize(y=2):
        plt.figure()
        plot_errorbars([(160, 1, 'A'), (170, 9, 'B')],
                       xlims=(145, 185), ylims=(-1, 1))


def plot_hypothesis1():
    with figsize(y=2.5):
        plt.figure()
        plt.errorbar([1, 2, 3], [170, 161, 169],
                     xerr=0, yerr=10, fmt='bo', capthick=2, capsize=10)

        plt.plot([1, 3], [180, 160], color='g', ls='--')
        plt.plot([1, 3], [170, 170], color='g', ls='--')
        plt.plot([1, 3], [160, 175], color='g', ls='--')
        plt.plot([1, 2, 3], [180, 152, 179], color='g', ls='--')
        plt.xlim(0,4); plt.ylim(150, 185)
        plt.xlabel('day')
        plt.ylabel('lbs')
        plt.tight_layout()


def plot_hypothesis2():
    with figsize(y=2.5):
        plt.figure()
        plt.errorbar(range(1, 11), [169, 170, 169,171, 170, 171, 169, 170, 169, 170],
                     xerr=0, yerr=6, fmt='bo', capthick=2, capsize=10)
        plt.plot([1, 10], [169, 170.5], color='g', ls='--')
        plt.xlim(0, 11); plt.ylim(150, 185)
        plt.xlabel('day')
        plt.ylabel('lbs')


def plot_hypothesis3():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    with figsize(y=2.5):
        plt.figure()

        plt.errorbar(range(1, 13), weights,
                     xerr=0, yerr=6, fmt='o', capthick=2, capsize=10)

        plt.xlim(0, 13); plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')


def plot_hypothesis4():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    with figsize(y=2.5):
        plt.figure()
        ave = np.sum(weights) / len(weights)
        plt.errorbar(range(1,13), weights, label='weights',
                     yerr=6, fmt='o', capthick=2, capsize=10)
        plt.plot([1, 12], [ave,ave], c='r', label='hypothesis')
        plt.xlim(0, 13); plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        show_legend()


def plot_hypothesis5():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    xs = range(1, len(weights)+1)
    line = np.poly1d(np.polyfit(xs, weights, 1))

    with figsize(y=2.5):
        plt.figure()
        plt.errorbar(range(1, 13), weights, label='weights',
                     yerr=5, fmt='o', capthick=2, capsize=10)
        plt.plot (xs, line(xs), c='r', label='hypothesis')
        plt.xlim(0, 13); plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        show_legend()


def plot_estimate_chart_1():
    with figsize(y=2.5):
        plt.figure()
        ax = plt.axes()
        ax.annotate('', xy=[1,159], xytext=[0,158],
                    arrowprops=dict(arrowstyle='->', ec='r',shrinkA=6, lw=3,shrinkB=5))
        plt.scatter ([0], [158], c='b')
        plt.scatter ([1], [159], c='r')
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        ax.xaxis.grid(True, which="major", linestyle='dotted')
        ax.yaxis.grid(True, which="major", linestyle='dotted')
        plt.tight_layout()


def plot_estimate_chart_2():
    with figsize(y=2.5):
        plt.figure()
        ax = plt.axes()
        ax.annotate('', xy=[1,159], xytext=[0,158],
                    arrowprops=dict(arrowstyle='->',
                                    ec='r', lw=3, shrinkA=6, shrinkB=5))
        plt.scatter ([0], [158.0], c='k',s=128)
        plt.scatter ([1], [164.2], c='b',s=128)
        plt.scatter ([1], [159], c='r', s=128)
        plt.text (1.0, 158.8, "prediction ($x_t)$", ha='center',va='top',fontsize=18,color='red')
        plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
        plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        ax.xaxis.grid(True, which="major", linestyle='dotted')
        ax.yaxis.grid(True, which="major", linestyle='dotted')


def plot_estimate_chart_3():
    with figsize(y=2.5):
        plt.figure()
        ax = plt.axes()
        ax.annotate('', xy=[1,159], xytext=[0,158],
                    arrowprops=dict(arrowstyle='->',
                                    ec='r', lw=3, shrinkA=6, shrinkB=5))

        ax.annotate('', xy=[1,159], xytext=[1,164.2],
                    arrowprops=dict(arrowstyle='-',
                                    ec='k', lw=3, shrinkA=8, shrinkB=8))

        est_y = (158 + .4*(164.2-158))
        plt.scatter ([0,1], [158.0,est_y], c='k',s=128)
        plt.scatter ([1], [164.2], c='b',s=128)
        plt.scatter ([1], [159], c='r', s=128)
        plt.text (1.0, 158.8, "prediction ($x_t)$", ha='center',va='top',fontsize=18,color='red')
        plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
        plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
        plt.text (0.95, est_y, "new estimate ($\hat{x}_{t}$)", ha='right', va='center',fontsize=18)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        ax.xaxis.grid(True, which="major", linestyle='dotted')
        ax.yaxis.grid(True, which="major", linestyle='dotted')



def create_predict_update_chart(box_bg = '#CCCCCC',
                arrow1 = '#88CCFF',
                arrow2 = '#88FF88'):
    plt.figure(figsize=(4, 3.), facecolor='w')
    ax = plt.axes((0, 0, 1, 1),
                  xticks=[], yticks=[], frameon=False)

    pc = Circle((4,5), 0.7, fc=box_bg)
    uc = Circle((6,5), 0.7, fc=box_bg)
    ax.add_patch (pc)
    ax.add_patch (uc)

    plt.text(4,5, "Predict\nStep",ha='center', va='center', fontsize=12)
    plt.text(6,5, "Update\nStep",ha='center', va='center', fontsize=12)

    #btm arrow from update to predict
    ax.annotate('',
                xy=(4.1, 4.5),  xycoords='data',
                xytext=(6, 4.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=pc,
                                patchA=uc,
                                connectionstyle="arc3,rad=-0.5"))
    #top arrow from predict to update
    ax.annotate('',
                xy=(6, 5.5),  xycoords='data',
                xytext=(4.1, 5.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=uc,
                                patchA=pc,
                                connectionstyle="arc3,rad=-0.5"))


    ax.annotate('Measurement ($\mathbf{z_k}$)',
                xy=(6.3, 5.6),  xycoords='data',
                xytext=(6,6), textcoords='data',
                size=14,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    # arrow from predict to state estimate
    ax.annotate('',
                xy=(4.0, 3.8),  xycoords='data',
                xytext=(4.0,4.3), textcoords='data',
                size=12,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('Initial\nConditions ($\mathbf{x_0}$)',
                xy=(4.05, 5.7),  xycoords='data',
                xytext=(2.5, 6.5), textcoords='data',
                size=14,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    plt.text (4, 3.7,'State Estimate ($\mathbf{\hat{x}_k}$)',
              ha='center', va='center', fontsize=14)
    plt.axis('equal')
    plt.xlim(2,10)


def show_residual_chart(show_eq=True, show_H=False):
    plt.figure(figsize=(11, 3.), facecolor='w')
    est_y = ((164.2-158)*.8 + 158)

    ax = plt.axes(xticks=[], yticks=[], frameon=False)
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))

    ax.annotate('', xy=[1,159], xytext=[1,164.2],
                arrowprops=dict(arrowstyle='-',
                                ec='k', lw=3, shrinkA=8, shrinkB=8))

    ax.annotate('', xy=(1., est_y), xytext=(0.9, est_y),
                arrowprops=dict(arrowstyle='->', ec='#004080',
                                lw=2,
                                shrinkA=3, shrinkB=4))


    plt.scatter ([0,1], [158.0,est_y], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.05, 158.8, r"prior $(\bar{x}_t)$", ha='center',va='top',fontsize=18,color='red')
    plt.text (0.5, 159.6, "prediction", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, r"measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, r"posterior ($x_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.text (1.02, est_y-1.5, "residual($y$)", ha='left', va='center',fontsize=18)
    if show_eq:
        if show_H:
            plt.text (1.02, est_y-2.2, r"$y=z-H\bar x_t$", ha='left', va='center',fontsize=18)
        else:
            plt.text (1.02, est_y-2.2, r"$y=z-\bar x_t$", ha='left', va='center',fontsize=18)
    plt.text (0.9, est_y, "new estimate ($x_t$)", ha='right', va='center',fontsize=18)
    plt.text (0.8, est_y-0.5, "(posterior)", ha='right', va='center',fontsize=18)
    if show_eq:
        plt.text (0.75, est_y-1.2, r"$\bar{x}_t + Ky$", ha='right', va='center',fontsize=18)
    plt.xlabel('time')
    ax.yaxis.set_label_position("right")
    plt.ylabel('state')
    plt.xlim(-0.1, 1.5)


def show_legend():
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def bar_plot(pos, x=None, ylim=(0,1), title=None, c='#30a2da',
             **kwargs):
    """ plot the values in `pos` as a bar plot.

    **Parameters**

    pos : list-like
        list of values to plot as bars

    x : list-like, optional
         If provided, specifies the x value for each value in pos. If not
         provided, the first pos element is plotted at x == 0, the second
         at 1, etc.

    ylim : (lower, upper), default = (0,1)
        specifies the lower and upper limits for the y-axis

    title : str, optional
        If specified, provides a title for the plot

    c : color, default='#30a2da'
        Color for the bars

    **kwargs : keywords, optional
        extra keyword arguments passed to ax.bar()

    """

    ax = plt.gca()
    if x is None:
        x = np.arange(len(pos))
    ax.bar(x, pos, color=c, **kwargs)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(np.asarray(x)+0.4, x)
    if title is not None:
        plt.title(title)


def plot_belief_vs_prior(belief, prior, **kwargs):
    """ plots two discrete probability distributions side by side, with
    titles "belief" and "prior"
    """

    plt.subplot(121)
    bar_plot(belief, title='belief', **kwargs)
    plt.subplot(122)
    bar_plot(prior, title='prior', **kwargs)


def plot_prior_vs_posterior(prior, posterior, reverse=False, **kwargs):
    """ plots two discrete probability distributions side by side, with
    titles "prior" and "posterior"
    """
    if reverse:
        plt.subplot(121)
        bar_plot(posterior, title='posterior', **kwargs)
        plt.subplot(122)
        bar_plot(prior, title='prior', **kwargs)
    else:
        plt.subplot(121)
        bar_plot(prior, title='prior', **kwargs)
        plt.subplot(122)
        bar_plot(posterior, title='posterior', **kwargs)


def set_labels(title=None, x=None, y=None):
    """ helps make code in book shorter. Optional set title, xlabel and ylabel
    """
    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)


def set_limits(x, y):
    """ helper function to make code in book shorter. Set the limits for the x
    and y axis.
    """

    plt.gca().set_xlim(x)
    plt.gca().set_ylim(y)

def plot_predictions(p, rng=None, label='Prediction'):
    if rng is None:
        rng = range(len(p))
    plt.scatter(rng, p, marker='v', s=40, edgecolor='r',
                facecolor='None', lw=2, label=label)



def plot_kf_output(xs, filter_xs, zs, title=None, aspect_equal=True):
    plot_filter(filter_xs[:, 0])
    plot_track(xs[:, 0])

    if zs is not None:
        plot_measurements(zs)
    show_legend()
    set_labels(title=title, y='meters', x='time (sec)')
    if aspect_equal:
        plt.gca().set_aspect('equal')
    plt.xlim((-1, len(xs)))
    plt.show()


def plot_measurements(xs, ys=None, color='k', lw=2, label='Measurements',
                      lines=False, **kwargs):
    """ Helper function to give a consistant way to display
    measurements in the book.
    """

    plt.autoscale(tight=True)
    if lines:
        if ys is not None:
            return plt.plot(xs, ys, color=color, lw=lw, ls='--', label=label, **kwargs)
        else:
            return plt.plot(xs, color=color, lw=lw, ls='--', label=label, **kwargs)
    else:
        if ys is not None:
            return plt.scatter(xs, ys, edgecolor=color, facecolor='none',
                        lw=2, label=label, **kwargs),
        else:
            return plt.scatter(range(len(xs)), xs, edgecolor=color, facecolor='none',
                        lw=2, label=label, **kwargs),


def plot_residual_limits(Ps, stds=1.):
    """ plots standand deviation given in Ps as a yellow shaded region. One std
    by default, use stds for a different choice (e.g. stds=3 for 3 standard
    deviations.
    """

    std = np.sqrt(Ps) * stds

    plt.plot(-std, color='k', ls=':', lw=2)
    plt.plot(std, color='k', ls=':', lw=2)
    plt.fill_between(range(len(std)), -std, std,
                 facecolor='#ffff00', alpha=0.3)


def plot_track(xs, ys=None, label='Track', c='k', lw=2, **kwargs):
    if ys is not None:
        return plt.plot(xs, ys, color=c, lw=lw, ls=':', label=label, **kwargs)
    else:
        return plt.plot(xs, color=c, lw=lw, ls=':', label=label, **kwargs)


def plot_filter(xs, ys=None, c='#013afe', label='Filter', var=None, **kwargs):
#def plot_filter(xs, ys=None, c='#6d904f', label='Filter', vars=None, **kwargs):


    if ys is None:
        ys = xs
        xs = range(len(ys))

    plt.plot(xs, ys, color=c, label=label, **kwargs)

    if var is None:
        return

    var = np.asarray(var)

    std = np.sqrt(var)
    std_top = ys+std
    std_btm = ys-std

    plt.plot(xs, ys+std, linestyle=':', color='k', lw=2)
    plt.plot(xs, ys-std, linestyle=':', color='k', lw=2)
    plt.fill_between(xs, std_btm, std_top,
                     facecolor='yellow', alpha=0.2)




def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxweight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()

    plt.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')

    plt.axis('off')
    plt.axis('equal')
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, -w/maxweight),
                      'black')
    if reenable:
        plt.ion()


if __name__ == "__main__":

    plot_errorbar1()
    plot_errorbar2()
    plot_errorbar3()
    plot_hypothesis1()
    plot_hypothesis2()
    plot_hypothesis3()
    plot_hypothesis4()
    plot_hypothesis5()
    plot_estimate_chart_1()
    plot_estimate_chart_2()
    plot_estimate_chart_3()
    create_predict_update_chart()
    show_residual_chart()
    show_residual_chart(True, True)
    plt.close('all')

    '''p = [0.2245871, 0.06288015, 0.06109133, 0.0581008, 0.09334062, 0.2245871,
     0.06288015, 0.06109133, 0.0581008,  0.09334062]*2
    bar_plot(p)
    plot_measurements(p)'''