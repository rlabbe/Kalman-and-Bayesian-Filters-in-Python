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


import kf_book.book_plots as book_plots
from kf_book.book_plots import figsize
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import matplotlib.pylab as plt


def plot_hypothesis1():
    with figsize(y=3.5):
        plt.figure()
        plt.errorbar([1, 2, 3], [170, 161, 169],
                     xerr=0, yerr=10, fmt='bo', capthick=2, capsize=10)

        plt.plot([1, 3], [180, 160], color='g', ls='--')
        plt.plot([1, 3], [170, 170], color='g', ls='--')
        plt.plot([1, 3], [160, 175], color='g', ls='--')
        plt.plot([1, 2, 3], [180, 152, 179], color='g', ls='--')
        plt.xlim(0,4)
        plt.ylim(150, 185)
        plt.xlabel('day')
        plt.ylabel('lbs')
        plt.grid(False)
        plt.tight_layout()


def plot_hypothesis2():
    with book_plots.figsize(y=2.5):
        plt.figure()
        plt.errorbar(range(1, 11), [169, 170, 169,171, 170, 171, 169, 170, 169, 170],
                     xerr=0, yerr=6, fmt='bo', capthick=2, capsize=10)
        plt.plot([1, 10], [169, 170.5], color='g', ls='--')

        plt.xlim(0, 11)
        plt.ylim(150, 185)
        plt.xlabel('day')
        plt.ylabel('lbs')
        plt.grid(False)

def plot_hypothesis3():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    with book_plots.figsize(y=2.5):
        plt.figure()

        plt.errorbar(range(1, 13), weights,
                     xerr=0, yerr=6, fmt='o', capthick=2, capsize=10)

        plt.xlim(0, 13)
        plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        plt.grid(False)


def plot_hypothesis4():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    with book_plots.figsize(y=2.5):
        plt.figure()
        ave = np.sum(weights) / len(weights)
        plt.errorbar(range(1,13), weights, label='weights',
                     yerr=6, fmt='o', capthick=2, capsize=10)
        plt.plot([1, 12], [ave,ave], c='r', label='hypothesis')
        plt.xlim(0, 13)
        plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        book_plots.show_legend()
        plt.grid(False)


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
        plt.xlim(0, 13)
        plt.ylim(145, 185)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        book_plots.show_legend()
        plt.grid(False)


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
        plt.text (1.0, 164.4, "measurement ($z_t$)",ha='center',va='bottom',fontsize=18,color='blue')
        plt.text (0.0, 159.8, "last estimate ($\hat{x}_{t-1}$)", ha='left', va='top',fontsize=18)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        ax.xaxis.grid(True, which="major", linestyle='dotted')
        ax.yaxis.grid(True, which="major", linestyle='dotted')
        plt.ylim(157, 164.5)


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
        plt.text (0, 159.8, "last estimate ($\hat{x}_{t-1}$)", ha='left', va='top',fontsize=18)
        plt.text (0.95, est_y, "estimate ($\hat{x}_{t}$)", ha='right', va='center',fontsize=18)
        plt.xlabel('day')
        plt.ylabel('weight (lbs)')
        ax.xaxis.grid(True, which="major", linestyle='dotted')
        ax.yaxis.grid(True, which="major", linestyle='dotted')
        plt.ylim(157, 164.5)


def plot_gh_results(weights, estimates, predictions, actual, time_step=0):
    n = len(weights)
    if time_step > 0:
        rng = range(1, n+1)
    else:
        rng = range(n, n+1)
    xs = range(n+1)
    book_plots.plot_measurements(range(1, len(weights)+1), weights, color='k', lines=False)
    book_plots.plot_filter(xs, estimates, marker='o', label='Estimates')
    book_plots.plot_track(xs[1:], predictions, c='r', marker='v', label='Predictions')
    plt.plot([xs[0], xs[-1]], actual, c='k', lw=1, label='Actual')
    plt.legend(loc=4)
    book_plots.set_labels(x='day', y='weight (lbs)')
    plt.xlim([-1, n+1])
    plt.ylim([156.0, 173])


def print_results(estimates, prediction, weight):
    print('previous estimate: {:.2f}, prediction: {:.2f}, estimate {:.2f}'.format(
          estimates[-2], prediction, weight))


def plot_g_h_results(measurements, filtered_data,
                     title='', z_label='Measurements',
                     **kwargs):

    book_plots.plot_filter(filtered_data, **kwargs)
    book_plots.plot_measurements(measurements, label=z_label)
    plt.legend(loc=4)
    plt.title(title)
    plt.gca().set_xlim(left=0,right=len(measurements))

    return

    import time
    if not interactive:
        book_plots.plot_filter(filtered_data, **kwargs)
        book_plots.plot_measurements(measurements, label=z_label)
        book_plots.show_legend()
        plt.title(title)
        plt.gca().set_xlim(left=0,right=len(measurements))
    else:
        for i in range(2, len(measurements)):
            book_plots.plot_filter(filtered_data, **kwargs)
            book_plots.plot_measurements(measurements, label=z_label)
            book_plots.show_legend()
            plt.title(title)
            plt.gca().set_xlim(left=0,right=len(measurements))
            plt.gca().canvas.draw()
            time.sleep(0.5)

if __name__ == '__main__':
    pass
