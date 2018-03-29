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
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import matplotlib.pylab as plt

def plot_gh_results(weights, estimates, predictions, time_step=0):

    n = len(weights)
    if time_step > 0:
        rng = range(1, n+1)
    else:
        rng = range(n, n+1)
    xs = range(n+1)
    pred, = book_plots.plot_track(xs[1:], predictions, c='r', marker='v')
    scale, = book_plots.plot_measurements(xs[1:], weights, color='k', lines=False)
    est, = book_plots.plot_filter(xs, estimates, marker='o')

    plt.legend([scale, est, pred], ['Measurement', 'Estimates', 'Predictions'], loc=4)
    book_plots.set_labels(x='day', y='weight (lbs)')
    plt.xlim([-1, n+1])
    plt.ylim([156.0, 173])


def print_results(estimates, prediction, weight):
    print('previous: {:.2f}, prediction: {:.2f} estimate {:.2f}'.format(
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
    plot_errorbar1()
