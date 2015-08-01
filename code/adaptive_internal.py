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

import book_plots as bp
import matplotlib.pyplot as plt


def plot_track_and_residuals(t, xs, z_xs, res):
    plt.subplot(121)
    if z_xs is not None:
        bp.plot_measurements(t, z_xs, label='z')
    bp.plot_filter(t, xs)
    plt.legend(loc=2)
    plt.xlabel('time (sec)')
    plt.ylabel('X')
    plt.title('estimates vs measurements')
    plt.subplot(122)
    # plot twice so it has the same color as the plot to the left!
    plt.plot(t, res)
    plt.plot(t, res)
    plt.xlabel('time (sec)')
    plt.ylabel('residual')
    plt.title('residuals')
    plt.show()
