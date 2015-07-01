# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:48:21 2015

@author: rlabbe
"""

import matplotlib.pyplot as plt
import book_plots as bp


def plot_track_and_residuals(t, xs, z_xs, res):
    plt.subplot(121)
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



