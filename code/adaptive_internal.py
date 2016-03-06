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

import code.book_plots as bp
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import matplotlib.pyplot as plt
import numpy as np


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


def plot_markov_chain():
    fig = plt.figure(figsize=(4,4), facecolor='w')
    ax = plt.axes((0, 0, 1, 1),
                  xticks=[], yticks=[], frameon=False)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 10)
    box_bg = '#DDDDDD'

    kf1c = Circle((4,5), 0.5, fc=box_bg)
    kf2c = Circle((6,5), 0.5, fc=box_bg)
    ax.add_patch (kf1c)
    ax.add_patch (kf2c)

    plt.text(4,5, "Straight",ha='center', va='center', fontsize=14)
    plt.text(6,5, "Turn",ha='center', va='center', fontsize=14)


    #btm
    plt.text(5, 3.9, ".05", ha='center', va='center', fontsize=18)
    ax.annotate('',
                xy=(4.1, 4.5),  xycoords='data',
                xytext=(6, 4.5), textcoords='data',
                size=10,
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                connectionstyle="arc3,rad=-0.5"))
    #top
    plt.text(5, 6.1, ".03", ha='center', va='center', fontsize=18)
    ax.annotate('',
                xy=(6, 5.5),  xycoords='data',
                xytext=(4.1, 5.5), textcoords='data',
                size=10,

                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                connectionstyle="arc3,rad=-0.5"))

    plt.text(3.5, 5.6, ".97", ha='center', va='center', fontsize=18)
    ax.annotate('',
                xy=(3.9, 5.5),  xycoords='data',
                xytext=(3.55, 5.2), textcoords='data',
                size=10,
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                connectionstyle="angle3,angleA=150,angleB=0"))

    plt.text(6.5, 5.6, ".95", ha='center', va='center', fontsize=18)
    ax.annotate('',
                xy=(6.1, 5.5),  xycoords='data',
                xytext=(6.45, 5.2), textcoords='data',
                size=10,
                arrowprops=dict(arrowstyle="->",
                                fc="0.2", ec="k",
                                connectionstyle="angle3,angleA=-150,angleB=2"))


    plt.axis('equal')
    plt.show()
    bp.end_interactive(fig)


def turning_target(N=600, turn_start=400):
    """ simulate a moving target blah"""

    #r = 1.
    dt = 1.
    phi_sim = np.array(
        [[1, dt, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, dt],
         [0, 0, 0, 1]])

    gam = np.array([[dt**2/2, 0],
                    [dt, 0],
                    [0, dt**2/2],
                    [0, dt]])

    x = np.array([[2000, 0, 10000, -15.]]).T

    simxs = []

    for i in range(N):
        x = np.dot(phi_sim, x)
        if i >= turn_start:
            x += np.dot(gam, np.array([[.075, .075]]).T)
        simxs.append(x)
    simxs = np.array(simxs)

    return simxs


if __name__ ==  "__main__":
    d = turning_target()