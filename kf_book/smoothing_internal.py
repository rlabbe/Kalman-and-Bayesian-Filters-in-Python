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

import matplotlib.pyplot as plt


def show_fixed_lag_numberline():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)

    # draw lines
    xmin = 1
    xmax = 9
    y = 5
    height = 1

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(4.5, y - height / 2., y + height / 2.)
    plt.vlines(6, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)
    plt.vlines(xmax-1, y - height / 2., y + height / 2.)

    # add numbers
    plt.text(xmin, y-1.1, '$x_0$', fontsize=20, horizontalalignment='center')
    plt.text(xmax, y-1.1, '$x_k$', fontsize=20, horizontalalignment='center')
    plt.text(xmax-1, y-1.1, '$x_{k-1}$', fontsize=20, horizontalalignment='center')
    plt.text(4.5, y-1.1, '$x_{k-N+1}$', fontsize=20, horizontalalignment='center')
    plt.text(6, y-1.1, '$x_{k-N+2}$', fontsize=20, horizontalalignment='center')
    plt.text(2.7, y-1.1, '.....', fontsize=20, horizontalalignment='center')
    plt.text(7.2, y-1.1, '.....', fontsize=20, horizontalalignment='center')

    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    #show_2d_transform()
    #show_sigma_selections()

    show_sigma_transform(True)
    #show_four_gps()
    #show_sigma_transform()
    #show_sigma_selections()

