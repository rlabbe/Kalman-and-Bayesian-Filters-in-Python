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

import filterpy.stats as stats
import matplotlib.pyplot as plt
import numpy as np


def plot1():
    P = np.array([[6, 2.5], [2.5, .6]])
    stats.plot_covariance_ellipse((10, 2), P, facecolor='g', alpha=0.2)


def plot2():
    P = np.array([[6, 2.5], [2.5, .6]])
    circle1=plt.Circle((10,0),3,color='#004080',fill=False,linewidth=4, alpha=.7)
    ax = plt.gca()
    ax.add_artist(circle1)
    plt.xlim(0,10)
    plt.ylim(0,3)
    P = np.array([[6, 2.5], [2.5, .6]])
    stats.plot_covariance_ellipse((10, 2), P, facecolor='g', alpha=0.2)

def plot3():
    P = np.array([[6, 2.5], [2.5, .6]])
    circle1=plt.Circle((10,0),3,color='#004080',fill=False,linewidth=4, alpha=.7)
    ax = plt.gca()
    ax.add_artist(circle1)
    plt.xlim(0,10)
    plt.ylim(0,3)
    plt.axhline(3, ls='--')
    stats.plot_covariance_ellipse((10, 2), P,  facecolor='g', alpha=0.2)

def plot4():
    P = np.array([[6, 2.5], [2.5, .6]])
    circle1=plt.Circle((10,0),3,color='#004080',fill=False,linewidth=4, alpha=.7)
    ax = plt.gca()
    ax.add_artist(circle1)
    plt.xlim(0,10)
    plt.ylim(0,3)
    plt.axhline(3, ls='--')
    stats.plot_covariance_ellipse((10, 2), P,  facecolor='g', alpha=0.2)
    plt.scatter([11.4], [2.65],s=200)
    plt.scatter([12], [3], c='r', s=200)
    plt.show()