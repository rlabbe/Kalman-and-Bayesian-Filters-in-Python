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
import filterpy.stats as stats
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, seed


def plot_dog_track(xs, dog, measurement_var, process_var):
    N = len(xs)
    bp.plot_track(dog)
    bp.plot_measurements(xs, label='Sensor')
    bp.set_labels('variance = {}, process variance = {}'.format(
              measurement_var, process_var), 'time', 'pos')
    plt.ylim([0, N])
    bp.show_legend()
    plt.show()


def print_gh(predict, update, z):
    predict_template = '{: 7.3f} {: 8.3f}'
    update_template = '{:.3f}\t{: 7.3f} {: 7.3f}'

    print(predict_template.format(predict[0], predict[1]),end='\t')
    print(update_template.format(z, update[0], update[1]))


def print_variance(positions):
    for i in range(0, len(positions), 5):
        print('\t{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                *[v[1] for v in positions[i:i+5]]))



def gaussian_vs_histogram():

    seed(15)
    xs = np.arange(0, 20, 0.1)
    ys = np.array([stats.gaussian(x-10, 0, 2) for x in xs])
    bar_ys = abs(ys + randn(len(xs)) * stats.gaussian(xs-10, 0, 10)/2)
    plt.gca().bar(xs[::5]-.25, bar_ys[::5], width=0.5, color='g')
    plt.plot(xs, ys, lw=3, color='k')
    plt.xlim(5, 15)


class DogSimulation(object):
    def __init__(self, x0=0, velocity=1,
                 measurement_var=0.0,
                 process_var=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            measurement_var: variance in measurement m^2
            process_var: variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.meas_std = sqrt(measurement_var)
        self.process_std = sqrt(process_var)

    def move(self, dt=1.0):
        """Compute new position of the dog in dt seconds."""
        dx = self.velocity + randn()*self.process_std
        self.x += dx * dt

    def sense_position(self):
        """ Returns measurement of new position in meters."""
        measurement = self.x + randn()*self.meas_std
        return measurement

    def move_and_sense(self):
        """ Move dog, and return measurement of new position in meters"""
        self.move()
        return self.sense_position()

