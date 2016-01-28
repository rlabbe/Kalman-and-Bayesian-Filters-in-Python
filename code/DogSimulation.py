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


import copy
import math
import numpy as np
from numpy.random import randn

class DogSimulation(object):

    def __init__(self, x0=0, velocity=1,
                 measurement_var=0.0, process_var=0.0):
        """ x0 - initial position
            velocity - (+=right, -=left)
            measurement_variance - variance in measurement m^2
            process_variance - variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.measurement_noise = math.sqrt(measurement_var)
        self.process_noise = math.sqrt(process_var)


    def move(self, dt=1.0):
        '''Compute new position of the dog assuming `dt` seconds have
        passed since the last update.'''
        # compute new position based on velocity. Add in some
        # process noise
        velocity = self.velocity + randn() * self.process_noise * dt
        self.x += velocity * dt


    def sense_position(self):
        # simulate measuring the position with noise
        return self.x + randn() * self.measurement_noise


    def move_and_sense(self, dt=1.0):
        self.move(dt)
        x = copy.deepcopy(self.x)
        return x, self.sense_position()


    def run_simulation(self, dt=1, count=1):
        """ simulate the dog moving over a period of time.

        **Returns**
        data : np.array[float, float]
            2D array, first column contains actual position of dog,
            second column contains the measurement of that position
        """
        return np.array([self.move_and_sense(dt) for i in range(count)])





