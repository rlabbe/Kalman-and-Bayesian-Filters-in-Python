# -*- coding: utf-8 -*-
"""
Created on Sun May 11 13:21:39 2014

@author: rlabbe
"""

from __future__ import print_function, division

from numpy.random import randn
import math

class DogSimulation(object):

    def __init__(self, x0=0, velocity=1,
                 measurement_variance=0.0, process_variance=0.0):
        """ x0 - initial position
            velocity - (+=right, -=left)
            measurement_variance - variance in measurement m^2
            process_variance - variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.measurement_noise = math.sqrt(measurement_variance)
        self.process_noise = math.sqrt(process_variance)


    def move(self, dt=1.0):
        '''Compute new position of the dog assuming `dt` seconds have
        passed since the last update.'''
        # compute new position based on velocity. Add in some
        # process noise
        velocity = self.velocity + randn() * self.process_noise
        self.x += velocity * dt

    def sense_position(self):
        # simulate measuring the position with noise
        measurement = self.x + randn() * self.measurement_noise
        return measurement

    def move_and_sense(self, dt=1.0):
        self.move(dt)

        return self.sense_position()
