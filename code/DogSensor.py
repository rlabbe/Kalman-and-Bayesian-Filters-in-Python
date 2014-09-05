# -*- coding: utf-8 -*-
"""
Created on Sun May 11 13:21:39 2014

@author: rlabbe
"""

from __future__ import print_function, division

import numpy.random as random
import math

class DogSensor(object):

    def __init__(self, x0=0, velocity=1, noise=0.0):
        """ x0 - initial position
            velocity - (+=right, -=left)
            noise - scaling factor for noise, 0== no noise
        """
        self.x = x0
        self.velocity = velocity
        self.noise = math.sqrt(noise)

    def sense(self):
        self.x = self.x + self.velocity
        return self.x + random.randn() * self.noise
